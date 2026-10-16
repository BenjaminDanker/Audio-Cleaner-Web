//! One live session = one connection from the web process (server/acw/stream/worker.py `handle`, ported).
//! Its thread reads frames, processes them and answers in order; events (captions, warnings) go out through a
//! small writer thread of their own, so a slow web process never holds up the caption decoders.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use crate::admission::{pin_current_thread, Decision};
use crate::aec::EchoCanceller;
use crate::captions::{Captioner, Emit};
use crate::dsp::LiveChain;
use crate::ffi::{OpusDecoder, OpusEncoder, OPUS_FRAME};
use crate::ipc::{self, Audio};
use crate::Engine;

const MAX_PENDING_REFS: usize = 50; // reference frames kept for mic frames that haven't arrived (1 s)
const EVENT_QUEUE: usize = 200;

fn send(w: &Mutex<UnixStream>, kind: u8, payload: &[u8]) -> std::io::Result<()> {
    let mut s = w.lock().unwrap();
    ipc::write_msg(&mut *s, kind, payload)
}

pub fn handle(engine: Arc<Engine>, stream: UnixStream) {
    let writer = match stream.try_clone() {
        Ok(w) => Arc::new(Mutex::new(w)),
        Err(_) => return,
    };
    let mut reader = stream;
    let (kind, payload) = match ipc::read_msg(&mut reader) {
        Ok(m) => m,
        Err(_) => return,
    };
    if kind != ipc::MSG_START {
        return;
    }
    let cfg: Value = match serde_json::from_slice(&payload) {
        Ok(v) => v,
        Err(_) => return,
    };
    let tier = cfg.get("denoise").and_then(|v| v.as_str()).map(str::to_owned);
    if tier.as_deref().is_some_and(|t| t != "light") {
        let _ = send(&writer, ipc::MSG_EVENT, json!({"type": "error", "code": "bad_tier"}).to_string().as_bytes());
        return;
    }
    let core = match engine.admission.lock().unwrap().decide(&cfg, &engine.host) {
        Decision::Admit(c) => c,
        Decision::Refuse(ev) => {
            if ev["code"] == "full" {
                let a = engine.admission.lock().unwrap();
                log!("live full: {} sessions, {:.1} of {:.1} units", a.active, a.units, a.capacity(&engine.host));
            }
            let _ = send(&writer, ipc::MSG_EVENT, ev.to_string().as_bytes());
            return;
        }
    };
    let cpus = engine.admission.lock().unwrap().cores[core].clone();
    pin_current_thread(&cpus);
    engine.repin_decoders();
    run(&engine, &cfg, core, tier.is_some(), reader, writer);
    engine.admission.lock().unwrap().release(&cfg, core);
    engine.repin_decoders();
}

fn run(engine: &Arc<Engine>, cfg: &Value, core: usize, denoise: bool, mut reader: UnixStream, writer: Arc<Mutex<UnixStream>>) {
    let flag = |k: &str| cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    let atten = cfg.get("atten_lim_db").and_then(|v| v.as_f64()).unwrap_or(100.0) as f32;
    let mut chain = match LiveChain::new(&engine.lib_df, if denoise { Some(engine.dfn_model.as_path()) } else { None }, atten) {
        Ok(c) => c,
        Err(e) => {
            log!("live session could not start: {e}");
            return;
        }
    };
    // Events: a bounded queue to a writer thread (a full queue drops the event, like the Python worker).
    let (tx, rx) = sync_channel::<Vec<u8>>(EVENT_QUEUE);
    let ev_writer = writer.clone();
    let events_thread = std::thread::spawn(move || {
        for ev in rx {
            if send(&ev_writer, ipc::MSG_EVENT, &ev).is_err() {
                break;
            }
        }
    });
    // Only the captioner sends events: when it is gone, the writer thread ends.
    let tx_emit: SyncSender<Vec<u8>> = tx;
    let emit: Emit = Arc::new(move |ev: Value| {
        if let Err(TrySendError::Full(_)) = tx_emit.try_send(ev.to_string().into_bytes()) {
            // the web side is far behind: drop it
        }
    });
    let captioner = if cfg.get("captions").and_then(|v| v.as_str()) == Some("local") {
        match Captioner::new(&engine.captions_dir, emit.clone()) {
            Ok(c) => Some(c),
            Err(e) => {
                log!("captions unavailable: {e}");
                None
            }
        }
    } else {
        None
    };
    drop(emit);
    let mut aec = if flag("echo_reference") { Some(EchoCanceller::new()) } else { None };
    let latency = chain.latency_samples + if aec.is_some() { crate::AEC_DELAY } else { 0 };
    let ready = json!({"type": "ready", "latencySamples": latency, "worker": 1, "cores": [engine.core_index(core)]});
    if send(&writer, ipc::MSG_EVENT, ready.to_string().as_bytes()).is_err() {
        finish(captioner, events_thread);
        return;
    }
    let tap_pcm = flag("tap_pcm");
    let return_audio = cfg.get("return_audio").and_then(|v| v.as_bool()).unwrap_or(true);
    let bitrate = cfg.get("opus_bitrate").and_then(|v| v.as_i64()).unwrap_or(96_000) as i32;
    let mut dec: Option<OpusDecoder> = None;
    let mut enc: Option<OpusEncoder> = None;
    let mut ref_dec: Option<OpusDecoder> = None;
    let mut refs: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
    let mut pcm = [0f32; OPUS_FRAME];

    loop {
        let (kind, payload) = match ipc::read_msg(&mut reader) {
            Ok(m) => m,
            Err(_) => break,
        };
        match kind {
            ipc::MSG_AUDIO => {
                let Some((seq, audio)) = ipc::unpack_audio(&payload) else { break };
                let opus = matches!(audio, Audio::Opus(_));
                let x: Vec<f32> = match audio {
                    Audio::Opus(pkt) => {
                        if dec.is_none() {
                            match OpusDecoder::new() {
                                Ok(d) => dec = Some(d),
                                Err(_) => break,
                            }
                        }
                        if let Err(e) = dec.as_mut().unwrap().decode(pkt, &mut pcm) {
                            log!("stream session ended: {e}"); // e.g. an Opus packet that isn't 20 ms
                            break;
                        }
                        pcm.to_vec()
                    }
                    Audio::F32(v) => v,
                };
                let x = match aec.as_mut() {
                    // a missing (or late) reference counts as silence
                    Some(a) => {
                        let r = refs.remove(&seq).unwrap_or_default();
                        let mut y = a.process(&x, &r);
                        y.resize(x.len(), 0.0);
                        y
                    }
                    None => x,
                };
                let y = chain.process(&x);
                if let Some(c) = captioner.as_ref() {
                    // The original (echo removed): recognizers do better on it than on denoised audio. The denoised
                    // level tells speech from background noise, so the decoder skips the silences between.
                    let level = chain.df.as_ref().map(|_| {
                        let ms = y.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / y.len().max(1) as f64;
                        (20.0 * (ms.sqrt() + 1e-9).log10()) as f32
                    });
                    c.feed(&x, level);
                }
                let out = if !return_audio {
                    ipc::pack_audio(seq, &[]) // an empty answer: the web side only times it
                } else if opus {
                    if enc.is_none() {
                        match OpusEncoder::new(bitrate) {
                            Ok(e) => enc = Some(e),
                            Err(_) => break,
                        }
                    }
                    match enc.as_mut().unwrap().encode(&y) {
                        Ok(pkt) => ipc::pack_opus(seq, pkt),
                        Err(_) => break,
                    }
                } else {
                    ipc::pack_audio(seq, &y)
                };
                if send(&writer, ipc::MSG_AUDIO, &out).is_err() {
                    break;
                }
                if tap_pcm && send(&writer, ipc::MSG_TAP, &ipc::pack_audio(seq, &x)).is_err() {
                    break; // extra-accurate captions hear the original, not the denoised audio
                }
            }
            ipc::MSG_SILENCE => {
                if let (Some(c), Some((_seq, n))) = (captioner.as_ref(), ipc::unpack_silence(&payload)) {
                    c.silence(n);
                }
            }
            ipc::MSG_REF => {
                if aec.is_none() {
                    continue;
                }
                let Some((seq, audio)) = ipc::unpack_audio(&payload) else { continue };
                let r = match audio {
                    Audio::Opus(pkt) => {
                        if ref_dec.is_none() {
                            ref_dec = OpusDecoder::new().ok();
                        }
                        let mut buf = [0f32; OPUS_FRAME];
                        match ref_dec.as_mut().map(|d| d.decode(pkt, &mut buf)) {
                            Some(Ok(())) => buf.to_vec(),
                            _ => continue,
                        }
                    }
                    Audio::F32(v) => v,
                };
                refs.insert(seq, r);
                while refs.len() > MAX_PENDING_REFS {
                    let first = *refs.keys().next().unwrap();
                    refs.remove(&first);
                }
            }
            ipc::MSG_CONFIG => {
                if let Ok(c) = serde_json::from_slice::<HashMap<String, Value>>(&payload) {
                    if let Some(db) = c.get("atten_lim_db").and_then(|v| v.as_f64()) {
                        chain.set_atten_lim(db as f32);
                    }
                }
            }
            ipc::MSG_END => {
                if let Some(c) = captioner.as_ref() {
                    c.flush(Duration::from_secs(3));
                }
                drop(captioner);
                let _ = events_thread.join(); // the last captions go out before the end
                let _ = send(&writer, ipc::MSG_END, b"");
                return;
            }
            _ => {}
        }
    }
    finish(captioner, events_thread);
}

fn finish(captioner: Option<Arc<Captioner>>, events_thread: std::thread::JoinHandle<()>) {
    if let Some(c) = captioner {
        c.close();
    }
    let _ = events_thread.join();
}

#[allow(dead_code)]
fn flush_writer(w: &Mutex<UnixStream>) {
    let _ = w.lock().unwrap().flush();
}
