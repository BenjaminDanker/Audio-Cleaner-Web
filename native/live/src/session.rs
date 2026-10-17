//! One live session = one connection from the web process (server/acw/stream/routes.py).
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
use crate::dsp::{Crossfade, DelayLine, LiveChain, NoiseMeter, SAMPLE_RATE};
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
    if kind == ipc::MSG_QUERY {
        let status = engine.status();
        let _ = send(&writer, ipc::MSG_EVENT, status.to_string().as_bytes());
        return;
    }
    if kind != ipc::MSG_START {
        return;
    }
    let cfg: Value = match serde_json::from_slice(&payload) {
        Ok(v) => v,
        Err(_) => return,
    };
    if !valid_features(&cfg) {
        let _ = send(&writer, ipc::MSG_EVENT, json!({"type": "error", "code": "bad_tier"}).to_string().as_bytes());
        return;
    }
    let decision = engine.admission.lock().unwrap().decide(&cfg, &engine.host); // (the lock ends here)
    let core = match decision {
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
    let (cfg, core) = run(&engine, cfg, core, reader, writer);
    engine.admission.lock().unwrap().release(&cfg, core);
}

/// The parts of a session that can change while it runs (MSG_CONFIG), next to the noise strength.
const FEATURES: [&str; 6] = ["denoise", "captions", "echo_reference", "tap_pcm", "trial", "may_open_core"];

fn valid_features(cfg: &Value) -> bool {
    let tier = |k: &str, ok: &[&str]| cfg.get(k).map_or(true, |v| v.is_null() || v.as_str().is_some_and(|t| ok.contains(&t)));
    tier("denoise", &["light"]) && tier("captions", &["local", "premium"])
}

fn is_on(cfg: &Value, k: &str) -> bool {
    cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(false)
}

fn local_captions(cfg: &Value) -> bool {
    cfg.get("captions").and_then(|v| v.as_str()) == Some("local")
}

fn denoise(cfg: &Value) -> bool {
    cfg.get("denoise").and_then(|v| v.as_str()) == Some("light")
}

/// The room test (the plugin's "Test my room", MSG_CONFIG room_test_seconds) listens through noise removal.
fn room_test_refused() -> Value {
    json!({"type": "room_test", "ok": false, "code": "needs_noise_removal",
           "message": "The room test listens through noise removal: turn noise removal on first."})
}

fn start_captioner(engine: &Engine, emit: &Emit, at: u64, first_line: u64) -> Option<Arc<Captioner>> {
    match Captioner::new(&engine.captions_dir, emit.clone(), at, first_line) {
        Ok(c) => Some(c),
        Err(e) => {
            log!("captions unavailable: {e}");
            None
        }
    }
}

/// Runs the session until it ends; returns its features as they were at the end and the core it ran on (what the
/// admission has to give back).
fn run(engine: &Arc<Engine>, mut cfg: Value, mut core: usize, mut reader: UnixStream, writer: Arc<Mutex<UnixStream>>) -> (Value, usize) {
    let atten = cfg.get("atten_lim_db").and_then(|v| v.as_f64()).unwrap_or(100.0) as f32;
    let mut chain = match LiveChain::new(&engine.lib_df, &engine.dfn_model, denoise(&cfg), atten) {
        Ok(c) => c,
        Err(e) => {
            log!("live session could not start: {e}");
            return (cfg, core);
        }
    };
    // Events: a bounded queue to a writer thread (a full queue drops the event: late captions are worthless). The
    // writer thread ends when the last sender is gone: the session's own (`emit`, dropped at the end) and the captioner's.
    let (tx, rx) = sync_channel::<Vec<u8>>(EVENT_QUEUE);
    let ev_writer = writer.clone();
    let events_thread = std::thread::spawn(move || {
        for ev in rx {
            if send(&ev_writer, ipc::MSG_EVENT, &ev).is_err() {
                break;
            }
        }
    });
    let tx_emit: SyncSender<Vec<u8>> = tx;
    let emit: Emit = Arc::new(move |ev: Value| {
        if let Err(TrySendError::Full(_)) = tx_emit.try_send(ev.to_string().into_bytes()) {
            // the web side is far behind: drop it
        }
    });
    let mut captioner = if local_captions(&cfg) { start_captioner(engine, &emit, 0, 0) } else { None };
    // The echo canceller delays the mic by AEC_DELAY: without it the mic goes through a delay line of that length, so
    // the session's delay is the same whether it runs or not.
    let mut aec = if is_on(&cfg, "echo_reference") { Some(EchoCanceller::new()) } else { None };
    let mut aec_dry = DelayLine::new(crate::AEC_DELAY);
    let mut aec_fade: Option<Crossfade> = None;
    let latency = chain.latency_samples + crate::AEC_DELAY;
    let ready = json!({"type": "ready", "latencySamples": latency, "worker": 1, "cores": [core]});
    if send(&writer, ipc::MSG_EVENT, ready.to_string().as_bytes()).is_err() {
        drop(emit);
        finish(captioner, events_thread);
        return (cfg, core);
    }
    let return_audio = cfg.get("return_audio").and_then(|v| v.as_bool()).unwrap_or(true);
    let bitrate = cfg.get("opus_bitrate").and_then(|v| v.as_i64()).unwrap_or(96_000) as i32;
    let mut dec: Option<OpusDecoder> = None;
    let mut enc: Option<OpusEncoder> = None;
    let mut ref_dec: Option<OpusDecoder> = None;
    let mut refs: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
    let mut pcm = [0f32; OPUS_FRAME];
    let mut samples: u64 = 0; // 48 kHz samples of the session so far, sound and silence (the captions' clock)
    let mut next_line: u64 = 0; // caption lines: a captioner started later goes on counting
    let mut room_test: Option<usize> = None; // seconds asked for, while chain.meter measures them

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
                samples += x.len() as u64;
                let dry = aec_dry.process(&x);
                let x = match aec.as_mut() {
                    // a missing (or late) reference counts as silence
                    Some(a) => {
                        let r = refs.remove(&seq).unwrap_or_default();
                        let mut y = a.process(&x, &r);
                        y.resize(x.len(), 0.0);
                        match aec_fade.as_mut() {
                            Some(f) => {
                                let (mixed, done) = f.mix(&dry, &y);
                                if done {
                                    if !f.to_wet() {
                                        aec = None;
                                    }
                                    aec_fade = None;
                                }
                                mixed
                            }
                            None => y,
                        }
                    }
                    None => dry,
                };
                let y = chain.process(&x);
                if let (Some(n), Some(m)) = (room_test, chain.meter.as_ref()) {
                    if m.samples >= n * SAMPLE_RATE {
                        let db = (m.figure_db() * 10.0).round() / 10.0;
                        emit(json!({"type": "room_test", "ok": true, "noiseDb": db, "seconds": n}));
                        chain.meter = None;
                        room_test = None;
                    }
                }
                if let Some(c) = captioner.as_ref() {
                    // The original (echo removed): recognizers do better on it than on denoised audio. The denoised
                    // level tells speech from background noise, so the decoder skips the silences between.
                    let level = chain.denoising().then(|| {
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
                if is_on(&cfg, "tap_pcm") && send(&writer, ipc::MSG_TAP, &ipc::pack_audio(seq, &x)).is_err() {
                    break; // extra-accurate captions hear the original, not the denoised audio
                }
            }
            ipc::MSG_SILENCE => {
                if let Some((_seq, n)) = ipc::unpack_silence(&payload) {
                    samples += n as u64;
                    if let Some(c) = captioner.as_ref() {
                        c.silence(n);
                    }
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
                let Ok(c) = serde_json::from_slice::<HashMap<String, Value>>(&payload) else { continue };
                if let Some(db) = c.get("atten_lim_db").and_then(|v| v.as_f64()) {
                    chain.set_atten_lim(db as f32);
                }
                if let Some(n) = c.get("room_test_seconds").and_then(|v| v.as_f64()) {
                    // measured at the noise remover (NoiseMeter): the processed seconds count, not the silence it skips
                    if chain.denoising() {
                        chain.meter = Some(NoiseMeter::new());
                        room_test = Some(n.clamp(5.0, 60.0).round() as usize);
                    } else {
                        emit(room_test_refused());
                    }
                }
                let Some(id) = c.get("id").cloned() else { continue }; // the strength alone: nothing to answer
                let mut new = cfg.clone();
                for k in FEATURES {
                    if let Some(v) = c.get(k) {
                        new[k] = v.clone();
                    }
                }
                let answer = if !valid_features(&new) {
                    json!({"type": "_config", "id": id, "ok": false, "code": "bad_tier"})
                } else {
                    let decided = engine.admission.lock().unwrap().change(&cfg, &new, core, &engine.host);
                    match decided {
                        Err(ev) => json!({"type": "_config", "id": id, "ok": false, "code": ev["code"]}),
                        Ok(target) => {
                            if target != core {
                                let cpus = engine.admission.lock().unwrap().cores[target].clone();
                                pin_current_thread(&cpus);
                                core = target;
                            }
                            if let Err(e) = chain.set_denoise(denoise(&new)) {
                                log!("noise removal could not start: {e}");
                                new["denoise"] = Value::Null;
                            }
                            if room_test.is_some() && !chain.denoising() {
                                chain.meter = None;
                                room_test = None;
                                emit(room_test_refused());
                            }
                            match (captioner.is_some(), local_captions(&new)) {
                                (false, true) => captioner = start_captioner(engine, &emit, samples, next_line),
                                (true, false) => {
                                    if let Some(old) = captioner.take() {
                                        old.flush(Duration::from_millis(300));
                                        next_line = old.next_line();
                                        old.close();
                                    }
                                }
                                _ => {}
                            }
                            let echo_on = aec_fade.as_ref().map_or(aec.is_some(), |f| f.to_wet());
                            if is_on(&new, "echo_reference") != echo_on {
                                match aec_fade.as_mut() {
                                    Some(f) => f.reverse(),
                                    None => {
                                        if aec.is_none() {
                                            aec = Some(EchoCanceller::new());
                                        }
                                        aec_fade = Some(Crossfade::new(!echo_on));
                                    }
                                }
                            }
                            cfg = new;
                            json!({"type": "_config", "id": id, "ok": true, "cores": [core],
                                   "denoise": cfg["denoise"].clone()})
                        }
                    }
                };
                if send(&writer, ipc::MSG_EVENT, answer.to_string().as_bytes()).is_err() {
                    break;
                }
            }
            ipc::MSG_END => {
                if let Some(c) = captioner.as_ref() {
                    c.flush(Duration::from_secs(3));
                }
                drop(captioner);
                drop(emit);
                let _ = events_thread.join(); // the last captions go out before the end
                let _ = send(&writer, ipc::MSG_END, b"");
                return (cfg, core);
            }
            _ => {}
        }
    }
    drop(emit);
    finish(captioner, events_thread);
    (cfg, core)
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
