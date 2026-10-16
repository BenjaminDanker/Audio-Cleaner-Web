//! Standard live captions: a streaming English recognizer (NeMo FastConformer transducer, 480 ms chunks, int8, via
//! the sherpa-onnx C API), a port of server/acw/stream/engine.py (_StreamingRecognizer, StreamingCaptioner).
//!
//! Sessions only queue audio; a decoder thread per recognizer owns all model calls and decodes every stream that has
//! a chunk ready in one batch. One recognizer does about 13x real time however many streams share it, so streams are
//! spread over a few recognizers (each one more copy of the model, ~220 MB, made only when the others are full).
//! Silence isn't decoded: from 2 s after the last speech until someone talks again the audio is only counted.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sherpa_onnx_sys as so;

use crate::dsp::{Decimate3, SAMPLE_RATE};

const INPUT_RATE: usize = 16_000;
const TICK: Duration = Duration::from_millis(50);
pub const STREAMS_PER_RECOGNIZER: usize = 10;

// A caption stream is shed, not queued, when the decoder falls behind it.
const MAX_BACKLOG: usize = 2 * SAMPLE_RATE;
const KEEP_BACKLOG: usize = SAMPLE_RATE / 2;
const RECOVERED_BACKLOG: usize = SAMPLE_RATE / 2;
const CAPTIONS_BEHIND_MESSAGE: &str = "Captions are running behind because live is very busy, so some words are being skipped. They catch up on their own.";

// Silence gate (engine.py): quieter than SPEECH_DBFS is silence; GATE_AFTER_S of it is decoded so the line ends; then
// skipped (counted) until speech, with PREROLL_FRAMES of the quiet kept for the start of the next words.
pub const SPEECH_DBFS: f32 = -50.0;
pub const GATE_AFTER_S: f32 = 2.0;
const PREROLL_FRAMES: usize = 15;

pub type Emit = Arc<dyn Fn(Value) + Send + Sync>;

struct Recognizer(*const so::OnlineRecognizer);
unsafe impl Send for Recognizer {}
unsafe impl Sync for Recognizer {}

/// Audio waiting for the decoder: samples to decode, or (Skip) samples only counted.
enum Piece {
    Audio(Vec<f32>),
    Skip(usize),
}

struct Shared {
    pending: Vec<Piece>,
}

/// State only the decoder thread touches.
struct DecoderSide {
    stream: *const so::OnlineStream,
    to16k: Decimate3,
    fed: u64, // 16 kHz samples given to the recognizer (or skipped)
    line_start: u64,
    last_text: String,
    line: u64,
    behind: bool,
}
unsafe impl Send for DecoderSide {}

pub struct Captioner {
    group: Arc<Group>,
    id: u64,
    shared: Mutex<Shared>,
    side: Mutex<DecoderSide>,
    emit: Emit,
    closed: AtomicBool,
    // the gate (session thread only, behind its own lock to keep Captioner Sync)
    gate: Mutex<Gate>,
}

struct Gate {
    quiet: f32,
    gated: bool,
    preroll: Vec<Vec<f32>>,
    skipped: usize,
    skipped_total: usize,
}

struct Group {
    rec: Recognizer,
    members: Mutex<HashMap<u64, Arc<Captioner>>>,
    tick: Mutex<()>, // held for a whole decoder tick: a stream that leaves waits for it
}

static GROUPS: OnceLock<Mutex<Vec<Arc<Group>>>> = OnceLock::new();
static DECODER_CPUS: Mutex<(u64, Vec<usize>)> = Mutex::new((0, Vec::new()));

/// The CPUs the caption decoders should run on (the cores that carry sessions); they move at their next tick.
pub fn set_decoder_cpus(cpus: Vec<usize>) {
    let mut d = DECODER_CPUS.lock().unwrap();
    if d.1 != cpus {
        d.0 += 1;
        d.1 = cpus;
    }
}
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn new_recognizer(model_dir: &Path, threads: i32) -> Result<Recognizer, String> {
    let p = |f: &str| cstr(&model_dir.join(f).to_string_lossy());
    let (enc, dec, join, tok) = (p("encoder.int8.onnx"), p("decoder.int8.onnx"), p("joiner.int8.onnx"), p("tokens.txt"));
    let (empty, cpu, greedy) = (cstr(""), cstr("cpu"), cstr("greedy_search"));
    let mut cfg: so::OnlineRecognizerConfig = unsafe { std::mem::zeroed() };
    cfg.feat_config = so::FeatureConfig { sample_rate: INPUT_RATE as i32, feature_dim: 80 };
    cfg.model_config.transducer = so::OnlineTransducerModelConfig { encoder: enc.as_ptr(), decoder: dec.as_ptr(), joiner: join.as_ptr() };
    cfg.model_config.tokens = tok.as_ptr();
    cfg.model_config.num_threads = threads;
    cfg.model_config.provider = cpu.as_ptr();
    cfg.model_config.model_type = empty.as_ptr();
    cfg.model_config.modeling_unit = empty.as_ptr();
    cfg.model_config.bpe_vocab = empty.as_ptr();
    cfg.decoding_method = greedy.as_ptr();
    cfg.max_active_paths = 4;
    cfg.enable_endpoint = 1;
    // Long silence with nothing said: reset. A pause after speech ends the line; long monologues still get lines.
    cfg.rule1_min_trailing_silence = 5.0;
    cfg.rule2_min_trailing_silence = 0.7;
    cfg.rule3_min_utterance_length = 12.0;
    cfg.hotwords_file = empty.as_ptr();
    cfg.hotwords_score = 1.5;
    cfg.rule_fsts = empty.as_ptr();
    cfg.rule_fars = empty.as_ptr();
    let rec = unsafe { so::SherpaOnnxCreateOnlineRecognizer(&cfg) };
    if rec.is_null() {
        return Err("could not load the caption model".into());
    }
    Ok(Recognizer(rec))
}

/// A recognizer group with room, made when all are full (each made group gets its decoder thread).
fn group_for(model_dir: &Path) -> Result<Arc<Group>, String> {
    let groups = GROUPS.get_or_init(|| Mutex::new(Vec::new()));
    let mut gs = groups.lock().unwrap();
    if let Some(g) = gs.iter().min_by_key(|g| g.members.lock().unwrap().len()) {
        if g.members.lock().unwrap().len() < STREAMS_PER_RECOGNIZER {
            return Ok(g.clone());
        }
    }
    let g = Arc::new(Group { rec: new_recognizer(model_dir, 2)?, members: Mutex::new(HashMap::new()), tick: Mutex::new(()) });
    let gt = g.clone();
    thread::Builder::new().name("live-captions".into()).spawn(move || decoder_loop(gt)).map_err(|e| e.to_string())?;
    log!("caption recognizer {} ready", gs.len() + 1);
    gs.push(g.clone());
    Ok(g)
}

fn decoder_loop(g: Arc<Group>) {
    let mut pinned = u64::MAX;
    loop {
        thread::sleep(TICK);
        {
            let d = DECODER_CPUS.lock().unwrap();
            if d.0 != pinned && !d.1.is_empty() {
                crate::admission::pin_current_thread(&d.1);
                pinned = d.0;
            }
        }
        let _tick = g.tick.lock().unwrap();
        let active: Vec<Arc<Captioner>> = g.members.lock().unwrap().values().cloned().collect();
        if active.is_empty() {
            continue;
        }
        for c in &active {
            c.take_audio();
        }
        let rec = g.rec.0;
        let mut ready: Vec<&Arc<Captioner>> = active.iter().filter(|c| c.is_ready(rec)).collect();
        while !ready.is_empty() {
            let streams: Vec<*const so::OnlineStream> = ready.iter().map(|c| c.side.lock().unwrap().stream).collect();
            unsafe { so::SherpaOnnxDecodeMultipleOnlineStreams(rec, streams.as_ptr(), streams.len() as i32) };
            ready.retain(|c| c.is_ready(rec));
        }
        for c in &active {
            c.publish(rec, false);
        }
    }
}

fn dbfs(x: &[f32]) -> f32 {
    let ms = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len().max(1) as f64;
    (20.0 * (ms.sqrt() + 1e-9).log10()) as f32
}

/// The model writes lowercase without punctuation: capitalise the line and "I" (alone, or in I'm, I've, I'll, I'd).
pub fn tidy(text: &str) -> String {
    let t = text.trim();
    let chars: Vec<char> = t.chars().collect();
    let mut out = String::with_capacity(t.len());
    for (i, &ch) in chars.iter().enumerate() {
        let word_start = i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '_');
        if ch == 'i' && word_start {
            let rest: String = chars[i + 1..].iter().collect();
            let after_ok = |s: &str| s.is_empty() || !s.chars().next().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false);
            let ok = rest.is_empty()
                || rest.starts_with(char::is_whitespace)
                || ["'m", "'ve", "'ll", "'d"].iter().any(|suf| rest.starts_with(suf) && after_ok(&rest[suf.len()..]));
            out.push(if ok { 'I' } else { ch });
        } else {
            out.push(ch);
        }
    }
    let mut c = out.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

impl Captioner {
    pub fn new(model_dir: &Path, emit: Emit) -> Result<Arc<Self>, String> {
        let group = group_for(model_dir)?;
        let stream = unsafe { so::SherpaOnnxCreateOnlineStream(group.rec.0) };
        if stream.is_null() {
            return Err("could not start a caption stream".into());
        }
        let c = Arc::new(Self {
            group: group.clone(),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            shared: Mutex::new(Shared { pending: Vec::new() }),
            side: Mutex::new(DecoderSide { stream, to16k: Decimate3::new(), fed: 0, line_start: 0, last_text: String::new(), line: 0, behind: false }),
            emit,
            closed: AtomicBool::new(false),
            gate: Mutex::new(Gate { quiet: 0.0, gated: false, preroll: Vec::new(), skipped: 0, skipped_total: 0 }),
        });
        group.members.lock().unwrap().insert(c.id, c.clone());
        Ok(c)
    }

    /// `level_dbfs`: how loud the speech in this frame is (the denoised frame when noise removal runs), or the
    /// frame's own level.
    pub fn feed(&self, x48: &[f32], level_dbfs: Option<f32>) {
        let level = level_dbfs.unwrap_or_else(|| dbfs(x48));
        let mut g = self.gate.lock().unwrap();
        if level >= SPEECH_DBFS {
            g.quiet = 0.0;
            let mut sh = self.shared.lock().unwrap();
            if g.gated {
                g.gated = false;
                sh.pending.push(Piece::Skip(g.skipped));
                for p in g.preroll.drain(..) {
                    sh.pending.push(Piece::Audio(p));
                }
                if g.skipped > 0 {
                    (self.emit)(json!({"type": "_idle", "samples": g.skipped})); // not decoded: a captions-only session isn't billed for it
                }
                g.skipped = 0;
            }
            sh.pending.push(Piece::Audio(x48.to_vec()));
            return;
        }
        g.quiet += x48.len() as f32 / SAMPLE_RATE as f32;
        if g.quiet <= GATE_AFTER_S {
            self.shared.lock().unwrap().pending.push(Piece::Audio(x48.to_vec()));
            return;
        }
        g.gated = true;
        g.preroll.push(x48.to_vec());
        if g.preroll.len() > PREROLL_FRAMES {
            let n = g.preroll.remove(0).len();
            g.skipped += n;
            g.skipped_total += n;
        }
    }

    /// `n` samples of silence the client didn't send (a muted source, digital silence): the line still gets its
    /// trailing silence (so it ends), the rest only moves the clock. Not billed in the first place.
    pub fn silence(&self, n: usize) {
        let mut g = self.gate.lock().unwrap();
        let need = if g.gated { 0 } else { (((GATE_AFTER_S - g.quiet) * SAMPLE_RATE as f32).max(0.0)) as usize };
        let fed = n.min(need);
        g.quiet += n as f32 / SAMPLE_RATE as f32;
        let mut sh = self.shared.lock().unwrap();
        if fed > 0 {
            sh.pending.push(Piece::Audio(vec![0.0; fed]));
        }
        if n > fed {
            g.gated = true;
            sh.pending.push(Piece::Skip(n - fed));
            g.skipped_total += n - fed;
        }
    }

    fn is_ready(&self, rec: *const so::OnlineRecognizer) -> bool {
        let s = self.side.lock().unwrap();
        unsafe { so::SherpaOnnxIsOnlineStreamReady(rec, s.stream) != 0 }
    }

    fn take_audio(&self) {
        let mut items = std::mem::take(&mut self.shared.lock().unwrap().pending);
        if items.is_empty() {
            return;
        }
        let mut side = self.side.lock().unwrap();
        let total: usize = items.iter().map(|p| if let Piece::Audio(a) = p { a.len() } else { 0 }).sum();
        if total > MAX_BACKLOG {
            // More than ~2 s behind (too many streams, or the box is busy): drop the oldest audio, say so once.
            let skips: usize = items.iter().map(|p| if let Piece::Skip(n) = p { *n } else { 0 }).sum();
            let mut kept: Vec<Vec<f32>> = Vec::new();
            let mut have = 0;
            for p in items.iter().rev() {
                if let Piece::Audio(a) = p {
                    if have >= KEEP_BACKLOG {
                        break;
                    }
                    let take = (KEEP_BACKLOG - have).min(a.len());
                    kept.push(a[a.len() - take..].to_vec());
                    have += take;
                }
            }
            kept.reverse();
            items = vec![Piece::Skip(total - have + skips)];
            items.extend(kept.into_iter().map(Piece::Audio));
            if !side.behind {
                side.behind = true;
                (self.emit)(json!({"type": "warning", "code": "captions_behind", "message": CAPTIONS_BEHIND_MESSAGE}));
            }
        } else if total < RECOVERED_BACKLOG {
            side.behind = false;
        }
        let mut run: Vec<f32> = Vec::new();
        items.push(Piece::Skip(0));
        for p in items {
            match p {
                Piece::Audio(a) => run.extend_from_slice(&a),
                Piece::Skip(n) => {
                    if !run.is_empty() {
                        let x = side.to16k.process(&run);
                        unsafe { so::SherpaOnnxOnlineStreamAcceptWaveform(side.stream, INPUT_RATE as i32, x.as_ptr(), x.len() as i32) };
                        side.fed += x.len() as u64;
                        run.clear();
                    }
                    side.fed += (n / 3) as u64; // skipped: the captions' times keep following the talk
                }
            }
        }
    }

    fn publish(&self, rec: *const so::OnlineRecognizer, last: bool) {
        let mut side = self.side.lock().unwrap();
        let raw = unsafe {
            let j = so::SherpaOnnxGetOnlineStreamResultAsJson(rec, side.stream);
            if j.is_null() {
                String::new()
            } else {
                let s = CStr::from_ptr(j).to_string_lossy().into_owned();
                so::SherpaOnnxDestroyOnlineStreamResultJson(j);
                s
            }
        };
        let text = serde_json::from_str::<Value>(&raw).ok().and_then(|v| v.get("text").and_then(|t| t.as_str()).map(tidy)).unwrap_or_default();
        let endpoint = last || unsafe { so::SherpaOnnxOnlineStreamIsEndpoint(rec, side.stream) != 0 };
        if !text.is_empty() && (endpoint || text != side.last_text) && !self.closed.load(Ordering::SeqCst) {
            let r = INPUT_RATE as f64;
            (self.emit)(json!({
                "type": "caption", "id": format!("l{}", side.line), "text": text, "final": endpoint,
                "start": ((side.line_start as f64 / r) * 1000.0).round() / 1000.0,
                "end": ((side.fed as f64 / r) * 1000.0).round() / 1000.0, "language": "en",
            }));
        }
        side.last_text = text.clone();
        if endpoint {
            unsafe { so::SherpaOnnxOnlineStreamReset(rec, side.stream) };
            side.line_start = side.fed;
            side.last_text.clear();
            if !text.is_empty() {
                side.line += 1;
            }
        }
    }

    /// End of stream: decode what is left (up to `timeout` for the decoder to catch up) and emit the last line.
    pub fn flush(&self, timeout: Duration) {
        {
            let mut g = self.gate.lock().unwrap();
            if g.gated {
                let skipped = g.skipped + g.preroll.iter().map(|p| p.len()).sum::<usize>();
                if skipped > 0 {
                    (self.emit)(json!({"type": "_idle", "samples": skipped}));
                }
                g.preroll.clear();
                g.skipped = 0;
            }
        }
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline && !self.shared.lock().unwrap().pending.is_empty() {
            thread::sleep(TICK);
        }
        self.group.members.lock().unwrap().remove(&self.id);
        let _tick = self.group.tick.lock().unwrap(); // the decoder's current tick is over: the stream is ours now
        self.take_audio();
        let rec = self.group.rec.0;
        let stream = self.side.lock().unwrap().stream;
        unsafe {
            so::SherpaOnnxOnlineStreamInputFinished(stream);
            while so::SherpaOnnxIsOnlineStreamReady(rec, stream) != 0 {
                so::SherpaOnnxDecodeOnlineStream(rec, stream);
            }
        }
        self.publish(rec, true);
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.group.members.lock().unwrap().remove(&self.id);
        drop(self.group.tick.lock().unwrap()); // returns once the decoder thread no longer touches it
    }
}

impl Drop for Captioner {
    fn drop(&mut self) {
        let s = self.side.get_mut().map(|s| s.stream).unwrap_or(std::ptr::null());
        if !s.is_null() {
            unsafe { so::SherpaOnnxDestroyOnlineStream(s) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tidy;

    #[test]
    fn tidy_like_python() {
        assert_eq!(tidy("i think i'm right and i'll go i"), "I think I'm right and I'll go I");
        assert_eq!(tidy("it is ice"), "It is ice");
        assert_eq!(tidy("hi i'd say"), "Hi I'd say");
        assert_eq!(tidy("  "), "");
    }
}
