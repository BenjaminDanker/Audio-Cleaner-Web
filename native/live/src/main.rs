//! acw-live: every live session of Audio Cleaner in one native process.
//!
//! It replaces the Python stream worker (server/acw/stream/worker.py) behind the same Unix-socket protocol, so the
//! web process doesn't change. Why: Python needs the GIL around every frame, so one Python process fell behind at
//! about ten sessions and the work had to be split over several processes (each with its own caption model). Here a
//! session is a thread that does all of its per-frame work (Opus, echo cancellation, noise removal, filters) without
//! a lock, on its own core.
//!
//! Configuration is the same environment as the Python worker (ACW_HOME, ACW_DATA_DIR, ACW_BIN_DIR,
//! ACW_MODELS_DIR, ACW_STREAM_SOCKET, ACW_ENV, ACW_MIN_JOB_CORES, ACW_LIVE_UNITS_PER_CORE,
//! ACW_STREAM_MAX_CAPTION_STREAMS). `python -m acw.stream.worker` starts this binary when it is installed
//! (~/audio-cleaner/bin/acw-live) unless ACW_LIVE_ENGINE=python.

macro_rules! log {
    ($($arg:tt)*) => {{
        eprintln!("acw-live: {}", format!($($arg)*));
    }};
}

mod admission;
mod aec;
mod captions;
mod dsp;
mod ffi;
mod ipc;
mod sandbox;
mod session;

use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use admission::{Admission, HostLoad};

/// How far the echo canceller delays the mic: 432 samples (9 ms), measured with an impulse, the same as the
/// Python worker's canceller.
pub const AEC_DELAY: usize = 432;

/// Units one physical core carries with good latency in this engine (2026-10-16, Ryzen 7 1700X: see
/// docs/operations.md), at the reference single-session model cost; other CPUs are scaled by calibrate().
const REFERENCE_UNITS_PER_CORE: f64 = 4.05;
const REFERENCE_FRAME_COST: f64 = 0.110;
/// Standard caption streams per live core (each recognizer decodes ~13x real time; silence isn't decoded).
const CAPTION_STREAMS_PER_CORE: usize = 5;

pub struct Engine {
    pub lib_df: PathBuf,
    pub dfn_model: PathBuf,
    pub captions_dir: PathBuf,
    pub admission: Mutex<Admission>,
    pub host: HostLoad,
}

impl Engine {
    /// The physical core index of live core `i` (live cores are the first ones).
    pub fn core_index(&self, i: usize) -> usize {
        i
    }

    /// Keep the caption decoders on the cores that carry sessions (file jobs are moved off those).
    pub fn repin_decoders(&self) {
        captions::set_decoder_cpus(self.admission.lock().unwrap().busy_cpus());
    }
}

fn env_path(key: &str, default: PathBuf) -> PathBuf {
    std::env::var_os(key).map(PathBuf::from).unwrap_or(default)
}

/// Scale the reference capacity by how fast this CPU runs one session's model (the best of a few short runs).
fn calibrate(lib: &Path, model: &Path) -> f64 {
    if let Ok(v) = std::env::var("ACW_LIVE_UNITS_PER_CORE") {
        if let Ok(v) = v.parse::<f64>() {
            return v;
        }
    }
    let reference = if ffi::df_shares_model(lib) { REFERENCE_UNITS_PER_CORE } else { 2.4 };
    let Ok(mut df) = ffi::DeepFilter::new(lib, model, 100.0) else { return reference };
    let mut seed: u32 = 7;
    let x: Vec<f32> = (0..96_000)
        .map(|_| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.1
        })
        .collect();
    let mut y = vec![0f32; x.len()];
    df.process(&x[..48_000], &mut y[..48_000]);
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = thread_cpu_time();
        df.process(&x, &mut y);
        best = best.min((thread_cpu_time() - t) / 2.0);
    }
    reference * REFERENCE_FRAME_COST / best.max(1e-3)
}

fn thread_cpu_time() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn main() {
    let home = env_path("ACW_HOME", PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("audio-cleaner"));
    let data_dir = env_path("ACW_DATA_DIR", home.join("data"));
    let bin_dir = env_path("ACW_BIN_DIR", home.join("bin"));
    let models_dir = env_path("ACW_MODELS_DIR", home.join("models"));
    let sock = env_path("ACW_STREAM_SOCKET", data_dir.join("stream.sock"));
    let production = std::env::var("ACW_ENV").map(|v| v == "production").unwrap_or(false);
    let lib_df = bin_dir.join("libdf.so");
    let dfn_model = models_dir.join("DeepFilterNet3_onnx.tar.gz");

    // Everything that needs /sys, /proc or the libraries happens before the sandbox closes them off.
    let cores = admission::physical_cores();
    let live = admission::max_live_cores(cores.len());
    let live_cores: Vec<Vec<usize>> = cores[..live].to_vec();
    let live_cpus: Vec<usize> = live_cores.iter().flatten().cloned().collect();
    admission::pin_current_thread(&live_cpus); // every thread started from here on inherits it
    if !ffi::opus_available() {
        log!("libopus is missing: Opus sessions will fail");
    }
    let upc = calibrate(&lib_df, &dfn_model);
    let caption_cap = std::env::var("ACW_STREAM_MAX_CAPTION_STREAMS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v > 0).unwrap_or(CAPTION_STREAMS_PER_CORE * live);
    let host = HostLoad::open();
    host.sample(&live_cpus);
    log!("live capacity: {live} cores x {upc:.2} units, {caption_cap} caption streams; shared model: {}", ffi::df_shares_model(&lib_df));

    // Sockets left by the Python workers (stream-2.sock ...): this engine serves every core from one socket.
    if let (Some(dir), Some(stem)) = (sock.parent(), sock.file_stem().and_then(|s| s.to_str())) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with(&format!("{stem}-")) && name.ends_with(".sock") {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::remove_file(&sock);
    let old = unsafe { libc::umask(0o177) };
    let listener = match UnixListener::bind(&sock) {
        Ok(l) => l,
        Err(e) => {
            log!("cannot listen on {}: {e}", sock.display());
            std::process::exit(1);
        }
    };
    unsafe { libc::umask(old) };
    let _ = std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o600));

    let sherpa_dir = sandbox::sherpa_lib_dir();
    match sandbox::restrict(&[bin_dir.as_path(), models_dir.as_path()], sherpa_dir.as_deref()) {
        Ok(status) => log!("listening on {} (landlock: {status})", sock.display()),
        Err(e) if production => {
            log!("refusing to run without the sandbox in production: {e}");
            std::process::exit(1);
        }
        Err(e) => log!("listening on {} (no landlock: {e})", sock.display()),
    }

    let engine = Arc::new(Engine {
        lib_df,
        dfn_model,
        captions_dir: models_dir.join("captions-en"),
        admission: Mutex::new(Admission::new(live_cores, upc, caption_cap)),
        host,
    });
    engine.repin_decoders();
    unsafe {
        libc::signal(libc::SIGTERM, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t);
    }
    // Host load about once a second, off the accept loop.
    {
        let e = engine.clone();
        let cpus = live_cpus.clone();
        std::thread::Builder::new()
            .name("hostload".into())
            .spawn(move || loop {
                let t = Instant::now();
                e.host.sample(&cpus);
                admission::sleep_until_next_second(t);
            })
            .ok();
    }
    listener.set_nonblocking(true).ok();
    while !STOP.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).ok();
                let e = engine.clone();
                let spawned = std::thread::Builder::new().name("live-session".into()).spawn(move || {
                    // A panic ends this session only; the others go on.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session::handle(e, stream)));
                });
                if let Err(err) = spawned {
                    log!("could not start a session thread: {err}");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(e) => {
                log!("accept failed: {e}");
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }
    log!("stopping");
}
