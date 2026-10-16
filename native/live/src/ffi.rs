//! The two C libraries the Python worker also uses, loaded at run time the same way (no build-time dependency):
//! libdf (DeepFilterNet3, native/deepfilter: one copy of the model weights shared by every instance) and libopus.

use std::ffi::{c_char, c_float, c_int, c_void, CString};
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;

use libloading::Library;

pub const HOP: usize = 480;

struct DfLib {
    _lib: Library,
    create: unsafe extern "C" fn(*const c_char, c_float, c_float, c_float, c_float, c_float, *const c_char) -> *mut c_void,
    process_frame: unsafe extern "C" fn(*mut c_void, *mut c_float, *mut c_float) -> c_float,
    set_atten_lim: unsafe extern "C" fn(*mut c_void, c_float),
    free: unsafe extern "C" fn(*mut c_void),
    shares_model: bool,
}

static DF: OnceLock<Result<DfLib, String>> = OnceLock::new();

fn df_lib(path: &Path) -> Result<&'static DfLib, String> {
    DF.get_or_init(|| unsafe {
        let lib = Library::new(path).map_err(|e| format!("libdf: {e}"))?;
        let create = *lib
            .get::<unsafe extern "C" fn(*const c_char, c_float, c_float, c_float, c_float, c_float, *const c_char) -> *mut c_void>(b"df_create_with_params\0")
            .map_err(|e| format!("libdf: {e}"))?;
        let process_frame = *lib
            .get::<unsafe extern "C" fn(*mut c_void, *mut c_float, *mut c_float) -> c_float>(b"df_process_frame\0")
            .map_err(|e| format!("libdf: {e}"))?;
        let set_atten_lim = *lib.get::<unsafe extern "C" fn(*mut c_void, c_float)>(b"df_set_atten_lim\0").map_err(|e| format!("libdf: {e}"))?;
        let free = *lib.get::<unsafe extern "C" fn(*mut c_void)>(b"df_free\0").map_err(|e| format!("libdf: {e}"))?;
        let shares_model = lib.get::<unsafe extern "C" fn() -> c_int>(b"df_shares_model\0").is_ok();
        Ok(DfLib { _lib: lib, create, process_frame, set_atten_lim, free, shares_model })
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Whether this libdf build shares one copy of the model between instances (df_shares_model).
pub fn df_shares_model(path: &Path) -> bool {
    df_lib(path).map(|l| l.shares_model).unwrap_or(false)
}

/// One stateful DeepFilterNet3 instance (one mono stream, 48 kHz, 480-sample hops). Both decoder stages always run
/// (skipping them on "clean" frames measurably hurts quality: acw_common/dfn.py).
pub struct DeepFilter {
    lib: &'static DfLib,
    st: *mut c_void,
    pub delay: usize,
}

// The state is only used by the thread that owns the session.
unsafe impl Send for DeepFilter {}

impl DeepFilter {
    pub fn new(lib_path: &Path, model: &Path, atten_lim_db: f32) -> Result<Self, String> {
        let lib = df_lib(lib_path)?;
        let delay = model_delay(model)?;
        let p = CString::new(model.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let st = unsafe { (lib.create)(p.as_ptr(), atten_lim_db, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY, 0.0, std::ptr::null()) };
        if st.is_null() {
            return Err("could not load DeepFilterNet model".into());
        }
        Ok(Self { lib, st, delay })
    }

    pub fn set_atten_lim(&mut self, db: f32) {
        unsafe { (self.lib.set_atten_lim)(self.st, db) }
    }

    /// `x.len()` must be a multiple of HOP; returns as many samples, delayed by `delay`.
    pub fn process(&mut self, x: &[f32], out: &mut [f32]) {
        let mut inp = [0f32; HOP];
        for (i, chunk) in x.chunks_exact(HOP).enumerate() {
            inp.copy_from_slice(chunk);
            unsafe { (self.lib.process_frame)(self.st, inp.as_mut_ptr(), out[i * HOP..(i + 1) * HOP].as_mut_ptr()) };
        }
    }
}

impl Drop for DeepFilter {
    fn drop(&mut self) {
        unsafe { (self.lib.free)(self.st) }
    }
}

/// Output delay of a libDF model tarball, from its config.ini: the STFT overlap (fft_size - hop_size) plus the
/// model's lookahead in hops (acw_common/dfn.py model_delay).
pub fn model_delay(model: &Path) -> Result<usize, String> {
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, usize>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(d) = cache.lock().unwrap().get(model) {
        return Ok(*d);
    }
    let f = std::fs::File::open(model).map_err(|e| format!("{}: {e}", model.display()))?;
    let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(f));
    let mut ini = String::new();
    for entry in ar.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        if entry.path().map_err(|e| e.to_string())?.to_string_lossy().ends_with("config.ini") {
            entry.read_to_string(&mut ini).map_err(|e| e.to_string())?;
            break;
        }
    }
    let get = |section: &str, key: &str| -> Option<usize> {
        let mut in_section = false;
        for line in ini.lines() {
            let l = line.trim();
            if l.starts_with('[') {
                in_section = l.trim_matches(|c| c == '[' || c == ']').eq_ignore_ascii_case(section);
            } else if in_section {
                if let Some((k, v)) = l.split_once('=') {
                    if k.trim() == key {
                        return v.trim().parse().ok();
                    }
                }
            }
        }
        None
    };
    let hop = get("df", "hop_size").ok_or("config.ini: no hop_size")?;
    let fft = get("df", "fft_size").ok_or("config.ini: no fft_size")?;
    let lookahead = get("df", "df_lookahead").unwrap_or(0).max(get("deepfilternet", "conv_lookahead").unwrap_or(0));
    let d = fft - hop + lookahead * hop;
    cache.lock().unwrap().insert(model.to_path_buf(), d);
    Ok(d)
}

// ------------------------------------------------------------------------------------------------- Opus

struct OpusLib {
    _lib: Library,
    dec_create: unsafe extern "C" fn(i32, c_int, *mut c_int) -> *mut c_void,
    decode_float: unsafe extern "C" fn(*mut c_void, *const u8, i32, *mut c_float, c_int, c_int) -> c_int,
    dec_destroy: unsafe extern "C" fn(*mut c_void),
    enc_create: unsafe extern "C" fn(i32, c_int, c_int, *mut c_int) -> *mut c_void,
    encode_float: unsafe extern "C" fn(*mut c_void, *const c_float, c_int, *mut u8, i32) -> i32,
    enc_ctl: unsafe extern "C" fn(*mut c_void, c_int, ...) -> c_int,
    enc_destroy: unsafe extern "C" fn(*mut c_void),
}

static OPUS: OnceLock<Result<OpusLib, String>> = OnceLock::new();

fn opus_lib() -> Result<&'static OpusLib, String> {
    OPUS.get_or_init(|| unsafe {
        let lib = Library::new("libopus.so.0").map_err(|e| format!("libopus: {e}"))?;
        macro_rules! sym {
            ($name:literal, $t:ty) => {
                *lib.get::<$t>($name).map_err(|e| format!("libopus: {e}"))?
            };
        }
        Ok(OpusLib {
            dec_create: sym!(b"opus_decoder_create\0", unsafe extern "C" fn(i32, c_int, *mut c_int) -> *mut c_void),
            decode_float: sym!(b"opus_decode_float\0", unsafe extern "C" fn(*mut c_void, *const u8, i32, *mut c_float, c_int, c_int) -> c_int),
            dec_destroy: sym!(b"opus_decoder_destroy\0", unsafe extern "C" fn(*mut c_void)),
            enc_create: sym!(b"opus_encoder_create\0", unsafe extern "C" fn(i32, c_int, c_int, *mut c_int) -> *mut c_void),
            encode_float: sym!(b"opus_encode_float\0", unsafe extern "C" fn(*mut c_void, *const c_float, c_int, *mut u8, i32) -> i32),
            enc_ctl: sym!(b"opus_encoder_ctl\0", unsafe extern "C" fn(*mut c_void, c_int, ...) -> c_int),
            enc_destroy: sym!(b"opus_encoder_destroy\0", unsafe extern "C" fn(*mut c_void)),
            _lib: lib,
        })
    })
    .as_ref()
    .map_err(|e| e.clone())
}

pub fn opus_available() -> bool {
    opus_lib().is_ok()
}

pub const OPUS_FRAME: usize = 960; // 20 ms: the only packet size clients send
const OPUS_APPLICATION_AUDIO: c_int = 2049;
const OPUS_SET_BITRATE: c_int = 4002;
const OPUS_SET_VBR: c_int = 4006;
const OPUS_SET_COMPLEXITY: c_int = 4010;

pub struct OpusDecoder {
    lib: &'static OpusLib,
    st: *mut c_void,
}
unsafe impl Send for OpusDecoder {}

impl OpusDecoder {
    pub fn new() -> Result<Self, String> {
        let lib = opus_lib()?;
        let mut err: c_int = 0;
        let st = unsafe { (lib.dec_create)(48_000, 1, &mut err) };
        if st.is_null() || err != 0 {
            return Err("opus decoder".into());
        }
        Ok(Self { lib, st })
    }

    /// One 20 ms packet -> 960 samples; anything else (a longer packet, garbage) is an error.
    pub fn decode(&mut self, pkt: &[u8], out: &mut [f32; OPUS_FRAME]) -> Result<(), String> {
        let n = unsafe { (self.lib.decode_float)(self.st, pkt.as_ptr(), pkt.len() as i32, out.as_mut_ptr(), OPUS_FRAME as c_int, 0) };
        if n != OPUS_FRAME as c_int {
            return Err("opus packets must be 20 ms".into());
        }
        Ok(())
    }
}

impl Drop for OpusDecoder {
    fn drop(&mut self) {
        unsafe { (self.lib.dec_destroy)(self.st) }
    }
}

pub struct OpusEncoder {
    lib: &'static OpusLib,
    st: *mut c_void,
    buf: Vec<u8>,
}
unsafe impl Send for OpusEncoder {}

impl OpusEncoder {
    pub fn new(bitrate: i32) -> Result<Self, String> {
        let lib = opus_lib()?;
        let mut err: c_int = 0;
        let st = unsafe { (lib.enc_create)(48_000, 1, OPUS_APPLICATION_AUDIO, &mut err) };
        if st.is_null() || err != 0 {
            return Err("opus encoder".into());
        }
        unsafe {
            (lib.enc_ctl)(st, OPUS_SET_BITRATE, bitrate.clamp(24_000, 160_000) as c_int);
            (lib.enc_ctl)(st, OPUS_SET_VBR, 1 as c_int);
            // At 96 kbps the default (10) buys nothing audible and costs about twice the encode time.
            (lib.enc_ctl)(st, OPUS_SET_COMPLEXITY, 5 as c_int);
        }
        Ok(Self { lib, st, buf: vec![0u8; 4000] })
    }

    pub fn encode(&mut self, y: &[f32]) -> Result<&[u8], String> {
        let clipped: Vec<f32> = y.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        let n = unsafe { (self.lib.encode_float)(self.st, clipped.as_ptr(), clipped.len() as c_int, self.buf.as_mut_ptr(), self.buf.len() as i32) };
        if n < 0 {
            return Err(format!("opus encode failed ({n})"));
        }
        Ok(&self.buf[..n as usize])
    }
}

impl Drop for OpusEncoder {
    fn drop(&mut self) {
        unsafe { (self.lib.enc_destroy)(self.st) }
    }
}
