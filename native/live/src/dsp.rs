//! Per-session processing: noise removal (DeepFilterNet3, an 80 Hz high-pass and a peak limiter), switchable
//! mid-session at a fixed delay, the room test's noise meter, and the 48 -> 16 kHz converter for the captions.

use std::path::{Path, PathBuf};

use crate::ffi::{model_delay, DeepFilter, HOP};

pub const SAMPLE_RATE: usize = 48_000;

/// butter(2, 80, btype="highpass", fs=48000, output="sos") from scipy, exactly.
const HPF: [f64; 6] = [0.9926225427561193, -1.9852450855122385, 0.9926225427561193, 1.0, -1.9851906578962617, 0.985299513128215];

/// Digital silence (a noise gate before our filter, a muted interface) isn't run through the model: once this many
/// samples of it have gone through (more than the model's 30 ms delay, so its buffers hold nothing but silence),
/// the output is silence too, until sound comes back.
const SILENT_PEAK: f32 = 1e-5;
const SILENT_SKIP_AFTER: usize = SAMPLE_RATE / 5;

/// A short crossfade from one signal to another (10 ms), for a part of the chain switched on or off mid-session.
pub struct Crossfade {
    to_wet: bool,
    pos: usize,
}

pub const FADE: usize = HOP;

impl Crossfade {
    pub fn new(to_wet: bool) -> Self {
        Self { to_wet, pos: 0 }
    }

    /// Turn around a fade that hasn't finished (switched back before it ended): it continues from where it is.
    pub fn reverse(&mut self) {
        self.to_wet = !self.to_wet;
        self.pos = FADE - self.pos.min(FADE);
    }

    pub fn to_wet(&self) -> bool {
        self.to_wet
    }

    /// `dry` and `wet` of one block mixed along the fade; true once it is over.
    pub fn mix(&mut self, dry: &[f32], wet: &[f32]) -> (Vec<f32>, bool) {
        let out = dry
            .iter()
            .zip(wet)
            .map(|(&d, &w)| {
                let t = (self.pos.min(FADE) as f32) / FADE as f32;
                self.pos += 1;
                let g = if self.to_wet { t } else { 1.0 - t };
                d * (1.0 - g) + w * g
            })
            .collect();
        (out, self.pos >= FADE)
    }
}

/// A fixed delay: what goes in comes out `delay` samples later (silence before the first input).
pub struct DelayLine {
    buf: Vec<f32>,
    delay: usize,
}

impl DelayLine {
    pub fn new(delay: usize) -> Self {
        Self { buf: vec![0.0; delay], delay }
    }

    pub fn process(&mut self, x: &[f32]) -> Vec<f32> {
        self.buf.extend_from_slice(x);
        let out: Vec<f32> = self.buf.drain(..x.len()).collect();
        debug_assert_eq!(self.buf.len(), self.delay);
        out
    }
}

/// The room test's noise figure (the plugin's "Test my room"), the same as pipeline.noise_figure_db and the website's
/// NoiseFigure: dB between the voice and what DeepFilterNet3 at full strength takes out, 99 when there is nothing to hear.
/// It is measured at the remover itself, whatever the session's strength: libDF limits the attenuation by mixing the input
/// back in, out = (1 - a) * full + a * input with a = 10^(-limit/20) (tract.rs, after the post filter), so with the input
/// `xd` lined up with the output `y`, full strength keeps (y - a*xd) / (1 - a) and takes out (xd - y) / (1 - a).
pub struct NoiseMeter {
    voice: f64,
    removed: f64,
    /// samples measured so far
    pub samples: usize,
}

impl NoiseMeter {
    pub fn new() -> Self {
        Self { voice: 0.0, removed: 0.0, samples: 0 }
    }

    /// One block: the remover's input, delayed by its latency, and its raw output at an attenuation limit of `atten_lim_db`.
    pub fn add(&mut self, xd: &[f32], y: &[f32], atten_lim_db: f32) {
        let a = if atten_lim_db >= 100.0 { 0.0 } else { 10f64.powf(-atten_lim_db as f64 / 20.0) };
        for (&x, &y) in xd.iter().zip(y) {
            let (x, y) = (x as f64, y as f64);
            self.voice += (y - a * x) * (y - a * x);
            self.removed += (x - y) * (x - y);
        }
        self.samples += xd.len().min(y.len());
    }

    pub fn figure_db(&self) -> f64 {
        if self.voice + self.removed <= 0.0 {
            return 99.0;
        }
        (10.0 * ((self.voice + 1e-12) / (self.removed + 1e-12)).log10()).min(99.0)
    }
}

/// Noise removal (DeepFilterNet3, an 80 Hz high-pass and a peak limiter), switchable mid-session. The output is always
/// delayed by the model's delay, whether the model runs or not, so a client's timing never changes with the settings:
/// without the model the audio passes through a delay line of the same length.
pub struct LiveChain {
    lib: PathBuf,
    model: PathBuf,
    atten_lim_db: f32,
    df: Option<DeepFilter>,
    pub latency_samples: usize,
    dry: DelayLine,
    history: Vec<f32>, // the latest input, to warm the model up when it is switched on
    fade: Option<Crossfade>,
    z: [f64; 2], // the high-pass state (direct form II transposed, like scipy's sosfilt)
    gain: f32,
    ceiling: f32,
    release: f32,
    silent_run: usize,
    pub skipped: usize,
    y: Vec<f32>,
    /// the room test, while one runs (session.rs sets it): fed with every block the model hears
    pub meter: Option<NoiseMeter>,
}

/// How much of the latest input a model switched on mid-session hears first (its output for it is thrown away): its
/// state then follows the room's noise when it starts being heard.
const WARM_UP: usize = 30 * HOP;

impl LiveChain {
    pub fn new(lib: &Path, model: &Path, denoise: bool, atten_lim_db: f32) -> Result<Self, String> {
        let latency_samples = model_delay(model)?;
        let df = if denoise { Some(DeepFilter::new(lib, model, atten_lim_db)?) } else { None };
        Ok(Self {
            lib: lib.to_path_buf(),
            model: model.to_path_buf(),
            atten_lim_db,
            df,
            latency_samples,
            dry: DelayLine::new(latency_samples),
            history: Vec::new(),
            fade: None,
            z: [0.0; 2],
            gain: 1.0,
            ceiling: 10f32.powf(-1.0 / 20.0),
            release: 10f32.powf(6.0 / 20.0 / 100.0), // recover ~6 dB per second at 100 hops/s
            silent_run: 0,
            skipped: 0,
            y: Vec::new(),
            meter: None,
        })
    }

    /// Noise removal is on (or fading in).
    pub fn denoising(&self) -> bool {
        match &self.fade {
            Some(f) => f.to_wet(),
            None => self.df.is_some(),
        }
    }

    pub fn set_atten_lim(&mut self, db: f32) {
        self.atten_lim_db = db.clamp(6.0, 100.0);
        if let Some(df) = self.df.as_mut() {
            df.set_atten_lim(self.atten_lim_db);
        }
    }

    /// Switch noise removal on or off from the next block, with a 10 ms crossfade.
    pub fn set_denoise(&mut self, on: bool) -> Result<(), String> {
        if on == self.denoising() {
            return Ok(());
        }
        if let Some(f) = self.fade.as_mut() {
            f.reverse(); // switched back before the last change finished fading
            return Ok(());
        }
        if on {
            let mut df = DeepFilter::new(&self.lib, &self.model, self.atten_lim_db)?;
            let n = (self.history.len() / HOP) * HOP;
            if n > 0 {
                let warm = &self.history[self.history.len() - n..];
                let mut out = vec![0f32; n];
                df.process(warm, &mut out);
            }
            self.df = Some(df);
            self.silent_run = 0;
        }
        self.fade = Some(Crossfade::new(on));
        Ok(())
    }

    /// N*480 samples in, N*480 out, delayed by `latency_samples`.
    pub fn process(&mut self, x: &[f32]) -> Vec<f32> {
        let dry = self.dry.process(x);
        self.history.extend_from_slice(x);
        if self.history.len() > 2 * WARM_UP {
            self.history.drain(..self.history.len() - WARM_UP);
        }
        if self.df.is_none() {
            return dry;
        }
        let wet = self.denoise(x, &dry);
        let Some(fade) = self.fade.as_mut() else { return wet };
        let (out, done) = fade.mix(&dry, &wet);
        if done {
            if !fade.to_wet() {
                self.df = None;
            }
            self.fade = None;
        }
        out
    }

    /// `dry`: the same block delayed by the model's latency, lined up with its output (for the room test).
    fn denoise(&mut self, x: &[f32], dry: &[f32]) -> Vec<f32> {
        let df = self.df.as_mut().expect("the model runs");
        if x.iter().all(|v| v.abs() < SILENT_PEAK) {
            self.silent_run += x.len();
            if self.silent_run > SILENT_SKIP_AFTER {
                self.skipped += x.len();
                return vec![0.0; x.len()];
            }
        } else {
            self.silent_run = 0;
        }
        self.y.resize(x.len(), 0.0);
        df.process(x, &mut self.y);
        if let Some(m) = self.meter.as_mut() {
            m.add(dry, &self.y, self.atten_lim_db);
        }
        let [b0, b1, b2, _, a1, a2] = HPF;
        let mut out = vec![0f32; x.len()];
        for (o, &v) in out.iter_mut().zip(self.y.iter()) {
            let xi = v as f64;
            let yi = b0 * xi + self.z[0];
            self.z[0] = b1 * xi - a1 * yi + self.z[1];
            self.z[1] = b2 * xi - a2 * yi;
            *o = yi as f32;
        }
        for hop in out.chunks_mut(HOP) {
            let peak = hop.iter().fold(0f32, |m, v| m.max(v.abs()));
            let target = if peak > 0.0 { (self.ceiling / peak).min(1.0) } else { 1.0 };
            let new = target.min(self.gain * self.release).min(1.0);
            let n = hop.len();
            for (i, v) in hop.iter_mut().enumerate() {
                let g = if new < self.gain {
                    // np.linspace(gain, new, n): both ends included
                    if n > 1 { self.gain + (new - self.gain) * (i as f32) / ((n - 1) as f32) } else { new }
                } else {
                    new
                };
                *v = (*v * g).clamp(-self.ceiling, self.ceiling);
            }
            self.gain = new;
        }
        out
    }
}

/// resample_poly(x, 1, 3)'s filter: firwin(61, 1/3, window=("kaiser", 5.0)), float32, from scipy.
const DECIM_TAPS: [f32; 61] = [
    -4.773070465844962e-19, -0.0005075758090242743, -0.0007171615143306553, 1.2363095201957368e-18, 0.001275830902159214,
    0.0016363257309421897, -2.32963673560862e-18, -0.002551690209656954, -0.003121165791526437, 3.7379654952199655e-18,
    0.0045264228247106075, 0.005382746923714876, -5.394486902495703e-18, -0.00746728852391243, -0.008729113265872002,
    7.188058902381676e-18, 0.011807636357843876, 0.013690670020878315, -8.973913720742836e-18, -0.01839848794043064,
    -0.021385829895734787, 1.0590672969563456e-17, 0.029341038316488266, 0.03484134003520012, -1.188119135761255e-17,
    -0.051828090101480484, -0.06626323610544205, 1.2713991835247262e-17, 0.13655221462249756, 0.2751477360725403,
    0.33353540301322937, 0.2751477360725403, 0.13655221462249756, 1.2713991835247262e-17, -0.06626323610544205,
    -0.051828090101480484, -1.188119135761255e-17, 0.03484134003520012, 0.029341038316488266, 1.0590672969563456e-17,
    -0.021385829895734787, -0.01839848794043064, -8.973913720742836e-18, 0.013690670020878315, 0.011807636357843876,
    7.188058902381676e-18, -0.008729113265872002, -0.00746728852391243, -5.394486902495703e-18, 0.005382746923714876,
    0.0045264228247106075, 3.7379654952199655e-18, -0.003121165791526437, -0.002551690209656954, -2.32963673560862e-18,
    0.0016363257309421897, 0.001275830902159214, 1.2363095201957368e-18, -0.0007171615143306553, -0.0005075758090242743,
    -4.773070465844962e-19,
];
const HALF: usize = 30;

/// 48 kHz -> 16 kHz, one batch at a time, as if the whole stream were converted in one go (the filter state carries
/// over, so batch edges add nothing).
pub struct Decimate3 {
    buf: Vec<f32>,
    next: usize,
}

impl Decimate3 {
    pub fn new() -> Self {
        Self { buf: vec![0.0; HALF], next: HALF }
    }

    pub fn process(&mut self, x: &[f32]) -> Vec<f32> {
        self.buf.extend_from_slice(x);
        if self.buf.len() < HALF + 1 || self.buf.len() - 1 - HALF < self.next {
            return Vec::new();
        }
        let last = self.buf.len() - 1 - HALF;
        let mut out = Vec::with_capacity((last - self.next) / 3 + 1);
        let mut c = self.next;
        while c <= last {
            // np.convolve(..., h, "valid") at centre c: sum_k buf[c - HALF + k] * h[60 - k]
            let w = &self.buf[c - HALF..=c + HALF];
            let mut s = 0f32;
            for k in 0..=2 * HALF {
                s += w[k] * DECIM_TAPS[2 * HALF - k];
            }
            out.push(s);
            c += 3;
        }
        let keep_from = c - HALF; // the next window starts here
        self.buf.drain(..keep_from);
        self.next = HALF;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimator_is_the_same_in_pieces_as_in_one_go() {
        let x: Vec<f32> = (0..48_000).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
        let whole = Decimate3::new().process(&x);
        let mut d = Decimate3::new();
        let mut parts = Vec::new();
        for chunk in x.chunks(977) {
            parts.extend(d.process(chunk));
        }
        assert_eq!(whole.len(), parts.len());
        assert!(whole.iter().zip(&parts).all(|(a, b)| (a - b).abs() < 1e-6));
        assert!((whole.len() as i64 - 16_000).abs() <= 11);
    }

    #[test]
    fn delay_line_delays_by_its_length() {
        let mut d = DelayLine::new(3);
        assert_eq!(d.process(&[1.0, 2.0]), vec![0.0, 0.0]);
        assert_eq!(d.process(&[3.0, 4.0]), vec![0.0, 1.0]);
    }

    /// libdf and the model where the engine is installed (~/audio-cleaner, or ACW_BIN_DIR / ACW_MODELS_DIR); None when absent.
    fn libdf_and_model() -> Option<(PathBuf, PathBuf)> {
        let home = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("audio-cleaner");
        let dir = |k: &str, d: &str| std::env::var_os(k).map(PathBuf::from).unwrap_or_else(|| home.join(d));
        let lib = dir("ACW_BIN_DIR", "bin").join("libdf.so");
        let model = dir("ACW_MODELS_DIR", "models").join("DeepFilterNet3_onnx.tar.gz");
        (lib.is_file() && model.is_file()).then_some((lib, model))
    }

    #[test]
    fn the_room_test_reads_the_full_strength_figure_at_any_strength() {
        let Some((lib, model)) = libdf_and_model() else {
            eprintln!("skipped: libdf or the model is not installed");
            return;
        };
        // a harmonic voice in words, under white noise about 10 dB down
        let mut seed = 1u32;
        let mut noise = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32 - 0.5
        };
        let x: Vec<f32> = (0..8 * SAMPLE_RATE)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                let word = (std::f32::consts::PI * 2.5 * t).sin().max(0.0);
                let v: f32 = (1..12).map(|k| (2.0 * std::f32::consts::PI * 140.0 * k as f32 * t).sin() / k as f32).sum();
                0.1 * v * word + 0.03 * noise()
            })
            .collect();
        let figure = |atten: f32| {
            let mut chain = LiveChain::new(&lib, &model, true, atten).unwrap();
            chain.meter = Some(NoiseMeter::new());
            for block in x.chunks(4 * HOP) {
                chain.process(block);
            }
            let m = chain.meter.unwrap();
            assert_eq!(m.samples, x.len());
            m.figure_db()
        };
        let (gentle, full) = (figure(12.0), figure(100.0));
        assert!(full < 30.0, "{full}");
        assert!((gentle - full).abs() < 0.5, "{gentle} at 12 dB, {full} at 100 dB");
        let mut quiet = NoiseMeter::new();
        quiet.add(&[0.0; HOP], &[0.0; HOP], 100.0);
        assert_eq!(quiet.figure_db(), 99.0);
    }

    #[test]
    fn crossfade_goes_from_dry_to_wet_in_one_hop_and_can_turn_around() {
        let dry = vec![0.0f32; FADE];
        let wet = vec![1.0f32; FADE];
        let mut f = Crossfade::new(true);
        let (half, done) = f.mix(&dry[..FADE / 2], &wet[..FADE / 2]);
        assert!(!done && half[0] == 0.0 && half.windows(2).all(|w| w[1] >= w[0]));
        f.reverse();
        let (rest, done) = f.mix(&dry[..FADE / 2], &wet[..FADE / 2]);
        assert!(done && (rest[0] - half[FADE / 2 - 1]).abs() < 0.01 && rest.last().unwrap() < &0.01);
    }
}
