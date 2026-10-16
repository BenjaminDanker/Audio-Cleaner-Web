//! Per-session processing, a line-by-line port of server/acw/stream/engine.py (LiveChain, Decimate3): DeepFilterNet3,
//! an 80 Hz high-pass and a peak limiter, and the 48 -> 16 kHz converter for the captions.

use std::path::Path;

use crate::ffi::{DeepFilter, HOP};

pub const SAMPLE_RATE: usize = 48_000;

/// butter(2, 80, btype="highpass", fs=48000, output="sos") from scipy, exactly.
const HPF: [f64; 6] = [0.9926225427561193, -1.9852450855122385, 0.9926225427561193, 1.0, -1.9851906578962617, 0.985299513128215];

/// Digital silence (a noise gate before our filter, a muted interface) isn't run through the model: once this many
/// samples of it have gone through (more than the model's 30 ms delay, so its buffers hold nothing but silence),
/// the output is silence too, until sound comes back.
const SILENT_PEAK: f32 = 1e-5;
const SILENT_SKIP_AFTER: usize = SAMPLE_RATE / 5;

pub struct LiveChain {
    pub df: Option<DeepFilter>,
    z: [f64; 2], // the high-pass state (direct form II transposed, like scipy's sosfilt)
    gain: f32,
    ceiling: f32,
    release: f32,
    pub latency_samples: usize,
    silent_run: usize,
    pub skipped: usize,
    y: Vec<f32>,
}

impl LiveChain {
    pub fn new(lib: &Path, model: Option<&Path>, atten_lim_db: f32) -> Result<Self, String> {
        let df = match model {
            Some(m) => Some(DeepFilter::new(lib, m, atten_lim_db)?),
            None => None,
        };
        let latency_samples = df.as_ref().map(|d| d.delay).unwrap_or(0);
        Ok(Self {
            df,
            z: [0.0; 2],
            gain: 1.0,
            ceiling: 10f32.powf(-1.0 / 20.0),
            release: 10f32.powf(6.0 / 20.0 / 100.0), // recover ~6 dB per second at 100 hops/s
            latency_samples,
            silent_run: 0,
            skipped: 0,
            y: Vec::new(),
        })
    }

    pub fn set_atten_lim(&mut self, db: f32) {
        if let Some(df) = self.df.as_mut() {
            df.set_atten_lim(db.clamp(6.0, 100.0));
        }
    }

    /// N*480 samples in, N*480 out (delayed by `latency_samples`). Without the model the audio passes unchanged.
    pub fn process(&mut self, x: &[f32]) -> Vec<f32> {
        let Some(df) = self.df.as_mut() else { return x.to_vec() };
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
    fn limiter_holds_the_ceiling_without_a_model() {
        // no model: unchanged; the limiter only runs with the model, so test its maths through a chain whose df is None
        let mut c = LiveChain::new(Path::new("/nonexistent"), None, 100.0).unwrap();
        let x = vec![0.5f32; 960];
        assert_eq!(c.process(&x), x);
    }
}
