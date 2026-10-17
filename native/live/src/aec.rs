//! Acoustic echo cancellation: WebRTC's AEC3 through `sonora` (a pure-Rust port, BSD-3-Clause). It needs the
//! far-end ("reference") signal: what went to the speakers, which the OBS plugin sends alongside the mic.
//! Measured on our synthetic room echo (server/tests/test_native_live.py's room): 42.8 dB of echo removed, the
//! near-end voice at 0.94 of its level, 1.1% of a core (the canceller we used before did 38 dB at ~2%).

use sonora::config::EchoCanceller as EcConfig;
use sonora::{AudioProcessing, Config, StreamConfig};

pub const FRAME: usize = 480; // 10 ms at 48 kHz

pub struct EchoCanceller {
    apm: AudioProcessing,
    mic: Vec<f32>,
    refbuf: Vec<f32>,
    scratch: Vec<f32>,
}

impl EchoCanceller {
    pub fn new() -> Self {
        let sc = StreamConfig::new(48_000, 1);
        let apm = AudioProcessing::builder()
            .config(Config { echo_canceller: Some(EcConfig::default()), ..Default::default() })
            .capture_config(sc)
            .render_config(sc)
            .build();
        Self { apm, mic: Vec::new(), refbuf: Vec::new(), scratch: vec![0.0; FRAME] }
    }

    /// mic and ref in step (any block size, 10 ms frames inside); a missing or short reference counts as silence.
    /// Returns the echo-free mic for every whole 10 ms frame so far.
    pub fn process(&mut self, mic: &[f32], reference: &[f32]) -> Vec<f32> {
        self.mic.extend_from_slice(mic);
        self.refbuf.extend_from_slice(reference);
        let frames = self.mic.len() / FRAME;
        let mut out = vec![0f32; frames * FRAME];
        for i in 0..frames {
            let (a, b) = (i * FRAME, (i + 1) * FRAME);
            let mut r = [0f32; FRAME];
            if self.refbuf.len() > a {
                let have = (self.refbuf.len() - a).min(FRAME);
                r[..have].copy_from_slice(&self.refbuf[a..a + have]);
            }
            for v in r.iter_mut() {
                *v = v.clamp(-1.0, 1.0);
            }
            let _ = self.apm.process_render_f32(&[&r], &mut [&mut self.scratch[..]]);
            let m: Vec<f32> = self.mic[a..b].iter().map(|v| v.clamp(-1.0, 1.0)).collect();
            if self.apm.process_capture_f32(&[&m], &mut [&mut out[a..b]]).is_err() {
                out[a..b].copy_from_slice(&m); // never silence the voice because the canceller hiccuped
            }
        }
        let used = frames * FRAME;
        self.mic.drain(..used);
        if self.refbuf.len() > used {
            self.refbuf.drain(..used);
        } else {
            self.refbuf.clear();
        }
        out
    }
}
