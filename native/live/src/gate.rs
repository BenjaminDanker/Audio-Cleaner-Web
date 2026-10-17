//! The captions' bookkeeping that needs no model, so it can be tested on its own: which audio the decoder gets
//! (the silence gate), and what happens when the decoder falls behind (shedding).
//!
//! Silence isn't decoded: from GATE_AFTER_S after the last speech (long enough for the line to end) until someone
//! talks again, audio is only counted (Piece::Skip), so the captions' times keep following the talk; the last
//! PREROLL_FRAMES of the quiet are kept and decoded with the next words. Measured on LibriSpeech with pauses: word
//! error rate 4.4% gated vs 4.7-5.2% not, 18% less decoding.

use crate::dsp::SAMPLE_RATE;

pub const SPEECH_DBFS: f32 = -50.0; // quieter than this is silence (of the denoised audio when noise removal runs)
pub const GATE_AFTER_S: f32 = 2.0;
pub const PREROLL_FRAMES: usize = 15;

// A stream more than MAX_BACKLOG behind its decoder keeps only the newest KEEP_BACKLOG; the incident is over once
// less than RECOVERED_BACKLOG waits.
pub const MAX_BACKLOG: usize = 2 * SAMPLE_RATE;
pub const KEEP_BACKLOG: usize = SAMPLE_RATE / 2;
pub const RECOVERED_BACKLOG: usize = SAMPLE_RATE / 2;

/// Audio waiting for the decoder: samples to decode, or samples only counted.
#[derive(Debug, PartialEq)]
pub enum Piece {
    Audio(Vec<f32>),
    Skip(usize),
}

pub fn dbfs(x: &[f32]) -> f32 {
    let ms = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len().max(1) as f64;
    (20.0 * (ms.sqrt() + 1e-9).log10()) as f32
}

#[derive(Default)]
pub struct Gate {
    quiet: f32,
    gated: bool,
    preroll: Vec<Vec<f32>>,
    skipped: usize, // since the gate closed, not counting the preroll
}

pub struct Fed {
    pub pieces: Vec<Piece>,
    /// Audio that was skipped and is now known to have been silence (billed audio a captions-only session gets back).
    pub idle: usize,
}

impl Gate {
    /// One frame of the session's audio; `level`: how loud the speech in it is.
    pub fn feed(&mut self, x: &[f32], level: f32) -> Fed {
        if level >= SPEECH_DBFS {
            self.quiet = 0.0;
            let mut pieces = Vec::new();
            let mut idle = 0;
            if self.gated {
                self.gated = false;
                pieces.push(Piece::Skip(self.skipped));
                pieces.extend(self.preroll.drain(..).map(Piece::Audio));
                idle = std::mem::take(&mut self.skipped);
            }
            pieces.push(Piece::Audio(x.to_vec()));
            return Fed { pieces, idle };
        }
        self.quiet += x.len() as f32 / SAMPLE_RATE as f32;
        if self.quiet <= GATE_AFTER_S {
            return Fed { pieces: vec![Piece::Audio(x.to_vec())], idle: 0 };
        }
        self.gated = true;
        self.preroll.push(x.to_vec());
        if self.preroll.len() > PREROLL_FRAMES {
            self.skipped += self.preroll.remove(0).len();
        }
        Fed { pieces: Vec::new(), idle: 0 }
    }

    /// `n` samples the client sent as silence (never billed): the line still gets its trailing silence, the rest
    /// only moves the clock.
    pub fn silence(&mut self, n: usize) -> Vec<Piece> {
        let need = if self.gated { 0 } else { ((GATE_AFTER_S - self.quiet) * SAMPLE_RATE as f32).max(0.0) as usize };
        let fed = n.min(need);
        self.quiet += n as f32 / SAMPLE_RATE as f32;
        let mut pieces = Vec::new();
        if fed > 0 {
            pieces.push(Piece::Audio(vec![0.0; fed]));
        }
        if n > fed {
            self.gated = true;
            pieces.push(Piece::Skip(n - fed));
        }
        pieces
    }

    /// End of the session: the billed audio that is still waiting behind a closed gate.
    pub fn finish(&mut self) -> usize {
        if !self.gated {
            return 0;
        }
        let n = self.skipped + self.preroll.iter().map(|p| p.len()).sum::<usize>();
        self.preroll.clear();
        self.skipped = 0;
        n
    }
}

/// When more than MAX_BACKLOG of audio waits, keep the newest KEEP_BACKLOG (the oldest dropped audio becomes a
/// Skip, so the times stay right). Returns the pieces to decode and whether audio was dropped.
pub fn shed(items: Vec<Piece>) -> (Vec<Piece>, bool) {
    let audio: usize = items.iter().map(|p| if let Piece::Audio(a) = p { a.len() } else { 0 }).sum();
    if audio <= MAX_BACKLOG {
        return (items, false);
    }
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
    let mut out = vec![Piece::Skip(audio - have + skips)];
    out.extend(kept.into_iter().map(Piece::Audio));
    (out, true)
}

pub fn audio_len(items: &[Piece]) -> usize {
    items.iter().map(|p| if let Piece::Audio(a) = p { a.len() } else { 0 }).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn talk() -> Vec<f32> {
        vec![0.1; 960]
    }

    fn quiet() -> Vec<f32> {
        vec![10f32.powf(-70.0 / 20.0); 960]
    }

    #[test]
    fn silence_is_skipped_after_the_line_ends_and_the_words_after_it_are_kept() {
        let mut g = Gate::default();
        let mut decoded = 0;
        let mut skipped = 0;
        for _ in 0..25 {
            decoded += audio_len(&g.feed(&talk(), 0.0).pieces);
        }
        for _ in 0..500 {
            let f = g.feed(&quiet(), dbfs(&quiet()));
            decoded += audio_len(&f.pieces);
        }
        assert_eq!(decoded, 25 * 960 + (GATE_AFTER_S * 50.0) as usize * 960);
        let back = g.feed(&talk(), 0.0);
        for p in &back.pieces {
            match p {
                Piece::Skip(n) => skipped += n,
                Piece::Audio(a) => decoded += a.len(),
            }
        }
        assert_eq!(back.idle, skipped); // the skipped audio is reported, once
        assert_eq!(skipped, (500 - 100 - PREROLL_FRAMES) * 960);
        assert_eq!(decoded, (25 + 100 + PREROLL_FRAMES + 1) * 960); // the preroll and the new words are decoded
    }

    #[test]
    fn a_denoised_level_decides_what_is_silence() {
        let mut g = Gate::default();
        let fan = vec![0.05f32; 960]; // loud by itself
        let decoded: usize = (0..300).map(|_| audio_len(&g.feed(&fan, -80.0).pieces)).sum();
        assert!(decoded < 300 * 960);
        assert!(g.finish() > 0);
    }

    #[test]
    fn silence_frames_end_the_line_then_only_count() {
        let mut g = Gate::default();
        g.feed(&talk(), 0.0);
        let p = g.silence(48_000 * 10);
        let tail = (GATE_AFTER_S * 48_000.0) as usize;
        assert_eq!(p, vec![Piece::Audio(vec![0.0; tail]), Piece::Skip(48_000 * 10 - tail)]);
        assert_eq!(g.finish(), 0); // silence frames were never billed: nothing to give back
    }

    #[test]
    fn a_stream_that_falls_behind_keeps_the_newest_half_second() {
        let items: Vec<Piece> = (0..300).map(|_| Piece::Audio(talk())).collect();
        let (out, dropped) = shed(items);
        assert!(dropped);
        assert_eq!(audio_len(&out), KEEP_BACKLOG);
        assert_eq!(out[0], Piece::Skip(300 * 960 - KEEP_BACKLOG));
        let few: Vec<Piece> = (0..10).map(|_| Piece::Audio(talk())).collect();
        assert!(!shed(few).1);
    }
}
