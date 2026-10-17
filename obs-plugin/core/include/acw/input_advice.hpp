// SPDX-License-Identifier: GPL-2.0-or-later
// A quick listen to the microphone: after about ten seconds it can say how noisy the room is, whether the speakers
// are heard in the microphone and whether the captions need the more accurate level, so the filter's status line
// can suggest the settings that fit ("Your room is quiet: noise removal can be turned off"). No OBS in here.
//
// The measurements and the thresholds are the website's own (web/src/media/analysis.js, web/src/recommend.js; the
// website's calibration: docs/research/recommend-2026-10.md). The noise figure is the one the service uses to choose
// its noise remover: the 90th minus the 10th percentile of the levels of 10 ms frames, digital silence left out.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace acw {

struct InputFindings {
	bool ready = false;          // enough microphone audio was heard to say anything
	bool final = false;          // all that will be kept has been kept (nothing more is listened to)
	double seconds = 0;          // of audio measured (muted stretches do not count)
	bool has_speech = false;     // someone spoke (the level moved up and down)
	double noise_db = 99;        // dB between the voice and the background
	double floor_dbfs = 0;       // the background level (10th percentile of the frame levels)
	double level_dbfs = 0;       // the voice level (90th percentile)
	double clip_fraction = 0;    // share of the samples that are flattened against full scale
	bool echo_known = false;     // the speakers' sound was available to compare with
	double echo_share = 0;       // share of the microphone's energy that the speakers' sound explains (0..1)
};

// What the filter is set to, for advice that fits it.
struct AdviceContext {
	bool denoise = true;
	int atten_lim_db = 100;       // 12 / 24 / 40 / 100
	std::string captions;         // "", "local" or "premium"
	std::string language;         // "" = detect, "en", "en,es", ...
	bool echo_requested = false;  // an echo reference is set up
	bool echo_active = false;     // ... and the service accepted it
};

// One suggestion: `id` for translations (the plugin's locale file), `text` the built-in English, with %1 for a detail.
struct AdviceNote {
	std::string id;
	std::string text;
	std::string arg;
};

struct InputAdvice {
	bool ready = false;
	bool quiet_room = false;       // noise removal can go
	int atten_lim_db = 100;        // the strength that fits a noisy room (12 / 24 / 40 / 100)
	bool echo_heard = false;
	std::string captions;          // "local" or "premium": the level that fits
	std::vector<AdviceNote> notes; // what to tell the user, most useful first; empty = the settings fit
};

class InputAnalyser {
public:
	// `with_reference`: the speakers' sound will be pushed too (it is compared with the microphone).
	// `seconds`: how much audio makes the findings ready; `max_seconds`: after this much nothing more is kept.
	explicit InputAnalyser(bool with_reference, double seconds = 10.0, double max_seconds = 20.0);

	// Audio thread: wait-free, no allocation. `mic` n samples at 48 kHz; `ref` the same time of what the speakers played,
	// or null (silence). Call it with the microphone audio that is heard, not with muted stretches.
	void push(const float *mic, const float *ref, size_t n);

	bool ready() const;

	// Any thread. The comparison with the speakers takes a few milliseconds; it is done at most once per five seconds
	// of new audio and remembered.
	InputFindings findings() const;

	static InputAdvice advise(const InputFindings &f, const AdviceContext &ctx);

private:
	static constexpr int kDecimate = 24;  // 48 kHz -> 2 kHz for the speaker comparison
	static constexpr size_t kLagMax = 800;  // 400 ms at 2 kHz

	const bool with_ref_;
	const size_t need_levels_;
	std::vector<float> levels_;           // dB per 10 ms frame, filled once
	std::vector<float> band_levels_;      // the same for the speech band (150 Hz - 3.5 kHz) only
	double band_sq_ = 0;
	double hp_[4] = {0, 0, 0, 0}, lp_[4] = {0, 0, 0, 0};  // filter memory: x1 x2 y1 y2
	std::atomic<uint32_t> n_levels_{0};
	double sum_sq_ = 0;
	uint32_t in_frame_ = 0;
	std::atomic<uint64_t> samples_{0};
	std::atomic<uint64_t> clipped_{0};
	uint32_t run_ = 0;
	std::vector<float> mic2_, ref2_;
	std::atomic<uint32_t> n_2k_{0};
	double acc_m_ = 0, acc_r_ = 0;
	uint32_t acc_n_ = 0;

	mutable std::mutex mu_;
	mutable uint32_t echo_at_ = 0;        // n_2k_ when the comparison was last made
	mutable double echo_share_ = 0;
};

}  // namespace acw
