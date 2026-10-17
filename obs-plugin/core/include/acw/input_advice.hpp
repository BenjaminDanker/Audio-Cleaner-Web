// SPDX-License-Identifier: GPL-2.0-or-later
// "Test my room": the person talks the way they do on stream for twenty seconds; the microphone is measured and the live
// settings that fit are offered as changes to the current ones. No OBS in here.
//
// The measurements and the thresholds are the website's own (web/src/media/analysis.js, web/src/recommend.js liveSettingsFor;
// calibration: docs/research/recommend-2026-10.md; web/tests/reco-unit.mjs checks that the numbers agree). The noise figure is the
// one the service uses to choose its noise remover: the 90th minus the 10th percentile of the levels of 10 ms frames, digital
// silence left out.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "acw/resampler.hpp"

namespace acw {

struct InputFindings {
	bool ready = false;          // enough microphone audio was heard to say anything
	double seconds = 0;          // of audio measured
	bool has_speech = false;     // someone spoke (the speech band's level moved up and down)
	double noise_db = 99;        // dB between the voice and the background
	double floor_dbfs = 0;       // the background level (10th percentile of the frame levels)
	double level_dbfs = 0;       // the voice level (90th percentile)
	double clip_fraction = 0;    // share of the samples that are flattened against full scale
};

// Measures the microphone (48 kHz): the levels of 10 ms frames, the speech band's, and clipping.
class InputAnalyser {
public:
	explicit InputAnalyser(double seconds);

	// Audio thread: wait-free, no allocation. Stops keeping anything once `seconds` are in.
	void push(const float *mic, size_t n);

	bool ready() const;
	double seconds_heard() const;  // cheap, any thread
	InputFindings findings() const;  // any thread

private:
	const size_t need_levels_;
	std::vector<float> levels_;       // dB per 10 ms frame, filled once
	std::vector<float> band_levels_;  // the same for the speech band (150 Hz - 3.5 kHz) only
	double band_sq_ = 0;
	double hp_[4] = {0, 0, 0, 0}, lp_[4] = {0, 0, 0, 0};  // filter memory: x1 x2 y1 y2
	std::atomic<uint32_t> n_levels_{0};
	double sum_sq_ = 0;
	uint32_t in_frame_ = 0;
	std::atomic<uint64_t> samples_{0};
	std::atomic<uint64_t> clipped_{0};
	uint32_t run_ = 0;
};

// What the filter is set to.
struct AdviceContext {
	bool denoise = true;
	int atten_lim_db = 100;  // 12 / 24 / 40 / 100
	std::string captions;    // "", "local" or "premium"
	std::string language;    // "" = detect, "en", "en,es", ...
};

// One thing to say: `id` for the plugin's locale file, `text` the built-in English with %1 for `arg`.
struct AdviceNote {
	std::string id;
	std::string text;
	std::string arg;
};

constexpr double kRoomTestSeconds = 20.0;

struct RoomSuggestion {
	bool done = false;                 // the whole stretch was heard
	bool heard_speech = false;
	AdviceContext settings;            // what to set: the current settings with the changes made
	std::vector<AdviceNote> changes;   // one per change: id "room.<what>", arg "<from> -> <to>"
	std::vector<AdviceNote> notes;     // what only the person can do (the microphone is too loud)
};

// Noise removal is never suggested off for live, however quiet the room sounded: twenty seconds don't show the keyboard,
// the game or the door of a whole stream, so a quiet room gets the gentlest strength instead.
class RoomTest {
public:
	// `rate`: the sample rate of the audio pushed; `max_block`: the most samples one push() gets.
	RoomTest(int rate, size_t max_block, double seconds = kRoomTestSeconds);

	// Audio thread: wait-free, no allocation.
	void push(const float *mic, size_t n);

	double seconds_left() const;
	bool done() const;
	RoomSuggestion suggest(const AdviceContext &now) const;

private:
	double seconds_;
	InputAnalyser a_;
	std::unique_ptr<RationalResampler> rs_;  // to 48 kHz, when OBS runs at another rate
	std::vector<float> buf_;
};

}  // namespace acw
