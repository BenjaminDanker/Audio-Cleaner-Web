// SPDX-License-Identifier: GPL-2.0-or-later
// "Test my room": the person talks the way they do on stream for twenty seconds; the live settings that fit are offered
// as changes to the current ones. No OBS in here.
//
// The thresholds are the website's own (web/src/recommend.js liveSettingsFor; calibration: docs/research/recommend-2026-10.md;
// web/tests/reco-unit.mjs checks that the numbers agree). The noise figure is the one the service uses to choose its noise
// remover: dB between the voice and what the noise remover at full strength takes out. The plugin sends Opus, so it can't
// line its own audio up with the cleaned audio: the service measures it at its noise remover (Session::request_room_test).
// The microphone itself is measured here: whether someone spoke (the speech band's level
// moves up and down) and clipping.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "acw/resampler.hpp"

namespace acw {

struct InputFindings {
	bool ready = false;          // enough microphone audio was heard to say anything
	double seconds = 0;          // of audio measured
	bool has_speech = false;     // someone spoke (the speech band's level moved up and down)
	double clip_fraction = 0;    // share of the samples that are flattened against full scale
};

// Measures the microphone (48 kHz): the speech band's level in 10 ms frames (digital silence left out), and clipping.
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
	std::vector<float> levels_;       // dB per 10 ms frame, filled once (to leave digital silence out)
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
	bool done = false;                 // the whole stretch was heard, and the noise figure (or why there is none) is in
	bool heard_speech = false;
	std::string error_code;            // the service could not measure the noise ("needs_noise_removal", ...): no suggestion
	std::string error_message;         // ... its own words for it
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

	// The service's answer (SessionSnapshot room_noise_db, or room_code and room_message). Any thread but the audio one.
	void set_noise(double noise_db);
	void set_failed(const std::string &code, const std::string &message);

	double seconds_left() const;  // of the microphone's stretch
	bool heard() const;           // the microphone's stretch is in
	bool done() const;            // ... and the noise figure, or why there is none
	RoomSuggestion suggest(const AdviceContext &now) const;

private:
	double seconds_;
	InputAnalyser a_;
	std::atomic<double> noise_db_;  // NaN until the service says
	std::atomic<bool> failed_{false};
	mutable std::mutex mu_;         // the failure's words
	std::string code_, message_;
	std::unique_ptr<RationalResampler> rs_;  // to 48 kHz, when OBS runs at another rate
	std::vector<float> buf_;
};

}  // namespace acw
