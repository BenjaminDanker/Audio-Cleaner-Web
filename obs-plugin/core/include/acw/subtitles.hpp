// SPDX-License-Identifier: GPL-2.0-or-later
// Subtitle files next to a recording: cues are collected while the recording runs (times counted from its start) and
// written as one SRT file per language when it stops.
#pragma once

#include <chrono>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace acw {

struct SubtitleCue {
	std::string lang;  // "" = the spoken language when it is not known (the file then has no language in its name)
	std::chrono::steady_clock::time_point start, end;
	std::string text;
};

// "00:01:02,345" for 62.345 seconds (negative = 0).
std::string srt_time(double seconds);

struct SrtCue {
	double start = 0, end = 0;  // seconds from the start of the recording
	std::string text;
};

// The text of an SRT file: cues in time order, no overlaps, long text wrapped into lines of at most `line_chars`.
std::string make_srt(std::vector<SrtCue> cues, size_t line_chars = 42);

// "/videos/2026-10-16 12-00-00.mkv" + "es" -> "/videos/2026-10-16 12-00-00.es.srt" (lang "" -> "... .srt").
std::string subtitle_path(const std::string &recording_path, const std::string &lang);

// Writes the file (to a temporary name first, then renames it). False on failure; `error` gets a short reason.
bool write_text_file(const std::string &path, const std::string &content, std::string *error = nullptr);

class SubtitleRecorder {
public:
	struct File {
		std::string lang;
		std::string srt;
		size_t cues = 0;
	};

	// A recording started now (or at `start`): forgets earlier cues and collects new ones.
	void begin(std::chrono::steady_clock::time_point start);
	bool active() const;
	// The recording was paused / went on (its own clock stands still meanwhile): speech in between is left out and the
	// cues after it are counted without the pause. No-ops while no recording runs.
	void pause(std::chrono::steady_clock::time_point now);
	void resume(std::chrono::steady_clock::time_point now);
	// Any thread; ignored while no recording runs. Cues that end before the recording started are dropped.
	void add(const SubtitleCue &cue);
	// The recording stopped: the files to write (one per language that has cues), and collecting ends.
	std::vector<File> finish();
	// Ends collecting without producing anything.
	void cancel();

private:
	static constexpr size_t kMaxCues = 20000;  // all languages together: a many-hour recording at full speed
	mutable std::mutex mu_;
	bool active_ = false;
	std::chrono::steady_clock::time_point t0_{};
	bool paused_ = false;
	std::chrono::steady_clock::time_point pause_start_{};
	std::vector<std::pair<std::chrono::steady_clock::time_point, std::chrono::steady_clock::time_point>> pauses_;  // finished ones
	double position(std::chrono::steady_clock::time_point t) const;  // seconds into the recording, pauses not counted
	std::map<std::string, std::vector<SrtCue>> cues_;
	size_t total_ = 0;
};

}  // namespace acw
