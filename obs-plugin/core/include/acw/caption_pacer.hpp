// SPDX-License-Identifier: GPL-2.0-or-later
// Turns final caption texts into short, paced lines for CEA-608/708 caption
// output. No backlog: stale lines are dropped.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace acw {

// Splits text into lines of at most `max_chars` Unicode code points, breaking
// at spaces where possible (UTF-8 aware; collapses whitespace).
std::vector<std::string> split_caption_lines(const std::string &text, size_t max_chars = 32);

class CaptionPacer {
public:
	// send(line, display_seconds) runs on the pacer thread.
	using SendFn = std::function<void(const std::string &line, double display_seconds)>;

	struct Options {
		size_t max_chars = 32;                // CEA-608 line width
		double chars_per_second = 24.0;       // stay below the ~60 byte/s 608 channel
		double min_gap_seconds = 0.25;
		double stale_seconds = 5.0;           // drop lines that waited longer than this
		size_t max_queue_lines = 6;
	};

	explicit CaptionPacer(SendFn send);
	CaptionPacer(SendFn send, const Options &opt);
	~CaptionPacer();
	CaptionPacer(const CaptionPacer &) = delete;
	CaptionPacer &operator=(const CaptionPacer &) = delete;

	// Any thread. Ignored while disabled.
	void push(const std::string &text);
	void set_enabled(bool on);

	// Stops the thread; waits at most `timeout_ms` for it, then detaches.
	void stop(int timeout_ms = 300);

	uint64_t sent_lines() const;
	uint64_t dropped_lines() const;

private:
	struct Item {
		std::string line;
		std::chrono::steady_clock::time_point queued;
	};
	struct State;
	std::shared_ptr<State> st_;
};

}  // namespace acw
