// SPDX-License-Identifier: GPL-2.0-or-later
// The text of an on-screen caption: the last one or two lines, rolled up at reading pace and cleared after a while of
// silence. A caption that is still being said (set_live) shows at once and grows word by word; its final version then
// takes its place without waiting for the reading pace (it was read as it came). OverlayText is the pure logic (times
// are plain seconds, so it is testable without waiting); OverlayDisplay runs it on its own thread and hands the text to
// a callback (the plugin writes it into an OBS text source).
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace acw {

class OverlayText {
public:
	struct Options {
		size_t max_lines = 2;          // lines on screen at once
		size_t line_chars = 42;        // characters per line
		double clear_after_s = 6.0;    // empty the screen this long after the last line was shown
		double chars_per_second = 25.; // reading pace: the next line of a long caption comes this much later
		double min_line_s = 0.7, max_line_s = 2.5;
		size_t max_pending = 10;       // more waiting lines than this: the oldest are dropped (no growing lag)
	};

	OverlayText() = default;
	explicit OverlayText(const Options &o) : opt_(o) {}

	void set_options(const Options &o) { opt_ = o; }

	// A new caption, `now` seconds on any steady scale (the same scale for every call). When `voice` had a caption
	// being said (set_live), this is its final version and replaces it.
	void push(const std::string &text, double now, uint64_t voice = 0);
	// What `voice` has said so far of the caption it is saying (replaces what it had; "" = nothing yet): shown below the
	// finished lines straight away. Several voices can be talking at once: each has its own lines.
	void set_live(uint64_t voice, const std::string &text, double now);
	// Moves on to `now`; true when the text changed.
	bool update(double now);
	// What is on screen: the lines joined with "\n" ("" = nothing).
	const std::string &text() const { return text_; }
	// Seconds from `now` until update() has something to change (the next line or the clearing); < 0 = nothing to wait for.
	double seconds_until_change(double now) const;
	// Empties the screen and forgets what waits.
	void clear();

private:
	struct Line {
		std::string text;
		double arrived;
	};
	struct Shown {
		std::string text;
		double ready_at;  // from then on a newer line may push it off the screen
	};
	double hold_for(const std::string &line) const;
	void rebuild();
	bool live_text() const;

	Options opt_;
	std::map<uint64_t, std::string> live_;  // per voice: the caption being said (the key stays until its final comes)
	std::deque<Line> pending_;
	std::deque<Shown> shown_;
	std::string text_;
	double last_reveal_ = -1e18;  // lines appear in order: never before the one that came up last
	double clear_at_ = 1e18;      // when the screen is emptied
};

// Several OverlayTexts (one per OBS text source), driven by one thread.
class OverlayDisplay {
public:
	// Called on the display's thread with the source name and its new text ("" = cleared).
	using SinkFn = std::function<void(const std::string &target, const std::string &text)>;

	explicit OverlayDisplay(SinkFn sink);
	~OverlayDisplay();
	OverlayDisplay(const OverlayDisplay &) = delete;
	OverlayDisplay &operator=(const OverlayDisplay &) = delete;

	// The targets that exist now, with the options of their text; targets no longer listed are cleared and dropped.
	void set_targets(const std::map<std::string, OverlayText::Options> &targets);
	// Any thread. Unknown targets are ignored.
	void push(const std::string &target, const std::string &text, uint64_t voice = 0);
	void set_live(const std::string &target, uint64_t voice, const std::string &text);
	// Empties every target (the text sources get "" at once).
	void clear_all();
	// Stops the thread (waits at most `timeout_ms`, then lets it go). Clears the targets first.
	void stop(int timeout_ms = 300);

private:
	struct State;
	std::shared_ptr<State> st_;
};

}  // namespace acw
