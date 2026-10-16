// SPDX-License-Identifier: GPL-2.0-or-later
// Holds caption lines back for a while, then hands them on in order: OBS's stream delay sends the video N seconds
// late, and a caption put on the stream "now" would ride on video from N seconds ago, ahead of the speech.
#pragma once

#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace acw {

class DelayedSender {
public:
	// send(line, display_seconds) runs on the sender's own thread, at or after the time given to `later`.
	using SendFn = std::function<void(const std::string &line, double display_seconds)>;

	// The same, for lines of several outputs: the tag says which one a line is for, and the outputs keep their own order
	// (a line for an output with a short delay may leave before an earlier line of an output with a long one).
	using TaggedSendFn = std::function<void(const std::string &line, double display_seconds, uint64_t tag)>;

	explicit DelayedSender(SendFn send, size_t max_lines = 600);
	explicit DelayedSender(TaggedSendFn send, size_t max_lines = 600);
	~DelayedSender();
	DelayedSender(const DelayedSender &) = delete;
	DelayedSender &operator=(const DelayedSender &) = delete;

	// Any thread. Lines go out in the order they were given (a later line never overtakes an earlier one).
	// Beyond `max_lines` waiting (a very long delay with constant talking), the oldest waiting line is dropped.
	void later(const std::string &line, double display_seconds, std::chrono::steady_clock::time_point due, uint64_t tag = 0);

	// Drops whatever is still waiting (the stream stopped).
	void clear();

	// Stops the thread; waits at most `timeout_ms` for it, then detaches. Waiting lines are dropped.
	void stop(int timeout_ms = 300);

	size_t waiting() const;

private:
	struct State;
	std::shared_ptr<State> st_;
};

}  // namespace acw
