// SPDX-License-Identifier: GPL-2.0-or-later
// "Only while streaming or recording": decides when the filter's sessions should run.
//
// A session is billed for every second it is connected, so with this setting on the filter keeps no session while
// nothing is going out. The gate gets the state of OBS's outputs (stream, recording, replay buffer, virtual camera) and
// the time, and says run or stop:
//   - run as soon as any output is starting or running (so processing is ready when the output begins);
//   - stop only when every output has been off for a grace period (10 s by default), so that a restart of the stream, or a
//     recording that stops and starts again, does not tear the session down and build it up again.
// No OBS here: the plugin feeds it from the frontend events, the tests feed it directly.
#pragma once

#include <chrono>
#include <mutex>

namespace acw {

enum class LiveOutput { Stream = 0, Recording = 1, Replay = 2, VirtualCam = 3 };

class LiveGate {
public:
	using Clock = std::chrono::steady_clock;
	static constexpr int kOutputs = 4;

	explicit LiveGate(std::chrono::milliseconds grace = std::chrono::seconds(10));

	// An output is starting or running (`active`), or it has stopped. Call it for every event; repeats are fine.
	void set_active(LiveOutput output, bool active, Clock::time_point now);

	// True while the sessions should run. Call as often as convenient; the stop is decided here, once the grace is over.
	bool run(Clock::time_point now);

	// Some output is starting or running right now (the grace period does not count).
	bool any_active() const;
	bool active(LiveOutput output) const;

	std::chrono::milliseconds grace() const { return grace_; }

private:
	mutable std::mutex mu_;
	std::chrono::milliseconds grace_;
	bool active_[kOutputs] = {false, false, false, false};
	bool running_ = false;
	Clock::time_point stop_at_{};  // when `running_` ends if nothing starts meanwhile (valid while no output is active)
};

}  // namespace acw
