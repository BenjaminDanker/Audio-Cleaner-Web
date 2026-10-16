// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/live_gate.hpp"

namespace acw {

LiveGate::LiveGate(std::chrono::milliseconds grace) : grace_(grace) {}

void LiveGate::set_active(LiveOutput output, bool active, Clock::time_point now)
{
	std::lock_guard<std::mutex> lk(mu_);
	const int i = static_cast<int>(output);
	if (i < 0 || i >= kOutputs)
		return;
	const bool was = active_[i];
	active_[i] = active;
	bool any = false;
	for (bool a : active_)
		any = any || a;
	if (any) {
		running_ = true;  // at once, not after the next poll
	} else if (running_ && was && !active) {
		stop_at_ = now + grace_;  // the last output just stopped: the grace period starts now
	}
}

bool LiveGate::run(Clock::time_point now)
{
	std::lock_guard<std::mutex> lk(mu_);
	for (bool a : active_)
		if (a)
			return true;
	if (running_ && now >= stop_at_)
		running_ = false;
	return running_;
}

bool LiveGate::any_active() const
{
	std::lock_guard<std::mutex> lk(mu_);
	for (bool a : active_)
		if (a)
			return true;
	return false;
}

bool LiveGate::active(LiveOutput output) const
{
	std::lock_guard<std::mutex> lk(mu_);
	const int i = static_cast<int>(output);
	return i >= 0 && i < kOutputs && active_[i];
}

}  // namespace acw
