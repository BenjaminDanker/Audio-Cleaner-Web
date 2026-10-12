// SPDX-License-Identifier: GPL-2.0-or-later
// Exponential backoff with jitter for reconnect attempts.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>

namespace acw {

class Backoff {
public:
	Backoff(std::chrono::milliseconds base = std::chrono::milliseconds(1000),
		std::chrono::milliseconds cap = std::chrono::milliseconds(60000),
		uint64_t seed = std::random_device{}())
		: base_(base), cap_(cap), rng_(seed)
	{
	}

	// Next delay: min(cap, base * 2^attempt), scaled by a random factor in [0.5, 1.0]
	// ("equal jitter") so that many clients do not reconnect in lock step.
	std::chrono::milliseconds next()
	{
		const int shift = std::min(attempt_, 20);
		int64_t d = base_.count() << shift;
		d = std::min<int64_t>(d, cap_.count());
		std::uniform_real_distribution<double> u(0.5, 1.0);
		++attempt_;
		return std::chrono::milliseconds(static_cast<int64_t>(static_cast<double>(d) * u(rng_)));
	}

	void reset() { attempt_ = 0; }
	int attempts() const { return attempt_; }

private:
	std::chrono::milliseconds base_, cap_;
	int attempt_ = 0;
	std::mt19937_64 rng_;
};

}  // namespace acw
