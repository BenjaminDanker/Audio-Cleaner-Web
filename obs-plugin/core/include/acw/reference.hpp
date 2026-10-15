// SPDX-License-Identifier: GPL-2.0-or-later
// The echo reference: what the speakers play, handed from the thread that
// captures it to the audio thread of the microphone filter.
#pragma once

#include "acw/common.hpp"
#include "acw/resampler.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace acw {

// One writer thread (push) and one reader thread (pull); neither ever waits.
// The reader takes whole blocks (its 20 ms frames) from a continuous stream, so
// the reference is not torn or full of holes, and keeps a small cushion so that
// jitter in when the two arrive rarely interrupts it. The echo canceller needs
// the reference to be no later than the echo in the microphone; the cushion
// (10 ms) and jitter are small next to the tens of ms the sound takes to get
// from the speakers to the microphone.
class ReferenceBuffer {
public:
	// host_rate: sample rate of the audio given to push() (mono).
	explicit ReferenceBuffer(int host_rate);

	// Writer: mono samples at host_rate.
	void push(const float *mono, size_t n);

	// Reader: the next n samples at 48 kHz, zeros where nothing arrived in time.
	// Returns false when the whole block is silence.
	bool pull(float *out, size_t n);

private:
	static constexpr size_t kRing = 1 << 14;  // 340 ms at 48 kHz
	static constexpr size_t kMaxPush = 2048;  // host-rate samples per inner step of push()
	static constexpr size_t kCushion = 480;   // 10 ms

	std::unique_ptr<RationalResampler> up_;
	std::vector<float> scratch_;  // writer: resampler output
	std::vector<float> ring_;
	std::atomic<uint64_t> written_{0};  // samples pushed so far (48 kHz)
	uint64_t read_ = 0;                 // reader: next sample to hand out
	bool primed_ = false;
};

}  // namespace acw
