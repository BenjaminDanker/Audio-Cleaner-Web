// SPDX-License-Identifier: GPL-2.0-or-later
// Fixed-delay output stage.
//
// The stage output is always the input delayed by D samples. The delay is
// chosen once and never changes. Each 20 ms block of output is taken either
// from the processed frame the server returned for it, if that frame arrived
// before the block's deadline, or from the delayed original audio. Source
// switches are cross-faded over kCrossfadeSamples (5 ms).
//
// Alignment: the server returns, for input frame k, the processed samples
// P[k*960 .. k*960+960) where P[i] is the cleaned version of input sample
// i - L (L = model latency, 1440 samples = 30 ms for "light"). Output block k
// therefore starts at output index  k*960 + (D - L)  and the original audio
// for the same block is x[i - L] = x[o - D].
//
// Everything in render()/push_input() is allocation-free, lock-free and
// wait-free; it runs on the OBS audio thread.
#pragma once

#include "acw/common.hpp"
#include "acw/stats.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace acw {

// Processed frames, written by the network thread and read by the audio
// thread. One slot per frame, validated by a sequence tag (SPSC, lock-free).
class ProcessedStore {
public:
	static constexpr size_t kSlots = 256;  // ~5.1 s of audio

	ProcessedStore();

	// network thread -------------------------------------------------------
	void publish(uint64_t seq, const float *pcm /* kFrameSamples */);
	void mark_sent(uint64_t seq);

	// audio thread ---------------------------------------------------------
	const float *get(uint64_t seq) const;  // nullptr if the frame did not arrive
	bool was_sent(uint64_t seq) const;

private:
	std::vector<float> pcm_;
	std::unique_ptr<std::atomic<uint64_t>[]> arrived_;  // seq + 1
	std::unique_ptr<std::atomic<uint64_t>[]> sent_;     // seq + 1
};

class OutputStage {
public:
	// delay_samples: D at 48 kHz. model_latency: expected L (samples).
	OutputStage(int delay_samples, int model_latency, ProcessedStore &store, Counters &counters);

	int delay_samples() const { return static_cast<int>(delay_); }

	// Called from any thread once the server reported its latency. Applied by
	// the audio thread before it has played any processed audio.
	void set_model_latency(int samples) { pending_latency_.store(samples, std::memory_order_release); }

	// Append 48 kHz mono input to the delay line (audio thread).
	void push_input(const float *x, size_t n);

	// Produce n output samples (audio thread).
	void render(float *out, size_t n);

private:
	enum class Mode : uint8_t { Orig, Proc };

	void apply_pending_latency();
	void ensure_decided(int64_t upto_block);
	void decide(int64_t block);
	Mode mode_of(int64_t block) const { return block < 0 ? Mode::Orig : modes_[block & 3]; }

	int64_t delay_;
	int64_t latency_;
	int64_t off_;  // output index of the start of block 0 = D - L
	ProcessedStore &store_;
	Counters &counters_;
	std::atomic<int> pending_latency_{-1};

	std::vector<float> ring_;  // original audio, indexed by absolute sample position
	size_t mask_;
	int64_t in_pos_ = 0;   // samples pushed so far
	int64_t out_pos_ = 0;  // next output index

	int64_t decided_upto_ = -1;
	Mode modes_[4] = {Mode::Orig, Mode::Orig, Mode::Orig, Mode::Orig};
	const float *ptrs_[4] = {nullptr, nullptr, nullptr, nullptr};
	bool played_proc_ = false;
};

}  // namespace acw
