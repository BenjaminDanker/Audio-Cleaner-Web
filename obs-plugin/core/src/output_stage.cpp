// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/output_stage.hpp"

#include <algorithm>
#include <cstring>

namespace acw {

// ---------------------------------------------------------------------------
// ProcessedStore

ProcessedStore::ProcessedStore()
	: pcm_(kSlots * kFrameSamples, 0.0f), arrived_(new std::atomic<uint64_t>[kSlots]),
	  sent_(new std::atomic<uint64_t>[kSlots])
{
	for (size_t i = 0; i < kSlots; ++i) {
		arrived_[i].store(0, std::memory_order_relaxed);
		sent_[i].store(0, std::memory_order_relaxed);
	}
}

void ProcessedStore::publish(uint64_t seq, const float *pcm)
{
	const size_t slot = static_cast<size_t>(seq % kSlots);
	// Invalidate first so a reader never sees a half-written reused slot.
	arrived_[slot].store(0, std::memory_order_release);
	std::memcpy(&pcm_[slot * kFrameSamples], pcm, kFrameSamples * sizeof(float));
	arrived_[slot].store(seq + 1, std::memory_order_release);
}

void ProcessedStore::mark_sent(uint64_t seq)
{
	sent_[static_cast<size_t>(seq % kSlots)].store(seq + 1, std::memory_order_release);
}

const float *ProcessedStore::get(uint64_t seq) const
{
	const size_t slot = static_cast<size_t>(seq % kSlots);
	if (arrived_[slot].load(std::memory_order_acquire) != seq + 1)
		return nullptr;
	return &pcm_[slot * kFrameSamples];
}

bool ProcessedStore::was_sent(uint64_t seq) const
{
	return sent_[static_cast<size_t>(seq % kSlots)].load(std::memory_order_acquire) == seq + 1;
}

// ---------------------------------------------------------------------------
// OutputStage

namespace {
constexpr size_t kRingSize = 1u << 17;  // 131072 samples (2.7 s) of original audio
}

OutputStage::OutputStage(int delay_samples, int model_latency, ProcessedStore &store, Counters &counters)
	: delay_(delay_samples), latency_(model_latency), off_(delay_samples - model_latency), store_(store),
	  counters_(counters), ring_(kRingSize, 0.0f), mask_(kRingSize - 1)
{
}

void OutputStage::push_input(const float *x, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		ring_[static_cast<size_t>(in_pos_ + static_cast<int64_t>(i)) & mask_] = x[i];
	in_pos_ += static_cast<int64_t>(n);
}

void OutputStage::apply_pending_latency()
{
	const int pl = pending_latency_.load(std::memory_order_acquire);
	if (pl < 0 || pl == latency_)
		return;
	// Re-grid only while no processed audio has been played (start of session).
	if (played_proc_)
		return;
	pending_latency_.store(-1, std::memory_order_relaxed);
	latency_ = pl;
	off_ = delay_ - latency_;
	// Everything up to the block that is about to be decided counts as decided
	// (original audio); decisions resume with the next block on the new grid.
	const int64_t rel = out_pos_ - off_ + static_cast<int64_t>(kCrossfadeSamples);
	decided_upto_ = rel >= 0 ? rel / static_cast<int64_t>(kFrameSamples) : -1;
	for (int i = 0; i < 4; ++i) {
		modes_[i] = Mode::Orig;
		ptrs_[i] = nullptr;
	}
}

void OutputStage::decide(int64_t block)
{
	const size_t i = static_cast<size_t>(block & 3);
	const float *p = store_.get(static_cast<uint64_t>(block));
	if (p) {
		modes_[i] = Mode::Proc;
		ptrs_[i] = p;
		counters_.used.fetch_add(1, std::memory_order_relaxed);
		played_proc_ = true;
	} else {
		modes_[i] = Mode::Orig;
		ptrs_[i] = nullptr;
		if (store_.was_sent(static_cast<uint64_t>(block)))
			counters_.late.fetch_add(1, std::memory_order_relaxed);
		else
			counters_.passthrough.fetch_add(1, std::memory_order_relaxed);
	}
}

void OutputStage::ensure_decided(int64_t upto_block)
{
	while (decided_upto_ < upto_block) {
		++decided_upto_;
		decide(decided_upto_);
	}
}

void OutputStage::render(float *out, size_t n)
{
	apply_pending_latency();

	constexpr int64_t kBlock = static_cast<int64_t>(kFrameSamples);
	constexpr int64_t kXf = static_cast<int64_t>(kCrossfadeSamples);

	for (size_t i = 0; i < n; ++i, ++out_pos_) {
		const int64_t src = out_pos_ - delay_;
		const float orig = (src >= 0 && src < in_pos_) ? ring_[static_cast<size_t>(src) & mask_] : 0.0f;

		const int64_t rel = out_pos_ - off_;
		// Decisions are taken kXf before a block starts, so the tail of the
		// previous block can fade to the original if the reply is missing.
		if (rel + kXf >= 0)
			ensure_decided((rel + kXf) / kBlock);

		if (rel < 0) {
			out[i] = orig;
			continue;
		}
		const int64_t b = rel / kBlock;
		const int64_t pos = rel - b * kBlock;
		if (modes_[b & 3] == Mode::Orig) {
			out[i] = orig;
			continue;
		}
		const float proc = ptrs_[b & 3][pos];
		float y = proc;
		if (pos < kXf && mode_of(b - 1) == Mode::Orig) {
			// orig -> proc, fade in over the head of the block
			const float w = static_cast<float>(pos + 1) / static_cast<float>(kXf + 1);
			y = orig * (1.0f - w) + proc * w;
		} else if (pos >= kBlock - kXf && mode_of(b + 1) == Mode::Orig) {
			// proc -> orig, fade out over the tail of the block
			const float w = static_cast<float>(pos - (kBlock - kXf) + 1) / static_cast<float>(kXf + 1);
			y = proc * (1.0f - w) + orig * w;
		}
		out[i] = y;
	}
}

}  // namespace acw
