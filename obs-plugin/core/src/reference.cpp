// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/reference.hpp"

#include <algorithm>
#include <cmath>

namespace acw {

ReferenceBuffer::ReferenceBuffer(int host_rate) : ring_(kRing, 0.0f)
{
	if (host_rate != kNetRate) {
		up_ = std::make_unique<RationalResampler>(host_rate, kNetRate);
		scratch_.assign(up_->max_output(kMaxPush) + 64, 0.0f);
	}
}

void ReferenceBuffer::push(const float *mono, size_t n)
{
	while (n > 0) {
		const size_t m = std::min(n, kMaxPush);
		const float *x = mono;
		size_t k = m;
		if (up_) {
			k = up_->process(mono, m, scratch_.data());
			x = scratch_.data();
		}
		uint64_t w = written_.load(std::memory_order_relaxed);
		for (size_t i = 0; i < k; ++i)
			ring_[(w + i) % kRing] = x[i];
		written_.store(w + k, std::memory_order_release);
		mono += m;
		n -= m;
	}
}

bool ReferenceBuffer::pull(float *out, size_t n)
{
	const uint64_t w = written_.load(std::memory_order_acquire);
	const uint64_t cushion = kCushion;
	// Start (and restart after a long gap): the newest audio, keeping a cushion against jitter in
	// when the reference arrives.
	if (!primed_ || w - read_ > n + cushion + 2 * n) {
		if (w < n + cushion) {
			std::fill(out, out + n, 0.0f);
			return false;
		}
		read_ = w - n - cushion;
		primed_ = true;
	}
	// Late (a whole block has not arrived yet): silence now, and the block comes with the next
	// call. The stream stays continuous: it just runs a block behind until the next restart.
	if (w - read_ < n) {
		std::fill(out, out + n, 0.0f);
		return false;
	}
	float peak = 0.0f;
	for (size_t i = 0; i < n; ++i) {
		out[i] = ring_[(read_ + i) % kRing];
		peak = std::max(peak, std::fabs(out[i]));
	}
	bool any = peak >= kSilenceLevel;  // quieter than this everywhere: the block is silence
	// The writer may have lapped us while we copied: then the copy is not trustworthy.
	if (written_.load(std::memory_order_acquire) - read_ > kRing) {
		std::fill(out, out + n, 0.0f);
		any = false;
	}
	read_ += n;
	return any;
}

}  // namespace acw
