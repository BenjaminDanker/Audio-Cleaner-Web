// SPDX-License-Identifier: GPL-2.0-or-later
// Streaming rational polyphase resampler (Kaiser-windowed sinc, linear phase).
//
// OBS may run at 44.1 kHz while the wire protocol is 48 kHz only; the filter
// resamples 44.1 -> 48 kHz for the network path and 48 -> 44.1 kHz on the way
// back. The two stages are exact rational converters (160/147 and 147/160),
// so there is no long-term drift between the input and output sample counts.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace acw {

class RationalResampler {
public:
	// taps_per_phase: FIR length per polyphase branch (quality / CPU).
	RationalResampler(int in_rate, int out_rate, int taps_per_phase = 48, double kaiser_beta = 8.0);

	int in_rate() const { return in_rate_; }
	int out_rate() const { return out_rate_; }

	// Group delay of the filter, expressed in seconds.
	double group_delay_seconds() const { return delay_sec_; }

	// Upper bound on the number of outputs `process()` writes for `n_in` inputs.
	size_t max_output(size_t n_in) const { return n_in * up_ / down_ + 2; }

	// Number of input samples that must be supplied so that `process()`
	// yields exactly `n_out` outputs from the current state. Exact for
	// down-converters (out_rate <= in_rate); an upper-bound-free "at least"
	// for up-converters, which is why the filter only uses it for 48k -> R.
	size_t input_needed(size_t n_out) const;

	// Consumes all `n_in` samples, writes every output that can be computed
	// and returns how many were written. `out` must hold max_output(n_in).
	size_t process(const float *in, size_t n_in, float *out);

	void reset();

private:
	int in_rate_, out_rate_;
	int64_t up_, down_;
	int taps_;
	double delay_sec_ = 0.0;
	std::vector<float> h_;      // polyphase coefficients, [phase][tap]
	std::vector<float> hist_;   // last taps_-1 input samples
	std::vector<float> work_;   // history + current block (preallocated, no audio-thread allocation)
	int64_t consumed_ = 0;      // input samples consumed so far
	int64_t next_p_ = 0;        // position (in 1/up input-sample units) of the next output
};

}  // namespace acw
