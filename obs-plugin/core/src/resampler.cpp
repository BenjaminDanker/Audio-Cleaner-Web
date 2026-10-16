// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/resampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace acw {

namespace {

constexpr double kPi = 3.14159265358979323846;

double bessel_i0(double x)
{
	double sum = 1.0, term = 1.0;
	const double q = x * x / 4.0;
	for (int k = 1; k < 60; ++k) {
		term *= q / (static_cast<double>(k) * k);
		sum += term;
		if (term < sum * 1e-17)
			break;
	}
	return sum;
}

// Dot product of n (a multiple of 8) floats with eight independent partial sums: no loop-carried dependency, so the
// compiler turns the inner loop into vector multiply-adds without needing fast-math.
inline float dot8(const float *__restrict h, const float *__restrict x, size_t n)
{
	float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	for (size_t k = 0; k < n; k += 8)
		for (size_t j = 0; j < 8; ++j)
			acc[j] += h[k + j] * x[k + j];
	return ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
}

}  // namespace

RationalResampler::RationalResampler(int in_rate, int out_rate, int taps_per_phase, double kaiser_beta)
	: in_rate_(in_rate), out_rate_(out_rate), taps_(std::max(8, taps_per_phase))
{
	const int64_t g = std::gcd<int64_t, int64_t>(in_rate, out_rate);
	up_ = out_rate / g;
	down_ = in_rate / g;
	step_i_ = down_ / up_;
	step_ph_ = down_ % up_;
	stride_ = (static_cast<size_t>(taps_) + 7) & ~static_cast<size_t>(7);

	// Prototype low-pass at the intermediate rate in_rate * up_.
	const int64_t len = static_cast<int64_t>(taps_) * up_;
	std::vector<double> proto(static_cast<size_t>(len));
	// Cut-off: just below the lower Nyquist, normalised to the intermediate rate.
	const double fc = 0.5 * std::min<double>(in_rate, out_rate) * 0.94 / (static_cast<double>(in_rate) * up_);
	const double mid = (static_cast<double>(len) - 1.0) / 2.0;
	const double i0b = bessel_i0(kaiser_beta);
	double sum = 0.0;
	for (int64_t n = 0; n < len; ++n) {
		const double t = static_cast<double>(n) - mid;
		const double x = 2.0 * kPi * fc * t;
		const double sinc = (std::fabs(t) < 1e-12) ? 2.0 * fc : std::sin(x) / (kPi * t);
		const double r = t / (mid > 0 ? mid : 1.0);
		const double w = bessel_i0(kaiser_beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0b;
		proto[static_cast<size_t>(n)] = sinc * w;
		sum += proto[static_cast<size_t>(n)];
	}
	// Unity DC gain per phase: the sum over all branches must equal up_.
	const double scale = static_cast<double>(up_) / sum;

	// Tap k of phase ph is proto[ph + k * up_] and multiplies the input k samples before the newest one. The row is stored
	// oldest sample first (k = taps_-1 ... 0) at the end of its stride_ floats.
	h_.assign(static_cast<size_t>(up_) * stride_, 0.0f);
	for (int64_t ph = 0; ph < up_; ++ph)
		for (int k = 0; k < taps_; ++k)
			h_[static_cast<size_t>(ph) * stride_ + (stride_ - 1 - static_cast<size_t>(k))] =
				static_cast<float>(proto[static_cast<size_t>(ph + k * up_)] * scale);

	// Linear phase: delay = (len - 1) / 2 samples of the intermediate rate.
	delay_sec_ = mid / (static_cast<double>(in_rate) * static_cast<double>(up_));
	work_.assign(stride_ - 1 + 4096, 0.0f);
	reset();
}

void RationalResampler::reset()
{
	hist_.assign(stride_ - 1, 0.0f);
	consumed_ = 0;
	next_i_ = 0;
	next_ph_ = 0;
}

size_t RationalResampler::input_needed(size_t n_out) const
{
	if (n_out == 0)
		return 0;
	// Output m (at position p) needs input index floor(p / up_) to be consumed.
	const int64_t p_last = next_i_ * up_ + next_ph_ + static_cast<int64_t>(n_out - 1) * down_;
	const int64_t need_total = p_last / up_ + 1;
	const int64_t need = need_total - consumed_;
	return need > 0 ? static_cast<size_t>(need) : 0;
}

size_t RationalResampler::process(const float *in, size_t n_in, float *out)
{
	constexpr size_t kBlock = 4096;
	const size_t hist = stride_ - 1;
	if (work_.size() < hist + kBlock)
		work_.resize(hist + kBlock);

	size_t n_out = 0;
	while (n_in > 0) {
		const size_t n = std::min(n_in, kBlock);
		std::copy(hist_.begin(), hist_.end(), work_.begin());
		std::copy(in, in + n, work_.begin() + static_cast<std::ptrdiff_t>(hist));

		const int64_t end = consumed_ + static_cast<int64_t>(n);
		int64_t i0 = next_i_;    // newest input index used by the next output
		int64_t ph = next_ph_;   // its phase
		const float *w = work_.data();
		while (i0 < end) {
			// The stride_ inputs i0-stride_+1 .. i0 start at work_[i0 - consumed_] (work_ begins with stride_-1 of history).
			out[n_out++] = dot8(&h_[static_cast<size_t>(ph) * stride_], w + (i0 - consumed_), stride_);
			i0 += step_i_;
			ph += step_ph_;
			if (ph >= up_) {
				ph -= up_;
				++i0;
			}
		}
		next_i_ = i0;
		next_ph_ = ph;
		std::copy(work_.begin() + static_cast<std::ptrdiff_t>(n), work_.begin() + static_cast<std::ptrdiff_t>(n + hist),
			  hist_.begin());
		consumed_ = end;
		in += n;
		n_in -= n;
	}
	return n_out;
}

}  // namespace acw
