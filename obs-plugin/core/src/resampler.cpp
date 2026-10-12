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

}  // namespace

RationalResampler::RationalResampler(int in_rate, int out_rate, int taps_per_phase, double kaiser_beta)
	: in_rate_(in_rate), out_rate_(out_rate), taps_(std::max(8, taps_per_phase))
{
	const int64_t g = std::gcd<int64_t, int64_t>(in_rate, out_rate);
	up_ = out_rate / g;
	down_ = in_rate / g;

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

	// h_[phase * taps_ + k] = proto[phase + k * up_]
	h_.assign(static_cast<size_t>(up_) * static_cast<size_t>(taps_), 0.0f);
	for (int64_t ph = 0; ph < up_; ++ph)
		for (int k = 0; k < taps_; ++k)
			h_[static_cast<size_t>(ph) * static_cast<size_t>(taps_) + static_cast<size_t>(k)] =
				static_cast<float>(proto[static_cast<size_t>(ph + k * up_)] * scale);

	// Linear phase: delay = (len - 1) / 2 samples of the intermediate rate.
	delay_sec_ = mid / (static_cast<double>(in_rate) * static_cast<double>(up_));
	work_.assign(static_cast<size_t>(taps_ - 1) + 4096, 0.0f);
	reset();
}

void RationalResampler::reset()
{
	hist_.assign(static_cast<size_t>(taps_ - 1), 0.0f);
	consumed_ = 0;
	next_p_ = 0;
}

size_t RationalResampler::input_needed(size_t n_out) const
{
	if (n_out == 0)
		return 0;
	// Output m (at position p) needs input index floor(p / up_) to be consumed.
	const int64_t p_last = next_p_ + static_cast<int64_t>(n_out - 1) * down_;
	const int64_t need_total = p_last / up_ + 1;
	const int64_t need = need_total - consumed_;
	return need > 0 ? static_cast<size_t>(need) : 0;
}

size_t RationalResampler::process(const float *in, size_t n_in, float *out)
{
	constexpr size_t kBlock = 4096;
	const size_t hist = static_cast<size_t>(taps_ - 1);
	if (work_.size() < hist + kBlock)
		work_.resize(hist + kBlock);

	size_t n_out = 0;
	while (n_in > 0) {
		const size_t n = std::min(n_in, kBlock);
		std::copy(hist_.begin(), hist_.end(), work_.begin());
		std::copy(in, in + n, work_.begin() + static_cast<std::ptrdiff_t>(hist));

		const int64_t end = consumed_ + static_cast<int64_t>(n);
		while (true) {
			const int64_t i0 = next_p_ / up_;  // newest input index used by this output
			if (i0 >= end)
				break;
			const int64_t ph = next_p_ - i0 * up_;
			const float *h = &h_[static_cast<size_t>(ph) * static_cast<size_t>(taps_)];
			// x[i0 - k] lives at work_[(i0 - consumed_) + hist - k]
			const float *x = &work_[static_cast<size_t>(i0 - consumed_) + hist];
			float acc = 0.0f;
			for (int k = 0; k < taps_; ++k)
				acc += h[k] * x[-static_cast<std::ptrdiff_t>(k)];
			out[n_out++] = acc;
			next_p_ += down_;
		}
		std::copy(work_.begin() + static_cast<std::ptrdiff_t>(n), work_.begin() + static_cast<std::ptrdiff_t>(n + hist),
			  hist_.begin());
		consumed_ = end;
		in += n;
		n_in -= n;
	}
	return n_out;
}

}  // namespace acw
