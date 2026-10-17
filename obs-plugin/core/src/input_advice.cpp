// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/input_advice.hpp"

#include <algorithm>
#include <cmath>

namespace acw {

namespace {

// Thresholds: the same as web/src/recommend.js (T), calibrated in docs/research/recommend-2026-10.md.
constexpr double kCleanAtLeast = 29.0;      // noise figure from which the room counts as quiet
constexpr double kHeavyBelow = 19.0;        // the service's own cut for its stronger noise remover (pipeline.HEAVY_BELOW_DB)
constexpr double kMaxBelow = 18.0;          // strength steps by noise figure: Maximum below this, then Strong, Balanced, Gentle
constexpr double kStrongBelow = 21.0;
constexpr double kBalancedBelow = 27.0;
constexpr double kCaptionsNoisyBelow = 18.0;
constexpr double kSpeechBandDyn = 6.8;      // the speech band's level moves at least this much (dB) when someone speaks
constexpr double kClipFraction = 0.002;
constexpr double kEchoShare = 0.03;
constexpr int kMinSilenceLevels = 10;
constexpr float kSilenceDb = -90.0f;
constexpr float kClipLevel = 0.999f;
constexpr uint32_t kClipRun = 3;
constexpr size_t kFrame = 480;  // 10 ms

// RBJ biquad (Butterworth) at 48 kHz, normalised: b0 b1 b2 a1 a2.
struct Biquad {
	double b0, b1, b2, a1, a2;
};

Biquad make_biquad(bool high, double fc)
{
	constexpr double kPi = 3.14159265358979323846;
	const double w = 2.0 * kPi * fc / 48000.0;
	const double c = std::cos(w), alpha = std::sin(w) / (2.0 * 0.70710678118654752);
	const double a0 = 1.0 + alpha;
	const double k = high ? 1.0 + c : 1.0 - c;
	return {k / 2 / a0, (high ? -k : k) / a0, k / 2 / a0, -2.0 * c / a0, (1.0 - alpha) / a0};
}

const Biquad &band_hp()
{
	static const Biquad b = make_biquad(true, 150.0);
	return b;
}

const Biquad &band_lp()
{
	static const Biquad b = make_biquad(false, 3500.0);
	return b;
}

double step(const Biquad &q, double *m, double x)
{
	const double y = q.b0 * x + q.b1 * m[0] + q.b2 * m[1] - q.a1 * m[2] - q.a2 * m[3];
	m[1] = m[0];
	m[0] = x;
	m[3] = m[2];
	m[2] = y;
	return y;
}

double percentile(const std::vector<float> &sorted, double q)
{
	const double pos = q / 100.0 * static_cast<double>(sorted.size() - 1);
	const size_t lo = static_cast<size_t>(pos);
	const size_t hi = std::min(sorted.size() - 1, lo + 1);
	return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
}

int strength_for(double noise_db)
{
	if (noise_db < kMaxBelow)
		return 100;
	if (noise_db < kStrongBelow)
		return 40;
	return noise_db < kBalancedBelow ? 24 : 12;
}

const char *strength_name(int db)
{
	switch (db) {
	case 12: return "Gentle";
	case 24: return "Balanced";
	case 40: return "Strong";
	default: return "Maximum";
	}
}

bool english_only(const std::string &language)
{
	return language.empty() || language == "en";
}

}  // namespace

InputAnalyser::InputAnalyser(bool with_reference, double seconds, double max_seconds)
	: with_ref_(with_reference), need_levels_(static_cast<size_t>(seconds * 100.0))
{
	levels_.assign(static_cast<size_t>(max_seconds * 100.0), 0.0f);
	band_levels_.assign(levels_.size(), 0.0f);
	if (with_ref_) {
		const size_t n = static_cast<size_t>(max_seconds * 48000.0 / kDecimate);
		mic2_.assign(n, 0.0f);
		ref2_.assign(n, 0.0f);
	}
}

void InputAnalyser::push(const float *mic, const float *ref, size_t n)
{
	const uint32_t cap = static_cast<uint32_t>(levels_.size());
	uint32_t count = n_levels_.load(std::memory_order_relaxed);
	if (count >= cap)
		return;  // all that is wanted has been kept
	uint64_t clipped = 0;
	for (size_t i = 0; i < n; ++i) {
		const float v = std::isfinite(mic[i]) ? mic[i] : 0.0f;
		sum_sq_ += static_cast<double>(v) * v;
		const double bv = step(band_lp(), lp_, step(band_hp(), hp_, v));
		band_sq_ += bv * bv;
		if (std::fabs(v) >= kClipLevel) {
			++run_;
			if (run_ == kClipRun)
				clipped += kClipRun;
			else if (run_ > kClipRun)
				++clipped;
		} else {
			run_ = 0;
		}
		if (++in_frame_ == kFrame) {
			if (count < cap) {
				band_levels_[count] = static_cast<float>(10.0 * std::log10(band_sq_ / kFrame + 1e-12));
				levels_[count++] = static_cast<float>(10.0 * std::log10(sum_sq_ / kFrame + 1e-12));
				n_levels_.store(count, std::memory_order_release);
			}
			sum_sq_ = 0;
			band_sq_ = 0;
			in_frame_ = 0;
		}
		if (with_ref_) {
			acc_m_ += v;
			acc_r_ += ref ? (std::isfinite(ref[i]) ? ref[i] : 0.0f) : 0.0f;
			if (++acc_n_ == kDecimate) {
				const uint32_t k = n_2k_.load(std::memory_order_relaxed);
				if (k < mic2_.size()) {
					mic2_[k] = static_cast<float>(acc_m_ / kDecimate);
					ref2_[k] = static_cast<float>(acc_r_ / kDecimate);
					n_2k_.store(k + 1, std::memory_order_release);
				}
				acc_m_ = acc_r_ = 0;
				acc_n_ = 0;
			}
		}
	}
	samples_.fetch_add(n, std::memory_order_relaxed);
	if (clipped)
		clipped_.fetch_add(clipped, std::memory_order_relaxed);
}

bool InputAnalyser::ready() const
{
	return n_levels_.load(std::memory_order_acquire) >= need_levels_;
}

double InputAnalyser::seconds_heard() const
{
	return n_levels_.load(std::memory_order_acquire) / 100.0;
}

InputFindings InputAnalyser::findings() const
{
	InputFindings f;
	const uint32_t n = n_levels_.load(std::memory_order_acquire);
	f.seconds = n / 100.0;
	f.final = n >= levels_.size();
	if (n < need_levels_)
		return f;
	std::vector<float> loud, band;
	loud.reserve(n);
	band.reserve(n);
	for (uint32_t i = 0; i < n; ++i)
		if (levels_[i] > kSilenceDb) {
			loud.push_back(levels_[i]);
			band.push_back(band_levels_[i]);
		}
	f.ready = true;
	const uint64_t samples = samples_.load(std::memory_order_relaxed);
	f.clip_fraction = samples ? static_cast<double>(clipped_.load(std::memory_order_relaxed)) / samples : 0.0;
	if (static_cast<int>(loud.size()) >= kMinSilenceLevels) {
		std::sort(loud.begin(), loud.end());
		const double p10 = percentile(loud, 10), p90 = percentile(loud, 90);
		f.noise_db = p90 - p10;
		f.floor_dbfs = p10;
		f.level_dbfs = p90;
		std::sort(band.begin(), band.end());
		f.has_speech = percentile(band, 90) - percentile(band, 10) >= kSpeechBandDyn;
	}
	if (with_ref_) {
		const uint32_t m = n_2k_.load(std::memory_order_acquire);
		std::lock_guard<std::mutex> lk(mu_);
		if (m >= kLagMax * 4 && (echo_at_ == 0 || m >= echo_at_ + 10000 || m >= mic2_.size())) {
			double er = 0, em = 0;
			for (uint32_t i = 0; i < m; ++i) {
				er += static_cast<double>(ref2_[i]) * ref2_[i];
				em += static_cast<double>(mic2_[i]) * mic2_[i];
			}
			double best = 0;
			if (er > 1e-9 && em > 1e-9) {
				for (size_t lag = 0; lag <= kLagMax; ++lag) {
					double c = 0, r2 = 0;
					for (uint32_t i = static_cast<uint32_t>(lag); i < m; ++i) {
						const double r = ref2_[i - lag];
						c += mic2_[i] * r;
						r2 += r * r;
					}
					if (r2 < 1e-12)
						continue;
					best = std::max(best, (c * c) / (r2 * em));
				}
			}
			echo_share_ = best;
			echo_at_ = m;
		}
		f.echo_known = echo_at_ != 0;
		f.echo_share = echo_share_;
	}
	return f;
}

InputAdvice InputAnalyser::advise(const InputFindings &f, const AdviceContext &ctx)
{
	InputAdvice a;
	if (!f.ready)
		return a;
	a.ready = true;
	a.captions = ctx.captions.empty() ? "local" : ctx.captions;
	if (!f.has_speech) {
		// nobody spoke while it listened: ask once more while it still listens, then say nothing (it will not listen again)
		if (!f.final)
			a.notes.push_back({"advice_speak", "Speak for a few seconds and this will suggest the best settings for your room.", ""});
		return a;
	}
	const bool clean = f.noise_db >= kCleanAtLeast;
	a.quiet_room = clean;
	a.atten_lim_db = clean ? 100 : strength_for(f.noise_db);
	if (clean) {
		if (ctx.denoise)
			a.notes.push_back({"advice_quiet", "Your room is quiet, so noise removal is not needed. Turn it off to save credit.", ""});
	} else if (!ctx.denoise && ctx.captions.empty()) {  // (with captions only, the sound is not what the person is after)
		a.notes.push_back({"advice_noisy", "Your room is noisy. Turn on noise removal (%1 suits it).", strength_name(a.atten_lim_db)});
	} else if (ctx.atten_lim_db != a.atten_lim_db) {
		a.notes.push_back({"advice_strength", "For your room, set the noise removal strength to %1.", strength_name(a.atten_lim_db)});
	}
	a.echo_heard = f.echo_known && f.echo_share >= kEchoShare;
	if (a.echo_heard) {
		if (ctx.echo_active)
			a.notes.push_back({"advice_echo_on", "Your speakers can be heard in your microphone. Echo cancellation is taking it out.", ""});
		else
			a.notes.push_back({"advice_echo_off", "Your speakers can be heard in your microphone. Wear headphones, or turn on echo cancellation.", ""});
	}
	if (f.clip_fraction > kClipFraction)
		a.notes.push_back({"advice_clip", "Your microphone is too loud and the sound distorts. Turn its level down.", ""});
	if (!ctx.captions.empty()) {
		const bool hard = f.noise_db < kCaptionsNoisyBelow || !english_only(ctx.language);
		if (hard) {
			a.captions = "premium";
			if (ctx.captions == "local")
				a.notes.push_back({"advice_captions_up", "Extra accurate captions will catch more words here.", ""});
		} else {
			a.captions = "local";
			if (ctx.captions == "premium")
				a.notes.push_back({"advice_captions_down", "Standard captions are enough for your clear voice.", ""});
		}
	}
	if (a.notes.empty())
		a.notes.push_back({"advice_fits", "The settings fit your room.", ""});
	return a;
}

// ------------------------------------------------------------------------------------------------- RoomTest

RoomTest::RoomTest(int rate, size_t max_block, double seconds)
	: seconds_(seconds), a_(false, seconds, seconds)
{
	if (rate != 48000) {
		rs_ = std::make_unique<RationalResampler>(rate, 48000);
		buf_.assign(rs_->max_output(max_block), 0.0f);
	}
}

void RoomTest::push(const float *mic, size_t n)
{
	if (done())
		return;
	if (!rs_) {
		a_.push(mic, nullptr, n);
		return;
	}
	const size_t m = rs_->process(mic, std::min(n, buf_.size() - 2), buf_.data());
	a_.push(buf_.data(), nullptr, m);
}

double RoomTest::seconds_left() const
{
	return std::max(0.0, seconds_ - a_.seconds_heard());
}

bool RoomTest::done() const
{
	return a_.ready();
}

RoomSuggestion RoomTest::suggest(const AdviceContext &now) const
{
	RoomSuggestion r;
	r.settings = now;
	const InputFindings f = a_.findings();
	r.done = f.ready;
	r.heard_speech = f.ready && f.has_speech;
	if (!r.heard_speech)
		return r;
	const InputAdvice adv = InputAnalyser::advise(f, now);
	auto change = [&](const char *id, const char *what, const std::string &from, const std::string &to) {
		r.changes.push_back({std::string("room.") + id, std::string(what) + ": %1", from + " \xE2\x86\x92 " + to});
	};
	if (adv.quiet_room) {
		r.settings.denoise = false;
		if (now.denoise)
			change("denoise", "Noise removal", "on", "off");
	} else {
		r.settings.denoise = true;
		r.settings.atten_lim_db = adv.atten_lim_db;
		if (!now.denoise)
			change("denoise", "Noise removal", "off", std::string("on, ") + strength_name(adv.atten_lim_db));
		else if (now.atten_lim_db != adv.atten_lim_db)
			change("strength", "Noise removal strength", strength_name(now.atten_lim_db), strength_name(adv.atten_lim_db));
	}
	if (!now.captions.empty() && adv.captions != now.captions) {
		auto name = [](const std::string &c) { return c == "premium" ? "Extra accurate" : "Standard"; };
		r.settings.captions = adv.captions;
		change("captions", "Captions", name(now.captions), name(adv.captions));
	}
	for (const AdviceNote &n : adv.notes)
		if (n.id == "advice_clip" || n.id == "advice_echo_off")
			r.notes.push_back(n);
	return r;
}

}  // namespace acw
