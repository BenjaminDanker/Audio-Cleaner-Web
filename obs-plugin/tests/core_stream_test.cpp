// SPDX-License-Identifier: GPL-2.0-or-later
//
// core_stream_test: drives acw_core exactly like the OBS filter does, against a
// live server, in real time.
//
//   core_stream_test --url ws://127.0.0.1:8791/api/stream/ws --key-file ~/audio-cleaner/dev/key
//       --in noisy.wav --out cleaned.wav [--clean clean.wav] [--captions local] [--rate 44100] ...
//
// The audio is fed in OBS-like irregular packets (10 ms and 1024-sample ticks),
// the delayed output is written to a WAV file, and RTT / late frames / noise
// reduction / captions are reported. The API key is read from a file and is
// never printed.
#include "acw/session.hpp"

#include "acw/reference.hpp"
#include "acw/resampler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace acw;
using Clock = std::chrono::steady_clock;

namespace {

struct Wav {
	std::vector<float> mono;
	int rate = 0;
};

uint32_t rd32(const unsigned char *p)
{
	return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t rd16(const unsigned char *p)
{
	return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

bool read_wav(const std::string &path, Wav &w)
{
	std::ifstream f(path, std::ios::binary);
	if (!f)
		return false;
	std::vector<unsigned char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (d.size() < 44 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4))
		return false;
	int fmt = 0, ch = 0, bits = 0;
	size_t pos = 12;
	while (pos + 8 <= d.size()) {
		const uint32_t sz = rd32(&d[pos + 4]);
		if (!std::memcmp(&d[pos], "fmt ", 4)) {
			fmt = rd16(&d[pos + 8]);
			ch = rd16(&d[pos + 10]);
			w.rate = static_cast<int>(rd32(&d[pos + 12]));
			bits = rd16(&d[pos + 22]);
		} else if (!std::memcmp(&d[pos], "data", 4)) {
			const size_t n = std::min<size_t>(sz, d.size() - pos - 8);
			const unsigned char *p = &d[pos + 8];
			if (ch < 1 || !((fmt == 1 && bits == 16) || (fmt == 3 && bits == 32)))
				return false;
			const size_t bytes = static_cast<size_t>(bits / 8) * static_cast<size_t>(ch);
			for (size_t i = 0; i + bytes <= n; i += bytes) {
				double s = 0;
				for (int c = 0; c < ch; ++c) {
					if (bits == 16)
						s += static_cast<int16_t>(rd16(p + i + 2 * static_cast<size_t>(c))) / 32768.0;
					else {
						uint32_t u = rd32(p + i + 4 * static_cast<size_t>(c));
						float v;
						std::memcpy(&v, &u, 4);
						s += v;
					}
				}
				w.mono.push_back(static_cast<float>(s / ch));
			}
			return true;
		}
		pos += 8 + sz + (sz & 1);
	}
	return false;
}

bool write_wav(const std::string &path, const std::vector<float> &x, int rate)
{
	std::ofstream f(path, std::ios::binary);
	if (!f)
		return false;
	auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
	auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };
	const uint32_t bytes = static_cast<uint32_t>(x.size() * 2);
	f.write("RIFF", 4);
	w32(36 + bytes);
	f.write("WAVEfmt ", 8);
	w32(16);
	w16(1);
	w16(1);
	w32(static_cast<uint32_t>(rate));
	w32(static_cast<uint32_t>(rate) * 2);
	w16(2);
	w16(16);
	f.write("data", 4);
	w32(bytes);
	for (float v : x) {
		const float c = std::min(1.0f, std::max(-1.0f, v));
		const int16_t s = static_cast<int16_t>(std::lrintf(c * 32767.0f));
		w16(static_cast<uint16_t>(s));
	}
	return static_cast<bool>(f);
}

std::string expand_home(const std::string &p)
{
	if (!p.empty() && p[0] == '~') {
		const char *h = std::getenv("HOME");
		return std::string(h ? h : "") + p.substr(1);
	}
	return p;
}

std::string read_key(const std::string &path)
{
	std::ifstream f(expand_home(path));
	std::string k;
	std::getline(f, k);
	while (!k.empty() && (k.back() == '\r' || k.back() == ' ' || k.back() == '\t'))
		k.pop_back();
	return k;
}

double rms(const std::vector<float> &x, size_t a, size_t b)
{
	if (b > x.size())
		b = x.size();
	if (a >= b)
		return 0;
	double e = 0;
	for (size_t i = a; i < b; ++i)
		e += static_cast<double>(x[i]) * x[i];
	return std::sqrt(e / static_cast<double>(b - a));
}

double db(double v)
{
	return 20.0 * std::log10(std::max(v, 1e-9));
}

double lag_estimate(const std::vector<float> &ref, const std::vector<float> &sig, size_t a, size_t b, int lo, int hi)
{
	auto corr = [&](int lag) {
		double s = 0;
		for (size_t i = a; i < b; ++i) {
			const long j = static_cast<long>(i) + lag;
			if (j >= 0 && j < static_cast<long>(sig.size()))
				s += static_cast<double>(ref[i]) * sig[static_cast<size_t>(j)];
		}
		return s;
	};
	int best = lo;
	double bv = -1e300;
	for (int l = lo; l <= hi; ++l) {
		const double c = corr(l);
		if (c > bv) {
			bv = c;
			best = l;
		}
	}
	return best;
}

struct Args {
	std::string url = "ws://127.0.0.1:8791/api/stream/ws";
	std::string key_file = "~/audio-cleaner/dev/key";
	std::string in, out, clean, reference;
	std::string captions = "";
	std::string language;
	std::string codec = "opus";
	int rate = 48000;
	int delay_ms = 150;
	int atten = 100;
	bool denoise = true;
	bool expect_fallback = false;
	bool realtime = true;
	double max_seconds = 0;  // limit the clip length (0 = all)
	int flush_ms = 12000;
};

void usage()
{
	std::printf("usage: core_stream_test --in noisy.wav [--out out.wav] [--clean clean.wav] [--url URL] [--key-file F]\n"
		    "       [--rate 48000|44100] [--delay-ms 80|150|250] [--denoise 0|1] [--captions local|premium]\n"
		    "       [--language en] [--atten 12|24|40|100] [--codec opus|pcm] [--expect-fallback] [--seconds N]\n"
		    "       [--reference speakers.wav]   what the speakers play: asks the server to cancel the echo\n");
}

}  // namespace

int main(int argc, char **argv)
{
	Args a;
	for (int i = 1; i < argc; ++i) {
		const std::string k = argv[i];
		auto val = [&]() -> std::string {
			if (i + 1 >= argc) {
				usage();
				std::exit(2);
			}
			return argv[++i];
		};
		if (k == "--url")
			a.url = val();
		else if (k == "--key-file")
			a.key_file = val();
		else if (k == "--in")
			a.in = expand_home(val());
		else if (k == "--out")
			a.out = expand_home(val());
		else if (k == "--clean")
			a.clean = expand_home(val());
		else if (k == "--reference")
			a.reference = expand_home(val());
		else if (k == "--captions")
			a.captions = val();
		else if (k == "--language")
			a.language = val();
		else if (k == "--codec")
			a.codec = val();
		else if (k == "--rate")
			a.rate = std::atoi(val().c_str());
		else if (k == "--delay-ms")
			a.delay_ms = std::atoi(val().c_str());
		else if (k == "--atten")
			a.atten = std::atoi(val().c_str());
		else if (k == "--denoise")
			a.denoise = std::atoi(val().c_str()) != 0;
		else if (k == "--seconds")
			a.max_seconds = std::atof(val().c_str());
		else if (k == "--flush-ms")
			a.flush_ms = std::atoi(val().c_str());
		else if (k == "--expect-fallback")
			a.expect_fallback = true;
		else if (k == "--no-realtime")
			a.realtime = false;
		else {
			usage();
			return 2;
		}
	}
	if (a.in.empty()) {
		usage();
		return 2;
	}

	Wav wav;
	if (!read_wav(a.in, wav)) {
		std::fprintf(stderr, "cannot read %s\n", a.in.c_str());
		return 2;
	}
	std::vector<float> in48 = wav.mono;
	if (wav.rate != kNetRate) {
		RationalResampler r(wav.rate, kNetRate);
		std::vector<float> o(r.max_output(in48.size()));
		o.resize(r.process(in48.data(), in48.size(), o.data()));
		in48 = o;
	}
	if (a.max_seconds > 0)
		in48.resize(std::min(in48.size(), static_cast<size_t>(a.max_seconds * kNetRate)));
	std::vector<float> clean48;
	if (!a.clean.empty()) {
		Wav c;
		if (read_wav(a.clean, c) && c.rate == kNetRate)
			clean48 = c.mono;
	}

	// Host-rate input (what OBS would deliver).
	std::vector<float> in = in48;
	if (a.rate != kNetRate) {
		RationalResampler r(kNetRate, a.rate);
		std::vector<float> o(r.max_output(in48.size()));
		o.resize(r.process(in48.data(), in48.size(), o.data()));
		in = o;
	}

	// Echo reference at the host rate, played in step with the input.
	std::vector<float> ref;
	std::shared_ptr<ReferenceBuffer> reference;
	if (!a.reference.empty()) {
		Wav w;
		if (!read_wav(a.reference, w)) {
			std::fprintf(stderr, "cannot read %s\n", a.reference.c_str());
			return 2;
		}
		ref = w.mono;
		if (w.rate != a.rate) {
			RationalResampler r(w.rate, a.rate);
			std::vector<float> o(r.max_output(ref.size()));
			o.resize(r.process(ref.data(), ref.size(), o.data()));
			ref = o;
		}
		reference = std::make_shared<ReferenceBuffer>(a.rate);
	}

	SessionConfig cfg;
	cfg.reference = reference;
	cfg.url = a.url;
	cfg.api_key = read_key(a.key_file);
	cfg.denoise = a.denoise;
	cfg.captions = a.captions;
	cfg.language = a.language;
	cfg.atten_lim_db = a.atten;
	cfg.delay_ms = a.delay_ms;
	cfg.host_rate = a.rate;
	cfg.client = "obs-audio-cleaner/0.1.0 (core_stream_test)";
	if (a.codec == "pcm")
		cfg.codec = Codec::Pcm16;
	else
		cfg.codec = Codec::Opus;
	if (cfg.api_key.empty()) {
		std::fprintf(stderr, "no API key in %s\n", a.key_file.c_str());
		return 2;
	}

	std::mutex mu;
	std::vector<std::string> captions;
	Session session(
		cfg,
		[&](const CaptionEvent &c) {
			std::lock_guard<std::mutex> lk(mu);
			captions.push_back(c.text);
			std::printf("CAPTION [%.2f-%.2f] %s\n", c.start, c.end, c.text.c_str());
		},
		[&](LogLevel lv, const std::string &m) {
			static const char *names[] = {"debug", "info", "warn", "error"};
			std::printf("[%s] %s\n", names[static_cast<int>(lv)], m.c_str());
		});

	const double delay_s = session.delay_seconds();
	const size_t delay_samples = static_cast<size_t>(std::lround(delay_s * a.rate));
	std::printf("input %.2f s @ %d Hz (%zu samples), codec=%s denoise=%d captions=%s, D = %.2f ms (%zu samples)\n",
		    static_cast<double>(in.size()) / a.rate, a.rate, in.size(), a.codec.c_str(), a.denoise ? 1 : 0,
		    a.captions.empty() ? "off" : a.captions.c_str(), delay_s * 1000.0, delay_samples);

	// Input = clip + silence tail of D so the whole delayed clip comes out.
	std::vector<float> fed = in;
	fed.resize(in.size() + delay_samples, 0.0f);
	std::vector<float> out = fed;  // processed in place, like OBS does

	// OBS-like packet sizes: 10 ms capture packets with an occasional 1024-frame tick.
	const size_t base = static_cast<size_t>(a.rate / 100);  // 480 or 441
	const size_t pattern[] = {base, base, base, 1024, base, base, base * 2, base, 1024, base};
	const size_t npat = sizeof(pattern) / sizeof(pattern[0]);

	session.start();
	const auto t0 = Clock::now();
	// The reference arrives on a thread of its own, in 10 ms chunks, like the capture callback of another source.
	std::atomic<bool> feeding{true};
	std::thread ref_thread;
	if (reference && a.realtime) {
		ref_thread = std::thread([&] {
			for (size_t i = 0; feeding && i * base < ref.size(); ++i) {
				std::this_thread::sleep_until(t0 + std::chrono::milliseconds(10 * (i + 1)));
				reference->push(&ref[i * base], std::min(base, ref.size() - i * base));
			}
		});
	}
	size_t pos = 0, k = 0;
	double worst_us = 0, sum_us = 0;
	size_t calls = 0;
	while (pos < out.size()) {
		const size_t m = std::min(pattern[k++ % npat], out.size() - pos);
		pos += m;
		if (a.realtime)
			std::this_thread::sleep_until(t0 + std::chrono::duration_cast<Clock::duration>(
								 std::chrono::duration<double>(static_cast<double>(pos) / a.rate)));
		if (reference && !a.realtime && pos <= ref.size())
			reference->push(&ref[pos - m], m);
		const auto c0 = Clock::now();
		session.process(&out[pos - m], m);
		const double us = std::chrono::duration<double, std::micro>(Clock::now() - c0).count();
		worst_us = std::max(worst_us, us);
		sum_us += us;
		++calls;
	}
	const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
	feeding = false;
	if (ref_thread.joinable())
		ref_thread.join();

	const SessionSnapshot snap = session.snapshot();
	std::printf("\nfed %zu samples in %.2f s wall time; session state: %s\n", fed.size(), wall, to_string(snap.state));
	std::printf("process() calls: %zu, mean %.1f us, worst %.1f us (audio-thread cost)\n", calls, sum_us / calls, worst_us);

	// Let the server flush outstanding captions and close.
	session.stop(a.expect_fallback ? 0 : a.flush_ms, 1500);
	const SessionSnapshot fin = session.snapshot();

	int failures = 0;
	auto expect = [&](bool ok, const char *what) {
		std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
		if (!ok)
			++failures;
	};

	std::printf("\nresults\n");
	std::printf("  output length %zu = input length %zu + D %zu\n", out.size(), in.size(), delay_samples);
	expect(out.size() == in.size() + delay_samples, "output length = input length + D");

	if (!a.out.empty())
		write_wav(a.out, out, a.rate);

	const std::vector<float> rtt = session.rtt_history();
	if (!rtt.empty()) {
		std::vector<float> s = rtt;
		std::sort(s.begin(), s.end());
		std::printf("  RTT over %zu frames: median %.1f ms, p95 %.1f ms, max %.1f ms\n", s.size(), s[s.size() / 2],
			    s[static_cast<size_t>(0.95 * static_cast<double>(s.size() - 1))], s.back());
	}
	std::printf("  frames: captured %llu sent %llu received %llu | played processed %llu, late %llu, passthrough %llu, dropped %llu, warm-up skipped %llu, bad %llu\n",
		    (unsigned long long)fin.captured, (unsigned long long)fin.sent, (unsigned long long)fin.received,
		    (unsigned long long)fin.used, (unsigned long long)fin.late, (unsigned long long)fin.passthrough,
		    (unsigned long long)fin.dropped, (unsigned long long)fin.warmup_discarded, (unsigned long long)fin.bad_frames);
	if (fin.sent > 0)
		std::printf("  traffic: uplink %.1f kbps, downlink %.1f kbps (audio frames incl. 16-byte headers, %s)\n",
			    fin.tx_bytes * 8.0 / 1000.0 / (static_cast<double>(fed.size()) / a.rate),
			    fin.rx_bytes * 8.0 / 1000.0 / (static_cast<double>(fed.size()) / a.rate), a.codec.c_str());
	std::printf("  reconnects: %u   status: %s\n", fin.reconnects, session.status_line().c_str());
	std::printf("  audio thread: worst process() call %.1f us\n", worst_us);

	// Output aligned to the input timeline (undo the fixed delay).
	auto aligned = [&](size_t i) -> float { return i + delay_samples < out.size() ? out[i + delay_samples] : 0.0f; };
	std::vector<float> y(in.size());
	for (size_t i = 0; i < y.size(); ++i)
		y[i] = aligned(i);

	if (a.expect_fallback) {
		// Server unreachable: the output must be the delayed original audio.
		double maxdiff = 0, err = 0, sig = 0;
		for (size_t i = 0; i < in.size(); ++i) {
			const double d = static_cast<double>(y[i]) - in[i];
			maxdiff = std::max(maxdiff, std::fabs(d));
			err += d * d;
			sig += static_cast<double>(in[i]) * in[i];
		}
		const double snr = 10.0 * std::log10(sig / std::max(err, 1e-30));
		std::printf("  fallback: max |output(delayed) - input| = %.2e, SNR of the delayed original = %.1f dB%s\n", maxdiff,
			    snr, a.rate == kNetRate ? " (bit exact path)" : " (44.1 -> 48 -> 44.1 kHz resampler round trip)");
		expect(a.rate == kNetRate ? maxdiff < 1e-6 : snr > 30.0, "passthrough returns the delayed original audio");
		expect(fin.reconnects >= 2, "reconnect attempts happened (backoff)");
		expect(fin.used == 0 && fin.sent == 0, "no processed audio was used and nothing was sent");
		expect(fin.state == SessionState::Stopped, "session stops cleanly");
		std::printf("  worst audio-thread call during outage: %.1f us\n", worst_us);
		return failures ? 1 : 0;
	}

	if (a.denoise) {
		// Alignment of the processed output with the input timeline.
		const size_t lo = in.size() / 4, hi = in.size() * 3 / 4;
		const int span = static_cast<int>(a.rate * 0.01);
		const double lag = lag_estimate(in, y, lo, std::min(hi, lo + static_cast<size_t>(a.rate)), -span, span);
		std::printf("  alignment of cleaned output vs. input (after removing D): %+.0f samples\n", lag);
		expect(std::fabs(lag) <= 2, "cleaned audio is time-aligned with the input (|lag| <= 2 samples)");

		// Noise reduction in noise-only segments, found with the clean reference when given,
		// otherwise as the quietest 0.5 s windows of the input.
		const size_t win = static_cast<size_t>(0.4 * a.rate);
		std::vector<size_t> starts;
		if (!clean48.empty()) {
			std::vector<float> c = clean48;
			if (a.rate != kNetRate) {
				RationalResampler r(kNetRate, a.rate);
				std::vector<float> o(r.max_output(c.size()));
				o.resize(r.process(c.data(), c.size(), o.data()));
				c = o;
			}
			for (size_t s = static_cast<size_t>(0.8 * a.rate); s + win < std::min(in.size(), c.size()); s += win / 2)
				if (db(rms(c, s, s + win)) < -55.0)
					starts.push_back(s);
		}
		if (starts.empty()) {
			std::vector<std::pair<double, size_t>> e;
			for (size_t s = static_cast<size_t>(0.8 * a.rate); s + win < in.size(); s += win / 2)
				e.push_back({rms(in, s, s + win), s});
			std::sort(e.begin(), e.end());
			for (size_t i = 0; i < std::min<size_t>(3, e.size()); ++i)
				starts.push_back(e[i].second);
			std::printf("  (no clean reference: using the %zu quietest input windows as noise-only)\n", starts.size());
		}
		double in_e = 0, out_e = 0;
		for (size_t s : starts) {
			// skip 0.25 s after the start of each window for the model/fade to settle
			const size_t a0 = s, b0 = s + win;
			in_e += std::pow(rms(in, a0, b0), 2);
			out_e += std::pow(rms(y, a0, b0), 2);
		}
		const double r_in = std::sqrt(in_e / static_cast<double>(starts.size()));
		const double r_out = std::sqrt(out_e / static_cast<double>(starts.size()));
		std::printf("  noise-only segments (%zu x %.0f ms): input RMS %.1f dBFS -> output RMS %.1f dBFS = %.1f dB reduction\n",
			    starts.size(), win * 1000.0 / a.rate, db(r_in), db(r_out), db(r_in) - db(r_out));
		expect(db(r_in) - db(r_out) > 10.0, "noise-only RMS reduced by more than 10 dB");

		// Speech must survive: overall RMS of the loudest windows stays within a few dB.
		std::vector<std::pair<double, size_t>> e;
		for (size_t s = 0; s + win < in.size(); s += win)
			e.push_back({rms(in, s, s + win), s});
		std::sort(e.rbegin(), e.rend());
		double sp_in = 0, sp_out = 0;
		const size_t nsp = std::min<size_t>(4, e.size());
		for (size_t i = 0; i < nsp; ++i) {
			sp_in += std::pow(rms(in, e[i].second, e[i].second + win), 2);
			sp_out += std::pow(rms(y, e[i].second, e[i].second + win), 2);
		}
		std::printf("  loudest windows (speech): input %.1f dBFS -> output %.1f dBFS\n", db(std::sqrt(sp_in / nsp)),
			    db(std::sqrt(sp_out / nsp)));
		expect(std::fabs(db(std::sqrt(sp_in / nsp)) - db(std::sqrt(sp_out / nsp))) < 6.0, "speech level preserved (within 6 dB)");
		expect(fin.late == 0, "zero late frames");
		expect(fin.used > 0.8 * static_cast<double>(fin.captured), "most output blocks were played from processed audio");
	}

	if (!a.captions.empty()) {
		std::lock_guard<std::mutex> lk(mu);
		std::printf("  captions received: %zu\n", captions.size());
		expect(!captions.empty(), "captions received");
	}
	return failures ? 1 : 0;
}
