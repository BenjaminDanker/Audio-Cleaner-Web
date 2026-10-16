// SPDX-License-Identifier: GPL-2.0-or-later
// Offline unit tests for acw_core (no network, no OBS).
#include "acw/backoff.hpp"
#include "acw/caption_pacer.hpp"
#include "acw/opus_codec.hpp"
#include "acw/output_stage.hpp"
#include "acw/protocol.hpp"
#include "acw/reference.hpp"
#include "acw/resampler.hpp"
#include "acw/session.hpp"
#include "acw/spsc.hpp"
#include "acw/update_check.hpp"
#include "acw/update_verify.hpp"
#include "update_test_util.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <thread>
#include <vector>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                                       \
	do {                                                                              \
		++g_checks;                                                               \
		if (!(cond)) {                                                            \
			++g_failures;                                                     \
			std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
		}                                                                         \
	} while (0)

#define CHECK_NEAR(a, b, tol)                                                                                  \
	do {                                                                                                   \
		++g_checks;                                                                                    \
		const double va = (a), vb = (b);                                                               \
		if (!(std::fabs(va - vb) <= (tol))) {                                                          \
			++g_failures;                                                                          \
			std::printf("  FAIL %s:%d: %s = %.6g, expected %.6g +- %g\n", __FILE__, __LINE__, #a, va, vb, \
				    static_cast<double>(tol));                                                 \
		}                                                                                              \
	} while (0)

using namespace acw;

static void test_spsc()
{
	std::printf("spsc queue\n");
	SpscQueue<int> q(8);
	CHECK(q.capacity() == 8);
	CHECK(q.front() == nullptr);
	for (int i = 0; i < 8; ++i)
		CHECK(q.push(i));
	CHECK(!q.push(99));
	CHECK(q.claim() == nullptr);
	for (int i = 0; i < 8; ++i) {
		int *f = q.front();
		CHECK(f && *f == i);
		q.pop();
	}
	CHECK(q.front() == nullptr);

	// Two threads, 1M items, order preserved.
	SpscQueue<uint32_t> big(64);
	constexpr uint32_t N = 1'000'000;
	std::atomic<bool> ok{true};
	std::thread prod([&] {
		for (uint32_t i = 0; i < N;) {
			if (big.push(i))
				++i;
		}
	});
	std::thread cons([&] {
		uint32_t expect = 0;
		while (expect < N) {
			if (uint32_t *v = big.front()) {
				if (*v != expect)
					ok = false;
				++expect;
				big.pop();
			}
		}
	});
	prod.join();
	cons.join();
	CHECK(ok.load());
}

static void test_protocol()
{
	std::printf("protocol framing + json\n");
	std::vector<float> pcm(960);
	for (size_t i = 0; i < pcm.size(); ++i)
		pcm[i] = std::sin(0.05f * static_cast<float>(i)) * 0.8f;
	pcm[3] = 2.0f;   // clipped
	pcm[4] = -2.0f;
	std::vector<uint8_t> buf(kHeaderSize + 2 * pcm.size());
	const size_t len = encode_frame_s16(0x01020304u, 0x1122334455667788ull, pcm.data(), pcm.size(), buf.data());
	CHECK(len == buf.size());
	CHECK(buf[0] == 1 && buf[1] == 1 && buf[2] == 0 && buf[3] == 0);
	CHECK(buf[4] == 0x04 && buf[7] == 0x01);  // little endian seq
	CHECK(buf[8] == 0x88 && buf[15] == 0x11);  // little endian ts_us
	FrameHeader h;
	std::vector<float> out(kMaxRxSamples);
	size_t n = 0;
	CHECK(decode_frame(buf.data(), len, h, out.data(), out.size(), n));
	CHECK(n == 960 && h.seq == 0x01020304u && h.ts_us == 0x1122334455667788ull && h.codec == 1);
	CHECK_NEAR(out[10], pcm[10], 1.0 / 16384);
	CHECK_NEAR(out[3], 32767.0 / 32768.0, 1e-4);
	CHECK_NEAR(out[4], -32767.0 / 32768.0, 1e-4);
	CHECK(!decode_frame(buf.data(), 10, h, out.data(), out.size(), n));
	buf[0] = 2;
	CHECK(!decode_frame(buf.data(), len, h, out.data(), out.size(), n));

	StartParams sp;
	sp.denoise = true;
	sp.captions = "local";
	sp.language = "";
	sp.atten_lim_db = 40;
	sp.client = "obs-audio-cleaner/test";
	const std::string s = make_start_message(sp);
	CHECK(s.find("\"type\":\"start\"") != std::string::npos);
	CHECK(s.find("\"denoise\":\"light\"") != std::string::npos);
	CHECK(s.find("\"captions\":\"local\"") != std::string::npos);
	CHECK(s.find("\"language\":null") != std::string::npos);
	CHECK(s.find("\"sampleRate\":48000") != std::string::npos);
	CHECK(s.find("\"attenLimDb\":40") != std::string::npos);
	CHECK(s.find("\"codec\":\"pcm_s16le\"") != std::string::npos && s.find("opusBitrate") == std::string::npos);
	sp.codec = "opus";
	sp.opus_bitrate = 64000;
	const std::string so = make_start_message(sp);
	CHECK(so.find("\"codec\":\"opus\"") != std::string::npos && so.find("\"opusBitrate\":64000") != std::string::npos);
	uint8_t hdr[kHeaderSize];
	encode_frame_header(kCodecOpus, 7, 99, hdr);
	FrameHeader oh;
	CHECK(parse_frame_header(hdr, sizeof hdr, oh) && oh.codec == 3 && oh.seq == 7 && oh.ts_us == 99 && oh.flags == 0);
	// echo reference: flags bit 0, nothing else changes; the start message only mentions it when asked for
	encode_frame_header(kCodecOpus, 7, 99, hdr, kFlagReference);
	CHECK(parse_frame_header(hdr, sizeof hdr, oh) && oh.flags == 1 && oh.codec == 3 && oh.seq == 7);
	CHECK(encode_frame_s16(5, 6, pcm.data(), pcm.size(), buf.data(), kFlagReference) == len && buf[2] == 1 && buf[3] == 0);
	CHECK(decode_frame(buf.data(), len, h, out.data(), out.size(), n) && h.flags == kFlagReference && n == 960);
	CHECK(s.find("echoReference") == std::string::npos && so.find("echoReference") == std::string::npos);
	sp.echo_reference = true;
	CHECK(make_start_message(sp).find("\"echoReference\":true") != std::string::npos);
	sp.denoise = false;
	sp.captions = "";
	const std::string s2 = make_start_message(sp);
	CHECK(s2.find("\"denoise\":null") != std::string::npos);
	CHECK(s2.find("\"captions\":null") != std::string::npos);

	Event ev;
	CHECK(parse_event(R"({"type":"ready","sessionId":"abc","sampleRate":48000,"hopSamples":480,"modelLatencyMs":30.0,"usdPerMinute":0.01,"maxSessionSeconds":43200})", ev));
	CHECK(ev.type == EventType::Ready && ev.session_id == "abc" && ev.model_latency_ms == 30.0 && ev.sample_rate == 48000);
	CHECK(!ev.echo_reference);  // an older server does not mention it
	CHECK(parse_event(R"({"type":"ready","sessionId":"abc","sampleRate":48000,"modelLatencyMs":39.0,"echoReference":true})", ev));
	CHECK(ev.type == EventType::Ready && ev.echo_reference && ev.model_latency_ms == 39.0);
	CHECK(parse_event(R"({"type":"caption","text":"héllo 😀","final":true,"start":1.5,"end":2.5})", ev));
	CHECK(ev.type == EventType::Caption && ev.is_final && ev.text == "h\xc3\xa9llo \xf0\x9f\x98\x80" && ev.end == 2.5);
	CHECK(parse_event(R"({"type":"caption","text":"par","final":false,"id":"item_1"})", ev));
	CHECK(ev.type == EventType::Caption && !ev.is_final && ev.caption_id == "item_1");
	CHECK(parse_event(R"({"type":"warning","code":"low_balance","secondsLeft":180,"availableMicros":0})", ev));
	CHECK(ev.type == EventType::Warning && ev.code == "low_balance" && ev.seconds_left == 180);
	CHECK(parse_event(R"({"type":"stopped","reason":"insufficient_funds"})", ev));
	CHECK(ev.type == EventType::Stopped && ev.code == "insufficient_funds");
	CHECK(parse_event(R"({"type":"error","code":"tos","message":"Accept the terms"})", ev));
	CHECK(ev.type == EventType::Error && ev.code == "tos");
	CHECK(parse_event(R"({"type":"pong","t":123456,"seconds":4.2})", ev));
	CHECK(ev.type == EventType::Pong && ev.has_pong_t && ev.pong_t == 123456);
	// retry hints and hostile numbers
	CHECK(parse_event(R"({"type":"error","code":"unavailable","message":"busy","retryAfterSeconds":90})", ev));
	CHECK(ev.type == EventType::Error && ev.retry_after_s == 90);
	CHECK(parse_event(R"({"type":"error","code":"x","retryAfter":99999999})", ev) && ev.retry_after_s == 3600);
	CHECK(parse_event(R"({"type":"error","code":"x","retryAfter":-4})", ev) && ev.retry_after_s == 0);
	CHECK(parse_event(R"({"type":"error","code":"x"})", ev) && ev.retry_after_s == 0);
	CHECK(parse_event(R"({"type":"warning","code":"low_balance","secondsLeft":1e300})", ev));
	CHECK(ev.seconds_left == 1000000000);
	CHECK(!parse_event("not json", ev));
	CHECK(parse_event(R"({"type":"something_new"})", ev) && ev.type == EventType::Unknown);
}

static void test_opus()
{
	std::printf("opus\n");
	CHECK(opus_available());
	OpusEnc enc(64000);
	OpusDec dec;
	CHECK(enc.ok() && dec.ok());
	const int la = enc.lookahead();
	std::printf("  encoder look-ahead %d samples (%.2f ms)\n", la, la / 48.0);
	CHECK(la > 100 && la < 960);
	const double kPi = 3.14159265358979323846;
	const int frames = 100;
	std::vector<float> x(static_cast<size_t>(frames) * 960), y;
	for (size_t i = 0; i < x.size(); ++i) {
		const double t = static_cast<double>(i) / 48000.0;
		x[i] = static_cast<float>(0.3 * std::sin(2 * kPi * 220 * t) * (0.6 + 0.4 * std::sin(2 * kPi * 3 * t)) +
					  0.1 * std::sin(2 * kPi * 1800 * t));
	}
	size_t bytes = 0;
	std::vector<uint8_t> pkt(kMaxOpusPacket);
	std::vector<float> d(kMaxRxSamples);
	for (int f = 0; f < frames; ++f) {
		const int n = enc.encode(&x[static_cast<size_t>(f) * 960], pkt.data(), static_cast<int>(pkt.size()));
		CHECK(n > 0);
		bytes += static_cast<size_t>(n);
		const int got = dec.decode(pkt.data(), n, d.data());
		CHECK(got == 960);
		y.insert(y.end(), d.begin(), d.begin() + 960);
	}
	// decoded[i] ~ x[i - look-ahead]
	double err = 0, sig = 0;
	for (size_t i = 9600; i + 960 < x.size(); ++i) {
		const double ref = x[i - static_cast<size_t>(la)], dv = y[i] - ref;
		err += dv * dv;
		sig += ref * ref;
	}
	const double snr = 10 * std::log10(sig / err);
	const double kbps = static_cast<double>(bytes) * 8 / (frames * 0.02) / 1000.0;
	std::printf("  round trip SNR %.1f dB at the codec delay, %.1f kbps payload\n", snr, kbps);
	CHECK(snr > 15.0);
	CHECK(kbps > 20 && kbps < 90);
	CHECK(dec.decode(pkt.data(), 0, d.data()) <= 0 || true);  // must not crash on empty input
	uint8_t junk[5] = {1, 2, 3, 4, 5};
	(void)dec.decode(junk, 5, d.data());  // must not crash on garbage
}

static void test_backoff()
{
	std::printf("backoff\n");
	Backoff b(std::chrono::milliseconds(1000), std::chrono::milliseconds(60000), 42);
	std::vector<int64_t> v;
	for (int i = 0; i < 10; ++i)
		v.push_back(b.next().count());
	for (int i = 0; i < 10; ++i) {
		const int64_t nominal = std::min<int64_t>(60000, 1000ll << i);
		CHECK(v[i] >= nominal / 2 && v[i] <= nominal);
	}
	CHECK(v[9] <= 60000);
	b.reset();
	const int64_t first = b.next().count();
	CHECK(first >= 500 && first <= 1000);
	// jitter: two different seeds must differ somewhere
	Backoff b2(std::chrono::milliseconds(1000), std::chrono::milliseconds(60000), 7);
	bool differs = false;
	Backoff b3(std::chrono::milliseconds(1000), std::chrono::milliseconds(60000), 8);
	for (int i = 0; i < 5; ++i)
		differs |= b2.next() != b3.next();
	CHECK(differs);
}

static void test_caption_lines()
{
	std::printf("caption line splitting\n");
	auto l = split_caption_lines("  Hello   world, this is a rather long caption sentence for testing.  ", 32);
	for (auto &s : l)
		CHECK(s.size() <= 32);
	CHECK(l.size() == 3 - 1 || l.size() == 3);
	std::string joined;
	for (auto &s : l)
		joined += (joined.empty() ? "" : " ") + s;
	CHECK(joined == "Hello world, this is a rather long caption sentence for testing.");
	auto l2 = split_caption_lines(std::string(70, 'x'), 32);
	CHECK(l2.size() == 3 && l2[0].size() == 32 && l2[2].size() == 6);
	// UTF-8: 40 two-byte characters => lines of 32 code points, never split mid-sequence
	std::string u;
	for (int i = 0; i < 40; ++i)
		u += "\xc3\xa9";
	auto l3 = split_caption_lines(u, 32);
	CHECK(l3.size() == 2 && l3[0].size() == 64 && l3[1].size() == 16);
	CHECK(split_caption_lines("   ", 32).empty());

	// Pacer: rapid pushes with a tiny queue drop the backlog; the newest line survives.
	std::mutex m;
	std::vector<std::string> sent;
	CaptionPacer::Options o;
	o.max_queue_lines = 2;
	o.min_gap_seconds = 0.05;
	o.chars_per_second = 1000;
	CaptionPacer p(
		[&](const std::string &line, double dur) {
			CHECK(dur >= 1.5 && dur <= 4.0);
			std::lock_guard<std::mutex> lk(m);
			sent.push_back(line);
		},
		o);
	for (int i = 0; i < 6; ++i)
		p.push("line " + std::to_string(i));
	std::this_thread::sleep_for(std::chrono::milliseconds(400));
	{
		std::lock_guard<std::mutex> lk(m);
		CHECK(!sent.empty() && sent.size() <= 4);
		CHECK(sent.back() == "line 5");
	}
	CHECK(p.dropped_lines() >= 2);
	p.set_enabled(false);
	p.push("ignored");
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	{
		std::lock_guard<std::mutex> lk(m);
		for (auto &s : sent)
			CHECK(s != "ignored");
	}
	p.stop();
}

static void test_caption_pacing()
{
	std::printf("caption conversion and pacing\n");
	// Typographic characters become plain ones, accents the channel has stay, missing letters are folded.
	CHECK(to_cea608("“It’s” – ok…") == "\"It's\" - ok...");
	CHECK(to_cea608("café ñandú") == "café ñandú");
	CHECK(to_cea608("Łódź Ść") == "Lódz Sc");  // Polish
	CHECK(to_cea608("şiğır") == "sigir");                     // Turkish
	CHECK(to_cea608("København æ") == "København ae");        // Danish
	double lost = -1;
	CHECK(to_cea608("hello", &lost) == "hello" && lost == 0.0);
	to_cea608("Привет мир", &lost);  // Russian
	CHECK(lost == 1.0);
	to_cea608("你好", &lost);  // Chinese
	CHECK(lost == 1.0);
	to_cea608("ok 😀", &lost);  // one emoji among three characters
	CHECK(lost > 0.3 && lost < 0.4);
	CHECK(utf8_count("héllo 😀") == 7);

	// A long caption is delivered whole and in order (it used to lose its beginning).
	std::mutex m;
	std::vector<std::string> sent;
	CaptionPacer::Options o;
	o.min_gap_seconds = 0.01;
	o.chars_per_second = 5000;
	std::string long_text;
	for (int i = 0; i < 40; ++i)
		long_text += "word" + std::to_string(100 + i) + " ";  // 8 lines of 32 characters or so
	const size_t want_lines = split_caption_lines(long_text, 32).size();
	CHECK(want_lines >= 8 && want_lines <= o.max_queue_lines);
	{
		CaptionPacer p(
			[&](const std::string &line, double) {
				std::lock_guard<std::mutex> lk(m);
				sent.push_back(line);
			},
			o);
		p.push(long_text);
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		p.stop();
		CHECK(p.dropped_lines() == 0);
	}
	{
		std::lock_guard<std::mutex> lk(m);
		CHECK(sent.size() == want_lines);
		CHECK(!sent.empty() && sent.front().rfind("word100", 0) == 0);
		std::string joined;
		for (auto &l : sent)
			joined += (joined.empty() ? "" : " ") + l;
		CHECK(joined + " " == long_text);
	}
	// A new caption makes room by dropping older captions' lines, never its own.
	sent.clear();
	o.max_queue_lines = 4;
	o.min_gap_seconds = 0.2;
	o.chars_per_second = 1000;
	{
		CaptionPacer p(
			[&](const std::string &line, double) {
				std::lock_guard<std::mutex> lk(m);
				sent.push_back(line);
			},
			o);
		p.push("old one");
		p.push("old two");
		p.push("old three");
		p.push("old four");
		p.push("fresh one two three four five six seven eight nine ten eleven twelve");  // 3 lines
		std::this_thread::sleep_for(std::chrono::milliseconds(1500));
		p.stop();
	}
	{
		std::lock_guard<std::mutex> lk(m);
		int fresh = 0;
		for (auto &l : sent)
			fresh += l.find("one two") != std::string::npos || l.find("ten") != std::string::npos ||
				 l.find("three four") != std::string::npos;
		CHECK(fresh >= 2);
		std::string all;
		for (auto &l : sent)
			all += l + " ";
		CHECK(all.find("twelve") != std::string::npos);
	}
	// Captions the channel cannot show at all are dropped, not sent as gibberish.
	sent.clear();
	{
		CaptionPacer p(
			[&](const std::string &line, double) {
				std::lock_guard<std::mutex> lk(m);
				sent.push_back(line);
			});
		p.push(u8"\u041f\u0440\u0438\u0432\u0435\u0442");
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		CHECK(p.dropped_lines() == 1 && p.sent_lines() == 0);
		p.stop();
	}
	CHECK(sent.empty());
}

static void test_resampler()
{
	std::printf("resampler\n");
	const double kPi = 3.14159265358979323846;
	// 44.1 -> 48 kHz: tone amplitude, frequency and exact sample counts for irregular chunks.
	RationalResampler up(44100, 48000), down(48000, 44100);
	std::mt19937 rng(1);
	std::vector<float> in, mid, out;
	const size_t total = 44100 * 2;
	for (size_t i = 0; i < total; ++i)
		in.push_back(0.5f * static_cast<float>(std::sin(2 * kPi * 1000.0 * static_cast<double>(i) / 44100.0)));
	size_t pos = 0;
	std::vector<float> tmp(8192);
	while (pos < total) {
		const size_t n = std::min<size_t>(total - pos, 100 + rng() % 1500);
		const size_t m = up.process(&in[pos], n, tmp.data());
		mid.insert(mid.end(), tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(m));
		pos += n;
	}
	CHECK(std::llabs(static_cast<long long>(mid.size()) - static_cast<long long>(total * 160 / 147)) <= 2);
	// Steady-state amplitude of the 48 kHz stream
	double peak = 0;
	for (size_t i = mid.size() / 2; i < mid.size() / 2 + 4800; ++i)
		peak = std::max<double>(peak, std::fabs(mid[i]));
	CHECK_NEAR(peak, 0.5, 0.005);
	// Frequency/delay: sample at index i should be sin(2 pi 1000 (i/48000 - delay))
	const double delay = up.group_delay_seconds();
	double max_err = 0;
	for (size_t i = mid.size() / 2; i < mid.size() / 2 + 4800; ++i) {
		const double ref = 0.5 * std::sin(2 * kPi * 1000.0 * (static_cast<double>(i) / 48000.0 - delay));
		max_err = std::max(max_err, std::fabs(ref - mid[i]));
	}
	CHECK_NEAR(max_err, 0.0, 0.01);
	CHECK(delay > 0.0002 && delay < 0.0006);

	// Down-conversion driven by output count: exactly n outputs every time.
	pos = 0;
	size_t produced = 0;
	bool exact = true;
	std::vector<float> o(4096);
	while (pos + 4000 < mid.size()) {
		const size_t want = 100 + rng() % 1000;
		const size_t need = down.input_needed(want);
		if (pos + need > mid.size())
			break;
		const size_t got = down.process(&mid[pos], need, o.data());
		if (got != want)
			exact = false;
		produced += got;
		out.insert(out.end(), o.begin(), o.begin() + static_cast<std::ptrdiff_t>(got));
		pos += need;
	}
	CHECK(exact);
	CHECK(produced > 40000);
	// Round trip (44.1 -> 48 -> 44.1): the tone comes back with the combined delay.
	const double total_delay = up.group_delay_seconds() + down.group_delay_seconds();
	double err2 = 0;
	for (size_t i = out.size() / 2; i < out.size() / 2 + 4410; ++i) {
		const double ref = 0.5 * std::sin(2 * kPi * 1000.0 * (static_cast<double>(i) / 44100.0 - total_delay));
		err2 = std::max(err2, std::fabs(ref - out[i]));
	}
	CHECK_NEAR(err2, 0.0, 0.01);

	// Passband flatness and image rejection at 44.1 -> 48.
	auto gain_at = [&](double f) {
		RationalResampler r(44100, 48000);
		std::vector<float> x(44100), y(60000);
		for (size_t i = 0; i < x.size(); ++i)
			x[i] = static_cast<float>(std::sin(2 * kPi * f * static_cast<double>(i) / 44100.0));
		const size_t m = r.process(x.data(), x.size(), y.data());
		double e = 0;
		for (size_t i = m / 2; i < m / 2 + 4800 && i < m; ++i)
			e += static_cast<double>(y[i]) * y[i];
		return std::sqrt(2.0 * e / 4800.0);  // sine amplitude from the RMS (independent of the sampling phase)
	};
	CHECK_NEAR(gain_at(100), 1.0, 0.01);
	CHECK_NEAR(gain_at(8000), 1.0, 0.01);
	CHECK_NEAR(gain_at(16000), 1.0, 0.02);
	// A 48 kHz tone above the 44.1 kHz Nyquist must be removed by the down-converter.
	{
		RationalResampler r(48000, 44100);
		std::vector<float> x(48000), y(60000);
		for (size_t i = 0; i < x.size(); ++i)
			x[i] = static_cast<float>(std::sin(2 * kPi * 23000.0 * static_cast<double>(i) / 48000.0));
		const size_t m = r.process(x.data(), x.size(), y.data());
		double p = 0;
		for (size_t i = m / 2; i < m / 2 + 4410 && i < m; ++i)
			p = std::max<double>(p, std::fabs(y[i]));
		CHECK(p < 0.01);
	}
}

// Simulates the network: frame k's reply P[k*960+j] = 0.5 * x[k*960 + j - L] arrives
// `rtt(k)` input samples after the frame was complete (or never).
static void test_output_stage()
{
	std::printf("output stage (fixed delay, deadlines, cross-fades)\n");
	constexpr int L = 1440;
	constexpr int D = 150 * 48;
	constexpr int total = 48000 * 6;
	const double kPi = 3.14159265358979323846;
	std::vector<float> x(total);
	for (int i = 0; i < total; ++i)
		x[static_cast<size_t>(i)] = 0.5f * static_cast<float>(std::sin(2 * kPi * 200.0 * i / 48000.0));
	auto xv = [&](long i) { return (i >= 0 && i < total) ? x[static_cast<size_t>(i)] : 0.0f; };

	const std::set<int> late{20, 21, 50};   // arrive after their deadline
	const std::set<int> lost{80};           // never arrive
	const int unsent_until = 3;             // frames 0..2 were not sent (connection still being made)
	auto rtt = [&](int k) { return late.count(k) ? 6200 : 1500; };  // input samples

	ProcessedStore store;
	Counters counters;
	OutputStage stage(D, L, store, counters);
	std::vector<float> out(total + 4800, 0.0f);

	const int chunk = 480;
	std::set<int> delivered;
	const int frames = total / 960;
	for (long pos = 0; pos + chunk <= total; pos += chunk) {
		// frames whose reply arrives before this call
		for (int k = 0; k < frames; ++k) {
			if (delivered.count(k) || k < unsent_until)
				continue;
			if (lost.count(k)) {
				if ((k + 1) * 960L <= pos)
					store.mark_sent(static_cast<uint64_t>(k));
				continue;
			}
			const long sent_at = (k + 1) * 960L;
			if (sent_at <= pos)
				store.mark_sent(static_cast<uint64_t>(k));
			if (sent_at + rtt(k) <= pos) {
				std::vector<float> p(960);
				for (int j = 0; j < 960; ++j)
					p[static_cast<size_t>(j)] = 0.5f * xv(k * 960L + j - L);
				store.publish(static_cast<uint64_t>(k), p.data());
				delivered.insert(k);
			}
		}
		stage.push_input(&x[static_cast<size_t>(pos)], chunk);
		stage.render(&out[static_cast<size_t>(pos)], chunk);
	}

	// Output sample o is the (maybe processed) input delayed by D.
	const long off = D - L;
	int interior_proc = 0, interior_orig = 0, bad = 0;
	double max_step = 0;
	for (long o = 0; o + chunk <= total; ++o) {
		const long rel = o - off;
		const float orig = xv(o - D);
		const float v = out[static_cast<size_t>(o)];
		if (o > 0)
			max_step = std::max<double>(max_step, std::fabs(v - out[static_cast<size_t>(o - 1)]));
		if (rel < 0) {
			if (v != orig)
				++bad;
			continue;
		}
		const long b = rel / 960, pos = rel % 960;
		const bool proc_block = b >= unsent_until && !late.count(static_cast<int>(b)) && !lost.count(static_cast<int>(b));
		const bool in_fade = pos < 240 || pos >= 960 - 240;
		if (!in_fade && b < frames - 8) {
			const float expect = proc_block ? 0.5f * orig : orig;
			if (std::fabs(v - expect) > 1e-6)
				++bad;
			proc_block ? ++interior_proc : ++interior_orig;
		} else {
			// inside a cross-fade: a convex combination of the two sources
			const float lo = std::min(orig, 0.5f * orig) - 1e-6f, hi = std::max(orig, 0.5f * orig) + 1e-6f;
			if (b < frames - 8 && (v < lo || v > hi))
				++bad;
		}
	}
	CHECK(bad == 0);
	CHECK(interior_proc > 100000);
	CHECK(interior_orig > 3000);
	// A hard switch between the sources would jump by up to 0.25; the fades keep steps small.
	CHECK(max_step < 0.03);
	CHECK(counters.late.load() == 4);  // 20, 21, 50 late + 80 lost
	CHECK(counters.passthrough.load() >= 3 && counters.passthrough.load() <= 6);
	CHECK(counters.used.load() > 100);
	std::printf("  used %llu late %llu passthrough %llu max_step %.4f\n",
		    static_cast<unsigned long long>(counters.used.load()), static_cast<unsigned long long>(counters.late.load()),
		    static_cast<unsigned long long>(counters.passthrough.load()), max_step);
}

// Session without a network: the audio must come out unchanged and delayed by
// exactly delay_seconds(), for 48 kHz and 44.1 kHz hosts and irregular chunk sizes.
static double estimate_lag(const std::vector<float> &in, const std::vector<float> &out, int lo, int hi)
{
	auto corr = [&](int lag) {
		double s = 0;
		const int n = static_cast<int>(in.size());
		for (int i = 20000; i < n - 20000; ++i)
			s += static_cast<double>(in[static_cast<size_t>(i)]) * out[static_cast<size_t>(i + lag)];
		return s;
	};
	int best = lo;
	double bv = -1e30;
	for (int l = lo; l <= hi; ++l) {
		const double c = corr(l);
		if (c > bv) {
			bv = c;
			best = l;
		}
	}
	const double a = corr(best - 1), b = bv, c = corr(best + 1);
	const double den = a - 2 * b + c;
	return best + (den != 0 ? 0.5 * (a - c) / den : 0.0);
}

static void test_session_passthrough(int rate, int delay_ms)
{
	std::printf("session fallback path @ %d Hz, D = %d ms\n", rate, delay_ms);
	SessionConfig cfg;
	cfg.host_rate = rate;
	cfg.delay_ms = delay_ms;
	cfg.api_key = "acw_unit_test";
	Session s(cfg, nullptr, nullptr);  // never started: no network
	// Band-limited noise so that the cross-correlation peak is unique.
	std::mt19937 rng(5);
	std::normal_distribution<float> nd(0.0f, 0.1f);
	const int n = rate * 3;
	std::vector<float> in(static_cast<size_t>(n)), out;
	float lp = 0;
	for (auto &v : in) {
		lp += 0.4f * (nd(rng) - lp);
		v = lp;
	}
	out = in;
	size_t pos = 0;
	const size_t sizes[] = {441, 1024, 480, 223, 960, 512, 1000};
	size_t k = 0;
	while (pos < out.size()) {
		const size_t m = std::min(sizes[k++ % 7], out.size() - pos);
		s.process(&out[pos], m);
		pos += m;
	}
	const double lag = estimate_lag(in, out, 0, static_cast<int>(rate * 0.4));
	const double expect = s.delay_seconds() * rate;
	std::printf("  measured delay %.2f samples, advertised %.2f samples (%.2f ms)\n", lag, expect, expect * 1000 / rate);
	CHECK_NEAR(lag, expect, 0.6);
	CHECK(expect > rate * delay_ms / 1000.0 - 1 && expect < rate * delay_ms / 1000.0 + 0.0015 * rate);
	// level preserved
	double e_in = 0, e_out = 0;
	for (int i = 30000; i < n - 30000; ++i) {
		e_in += static_cast<double>(in[static_cast<size_t>(i)]) * in[static_cast<size_t>(i)];
		e_out += static_cast<double>(out[static_cast<size_t>(i + static_cast<int>(std::lround(expect)))]) *
			 out[static_cast<size_t>(i + static_cast<int>(std::lround(expect)))];
	}
	CHECK_NEAR(std::sqrt(e_out / e_in), 1.0, 0.03);
	const SessionSnapshot snap = s.snapshot();
	CHECK(snap.used == 0 && snap.late == 0 && snap.passthrough > 10);
	CHECK(snap.state == SessionState::Idle);
}

static void test_reference_buffer()
{
	std::printf("echo reference buffer\n");
	// A counter pushed in irregular chunks comes out as the same counter, whole blocks, no holes.
	ReferenceBuffer rb(48000);
	std::vector<float> block(960);
	CHECK(!rb.pull(block.data(), 960));  // nothing yet: silence
	for (float v : block)
		CHECK(v == 0.0f);
	float next = 1.0f;
	const size_t chunks[] = {480, 441, 519, 300, 660, 1};  // 10 ms-ish ticks; two ticks make one 20 ms block
	float last = 0;
	int blocks = 0;
	bool contiguous = true;
	for (int tick = 0; tick < 400; ++tick) {
		std::vector<float> x(chunks[tick % 6]);
		for (float &v : x)
			v = next++;
		rb.push(x.data(), x.size());
		if (tick % 2 == 0 || !rb.pull(block.data(), 960))
			continue;
		if (last != 0 && block[0] != last + 1)
			contiguous = false;
		for (size_t i = 1; i < block.size(); ++i)
			contiguous = contiguous && block[i] == block[i - 1] + 1;
		last = block.back();
		++blocks;
	}
	CHECK(contiguous && blocks > 100);
	// Far behind (a long pause on the reading side): it jumps to the newest audio, a little behind it.
	std::vector<float> big(48000, 0.5f);
	for (size_t i = 0; i < big.size(); ++i)
		big[i] = next++;
	rb.push(big.data(), big.size());
	CHECK(rb.pull(block.data(), 960));
	CHECK(block[0] > next - 3000 && block[959] < next - 480);
	// Silence is reported as silence, other sample rates are converted to 48 kHz.
	ReferenceBuffer quiet(48000);
	std::vector<float> zeros(4800, 0.0f);
	quiet.push(zeros.data(), zeros.size());
	CHECK(!quiet.pull(block.data(), 960));
	ReferenceBuffer rb44(44100);
	std::vector<float> tone(44100);
	for (size_t i = 0; i < tone.size(); ++i)
		tone[i] = 0.5f * std::sin(0.05f * static_cast<float>(i));
	rb44.push(tone.data(), tone.size());
	int got = 0;
	double e = 0;
	for (int tick = 0; tick < 200; ++tick) {  // a second of 10 ms ticks, a block every other tick
		rb44.push(tone.data() + (tick % 100) * 441, 441);
		if (tick % 2 == 1 && rb44.pull(block.data(), 960) && tick > 20) {
			++got;
			for (float v : block)
				e += static_cast<double>(v) * v;
		}
	}
	CHECK(got >= 85 && got <= 90);  // two seconds, minus the first 0.4
	CHECK_NEAR(std::sqrt(e / (got * 960.0)), 0.5 / std::sqrt(2.0), 0.03);
}

static void test_session_texts()
{
	std::printf("session texts and address rules\n");
	CHECK(server_url_allowed("wss://mc-multi-world.space/audio/api/stream/ws"));
	CHECK(server_url_allowed("ws://127.0.0.1:8791/api/stream/ws"));
	CHECK(server_url_allowed("ws://localhost/x"));
	CHECK(server_url_allowed("ws://[::1]:9/x"));
	CHECK(server_url_allowed("ws://192.168.1.20/x") && server_url_allowed("ws://10.1.2.3:80/x"));
	CHECK(server_url_allowed("ws://172.16.0.1/x") && server_url_allowed("ws://172.31.9.9/x"));
	CHECK(server_url_allowed("ws://studio.local/x"));
	CHECK(!server_url_allowed("ws://172.32.0.1/x") && !server_url_allowed("ws://172.15.0.1/x"));
	CHECK(!server_url_allowed("ws://example.com/x") && !server_url_allowed("ws://8.8.8.8/x"));
	CHECK(!server_url_allowed("ws://127.0.0.1.evil.example/x"));
	CHECK(!server_url_allowed("ws://localhost@evil.example/x"));  // user info trick
	CHECK(!server_url_allowed("http://127.0.0.1/x") && !server_url_allowed(""));
	CHECK(std::string(latency_preset_name(80)) == "Low" && std::string(latency_preset_name(150)) == "Balanced" &&
	      std::string(latency_preset_name(250)) == "Safe");

	SessionConfig cfg;
	cfg.api_key = "x";
	cfg.url = "ws://example.com/x";
	Session refused(cfg, nullptr, nullptr);
	refused.start();
	CHECK(refused.snapshot().state == SessionState::Stopped);
	CHECK(refused.status_line().find("local network") != std::string::npos);

	cfg.url = "ws://127.0.0.1:1/x";
	Session idle(cfg, nullptr, nullptr);
	CHECK(idle.status_line().rfind("Waiting for audio from OBS.", 0) == 0);
	CHECK(idle.status_line().find("RTT") == std::string::npos);
	CHECK(idle.status_details().find("idle") != std::string::npos);
	// Nothing to retry unless blocked.
	idle.retry_now();
	CHECK(idle.snapshot().state == SessionState::Idle);
}

static void test_session_config()
{
	std::printf("session configuration\n");
	// captions only: no delay, audio untouched
	SessionConfig cfg;
	cfg.denoise = false;
	cfg.captions = "local";
	cfg.api_key = "x";
	Session s(cfg, nullptr, nullptr);
	std::vector<float> a(1000, 0.25f);
	s.process(a.data(), a.size());
	CHECK(s.delay_seconds() == 0.0);
	for (float v : a)
		CHECK(v == 0.25f);
	// unsupported host rate: audio untouched, flagged
	SessionConfig bad;
	bad.host_rate = 96000;
	bad.api_key = "x";
	Session s2(bad, nullptr, nullptr);
	std::vector<float> b(1000, 0.5f);
	s2.process(b.data(), b.size());
	CHECK(s2.delay_seconds() == 0.0 && b[10] == 0.5f);
	CHECK(s2.snapshot().passthrough_only);
	// bad URL / missing key: refuses to start
	SessionConfig nk;
	nk.api_key = "";
	Session s3(nk, nullptr, nullptr);
	s3.start();
	CHECK(s3.snapshot().state == SessionState::Stopped);
	SessionConfig nu;
	nu.api_key = "k";
	nu.url = "http://example.com/x";
	Session s4(nu, nullptr, nullptr);
	s4.start();
	CHECK(s4.snapshot().state == SessionState::Stopped);
	CHECK(s4.status_line().find("wss://") != std::string::npos);
	// the API key is never part of the status text
	SessionConfig sk;
	sk.api_key = "acw_SECRET_KEY_VALUE";
	sk.url = "ws://127.0.0.1:1/x";
	Session s5(sk, nullptr, nullptr);
	CHECK(s5.status_line().find("SECRET") == std::string::npos);
}

// ---------------------------------------------------------------------------------------------------
// update notice: version numbers and the version file

static void test_update_versions()
{
	std::printf("update check: versions\n");
	using namespace acw;
	Version v;
	CHECK(parse_version("0.1.57", v) && v.part[0] == 0 && v.part[1] == 1 && v.part[2] == 57 && v.part[3] == 0);
	CHECK(parse_version("12.345678.9", v) && v.part[1] == 345678);
	CHECK(parse_version("2", v) && v.part[0] == 2);
	CHECK(parse_version("1.2.3.4", v) && v.part[3] == 4);
	const char *bad[] = {"",      " ",       "v1.2.3", "1.2.3-beta", "1..3",   ".1",      "1.",      "1.2.3.4.5", "1.2.x",
			     "1,2,3", "1.2.3 ",  " 1.2.3", "1.2.3\n",    "-1.2.3", "+1.2.3",  "1234567.0"};
	for (const char *b : bad) {
		Version keep;
		keep.part[0] = 99;
		Version probe = keep;
		CHECK(!parse_version(b, probe));
		CHECK(probe.part[0] == 99);  // untouched on failure
	}
	CHECK(!parse_version(std::string("1.2\0" "3", 4), v));

	// Numeric, not alphabetic: 0.1.9 < 0.1.57 < 0.1.100 < 0.2.0 < 1.0.0
	const char *order[] = {"0.1.0", "0.1.9", "0.1.57", "0.1.100", "0.2.0", "1.0.0", "1.0.0.1", "1.0.1", "10.0.0"};
	const int n = static_cast<int>(sizeof(order) / sizeof(order[0]));
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j) {
			Version a, b;
			CHECK(parse_version(order[i], a) && parse_version(order[j], b));
			const int c = compare_versions(a, b);
			CHECK((c < 0) == (i < j) && (c == 0) == (i == j));
			CHECK(is_newer_version(order[i], order[j]) == (i > j));
		}
	CHECK(is_newer_version("1.0", "0.9.9"));
	{
		Version a, b;
		CHECK(parse_version("1.0", a) && parse_version("1.0.0.0", b) && compare_versions(a, b) == 0);
	}
	CHECK(!is_newer_version("garbage", "0.1.0"));
	CHECK(!is_newer_version("0.2.0", "garbage"));
	CHECK(!is_newer_version("0.1.0", "0.1.0"));
}

static std::string entry_json(const std::string &name, const std::string &size, const std::string &sha512 = std::string(128, 'b'))
{
	return "{\"name\":\"" + name + "\",\"size\":" + size + ",\"sha256\":\"" + std::string(64, 'a') + "\",\"sha512\":\"" + sha512 + "\"}";
}

static void test_update_manifest()
{
	std::printf("update check: manifest\n");
	using namespace acw;
	UpdateManifest m;

	const std::string win = entry_json("audio-cleaner-obs-plugin-0.1.60-windows.zip", "3000000");
	const std::string mac = entry_json("audio-cleaner-obs-plugin-0.1.60-mac.tar.xz", "4000000");
	const std::string deb = entry_json("audio-cleaner-obs-plugin-0.1.60-ubuntu-debian.deb", "2000000");
	const std::string oth = entry_json("audio-cleaner-obs-plugin-0.1.60-linux-other.tar.xz", "2000000");
	const std::string full = "{\"files\":{\"linux-other\":" + oth + ",\"macos\":" + mac + ",\"ubuntu-debian\":" + deb +
				 ",\"windows-x64\":" + win + "},\"published\":\"2026-10-16T10:00:00Z\",\"version\":\"0.1.60\"}";
	CHECK(parse_update_manifest(full, m));
	CHECK(m.version == "0.1.60" && m.files.size() == 4);
	CHECK(m.files.at(kKeyWindows).name == "audio-cleaner-obs-plugin-0.1.60-windows.zip");
	CHECK(m.files.at(kKeyWindows).size == 3000000);
	CHECK(m.files.at(kKeyWindows).sha512 == std::string(128, 'b'));
	CHECK(m.files.at(kKeyWindows).sha256 == std::string(64, 'a'));
	CHECK(m.files.at(kKeyMac).name.size() > 0 && m.files.at(kKeyDebian).name.size() > 0 && m.files.at(kKeyLinuxOther).name.size() > 0);

	// Whitespace and key order do not matter; fields we do not know are ignored; the sha256 is optional.
	CHECK(parse_update_manifest("{\"version\": \"1.2.3\", \"x\": [1], \"files\": {\"macos\": {\"sha512\": \"" + std::string(128, 'c') +
					    "\", \"size\": 5, \"name\": \"p.tar.xz\", \"extra\": null}}}",
				    m));
	CHECK(m.version == "1.2.3" && m.files.size() == 1 && m.files.at(kKeyMac).sha256.empty());

	// Entries that cannot be used are left out, the rest of the manifest stays usable.
	const std::string odd = "{\"version\":\"0.1.60\",\"files\":{"
				"\"windows-x64\":" + entry_json("../evil.zip", "5") + ","
				"\"macos\":" + entry_json("a.zip", "5") + ","                      // wrong ending for its kind
				"\"ubuntu-debian\":" + entry_json("a.deb", "0") + ","              // empty
				"\"linux-other\":" + entry_json("a.tar.xz", "67108865") + ","      // over 64 MiB
				"\"beos\":" + entry_json("a.zip", "5") + "}}";                     // a kind we do not know
	CHECK(parse_update_manifest(odd, m) && m.files.empty());
	CHECK(parse_update_manifest("{\"version\":\"0.1.60\",\"files\":{\"macos\":" + entry_json("a.tar.xz", "67108864") + "}}", m) &&
	      m.files.size() == 1);
	CHECK(parse_update_manifest("{\"version\":\"0.1.60\",\"files\":{\"macos\":" + entry_json("a.tar.xz", "5", std::string(127, 'b')) + "}}", m) &&
	      m.files.empty());  // a short digest
	CHECK(parse_update_manifest("{\"version\":\"0.1.60\",\"files\":{\"macos\":" + entry_json("a.tar.xz", "5", std::string(128, 'B')) + "}}", m) &&
	      m.files.empty());  // upper case: the CI writes lower case
	CHECK(parse_update_manifest("{\"version\":\"0.1.60\",\"files\":{\"macos\":" + entry_json("a.tar.xz", "-5") + "}}", m) && m.files.empty());
	CHECK(parse_update_manifest("{\"version\":\"0.1.60\",\"files\":{\"macos\":" + entry_json("a.tar.xz", "\"5\"") + "}}", m) && m.files.empty());
	CHECK(parse_update_manifest("{\"version\":\"0.1.60\",\"files\":{\"macos\":" + entry_json("a.tar.xz", "5.5") + "}}", m) && m.files.empty());
	CHECK(parse_update_manifest("{\"version\":\"0.1.60\",\"files\":{\"macos\":[1]}}", m) && m.files.empty());
	CHECK(parse_update_manifest("{\"version\":\"0.1.60\",\"files\":{}}", m) && m.files.empty());

	// Not a manifest.
	const char *bad[] = {"",
			     "null",
			     "[]",
			     "{}",
			     "{\"version\":\"0.1.60\"}",                // no files
			     "{\"files\":{}}",                          // no version
			     "{\"version\":\"0.1.60\",\"files\":[]}",
			     "{\"version\":\"0.1.60\",\"files\":null}",
			     "{\"version\":1,\"files\":{}}",
			     "{\"version\":\"v1.2.3\",\"files\":{}}",
			     "{\"version\":\"1.2.3-rc1\",\"files\":{}}",
			     "{\"version\":\"\",\"files\":{}}",
			     "{\"Version\":\"1.2.3\",\"files\":{}}",
			     "<html><body>404</body></html>",
			     "{\"version\":\"1.2.3\",\"files\":{}} trailing",
			     "{\"version\":\"1.2.3\",\"files\":{}"};
	for (const char *b : bad) {
		UpdateManifest keep;
		keep.version = "kept";
		UpdateManifest probe = keep;
		CHECK(!parse_update_manifest(b, probe));
		CHECK(probe.version == "kept");
	}

	// File names: a plain name and nothing else.
	CHECK(valid_update_file_name("audio-cleaner-obs-plugin-0.1.60-windows.zip"));
	CHECK(valid_update_file_name("a"));
	CHECK(valid_update_file_name("A_b-c.1.tar.xz"));
	CHECK(valid_update_file_name(std::string(120, 'a')));
	for (const char *n : {"", ".hidden", "-x", "_x", "../x.zip", "..", ".", "a/b.zip", "a\\b.zip", "/etc/passwd", "C:x.zip", "a b.zip",
			      "a..b.zip", "x.zip\n", "x%2e.zip", "x?.zip", "x#.zip", "x:y.zip", "x;y.zip", "x&y.zip", "x$y.zip", "x`y.zip", "x\"y.zip",
			      "x'y.zip", "x|y.zip", "x*y.zip", "ü.zip", "x\xC3\xBC.zip"})
		CHECK(!valid_update_file_name(n));
	CHECK(!valid_update_file_name(std::string(121, 'a')));
	CHECK(!valid_update_file_name(std::string("a\0b", 3)));
	CHECK(std::string(update_file_suffix(kKeyWindows)) == ".zip");
	CHECK(std::string(update_file_suffix(kKeyMac)) == ".tar.xz");
	CHECK(std::string(update_file_suffix(kKeyDebian)) == ".deb");
	CHECK(std::string(update_file_suffix(kKeyLinuxOther)) == ".tar.xz");
	CHECK(std::string(update_file_suffix("nope")).empty());
}

static void test_update_crypto()
{
	std::printf("update check: base64, SHA-512, Ed25519\n");
	using namespace acw;

	// base64
	std::vector<uint8_t> b;
	CHECK(base64_decode("", b) && b.empty());
	CHECK(base64_decode("Zg==", b) && b.size() == 1 && b[0] == 'f');
	CHECK(base64_decode("Zm8=", b) && b.size() == 2);
	CHECK(base64_decode("Zm9v", b) && b.size() == 3);
	CHECK(base64_decode("Zm9vYg==", b) && b.size() == 4);
	CHECK(base64_decode("+/+/", b) && b.size() == 3 && b[0] == 0xfb && b[1] == 0xff && b[2] == 0xbf);
	for (const char *bad : {"Zg=", "Zg", "Zg=a", "=Zg=", "Z===", "Zg==Zg==", "Zm9v ", " Zm9v", "Zm9v\n", "Zm-_", "Zh==", "Zm9=", "Zm9\x01"})
		CHECK(!base64_decode(bad, b));
	for (size_t n = 0; n < 40; ++n) {
		std::vector<uint8_t> v(n);
		for (size_t i = 0; i < n; ++i)
			v[i] = static_cast<uint8_t>(i * 37 + n);
		const std::string e = base64_encode(v.data(), v.size());
		std::vector<uint8_t> back;
		CHECK(base64_decode(e, back) && back == v);
	}
	CHECK(is_hex_digest(std::string(128, '0'), 128) && !is_hex_digest(std::string(127, '0'), 128) && !is_hex_digest(std::string(128, 'A'), 128));
	CHECK(!is_hex_digest(std::string(127, '0') + "g", 128));

	// SHA-512 (FIPS 180 vectors)
	CHECK(sha512_hex("") ==
	      "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
	CHECK(sha512_hex("abc") ==
	      "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
	CHECK(sha512_hex("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu") ==
	      "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909");
	{
		// in pieces of every size = in one go
		std::string data(1000, 'x');
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = static_cast<char>(i * 7);
		const std::string whole = sha512_hex(data);
		for (size_t piece : {1u, 3u, 127u, 128u, 129u, 500u}) {
			Sha512 h;
			for (size_t at = 0; at < data.size(); at += piece)
				h.update(data.data() + at, std::min(piece, data.size() - at));
			CHECK(h.finish_hex() == whole);
		}
	}

	// Ed25519: a signature made by `openssl pkeyutl -sign -rawin` (what the CI does) over the exact bytes.
	const std::string msg = "{\"files\":{},\"version\":\"0.1.60\"}";
	const char *pub = "FDHGbeQu9FAd51SLwN5L6xua+/Vw9RJCnxbIItVoclA=";
	const char sig_hex[] =
		"e5404e1119523eb0186a1f88d4b7660cc87e67765494a8a151c945e6999dd955b7c951cdd329f08f9390555862d4070ebca8201b6c7a22daaf794307a588f10f";
	std::string sig;
	for (size_t i = 0; i < 128; i += 2)
		sig += static_cast<char>(std::stoi(std::string(sig_hex + i, 2), nullptr, 16));
	CHECK(sig.size() == 64);
	CHECK(verify_ed25519(msg, sig, pub));
	// tampered message, tampered signature (every byte position), another key, malformed inputs
	CHECK(!verify_ed25519(msg + " ", sig, pub));
	CHECK(!verify_ed25519("{\"files\":{},\"version\":\"0.1.61\"}", sig, pub));
	CHECK(!verify_ed25519("", sig, pub));
	for (size_t i = 0; i < 64; ++i) {
		std::string t = sig;
		t[i] = static_cast<char>(t[i] ^ 0x01);
		CHECK(!verify_ed25519(msg, t, pub));
	}
	CHECK(!verify_ed25519(msg, sig, kUpdatePublicKeyBase64));  // the real key did not sign this
	CHECK(!verify_ed25519(msg, sig.substr(0, 63), pub));
	CHECK(!verify_ed25519(msg, sig + "x", pub));
	CHECK(!verify_ed25519(msg, "", pub));
	CHECK(!verify_ed25519(msg, sig, ""));
	CHECK(!verify_ed25519(msg, sig, "FDHGbeQu9FAd51SLwN5L6xua+/Vw9RJCnxbIItVoclA"));       // no padding
	CHECK(!verify_ed25519(msg, sig, "FDHGbeQu9FAd51SLwN5L6xua+/Vw9RJCnxbIItVo"));          // too short
	CHECK(!verify_ed25519(msg, sig, "!DHGbeQu9FAd51SLwN5L6xua+/Vw9RJCnxbIItVoclA="));      // not base64
	CHECK(!verify_ed25519(msg, std::string(64, '\0'), pub));
	CHECK(base64_decode(kUpdatePublicKeyBase64, b) && b.size() == 32);  // the embedded key is well-formed

	// ... and our own signer agrees with itself (the loopback tests use it)
	const updtest::KeyPair k = updtest::make_key(1), other = updtest::make_key(2);
	CHECK(verify_ed25519(msg, updtest::sign(k, msg), k.pub_b64()));
	CHECK(!verify_ed25519(msg, updtest::sign(k, msg), other.pub_b64()));
	CHECK(!verify_ed25519(msg + "x", updtest::sign(k, msg), k.pub_b64()));
}

int main()
{
	test_spsc();
	test_protocol();
	test_opus();
	test_backoff();
	test_caption_lines();
	test_caption_pacing();
	test_resampler();
	test_output_stage();
	test_session_passthrough(48000, 150);
	test_session_passthrough(44100, 150);
	test_session_passthrough(44100, 80);
	test_reference_buffer();
	test_session_config();
	test_session_texts();
	test_update_versions();
	test_update_manifest();
	test_update_crypto();
	std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
	return g_failures ? 1 : 0;
}
