// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/session.hpp"

#include "acw/backoff.hpp"
#include "acw/languages.hpp"
#include "acw/opus_codec.hpp"
#include "acw/output_stage.hpp"
#include "acw/protocol.hpp"
#include "acw/resampler.hpp"
#include "acw/spsc.hpp"
#include "acw/stats.hpp"
#include "tls_options.hpp"

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

namespace acw {

using Clock = std::chrono::steady_clock;

const char *to_string(SessionState s)
{
	switch (s) {
	case SessionState::Idle:
		return "idle";
	case SessionState::Connecting:
		return "connecting";
	case SessionState::Handshaking:
		return "handshaking";
	case SessionState::Streaming:
		return "streaming";
	case SessionState::Backoff:
		return "reconnecting";
	case SessionState::Stopped:
		return "stopped";
	case SessionState::Blocked:
		return "blocked";
	}
	return "?";
}

namespace {

// Microseconds on a process-wide monotonic clock; also the `ts_us` we put in
// frames (opaque to the server, echoed back for RTT measurement).
const Clock::time_point &clock_epoch()
{
	static const Clock::time_point t0 = Clock::now();
	return t0;
}
uint64_t now_us()
{
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - clock_epoch()).count()) + 1;
}

struct Frame {
	uint32_t seq = 0;
	uint64_t ts_us = 0;
	float pcm[kFrameSamples] = {};
	float ref[kFrameSamples] = {};  // what the speakers played meanwhile (echo cancellation)
	bool has_ref = false;           // ... and it is not silence
	bool silent = false;            // the microphone block is silence (muted, or every sample below kSilenceLevel)
};

constexpr size_t kMaxBlock = 2048;           // host-rate samples handled per inner step
constexpr size_t kQueueFrames = 32;          // 640 ms between the audio thread and the network thread
constexpr size_t kMaxTxBuffered = 40 * 1024; // bytes queued in the socket before we drop frames
constexpr int kPingIntervalMs = 5000;
constexpr uint64_t kFrameWakeSlackUs = 500;  // the network thread wakes this long after a frame is due: the audio thread has stored it by then
constexpr uint64_t kStaleAudioUs = 40000;    // a predicted frame this much overdue: the audio is not flowing (any more)
constexpr int kRxWatchdogMs = 20000;
constexpr int kIdleDisconnectMs = 10000;     // no audio from the host for this long: close the session
constexpr size_t kRttRing = 8192;
constexpr int kLateWindowS = 6;          // seconds over which late frames are counted
constexpr uint64_t kLateMinFrames = 100;  // ... needs at least this many frames (2 s) before judging
constexpr double kLateSetPct = 30.0;     // more than this many percent late: the link is too slow
constexpr double kLateClearPct = 15.0;   // ... until it is below this again
constexpr int kFramingMs = 25;           // 20 ms frame + 5 ms cross-fade: what the delay budget loses to framing
constexpr size_t kMaxCaptionBytes = 4000;  // a translation longer than this is cut (the service sends sentences)
constexpr int kOpusMinBitrate = 24000;   // the lowest the server accepts: enough when only captions are wanted

bool is_secure_url(const std::string &u)
{
	return u.rfind("wss://", 0) == 0;
}
bool is_plain_url(const std::string &u)
{
	return u.rfind("ws://", 0) == 0;
}

// Host part of a ws:// or wss:// URL (without port, brackets kept for IPv6).
std::string url_host(const std::string &u)
{
	const size_t b = u.find("://");
	if (b == std::string::npos)
		return {};
	size_t i = b + 3;
	const size_t at = u.find('@', i);  // user info
	const size_t slash = u.find_first_of("/?#", i);
	if (at != std::string::npos && (slash == std::string::npos || at < slash))
		i = at + 1;
	size_t e = u.find_first_of("/?#", i);
	std::string hp = u.substr(i, e == std::string::npos ? std::string::npos : e - i);
	if (!hp.empty() && hp[0] == '[') {
		const size_t rb = hp.find(']');
		return rb == std::string::npos ? hp : hp.substr(0, rb + 1);
	}
	const size_t colon = hp.find(':');
	return colon == std::string::npos ? hp : hp.substr(0, colon);
}

// Unencrypted connections are only acceptable to this computer or the local network.
bool is_local_host(std::string h)
{
	for (char &c : h)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	if (h == "localhost" || h == "[::1]")
		return true;
	if (h.size() > 6 && h.compare(h.size() - 6, 6, ".local") == 0)
		return true;
	// Dotted quad: four numbers 0-255 and nothing else.
	int part[4] = {0, 0, 0, 0};
	int idx = 0, digits = 0;
	for (char ch : h) {
		if (ch >= '0' && ch <= '9') {
			part[idx] = part[idx] * 10 + (ch - '0');
			if (++digits > 3 || part[idx] > 255)
				return false;
		} else if (ch == '.' && digits > 0 && idx < 3) {
			++idx;
			digits = 0;
		} else {
			return false;
		}
	}
	if (idx != 3 || digits == 0)
		return false;
	const int a = part[0], b = part[1];
	return a == 127 || a == 10 || (a == 192 && b == 168) || (a == 172 && b >= 16 && b <= 31) || (a == 169 && b == 254);
}

std::atomic<int> g_bg_threads{0};  // network threads that are running (detached ones included)

}  // namespace

// ---------------------------------------------------------------------------

struct Session::Impl : std::enable_shared_from_this<Session::Impl> {
	// Everything the network side learns about one WebSocket attempt.
	struct Attempt {
		std::mutex mu;
		std::condition_variable cv;
		bool opened = false;
		bool closed = false;
		bool ready = false;
		int http_status = 0;
		std::string reason;                // transport-level close/error text
		std::string close_reason;          // the reason text of the server's close frame (it can be a code such as "too_many_messages")
		std::string server_code, server_msg;  // `error` event
		int retry_after_s = 0;             // ... and its optional retry hint
		std::string stopped_reason;        // `stopped` event
		Event ready_ev;
		std::unique_ptr<OpusDec> opus_dec;  // used by the IX thread only
		std::atomic<int64_t> first_valid_seq{INT64_MAX};
		std::atomic<uint64_t> last_rx_us{0};
	};

	struct Outcome {
		std::string reason;      // technical, for the log and the details line
		std::string message;     // plain words for the user
		std::string code;        // the reason as a word ("no_audio", "trial_over", ...), for the plugin's own texts
		std::string arg;         // a detail for those texts (the server's own message)
		bool keep_backoff = false;  // do not count a long session as a success (it ended for a reason that may repeat)
		int min_wait_ms = 0;
		int streamed_ms = 0;
		bool immediate = false;  // reconnect without waiting (server restarted the session)
		bool give_up = false;    // only the user can fix this: stop retrying until retry_now()
		bool slow = false;       // the server is limiting us (busy, too many sessions, rate limit): back off hard
	};

	SessionConfig cfg;
	CaptionCallback on_caption;
	TranslationCallback on_translation;
	LogCallback on_log;
	std::mutex cb_mu;
	bool cb_enabled = true;

	// ---- audio-thread side -------------------------------------------------
	Counters counters;
	ProcessedStore store;
	std::unique_ptr<OutputStage> stage;
	std::unique_ptr<RationalResampler> up, down;
	std::vector<float> in48, out48;
	bool resampled = false;
	bool supported = true;
	bool use_opus = false;
	int opus_extra_latency = 0;  // codec delay of both directions (samples @ 48 kHz), added to the model latency
	double delay_sec = 0.0;
	Frame cur;
	size_t fill = 0;
	float cur_peak = 0.0f;  // loudest sample of the microphone block being assembled
	uint32_t next_seq = 0;
	// What the network thread needs to know when the next frame will be complete, so that it can sleep until then instead
	// of polling: the audio thread only stores these (after each process() call, the last one is the release).
	std::atomic<uint32_t> sched_fill{0};      // samples of the next frame already in hand (48 kHz)
	std::atomic<uint32_t> sched_chunk{0};     // samples (48 kHz) the last process() call delivered: OBS hands them over at this pace
	std::atomic<uint64_t> sched_call_us{0};   // when that call happened (now_us() clock); 0 = none yet
	SpscQueue<Frame> queue{kQueueFrames};
	std::atomic<bool> streaming{false};
	std::atomic<bool> echo_active{false};  // the server accepted the echo reference
	std::atomic<bool> silence_active{false};  // the server takes silence frames
	std::atomic<int> atten_db{100};        // current attenuation limit (UI thread writes, network thread reads)
	std::atomic<int64_t> origin_ticks{0};  // steady-clock time of the audio second 0 of this connection (0 = none yet)
	std::atomic<uint64_t> last_process_us{0};

	// ---- thread control ----------------------------------------------------
	std::thread th;
	std::mutex mu;
	std::condition_variable cv;
	std::atomic<bool> stop_req{false};
	std::atomic<bool> started{false};
	std::atomic<int> flush_ms{0};
	std::mutex done_mu;
	std::condition_variable done_cv;
	bool net_done = false;
	std::atomic<int> pending_atten{-1};
	// A change of what the session does (change()): waiting to be sent, then sent and waiting for the answer.
	std::mutex chg_mu;
	std::optional<Features> pending_change;  // chg_mu
	std::optional<Features> sent_change;     // chg_mu
	std::string sent_change_id;              // chg_mu
	uint32_t change_seq = 0;                 // chg_mu
	std::atomic<bool> retry_req{false};
	std::atomic<bool> link_slow{false};
	std::atomic<int> late_x10{0};  // recent share of late frames, in 0.1 %

	// ---- status ------------------------------------------------------------
	mutable std::mutex st_mu;
	SessionState state = SessionState::Idle;
	std::string detail;
	std::string message;
	std::string session_id;
	double usd_per_minute = 0, model_latency_ms = 0;
	int backoff_s = 0;
	bool low_balance = false;
	int seconds_left = -1;
	std::string warning;
	std::string reason_code, reason_arg;
	std::vector<std::string> translate_active;
	bool translate_active_known = false;
	std::vector<std::string> translate_failed;
	bool captions_failed = false;
	Features feat;               // what the session does now (as the service agreed to it)
	std::string change_refused;  // why the last change could not be made ("" = it was, or none was asked)
	std::vector<std::string> other_warnings;  // texts of warnings that have no plugin text of their own
	double ping_rtt_ms = 0;
	std::vector<float> rtt_ring;
	size_t rtt_n = 0;
	double rtt_avg = 0, rtt_last = 0;
	bool insecure = false;

	std::vector<float> rx;  // IX thread scratch

	// -----------------------------------------------------------------------
	explicit Impl(const SessionConfig &c, CaptionCallback cc, LogCallback lc)
		: cfg(c), on_caption(std::move(cc)), on_log(std::move(lc)), rtt_ring(kRttRing, 0.0f), rx(kMaxRxSamples)
	{
		(void)now_us();  // initialise the clock epoch off the audio thread
		ix::initNetSystem();
		insecure = is_plain_url(cfg.url);

		if (cfg.host_rate < 8000 || cfg.host_rate > kNetRate)
			supported = false;
		cfg.atten_lim_db = std::min(100, std::max(6, cfg.atten_lim_db));
		atten_db.store(cfg.atten_lim_db);

		if (cfg.codec == Codec::Opus) {
			OpusEnc probe(cfg.opus_bitrate);
			if (opus_available() && probe.ok()) {
				use_opus = true;
				// Each direction delays the audio by the encoder look-ahead.
				opus_extra_latency = 2 * probe.lookahead();
			} else {
				cfg.codec = Codec::Pcm16;  // built without libopus: use PCM
			}
		}

		feat.denoise = cfg.denoise;
		feat.captions = cfg.captions;
		feat.language = cfg.language;
		if (!cfg.captions.empty())
			feat.translate = cfg.translate;
		feat.echo_reference = cfg.echo && cfg.reference;
		if (supported && cfg.host_rate != kNetRate) {
			resampled = true;
			up = std::make_unique<RationalResampler>(cfg.host_rate, kNetRate);
			down = std::make_unique<RationalResampler>(kNetRate, cfg.host_rate);
			const size_t cap = up->max_output(kMaxBlock) + 64;
			in48.assign(cap, 0.0f);
			out48.assign(cap + 64, 0.0f);
		}
		if (supported && cfg.denoise) {
			const int d = cfg.delay_ms * (kNetRate / 1000);
			stage = std::make_unique<OutputStage>(d, cfg.model_latency_samples + opus_extra_latency, store, counters);
			delay_sec = static_cast<double>(d) / kNetRate;
			if (resampled)
				delay_sec += up->group_delay_seconds() + down->group_delay_seconds();
		}
		set_state(SessionState::Idle, supported ? "waiting for audio" : "unsupported sample rate: audio passed through",
			  supported ? "Waiting for audio from OBS." : "");
	}

	~Impl() { ix::uninitNetSystem(); }

	// ---- logging / callbacks -------------------------------------------------
	void log(LogLevel lv, const std::string &msg)
	{
		std::lock_guard<std::mutex> lk(cb_mu);
		if (cb_enabled && on_log) {
			try {
				on_log(lv, msg);
			} catch (...) {
			}
		}
	}
	// The callbacks belong to the plugin; whatever they do must never take the network thread (and with it the
	// stream's audio path) down: an exception ends here.
	void emit_caption(const CaptionEvent &ev)
	{
		std::lock_guard<std::mutex> lk(cb_mu);
		if (cb_enabled && on_caption) {
			try {
				on_caption(ev);
			} catch (...) {
			}
		}
	}
	void emit_translation(const TranslationEvent &ev)
	{
		std::lock_guard<std::mutex> lk(cb_mu);
		if (cb_enabled && on_translation) {
			try {
				on_translation(ev);
			} catch (...) {
			}
		}
	}
	Clock::time_point origin() const
	{
		const int64_t t = origin_ticks.load(std::memory_order_acquire);
		return t == 0 ? Clock::time_point{} : Clock::time_point(Clock::duration(t));
	}
	void disable_callbacks()
	{
		std::lock_guard<std::mutex> lk(cb_mu);
		cb_enabled = false;
	}

	void set_state(SessionState s, const std::string &d = std::string(), const std::string &m = std::string(),
		       const std::string &code = std::string(), const std::string &arg = std::string())
	{
		std::lock_guard<std::mutex> lk(st_mu);
		state = s;
		detail = d;
		message = m;
		reason_code = code;
		reason_arg = arg;
		if (s != SessionState::Backoff)
			backoff_s = 0;
	}

	void add_rtt(double ms)
	{
		std::lock_guard<std::mutex> lk(st_mu);
		rtt_ring[rtt_n % kRttRing] = static_cast<float>(ms);
		++rtt_n;
		rtt_last = ms;
		rtt_avg = rtt_n == 1 ? ms : rtt_avg + 0.05 * (ms - rtt_avg);
	}

	// =========================================================================
	// Audio thread
	// =========================================================================

	void process(float *io, size_t n, bool muted)
	{
		const uint64_t t0 = now_us();
		last_process_us.store(t0, std::memory_order_relaxed);  // the network thread polls this
		if (!supported)
			return;
		size_t total48 = 0;
		while (n > 0) {
			const size_t m = std::min(n, kMaxBlock);
			total48 += process_block(io, m, muted);
			io += m;
			n -= m;
		}
		if (total48 > 0) {
			sched_fill.store(static_cast<uint32_t>(fill), std::memory_order_relaxed);
			sched_chunk.store(static_cast<uint32_t>(total48), std::memory_order_relaxed);
			sched_call_us.store(t0, std::memory_order_release);
		}
	}

	// Returns the number of 48 kHz samples this block gave the frame assembly.
	size_t process_block(float *io, size_t m, bool muted)
	{
		const float *x48 = io;
		size_t n48 = m;
		if (resampled) {
			n48 = up->process(io, m, in48.data());
			x48 = in48.data();
		}
		if (stage)
			stage->push_input(x48, n48);
		assemble(x48, n48, muted);

		if (!stage)
			return n48;  // captions only: the audio is not altered
		if (!resampled) {
			stage->render(io, m);
			return n48;
		}
		size_t need = down->input_needed(m);
		need = std::min(need, out48.size());
		stage->render(out48.data(), need);
		const size_t got = down->process(out48.data(), need, io);
		for (size_t i = got; i < m; ++i)  // cannot happen for host_rate <= 48 kHz; stay safe
			io[i] = got ? io[got - 1] : 0.0f;
		return n48;
	}

	void assemble(const float *x, size_t n, bool muted)
	{
		while (n > 0) {
			const size_t take = std::min(n, kFrameSamples - fill);
			float *dst = cur.pcm + fill;
			if (muted) {
				// Muted in OBS: the service gets silence for these samples (they are not heard, so they are not captioned).
				std::fill(dst, dst + take, 0.0f);
			} else {
				float peak = cur_peak;
				for (size_t i = 0; i < take; ++i) {  // a NaN/Inf from a misbehaving source must not reach the encoder
					const float v = std::isfinite(x[i]) ? x[i] : 0.0f;
					dst[i] = v;
					peak = std::max(peak, std::fabs(v));
				}
				cur_peak = peak;
			}
			fill += take;
			x += take;
			n -= take;
			if (fill < kFrameSamples)
				continue;
			fill = 0;
			cur.seq = next_seq++;
			cur.ts_us = now_us();
			cur.silent = cur_peak < kSilenceLevel;
			cur_peak = 0.0f;
			cur.has_ref = cfg.reference && cfg.reference->pull(cur.ref, kFrameSamples);  // also keeps its clock moving
			counters.captured.fetch_add(1, std::memory_order_relaxed);
			if (!streaming.load(std::memory_order_acquire))
				continue;  // not connected: nothing to send (original audio is played)
			Frame *slot = queue.claim();
			if (!slot) {
				counters.dropped.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			*slot = cur;
			queue.commit();  // the network thread polls the queue (a few ms): no wake-up call from here
		}
	}

	// =========================================================================
	// Network thread
	// =========================================================================

	bool stopping() const { return stop_req.load(std::memory_order_acquire); }

	// Sleeps up to `ms`; returns false when a stop was requested.
	bool sleep_interruptible(std::chrono::milliseconds ms)
	{
		std::unique_lock<std::mutex> lk(mu);
		return !cv.wait_for(lk, ms, [&] { return stopping(); });
	}

	bool audio_is_flowing() const
	{
		const uint64_t t = last_process_us.load(std::memory_order_relaxed);
		return t != 0 && now_us() - t < 2'000'000;
	}

	// When (now_us() clock) the audio thread will complete the next 20 ms frame, worked out from the input sample clock:
	// OBS hands the audio over in chunks of about the same size at a steady pace, so the frame is complete in the call
	// that brings the missing samples, k calls after the last one. 0 = unknown (no audio yet).
	uint64_t next_frame_due_us() const
	{
		const uint64_t call = sched_call_us.load(std::memory_order_acquire);
		if (call == 0)
			return 0;
		const uint64_t fill_now = sched_fill.load(std::memory_order_relaxed);
		const uint64_t chunk = sched_chunk.load(std::memory_order_relaxed);
		if (chunk == 0 || fill_now >= kFrameSamples)
			return 0;
		const uint64_t calls = (kFrameSamples - fill_now + chunk - 1) / chunk;  // calls until the frame is complete
		return call + calls * chunk * 1'000'000 / kNetRate;
	}

	static Clock::time_point time_of_us(uint64_t us) { return clock_epoch() + std::chrono::microseconds(us - 1); }

	// Sleeps until there is something to do: the next frame is due, a ping, the late-frame window or a watchdog is
	// due, a stop is requested or the server closed the connection. (This replaces a poll every 2 ms.)
	void sleep_until_work(Clock::time_point last_ping, Clock::time_point last_window)
	{
		const Clock::time_point now = Clock::now();
		const uint64_t now_u = now_us();
		Clock::time_point wake = now + std::chrono::milliseconds(250);  // the watchdogs below are checked at least this often
		wake = std::min(wake, last_ping + std::chrono::milliseconds(kPingIntervalMs));
		if (stage)
			wake = std::min(wake, last_window + std::chrono::seconds(1));
		const uint64_t due = next_frame_due_us();
		if (due != 0 && now_u < due + kStaleAudioUs) {
			uint64_t at = due + kFrameWakeSlackUs;
			if (at <= now_u)
				at = now_u + 1000;  // overdue: the audio thread is late, look again in a moment
			wake = std::min(wake, time_of_us(at));
		} else {
			wake = std::min(wake, now + std::chrono::milliseconds(50));  // no audio coming: wake rarely, but notice it starting
		}
		std::unique_lock<std::mutex> lk(mu);
		cv.wait_until(lk, wake, [&] { return stopping() || queue.front() != nullptr; });
	}

	// Blocked: waits until the user asks for another try (or the session is stopped).
	void wait_for_retry()
	{
		std::unique_lock<std::mutex> lk(mu);
		cv.wait(lk, [&] { return stopping() || retry_req.load(); });
		retry_req.store(false);
	}

	// ---- late frames ---------------------------------------------------------------
	// Once a second: how many of the frames decided in the last few seconds came back too late? A fixed
	// delay D leaves the network a fixed time budget; beyond it every frame misses its deadline and the
	// listener silently gets the unprocessed audio, so the user is told.
	struct LateWindow {
		uint64_t used[kLateWindowS + 1] = {}, late[kLateWindowS + 1] = {};
		int n = 0;
	};

	void update_late_window(LateWindow &w)
	{
		constexpr int kN = kLateWindowS + 1;
		w.used[w.n % kN] = counters.used.load(std::memory_order_relaxed);
		w.late[w.n % kN] = counters.late.load(std::memory_order_relaxed);
		++w.n;
		const int back = std::min(w.n - 1, kLateWindowS);
		if (back < 2)
			return;
		const int oldest = (w.n - 1 - back) % kN, newest = (w.n - 1) % kN;
		const uint64_t du = w.used[newest] - w.used[oldest], dl = w.late[newest] - w.late[oldest];
		if (du + dl < kLateMinFrames)
			return;  // too few frames to judge (e.g. just connected)
		const double pct = 100.0 * static_cast<double>(dl) / static_cast<double>(du + dl);
		late_x10.store(static_cast<int>(pct * 10.0), std::memory_order_relaxed);
		const bool slow = link_slow.load(std::memory_order_relaxed);
		if (!slow && pct > kLateSetPct) {
			link_slow.store(true, std::memory_order_relaxed);
			log(LogLevel::Warning, "most processed audio is arriving too late for the " + std::string(latency_preset_name(cfg.delay_ms)) +
						       " (" + std::to_string(cfg.delay_ms) + " ms) delay setting; the original audio is played instead");
		} else if (slow && pct < kLateClearPct) {
			link_slow.store(false, std::memory_order_relaxed);
			log(LogLevel::Info, "processed audio is arriving in time again");
		}
	}

	void net_main()
	{
		Backoff backoff(std::chrono::milliseconds(1000), std::chrono::milliseconds(60000));
		// For refusals that come from limits on the server side (busy, two sessions at most, rate limit): each
		// attempt counts against the account's hourly start allowance, so retry slowly.
		Backoff slow_backoff(std::chrono::milliseconds(40000), std::chrono::milliseconds(300000));
		bool first = true;
		while (!stopping()) {
			if (!supported) {  // nothing is sent for an unsupported sample rate: do not hold a session open
				sleep_interruptible(std::chrono::milliseconds(500));
				continue;
			}
			// Connect lazily: only once the host actually feeds audio.
			if (!audio_is_flowing()) {
				set_state(SessionState::Idle, "waiting for audio", "Waiting for audio from OBS.");
				sleep_interruptible(std::chrono::milliseconds(100));
				continue;
			}
			if (!first)
				counters.reconnects.fetch_add(1, std::memory_order_relaxed);
			first = false;

			Outcome out;
			try {
				out = run_attempt();
			} catch (const std::exception &e) {
				// Nothing in here may take the thread (and OBS with it) down: the audio goes on unprocessed.
				out = Outcome{};
				out.reason = std::string("internal error: ") + e.what();
				out.message = "Something went wrong inside the plugin. Reconnecting.";
				out.min_wait_ms = 5000;
			} catch (...) {
				out = Outcome{};
				out.reason = "internal error";
				out.message = "Something went wrong inside the plugin. Reconnecting.";
				out.min_wait_ms = 5000;
			}
			streaming.store(false, std::memory_order_release);
			if (stopping())
				break;
			if (out.streamed_ms >= 30000 && !out.keep_backoff) {
				backoff.reset();
				slow_backoff.reset();
			}
			if (out.give_up) {
				set_state(SessionState::Blocked, out.reason, out.message, out.code, out.arg);
				log(LogLevel::Warning, out.message + " (" + out.reason + ") Not retrying until you press Try again or change a setting.");
				wait_for_retry();
				backoff.reset();
				slow_backoff.reset();
				continue;
			}
			std::chrono::milliseconds wait = out.immediate ? std::chrono::milliseconds(500)
									: (out.slow ? slow_backoff.next() : backoff.next());
			wait = std::max(wait, std::chrono::milliseconds(out.min_wait_ms));
			{
				std::lock_guard<std::mutex> lk(st_mu);
				state = SessionState::Backoff;
				detail = out.reason;
				message = out.message;
				reason_code = out.code;
				reason_arg = out.arg;
				backoff_s = static_cast<int>((wait.count() + 999) / 1000);
			}
			log(LogLevel::Warning, "connection lost (" + out.reason + "); retrying in " + std::to_string(backoff_s) +
						       " s, audio is passed through meanwhile");
			sleep_interruptible(wait);
		}
		streaming.store(false);
		set_state(SessionState::Stopped, "stopped", "Stopped.");
		{
			std::lock_guard<std::mutex> lk(done_mu);
			net_done = true;
		}
		done_cv.notify_all();
	}

	static bool waits_for(Attempt &att, std::chrono::milliseconds ms, const std::function<bool()> &pred,
			      const std::atomic<bool> &stop)
	{
		std::unique_lock<std::mutex> lk(att.mu);
		const auto deadline = Clock::now() + ms;
		while (!pred()) {
			if (stop.load(std::memory_order_acquire) || Clock::now() >= deadline)
				return pred();
			att.cv.wait_for(lk, std::chrono::milliseconds(50));
		}
		return true;
	}

	std::string preset_text() const
	{
		return std::string(latency_preset_name(cfg.delay_ms)) + " (" + std::to_string(cfg.delay_ms) + " ms)";
	}

	// What an `error` event (at start, or later) means for the user and for retrying.
	// Permanent ones (only a changed setting or account fixes them) stop the retries; limits on the server
	// side back off hard; the rest is retried with the normal backoff.
	void classify_server_error(const std::string &code, const std::string &msg, int retry_after_s, Outcome &out) const
	{
		out.reason = "server: " + code + (msg.empty() ? "" : " - " + msg);
		out.code = code;
		out.arg = msg;
		if (code == "insufficient_funds") {
			out.give_up = true;
			out.message = "There is not enough balance to start live processing." + (msg.empty() ? std::string() : " " + msg) +
				      " Add balance on the website, then press Try again.";
		} else if (code == "needs_topup") {
			out.give_up = true;
			out.message = "Extra accurate captions need a first top-up on the website. Add balance there (or use Standard captions), "
				      "then press Try again.";
		} else if (code == "trial_over") {
			out.give_up = true;
			out.message = "Your free trial of live processing is over. Add balance on the website, then press Try again.";
		} else if (code == "trial_busy") {
			out.slow = true;
			out.min_wait_ms = 60000;
			out.message = "Live processing for free trials is busy right now. Trying again in a few minutes.";
		} else if (code == "rate_limited" || code == "too_many_messages") {
			out.code = "too_many_messages";
			out.min_wait_ms = 20000;
			out.message = "The service closed the connection because this plugin sent too many messages. Reconnecting in a moment.";
		} else if (code == "tos") {
			out.give_up = true;
			out.message = "Please accept the terms of service on the website, then press Try again.";
		} else if (code == "bad_config") {
			out.give_up = true;
			out.message = "The server did not accept these settings." + (msg.empty() ? std::string() : " " + msg) +
				      " Change them, then press Try again.";
		} else if (code == "bad_tier") {
			out.give_up = true;
			out.message = "These settings are not available. Change them, then press Try again.";
		} else if (code == "nothing_enabled") {
			out.give_up = true;
			out.message = "Turn on noise removal or captions, then press Try again.";
		} else if (code == "bad_start") {
			out.give_up = true;
			out.message = "The server could not read this plugin's request. Please update the plugin.";
		} else if (code == "too_many_sessions") {
			out.slow = true;
			out.min_wait_ms = 30000;
			out.message = "Your account already has four live sessions running (other Audio Cleaner filters, or another "
				      "computer). Close one of them; this one keeps trying.";
		} else if (code == "full") {
			out.slow = true;
			out.min_wait_ms = 60000;
			out.message = msg.empty() ? std::string("Live processing is full right now. Trying again in a few minutes.") : msg;
		} else if (code == "unavailable") {
			out.slow = true;
			out.min_wait_ms = 30000;
			out.message = "Live processing is busy or offline right now. Trying again soon.";
		} else if (code == "too_fast") {
			out.min_wait_ms = 5000;
			out.message = "Audio was sent faster than real time (this can happen after the computer sleeps). Reconnecting.";
		} else if (code == "worker_lost" || code == "bad_frame") {
			out.min_wait_ms = 5000;
			out.message = "Processing stopped unexpectedly on the server. Reconnecting.";
		} else {
			out.min_wait_ms = 5000;
			out.message = "The server reported a problem" + (msg.empty() ? std::string(".") : ": " + msg) + " Reconnecting.";
		}
		out.min_wait_ms = std::max(out.min_wait_ms, retry_after_s * 1000);
	}

	// Reasons the service may give in the close frame of the socket instead of (or as well as) an event.
	static bool is_close_code(const std::string &r)
	{
		return r == "too_many_messages" || r == "no_audio" || r == "trial_over" || r == "trial_busy" || r == "needs_topup" ||
		       r == "insufficient_funds" || r == "captions_unavailable" || r == "idle_timeout" || r == "account_closed";
	}

	// A `stopped` event: the service ended the session on purpose.
	static void classify_stop(const std::string &stopped, Outcome &out)
	{
		out.reason = "server stopped the session: " + stopped;
		out.code = stopped;
		if (stopped == "max_duration") {
			out.immediate = true;
			out.message = "The session reached its time limit; starting a new one.";
		} else if (stopped == "insufficient_funds") {
			out.give_up = true;
			out.message = "Your balance ran out, so live processing stopped. Add balance on the website, then press "
				      "Try again.";
		} else if (stopped == "account_closed") {
			out.give_up = true;
			out.message = "Your account is not active any more. Check it on the website, then press Try again.";
		} else if (stopped == "trial_over") {
			out.give_up = true;
			out.message = "Your free trial of live processing is over. Add balance on the website, then press Try again.";
		} else if (stopped == "no_audio" || stopped == "idle_timeout") {
			// The microphone audio did not arrive steadily (or at all). Nothing is wrong with the account: start again
			// as soon as audio flows (the loop only connects while it does); the backoff keeps growing if it repeats.
			out.keep_backoff = true;
			out.code = "no_audio";
			out.message = "The service stopped this session because your microphone audio was not arriving steadily. It starts "
				      "again by itself when audio flows.";
		} else if (stopped == "captions_unavailable") {
			out.slow = true;
			out.min_wait_ms = 300000;
			out.message = "The caption service stopped working, so this session ended (nothing is billed for it). Trying again "
				      "in a few minutes.";
		} else if (stopped == "too_many_messages" || stopped == "rate_limited") {
			out.code = "too_many_messages";
			out.min_wait_ms = 20000;
			out.message = "The service closed the connection because this plugin sent too many messages. Reconnecting in a moment.";
		} else {
			out.code.clear();
			out.min_wait_ms = 120000;
			out.message = "The server ended the session. Trying again in a moment.";
		}
	}

	// The WebSocket upgrade failed (HTTP status) or the connection could not be made at all.
	void classify_connect_failure(int http_status, Outcome &out) const
	{
		if (http_status == 401) {
			out.give_up = true;
			out.message = "The server did not accept the API key. Check the key in this filter's settings, then press Try again.";
		} else if (http_status == 403) {
			// The server answers 403 for a key without the live scope, a suspended account and for too many
			// recent sessions alike, so the text names all three and the retry is slow.
			out.slow = true;
			out.min_wait_ms = 120000;
			out.message = "The server refused the connection. Check that the API key was made with \"OBS plugin and live\" "
				      "ticked and that your account is active. If you connected many times in the last hour, this "
				      "clears by itself in a few minutes.";
		} else if (http_status == 429) {
			out.slow = true;
			out.min_wait_ms = 120000;
			out.message = "Too many connection attempts. Waiting a few minutes before trying again.";
		} else if (http_status >= 400 && http_status < 500 && http_status != 408 && http_status != 425) {
			out.give_up = true;
			out.message = "The server rejected the connection (error " + std::to_string(http_status) +
				      "). Check the server address in the advanced settings, then press Try again.";
		} else if (http_status >= 500) {
			out.min_wait_ms = 5000;
			out.message = "The server is having trouble (error " + std::to_string(http_status) + "). Trying again soon.";
		} else {
			out.message = "Could not reach the server. Check your internet connection.";
		}
	}

	Outcome run_attempt()
	{
		Outcome out;
		auto att = std::make_shared<Attempt>();
		att->last_rx_us.store(now_us());
		set_state(SessionState::Connecting, "connecting to " + cfg.url, "Connecting...");

		ix::WebSocket ws;
		ws.setUrl(cfg.url);
		ix::WebSocketHttpHeaders headers;
		headers["Authorization"] = "Bearer " + cfg.api_key;  // never logged
		headers["User-Agent"] = cfg.client;
		ws.setExtraHeaders(headers);
		ws.disableAutomaticReconnection();  // we run our own backoff
		ws.disablePerMessageDeflate();
		ws.setHandshakeTimeout(10);
		ws.setTLSOptions(detail::make_tls_options(cfg.ca_file));

		if (use_opus)
			att->opus_dec = std::make_unique<OpusDec>();
		Impl *self = this;
		ws.setOnMessageCallback([self, att](const ix::WebSocketMessagePtr &msg) {
			switch (msg->type) {
			case ix::WebSocketMessageType::Open: {
				std::lock_guard<std::mutex> lk(att->mu);
				att->opened = true;
				att->cv.notify_all();
				break;
			}
			case ix::WebSocketMessageType::Close: {
				std::lock_guard<std::mutex> lk(att->mu);
				att->closed = true;
				att->close_reason = msg->closeInfo.reason;
				if (att->reason.empty())
					att->reason = "closed by server (code " + std::to_string(msg->closeInfo.code) +
						      (msg->closeInfo.reason.empty() ? "" : ", " + msg->closeInfo.reason) + ")";
				att->cv.notify_all();
				self->cv.notify_all();  // the network thread may be asleep until the next frame is due
				break;
			}
			case ix::WebSocketMessageType::Error: {
				std::lock_guard<std::mutex> lk(att->mu);
				att->closed = true;
				att->http_status = msg->errorInfo.http_status;
				att->reason = msg->errorInfo.reason;
				att->cv.notify_all();
				self->cv.notify_all();
				break;
			}
			case ix::WebSocketMessageType::Message:
				att->last_rx_us.store(now_us(), std::memory_order_relaxed);
				if (msg->binary)
					self->on_binary(*att, msg->str);
				else
					self->on_text(*att, msg->str);
				break;
			default:
				att->last_rx_us.store(now_us(), std::memory_order_relaxed);
				break;
			}
		});
		ws.start();

		// ---- connect + upgrade --------------------------------------------------
		const std::atomic<bool> &stop = stop_req;
		if (!waits_for(*att, std::chrono::seconds(15), [&] { return att->opened || att->closed; }, stop) ||
		    !att->opened) {
			out.reason = describe_failure(*att, "could not connect");
			int hs;
			{
				std::lock_guard<std::mutex> lk(att->mu);
				hs = att->http_status;
			}
			classify_connect_failure(hs, out);
			ws.stop();
			return out;
		}

		// ---- start / ready --------------------------------------------------------
		set_state(SessionState::Handshaking, "starting session", "Connecting...");
		{
			Features f;
			{
				std::lock_guard<std::mutex> lk(st_mu);
				f = feat;
			}
			StartParams sp;
			sp.denoise = f.denoise;
			sp.captions = f.captions;
			sp.language = f.language;
			sp.atten_lim_db = atten_db.load();
			sp.codec = use_opus ? "opus" : "pcm_s16le";
			// Without an output delay (a session started captions only) the returned audio is not used: the smallest
			// stream, or none at all.
			sp.opus_bitrate = use_opus ? (stage ? cfg.opus_bitrate : kOpusMinBitrate) : 0;
			sp.client = cfg.client;
			sp.echo_reference = f.echo_reference;
			sp.return_audio = stage != nullptr;
			if (!f.captions.empty())
				sp.translate = f.translate;  // translations need captions
			if (!ws.sendText(make_start_message(sp)).success) {
				out.reason = "could not send start message";
				out.message = "Could not send the start request. Trying again.";
				ws.stop();
				return out;
			}
		}
		if (!waits_for(*att, std::chrono::seconds(30), [&] { return att->ready || att->closed; }, stop) ||
		    !att->ready) {
			out.reason = describe_failure(*att, "server did not answer the start message");
			std::string code, msg;
			int retry_after = 0;
			{
				std::lock_guard<std::mutex> lk(att->mu);
				code = att->server_code;
				msg = att->server_msg;
				retry_after = att->retry_after_s;
			}
			if (!code.empty())
				classify_server_error(code, msg, retry_after, out);
			else
				out.message = "The server did not answer the start request. Trying again.";
			ws.stop();
			return out;
		}
		Event ready;
		{
			std::lock_guard<std::mutex> lk(att->mu);
			ready = att->ready_ev;
		}
		if (ready.sample_rate != 0 && ready.sample_rate != kNetRate) {
			out.reason = "server sample rate " + std::to_string(ready.sample_rate) + " is not supported";
			out.message = "This plugin and the server do not use the same audio format. Please update the plugin.";
			out.give_up = true;
			ws.stop();
			return out;
		}
		// The numbers in `ready` decide the output timing: check them before they are used.
		const double ml = ready.model_latency_ms;
		if (!std::isfinite(ml) || ml < 0.0 || ml > 2000.0) {
			out.reason = "invalid modelLatencyMs in the server's ready message";
			out.message = "The server sent a reply this plugin does not understand. Trying again.";
			out.min_wait_ms = 30000;
			ws.stop();
			return out;
		}
		const double total_latency_ms = ml + opus_extra_latency * 1000.0 / kNetRate;
		if (stage && static_cast<double>(cfg.delay_ms) - total_latency_ms - kFramingMs <= 0.0) {
			// Even with a perfect network no processed frame could arrive in time.
			out.reason = "model latency " + std::to_string(static_cast<int>(total_latency_ms)) + " ms does not fit the " +
				     std::to_string(cfg.delay_ms) + " ms delay";
			out.message = "The " + preset_text() + " delay setting is too short for this server. Choose a longer one in "
				      "this filter's settings.";
			out.give_up = true;
			ws.stop();
			return out;
		}
		echo_active.store(cfg.reference && ready.echo_reference);  // follows changes (a `config` answer) from here
		const bool send_silence = ready.silence_frames;  // silent blocks go out as silence frames, only if the service takes them
		silence_active.store(send_silence);
		{
			std::lock_guard<std::mutex> lk(st_mu);
			if (feat.echo_reference && !ready.echo_reference)
				log(LogLevel::Warning, "the server does not offer echo cancellation: continuing without it");
		}
		const int latency = static_cast<int>(std::lround(ml * (kNetRate / 1000.0))) + opus_extra_latency;
		const int warm_frames = (latency + static_cast<int>(kFrameSamples) - 1) / static_cast<int>(kFrameSamples);
		if (stage)
			stage->set_model_latency(latency);
		{
			std::lock_guard<std::mutex> lk(st_mu);
			session_id = ready.session_id;
			usd_per_minute = ready.usd_per_minute;
			model_latency_ms = ready.model_latency_ms;
			translate_active = ready.translate;
			translate_active_known = ready.has_translate;
		}
		set_state(SessionState::Streaming, "connected");
		{
			char b[160];
			std::snprintf(b, sizeof b, "connected (%s, model latency %.0f ms, %.4f USD/min, delay %d ms)",
				      use_opus ? "opus" : "pcm_s16le", ready.model_latency_ms, ready.usd_per_minute, cfg.delay_ms);
			log(LogLevel::Info, b);
		}

		// ---- stream ------------------------------------------------------------------
		while (queue.front())
			queue.pop();  // anything left from before the connection is stale
		streaming.store(true, std::memory_order_release);

		const auto t_start = Clock::now();
		auto last_ping = t_start;
		auto last_window = t_start;
		LateWindow window;
		uint64_t last_frame_us = now_us();
		bool first_frame = true;
		std::string tx(kHeaderSize + std::max(2 * kFrameSamples, kMaxOpusPacket), '\0');
		std::unique_ptr<OpusEnc> enc;
		if (use_opus)
			enc = std::make_unique<OpusEnc>(cfg.opus_bitrate);  // fresh state for every connection
		std::unique_ptr<OpusEnc> ref_enc;  // the reference is a stream of its own (made when echo removal is first on)

		while (!stopping()) {
			{
				std::lock_guard<std::mutex> lk(att->mu);
				if (att->closed)
					break;
			}
			while (Frame *f = queue.front()) {
				last_frame_us = now_us();
				if (first_frame) {
					att->first_valid_seq.store(static_cast<int64_t>(f->seq) + warm_frames, std::memory_order_release);
					first_frame = false;
				}
				if (ws.bufferedAmount() > kMaxTxBuffered) {
					counters.dropped.fetch_add(1, std::memory_order_relaxed);
					queue.pop();
					continue;
				}
				// Encodes one frame into `tx`: its length, or 0 when the encoder failed.
				auto encode = [&](OpusEnc *e, const float *pcm, uint16_t flags) -> size_t {
					uint8_t *txp = reinterpret_cast<uint8_t *>(&tx[0]);
					if (!e)
						return encode_frame_s16(f->seq, f->ts_us, pcm, kFrameSamples, txp, flags);
					const int n = e->encode(pcm, txp + kHeaderSize, static_cast<int>(kMaxOpusPacket));
					if (n <= 0)
						return 0;
					encode_frame_header(kCodecOpus, f->seq, f->ts_us, txp, flags);
					return kHeaderSize + static_cast<size_t>(n);
				};
				auto send = [&](size_t len) -> bool {
					const size_t cap = tx.size();
					tx.resize(len);
					const bool ok = ws.sendBinary(tx).success;
					tx.resize(cap);
					if (ok)
						counters.tx_bytes.fetch_add(len, std::memory_order_relaxed);
					return ok;
				};
				// Silence goes out as a 18-byte silence frame when the server takes them (nothing is encoded, it costs the
				// server nothing and is not billed); otherwise it is encoded like any other audio.
				auto silence = [&](uint16_t flags) -> size_t {
					return encode_frame_silence(f->seq, f->ts_us, kFrameSamples, reinterpret_cast<uint8_t *>(&tx[0]), flags);
				};
				// The reference goes first, so the server has it when the microphone frame arrives. A silent reference
				// is a silence frame, or (a server that does not take them) skipped. The microphone frames are never
				// skipped, silent or not: they are what keeps the session alive on the server.
				if (echo_active.load(std::memory_order_relaxed)) {
					if (use_opus && !ref_enc)
						ref_enc = std::make_unique<OpusEnc>(cfg.opus_bitrate);
					size_t rlen = 0;
					if (f->has_ref)
						rlen = encode(ref_enc.get(), f->ref, kFlagReference);
					else if (send_silence)
						rlen = silence(kFlagReference);
					if (rlen && send(rlen))
						counters.reference_sent.fetch_add(1, std::memory_order_relaxed);
				}
				const bool as_silence = send_silence && f->silent;
				const size_t len = as_silence ? silence(0) : encode(enc.get(), f->pcm, 0);
				if (!len) {
					counters.dropped.fetch_add(1, std::memory_order_relaxed);
					queue.pop();
					continue;
				}
				if (static_cast<int64_t>(f->seq) >= att->first_valid_seq.load(std::memory_order_relaxed))
					store.mark_sent(f->seq);
				if (send(len)) {
					counters.sent.fetch_add(1, std::memory_order_relaxed);
					if (as_silence)
						counters.silent_sent.fetch_add(1, std::memory_order_relaxed);
				} else
					counters.dropped.fetch_add(1, std::memory_order_relaxed);
				queue.pop();
			}

			const auto now = Clock::now();
			if (now - last_ping >= std::chrono::milliseconds(kPingIntervalMs)) {
				ws.sendText(make_ping_message(now_us()));
				last_ping = now;
			}
			if (now - last_window >= std::chrono::seconds(1)) {
				last_window = now;
				if (stage)
					update_late_window(window);
			}
			const int atten = pending_atten.exchange(-1);
			if (atten >= 0)
				ws.sendText(make_config_message(atten));
			{
				std::lock_guard<std::mutex> lk(chg_mu);
				if (pending_change && !sent_change) {  // one change at a time: the next waits for the answer
					sent_change_id = "c" + std::to_string(++change_seq);
					sent_change = std::move(pending_change);
					pending_change.reset();
					ws.sendText(make_change_message(*sent_change, sent_change_id));
				}
			}

			if (now_us() - att->last_rx_us.load(std::memory_order_relaxed) > uint64_t(kRxWatchdogMs) * 1000) {
				out.reason = "no data from the server for " + std::to_string(kRxWatchdogMs / 1000) + " s";
				out.message = "The server stopped answering. Reconnecting.";
				break;
			}
			if (now_us() - last_frame_us > uint64_t(kIdleDisconnectMs) * 1000 && !audio_is_flowing()) {
				// The host stopped feeding audio (filter muted/disabled upstream): close quietly.
				out.reason = "no audio from OBS";
				out.message = "Waiting for audio from OBS.";
				out.immediate = true;
				out.min_wait_ms = 0;
				break;
			}

			counters.net_loops.fetch_add(1, std::memory_order_relaxed);
			sleep_until_work(last_ping, last_window);
		}
		streaming.store(false, std::memory_order_release);
		origin_ticks.store(0, std::memory_order_release);
		echo_active.store(false);
		{
			// A change the connection ended before answering: the next connection starts with it (or with the one
			// asked for since), and the service checks it then.
			std::lock_guard<std::mutex> lk(chg_mu);
			std::optional<Features> next = pending_change ? pending_change : sent_change;
			pending_change.reset();
			sent_change.reset();
			if (next) {
				std::lock_guard<std::mutex> lk2(st_mu);
				feat = *next;
			}
		}
		silence_active.store(false);
		link_slow.store(false);
		late_x10.store(0);
		out.streamed_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_start).count());

		if (stopping()) {
			// Graceful end: the server flushes outstanding captions, then closes.
			ws.sendText(make_stop_message());
			const int fm = flush_ms.load();
			if (fm > 0)
				waits_for(*att, std::chrono::milliseconds(fm), [&] { return att->closed; }, std::atomic<bool>{false});
			ws.stop();
			return out;
		}

		if (out.reason.empty()) {
			std::string stopped, code, msg, transport, close_reason;
			int retry_after = 0;
			{
				std::lock_guard<std::mutex> lk(att->mu);
				stopped = att->stopped_reason;
				code = att->server_code;
				msg = att->server_msg;
				retry_after = att->retry_after_s;
				transport = att->reason;
				close_reason = att->close_reason;
			}
			if (stopped.empty() && code.empty() && is_close_code(close_reason))
				code = close_reason;  // the service can name the reason in the close frame alone
			if (!stopped.empty()) {
				classify_stop(stopped, out);
			} else if (!code.empty()) {
				classify_server_error(code, msg, retry_after, out);
			} else {
				out.reason = transport.empty() ? "connection closed" : transport;
				out.message = "The connection to the server was lost. Reconnecting.";
			}
		}
		ws.stop();
		return out;
	}

	static std::string describe_failure(Attempt &att, const char *fallback)
	{
		std::lock_guard<std::mutex> lk(att.mu);
		std::string r = att.reason;
		if (att.http_status > 0)
			r += (r.empty() ? "" : " ") + std::string("(HTTP ") + std::to_string(att.http_status) + ")";
		if (!att.server_code.empty())
			r = "server: " + att.server_code + (att.server_msg.empty() ? "" : " - " + att.server_msg);
		return r.empty() ? fallback : r;
	}

	// ---- IX thread handlers -----------------------------------------------------
	void on_binary(Attempt &att, const std::string &data)
	{
		FrameHeader h;
		size_t n = 0;
		const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data.data());
		if (!parse_frame_header(bytes, data.size(), h)) {
			counters.bad_frames.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		if (!stage) {
			// No output delay (started captions only): the audio that comes back is not used, so it is not decoded either.
			if (data.size() <= kHeaderSize) {
				counters.bad_frames.fetch_add(1, std::memory_order_relaxed);
				return;
			}
			n = kFrameSamples;
		} else if (h.codec == kCodecOpus) {
			const int got = att.opus_dec ? att.opus_dec->decode(bytes + kHeaderSize, static_cast<int>(data.size() - kHeaderSize),
									     rx.data())
						     : -1;
			if (got <= 0) {
				counters.bad_frames.fetch_add(1, std::memory_order_relaxed);
				return;
			}
			n = static_cast<size_t>(got);
		} else if (!decode_frame(bytes, data.size(), h, rx.data(), rx.size(), n)) {
			counters.bad_frames.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		counters.rx_bytes.fetch_add(data.size(), std::memory_order_relaxed);
		const uint64_t t = now_us();
		if (t >= h.ts_us)
			add_rtt(static_cast<double>(t - h.ts_us) / 1000.0);
		counters.received.fetch_add(1, std::memory_order_relaxed);
		if (n != kFrameSamples) {
			counters.bad_frames.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		if (static_cast<int64_t>(h.seq) < att.first_valid_seq.load(std::memory_order_acquire)) {
			counters.warmup_discarded.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		if (stage)
			store.publish(h.seq, rx.data());
	}

	void on_text(Attempt &att, const std::string &text)
	{
		Event ev;
		if (!parse_event(text, ev))
			return;
		switch (ev.type) {
		case EventType::Ready: {
			// Second 0 of the service's clock (the audio it counts starts with the session): events follow in order.
			origin_ticks.store(Clock::now().time_since_epoch().count(), std::memory_order_release);
			{
				// A new connection starts with a clean status. Cleared here, on this thread, before any message that
				// follows `ready` (a warning can come right after it, before the connecting thread wakes up).
				std::lock_guard<std::mutex> st(st_mu);
				low_balance = false;
				seconds_left = -1;
				warning.clear();
				translate_failed.clear();
				captions_failed = false;
				other_warnings.clear();
			}
			std::lock_guard<std::mutex> lk(att.mu);
			att.ready_ev = ev;
			att.ready = true;
			att.cv.notify_all();
			break;
		}
		case EventType::Caption:
			if (ev.is_final && !ev.text.empty()) {
				CaptionEvent ce;
				ce.text = ev.text;
				ce.is_final = true;
				ce.start = ev.start;
				ce.end = ev.end;
				ce.id = ev.caption_id;
				ce.origin = origin();
				emit_caption(ce);
			}
			break;
		case EventType::Translation:
			if (!ev.text.empty() && !ev.lang.empty()) {
				TranslationEvent te;
				te.lang = normalise_language(ev.lang);
				te.text = ev.text.size() > kMaxCaptionBytes ? ev.text.substr(0, kMaxCaptionBytes) : ev.text;
				te.of = ev.of;
				te.start = ev.start;
				te.end = ev.end;
				te.origin = origin();
				if (te.lang != "auto")
					emit_translation(te);
			}
			break;
		case EventType::Warning: {
			std::string msg;
			if (ev.code == "low_balance") {
				{
					std::lock_guard<std::mutex> lk(st_mu);
					low_balance = true;
					seconds_left = ev.seconds_left;
				}
				msg = "balance running low";
				if (ev.seconds_left >= 0)
					msg += " (about " + std::to_string(ev.seconds_left) + " s of live processing left)";
			} else if (ev.code == "translation_unavailable") {
				const std::string lang = normalise_language(ev.lang);
				msg = "translation into " + language_name(lang) + " stopped" + (ev.message.empty() ? "" : ": " + ev.message);
				std::lock_guard<std::mutex> lk(st_mu);
				if (std::find(translate_failed.begin(), translate_failed.end(), lang) == translate_failed.end() && translate_failed.size() < 8)
					translate_failed.push_back(lang);
			} else if (ev.code == "captions_unavailable") {
				msg = ev.message.empty() ? "captions stopped" : ev.message;
				std::lock_guard<std::mutex> lk(st_mu);
				captions_failed = true;
			} else {
				msg = ev.message.empty() ? ev.code : ev.message;
				std::lock_guard<std::mutex> lk(st_mu);
				warning = msg;
			}
			log(LogLevel::Warning, "server warning: " + msg);
			break;
		}
		case EventType::Stopped: {
			std::lock_guard<std::mutex> lk(att.mu);
			att.stopped_reason = ev.code;
			att.cv.notify_all();
			break;
		}
		case EventType::Error: {
			std::lock_guard<std::mutex> lk(att.mu);
			att.server_code = ev.code;
			att.server_msg = ev.message;
			att.retry_after_s = ev.retry_after_s;
			att.cv.notify_all();
			break;
		}
		case EventType::Config: {
			// The answer to a change (change()): what the session does from now on, or why nothing changed.
			std::optional<Features> asked;
			{
				std::lock_guard<std::mutex> lk(chg_mu);
				if (!sent_change || ev.id != sent_change_id)
					break;
				asked = std::move(sent_change);
				sent_change.reset();
			}
			cv.notify_all();  // a change waiting behind it can go out
			std::lock_guard<std::mutex> lk(st_mu);
			if (ev.ok) {
				feat = *asked;
				usd_per_minute = ev.usd_per_minute;
				translate_active = ev.translate;
				translate_active_known = ev.has_translate;
				translate_failed.clear();
				captions_failed = false;
				change_refused.clear();
				echo_active.store(cfg.reference && ev.echo_reference);
			} else {
				change_refused = ev.message.empty() ? "The service could not make that change." : ev.message;
			}
			break;
		}
		case EventType::Pong:
			if (ev.has_pong_t) {
				const uint64_t t = now_us();
				if (t >= ev.pong_t) {
					std::lock_guard<std::mutex> lk(st_mu);
					ping_rtt_ms = static_cast<double>(t - ev.pong_t) / 1000.0;
				}
			}
			break;
		case EventType::Unknown:
			break;
		}
	}
};

// ---------------------------------------------------------------------------

Session::Session(const SessionConfig &cfg, CaptionCallback on_caption, LogCallback on_log)
	: impl_(std::make_shared<Impl>(cfg, std::move(on_caption), std::move(on_log)))
{
}

Session::~Session()
{
	stop(0, 500);
}

void Session::set_translation_callback(TranslationCallback cb)
{
	std::lock_guard<std::mutex> lk(impl_->cb_mu);
	impl_->on_translation = std::move(cb);
}

bool Session::started() const
{
	return impl_->started.load();
}

void Session::start()
{
	Impl &d = *impl_;
	bool expected = false;
	if (!d.started.compare_exchange_strong(expected, true))
		return;
	auto refuse = [&](const std::string &detail, const std::string &message) {
		d.set_state(SessionState::Stopped, detail, message);
		d.log(LogLevel::Warning, message);
		d.stop_req.store(true);
		std::lock_guard<std::mutex> lk(d.done_mu);
		d.net_done = true;
	};
	if (is_secure_url(d.cfg.url) == false && !is_plain_url(d.cfg.url)) {
		refuse("server URL must start with wss://", "The server address must start with wss://. Check it in the advanced settings.");
		return;
	}
	if (d.insecure && !is_local_host(url_host(d.cfg.url))) {
		// The API key and the audio would cross the internet in clear text.
		refuse("ws:// is only allowed for this computer or the local network",
		       "Unencrypted ws:// addresses only work for this computer or your local network. Use a wss:// address.");
		return;
	}
	if (d.cfg.api_key.empty()) {
		refuse("no API key configured", "Enter an API key to start.");
		return;
	}
	if (d.insecure)
		d.log(LogLevel::Warning, "the server URL uses ws:// (unencrypted): the API key and audio travel in clear text; "
					 "use this for local testing only");
	g_bg_threads.fetch_add(1);
	try {
		auto self = impl_;
		d.th = std::thread([self]() mutable {
			self->net_main();
			self.reset();  // may free the session state: nothing of the plugin's data is touched after this
			g_bg_threads.fetch_sub(1);
		});
	} catch (const std::exception &e) {
		g_bg_threads.fetch_sub(1);
		d.set_state(SessionState::Stopped, std::string("could not start network thread: ") + e.what(),
			    "Could not start the network thread.");
		std::lock_guard<std::mutex> lk(d.done_mu);
		d.net_done = true;
	}
}

bool wait_for_background_threads(int ms)
{
	const auto deadline = Clock::now() + std::chrono::milliseconds(std::max(0, ms));
	while (g_bg_threads.load() > 0 && Clock::now() < deadline)
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	return g_bg_threads.load() == 0;
}

bool server_url_allowed(const std::string &url)
{
	return is_secure_url(url) || (is_plain_url(url) && is_local_host(url_host(url)));
}

const char *latency_preset_name(int delay_ms)
{
	if (delay_ms <= 80)
		return "Low";
	if (delay_ms <= 150)
		return "Balanced";
	return "Safe";
}

void Session::request_stop()
{
	Impl &d = *impl_;
	{
		std::lock_guard<std::mutex> lk(d.mu);
		d.stop_req.store(true, std::memory_order_release);
	}
	d.cv.notify_all();
}

void Session::retry_now()
{
	Impl &d = *impl_;
	{
		std::lock_guard<std::mutex> lk(d.st_mu);
		if (d.state != SessionState::Blocked)
			return;
	}
	{
		std::lock_guard<std::mutex> lk(d.mu);
		d.retry_req.store(true);
	}
	d.cv.notify_all();
}

void Session::stop(int flush_ms, int join_timeout_ms)
{
	Impl &d = *impl_;
	if (!d.started.load()) {
		d.disable_callbacks();
		return;
	}
	d.flush_ms.store(std::max(0, flush_ms));
	{
		std::lock_guard<std::mutex> lk(d.mu);
		d.stop_req.store(true, std::memory_order_release);
	}
	d.cv.notify_all();
	if (d.th.joinable()) {
		bool done;
		{
			std::unique_lock<std::mutex> lk(d.done_mu);
			done = d.done_cv.wait_for(lk, std::chrono::milliseconds(std::max(0, flush_ms) + std::max(0, join_timeout_ms)),
						  [&] { return d.net_done; });
		}
		if (done) {
			d.th.join();
		} else {
			// Never hold up OBS shutdown: the thread owns a reference to the state and exits on its own.
			d.log(LogLevel::Warning, "network thread did not stop in time; detaching it");
			d.th.detach();
		}
	}
	d.disable_callbacks();
}

void Session::process(float *mono, size_t n, bool muted)
{
	impl_->process(mono, n, muted);
}

double Session::delay_seconds() const
{
	return impl_->supported ? impl_->delay_sec : 0.0;
}

bool Session::change(const SessionConfig &next)
{
	Impl &d = *impl_;
	const SessionConfig &c = d.cfg;
	if (next.url != c.url || next.api_key != c.api_key || next.delay_ms != c.delay_ms || next.codec != c.codec ||
	    next.host_rate != c.host_rate || next.reference != c.reference)
		return false;
	if (next.denoise && !d.stage && d.supported)
		return false;  // noise removal in a session whose audio was never delayed: a new session (with the delay) it is
	Features f;
	f.denoise = next.denoise;
	f.captions = next.captions;
	f.language = next.language;
	if (!next.captions.empty())
		f.translate = next.translate;
	f.echo_reference = next.echo && next.reference;
	{
		std::lock_guard<std::mutex> lk(d.st_mu);
		if (!d.streaming.load()) {
			d.feat = f;  // not connected: the next connection starts with it
			d.change_refused.clear();
			return true;
		}
	}
	{
		std::lock_guard<std::mutex> lk(d.chg_mu);
		d.pending_change = f;  // replaces one that hasn't gone out yet
	}
	d.cv.notify_all();
	return true;
}

void Session::set_atten_lim_db(int db)
{
	db = std::min(100, std::max(6, db));
	impl_->atten_db.store(db);  // used by the next (re)connect
	impl_->pending_atten.store(db);
	impl_->cv.notify_all();
}

std::vector<float> Session::rtt_history() const
{
	std::lock_guard<std::mutex> lk(impl_->st_mu);
	const size_t n = std::min(impl_->rtt_n, kRttRing);
	std::vector<float> v;
	v.reserve(n);
	const size_t start = impl_->rtt_n - n;
	for (size_t i = 0; i < n; ++i)
		v.push_back(impl_->rtt_ring[(start + i) % kRttRing]);
	return v;
}

SessionSnapshot Session::snapshot() const
{
	const Impl &d = *impl_;
	SessionSnapshot s;
	{
		std::lock_guard<std::mutex> lk(d.st_mu);
		s.state = d.state;
		s.detail = d.detail;
		s.message = d.message;
		s.session_id = d.session_id;
		s.usd_per_minute = d.usd_per_minute;
		s.model_latency_ms = d.model_latency_ms;
		s.backoff_seconds = d.backoff_s;
		s.low_balance = d.low_balance;
		s.seconds_left = d.seconds_left;
		s.warning = d.warning;
		s.reason_code = d.reason_code;
		s.reason_arg = d.reason_arg;
		s.translate_active = d.translate_active;
		s.translate_active_known = d.translate_active_known;
		s.translate_failed = d.translate_failed;
		s.captions_failed = d.captions_failed;
		s.ping_rtt_ms = d.ping_rtt_ms;
		s.rtt_last_ms = d.rtt_last;
		s.rtt_avg_ms = d.rtt_avg;
		const size_t n = std::min<size_t>(d.rtt_n, 2048);
		if (n) {
			std::vector<float> v(n);
			for (size_t i = 0; i < n; ++i)
				v[i] = d.rtt_ring[(d.rtt_n - n + i) % kRttRing];
			s.rtt_max_ms = *std::max_element(v.begin(), v.end());
			const size_t k = static_cast<size_t>(0.95 * static_cast<double>(n - 1));
			std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
			s.rtt_p95_ms = v[k];
		}
	}
	{
		std::lock_guard<std::mutex> lk(d.st_mu);
		if (!d.feat.captions.empty())
			s.translate_requested = d.feat.translate;
		s.echo_requested = d.feat.echo_reference;
		s.change_refused = d.change_refused;
	}
	s.insecure = d.insecure;
	s.codec = d.use_opus ? "opus" : "pcm_s16le";
	s.echo_active = d.echo_active.load();
	s.silence_active = d.silence_active.load();
	s.passthrough_only = !d.supported;
	s.delay_ms = d.supported ? d.delay_sec * 1000.0 : 0.0;
	s.captured = d.counters.captured.load();
	s.sent = d.counters.sent.load();
	s.received = d.counters.received.load();
	s.used = d.counters.used.load();
	s.late = d.counters.late.load();
	s.passthrough = d.counters.passthrough.load();
	s.dropped = d.counters.dropped.load();
	s.silent_sent = d.counters.silent_sent.load();
	s.net_loops = d.counters.net_loops.load();
	s.bad_frames = d.counters.bad_frames.load();
	s.warmup_discarded = d.counters.warmup_discarded.load();
	s.tx_bytes = d.counters.tx_bytes.load();
	s.rx_bytes = d.counters.rx_bytes.load();
	s.reconnects = d.counters.reconnects.load();
	s.link_too_slow = d.link_slow.load();
	s.late_percent = d.late_x10.load() / 10.0;
	return s;
}

namespace {

std::string replace_first(std::string text, const std::string &from, const std::string &to)
{
	const size_t at = text.find(from);
	if (at != std::string::npos)
		text.replace(at, from.size(), to);
	return text;
}

// "Spanish", "Spanish and German", "Spanish, German and French".
std::string join_names(const std::vector<std::string> &codes)
{
	std::string out;
	for (size_t i = 0; i < codes.size(); ++i) {
		if (i > 0)
			out += i + 1 == codes.size() ? " and " : ", ";
		out += language_name(codes[i]);
	}
	return out;
}

}  // namespace

std::string Session::status_line(const Localizer &loc) const
{
	const Impl &d = *impl_;
	SessionSnapshot s = snapshot();
	Features f;
	{
		std::lock_guard<std::mutex> lk(d.st_mu);
		f = d.feat;
	}
	const bool cleaning = f.denoise && d.supported;
	// A text in the user's language, or the built-in English one. "%1" is replaced by `arg`.
	auto say = [&](const std::string &id, const std::string &builtin, const std::string &arg = std::string()) {
		std::string t = loc ? loc(id) : std::string();
		return t.empty() ? builtin : replace_first(t, "%1", arg);
	};
	if (!s.reason_code.empty() && (s.state == SessionState::Backoff || s.state == SessionState::Blocked)) {
		const std::string t = loc ? loc("reason." + s.reason_code) : std::string();
		if (!t.empty())
			s.message = replace_first(t, "%1", s.reason_arg);
	}
	std::string out;
	switch (s.state) {
	case SessionState::Streaming:
		if (s.link_too_slow) {
			out = "Your connection is too slow for the " + d.preset_text() +
			      " delay setting, so your audio is not being cleaned.";
			out += d.cfg.delay_ms < 250 ? " Choose a longer delay (Balanced or Safe) in this filter's settings."
						    : " Check your internet connection.";
		} else {
			out = cleaning ? "Connected. Your audio is being cleaned." : "Connected.";
			if (!f.captions.empty() && !s.captions_failed)
				out += " Captions are running.";
			// Translations: the ones the service switched on, and the ones it did not.
			std::vector<std::string> on, off;
			for (const std::string &l : s.translate_requested) {
				const bool active = std::find(s.translate_active.begin(), s.translate_active.end(), l) != s.translate_active.end();
				const bool failed = std::find(s.translate_failed.begin(), s.translate_failed.end(), l) != s.translate_failed.end();
				if (failed)
					continue;  // said below
				(active ? on : off).push_back(l);
			}
			if (!on.empty() && !s.captions_failed)
				out += " " + say("status.translating", "Translating into " + join_names(on) + ".", join_names(on));
			if (!off.empty())
				out += " " + say("warning.translation_not_started",
						 "The service did not switch on translation into " + join_names(off) + ".", join_names(off));
		}
		if (s.echo_active)
			out += " Echo cancellation is on.";
		else if (s.echo_requested)
			out += " Echo cancellation is not available on this server.";
		break;
	case SessionState::Backoff:
		out = s.message.empty() ? s.detail : s.message;
		if (cleaning)
			out += " Your audio keeps playing without cleaning.";
		out += " Trying again in " + std::to_string(s.backoff_seconds) + " s.";
		break;
	case SessionState::Blocked:
		out = s.message.empty() ? s.detail : s.message;
		break;
	case SessionState::Connecting:
	case SessionState::Handshaking:
		out = "Connecting...";
		break;
	case SessionState::Idle:
		out = s.message.empty() ? "Waiting for audio from OBS." : s.message;
		break;
	case SessionState::Stopped:
		out = s.message.empty() ? "Stopped." : s.message;
		break;
	}
	if (s.low_balance) {
		out += " Your balance is running low";
		if (s.seconds_left >= 0)
			out += " (about " + std::to_string((s.seconds_left + 59) / 60) + " min left)";
		out += ".";
	}
	if (s.captions_failed)
		out += " " + say("warning.captions_unavailable",
				 "The caption service had trouble, so captions have stopped" +
					 std::string(cleaning ? "; your audio is not affected." : "."));
	for (const std::string &l : s.translate_failed)
		out += " " + say("warning.translation_unavailable",
				 "Translation into " + language_name(l) + " stopped because the translation service had trouble; the other captions go on.",
				 language_name(l));
	if (!s.change_refused.empty())
		out += " " + say("warning.change_refused", "That change could not be made: " + s.change_refused +
								     " The session goes on with the earlier settings.", s.change_refused);
	if (!s.warning.empty())
		out += " " + s.warning;
	if (s.insecure)
		out += " Warning: this connection is not encrypted.";
	if (s.passthrough_only)
		out += " Your audio is not changed: OBS uses a sample rate this filter does not support.";
	return out;
}

std::string Session::status_details() const
{
	const SessionSnapshot s = snapshot();
	char buf[160];
	std::string out = to_string(s.state);
	if (!s.codec.empty())
		out += ", " + s.codec;
	if (s.delay_ms > 0) {
		std::snprintf(buf, sizeof buf, ", delay %.0f ms", s.delay_ms);
		out += buf;
	}
	if (s.state == SessionState::Streaming && (s.rtt_last_ms > 0 || s.received > 0)) {
		std::snprintf(buf, sizeof buf, ", round trip %.0f ms (p95 %.0f)", s.rtt_avg_ms, s.rtt_p95_ms);
		out += buf;
	} else if (s.state == SessionState::Streaming && s.ping_rtt_ms > 0) {
		// Captions only: no audio comes back, so the round trip is the one of the pings.
		std::snprintf(buf, sizeof buf, ", round trip %.0f ms", s.ping_rtt_ms);
		out += buf;
	}
	if (s.state == SessionState::Streaming || s.captured > 0) {
		out += ", frames in time " + std::to_string(s.used) + ", late " + std::to_string(s.late) + ", unprocessed " +
		       std::to_string(s.passthrough);
		if (s.dropped)
			out += ", dropped " + std::to_string(s.dropped);
		if (s.silent_sent)
			out += ", silent frames (not billed) " + std::to_string(s.silent_sent);
		if (s.late_percent > 0) {
			std::snprintf(buf, sizeof buf, ", recently late %.0f%%", s.late_percent);
			out += buf;
		}
	}
	if (s.reconnects)
		out += ", reconnects " + std::to_string(s.reconnects);
	if (!s.detail.empty() && s.state != SessionState::Streaming)
		out += "; " + s.detail;
	return out;
}

}  // namespace acw
