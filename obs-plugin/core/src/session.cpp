// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/session.hpp"

#include "acw/backoff.hpp"
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
uint64_t now_us()
{
	static const Clock::time_point t0 = Clock::now();
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()) + 1;
}

struct Frame {
	uint32_t seq = 0;
	uint64_t ts_us = 0;
	float pcm[kFrameSamples] = {};
	float ref[kFrameSamples] = {};  // what the speakers played meanwhile (echo cancellation)
	bool has_ref = false;           // ... and it is not silence
};

constexpr size_t kMaxBlock = 2048;           // host-rate samples handled per inner step
constexpr size_t kQueueFrames = 32;          // 640 ms between the audio thread and the network thread
constexpr size_t kMaxTxBuffered = 40 * 1024; // bytes queued in the socket before we drop frames
constexpr int kPingIntervalMs = 5000;
constexpr int kRxWatchdogMs = 20000;
constexpr int kIdleDisconnectMs = 10000;     // no audio from the host for this long: close the session
constexpr size_t kRttRing = 8192;
constexpr int kLateWindowS = 6;          // seconds over which late frames are counted
constexpr uint64_t kLateMinFrames = 100;  // ... needs at least this many frames (2 s) before judging
constexpr double kLateSetPct = 30.0;     // more than this many percent late: the link is too slow
constexpr double kLateClearPct = 15.0;   // ... until it is below this again
constexpr int kFramingMs = 25;           // 20 ms frame + 5 ms cross-fade: what the delay budget loses to framing
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
		int min_wait_ms = 0;
		int streamed_ms = 0;
		bool immediate = false;  // reconnect without waiting (server restarted the session)
		bool give_up = false;    // only the user can fix this: stop retrying until retry_now()
		bool slow = false;       // the server is limiting us (busy, too many sessions, rate limit): back off hard
	};

	SessionConfig cfg;
	CaptionCallback on_caption;
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
	uint32_t next_seq = 0;
	SpscQueue<Frame> queue{kQueueFrames};
	std::atomic<bool> streaming{false};
	std::atomic<bool> echo_active{false};  // the server accepted the echo reference
	std::atomic<int> atten_db{100};        // current attenuation limit (UI thread writes, network thread reads)
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
		if (cb_enabled && on_log)
			on_log(lv, msg);
	}
	void emit_caption(const CaptionEvent &ev)
	{
		std::lock_guard<std::mutex> lk(cb_mu);
		if (cb_enabled && on_caption)
			on_caption(ev);
	}
	void disable_callbacks()
	{
		std::lock_guard<std::mutex> lk(cb_mu);
		cb_enabled = false;
	}

	void set_state(SessionState s, const std::string &d = std::string(), const std::string &m = std::string())
	{
		std::lock_guard<std::mutex> lk(st_mu);
		state = s;
		detail = d;
		message = m;
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

	void process(float *io, size_t n)
	{
		last_process_us.store(now_us(), std::memory_order_relaxed);  // the network thread polls this
		if (!supported)
			return;
		while (n > 0) {
			const size_t m = std::min(n, kMaxBlock);
			process_block(io, m);
			io += m;
			n -= m;
		}
	}

	void process_block(float *io, size_t m)
	{
		const float *x48 = io;
		size_t n48 = m;
		if (resampled) {
			n48 = up->process(io, m, in48.data());
			x48 = in48.data();
		}
		if (stage)
			stage->push_input(x48, n48);
		assemble(x48, n48);

		if (!stage)
			return;  // captions only: the audio is not altered
		if (!resampled) {
			stage->render(io, m);
			return;
		}
		size_t need = down->input_needed(m);
		need = std::min(need, out48.size());
		stage->render(out48.data(), need);
		const size_t got = down->process(out48.data(), need, io);
		for (size_t i = got; i < m; ++i)  // cannot happen for host_rate <= 48 kHz; stay safe
			io[i] = got ? io[got - 1] : 0.0f;
	}

	void assemble(const float *x, size_t n)
	{
		while (n > 0) {
			const size_t take = std::min(n, kFrameSamples - fill);
			float *dst = cur.pcm + fill;
			for (size_t i = 0; i < take; ++i)  // a NaN/Inf from a misbehaving source must not reach the encoder
				dst[i] = std::isfinite(x[i]) ? x[i] : 0.0f;
			fill += take;
			x += take;
			n -= take;
			if (fill < kFrameSamples)
				continue;
			fill = 0;
			cur.seq = next_seq++;
			cur.ts_us = now_us();
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

			Outcome out = run_attempt();
			if (stopping())
				break;
			if (out.streamed_ms >= 30000) {
				backoff.reset();
				slow_backoff.reset();
			}
			if (out.give_up) {
				set_state(SessionState::Blocked, out.reason, out.message);
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
		if (code == "insufficient_funds") {
			out.give_up = true;
			out.message = "There is not enough balance to start live processing." + (msg.empty() ? std::string() : " " + msg) +
				      " Add balance on the website, then press Try again.";
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
			out.message = "Your account already has two live sessions running (other Audio Cleaner filters, or another "
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
				if (att->reason.empty())
					att->reason = "closed by server (code " + std::to_string(msg->closeInfo.code) +
						      (msg->closeInfo.reason.empty() ? "" : ", " + msg->closeInfo.reason) + ")";
				att->cv.notify_all();
				break;
			}
			case ix::WebSocketMessageType::Error: {
				std::lock_guard<std::mutex> lk(att->mu);
				att->closed = true;
				att->http_status = msg->errorInfo.http_status;
				att->reason = msg->errorInfo.reason;
				att->cv.notify_all();
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
			StartParams sp;
			sp.denoise = cfg.denoise;
			sp.captions = cfg.captions;
			sp.language = cfg.language;
			sp.atten_lim_db = atten_db.load();
			sp.codec = use_opus ? "opus" : "pcm_s16le";
			// Captions only: the returned audio is not used, so ask for the smallest stream.
			sp.opus_bitrate = use_opus ? (cfg.denoise ? cfg.opus_bitrate : kOpusMinBitrate) : 0;
			sp.client = cfg.client;
			sp.echo_reference = cfg.reference != nullptr;
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
		if (cfg.denoise && static_cast<double>(cfg.delay_ms) - total_latency_ms - kFramingMs <= 0.0) {
			// Even with a perfect network no processed frame could arrive in time.
			out.reason = "model latency " + std::to_string(static_cast<int>(total_latency_ms)) + " ms does not fit the " +
				     std::to_string(cfg.delay_ms) + " ms delay";
			out.message = "The " + preset_text() + " delay setting is too short for this server. Choose a longer one in "
				      "this filter's settings.";
			out.give_up = true;
			ws.stop();
			return out;
		}
		const bool send_ref = cfg.reference && ready.echo_reference;
		echo_active.store(send_ref);
		if (cfg.reference && !send_ref)
			log(LogLevel::Warning, "the server does not offer echo cancellation: continuing without it");
		const int latency = static_cast<int>(std::lround(ml * (kNetRate / 1000.0))) + opus_extra_latency;
		const int warm_frames = (latency + static_cast<int>(kFrameSamples) - 1) / static_cast<int>(kFrameSamples);
		if (stage && cfg.denoise)
			stage->set_model_latency(latency);
		{
			std::lock_guard<std::mutex> lk(st_mu);
			session_id = ready.session_id;
			usd_per_minute = ready.usd_per_minute;
			model_latency_ms = ready.model_latency_ms;
			low_balance = false;
			seconds_left = -1;
			warning.clear();
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
		std::unique_ptr<OpusEnc> ref_enc;  // the reference is a stream of its own
		if (use_opus && send_ref)
			ref_enc = std::make_unique<OpusEnc>(cfg.opus_bitrate);

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
				// The reference goes first, so the server has it when the microphone frame arrives.
				// (Silent reference frames are skipped; the microphone frames are never skipped, silent or
				// not: they are what keeps the session alive on the server.)
				if (send_ref && f->has_ref) {
					const size_t rlen = encode(ref_enc.get(), f->ref, kFlagReference);
					if (rlen && send(rlen))
						counters.reference_sent.fetch_add(1, std::memory_order_relaxed);
				}
				const size_t len = encode(enc.get(), f->pcm, 0);
				if (!len) {
					counters.dropped.fetch_add(1, std::memory_order_relaxed);
					queue.pop();
					continue;
				}
				if (static_cast<int64_t>(f->seq) >= att->first_valid_seq.load(std::memory_order_relaxed))
					store.mark_sent(f->seq);
				if (send(len))
					counters.sent.fetch_add(1, std::memory_order_relaxed);
				else
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
				if (cfg.denoise)
					update_late_window(window);
			}
			const int atten = pending_atten.exchange(-1);
			if (atten >= 0)
				ws.sendText(make_config_message(atten));

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

			std::unique_lock<std::mutex> lk(mu);
			cv.wait_for(lk, std::chrono::milliseconds(2), [&] { return stopping(); });
		}
		streaming.store(false, std::memory_order_release);
		echo_active.store(false);
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
			std::string stopped, code, msg, transport;
			int retry_after = 0;
			{
				std::lock_guard<std::mutex> lk(att->mu);
				stopped = att->stopped_reason;
				code = att->server_code;
				msg = att->server_msg;
				retry_after = att->retry_after_s;
				transport = att->reason;
			}
			if (!stopped.empty()) {
				out.reason = "server stopped the session: " + stopped;
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
				} else {
					out.min_wait_ms = 120000;
					out.message = "The server ended the session. Trying again in a moment.";
				}
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
		if (!cfg.denoise) {
			// Captions only: the audio that comes back is not used, so it is not decoded either.
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
		if (cfg.denoise)
			store.publish(h.seq, rx.data());
	}

	void on_text(Attempt &att, const std::string &text)
	{
		Event ev;
		if (!parse_event(text, ev))
			return;
		switch (ev.type) {
		case EventType::Ready: {
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
				emit_caption(ce);
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

void Session::process(float *mono, size_t n)
{
	impl_->process(mono, n);
}

double Session::delay_seconds() const
{
	return impl_->supported ? impl_->delay_sec : 0.0;
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
	s.insecure = d.insecure;
	s.codec = d.use_opus ? "opus" : "pcm_s16le";
	s.echo_requested = d.cfg.reference != nullptr;
	s.echo_active = d.echo_active.load();
	s.passthrough_only = !d.supported;
	s.delay_ms = d.supported ? d.delay_sec * 1000.0 : 0.0;
	s.captured = d.counters.captured.load();
	s.sent = d.counters.sent.load();
	s.received = d.counters.received.load();
	s.used = d.counters.used.load();
	s.late = d.counters.late.load();
	s.passthrough = d.counters.passthrough.load();
	s.dropped = d.counters.dropped.load();
	s.bad_frames = d.counters.bad_frames.load();
	s.warmup_discarded = d.counters.warmup_discarded.load();
	s.tx_bytes = d.counters.tx_bytes.load();
	s.rx_bytes = d.counters.rx_bytes.load();
	s.reconnects = d.counters.reconnects.load();
	s.link_too_slow = d.link_slow.load();
	s.late_percent = d.late_x10.load() / 10.0;
	return s;
}

std::string Session::status_line() const
{
	const Impl &d = *impl_;
	const SessionSnapshot s = snapshot();
	const bool cleaning = d.cfg.denoise && d.supported;
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
			if (!d.cfg.captions.empty())
				out += " Captions are running.";
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
	}
	if (s.state == SessionState::Streaming || s.captured > 0) {
		out += ", frames in time " + std::to_string(s.used) + ", late " + std::to_string(s.late) + ", unprocessed " +
		       std::to_string(s.passthrough);
		if (s.dropped)
			out += ", dropped " + std::to_string(s.dropped);
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
