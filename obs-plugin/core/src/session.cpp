// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/session.hpp"

#include "acw/backoff.hpp"
#include "acw/opus_codec.hpp"
#include "acw/output_stage.hpp"
#include "acw/protocol.hpp"
#include "acw/resampler.hpp"
#include "acw/spsc.hpp"
#include "acw/stats.hpp"

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
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

bool is_secure_url(const std::string &u)
{
	return u.rfind("wss://", 0) == 0;
}
bool is_plain_url(const std::string &u)
{
	return u.rfind("ws://", 0) == 0;
}

#if defined(ACW_TLS_MBEDTLS) && !defined(_WIN32)
// mbedTLS has no system trust store: use the distribution's CA bundle.
std::string find_ca_bundle()
{
	static const char *candidates[] = {
		"/etc/ssl/certs/ca-certificates.crt",  // Debian, Ubuntu
		"/etc/pki/tls/certs/ca-bundle.crt",    // Fedora, RHEL
		"/etc/ssl/ca-bundle.pem",              // openSUSE
		"/etc/ssl/cert.pem",                   // Alpine, macOS
	};
	for (const char *c : candidates)
		if (std::ifstream(c).good())
			return c;
	return {};
}
#endif

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
		std::string stopped_reason;        // `stopped` event
		Event ready_ev;
		std::unique_ptr<OpusDec> opus_dec;  // used by the IX thread only
		std::atomic<int64_t> first_valid_seq{INT64_MAX};
		std::atomic<uint64_t> last_rx_us{0};
	};

	struct Outcome {
		std::string reason;
		int min_wait_ms = 0;
		int streamed_ms = 0;
		bool immediate = false;  // reconnect without waiting (server restarted the session)
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

	// ---- status ------------------------------------------------------------
	mutable std::mutex st_mu;
	SessionState state = SessionState::Idle;
	std::string detail;
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
		set_state(SessionState::Idle, supported ? "waiting for audio" : "unsupported sample rate: audio passed through");
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

	void set_state(SessionState s, const std::string &d = std::string())
	{
		std::lock_guard<std::mutex> lk(st_mu);
		state = s;
		detail = d;
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
		if (last_process_us.exchange(now_us(), std::memory_order_relaxed) == 0)
			cv.notify_all();  // first audio: let the network thread connect right away
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
			std::memcpy(cur.pcm + fill, x, take * sizeof(float));
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
			queue.commit();
			cv.notify_one();
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

	void net_main()
	{
		Backoff backoff(std::chrono::milliseconds(1000), std::chrono::milliseconds(60000));
		bool first = true;
		while (!stopping()) {
			// Connect lazily: only once the host actually feeds audio.
			if (!audio_is_flowing()) {
				set_state(SessionState::Idle, "waiting for audio");
				sleep_interruptible(std::chrono::milliseconds(100));
				continue;
			}
			if (!first)
				counters.reconnects.fetch_add(1, std::memory_order_relaxed);
			first = false;

			Outcome out = run_attempt();
			if (stopping())
				break;
			if (out.streamed_ms >= 30000)
				backoff.reset();
			std::chrono::milliseconds wait = out.immediate ? std::chrono::milliseconds(500) : backoff.next();
			wait = std::max(wait, std::chrono::milliseconds(out.min_wait_ms));
			{
				std::lock_guard<std::mutex> lk(st_mu);
				state = SessionState::Backoff;
				detail = out.reason;
				backoff_s = static_cast<int>((wait.count() + 999) / 1000);
			}
			log(LogLevel::Warning, "connection lost (" + out.reason + "); retrying in " + std::to_string(backoff_s) +
						       " s, audio is passed through meanwhile");
			sleep_interruptible(wait);
		}
		streaming.store(false);
		set_state(SessionState::Stopped, "stopped");
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

	static bool is_fatal_code(const std::string &c)
	{
		return c == "insufficient_funds" || c == "tos" || c == "bad_config" || c == "bad_tier" ||
		       c == "nothing_enabled" || c == "bad_start";
	}

	Outcome run_attempt()
	{
		Outcome out;
		auto att = std::make_shared<Attempt>();
		att->last_rx_us.store(now_us());
		set_state(SessionState::Connecting, "connecting to " + cfg.url);

		ix::WebSocket ws;
		ws.setUrl(cfg.url);
		ix::WebSocketHttpHeaders headers;
		headers["Authorization"] = "Bearer " + cfg.api_key;  // never logged
		headers["User-Agent"] = cfg.client;
		ws.setExtraHeaders(headers);
		ws.disableAutomaticReconnection();  // we run our own backoff
		ws.disablePerMessageDeflate();
		ws.setHandshakeTimeout(10);
		{
			ix::SocketTLSOptions tls;
			tls.caFile = cfg.ca_file.empty() ? "SYSTEM" : cfg.ca_file;  // certificate + hostname are always verified
#if defined(ACW_TLS_MBEDTLS) && !defined(_WIN32)
			if (cfg.ca_file.empty()) {
				const std::string bundle = find_ca_bundle();
				if (!bundle.empty())
					tls.caFile = bundle;
			}
#endif
			ws.setTLSOptions(tls);
		}

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
			if (att->http_status == 401 || att->http_status == 403) {
				out.min_wait_ms = 60000;
				out.reason = "authentication failed (HTTP " + std::to_string(att->http_status) +
					     "): check the API key and that it has the 'stream' scope";
			} else if (att->http_status == 429) {
				out.min_wait_ms = 60000;
			}
			ws.stop();
			return out;
		}

		// ---- start / ready --------------------------------------------------------
		set_state(SessionState::Handshaking, "starting session");
		{
			StartParams sp;
			sp.denoise = cfg.denoise;
			sp.captions = cfg.captions;
			sp.language = cfg.language;
			sp.atten_lim_db = cfg.atten_lim_db;
			sp.codec = use_opus ? "opus" : "pcm_s16le";
			sp.opus_bitrate = use_opus ? cfg.opus_bitrate : 0;
			sp.client = cfg.client;
			sp.echo_reference = cfg.reference != nullptr;
			if (!ws.sendText(make_start_message(sp)).success) {
				out.reason = "could not send start message";
				ws.stop();
				return out;
			}
		}
		if (!waits_for(*att, std::chrono::seconds(30), [&] { return att->ready || att->closed; }, stop) ||
		    !att->ready) {
			out.reason = describe_failure(*att, "server did not answer the start message");
			if (!att->server_code.empty()) {
				// "full": the server has no room for this setup right now; the
				// message lists what is still available. Retry slowly.
				out.min_wait_ms = is_fatal_code(att->server_code) || att->server_code == "full" ? 60000 : 15000;
				out.reason = "server: " + att->server_code + (att->server_msg.empty() ? "" : " - " + att->server_msg);
			}
			ws.stop();
			return out;
		}
		const Event ready = att->ready_ev;
		const bool send_ref = cfg.reference && ready.echo_reference;
		echo_active.store(send_ref);
		if (cfg.reference && !send_ref)
			log(LogLevel::Warning, "the server does not offer echo cancellation: continuing without it");
		if (ready.sample_rate != 0 && ready.sample_rate != kNetRate) {
			out.reason = "server sample rate " + std::to_string(ready.sample_rate) + " is not supported";
			out.min_wait_ms = 60000;
			ws.stop();
			return out;
		}
		const int latency =
			static_cast<int>(std::lround(ready.model_latency_ms * (kNetRate / 1000.0))) + opus_extra_latency;
		const int warm_frames = static_cast<int>((latency + kFrameSamples - 1) / kFrameSamples);
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
			const int atten = pending_atten.exchange(-1);
			if (atten >= 0)
				ws.sendText(make_config_message(atten));

			if (now_us() - att->last_rx_us.load(std::memory_order_relaxed) > uint64_t(kRxWatchdogMs) * 1000) {
				out.reason = "no data from the server for " + std::to_string(kRxWatchdogMs / 1000) + " s";
				break;
			}
			if (now_us() - last_frame_us > uint64_t(kIdleDisconnectMs) * 1000 && !audio_is_flowing()) {
				// The host stopped feeding audio (filter muted/disabled upstream): close quietly.
				out.reason = "no audio from OBS";
				out.immediate = true;
				out.min_wait_ms = 0;
				break;
			}

			std::unique_lock<std::mutex> lk(mu);
			cv.wait_for(lk, std::chrono::milliseconds(4), [&] { return stopping() || queue.front() != nullptr; });
		}
		streaming.store(false, std::memory_order_release);
		echo_active.store(false);
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
			std::lock_guard<std::mutex> lk(att->mu);
			if (!att->stopped_reason.empty()) {
				out.reason = "server stopped the session: " + att->stopped_reason;
				if (att->stopped_reason == "max_duration")
					out.immediate = true;
				else
					out.min_wait_ms = 120000;
			} else if (!att->server_code.empty()) {
				out.reason = "server: " + att->server_code + (att->server_msg.empty() ? "" : " - " + att->server_msg);
				out.min_wait_ms = is_fatal_code(att->server_code) ? 60000 : 5000;
			} else {
				out.reason = att->reason.empty() ? "connection closed" : att->reason;
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
		if (h.codec == kCodecOpus) {
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
				msg = ev.code + (ev.message.empty() ? "" : ": " + ev.message);
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
	if (is_secure_url(d.cfg.url) == false && !is_plain_url(d.cfg.url)) {
		d.set_state(SessionState::Stopped, "server URL must start with wss:// (or ws:// for local testing)");
		d.stop_req.store(true);
		std::lock_guard<std::mutex> lk(d.done_mu);
		d.net_done = true;
		return;
	}
	if (d.cfg.api_key.empty()) {
		d.set_state(SessionState::Stopped, "no API key configured");
		d.stop_req.store(true);
		std::lock_guard<std::mutex> lk(d.done_mu);
		d.net_done = true;
		return;
	}
	if (d.insecure)
		d.log(LogLevel::Warning, "the server URL uses ws:// (unencrypted): the API key and audio travel in clear text; "
					 "use this for local testing only");
	try {
		auto self = impl_;
		d.th = std::thread([self] { self->net_main(); });
	} catch (const std::exception &e) {
		d.set_state(SessionState::Stopped, std::string("could not start network thread: ") + e.what());
		std::lock_guard<std::mutex> lk(d.done_mu);
		d.net_done = true;
	}
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
	impl_->cfg.atten_lim_db = db;  // used by the next (re)connect
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
	return s;
}

std::string Session::status_line() const
{
	const SessionSnapshot s = snapshot();
	char buf[64];
	std::string out;
	switch (s.state) {
	case SessionState::Streaming:
		out = "Connected";
		if (s.rtt_last_ms > 0 || s.received > 0) {
			std::snprintf(buf, sizeof buf, ", RTT %.0f ms (p95 %.0f)", s.rtt_avg_ms, s.rtt_p95_ms);
			out += buf;
		}
		break;
	case SessionState::Backoff:
		out = "Disconnected - audio passes through; retrying in " + std::to_string(s.backoff_seconds) + " s (" + s.detail + ")";
		break;
	case SessionState::Connecting:
	case SessionState::Handshaking:
		out = "Connecting...";
		break;
	case SessionState::Idle:
		out = s.detail.empty() ? "Idle" : "Idle (" + s.detail + ")";
		break;
	case SessionState::Stopped:
		out = "Stopped (" + s.detail + ")";
		break;
	}
	if (s.delay_ms > 0) {
		std::snprintf(buf, sizeof buf, " | delay %.0f ms", s.delay_ms);
		out += buf;
	}
	if (s.echo_active)
		out += " | echo cancellation on";
	if (s.state == SessionState::Streaming || s.captured > 0) {
		out += " | frames ok " + std::to_string(s.used) + ", late " + std::to_string(s.late) + ", passthrough " +
		       std::to_string(s.passthrough);
		if (s.dropped)
			out += ", dropped " + std::to_string(s.dropped);
	}
	if (s.reconnects)
		out += " | reconnects " + std::to_string(s.reconnects);
	if (s.insecure)
		out += " | WARNING: unencrypted ws:// connection";
	if (s.low_balance) {
		out += " | LOW BALANCE";
		if (s.seconds_left >= 0)
			out += " (~" + std::to_string(s.seconds_left) + " s left)";
	}
	if (!s.warning.empty())
		out += " | " + s.warning;
	if (s.passthrough_only)
		out += " | audio not altered: unsupported sample rate";
	return out;
}

}  // namespace acw
