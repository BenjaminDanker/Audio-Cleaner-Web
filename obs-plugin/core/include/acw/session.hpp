// SPDX-License-Identifier: GPL-2.0-or-later
// A live Audio Cleaner session: ring buffers + reframer + resampler +
// WebSocket client thread + fixed-delay output stage. OBS independent.
#pragma once

#include "acw/common.hpp"
#include "acw/reference.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace acw {

// Audio codec on the wire (docs/streaming-protocol.md section 2).
enum class Codec {
	Opus,   // one 20 ms Opus packet per frame, ~60 kbps each way (default)
	Pcm16,  // pcm_s16le, ~770 kbps each way (fallback)
};

struct SessionConfig {
	std::string url = "wss://mc-multi-world.space/audio/api/stream/ws";
	std::string api_key;               // never logged
	bool denoise = true;               // tier "light"
	std::string captions;              // "", "local" or "premium"
	std::string language;              // "" = auto
	int atten_lim_db = 100;            // attenuation limit: 18 / 40 / 100
	int delay_ms = 150;                // fixed output delay D: 80 / 150 / 250
	Codec codec = Codec::Opus;
	int opus_bitrate = 64000;          // bits/s, both directions
	int host_rate = kNetRate;          // sample rate of the audio passed to process()
	std::string client = "obs-audio-cleaner/0.1.0";
	std::string ca_file;               // optional CA bundle path ("" = system trust store)
	int model_latency_samples = 1440;  // expected; replaced by the server's ready message
	// Echo cancellation: what the speakers play (the owner pushes to it, at host_rate).
	// Set = ask the server to remove that echo from the audio; sent next to each microphone frame.
	std::shared_ptr<ReferenceBuffer> reference;
};

enum class SessionState { Idle, Connecting, Handshaking, Streaming, Backoff, Stopped };

const char *to_string(SessionState s);

struct SessionSnapshot {
	SessionState state = SessionState::Idle;
	std::string detail;            // last error / reason for the current state
	std::string session_id;
	bool insecure = false;         // ws:// URL
	bool passthrough_only = false; // unsupported configuration: audio is not altered
	std::string codec;             // codec actually in use: "opus" or "pcm_s16le"
	bool echo_requested = false;   // an echo reference is configured
	bool echo_active = false;      // ... and the server accepted it
	double delay_ms = 0;           // total delay applied to the audio (0 when captions only)
	double model_latency_ms = 0;
	double usd_per_minute = 0;
	double rtt_last_ms = 0, rtt_avg_ms = 0, rtt_p95_ms = 0, rtt_max_ms = 0;
	double ping_rtt_ms = 0;
	uint64_t captured = 0, sent = 0, received = 0, used = 0, late = 0, passthrough = 0, dropped = 0;
	uint64_t bad_frames = 0, warmup_discarded = 0;
	uint64_t tx_bytes = 0, rx_bytes = 0;  // audio traffic (frame headers included)
	uint32_t reconnects = 0;
	int backoff_seconds = 0;       // when state == Backoff: seconds until the next attempt
	bool low_balance = false;
	int seconds_left = -1;
	std::string warning;           // last warning text (e.g. captions unavailable)
};

struct CaptionEvent {
	std::string text;
	bool is_final = false;
	double start = 0, end = 0;
};

enum class LogLevel { Debug, Info, Warning, Error };

class Session {
public:
	using CaptionCallback = std::function<void(const CaptionEvent &)>;
	using LogCallback = std::function<void(LogLevel, const std::string &)>;

	Session(const SessionConfig &cfg, CaptionCallback on_caption, LogCallback on_log);
	~Session();
	Session(const Session &) = delete;
	Session &operator=(const Session &) = delete;

	// Starts the network thread (idempotent, cheap: safe on the audio thread).
	void start();

	// Stops the session. flush_ms > 0 waits that long for the server to flush
	// outstanding captions after the `stop` message. Never blocks longer than
	// flush_ms + join_timeout_ms; a stuck thread is detached.
	void stop(int flush_ms = 0, int join_timeout_ms = 800);

	bool started() const;

	// Audio thread: processes `n` mono samples at cfg.host_rate in place.
	// Wait-free; no allocation, locks or system calls (except the clock).
	void process(float *mono, size_t n);

	// Delay (seconds) that process() applies to the audio; use it to back-date
	// output timestamps. 0 when the audio is not altered.
	double delay_seconds() const;

	// Live strength change (sent to the server as a `config` message).
	void set_atten_lim_db(int db);

	SessionSnapshot snapshot() const;
	std::string status_line() const;

	// Round-trip times of the most recent frames (ms), oldest first.
	std::vector<float> rtt_history() const;

private:
	struct Impl;
	std::shared_ptr<Impl> impl_;
};

}  // namespace acw
