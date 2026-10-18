// SPDX-License-Identifier: GPL-2.0-or-later
// A live Audio Cleaner session: ring buffers + reframer + resampler +
// WebSocket client thread + fixed-delay output stage. OBS independent.
#pragma once

#include "acw/caption_events.hpp"
#include "acw/common.hpp"
#include "acw/reference.hpp"

#include <cstdint>
#include <functional>
#include <limits>
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
	std::string url = "wss://audiocleaner.live/api/stream/ws";
	std::string api_key;               // never logged
	bool denoise = true;               // tier "light"
	std::string captions;              // "", "local" (standard, English) or "premium" (extra accurate)
	std::string language;              // "" = auto; "en,es" = someone who switches between these (up to three)
	std::vector<std::string> translate;  // 2-letter codes to translate the captions into (up to 3; needs captions)
	int atten_lim_db = 100;            // attenuation limit: 12 / 24 / 40 / 100
	int delay_ms = 150;                // fixed output delay D: 80 / 150 / 250
	Codec codec = Codec::Opus;
	int opus_bitrate = 64000;          // bits/s, both directions
	int host_rate = kNetRate;          // sample rate of the audio passed to process()
	std::string client = "obs-audio-cleaner/0.1.0";
	std::string ca_file;               // optional CA bundle path ("" = system trust store)
	int model_latency_samples = 1440;  // expected; replaced by the server's ready message
	// Echo cancellation: what the speakers play (the owner pushes to it, at host_rate), and whether to ask the server to
	// remove that echo from the audio (sent next to each microphone frame). Echo removal can be switched on and off
	// while the session runs (change()) as long as the buffer is there.
	std::shared_ptr<ReferenceBuffer> reference;
	bool echo = false;
};

// Blocked: the server refused the session for a reason only the user can fix (wrong key, no balance,
// unsupported settings, ...). The session does not retry until retry_now() is called or it is replaced.
enum class SessionState { Idle, Connecting, Handshaking, Streaming, Backoff, Stopped, Blocked };

const char *to_string(SessionState s);

struct SessionSnapshot {
	SessionState state = SessionState::Idle;
	std::string detail;            // technical reason for the current state (for logs and the details line)
	std::string message;           // the same in plain words for the user ("" = nothing to say)
	std::string session_id;
	bool insecure = false;         // ws:// URL
	bool passthrough_only = false; // unsupported configuration: audio is not altered
	std::string codec;             // codec actually in use: "opus" or "pcm_s16le"
	bool echo_requested = false;   // echo removal is asked for
	bool echo_active = false;      // ... and the server accepted it
	double delay_ms = 0;           // total delay applied to the audio (0 when captions only)
	double model_latency_ms = 0;
	double usd_per_minute = 0;
	double rtt_last_ms = 0, rtt_avg_ms = 0, rtt_p95_ms = 0, rtt_max_ms = 0;
	double ping_rtt_ms = 0;
	uint64_t captured = 0, sent = 0, received = 0, used = 0, late = 0, passthrough = 0, dropped = 0;
	uint64_t silent_sent = 0;      // of `sent`: silence frames (nothing encoded, nothing billed)
	uint64_t net_loops = 0;        // how many times the network thread woke up while streaming
	bool silence_active = false;   // the service takes silence frames
	uint64_t bad_frames = 0, warmup_discarded = 0;
	uint64_t tx_bytes = 0, rx_bytes = 0;  // audio traffic (frame headers included)
	uint32_t reconnects = 0;
	int backoff_seconds = 0;       // when state == Backoff: seconds until the next attempt
	bool low_balance = false;
	int seconds_left = -1;
	std::string warning;           // last warning text (e.g. captions unavailable)
	std::string reason_code;       // why the session is not streaming ("no_audio", "trial_over", ...; "" = no particular reason)
	std::string reason_arg;        // a detail for that reason (the service's own message)
	// Translations: asked for, switched on by the service (the `ready` message; known only when it said so) and the
	// ones it gave up on (a `translation_unavailable` warning).
	std::vector<std::string> translate_requested, translate_active, translate_failed;
	bool translate_active_known = false;
	bool captions_failed = false;  // a `captions_unavailable` warning: the captions stopped (the audio goes on)
	// A change of settings the service could not make (its message): the session goes on as it was. "" = none.
	std::string change_refused;
	// Recent frames (last few seconds) that came back too late to be used. link_too_slow: most of them did,
	// i.e. the connection (or the server) cannot keep up with the chosen delay.
	bool link_too_slow = false;
	double late_percent = 0;
	// "Test my room" (request_room_test()): the noise figure the service measured at its noise remover (dB between the
	// voice and what it takes out), NaN until it is known; or why there is none (the service's code and message).
	double room_noise_db = std::numeric_limits<double>::quiet_NaN();
	std::string room_code, room_message;
};

enum class LogLevel { Debug, Info, Warning, Error };

class Session {
public:
	using CaptionCallback = std::function<void(const CaptionEvent &)>;
	using TranslationCallback = std::function<void(const TranslationEvent &)>;
	using LogCallback = std::function<void(LogLevel, const std::string &)>;
	// Plain-words texts of the user's language: gets an id ("reason.no_audio", "warning.translation_unavailable", ...)
	// and returns the text ("%1" is replaced by a detail such as a language name), or "" to use the built-in English.
	using Localizer = std::function<std::string(const std::string &id)>;

	Session(const SessionConfig &cfg, CaptionCallback on_caption, LogCallback on_log);
	~Session();
	Session(const Session &) = delete;
	Session &operator=(const Session &) = delete;

	// Where translations go. Call before start(); runs on the network thread, so it must return quickly.
	void set_translation_callback(TranslationCallback cb);

	// Starts the network thread (idempotent). The thread idles until audio flows, so starting early is free.
	// Not for the audio thread (it creates a thread).
	void start();

	// Stops the session. flush_ms > 0 waits that long for the server to flush
	// outstanding captions after the `stop` message. Never blocks longer than
	// flush_ms + join_timeout_ms; a stuck thread is detached.
	void stop(int flush_ms = 0, int join_timeout_ms = 800);

	bool started() const;

	// Audio thread: processes `n` mono samples at cfg.host_rate in place.
	// Wait-free; no allocation, locks or system calls (except the clock).
	// `muted`: the source is muted in OBS (its audio is not mixed in): what is sent to the service for these samples is
	// silence, so a muted microphone is not captioned, and with a service that takes silence frames it costs nothing.
	void process(float *mono, size_t n, bool muted = false);

	// Delay (seconds) that process() applies to the audio; use it to back-date
	// output timestamps. 0 when the audio is not altered.
	double delay_seconds() const;

	// Asks the network thread to finish, without waiting for it (any thread, wait-free apart from a mutex).
	// stop() still has to be called to join it. Use it to end billing at once.
	void request_stop();

	// Live strength change (sent to the server as a `config` message).
	void set_atten_lim_db(int db);

	// "Test my room": asks the service to measure the noise at its noise remover over the next `seconds` (5-60) of audio
	// (a `room_test` message, sent again if the connection is made anew before the answer). Forgets the last result;
	// snapshot() has the new one when it comes. Needs noise removal on: otherwise the answer is room_code
	// "needs_noise_removal".
	void request_room_test(int seconds = 20);

	// Changes what the session does (noise removal, captions, their language, translations, echo removal) without a
	// new session: the audio goes on and the billing follows from that moment. False when `next` needs a new session:
	// a setting that isn't one of those differs, or noise removal is switched on in a session that started without it
	// (its audio was never delayed, so there is no room for the cleaned audio). If the service refuses the change, the
	// session goes on as it was and snapshot().change_refused says why.
	bool change(const SessionConfig &next);

	// Blocked sessions try again now (after the user fixed the problem). Otherwise a no-op.
	void retry_now();

	SessionSnapshot snapshot() const;
	// One or two plain sentences for the user.
	std::string status_line(const Localizer &loc = nullptr) const;
	// Technical line: round-trip time, frame counters, reconnects, codec.
	std::string status_details() const;

	// Round-trip times of the most recent frames (ms), oldest first.
	std::vector<float> rtt_history() const;

private:
	struct Impl;
	std::shared_ptr<Impl> impl_;
};

// True when the server address may be used: wss:// to anywhere, ws:// (unencrypted) only to this computer
// or the local network (localhost, 127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x, *.local, ::1).
bool server_url_allowed(const std::string &url);

// Name of a delay preset (80 / 150 / 250 ms): "Low", "Balanced", "Safe".
const char *latency_preset_name(int delay_ms);

// Waits up to `ms` for network threads that stop() had to detach. Call when the plugin unloads, before
// its code can go away. True when none is left.
bool wait_for_background_threads(int ms);

}  // namespace acw
