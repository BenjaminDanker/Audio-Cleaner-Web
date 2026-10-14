// SPDX-License-Identifier: GPL-2.0-or-later
// Live streaming protocol v1 (docs/streaming-protocol.md): framing and JSON.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace acw {

struct FrameHeader {
	uint8_t version = 0;
	uint8_t codec = 0;
	uint16_t flags = 0;
	uint32_t seq = 0;
	uint64_t ts_us = 0;
};

// Binary audio frames --------------------------------------------------------

// Writes header + pcm_s16le payload; `out` must hold kHeaderSize + 2 * n bytes.
// Returns the number of bytes written.
size_t encode_frame_s16(uint32_t seq, uint64_t ts_us, const float *pcm, size_t n, uint8_t *out);

// Writes only the 16-byte header (any codec, e.g. Opus where the payload follows).
void encode_frame_header(uint8_t codec, uint32_t seq, uint64_t ts_us, uint8_t *out);

// Parses the 16-byte header; false when too short or the version is unknown.
bool parse_frame_header(const uint8_t *data, size_t len, FrameHeader &hdr);

// Parses a frame (pcm_s16le or f32le). Returns false on malformed input.
// `out` receives at most `max_samples`; `n_samples` is the payload length.
bool decode_frame(const uint8_t *data, size_t len, FrameHeader &hdr, float *out, size_t max_samples,
		  size_t &n_samples);

// JSON messages --------------------------------------------------------------

struct StartParams {
	bool denoise = true;                 // tier "light", or "low_latency" when low_latency is set
	bool low_latency = false;
	std::string captions;                // "", "local" or "premium"
	std::string language;                // "" = auto
	int atten_lim_db = 100;
	std::string codec = "pcm_s16le";     // "opus" or "pcm_s16le"
	int opus_bitrate = 0;                // downlink bitrate requested from the server (opus only; 0 = server default)
	std::string client;                  // e.g. "obs-audio-cleaner/0.1.0"
};

std::string make_start_message(const StartParams &p);
std::string make_ping_message(uint64_t t);
std::string make_config_message(int atten_lim_db);
std::string make_stop_message();

enum class EventType { Unknown, Ready, Caption, Warning, Stopped, Error, Pong };

struct Event {
	EventType type = EventType::Unknown;
	// ready
	std::string session_id;
	double model_latency_ms = 0.0;
	double usd_per_minute = 0.0;
	int hop_samples = 0;
	int sample_rate = 0;
	int max_session_seconds = 0;
	// caption
	std::string text;
	bool is_final = false;
	double start = 0.0, end = 0.0;
	std::string caption_id;
	// warning / error / stopped
	std::string code;      // warning/error code, or stopped.reason
	std::string message;
	int seconds_left = -1;
	// pong
	uint64_t pong_t = 0;
	bool has_pong_t = false;
};

bool parse_event(const std::string &text, Event &ev);

}  // namespace acw
