// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/protocol.hpp"

#include "acw/common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <nlohmann/json.hpp>

namespace acw {

namespace {

inline void put_u16(uint8_t *p, uint16_t v)
{
	p[0] = static_cast<uint8_t>(v & 0xff);
	p[1] = static_cast<uint8_t>(v >> 8);
}
inline void put_u32(uint8_t *p, uint32_t v)
{
	for (int i = 0; i < 4; ++i)
		p[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xff);
}
inline void put_u64(uint8_t *p, uint64_t v)
{
	for (int i = 0; i < 8; ++i)
		p[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xff);
}
inline uint32_t get_u32(const uint8_t *p)
{
	return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
	       (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t get_u64(const uint8_t *p)
{
	uint64_t v = 0;
	for (int i = 7; i >= 0; --i)
		v = (v << 8) | p[i];
	return v;
}

}  // namespace

size_t encode_frame_s16(uint32_t seq, uint64_t ts_us, const float *pcm, size_t n, uint8_t *out, uint16_t flags)
{
	out[0] = kProtoVersion;
	out[1] = kCodecS16;
	put_u16(out + 2, flags);
	put_u32(out + 4, seq);
	put_u64(out + 8, ts_us);
	uint8_t *p = out + kHeaderSize;
	for (size_t i = 0; i < n; ++i) {
		float x = pcm[i];
		if (!(x == x))  // NaN guard
			x = 0.0f;
		x = std::min(1.0f, std::max(-1.0f, x));
		const int v = static_cast<int>(std::lrintf(x * 32767.0f));
		put_u16(p + 2 * i, static_cast<uint16_t>(static_cast<int16_t>(v)));
	}
	return kHeaderSize + 2 * n;
}

void encode_frame_header(uint8_t codec, uint32_t seq, uint64_t ts_us, uint8_t *out, uint16_t flags)
{
	out[0] = kProtoVersion;
	out[1] = codec;
	put_u16(out + 2, flags);
	put_u32(out + 4, seq);
	put_u64(out + 8, ts_us);
}

size_t encode_frame_silence(uint32_t seq, uint64_t ts_us, size_t samples, uint8_t *out, uint16_t flags)
{
	if (samples < kSilenceMinSamples || samples > kMaxRxSamples || samples % kSilenceStep != 0)
		return 0;
	encode_frame_header(kCodecSilence, seq, ts_us, out, flags);
	put_u16(out + kHeaderSize, static_cast<uint16_t>(samples));
	return kHeaderSize + kSilencePayload;
}

bool parse_silence_samples(const uint8_t *data, size_t len, size_t &samples)
{
	FrameHeader h;
	if (len != kHeaderSize + kSilencePayload || !parse_frame_header(data, len, h) || h.codec != kCodecSilence)
		return false;
	const size_t n = static_cast<size_t>(data[kHeaderSize]) | (static_cast<size_t>(data[kHeaderSize + 1]) << 8);
	if (n < kSilenceMinSamples || n > kMaxRxSamples || n % kSilenceStep != 0)
		return false;
	samples = n;
	return true;
}

bool parse_frame_header(const uint8_t *data, size_t len, FrameHeader &hdr)
{
	if (len < kHeaderSize)
		return false;
	hdr.version = data[0];
	hdr.codec = data[1];
	hdr.flags = static_cast<uint16_t>(data[2] | (data[3] << 8));
	hdr.seq = get_u32(data + 4);
	hdr.ts_us = get_u64(data + 8);
	return hdr.version == kProtoVersion;
}

bool decode_frame(const uint8_t *data, size_t len, FrameHeader &hdr, float *out, size_t max_samples,
		  size_t &n_samples)
{
	if (!parse_frame_header(data, len, hdr))
		return false;
	const uint8_t *p = data + kHeaderSize;
	const size_t payload = len - kHeaderSize;
	if (hdr.codec == kCodecSilence) {
		size_t n = 0;
		if (!parse_silence_samples(data, len, n) || n > max_samples)
			return false;
		std::fill(out, out + n, 0.0f);
		n_samples = n;
		return true;
	}
	if (hdr.codec == kCodecS16) {
		if (payload % 2)
			return false;
		n_samples = payload / 2;
		if (n_samples > max_samples)
			return false;
		for (size_t i = 0; i < n_samples; ++i) {
			const int16_t v = static_cast<int16_t>(static_cast<uint16_t>(p[2 * i] | (p[2 * i + 1] << 8)));
			out[i] = static_cast<float>(v) * (1.0f / 32768.0f);
		}
		return true;
	}
	if (hdr.codec == kCodecF32) {
		if (payload % 4)
			return false;
		n_samples = payload / 4;
		if (n_samples > max_samples)
			return false;
		for (size_t i = 0; i < n_samples; ++i) {
			const uint32_t u = get_u32(p + 4 * i);
			float f;
			static_assert(sizeof(f) == sizeof(u), "float must be 32 bit");
			std::memcpy(&f, &u, sizeof(f));
			out[i] = std::isfinite(f) ? f : 0.0f;
		}
		return true;
	}
	return false;
}

namespace {

// "" -> null, "en" -> "en", "en,es" -> ["en","es"]: someone who switches between languages (extra-accurate captions
// follow the switches), up to three.
nlohmann::json language_json(const std::string &language)
{
	using nlohmann::json;
	if (language.empty())
		return json(nullptr);
	if (language.find(',') == std::string::npos)
		return json(language);
	json list = json::array();
	size_t from = 0;
	while (from <= language.size()) {
		const size_t comma = language.find(',', from);
		const std::string one = language.substr(from, comma == std::string::npos ? std::string::npos : comma - from);
		if (!one.empty() && list.size() < 3)
			list.push_back(one);
		if (comma == std::string::npos)
			break;
		from = comma + 1;
	}
	return list;
}

}  // namespace

std::string make_change_message(const Features &f, const std::string &id)
{
	using nlohmann::json;
	json j;
	j["type"] = "config";
	j["id"] = id;
	j["denoise"] = f.denoise ? json("light") : json(nullptr);
	j["captions"] = f.captions == "local" || f.captions == "premium" ? json(f.captions) : json(nullptr);
	j["language"] = language_json(f.language);
	j["translate"] = f.captions.empty() ? std::vector<std::string>{} : f.translate;
	j["echoReference"] = f.echo_reference;
	return j.dump();
}

std::string make_start_message(const StartParams &p)
{
	using nlohmann::json;
	json j;
	j["type"] = "start";
	j["version"] = 1;
	j["denoise"] = p.denoise ? json("light") : json(nullptr);
	if (p.captions == "local" || p.captions == "premium")
		j["captions"] = p.captions;
	else
		j["captions"] = nullptr;
	j["language"] = language_json(p.language);
	j["sampleRate"] = kNetRate;
	j["codec"] = p.codec;
	if (p.codec == "opus" && p.opus_bitrate > 0)
		j["opusBitrate"] = p.opus_bitrate;
	j["attenLimDb"] = p.atten_lim_db;
	if (p.echo_reference)
		j["echoReference"] = true;
	if (!p.translate.empty())
		j["translate"] = p.translate;
	if (!p.return_audio)
		j["returnAudio"] = false;  // captions only: no audio frames come back
	j["client"] = p.client;
	return j.dump();
}

std::string make_ping_message(uint64_t t)
{
	nlohmann::json j;
	j["type"] = "ping";
	j["t"] = t;
	return j.dump();
}

std::string make_config_message(int atten_lim_db)
{
	nlohmann::json j;
	j["type"] = "config";
	j["attenLimDb"] = atten_lim_db;
	return j.dump();
}

std::string make_stop_message()
{
	return "{\"type\":\"stop\"}";
}

bool parse_event(const std::string &text, Event &ev)
{
	using nlohmann::json;
	json j = json::parse(text, nullptr, /*allow_exceptions=*/false);
	if (!j.is_object())
		return false;
	auto str = [&](const char *k, std::string &dst) {
		auto it = j.find(k);
		if (it != j.end() && it->is_string())
			dst = it->get<std::string>();
	};
	// Numbers from the network are never trusted: the session checks the doubles it uses (they may be
	// infinite), integers are clamped here (casting an out-of-range double to int is undefined behaviour).
	auto num = [&](const char *k, double &dst) {
		auto it = j.find(k);
		if (it != j.end() && it->is_number())
			dst = it->get<double>();
	};
	auto integer = [&](const char *k, int &dst) {
		auto it = j.find(k);
		if (it != j.end() && it->is_number()) {
			const double v = it->get<double>();
			if (std::isfinite(v))
				dst = static_cast<int>(std::min(1e9, std::max(-1e9, v)));
		}
	};

	std::string type;
	str("type", type);
	ev = Event{};
	if (type == "ready") {
		ev.type = EventType::Ready;
		str("sessionId", ev.session_id);
		num("modelLatencyMs", ev.model_latency_ms);
		num("usdPerMinute", ev.usd_per_minute);
		integer("hopSamples", ev.hop_samples);
		integer("sampleRate", ev.sample_rate);
		integer("maxSessionSeconds", ev.max_session_seconds);
		if (auto it = j.find("echoReference"); it != j.end() && it->is_boolean())
			ev.echo_reference = it->get<bool>();
		if (auto it = j.find("silenceFrames"); it != j.end() && it->is_boolean())
			ev.silence_frames = it->get<bool>();
		if (auto it = j.find("translate"); it != j.end() && it->is_array()) {
			ev.has_translate = true;
			for (const auto &e : *it)
				if (e.is_string() && ev.translate.size() < 8)
					ev.translate.push_back(e.get<std::string>());
		}
	} else if (type == "translation") {
		ev.type = EventType::Translation;
		str("lang", ev.lang);
		str("text", ev.text);
		str("of", ev.of);
		num("start", ev.start);
		num("end", ev.end);
		ev.is_final = true;
	} else if (type == "caption") {
		ev.type = EventType::Caption;
		str("text", ev.text);
		auto it = j.find("final");
		ev.is_final = it != j.end() && it->is_boolean() && it->get<bool>();
		num("start", ev.start);
		num("end", ev.end);
		str("id", ev.caption_id);
	} else if (type == "warning") {
		ev.type = EventType::Warning;
		str("code", ev.code);
		str("lang", ev.lang);
		str("message", ev.message);
		integer("secondsLeft", ev.seconds_left);
	} else if (type == "stopped") {
		ev.type = EventType::Stopped;
		str("reason", ev.code);
	} else if (type == "error") {
		ev.type = EventType::Error;
		str("code", ev.code);
		str("message", ev.message);
		int ra = 0;
		integer("retryAfter", ra);  // seconds; the server may add it to any error
		integer("retryAfterSeconds", ra);
		ev.retry_after_s = std::min(3600, std::max(0, ra));
	} else if (type == "config") {
		ev.type = EventType::Config;
		str("id", ev.id);
		str("code", ev.code);
		str("message", ev.message);
		num("usdPerMinute", ev.usd_per_minute);
		if (auto it = j.find("ok"); it != j.end() && it->is_boolean())
			ev.ok = it->get<bool>();
		if (auto it = j.find("echoReference"); it != j.end() && it->is_boolean())
			ev.echo_reference = it->get<bool>();
		if (auto it = j.find("translate"); it != j.end() && it->is_array()) {
			ev.has_translate = true;
			for (const auto &e : *it)
				if (e.is_string() && ev.translate.size() < 8)
					ev.translate.push_back(e.get<std::string>());
		}
	} else if (type == "pong") {
		ev.type = EventType::Pong;
		auto it = j.find("t");
		if (it != j.end() && it->is_number_unsigned()) {
			ev.pong_t = it->get<uint64_t>();
			ev.has_pong_t = true;
		}
	} else {
		ev.type = EventType::Unknown;
	}
	return true;
}

}  // namespace acw
