// SPDX-License-Identifier: GPL-2.0-or-later
// Thin RAII wrappers around libopus for 48 kHz mono, 20 ms frames.
#pragma once

#include "acw/common.hpp"

#include <cstddef>
#include <cstdint>

struct OpusEncoder;
struct OpusDecoder;

namespace acw {

constexpr size_t kMaxOpusPacket = 1276;  // largest packet libopus produces

// True when acw_core was built with libopus.
bool opus_available();

class OpusEnc {
public:
	explicit OpusEnc(int bitrate);
	~OpusEnc();
	OpusEnc(const OpusEnc &) = delete;
	OpusEnc &operator=(const OpusEnc &) = delete;

	bool ok() const { return enc_ != nullptr; }
	// Encoder look-ahead in samples at 48 kHz (the codec delay).
	int lookahead() const { return lookahead_; }
	// Encodes kFrameSamples samples. Returns the packet size or <= 0 on error.
	int encode(const float *pcm, uint8_t *out, int capacity);

private:
	OpusEncoder *enc_ = nullptr;
	int lookahead_ = 0;
};

class OpusDec {
public:
	OpusDec();
	~OpusDec();
	OpusDec(const OpusDec &) = delete;
	OpusDec &operator=(const OpusDec &) = delete;

	bool ok() const { return dec_ != nullptr; }
	// Decodes one packet to kFrameSamples samples. Returns the sample count or <= 0 on error.
	int decode(const uint8_t *packet, int len, float *out);

private:
	OpusDecoder *dec_ = nullptr;
};

}  // namespace acw
