// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/opus_codec.hpp"

#ifdef ACW_HAVE_OPUS
#include <opus.h>
#endif

namespace acw {

#ifdef ACW_HAVE_OPUS

bool opus_available()
{
	return true;
}

OpusEnc::OpusEnc(int bitrate)
{
	int err = 0;
	enc_ = opus_encoder_create(kNetRate, 1, OPUS_APPLICATION_AUDIO, &err);
	if (err != OPUS_OK || !enc_) {
		enc_ = nullptr;
		return;
	}
	opus_encoder_ctl(enc_, OPUS_SET_BITRATE(bitrate));
	// Complexity 5: libopus defaults to 9-10, which costs several times the encode time at these bitrates for speech
	// that sounds the same (the server resamples and cleans it anyway).
	opus_encoder_ctl(enc_, OPUS_SET_COMPLEXITY(5));
	opus_encoder_ctl(enc_, OPUS_SET_VBR(1));
	opus_encoder_ctl(enc_, OPUS_SET_INBAND_FEC(0));
	opus_encoder_ctl(enc_, OPUS_SET_DTX(0));
	opus_int32 la = 0;
	opus_encoder_ctl(enc_, OPUS_GET_LOOKAHEAD(&la));
	lookahead_ = static_cast<int>(la);
}

OpusEnc::~OpusEnc()
{
	if (enc_)
		opus_encoder_destroy(enc_);
}

int OpusEnc::encode(const float *pcm, uint8_t *out, int capacity)
{
	if (!enc_)
		return -1;
	return opus_encode_float(enc_, pcm, static_cast<int>(kFrameSamples), out, capacity);
}

OpusDec::OpusDec()
{
	int err = 0;
	dec_ = opus_decoder_create(kNetRate, 1, &err);
	if (err != OPUS_OK)
		dec_ = nullptr;
}

OpusDec::~OpusDec()
{
	if (dec_)
		opus_decoder_destroy(dec_);
}

int OpusDec::decode(const uint8_t *packet, int len, float *out)
{
	if (!dec_)
		return -1;
	return opus_decode_float(dec_, packet, len, out, static_cast<int>(kMaxRxSamples), 0);
}

#else  // built without libopus: the session falls back to PCM

bool opus_available()
{
	return false;
}
OpusEnc::OpusEnc(int) {}
OpusEnc::~OpusEnc() = default;
int OpusEnc::encode(const float *, uint8_t *, int)
{
	return -1;
}
OpusDec::OpusDec() = default;
OpusDec::~OpusDec() = default;
int OpusDec::decode(const uint8_t *, int, float *)
{
	return -1;
}

#endif

}  // namespace acw
