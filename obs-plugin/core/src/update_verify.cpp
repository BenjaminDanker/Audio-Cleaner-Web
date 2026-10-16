// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/update_verify.hpp"

extern "C" {
#include "monocypher-ed25519.h"
}

namespace acw {

namespace {
const char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64_value(char c)
{
	if (c >= 'A' && c <= 'Z')
		return c - 'A';
	if (c >= 'a' && c <= 'z')
		return c - 'a' + 26;
	if (c >= '0' && c <= '9')
		return c - '0' + 52;
	if (c == '+')
		return 62;
	if (c == '/')
		return 63;
	return -1;
}
}  // namespace

bool base64_decode(const std::string &text, std::vector<uint8_t> &out)
{
	if (text.size() % 4 != 0)
		return false;
	std::vector<uint8_t> r;
	r.reserve(text.size() / 4 * 3);
	for (size_t i = 0; i < text.size(); i += 4) {
		int v[4];
		int pad = 0;
		for (int k = 0; k < 4; ++k) {
			const char c = text[i + k];
			if (c == '=') {
				// padding only in the last group, only at its end
				if (i + 4 != text.size() || k < 2)
					return false;
				++pad;
				v[k] = 0;
			} else {
				if (pad > 0)
					return false;
				v[k] = b64_value(c);
				if (v[k] < 0)
					return false;
			}
		}
		const uint32_t n = (static_cast<uint32_t>(v[0]) << 18) | (static_cast<uint32_t>(v[1]) << 12) |
				   (static_cast<uint32_t>(v[2]) << 6) | static_cast<uint32_t>(v[3]);
		r.push_back(static_cast<uint8_t>(n >> 16));
		if (pad < 2)
			r.push_back(static_cast<uint8_t>(n >> 8));
		if (pad < 1)
			r.push_back(static_cast<uint8_t>(n));
		// The unused bits of the last character must be zero (one text per value).
		if (pad == 2 && (v[1] & 0x0f))
			return false;
		if (pad == 1 && (v[2] & 0x03))
			return false;
	}
	out = std::move(r);
	return true;
}

std::string base64_encode(const uint8_t *data, size_t size)
{
	std::string s;
	s.reserve((size + 2) / 3 * 4);
	for (size_t i = 0; i < size; i += 3) {
		const uint32_t n = (static_cast<uint32_t>(data[i]) << 16) | (i + 1 < size ? static_cast<uint32_t>(data[i + 1]) << 8 : 0) |
				   (i + 2 < size ? data[i + 2] : 0);
		s += kB64[(n >> 18) & 63];
		s += kB64[(n >> 12) & 63];
		s += i + 1 < size ? kB64[(n >> 6) & 63] : '=';
		s += i + 2 < size ? kB64[n & 63] : '=';
	}
	return s;
}

std::string hex_encode(const uint8_t *data, size_t size)
{
	static const char digits[] = "0123456789abcdef";
	std::string s;
	s.reserve(size * 2);
	for (size_t i = 0; i < size; ++i) {
		s += digits[data[i] >> 4];
		s += digits[data[i] & 15];
	}
	return s;
}

bool is_hex_digest(const std::string &text, size_t digits)
{
	if (text.size() != digits)
		return false;
	for (char c : text)
		if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
			return false;
	return true;
}

bool verify_ed25519(const std::string &message, const std::string &signature, const std::string &public_key_b64)
{
	std::vector<uint8_t> key;
	if (!base64_decode(public_key_b64, key) || key.size() != 32 || signature.size() != 64)
		return false;
	return crypto_ed25519_check(reinterpret_cast<const uint8_t *>(signature.data()), key.data(),
				    reinterpret_cast<const uint8_t *>(message.data()), message.size()) == 0;
}

struct Sha512::Impl {
	crypto_sha512_ctx ctx;
};

Sha512::Sha512() : impl_(new Impl)
{
	crypto_sha512_init(&impl_->ctx);
}
Sha512::~Sha512() = default;

void Sha512::update(const void *data, size_t size)
{
	crypto_sha512_update(&impl_->ctx, static_cast<const uint8_t *>(data), size);
}

std::string Sha512::finish_hex()
{
	uint8_t h[64];
	crypto_sha512_final(&impl_->ctx, h);
	return hex_encode(h, sizeof h);
}

std::string sha512_hex(const std::string &data)
{
	Sha512 h;
	h.update(data.data(), data.size());
	return h.finish_hex();
}

}  // namespace acw
