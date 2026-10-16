// SPDX-License-Identifier: GPL-2.0-or-later
// What the plugin update trusts: an Ed25519 signature over the manifest, and SHA-512 hashes of the files it lists
// (both from Monocypher). OBS independent.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace acw {

// The key that signs the update manifests (the private half is the CI secret UPDATE_SIGNING_KEY): 32 raw bytes,
// base64.
constexpr const char *kUpdatePublicKeyBase64 = "lkLV2nf15eOuttdcqpieY2QJFyPsF4WOIwidisbp6Hw=";

// Standard base64 with padding, nothing else (no whitespace, no URL alphabet). False for anything malformed.
bool base64_decode(const std::string &text, std::vector<uint8_t> &out);
std::string base64_encode(const uint8_t *data, size_t size);

// Lowercase hex.
std::string hex_encode(const uint8_t *data, size_t size);
// 128 lowercase hex digits (a SHA-512)?  64 for a SHA-256.
bool is_hex_digest(const std::string &text, size_t digits);

// True when `signature` (exactly 64 raw bytes) is a valid Ed25519 signature of `message` under the public key
// `public_key_b64` (base64 of 32 raw bytes). A malformed key or signature is a plain false.
bool verify_ed25519(const std::string &message, const std::string &signature, const std::string &public_key_b64);

// SHA-512, streamed.
class Sha512 {
public:
	Sha512();
	~Sha512();
	Sha512(const Sha512 &) = delete;
	Sha512 &operator=(const Sha512 &) = delete;
	void update(const void *data, size_t size);
	// The digest as 128 lowercase hex digits. The object is spent afterwards.
	std::string finish_hex();

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

std::string sha512_hex(const std::string &data);

}  // namespace acw
