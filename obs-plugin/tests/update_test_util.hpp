// SPDX-License-Identifier: GPL-2.0-or-later
// Shared by the update tests: a throw-away signing key and a manifest builder.
#pragma once

#include "acw/update_verify.hpp"

extern "C" {
#include "monocypher-ed25519.h"
}

#include <cstdint>
#include <string>

namespace updtest {

struct KeyPair {
	uint8_t secret[64];
	uint8_t pub[32];
	std::string pub_b64() const { return acw::base64_encode(pub, 32); }
};

// A key made from a one-byte seed: seed 1 is "ours" in the tests, seed 2 is somebody else's.
inline KeyPair make_key(uint8_t seed_byte)
{
	uint8_t seed[32];
	for (int i = 0; i < 32; ++i)
		seed[i] = static_cast<uint8_t>(seed_byte + i);
	KeyPair k;
	crypto_ed25519_key_pair(k.secret, k.pub, seed);
	return k;
}

// The raw 64-byte signature of `message`.
inline std::string sign(const KeyPair &k, const std::string &message)
{
	uint8_t sig[64];
	crypto_ed25519_sign(sig, k.secret, reinterpret_cast<const uint8_t *>(message.data()), message.size());
	return std::string(reinterpret_cast<const char *>(sig), 64);
}

// One entry of "files".
inline std::string file_entry(const std::string &name, const std::string &payload)
{
	return "{\"name\":\"" + name + "\",\"sha256\":\"" + std::string(64, 'a') + "\",\"sha512\":\"" + acw::sha512_hex(payload) +
	       "\",\"size\":" + std::to_string(payload.size()) + "}";
}

}  // namespace updtest
