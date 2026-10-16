// SPDX-License-Identifier: GPL-2.0-or-later
// Audio Cleaner OBS plugin - shared constants.
#pragma once

#include <cstddef>
#include <cstdint>

namespace acw {

// Wire format (docs/streaming-protocol.md): mono 48 kHz, frames of 10-100 ms
// in 10 ms steps. We always send 20 ms frames.
constexpr int kNetRate = 48000;
constexpr size_t kFrameSamples = 960;      // 20 ms @ 48 kHz
constexpr size_t kMaxRxSamples = 4800;     // 100 ms: the largest frame the server may return
constexpr size_t kCrossfadeSamples = 240;  // 5 ms (design limit: <= 10 ms)

constexpr uint8_t kProtoVersion = 1;
constexpr uint8_t kCodecS16 = 1;
constexpr uint8_t kCodecF32 = 2;
constexpr uint8_t kCodecOpus = 3;  // exactly one 20 ms Opus packet per frame
constexpr uint8_t kCodecSilence = 4;  // no audio: a u16 (little endian) count of 48 kHz samples of silence (480..4800, multiple of 480)
constexpr size_t kSilencePayload = 2;
constexpr size_t kSilenceMinSamples = 480;
constexpr size_t kSilenceStep = 480;
constexpr size_t kHeaderSize = 16;
constexpr uint16_t kFlagReference = 1;  // frame flags bit 0: echo reference (what the speakers play), not microphone audio

// A block whose every sample is below this (absolute value) is digital silence: it is sent as a silence frame, not encoded.
constexpr float kSilenceLevel = 1e-5f;

// Latency presets (fixed output delay D, chosen at session start).
enum class LatencyPreset { Low = 80, Balanced = 150, Safe = 250 };

constexpr int delay_ms_for(LatencyPreset p) { return static_cast<int>(p); }

}  // namespace acw
