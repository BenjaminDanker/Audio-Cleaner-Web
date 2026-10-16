// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <cstdint>

namespace acw {

// Counters touched from several threads (all relaxed atomics).
struct Counters {
	std::atomic<uint64_t> captured{0};          // 20 ms frames assembled from the input
	std::atomic<uint64_t> sent{0};              // frames sent to the server
	std::atomic<uint64_t> reference_sent{0};    // echo-reference frames sent (silent ones only as silence frames, and only when the server takes them)
	std::atomic<uint64_t> silent_sent{0};       // microphone frames sent as silence frames (not encoded, not billed)
	std::atomic<uint64_t> net_loops{0};         // rounds of the network thread's streaming loop (how often it woke up)
	std::atomic<uint64_t> received{0};          // frames received back (valid, in time or not)
	std::atomic<uint64_t> used{0};              // output blocks played from processed audio
	std::atomic<uint64_t> late{0};              // sent, but the reply missed its deadline or never came
	std::atomic<uint64_t> passthrough{0};       // not sent (disconnected / warm-up): original audio played
	std::atomic<uint64_t> dropped{0};           // frames lost locally (queue or TCP back-pressure)
	std::atomic<uint64_t> bad_frames{0};        // malformed replies
	std::atomic<uint64_t> warmup_discarded{0};  // replies covering the model warm-up after (re)connect
	std::atomic<uint64_t> tx_bytes{0};          // audio payload+header bytes sent
	std::atomic<uint64_t> rx_bytes{0};          // audio bytes received
	std::atomic<uint32_t> reconnects{0};
};

}  // namespace acw
