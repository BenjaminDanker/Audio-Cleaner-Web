// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/output_select.hpp"

#include <algorithm>

namespace acw {

std::vector<size_t> pick_caption_outputs(const std::vector<OutputInfo> &outputs, const CaptionTargets &targets)
{
	std::vector<size_t> picked;
	std::vector<uintptr_t> seen;
	// An output the frontend names (stream, recording, replay buffer) is never also one of "the others", even when its
	// own box is off: that box decides.
	std::vector<uintptr_t> named;
	for (const OutputInfo &o : outputs)
		if (o.role != OutputRole::Other)
			named.push_back(o.id);
	for (size_t i = 0; i < outputs.size(); ++i) {
		const OutputInfo &o = outputs[i];
		if (o.id == 0 || !o.active)
			continue;
		bool wanted = false;
		switch (o.role) {
		case OutputRole::Stream:
			wanted = targets.stream;
			break;
		case OutputRole::Recording:
			wanted = targets.recording;
			break;
		case OutputRole::Replay:
			wanted = targets.replay;
			break;
		case OutputRole::Other:
			wanted = targets.others && o.encoded && o.has_video_encoder &&
				 std::find(named.begin(), named.end(), o.id) == named.end();
			break;
		}
		if (!wanted)
			continue;
		// The same output can be listed more than once (e.g. as the stream and in the general list), or one output
		// serve as both stream and recording: it gets each caption once. The first listing decides.
		if (std::find(seen.begin(), seen.end(), o.id) != seen.end())
			continue;
		seen.push_back(o.id);
		picked.push_back(i);
	}
	return picked;
}

uint64_t output_tag(const OutputInfo &o)
{
	switch (o.role) {
	case OutputRole::Stream:
		return 1;
	case OutputRole::Recording:
		return 2;
	case OutputRole::Replay:
		return 3;
	case OutputRole::Other:
		break;
	}
	uint64_t h = 1469598103934665603ull;  // FNV-1a
	for (unsigned char c : o.name) {
		h ^= c;
		h *= 1099511628211ull;
	}
	return h | 0x100ull;  // never one of the small constants above
}

}  // namespace acw
