// SPDX-License-Identifier: GPL-2.0-or-later
// Which of OBS's outputs get closed captions. The plugin describes every output it can see (the stream, the
// recording and the replay buffer that the frontend names, plus all others from obs_enum_outputs, e.g. those made by
// multistream plugins); this decides, without any OBS call, which of them to caption.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace acw {

enum class OutputRole { Stream, Recording, Replay, Other };

struct OutputInfo {
	uintptr_t id = 0;       // identity of the output (its address): the same output listed twice is captioned once
	std::string name;       // OBS's name for it (stable for outputs of other plugins)
	OutputRole role = OutputRole::Other;  // Stream/Recording/Replay only for the ones the frontend named
	bool active = false;    // running now
	bool encoded = false;   // takes encoded packets (OBS_OUTPUT_ENCODED): only those can carry captions
	bool has_video_encoder = false;
	uint32_t delay_s = 0;   // obs_output_get_active_delay: the video leaves this many seconds late
};

struct CaptionTargets {
	bool stream = true;
	bool recording = true;
	bool replay = false;
	bool others = false;  // outputs of other plugins (multistream and the like)
};

// Indexes into `outputs` of the ones to caption, each output at most once.
//   - the frontend's stream/recording/replay output: when its box is ticked and it runs;
//   - any other output: only when "others" is ticked, it runs, takes encoded packets and has a video encoder
//     (an audio-only or raw output cannot carry a caption, and one that is not running ignores it).
// The roles are exclusive: an output the frontend names as the stream is never also treated as "other".
std::vector<size_t> pick_caption_outputs(const std::vector<OutputInfo> &outputs, const CaptionTargets &targets);

// A number that names an output for later (a delayed caption finds its output again with it): small constants for the
// three frontend roles, a hash of the name for the others.
uint64_t output_tag(const OutputInfo &o);

}  // namespace acw
