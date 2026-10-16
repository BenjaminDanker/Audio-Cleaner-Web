// SPDX-License-Identifier: GPL-2.0-or-later
// What a live session hands to the plugin: final captions and their translations.
#pragma once

#include <chrono>
#include <string>

namespace acw {

struct CaptionEvent {
	std::string text;
	bool is_final = false;
	double start = 0, end = 0;  // seconds of audio since the session started
	std::string id;             // the service's id for this caption ("" = none); translations refer to it
	// Where "seconds since the session started" is on this computer's steady clock (default = unknown).
	std::chrono::steady_clock::time_point origin{};
};

// A final caption in another language, sent a moment after the caption it translates.
struct TranslationEvent {
	std::string lang;  // 2-letter code
	std::string text;
	std::string of;    // id of the caption it translates ("" = unknown)
	double start = 0, end = 0;
	std::chrono::steady_clock::time_point origin{};
};

// Several voices can share the captions: a filter per microphone, or the two sides of a stereo source split into two
// sessions. Each line then starts with its speaker's name ("" = no label), and the ids of the second side get a prefix
// so its lines never replace the first side's (both sessions number their lines from the start).
inline void label_speaker(CaptionEvent &ev, const std::string &speaker, const std::string &id_prefix = std::string())
{
	if (!speaker.empty() && !ev.text.empty())
		ev.text = speaker + ": " + ev.text;
	if (!id_prefix.empty() && !ev.id.empty())
		ev.id = id_prefix + ev.id;
}

inline void label_speaker(TranslationEvent &ev, const std::string &speaker, const std::string &id_prefix = std::string())
{
	if (!speaker.empty() && !ev.text.empty())
		ev.text = speaker + ": " + ev.text;
	if (!id_prefix.empty() && !ev.of.empty())
		ev.of = id_prefix + ev.of;
}

}  // namespace acw
