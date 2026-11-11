// SPDX-License-Identifier: GPL-2.0-or-later
// What a live session hands to the plugin: captions (word by word while they are said, then final) and their translations.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace acw {

struct CaptionEvent {
	std::string text;
	// false: the caption while it is still being said, as far as it is settled (see settled_words); every one of
	// these is followed by the next version of the same caption (same `id`), the last of which is final.
	bool is_final = false;
	double start = 0, end = 0;  // seconds of audio since the session started
	std::string id;             // the service's id for this caption ("" = none); translations refer to it
	// Where "seconds since the session started" is on this computer's steady clock (default = unknown).
	std::chrono::steady_clock::time_point origin{};
	uint64_t voice = 0;  // who spoke when several voices share the captions (see label_speaker); 0 = the only one
};

// A final caption in another language, sent a moment after the caption it translates.
struct TranslationEvent {
	std::string lang;  // 2-letter code
	std::string text;
	std::string of;    // id of the caption it translates ("" = unknown)
	double start = 0, end = 0;
	std::chrono::steady_clock::time_point origin{};
	uint64_t voice = 0;  // as in CaptionEvent
};

// The part of a caption still being said that is safe to show: every word but the last, which may be cut off
// ("hel" before "hello"); it goes up once the next word has started, or with the final caption. This is the balance
// between speed and accuracy: about one word behind the talk, and no half-heard word ever appears. "" = nothing yet.
inline std::string settled_words(const std::string &partial)
{
	auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
	size_t end = partial.size();
	while (end > 0 && blank(partial[end - 1]))
		--end;  // trailing space: the last word may still go on
	while (end > 0 && !blank(partial[end - 1]))
		--end;  // the last word
	while (end > 0 && blank(partial[end - 1]))
		--end;
	size_t begin = 0;
	while (begin < end && blank(partial[begin]))
		++begin;
	return partial.substr(begin, end - begin);
}

// Several voices can share the captions: a filter per microphone, or the two sides of a stereo source split into two
// sessions. Each line then starts with its speaker's name ("" = no label), and the ids of the second side get a prefix
// so its lines never replace the first side's (both sessions number their lines from the start). `voice` is a number
// of its own for each of them: two people can talk at once, so only a voice's own lines follow one another.
inline void label_speaker(CaptionEvent &ev, const std::string &speaker, const std::string &id_prefix = std::string(),
			  uint64_t voice = 0)
{
	ev.voice = voice;
	if (!speaker.empty() && !ev.text.empty())
		ev.text = speaker + ": " + ev.text;
	if (!id_prefix.empty() && !ev.id.empty())
		ev.id = id_prefix + ev.id;
}

inline void label_speaker(TranslationEvent &ev, const std::string &speaker, const std::string &id_prefix = std::string(),
			  uint64_t voice = 0)
{
	ev.voice = voice;
	if (!speaker.empty() && !ev.text.empty())
		ev.text = speaker + ": " + ev.text;
	if (!id_prefix.empty() && !ev.of.empty())
		ev.of = id_prefix + ev.of;
}

}  // namespace acw
