// SPDX-License-Identifier: GPL-2.0-or-later
// Where the captions and translations of one filter go: the closed-caption channel (one language), an on-screen
// text source per language, and the subtitle recorder. OBS independent: the plugin supplies the three sinks.
#pragma once

#include "acw/caption_events.hpp"
#include "acw/overlay_text.hpp"
#include "acw/subtitles.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace acw {

struct RouterConfig {
	std::string spoken;                   // language code of the spoken captions ("" = not known: detected automatically)
	std::vector<std::string> translate;   // translation language of each slot (up to 3); "" = unused slot
	// Closed captions: one language. "" = the spoken one, else one of `translate`.
	std::string closed_caption_language;
	bool closed_captions = false;         // an output wants closed captions at all (and the language fits their format)
	// On-screen text sources by OBS name ("" = none): for the spoken language and for each translation slot.
	std::string overlay_spoken;
	std::vector<std::string> overlay;
	OverlayText::Options overlay_options;
	bool subtitles = false;               // keep cues for the subtitle files of recordings
};

class CaptionRouter {
public:
	struct Sinks {
		std::function<void(const std::string &text)> closed_caption;                                  // the paced caption channel
		std::function<void(const std::string &source, const std::string &text)> overlay;               // set a text source's text
		std::function<void(const SubtitleCue &cue)> subtitle;                                          // a cue for the recording's files
	};

	explicit CaptionRouter(Sinks sinks);
	~CaptionRouter();
	CaptionRouter(const CaptionRouter &) = delete;
	CaptionRouter &operator=(const CaptionRouter &) = delete;

	// Any thread; takes effect for the next event. Text sources that are no longer used are emptied.
	void configure(const RouterConfig &cfg);

	// The network thread. Both return quickly.
	void on_caption(const CaptionEvent &ev);
	void on_translation(const TranslationEvent &ev);

	// Empties the on-screen text (e.g. the session ended) without forgetting the configuration.
	void clear_overlays();
	// Ends the overlay thread (empties the sources first). Events after this are ignored.
	void stop();

private:
	Sinks sinks_;
	std::mutex mu_;
	RouterConfig cfg_;
	std::map<std::string, std::string> last_of_;  // per language: id of the caption translated last (a repeat is dropped)
	std::unique_ptr<OverlayDisplay> display_;
	bool stopped_ = false;
};

}  // namespace acw
