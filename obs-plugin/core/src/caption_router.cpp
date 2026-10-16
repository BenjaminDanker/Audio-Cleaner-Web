// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/caption_router.hpp"

#include <algorithm>

namespace acw {

namespace {

using Clock = std::chrono::steady_clock;

// When a cue starts and ends on this computer's clock. The service counts seconds of audio from the start of the
// connection (`origin`); without that, or without sensible times, the cue ends when it arrived.
void cue_times(Clock::time_point origin, double start, double end, Clock::time_point arrival, Clock::time_point &s,
	       Clock::time_point &e)
{
	auto secs = [](double v) { return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(v)); };
	if (origin != Clock::time_point{} && end > start && start >= 0.0 && end < 1e7) {
		s = origin + secs(start);
		e = origin + secs(end);
		return;
	}
	e = arrival;
	s = arrival - secs(std::min(6.0, std::max(1.0, end - start)));
}

}  // namespace

CaptionRouter::CaptionRouter(Sinks sinks) : sinks_(std::move(sinks)) {}

CaptionRouter::~CaptionRouter()
{
	stop();
}

void CaptionRouter::configure(const RouterConfig &cfg)
{
	std::unique_ptr<OverlayDisplay> retire;  // destroyed outside the lock (its thread sends the last empty texts)
	{
		std::lock_guard<std::mutex> lk(mu_);
		if (stopped_)
			return;
		cfg_ = cfg;
		std::map<std::string, OverlayText::Options> targets;
		if (!cfg_.overlay_spoken.empty())
			targets[cfg_.overlay_spoken] = cfg_.overlay_options;
		for (size_t i = 0; i < cfg_.translate.size() && i < cfg_.overlay.size(); ++i)
			if (!cfg_.translate[i].empty() && !cfg_.overlay[i].empty())
				targets[cfg_.overlay[i]] = cfg_.overlay_options;
		if (targets.empty()) {
			retire = std::move(display_);
		} else {
			if (!display_)
				display_ = std::make_unique<OverlayDisplay>(sinks_.overlay);
			display_->set_targets(targets);
		}
	}
}

void CaptionRouter::on_caption(const CaptionEvent &ev)
{
	if (!ev.is_final || ev.text.empty())
		return;
	const Clock::time_point arrival = Clock::now();
	bool to_cc = false;
	std::string spoken;
	bool subtitles = false;
	{
		std::lock_guard<std::mutex> lk(mu_);
		if (stopped_)
			return;
		to_cc = cfg_.closed_captions && cfg_.closed_caption_language.empty();
		if (display_ && !cfg_.overlay_spoken.empty())
			display_->push(cfg_.overlay_spoken, ev.text);
		spoken = cfg_.spoken;
		subtitles = cfg_.subtitles;
	}
	if (to_cc && sinks_.closed_caption)
		sinks_.closed_caption(ev.text);
	if (subtitles && sinks_.subtitle) {
		SubtitleCue c;
		c.lang = spoken;
		c.text = ev.text;
		cue_times(ev.origin, ev.start, ev.end, arrival, c.start, c.end);
		sinks_.subtitle(c);
	}
}

void CaptionRouter::on_translation(const TranslationEvent &ev)
{
	if (ev.text.empty() || ev.lang.empty())
		return;
	const Clock::time_point arrival = Clock::now();
	bool to_cc = false;
	bool subtitles = false;
	{
		std::lock_guard<std::mutex> lk(mu_);
		if (stopped_)
			return;
		size_t slot = cfg_.translate.size();
		for (size_t i = 0; i < cfg_.translate.size(); ++i)
			if (cfg_.translate[i] == ev.lang) {
				slot = i;
				break;
			}
		if (slot == cfg_.translate.size())
			return;  // a language that was not asked for (or no longer is)
		if (!ev.of.empty()) {
			auto &last = last_of_[ev.lang];
			if (last.first == ev.of && last.second == ev.origin)
				return;  // the same caption translated twice
			last = {ev.of, ev.origin};
		}
		to_cc = cfg_.closed_captions && cfg_.closed_caption_language == ev.lang;
		if (display_ && slot < cfg_.overlay.size() && !cfg_.overlay[slot].empty())
			display_->push(cfg_.overlay[slot], ev.text);
		subtitles = cfg_.subtitles;
	}
	if (to_cc && sinks_.closed_caption)
		sinks_.closed_caption(ev.text);
	if (subtitles && sinks_.subtitle) {
		SubtitleCue c;
		c.lang = ev.lang;
		c.text = ev.text;
		cue_times(ev.origin, ev.start, ev.end, arrival, c.start, c.end);
		sinks_.subtitle(c);
	}
}

void CaptionRouter::clear_overlays()
{
	std::lock_guard<std::mutex> lk(mu_);
	if (display_)
		display_->clear_all();
}

void CaptionRouter::stop()
{
	std::unique_ptr<OverlayDisplay> d;
	{
		std::lock_guard<std::mutex> lk(mu_);
		stopped_ = true;
		d = std::move(display_);
	}
	if (d)
		d->stop(300);
}

}  // namespace acw
