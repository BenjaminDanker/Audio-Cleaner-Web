/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors
SPDX-License-Identifier: GPL-2.0-or-later
*/

#include "caption-output.h"

#include "acw/caption_pacer.hpp"
#include "acw/delayed_sender.hpp"

#include <obs-module.h>
#ifdef ACW_HAVE_FRONTEND_API
#include <obs-frontend-api.h>
#endif

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

namespace {

std::mutex g_mu;
std::unique_ptr<acw::CaptionPacer> g_pacer;
// With a stream delay set in OBS, the stream's video goes out that many seconds late, and a caption is put on the
// video that leaves next: sent at once it would show before the words are heard. Stream lines wait that long here.
// The recording is not delayed, so its lines go at once.
std::shared_ptr<acw::DelayedSender> g_delayed;

#ifdef ACW_HAVE_FRONTEND_API
// Puts the line on `output` when it is running. Takes over the reference the frontend returned.
void send_to(obs_output_t *output, const std::string &line, double seconds)
{
	if (!output)
		return;
	if (obs_output_active(output))
		obs_output_output_caption_text2(output, line.c_str(), seconds);
	obs_output_release(output);
}
#endif

// Runs on the pacer thread: one <= 32 character line at a time, to the stream and to the recording.
void send_line(const std::string &line, double seconds)
{
#ifdef ACW_HAVE_FRONTEND_API
	obs_output_t *stream = obs_frontend_get_streaming_output();
	obs_output_t *rec = obs_frontend_get_recording_output();
	if (rec == stream && rec) {  // one output serving as both: do not caption it twice
		obs_output_release(rec);
		rec = nullptr;
	}
	const uint32_t delay = stream ? obs_output_get_active_delay(stream) : 0;
	std::shared_ptr<acw::DelayedSender> later;
	if (delay > 0) {
		std::lock_guard<std::mutex> lk(g_mu);
		later = g_delayed;
	}
	if (later) {
		obs_output_release(stream);
		later->later(line, seconds, std::chrono::steady_clock::now() + std::chrono::seconds(delay));
	} else {
		send_to(stream, line, seconds);
	}
	send_to(rec, line, seconds);
#else
	(void)line;
	(void)seconds;
#endif
}

}  // namespace

extern "C" void caption_output_init(void)
{
	std::lock_guard<std::mutex> lk(g_mu);
	if (!g_pacer)
		g_pacer = std::make_unique<acw::CaptionPacer>(send_line);
#ifdef ACW_HAVE_FRONTEND_API
	if (!g_delayed)
		g_delayed = std::make_shared<acw::DelayedSender>([](const std::string &line, double seconds) {
			send_to(obs_frontend_get_streaming_output(), line, seconds);  // the stream may have restarted: ask again
		});
#endif
}

extern "C" void caption_output_shutdown(void)
{
	std::unique_ptr<acw::CaptionPacer> p;
	std::shared_ptr<acw::DelayedSender> d;
	{
		std::lock_guard<std::mutex> lk(g_mu);
		p = std::move(g_pacer);
		d = std::move(g_delayed);
	}
	if (p)
		p->stop(300);  // never holds up OBS shutdown for longer than this
	if (d)
		d->stop(300);
}

extern "C" void caption_output_push(const char *text)
{
	if (!text)
		return;
	std::lock_guard<std::mutex> lk(g_mu);
	if (g_pacer)
		g_pacer->push(text);
}
