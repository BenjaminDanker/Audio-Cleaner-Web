/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors
SPDX-License-Identifier: GPL-2.0-or-later
*/

#include "subtitle-files.h"

#include <obs-module.h>
#include <plugin-support.h>
#include <util/bmem.h>
#ifdef ACW_HAVE_FRONTEND_API
#include <obs-frontend-api.h>
#endif

#include <chrono>
#include <string>

namespace {

acw::SubtitleRecorder g_recorder;

#ifdef ACW_HAVE_FRONTEND_API

// The recording stopped: one file per language, named after the recording.
void write_files()
{
	std::vector<acw::SubtitleRecorder::File> files = g_recorder.finish();
	if (files.empty())
		return;
	char *last = obs_frontend_get_last_recording();
	const std::string recording = last ? last : "";
	bfree(last);
	if (recording.empty()) {
		obs_log(LOG_WARNING, "subtitle files: OBS did not say where the recording went, so none were written");
		return;
	}
	for (const acw::SubtitleRecorder::File &f : files) {
		const std::string path = acw::subtitle_path(recording, f.lang);
		std::string error;
		if (acw::write_text_file(path, f.srt, &error))
			obs_log(LOG_INFO, "subtitle file written: %s (%zu captions)", path.c_str(), f.cues);
		else
			obs_log(LOG_WARNING, "subtitle file not written (%s): %s", path.c_str(), error.c_str());
	}
}

void on_frontend_event(enum obs_frontend_event event, void *)
{
	try {
		switch (event) {
		case OBS_FRONTEND_EVENT_RECORDING_STARTED:
			g_recorder.begin(std::chrono::steady_clock::now());
			break;
		case OBS_FRONTEND_EVENT_RECORDING_PAUSED:
			g_recorder.pause(std::chrono::steady_clock::now());
			break;
		case OBS_FRONTEND_EVENT_RECORDING_UNPAUSED:
			g_recorder.resume(std::chrono::steady_clock::now());
			break;
		case OBS_FRONTEND_EVENT_RECORDING_STOPPED:
			write_files();
			break;
		case OBS_FRONTEND_EVENT_EXIT:
			g_recorder.cancel();
			break;
		default:
			break;
		}
	} catch (...) {
		// Subtitle files are an extra: whatever goes wrong here never reaches OBS.
	}
}

#endif

}  // namespace

extern "C" void subtitle_files_init(void)
{
#ifdef ACW_HAVE_FRONTEND_API
	obs_frontend_add_event_callback(on_frontend_event, nullptr);
#endif
}

extern "C" void subtitle_files_shutdown(void)
{
#ifdef ACW_HAVE_FRONTEND_API
	obs_frontend_remove_event_callback(on_frontend_event, nullptr);
#endif
	g_recorder.cancel();
}

void subtitle_files_add(const acw::SubtitleCue &cue)
{
	g_recorder.add(cue);
}
