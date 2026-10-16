/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>
#include <plugin-support.h>

#include "caption-output.h"
#include "live-state.h"
#include "subtitle-files.h"
#include "text-sources.h"
#include "update-notice.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

MODULE_EXPORT const char *obs_module_description(void)
{
	return "Cloud noise removal and live captions for audio sources (Audio Cleaner)";
}

/* audio-cleaner-filter.cpp */
void audio_cleaner_register_filter(void);
void audio_cleaner_shutdown(void);

bool obs_module_load(void)
{
	caption_output_init();
	text_sources_init();
	subtitle_files_init();
	live_state_init(); /* before the filters exist: they ask what the outputs do when they are created */
	audio_cleaner_register_filter();
	update_notice_start(); /* a background thread; never blocks the start of OBS */
	obs_log(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
	return true;
}

void obs_module_unload(void)
{
	update_notice_stop();
	live_state_shutdown();
	subtitle_files_shutdown();
	text_sources_shutdown();
	caption_output_shutdown();
	audio_cleaner_shutdown(); /* network threads that could not be joined in time must be gone before the code is */
	obs_log(LOG_INFO, "plugin unloaded");
}
