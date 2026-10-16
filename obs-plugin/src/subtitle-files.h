/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors
SPDX-License-Identifier: GPL-2.0-or-later
*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Subtitle files next to recordings: collects the captions of every filter while OBS records and writes one .srt file per
 * language when the recording stops. */
void subtitle_files_init(void);
void subtitle_files_shutdown(void);

#ifdef __cplusplus
}

#include "acw/subtitles.hpp"

/* One cue (any thread); ignored while OBS does not record. */
void subtitle_files_add(const acw::SubtitleCue &cue);
#endif
