/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors
SPDX-License-Identifier: GPL-2.0-or-later
*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* One paced caption channel shared by all filter instances (CEA-608/708
 * bandwidth is limited, so several mics must not talk over each other). */
void caption_output_init(void);
void caption_output_shutdown(void);

/* Queue caption text for the streaming output; a no-op while not streaming. */
void caption_output_push(const char *text);

#ifdef __cplusplus
}
#endif
