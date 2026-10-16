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

/* Which outputs the closed captions go to (bits). A filter registers its own choice under its own address (any
 * pointer that stays the same for the filter's life); the channel serves the union of all filters' choices.
 * 0 removes the filter's entry. */
#define CAPTION_TARGET_STREAM 1u
#define CAPTION_TARGET_RECORDING 2u
#define CAPTION_TARGET_REPLAY 4u
#define CAPTION_TARGET_OTHERS 8u /* outputs made by other plugins (multistream and the like) */
void caption_output_set_targets(const void *owner, unsigned targets);

/* Queue caption text for the chosen outputs; a no-op while none of them runs. */
void caption_output_push(const char *text);

#ifdef __cplusplus
}
#endif
