/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors
SPDX-License-Identifier: GPL-2.0-or-later
*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* "Only while streaming or recording": watches OBS's outputs (stream, recording, replay buffer, virtual camera) through
 * the frontend events, once for the whole plugin, and answers whether the filters' sessions should run (acw::LiveGate
 * makes the decision). Without the frontend API (ACW_HAVE_FRONTEND_API undefined) nothing can be watched: the answer is
 * always yes, as before this setting existed. */
void live_state_init(void);
void live_state_shutdown(void);

#ifdef __cplusplus
}

#include <functional>

/* True when this build can watch the outputs at all (the setting means nothing without it). */
bool live_state_available();

/* Should the sessions run now? Any thread; cheap. */
bool live_state_wanted();

/* Asks OBS what is running and adds what no event told us about (a missed event). Any thread. */
void live_state_poll();

/* Called after a frontend event changed what the outputs do (on the thread that delivers events), so that the filters
 * can start their sessions at once. The listener must not call back into OBS's frontend. */
void live_state_set_listener(std::function<void()> listener);
#endif
