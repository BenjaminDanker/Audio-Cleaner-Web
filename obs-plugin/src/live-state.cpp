/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors
SPDX-License-Identifier: GPL-2.0-or-later
*/

#include "live-state.h"

#include "acw/live_gate.hpp"

#include <obs-module.h>
#ifdef ACW_HAVE_FRONTEND_API
#include <obs-frontend-api.h>
#endif

#include <chrono>
#include <memory>
#include <mutex>

namespace {

acw::LiveGate g_gate;  // thread-safe itself
std::mutex g_listener_mu;
std::function<void()> g_listener;

#ifdef ACW_HAVE_FRONTEND_API

void notify()
{
	std::function<void()> fn;
	{
		std::lock_guard<std::mutex> lk(g_listener_mu);
		fn = g_listener;
	}
	if (fn)
		fn();
}

void mark(acw::LiveOutput o, bool active)
{
	g_gate.set_active(o, active, std::chrono::steady_clock::now());
}

// What OBS says is running: only ever adds (an output that is "starting" is not running yet, and its event told us).
bool poll_outputs()
{
	bool added = false;
	const struct {
		acw::LiveOutput o;
		bool on;
	} now[] = {
		{acw::LiveOutput::Stream, obs_frontend_streaming_active()},
		{acw::LiveOutput::Recording, obs_frontend_recording_active()},
		{acw::LiveOutput::Replay, obs_frontend_replay_buffer_active()},
		{acw::LiveOutput::VirtualCam, obs_frontend_virtualcam_active()},
	};
	for (const auto &n : now) {
		if (n.on && !g_gate.active(n.o)) {
			mark(n.o, true);
			added = true;
		}
	}
	return added;
}

void on_frontend_event(enum obs_frontend_event event, void *)
{
	try {
		switch (event) {
		// An output that is starting counts at once, so the session is ready when the output begins; one that is stopping
		// still runs until it has stopped.
		case OBS_FRONTEND_EVENT_STREAMING_STARTING:
		case OBS_FRONTEND_EVENT_STREAMING_STARTED:
			mark(acw::LiveOutput::Stream, true);
			break;
		case OBS_FRONTEND_EVENT_STREAMING_STOPPED:
			mark(acw::LiveOutput::Stream, false);
			break;
		case OBS_FRONTEND_EVENT_RECORDING_STARTING:
		case OBS_FRONTEND_EVENT_RECORDING_STARTED:
			mark(acw::LiveOutput::Recording, true);
			break;
		case OBS_FRONTEND_EVENT_RECORDING_STOPPED:
			mark(acw::LiveOutput::Recording, false);
			break;
		case OBS_FRONTEND_EVENT_REPLAY_BUFFER_STARTING:
		case OBS_FRONTEND_EVENT_REPLAY_BUFFER_STARTED:
			mark(acw::LiveOutput::Replay, true);
			break;
		case OBS_FRONTEND_EVENT_REPLAY_BUFFER_STOPPED:
			mark(acw::LiveOutput::Replay, false);
			break;
		case OBS_FRONTEND_EVENT_VIRTUALCAM_STARTED:
			mark(acw::LiveOutput::VirtualCam, true);
			break;
		case OBS_FRONTEND_EVENT_VIRTUALCAM_STOPPED:
			mark(acw::LiveOutput::VirtualCam, false);
			break;
		case OBS_FRONTEND_EVENT_FINISHED_LOADING:
			poll_outputs();  // whatever was running before this plugin could listen
			break;
		default:
			return;
		}
		notify();
	} catch (...) {
		// Never reaches OBS: the worst case is that the sessions follow the state a second later (live_state_poll).
	}
}

#endif  // ACW_HAVE_FRONTEND_API

}  // namespace

extern "C" void live_state_init(void)
{
#ifdef ACW_HAVE_FRONTEND_API
	obs_frontend_add_event_callback(on_frontend_event, nullptr);
	try {
		poll_outputs();  // the plugin may load while something already runs
	} catch (...) {
	}
#endif
}

extern "C" void live_state_shutdown(void)
{
#ifdef ACW_HAVE_FRONTEND_API
	obs_frontend_remove_event_callback(on_frontend_event, nullptr);
#endif
	live_state_set_listener(nullptr);
}

bool live_state_available()
{
#ifdef ACW_HAVE_FRONTEND_API
	return true;
#else
	return false;
#endif
}

bool live_state_wanted()
{
#ifdef ACW_HAVE_FRONTEND_API
	return g_gate.run(std::chrono::steady_clock::now());
#else
	return true;
#endif
}

void live_state_poll()
{
#ifdef ACW_HAVE_FRONTEND_API
	try {
		poll_outputs();
	} catch (...) {
	}
#endif
}

void live_state_set_listener(std::function<void()> listener)
{
	std::lock_guard<std::mutex> lk(g_listener_mu);
	g_listener = std::move(listener);
}
