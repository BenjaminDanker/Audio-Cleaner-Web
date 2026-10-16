/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors
SPDX-License-Identifier: GPL-2.0-or-later
*/

#include "caption-output.h"

#include "acw/caption_pacer.hpp"
#include "acw/delayed_sender.hpp"
#include "acw/output_select.hpp"

#include <obs-module.h>
#ifdef ACW_HAVE_FRONTEND_API
#include <obs-frontend-api.h>
#endif

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::mutex g_mu;
std::unique_ptr<acw::CaptionPacer> g_pacer;
// With a stream delay set in OBS, the stream's video goes out that many seconds late, and a caption is put on the
// video that leaves next: sent at once it would show before the words are heard. Lines for an output with a delay
// wait that long here (each output has its own delay, e.g. a multistream output; the recording has none).
std::shared_ptr<acw::DelayedSender> g_delayed;
std::map<const void *, unsigned> g_targets;  // per filter

#ifdef ACW_HAVE_FRONTEND_API

acw::CaptionTargets current_targets()
{
	unsigned mask = 0;
	{
		std::lock_guard<std::mutex> lk(g_mu);
		for (const auto &kv : g_targets)
			mask |= kv.second;
	}
	acw::CaptionTargets t;
	t.stream = (mask & CAPTION_TARGET_STREAM) != 0;
	t.recording = (mask & CAPTION_TARGET_RECORDING) != 0;
	t.replay = (mask & CAPTION_TARGET_REPLAY) != 0;
	t.others = (mask & CAPTION_TARGET_OTHERS) != 0;
	return t;
}

// The outputs OBS has right now, each with a reference that is released when this goes away.
struct Outputs {
	std::vector<acw::OutputInfo> infos;
	std::vector<obs_output_t *> outputs;  // parallel to infos; owned references

	Outputs() = default;
	Outputs(const Outputs &) = delete;
	Outputs &operator=(const Outputs &) = delete;
	~Outputs()
	{
		for (obs_output_t *o : outputs)
			obs_output_release(o);
	}

	bool has(obs_output_t *o) const
	{
		for (obs_output_t *x : outputs)
			if (x == o)
				return true;
		return false;
	}

	// Takes over the reference (null is ignored).
	void add(obs_output_t *o, acw::OutputRole role)
	{
		if (!o)
			return;
		if (has(o)) {
			obs_output_release(o);
			return;
		}
		acw::OutputInfo info;
		info.id = reinterpret_cast<uintptr_t>(o);
		const char *name = obs_output_get_name(o);
		info.name = name ? name : "";
		info.role = role;
		info.active = obs_output_active(o);
		info.encoded = (obs_output_get_flags(o) & OBS_OUTPUT_ENCODED) != 0;
		info.has_video_encoder = obs_output_get_video_encoder2(o, 0) != nullptr;
		info.delay_s = obs_output_get_active_delay(o);
		infos.push_back(std::move(info));
		outputs.push_back(o);
	}
};

bool collect_other(void *param, obs_output_t *o)
{
	auto *out = static_cast<Outputs *>(param);
	if (!out->has(o))
		out->add(obs_output_get_ref(o), acw::OutputRole::Other);
	return true;
}

// The stream, the recording and the replay buffer that the frontend names, then (when wanted) all the others.
void gather(Outputs &out, bool others)
{
	out.add(obs_frontend_get_streaming_output(), acw::OutputRole::Stream);
	out.add(obs_frontend_get_recording_output(), acw::OutputRole::Recording);
	out.add(obs_frontend_get_replay_buffer_output(), acw::OutputRole::Replay);
	if (others)
		obs_enum_outputs(collect_other, &out);
}

void put_on(obs_output_t *output, const std::string &line, double seconds)
{
	if (obs_output_active(output))
		obs_output_output_caption_text2(output, line.c_str(), seconds);
}

// Runs on the pacer thread: one <= 32 character line at a time, to every chosen output that runs.
void send_line(const std::string &line, double seconds)
{
	const acw::CaptionTargets targets = current_targets();
	Outputs outs;
	gather(outs, targets.others);
	std::shared_ptr<acw::DelayedSender> later;
	{
		std::lock_guard<std::mutex> lk(g_mu);
		later = g_delayed;
	}
	for (size_t i : acw::pick_caption_outputs(outs.infos, targets)) {
		const acw::OutputInfo &info = outs.infos[i];
		if (info.delay_s > 0 && later)
			later->later(line, seconds, std::chrono::steady_clock::now() + std::chrono::seconds(info.delay_s), acw::output_tag(info));
		else
			put_on(outs.outputs[i], line, seconds);
	}
}

// Runs on the delayed sender's thread: the output may have stopped or restarted meanwhile, so it is looked up again.
void send_delayed(const std::string &line, double seconds, uint64_t tag)
{
	const acw::CaptionTargets targets = current_targets();
	Outputs outs;
	gather(outs, targets.others);
	for (size_t i : acw::pick_caption_outputs(outs.infos, targets))
		if (acw::output_tag(outs.infos[i]) == tag)
			put_on(outs.outputs[i], line, seconds);
}

#endif  // ACW_HAVE_FRONTEND_API

}  // namespace

extern "C" void caption_output_init(void)
{
	std::lock_guard<std::mutex> lk(g_mu);
#ifdef ACW_HAVE_FRONTEND_API
	if (!g_pacer)
		g_pacer = std::make_unique<acw::CaptionPacer>(send_line);
	if (!g_delayed)
		g_delayed = std::make_shared<acw::DelayedSender>(acw::DelayedSender::TaggedSendFn(send_delayed));
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
		g_targets.clear();
	}
	if (p)
		p->stop(300);  // never holds up OBS shutdown for longer than this
	if (d)
		d->stop(300);
}

extern "C" void caption_output_set_targets(const void *owner, unsigned targets)
{
	std::lock_guard<std::mutex> lk(g_mu);
	if (targets)
		g_targets[owner] = targets;
	else
		g_targets.erase(owner);
}

extern "C" void caption_output_push(const char *text)
{
	if (!text)
		return;
	std::lock_guard<std::mutex> lk(g_mu);
	if (g_pacer)
		g_pacer->push(text);
}
