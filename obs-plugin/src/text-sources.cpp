/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors
SPDX-License-Identifier: GPL-2.0-or-later
*/

#include "text-sources.h"

#include <obs-module.h>
#ifdef ACW_HAVE_FRONTEND_API
#include <obs-frontend-api.h>
#endif

#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace {

// Text sources made here live in no scene, and a source nobody holds is destroyed: keep one reference to each until
// the scene collection changes (a new collection must not inherit them) or OBS exits. A scene that gets the source
// ("Add Existing") holds its own reference.
std::mutex g_mu;
std::vector<obs_source_t *> g_kept;

void release_kept()
{
	std::vector<obs_source_t *> kept;
	{
		std::lock_guard<std::mutex> lk(g_mu);
		kept.swap(g_kept);
	}
	for (obs_source_t *s : kept)
		obs_source_release(s);
}

#ifdef ACW_HAVE_FRONTEND_API
void on_frontend_event(enum obs_frontend_event event, void *)
{
	if (event == OBS_FRONTEND_EVENT_SCENE_COLLECTION_CHANGING || event == OBS_FRONTEND_EVENT_EXIT)
		release_kept();
}
#endif

bool is_text_source(obs_source_t *src)
{
	const char *id = obs_source_get_unversioned_id(src);
	return id && (std::strcmp(id, "text_gdiplus") == 0 || std::strcmp(id, "text_ft2_source") == 0);
}

const char *default_face()
{
#if defined(_WIN32)
	return "Arial";
#elif defined(__APPLE__)
	return "Helvetica";
#else
	return "Sans Serif";
#endif
}

// The newest text source type this OBS has: GDI+ text on Windows, FreeType2 text everywhere (and on Windows when GDI+
// is missing).
const char *text_source_type(bool &gdiplus)
{
	gdiplus = false;
#if defined(_WIN32)
	if (const char *id = obs_get_latest_input_type_id("text_gdiplus")) {
		gdiplus = true;
		return id;
	}
#endif
	return obs_get_latest_input_type_id("text_ft2_source");
}

// Large white bold text with a dark outline, wrapped in a box 1600 wide, at the bottom: ready for a 1080p scene.
obs_data_t *caption_text_settings(bool gdiplus)
{
	obs_data_t *s = obs_data_create();
	obs_data_t *font = obs_data_create();
	obs_data_set_string(font, "face", default_face());
	obs_data_set_string(font, "style", "Bold");
	obs_data_set_int(font, "size", 56);
	obs_data_set_int(font, "flags", 1);  // OBS_FONT_BOLD
	obs_data_set_obj(s, "font", font);
	obs_data_release(font);
	obs_data_set_string(s, "text", "");
	obs_data_set_bool(s, "outline", true);
	if (gdiplus) {
		obs_data_set_int(s, "color", 0xFFFFFF);
		obs_data_set_int(s, "opacity", 100);
		obs_data_set_int(s, "outline_size", 4);
		obs_data_set_int(s, "outline_color", 0x000000);
		obs_data_set_int(s, "outline_opacity", 100);
		obs_data_set_string(s, "align", "center");
		obs_data_set_string(s, "valign", "bottom");
		obs_data_set_bool(s, "extents", true);
		obs_data_set_bool(s, "extents_wrap", true);
		obs_data_set_int(s, "extents_cx", 1600);
		obs_data_set_int(s, "extents_cy", 200);
	} else {
		obs_data_set_int(s, "color1", 0xFFFFFFFF);
		obs_data_set_int(s, "color2", 0xFFFFFFFF);
		obs_data_set_bool(s, "word_wrap", true);
		obs_data_set_int(s, "custom_width", 1600);
	}
	return s;
}

struct CreateTask {
	std::string name;
	std::function<void(TextSourceResult)> done;
};

void create_task(void *param)
{
	std::unique_ptr<CreateTask> t(static_cast<CreateTask *>(param));
	TextSourceResult result = TextSourceResult::NoTextSource;
	if (obs_source_t *existing = obs_get_source_by_name(t->name.c_str())) {
		result = is_text_source(existing) ? TextSourceResult::AlreadyThere : TextSourceResult::NameTaken;
		obs_source_release(existing);
	} else {
		bool gdiplus = false;
		if (const char *type = text_source_type(gdiplus)) {
			obs_data_t *settings = caption_text_settings(gdiplus);
			obs_source_t *src = obs_source_create(type, t->name.c_str(), settings, nullptr);
			obs_data_release(settings);
			if (src) {
				std::lock_guard<std::mutex> lk(g_mu);
				g_kept.push_back(src);  // the reference stays with us (see above)
				result = TextSourceResult::Created;
			}
		}
	}
	try {
		if (t->done)
			t->done(result);
	} catch (...) {
	}
}

bool add_if_text(void *param, obs_source_t *src)
{
	if (is_text_source(src))
		(*static_cast<std::function<void(const char *)> *>(param))(obs_source_get_name(src));
	return true;
}

}  // namespace

extern "C" void text_sources_init(void)
{
#ifdef ACW_HAVE_FRONTEND_API
	obs_frontend_add_event_callback(on_frontend_event, nullptr);
#endif
}

extern "C" void text_sources_shutdown(void)
{
#ifdef ACW_HAVE_FRONTEND_API
	obs_frontend_remove_event_callback(on_frontend_event, nullptr);
#endif
	release_kept();
}

void text_source_set_text(const char *name, const char *text)
{
	if (!name || !*name)
		return;
	obs_source_t *src = obs_get_source_by_name(name);
	if (!src)
		return;
	// Only the text changes: obs_source_update merges this into the source's settings, and for a video source it
	// applies them on the graphics thread (so calling it from any thread is the usual way).
	obs_data_t *patch = obs_data_create();
	obs_data_set_string(patch, "text", text ? text : "");
	obs_source_update(src, patch);
	obs_data_release(patch);
	obs_source_release(src);
}

bool text_source_exists(const char *name)
{
	if (!name || !*name)
		return false;
	obs_source_t *src = obs_get_source_by_name(name);
	if (src)
		obs_source_release(src);
	return src != nullptr;
}

void text_source_create(const std::string &name, std::function<void(TextSourceResult)> done)
{
	auto *t = new CreateTask{name, std::move(done)};
	obs_queue_task(OBS_TASK_UI, create_task, t, false);
}

void text_sources_list(std::function<void(const char *name)> add)
{
	obs_enum_sources(add_if_text, &add);
}
