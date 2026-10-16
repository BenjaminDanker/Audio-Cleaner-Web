/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors
SPDX-License-Identifier: GPL-2.0-or-later
*/

#pragma once

#include <obs-module.h>

#ifdef __cplusplus
extern "C" {
#endif

void text_sources_init(void);
/* Lets go of the text sources this plugin created and kept alive. */
void text_sources_shutdown(void);

#ifdef __cplusplus
}

#include <functional>
#include <string>

/* Sets the "text" of the OBS text source called `name` (any thread; a missing source is ignored). */
void text_source_set_text(const char *name, const char *text);

/* True when a source with this name exists. */
bool text_source_exists(const char *name);

enum class TextSourceResult {
	Created,       /* a new text source, not in any scene yet */
	AlreadyThere,  /* a text source with that name exists already: it is simply used */
	NameTaken,     /* the name belongs to a source that is not a text source */
	NoTextSource,  /* this OBS has no text source type we know */
};

/* Creates a text source called `name` (nothing is added to any scene: the user does that with "Add Existing") on
 * OBS's UI thread, then calls `done` there. From the UI thread itself (a button click) this happens at once. */
void text_source_create(const std::string &name, std::function<void(TextSourceResult)> done);

/* The names of all text sources (the kinds this plugin can write to), in the order OBS lists them. */
void text_sources_list(std::function<void(const char *name)> add);
#endif
