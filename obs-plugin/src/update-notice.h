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

/*
 * Plugin updates: once per OBS session, on a thread of its own, the plugin reads the signed manifest of the newest
 * plugin from our download server and compares it with its own version. If it is newer, the filter's properties
 * say so with a button; the button downloads this computer's file, checks it against the signed manifest and
 * installs it (acw_core update_download / update_install): on Windows and for plugin folders it is swapped in when
 * OBS closes, a .deb install is handed to the system's package installer. The check can be turned off; the choice
 * is global (OBS module config, settings.json), not per filter.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* obs_module_load: reads the setting; unless it is off, starts the one check of this session (in the
 * background, a few seconds after OBS has started loading). */
void update_notice_start(void);
/* obs_module_unload: ends the check and joins its thread. */
void update_notice_stop(void);

/* "Check for updates when OBS starts". */
bool update_notice_enabled(void);
/* Saves the choice. Turning it on starts this session's check if it has not run yet. */
void update_notice_set_enabled(bool on);

/* True when the check is on and found a newer version; copies its number into `latest`. */
bool update_notice_available(char *latest, size_t size);
/* Where the update stands after the button was pressed. */
enum update_install_state {
	UPDATE_INSTALL_IDLE = 0,       /* not started */
	UPDATE_INSTALL_WORKING,        /* downloading and checking */
	UPDATE_INSTALL_ON_EXIT,        /* ready: it installs when OBS closes */
	UPDATE_INSTALL_DONE_RESTART,   /* installed already: restart OBS */
	UPDATE_INSTALL_INSTALLER_OPEN, /* the system's package installer was opened with the file */
	UPDATE_INSTALL_FOLDER_SHOWN,   /* cannot be replaced from here: the folder with the file was opened; detail = where it goes */
	UPDATE_INSTALL_FAILED,         /* detail = why, in plain words */
};
/* Starts downloading and installing the newer version (in the background; returns at once). */
void update_notice_install(void);
/* The state of that, with its detail text (may be empty). */
enum update_install_state update_notice_install_state(char *detail, size_t size);

#ifdef __cplusplus
}
#endif
