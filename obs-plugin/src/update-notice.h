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
 * "A newer version of the plugin exists": once per OBS session, on a thread of its own, the plugin reads the
 * small version file of the newest release from GitHub and compares it with its own version. If the file
 * says newer, the filter's properties show a notice and a button that opens the release page. Nothing is
 * downloaded or installed (a loaded plugin cannot replace itself). The check can be turned off; the choice
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
/* Opens the release page in the browser (a fixed address on our own GitHub releases). */
void update_notice_open_download_page(void);

#ifdef __cplusplus
}
#endif
