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

/* See update-notice.h. Nothing here runs on the audio thread, and nothing blocks OBS: the request is made by a
 * thread of our own (acw_core's UpdateCheck, HTTPS with verified certificates, short timeouts, tiny size limit). */

#include <obs-module.h>
#include <util/platform.h>
#include <plugin-support.h>

#include "update-notice.h"

#include "acw/update_check.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <crt_externs.h>  // a dylib cannot name `environ` directly
#define ACW_ENVIRON (*_NSGetEnviron())
#else
extern char **environ;
#define ACW_ENVIRON environ
#endif
#endif

#define CONFIG_FILE "settings.json"
#define CONFIG_KEY "check_updates"
#define START_DELAY_S 3  /* let OBS finish starting before the first request */
#define STOP_WAIT_MS 3000

namespace {

std::mutex g_mu;
std::condition_variable g_cv;
bool g_loaded = false;       // the setting was read
bool g_enabled = true;       // "Check for updates when OBS starts"
bool g_started = false;      // this session's check was started (it runs once per OBS session)
bool g_stop = false;         // the plugin is unloading
bool g_done = false;         // the check thread has finished
bool g_found = false;        // the check found a newer version
std::string g_latest;        // ... its number
std::string g_url;           // ... and the page for it (always on our releases)
std::thread g_thread;
std::shared_ptr<acw::UpdateCheck> g_check;

// The settings file lives in the module's own config folder, so it is the same for every filter and scene.
char *config_file()
{
	return obs_module_config_path(CONFIG_FILE);
}

bool read_enabled()
{
	bool on = true;
	if (char *path = config_file()) {
		if (obs_data_t *d = obs_data_create_from_json_file_safe(path, ".bak")) {
			obs_data_set_default_bool(d, CONFIG_KEY, true);
			on = obs_data_get_bool(d, CONFIG_KEY);
			obs_data_release(d);
		}
		bfree(path);
	}
	return on;
}

void write_enabled(bool on)
{
	char *dir = obs_module_config_path("");
	char *path = config_file();
	if (dir && path) {
		os_mkdirs(dir);
		if (obs_data_t *d = obs_data_create()) {
			obs_data_set_bool(d, CONFIG_KEY, on);
			if (!obs_data_save_json_safe(d, path, ".tmp", ".bak"))
				obs_log(LOG_WARNING, "could not save the update check setting");
			obs_data_release(d);
		}
	}
	bfree(dir);
	bfree(path);
}

void worker(std::shared_ptr<acw::UpdateCheck> check, int delay_s)
{
	if (delay_s > 0) {
		std::unique_lock<std::mutex> lk(g_mu);
		g_cv.wait_for(lk, std::chrono::seconds(delay_s), [] { return g_stop; });
		if (g_stop) {
			g_done = true;
			g_cv.notify_all();
			return;
		}
	}

	acw::UpdateCheckOptions opt;
	opt.user_agent = std::string("obs-audio-cleaner/") + PLUGIN_VERSION;
#ifdef ACW_UPDATE_TEST_URL /* only for a test harness that stands in for libobs; never defined in a real build */
	opt.url = ACW_UPDATE_TEST_URL;
	opt.allow_local_http = true;
#endif
	const acw::UpdateCheckResult r = check->run(PLUGIN_VERSION, opt);

	{
		std::lock_guard<std::mutex> lk(g_mu);
		if (r.status == acw::UpdateStatus::UpdateAvailable) {
			g_found = true;
			g_latest = r.info.version;
			g_url = r.info.url;
		}
		g_done = true;
		g_cv.notify_all();
	}
	switch (r.status) {
	case acw::UpdateStatus::UpdateAvailable:
		obs_log(LOG_INFO, "update available: Audio Cleaner plugin %s (you have %s): %s", r.info.version.c_str(),
			PLUGIN_VERSION, r.info.url.c_str());
		break;
	case acw::UpdateStatus::UpToDate:
		obs_log(LOG_INFO, "the plugin is up to date (%s)", PLUGIN_VERSION);
		break;
	case acw::UpdateStatus::Failed:
		obs_log(LOG_INFO, "could not check for a newer plugin version: %s", r.error.c_str());
		break;
	}
}

// Called with the lock held. The thread is created at most once per session.
void start_check_locked(int delay_s)
{
	if (g_started || g_stop || !g_enabled)
		return;
	g_started = true;
	g_done = false;
	g_check = std::make_shared<acw::UpdateCheck>();
	g_thread = std::thread(worker, g_check, delay_s);
}

bool open_url(const std::string &url)
{
#if defined(_WIN32)
	const std::wstring wide(url.begin(), url.end());  // the address is plain ASCII (release_url_allowed)
	return reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", wide.c_str(), nullptr, nullptr, SW_SHOWNORMAL)) > 32;
#else
	/* No shell sees the address as code: it is an argument, never part of a command line. */
	pid_t pid = 0;
#if defined(__APPLE__)
	const char *const argv[] = {"open", url.c_str(), nullptr};
	if (posix_spawn(&pid, "/usr/bin/open", nullptr, nullptr, const_cast<char *const *>(argv), ACW_ENVIRON) != 0)
		return false;
#else
	/* xdg-open can keep running for a while: the shell starts it in the background and ends at once. */
	const char *const argv[] = {"sh", "-c", "xdg-open \"$1\" >/dev/null 2>&1 &", "sh", url.c_str(), nullptr};
	if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, const_cast<char *const *>(argv), ACW_ENVIRON) != 0)
		return false;
#endif
	int status = 0;
	if (waitpid(pid, &status, 0) < 0)
		return false;
	return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

}  // namespace

extern "C" void update_notice_start(void)
{
	const bool on = read_enabled();
	std::lock_guard<std::mutex> lk(g_mu);
	g_loaded = true;
	g_enabled = on;
	g_stop = false;
	start_check_locked(START_DELAY_S);
}

extern "C" void update_notice_stop(void)
{
	std::shared_ptr<acw::UpdateCheck> check;
	std::thread th;
	bool joinable = false;
	{
		std::unique_lock<std::mutex> lk(g_mu);
		g_stop = true;
		check = g_check;
		g_cv.notify_all();
		if (check)
			check->cancel();
		if (g_thread.joinable()) {
			joinable = g_cv.wait_for(lk, std::chrono::milliseconds(STOP_WAIT_MS), [] { return g_done; });
			th = std::move(g_thread);
		}
	}
	if (th.joinable()) {
		if (joinable)
			th.join();
		else
			th.detach();  // stuck in the network library: it cannot be waited for forever (same as acw_core's threads)
	}
	std::lock_guard<std::mutex> lk(g_mu);
	g_check.reset();
	g_started = false;
	g_found = false;
}

extern "C" bool update_notice_enabled(void)
{
	std::lock_guard<std::mutex> lk(g_mu);
	if (!g_loaded) {
		/* properties can be asked for before the module finished loading: read the file, do not guess */
		g_enabled = read_enabled();
		g_loaded = true;
	}
	return g_enabled;
}

extern "C" void update_notice_set_enabled(bool on)
{
	{
		std::lock_guard<std::mutex> lk(g_mu);
		if (g_loaded && g_enabled == on)
			return;
		g_loaded = true;
		g_enabled = on;
		start_check_locked(0);  // turned on in the middle of a session: the check that did not run yet, now
	}
	write_enabled(on);
	obs_log(LOG_INFO, "checking for plugin updates when OBS starts: %s", on ? "on" : "off");
}

extern "C" bool update_notice_available(char *latest, size_t size)
{
	std::lock_guard<std::mutex> lk(g_mu);
	if (!g_enabled || !g_found || !latest || size == 0)
		return false;
	const size_t n = std::min(g_latest.size(), size - 1);
	std::memcpy(latest, g_latest.data(), n);
	latest[n] = '\0';
	return true;
}

extern "C" void update_notice_open_download_page(void)
{
	std::string url;
	{
		std::lock_guard<std::mutex> lk(g_mu);
		url = g_found ? g_url : std::string();
	}
	if (!acw::release_url_allowed(url))
		url = acw::kReleasePageUrl;
	if (!open_url(url))
		obs_log(LOG_WARNING, "could not open the browser; the update is at %s", url.c_str());
}
