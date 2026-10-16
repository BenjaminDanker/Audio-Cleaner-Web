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


/* See update-notice.h. Nothing here runs on the audio thread, and nothing blocks OBS: the check and the update
 * each run on a thread of their own (acw_core: HTTPS with verified certificates, short timeouts, size limits, the
 * manifest's Ed25519 signature and the file's SHA-512 checked before anything is used). */

#include <obs-module.h>
#include <util/platform.h>
#include <plugin-support.h>

#include "update-notice.h"

#include "acw/update_check.hpp"
#include "acw/update_download.hpp"
#include "acw/update_install.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
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
#else
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
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
acw::UpdateManifest g_manifest;  // ... its signed manifest
std::thread g_thread;
std::shared_ptr<acw::UpdateCheck> g_check;

// The update itself (after the button)
update_install_state g_install = UPDATE_INSTALL_IDLE;
std::string g_install_detail;
bool g_install_running = false;
std::thread g_install_thread;
std::shared_ptr<acw::UpdateDownload> g_download;

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

std::string user_agent()
{
	return std::string("obs-audio-cleaner/") + PLUGIN_VERSION;
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
	opt.user_agent = user_agent();
#ifdef ACW_UPDATE_TEST_URL /* only for a test harness that stands in for libobs; never defined in a real build */
	opt.base_url = ACW_UPDATE_TEST_URL;
	opt.allow_local_http = true;
#endif
	const acw::UpdateCheckResult r = check->run(PLUGIN_VERSION, opt);

	{
		std::lock_guard<std::mutex> lk(g_mu);
		if (r.status == acw::UpdateStatus::UpdateAvailable) {
			g_found = true;
			g_manifest = r.manifest;
		}
		g_done = true;
		g_cv.notify_all();
	}
	switch (r.status) {
	case acw::UpdateStatus::UpdateAvailable:
		obs_log(LOG_INFO, "update available: Audio Cleaner plugin %s (you have %s)", r.manifest.version.c_str(),
			PLUGIN_VERSION);
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

// The path of this plugin's own module file (UTF-8), "" when it cannot be found.
std::string own_module_path()
{
#if defined(_WIN32)
	HMODULE module = nullptr;
	if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(&update_notice_start), &module))
		return {};
	std::wstring wide(32768, L'\0');
	const DWORD n = GetModuleFileNameW(module, wide.data(), static_cast<DWORD>(wide.size()));
	if (n == 0 || n >= wide.size())
		return {};
	wide.resize(n);
	const int len = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
	std::string out(static_cast<size_t>(std::max(len, 0)), '\0');
	if (len > 0)
		WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), out.data(), len, nullptr, nullptr);
	return out;
#else
	Dl_info info{};
	if (!dladdr(reinterpret_cast<void *>(&update_notice_start), &info) || !info.dli_fname)
		return {};
	char real[PATH_MAX];
	return realpath(info.dli_fname, real) ? std::string(real) : std::string(info.dli_fname);
#endif
}

unsigned long own_pid()
{
#if defined(_WIN32)
	return GetCurrentProcessId();
#else
	return static_cast<unsigned long>(getpid());
#endif
}

void set_install(update_install_state state, const std::string &detail)
{
	std::lock_guard<std::mutex> lk(g_mu);
	g_install = state;
	g_install_detail = detail;
}

// Downloads this computer's file, checks it and hands it to the installer. Runs on its own thread.
void install_worker(acw::UpdateManifest manifest, std::shared_ptr<acw::UpdateDownload> download)
{
	auto fail = [](const std::string &why) {
		obs_log(LOG_WARNING, "plugin update failed: %s", why.c_str());
		set_install(UPDATE_INSTALL_FAILED, why);
	};
	const std::string module = own_module_path();
	const acw::InstallTarget target = acw::classify_install(acw::host_os(), module);
	auto file = manifest.files.find(target.key);
	if (module.empty() || target.kind == acw::InstallKind::Unknown) {
		fail("this copy of the plugin is installed somewhere the updater does not know; download it from the website instead");
	} else if (file == manifest.files.end()) {
		fail("the new version has no file for this computer yet");
	} else {
		std::string dir, error;
		if (!acw::make_update_work_dir(dir, error)) {
			fail(error);
		} else {
			acw::DownloadOptions opt;
			opt.user_agent = user_agent();
#ifdef ACW_UPDATE_TEST_URL
			opt.base_url = ACW_UPDATE_TEST_URL;
			opt.allow_local_http = true;
#endif
			const acw::DownloadResult got = download->run(file->second, dir, opt);
			if (got.cancelled) {
				set_install(UPDATE_INSTALL_IDLE, {});
			} else if (!got.ok) {
				fail(got.error);
			} else {
				acw::ApplyRequest req;
				req.file = got.path;
				req.file_sha512 = file->second.sha512;
				req.target = target;
				req.pid = own_pid();
				req.module_file = module;
				req.work_dir = dir;
				req.log = acw::update_log_path();
#if !defined(_WIN32) && !defined(__APPLE__)
				req.immediate = std::getenv("FLATPAK_ID") != nullptr;  // processes in the sandbox end with OBS
#endif
				const acw::ApplyResult r = acw::apply_update(req);
				switch (r.outcome) {
				case acw::ApplyOutcome::InstallsOnExit:
					set_install(UPDATE_INSTALL_ON_EXIT, {});
					break;
				case acw::ApplyOutcome::InstalledNow:
					set_install(UPDATE_INSTALL_DONE_RESTART, {});
					break;
				case acw::ApplyOutcome::InstallerOpened:
					set_install(UPDATE_INSTALL_INSTALLER_OPEN, {});
					break;
				case acw::ApplyOutcome::FolderShown:
					set_install(UPDATE_INSTALL_FOLDER_SHOWN, r.detail);
					break;
				case acw::ApplyOutcome::Failed:
					fail(r.error);
					break;
				}
				obs_log(LOG_INFO, "plugin update %s: %s", manifest.version.c_str(),
					r.outcome == acw::ApplyOutcome::Failed ? "failed" : "handed to the installer");
			}
		}
	}
	std::lock_guard<std::mutex> lk(g_mu);
	g_install_running = false;
	g_cv.notify_all();
}

// Ends a thread: joined when it finishes in time, else left to finish on its own (it only holds shared state).
void finish_thread(std::thread &th, std::unique_lock<std::mutex> &lk, bool (*done)())
{
	if (!th.joinable())
		return;
	const bool in_time = g_cv.wait_for(lk, std::chrono::milliseconds(STOP_WAIT_MS), done);
	std::thread t = std::move(th);
	lk.unlock();
	if (in_time)
		t.join();
	else
		t.detach();  // stuck in the network library: it cannot be waited for forever (same as acw_core's threads)
	lk.lock();
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
	std::unique_lock<std::mutex> lk(g_mu);
	g_stop = true;
	g_cv.notify_all();
	if (g_check)
		g_check->cancel();
	if (g_download)
		g_download->cancel();
	finish_thread(g_thread, lk, [] { return g_done; });
	finish_thread(g_install_thread, lk, [] { return !g_install_running; });
	g_check.reset();
	g_download.reset();
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
	const size_t n = std::min(g_manifest.version.size(), size - 1);
	std::memcpy(latest, g_manifest.version.data(), n);
	latest[n] = '\0';
	return true;
}

extern "C" void update_notice_install(void)
{
	std::lock_guard<std::mutex> lk(g_mu);
	const bool again = g_install == UPDATE_INSTALL_IDLE || g_install == UPDATE_INSTALL_FAILED;
	if (!g_found || g_stop || g_install_running || !again)
		return;
	if (g_install_thread.joinable())
		g_install_thread.join();  // the previous try has finished (g_install_running is false)
	g_install = UPDATE_INSTALL_WORKING;
	g_install_detail.clear();
	g_install_running = true;
	g_download = std::make_shared<acw::UpdateDownload>();
	g_install_thread = std::thread(install_worker, g_manifest, g_download);
}

extern "C" enum update_install_state update_notice_install_state(char *detail, size_t size)
{
	std::lock_guard<std::mutex> lk(g_mu);
	if (detail && size) {
		const size_t n = std::min(g_install_detail.size(), size - 1);
		std::memcpy(detail, g_install_detail.data(), n);
		detail[n] = '\0';
	}
	return g_install;
}
