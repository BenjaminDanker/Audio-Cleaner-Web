// SPDX-License-Identifier: GPL-2.0-or-later
// Downloads the update file of this kind of computer from the plugin's own download address and proves it is
// the file the signed manifest lists (size and SHA-512) before anyone may use it. OBS independent.
#pragma once

#include "acw/update_check.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace acw {

struct DownloadOptions {
	std::string base_url = kUpdateBaseUrl;  // a folder, ends with '/'
	std::string user_agent = "obs-audio-cleaner";
	std::string ca_file;                    // "" = the system's trust store
	int connect_timeout_s = 10;
	int transfer_timeout_s = 300;           // the whole transfer
	int max_redirects = 4;
	bool allow_local_http = false;          // tests only
};

struct DownloadResult {
	bool ok = false;
	std::string path;   // set when ok: <dest_dir>/<file name>, complete and verified
	std::string error;  // set when not ok: the reason in plain words (it is shown to the user)
	bool cancelled = false;
};

// One download. run() blocks (use a thread of your own); cancel() may be called from any thread.
class UpdateDownload {
public:
	UpdateDownload();
	~UpdateDownload();
	UpdateDownload(const UpdateDownload &) = delete;
	UpdateDownload &operator=(const UpdateDownload &) = delete;

	// Fetches base_url + file.name into dest_dir (UTF-8, must exist). The bytes go to "<name>.part" and are renamed
	// only once their count and SHA-512 match `file`; a failed download leaves nothing behind. A file name that is
	// not a plain name is refused before anything is sent.
	DownloadResult run(const UpdateFile &file, const std::string &dest_dir, const DownloadOptions &opt = {});
	void cancel();
	// Bytes received so far (of the current attempt); 0 before it starts.
	uint64_t bytes_done() const;

private:
	struct Impl;
	std::shared_ptr<Impl> impl_;
};

}  // namespace acw
