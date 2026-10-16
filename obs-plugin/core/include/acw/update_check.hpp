// SPDX-License-Identifier: GPL-2.0-or-later
// "Is there a newer version of the plugin?": version numbers, the small version.json that every release
// carries, and the HTTPS request that reads it. OBS independent; never touches the audio path. The plugin
// runs the request once per OBS start on its own thread and only ever shows a notice: nothing is downloaded.
#pragma once

#include <atomic>
#include <memory>
#include <string>

namespace acw {

// Where the newest build's version file lives, and the page people are sent to. Both are fixed: the address
// of the page is never taken from the file unless it starts with kReleasePagePrefix.
constexpr const char *kVersionJsonUrl =
	"https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/download/obs-plugin-latest/version.json";
constexpr const char *kReleasePageUrl = "https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/tag/obs-plugin-latest";
constexpr const char *kReleasePagePrefix = "https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/";

// MAJOR.MINOR.PATCH (a build is MAJOR.MINOR.<CI run number>). Up to four numeric parts are understood; a
// missing part counts as 0.
struct Version {
	int part[4] = {0, 0, 0, 0};
};

// Digits separated by single dots, 1 to 4 parts, each 1 to 6 digits. Anything else (a leading "v", spaces,
// "1.2.3-beta", an empty part) is refused.
bool parse_version(const std::string &text, Version &out);
// <0, 0, >0 like strcmp.
int compare_versions(const Version &a, const Version &b);
// True when `latest` and `current` are both valid and latest > current.
bool is_newer_version(const std::string &latest, const std::string &current);

// What version.json says: {"version": "0.1.57", "url": "https://github.com/.../releases/tag/obs-plugin-latest"}.
struct UpdateInfo {
	std::string version;  // valid (parse_version accepts it)
	std::string url;      // always an address that starts with kReleasePagePrefix (kReleasePageUrl when the file has none)
};
// False when the text is not a JSON object with a valid "version".
bool parse_update_json(const std::string &body, UpdateInfo &out);
// True for an address the "Download" button may open: starts with kReleasePagePrefix, plain printable characters.
bool release_url_allowed(const std::string &url);

struct UpdateCheckOptions {
	std::string url = kVersionJsonUrl;
	std::string user_agent = "obs-audio-cleaner";  // "obs-audio-cleaner/<plugin version>"
	std::string ca_file;                           // "" = the system's trust store
	int connect_timeout_s = 5;
	int transfer_timeout_s = 8;                    // per request, redirects included
	int total_timeout_s = 20;                      // for the whole check
	size_t max_bytes = 4096;                       // a longer reply is refused
	int max_redirects = 4;
	bool allow_local_http = false;                 // tests: http:// to 127.x / the local network (https is always fine)
};

enum class UpdateStatus { UpdateAvailable, UpToDate, Failed };

struct UpdateCheckResult {
	UpdateStatus status = UpdateStatus::Failed;
	UpdateInfo info;    // set unless Failed
	std::string error;  // set when Failed: a technical reason for the log
};

// One check. run() blocks (use a thread of your own); cancel() may be called from any thread and makes run()
// return soon. Certificates and host names are always verified; only https:// is followed (also across
// redirects), plus http:// to this computer when allow_local_http is set.
class UpdateCheck {
public:
	UpdateCheck();
	~UpdateCheck();
	UpdateCheck(const UpdateCheck &) = delete;
	UpdateCheck &operator=(const UpdateCheck &) = delete;

	UpdateCheckResult run(const std::string &current_version, const UpdateCheckOptions &opt = {});
	void cancel();

private:
	struct Impl;
	std::shared_ptr<Impl> impl_;
};

}  // namespace acw
