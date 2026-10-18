// SPDX-License-Identifier: GPL-2.0-or-later
// "Is there a newer version of the plugin?": version numbers, the signed manifest that every build is published
// with, and the HTTPS requests that read it. OBS independent; never touches the audio path. The plugin runs the
// check once per OBS start on its own thread; the download and the installation are in update_download.hpp and
// update_install.hpp.
//
// The plugin talks to one place only, kUpdateBaseUrl: the home server of the service, which keeps a verified copy
// of every release. It holds manifest.json, manifest.json.sig (a raw 64-byte Ed25519 signature over the exact bytes
// of manifest.json) and the files the manifest lists, all directly under that address.
#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace acw {

// The only address the plugin fetches updates from (a folder: it ends with a slash).
constexpr const char *kUpdateBaseUrl = "https://audiocleaner.live/api/downloads/obs-plugin/";

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

// ---------------------------------------------------------------------------------------------------
// the manifest

// The kinds of computer a build exists for (the keys of "files"), and the ending of each one's file name.
constexpr const char *kKeyWindows = "windows-x64";    // .zip: audio-cleaner/bin/64bit/*.dll + audio-cleaner/data/...
constexpr const char *kKeyMac = "macos";              // .tar.xz: audio-cleaner.plugin
constexpr const char *kKeyDebian = "ubuntu-debian";   // .deb
constexpr const char *kKeyLinuxOther = "linux-other"; // .tar.xz: audio-cleaner/bin/64bit/*.so + audio-cleaner/data/...

constexpr uint64_t kMaxUpdateFileBytes = 64ull * 1024 * 1024;  // a larger file is not an update of this plugin

struct UpdateFile {
	std::string name;    // a plain file name (valid_update_file_name), with the ending that goes with its kind
	uint64_t size = 0;   // 1 .. kMaxUpdateFileBytes
	std::string sha256;  // 64 lowercase hex digits (informational; the plugin checks the SHA-512)
	std::string sha512;  // 128 lowercase hex digits
};

struct UpdateManifest {
	std::string version;                      // valid (parse_version accepts it)
	std::map<std::string, UpdateFile> files;  // only well-formed entries of the four known kinds
};

// A plain file name: 1 to 120 characters of letters, digits, '.', '_' and '-', starting with a letter or digit,
// no "..". Never a path.
bool valid_update_file_name(const std::string &name);
// The ending a file of this kind must have (".zip", ".tar.xz", ".deb"); "" for an unknown kind.
const char *update_file_suffix(const std::string &key);

// {"version": "0.1.60", "files": {"windows-x64": {"name", "size", "sha256", "sha512"}, ...}}. False when the text
// is not a JSON object with a valid "version" and a "files" object. Entries that are malformed, of an unknown kind
// or with a name that is not allowed are left out (so find() will not offer them); extra fields are ignored.
bool parse_update_manifest(const std::string &body, UpdateManifest &out);

// ---------------------------------------------------------------------------------------------------
// the check

struct UpdateCheckOptions {
	std::string base_url = kUpdateBaseUrl;  // a folder, ends with '/'
	std::string public_key;                 // base64 of the 32-byte key; "" = kUpdatePublicKeyBase64
	std::string user_agent = "obs-audio-cleaner";  // "obs-audio-cleaner/<plugin version>"
	std::string ca_file;                           // "" = the system's trust store
	int connect_timeout_s = 5;
	int transfer_timeout_s = 8;                    // per request
	int total_timeout_s = 20;                      // for the whole check (both files)
	size_t max_manifest_bytes = 16384;             // a longer reply is refused
	int max_redirects = 4;
	bool allow_local_http = false;                 // tests: http:// to 127.x / the local network (https is always fine)
};

enum class UpdateStatus { UpdateAvailable, UpToDate, Failed };

struct UpdateCheckResult {
	UpdateStatus status = UpdateStatus::Failed;
	UpdateManifest manifest;  // the signed manifest; set unless Failed
	std::string error;        // set when Failed: a technical reason for the log
};

// One check. run() blocks (use a thread of your own); cancel() may be called from any thread and makes run()
// return soon. Certificates and host names are always verified; only https:// is followed (also across
// redirects, which must stay on the same server), plus http:// to this computer when allow_local_http is set.
// The manifest is used only when its signature verifies.
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
