// SPDX-License-Identifier: GPL-2.0-or-later
// Installing a downloaded (and verified) update. A loaded plugin cannot replace itself, so the swap is done by a
// small script that waits for this OBS process to exit; the plugin only writes that script (fixed text, every
// path quoted), checks the files, and starts it. OBS independent: the plugin tells it where its own module file
// is and which process it runs in.
//
//   Windows  zip with audio-cleaner/bin/64bit/*.dll + audio-cleaner/data/...: a PowerShell script waits for the
//            process, unpacks the zip next to the install folder and swaps the folders (elevated, with one UAC
//            prompt, when the folder is not writable for the user).
//   macOS    tar.xz with audio-cleaner.plugin: unpacked next to the installed bundle right away, swapped in on exit
//            by an sh script (only when the folder is writable; otherwise the file is shown to the user).
//   Linux    .deb (plugin under /usr): handed to the system's package installer. Anything else
//            (.../audio-cleaner/bin/64bit/*.so): like macOS.
// Every step is undone when it fails: the old plugin stays in place and works.
#pragma once

#include <string>

namespace acw {

enum class HostOs { Windows, MacOS, Linux };
HostOs host_os();  // the system this was compiled for

enum class InstallKind {
	Unknown,         // the module is somewhere the layout of our packages does not describe
	WindowsFolder,   // <root>\bin\64bit\<name>.dll
	MacBundle,       // <folder>/<name>.plugin/Contents/MacOS/<name>
	LinuxFolder,     // <root>/bin/64bit/<name>.so
	LinuxPackage,    // /usr/...: installed by the package manager
};

struct InstallTarget {
	InstallKind kind = InstallKind::Unknown;
	std::string key;   // which file of the manifest fits (kKeyWindows, ...); "" when Unknown
	std::string root;  // WindowsFolder / LinuxFolder: the plugin folder; MacBundle: the .plugin bundle (as given)
};

// Where this plugin is installed, from the path of its own module file (UTF-8, as the operating system reports it).
InstallTarget classify_install(HostOs os, const std::string &module_path);

// The name of the plugin's folder / module (the packages are built with it).
constexpr const char *kPluginName = "audio-cleaner";

// ---------------------------------------------------------------------------------------------------
// the scripts (pure text; the tests read them)

// Text safe for a quoted script argument: not empty, at most 1000 bytes, no control characters.
bool plain_path_text(const std::string &text);
// PowerShell single-quoted literal ('...' with ' and the typographic quotes doubled).
std::string ps_quote(const std::string &text);
// POSIX shell single-quoted literal.
std::string sh_quote(const std::string &text);
// Base64 of the UTF-16LE text, as `powershell -EncodedCommand` wants it.
std::string utf16le_base64(const std::string &utf8);

struct WindowsScriptArgs {
	unsigned long pid = 0;  // the OBS process to wait for
	std::string zip;        // the verified download
	std::string zip_sha512; // ... its SHA-512 (the script checks it again right before it unpacks)
	std::string root;       // the installed plugin folder
	std::string stage;      // a new folder next to `root` (same drive), removed at the end
	std::string work_dir;   // holds the zip and the script; removed at the end
	std::string log;        // one line per step
};
// The text of the PowerShell script (UTF-8; the file gets a byte order mark so that Windows PowerShell reads the
// paths right).
std::string windows_update_script(const WindowsScriptArgs &a);
// The command that runs the script file only if its bytes have the given SHA-512 (for the elevated start, where
// the file must not be changed between the prompt and the run). UTF-8 text for utf16le_base64().
std::string windows_guarded_command(const std::string &script_path, const std::string &script_sha512);

struct PosixScriptArgs {
	long pid = 0;           // the OBS process to wait for; 0 = do not wait
	std::string target;     // the installed plugin (folder or bundle)
	std::string staged;     // the new one, unpacked and checked, on the same file system
	std::string stage_dir;  // holds `staged` and the old plugin while swapping; removed at the end
	std::string work_dir;   // holds the script; removed at the end
	std::string log;        // one line per step
};
std::string posix_update_script(const PosixScriptArgs &a);

// ---------------------------------------------------------------------------------------------------
// doing it

enum class ApplyOutcome {
	InstallsOnExit,   // a script is waiting for OBS to close, then swaps the plugin
	InstalledNow,     // swapped already (needs a restart of OBS)
	InstallerOpened,  // the .deb was handed to the system's package installer
	FolderShown,      // the plugin cannot be replaced from here: the folder with the file was opened, `detail` is where it goes
	Failed,
};

struct ApplyRequest {
	std::string file;          // the verified download (UTF-8 path)
	std::string file_sha512;   // its SHA-512 hex
	InstallTarget target;
	unsigned long pid = 0;     // this OBS process
	std::string module_file;   // the plugin's own module file (its name is checked in the new package)
	std::string work_dir;      // from make_update_work_dir(); the download lives in it
	std::string log;           // a log file path for the scripts
	bool open_ui = true;       // false in tests: nothing is opened for the user (installer, folder window)
	bool immediate = false;    // POSIX: swap now instead of on exit (the OBS in a Flatpak sandbox: processes there
	                           // end with OBS, and a loaded library on Linux can be replaced on disk)
};

struct ApplyResult {
	ApplyOutcome outcome = ApplyOutcome::Failed;
	std::string error;   // Failed: the reason in plain words
	std::string detail;  // FolderShown: the folder to copy the plugin into
};

// A fresh private folder for one update (the download, the script). Old ones are tidied up.
bool make_update_work_dir(std::string &dir, std::string &error);
// Where the scripts log (in the user's temp folder).
std::string update_log_path();

// Starts the installation of the downloaded file. Returns at once (the script runs on its own).
ApplyResult apply_update(const ApplyRequest &req);

}  // namespace acw
