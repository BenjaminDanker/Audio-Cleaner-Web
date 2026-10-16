// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/update_install.hpp"

#include "acw/update_check.hpp"
#include "acw/update_verify.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

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
#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
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

namespace acw {

namespace fs = std::filesystem;

HostOs host_os()
{
#if defined(_WIN32)
	return HostOs::Windows;
#elif defined(__APPLE__)
	return HostOs::MacOS;
#else
	return HostOs::Linux;
#endif
}

// ---------------------------------------------------------------------------------------------------
// where is the plugin installed

namespace {

std::vector<std::string> split(const std::string &s, char sep)
{
	std::vector<std::string> out;
	size_t at = 0;
	for (;;) {
		const size_t next = s.find(sep, at);
		out.push_back(s.substr(at, next == std::string::npos ? std::string::npos : next - at));
		if (next == std::string::npos)
			break;
		at = next + 1;
	}
	return out;
}

std::string join(const std::vector<std::string> &parts, size_t count, char sep)
{
	std::string s;
	for (size_t i = 0; i < count; ++i) {
		if (i)
			s += sep;
		s += parts[i];
	}
	return s;
}

std::string lower(std::string s)
{
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return s;
}

bool ends_with_ci(const std::string &s, const std::string &suffix)
{
	return s.size() >= suffix.size() && lower(s.substr(s.size() - suffix.size())) == suffix;
}

bool starts_with(const std::string &s, const std::string &prefix)
{
	return s.compare(0, prefix.size(), prefix) == 0;
}

InstallTarget classify_windows(std::string path)
{
	std::replace(path.begin(), path.end(), '/', '\\');
	if (starts_with(path, "\\\\?\\") && !starts_with(path, "\\\\?\\UNC\\"))
		path = path.substr(4);
	const std::vector<std::string> p = split(path, '\\');
	const size_t n = p.size();
	InstallTarget t;
	if (n < 5 || !ends_with_ci(p[n - 1], ".dll") || lower(p[n - 2]) != "64bit" || lower(p[n - 3]) != "bin")
		return t;
	const size_t root_parts = n - 3;
	// "C:\bin\64bit\x.dll" would make the drive the plugin folder: not a layout of ours
	if (root_parts < 2 || (root_parts == 2 && p[0].size() == 2 && p[0][1] == ':' ))
		return t;
	const bool unc = p[0].empty() && p[1].empty();  // \\server\share\...
	for (size_t i = 1; i < root_parts; ++i)
		if (p[i].empty() && !(unc && i == 1))
			return t;
	t.kind = InstallKind::WindowsFolder;
	t.key = kKeyWindows;
	t.root = join(p, root_parts, '\\');
	return t;
}

InstallTarget classify_mac(const std::string &path)
{
	const std::vector<std::string> p = split(path, '/');
	const size_t n = p.size();
	InstallTarget t;
	if (n < 5 || p[0] != "" )
		return t;
	const size_t i = n - 4;  // <bundle>.plugin / Contents / MacOS / <binary>
	if (!ends_with_ci(p[i], ".plugin") || p[i].size() <= 7 || p[i + 1] != "Contents" || p[i + 2] != "MacOS" || p[i + 3].empty())
		return t;
	t.kind = InstallKind::MacBundle;
	t.key = kKeyMac;
	t.root = join(p, i + 1, '/');
	return t;
}

InstallTarget classify_linux(const std::string &path)
{
	InstallTarget t;
	if (starts_with(path, "/usr/") && !starts_with(path, "/usr/local/")) {
		t.kind = InstallKind::LinuxPackage;
		t.key = kKeyDebian;
		return t;
	}
	const std::vector<std::string> p = split(path, '/');
	const size_t n = p.size();
	if (n < 5 || p[0] != "" || !ends_with_ci(p[n - 1], ".so") || p[n - 2] != "64bit" || p[n - 3] != "bin")
		return t;
	for (size_t i = 1; i < n - 3; ++i)
		if (p[i].empty())
			return t;
	t.kind = InstallKind::LinuxFolder;
	t.key = kKeyLinuxOther;
	t.root = join(p, n - 3, '/');
	return t;
}

}  // namespace

InstallTarget classify_install(HostOs os, const std::string &module_path)
{
	switch (os) {
	case HostOs::Windows:
		return classify_windows(module_path);
	case HostOs::MacOS:
		return classify_mac(module_path);
	case HostOs::Linux:
		return classify_linux(module_path);
	}
	return {};
}

// ---------------------------------------------------------------------------------------------------
// script text

bool plain_path_text(const std::string &text)
{
	if (text.empty() || text.size() > 1000)
		return false;
	for (char ch : text)
		if (static_cast<unsigned char>(ch) < 0x20 || ch == 0x7f)
			return false;
	return true;
}

std::string ps_quote(const std::string &text)
{
	// PowerShell also reads U+2018 .. U+201B as single quotes: they are doubled like the ASCII one.
	std::string out = "'";
	for (size_t i = 0; i < text.size(); ++i) {
		const unsigned char c = static_cast<unsigned char>(text[i]);
		if (c == '\'') {
			out += "''";
		} else if (c == 0xE2 && i + 2 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x80 &&
			   static_cast<unsigned char>(text[i + 2]) >= 0x98 && static_cast<unsigned char>(text[i + 2]) <= 0x9B) {
			out.append(text, i, 3);
			out.append(text, i, 3);
			i += 2;
		} else {
			out += text[i];
		}
	}
	out += '\'';
	return out;
}

std::string sh_quote(const std::string &text)
{
	std::string out = "'";
	for (char c : text) {
		if (c == '\'')
			out += "'\\''";
		else
			out += c;
	}
	out += '\'';
	return out;
}

std::string utf16le_base64(const std::string &utf8)
{
	std::vector<uint8_t> bytes;
	for (size_t i = 0; i < utf8.size();) {
		const unsigned char c = static_cast<unsigned char>(utf8[i]);
		uint32_t cp = 0;
		size_t len = 1;
		if (c < 0x80) {
			cp = c;
		} else if ((c & 0xE0) == 0xC0 && i + 1 < utf8.size()) {
			cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(utf8[i + 1]) & 0x3Fu);
			len = 2;
		} else if ((c & 0xF0) == 0xE0 && i + 2 < utf8.size()) {
			cp = ((c & 0x0Fu) << 12) | ((static_cast<unsigned char>(utf8[i + 1]) & 0x3Fu) << 6) |
			     (static_cast<unsigned char>(utf8[i + 2]) & 0x3Fu);
			len = 3;
		} else if ((c & 0xF8) == 0xF0 && i + 3 < utf8.size()) {
			cp = ((c & 0x07u) << 18) | ((static_cast<unsigned char>(utf8[i + 1]) & 0x3Fu) << 12) |
			     ((static_cast<unsigned char>(utf8[i + 2]) & 0x3Fu) << 6) | (static_cast<unsigned char>(utf8[i + 3]) & 0x3Fu);
			len = 4;
		} else {
			cp = 0xFFFD;
		}
		i += len;
		auto put = [&](uint32_t unit) {
			bytes.push_back(static_cast<uint8_t>(unit & 0xFF));
			bytes.push_back(static_cast<uint8_t>(unit >> 8));
		};
		if (cp >= 0x10000) {
			cp -= 0x10000;
			put(0xD800 + (cp >> 10));
			put(0xDC00 + (cp & 0x3FF));
		} else {
			put(cp);
		}
	}
	return base64_encode(bytes.data(), bytes.size());
}

std::string windows_update_script(const WindowsScriptArgs &a)
{
	std::string s;
	s += "# Audio Cleaner OBS plugin: finishes an update once OBS has closed. Written by the plugin, runs once.\n";
	s += "$ErrorActionPreference = 'Stop'\n";
	s += "$ObsPid = " + std::to_string(a.pid) + "\n";
	s += "$Zip = " + ps_quote(a.zip) + "\n";
	s += "$ZipSha512 = " + ps_quote(a.zip_sha512) + "\n";
	s += "$Root = " + ps_quote(a.root) + "\n";
	s += "$Stage = " + ps_quote(a.stage) + "\n";
	s += "$WorkDir = " + ps_quote(a.work_dir) + "\n";
	s += "$LogFile = " + ps_quote(a.log) + "\n";
	s += R"PS(
function Write-Log([string]$Text) {
    try { Add-Content -LiteralPath $LogFile -Value ((Get-Date -Format 's') + ' ' + $Text) -Encoding UTF8 } catch { }
}
function Join-All {
    $path = $args[0]
    foreach ($part in $args[1..($args.Length - 1)]) { $path = Join-Path $path $part }
    return $path
}
function Invoke-Retry([scriptblock]$Action) {
    for ($try = 1; $try -le 20; $try++) {
        try { & $Action; return } catch { if ($try -eq 20) { throw }; Start-Sleep -Seconds 1 }
    }
}

$moved = $false
try {
    Write-Log 'waiting for OBS to close'
    $waited = 0
    while ((Get-Process -Id $ObsPid -ErrorAction SilentlyContinue) -and $waited -lt 86400) { Start-Sleep -Seconds 1; $waited++ }
    if (Get-Process -Id $ObsPid -ErrorAction SilentlyContinue) { throw 'OBS did not close' }
    Start-Sleep -Seconds 2

    # Read once, check, and unpack that copy: the downloaded file cannot change in between.
    $bytes = [IO.File]::ReadAllBytes($Zip)
    $hash = ([BitConverter]::ToString([Security.Cryptography.SHA512]::Create().ComputeHash($bytes)).Replace('-', '')).ToLowerInvariant()
    if ($hash -ne $ZipSha512) { throw 'the downloaded file was changed' }
    if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
    [void][IO.Directory]::CreateDirectory($Stage)
    $StageZip = Join-Path $Stage 'update.zip'
    [IO.File]::WriteAllBytes($StageZip, $bytes)
    $Unpacked = Join-Path $Stage 'new'
    Expand-Archive -LiteralPath $StageZip -DestinationPath $Unpacked -Force
    $NewRoot = Join-Path $Unpacked 'audio-cleaner'
    if (-not (Test-Path -LiteralPath (Join-All $NewRoot 'bin' '64bit' 'audio-cleaner.dll') -PathType Leaf)) { throw 'the update does not contain the plugin' }

    # Swap: the old folder is put aside, the new one moved in, and the old one comes back if that fails.
    $Old = Join-Path $Stage 'old'
    if (Test-Path -LiteralPath $Root) {
        Invoke-Retry { Move-Item -LiteralPath $Root -Destination $Old }
        $moved = $true
    }
    try {
        Move-Item -LiteralPath $NewRoot -Destination $Root
    } catch {
        if (Test-Path -LiteralPath $Root) { Remove-Item -LiteralPath $Root -Recurse -Force -ErrorAction SilentlyContinue }
        if ($moved) { Move-Item -LiteralPath $Old -Destination $Root }
        throw
    }
    Write-Log 'installed'
} catch {
    Write-Log ('failed, the old plugin stays: ' + $_.Exception.Message)
} finally {
    foreach ($leftover in @($Stage, $WorkDir)) {
        try { if (Test-Path -LiteralPath $leftover) { Remove-Item -LiteralPath $leftover -Recurse -Force } } catch { }
    }
}
)PS";
	return s;
}

std::string windows_guarded_command(const std::string &script_path, const std::string &script_sha512)
{
	std::string s;
	s += "$b = [IO.File]::ReadAllBytes(" + ps_quote(script_path) + "); ";
	s += "$h = ([BitConverter]::ToString([Security.Cryptography.SHA512]::Create().ComputeHash($b)).Replace('-', '')).ToLowerInvariant(); ";
	s += "if ($h -ne " + ps_quote(script_sha512) + ") { exit 1 }; ";
	s += "Invoke-Expression ([Text.Encoding]::UTF8.GetString($b, 3, $b.Length - 3))";  // 3 = the byte order mark
	return s;
}

std::string posix_update_script(const PosixScriptArgs &a)
{
	std::string s;
	s += "#!/bin/sh\n";
	s += "# Audio Cleaner OBS plugin: finishes an update once OBS has closed. Written by the plugin, runs once.\n";
	s += "PID=" + sh_quote(std::to_string(a.pid)) + "\n";
	s += "TARGET=" + sh_quote(a.target) + "\n";
	s += "STAGED=" + sh_quote(a.staged) + "\n";
	s += "STAGE_DIR=" + sh_quote(a.stage_dir) + "\n";
	s += "WORK_DIR=" + sh_quote(a.work_dir) + "\n";
	s += "LOG=" + sh_quote(a.log) + "\n";
	s += R"SH(
log() { printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$1" >> "$LOG" 2>/dev/null; }

if [ "$PID" != 0 ]; then
    log 'waiting for OBS to close'
    n=0
    while kill -0 "$PID" 2>/dev/null; do
        n=$((n + 1))
        if [ "$n" -gt 86400 ]; then log 'OBS did not close'; exit 1; fi
        sleep 1
    done
    sleep 1
fi

OLD="$STAGE_DIR/old"
KEEP=0
if [ ! -d "$STAGED" ]; then
    log 'failed, the old plugin stays: the new files are gone'
elif [ -e "$TARGET" ] || [ -h "$TARGET" ]; then
    if ! mv "$TARGET" "$OLD"; then
        log 'failed, the old plugin stays: could not move it aside'
    elif mv "$STAGED" "$TARGET"; then
        log 'installed'
    else
        rm -rf "$TARGET"
        if mv "$OLD" "$TARGET"; then
            log 'failed, the old plugin was put back'
        else
            KEEP=1
            log "failed, the old plugin is in $OLD"
        fi
    fi
elif mv "$STAGED" "$TARGET"; then
    log 'installed'
else
    log 'failed, could not move the new plugin in'
fi

if [ "$KEEP" = 0 ]; then rm -rf "$STAGE_DIR"; fi
rm -rf "$WORK_DIR"
)SH";
	return s;
}

// ---------------------------------------------------------------------------------------------------
// work folder

std::string update_log_path()
{
	std::error_code ec;
	return (fs::temp_directory_path(ec) / "audio-cleaner-update.log").u8string();
}

bool make_update_work_dir(std::string &dir, std::string &error)
{
	std::error_code ec;
	const fs::path base = fs::temp_directory_path(ec);
	if (ec) {
		error = "there is no temporary folder";
		return false;
	}
	// Folders of earlier attempts that nobody finished: a week old is enough.
	for (fs::directory_iterator it(base, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
		const std::string name = it->path().filename().u8string();
		if (name.compare(0, 11, "acw-update-") != 0)
			continue;
		std::error_code e2;
		if (it->is_symlink(e2) || !it->is_directory(e2))
			continue;
		const auto age = fs::file_time_type::clock::now() - it->last_write_time(e2);
		if (!e2 && age > std::chrono::hours(24 * 7))
			fs::remove_all(it->path(), e2);
	}
#if defined(_WIN32)
	std::random_device rd;
	for (int i = 0; i < 20; ++i) {
		char name[40];
		std::snprintf(name, sizeof name, "acw-update-%08x%08x", rd(), rd());
		const fs::path p = base / name;
		std::error_code e2;
		if (fs::create_directory(p, e2)) {
			dir = p.u8string();
			return true;
		}
	}
	error = "could not make a folder for the update";
	return false;
#else
	std::string templ = (base / "acw-update-XXXXXX").string();
	std::vector<char> buf(templ.begin(), templ.end());
	buf.push_back('\0');
	if (!mkdtemp(buf.data())) {  // mode 0700
		error = "could not make a folder for the update";
		return false;
	}
	dir = buf.data();
	return true;
#endif
}

// ---------------------------------------------------------------------------------------------------
// doing it

namespace {

bool write_text_file(const fs::path &path, const std::string &text, bool bom)
{
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f)
		return false;
	if (bom)
		f.write("\xEF\xBB\xBF", 3);
	f.write(text.data(), static_cast<std::streamsize>(text.size()));
	f.flush();
	return static_cast<bool>(f);
}

ApplyResult failed(const std::string &why)
{
	ApplyResult r;
	r.outcome = ApplyOutcome::Failed;
	r.error = why;
	return r;
}

#if defined(_WIN32)

std::wstring widen(const std::string &utf8)
{
	if (utf8.empty())
		return {};
	const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
	if (n <= 0)
		return {};
	std::wstring w(static_cast<size_t>(n), L'\0');
	MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()), &w[0], n);
	return w;
}

// Can this user make (and so rename) things in the folder? Tried for real: access lists are too many to read.
bool dir_writable(const std::wstring &dir)
{
	wchar_t name[64];
	swprintf(name, 64, L"\\.acw-write-test-%lu-%lu", GetCurrentProcessId(), GetTickCount());
	const HANDLE h = CreateFileW((dir + name).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
				     FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
	if (h == INVALID_HANDLE_VALUE)
		return false;
	CloseHandle(h);
	return true;
}

std::wstring powershell_exe()
{
	wchar_t sys[MAX_PATH];
	const UINT n = GetSystemDirectoryW(sys, MAX_PATH);
	if (n == 0 || n >= MAX_PATH)
		return {};
	return std::wstring(sys, n) + L"\\WindowsPowerShell\\v1.0\\powershell.exe";
}

ApplyResult apply_windows(const ApplyRequest &req)
{
	const InstallTarget &t = req.target;
	std::string root = t.root;
	const size_t cut = root.find_last_of('\\');
	if (cut == std::string::npos || cut == 0)
		return failed("this OBS installation cannot be updated from here");
	const std::string parent = root.substr(0, cut);
	const fs::path work = fs::u8path(req.work_dir);
	const std::string name = root.substr(cut + 1);
	if (name.empty() || !plain_path_text(root) || !plain_path_text(req.file) || !plain_path_text(req.work_dir) || !plain_path_text(req.log))
		return failed("a folder name cannot be used");
	const std::wstring wparent = widen(parent), wroot = widen(root), wexe = powershell_exe();
	if (wparent.empty() || wroot.empty() || wexe.empty())
		return failed("a folder name cannot be used");

	WindowsScriptArgs a;
	a.pid = req.pid;
	a.zip = req.file;
	a.zip_sha512 = req.file_sha512;
	a.root = root;
	a.stage = parent + "\\.acw-update-" + std::to_string(req.pid);
	a.work_dir = req.work_dir;
	a.log = req.log;
	const fs::path script = work / "acw-update.ps1";
	if (!write_text_file(script, windows_update_script(a), /*bom=*/true))
		return failed("the update could not be prepared");
	const std::string script_utf8 = script.u8string();

	const bool elevate = !dir_writable(wparent) || (GetFileAttributesW(wroot.c_str()) != INVALID_FILE_ATTRIBUTES && !dir_writable(wroot));
	const std::wstring wscript = widen(script_utf8);
	if (!elevate) {
		std::wstring cmd = L"\"" + wexe + L"\" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"" + wscript + L"\"";
		std::vector<wchar_t> buf(cmd.begin(), cmd.end());
		buf.push_back(L'\0');
		STARTUPINFOW si{};
		si.cb = sizeof si;
		PROCESS_INFORMATION pi{};
		// Outside OBS's job object when it allows that: the script must outlive OBS.
		BOOL ok = CreateProcessW(wexe.c_str(), buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB,
					 nullptr, nullptr, &si, &pi);
		if (!ok && GetLastError() == ERROR_ACCESS_DENIED)
			ok = CreateProcessW(wexe.c_str(), buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
		if (!ok)
			return failed("the update could not be started");
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
	} else {
		// One UAC prompt. The script file sits in a folder the user can write to: the elevated start runs it only
		// if its bytes are the ones written above.
		std::ifstream in(script, std::ios::binary);
		const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		const std::string guard = windows_guarded_command(script_utf8, sha512_hex(bytes));
		const std::wstring params = L"-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -EncodedCommand " + widen(utf16le_base64(guard));
		SHELLEXECUTEINFOW sei{};
		sei.cbSize = sizeof sei;
		sei.fMask = SEE_MASK_NOASYNC | SEE_MASK_FLAG_NO_UI;
		sei.lpVerb = L"runas";
		sei.lpFile = wexe.c_str();
		sei.lpParameters = params.c_str();
		sei.nShow = SW_HIDE;
		if (!ShellExecuteExW(&sei)) {
			std::error_code ec;
			fs::remove(script, ec);
			return failed(GetLastError() == ERROR_CANCELLED ? "the update was not allowed to change the plugin folder"
									 : "the update could not be started");
		}
	}
	ApplyResult r;
	r.outcome = ApplyOutcome::InstallsOnExit;
	return r;
}

void show_file(const ApplyRequest &req)
{
	if (!req.open_ui)
		return;
	const std::wstring params = L"/select,\"" + widen(req.file) + L"\"";
	ShellExecuteW(nullptr, L"open", L"explorer.exe", params.c_str(), nullptr, SW_SHOWNORMAL);
}

#else  // POSIX

// Runs a program (looked up in PATH) with output thrown away; its exit code, -1 when it could not be started.
// The arguments are an argument list: no shell reads them.
int run_program(const std::vector<std::string> &args, bool wait)
{
	std::vector<char *> argv;
	for (const std::string &a : args)
		argv.push_back(const_cast<char *>(a.c_str()));
	argv.push_back(nullptr);
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
	posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
	posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
	pid_t pid = 0;
	const int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, argv.data(), ACW_ENVIRON);
	posix_spawn_file_actions_destroy(&fa);
	if (rc != 0)
		return -1;
	if (!wait)
		return 0;
	int status = 0;
	while (waitpid(pid, &status, 0) < 0)
		if (errno != EINTR)
			return -1;
	return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// Starts a program that may keep running (xdg-open, the update script): a shell starts it in the background
// and ends at once, so nothing is left to wait for. `script_and_args` are arguments of the shell's command line
// ($1, $2 ...), never part of its text.
bool run_detached(const std::string &command, const std::vector<std::string> &positional)
{
	std::vector<std::string> args = {"sh", "-c", command, "sh"};
	args.insert(args.end(), positional.begin(), positional.end());
	return run_program(args, true) == 0;
}

bool dir_writable(const std::string &dir)
{
	return access(dir.c_str(), W_OK | X_OK) == 0;
}

std::string parent_of(const std::string &path)
{
	const size_t cut = path.find_last_of('/');
	if (cut == std::string::npos || cut == 0)
		return cut == 0 ? "/" : ".";
	return path.substr(0, cut);
}

bool is_plain_dir(const fs::path &p)
{
	std::error_code ec;
	const fs::file_status st = fs::symlink_status(p, ec);
	return !ec && fs::is_directory(st);
}

bool is_plain_file(const fs::path &p)
{
	std::error_code ec;
	const fs::file_status st = fs::symlink_status(p, ec);
	return !ec && fs::is_regular_file(st);
}

void open_with_system(const std::string &what)
{
#if defined(__APPLE__)
	run_program({"/usr/bin/open", what}, true);
#else
	run_detached("xdg-open \"$1\" >/dev/null 2>&1 &", {what});
#endif
}

void show_file(const ApplyRequest &req)
{
	if (!req.open_ui)
		return;
#if defined(__APPLE__)
	run_program({"/usr/bin/open", "-R", req.file}, true);  // Finder, with the file selected
#else
	open_with_system(parent_of(req.file));
#endif
}

ApplyResult apply_posix(const ApplyRequest &req)
{
	const InstallTarget &t = req.target;
	const bool mac = t.kind == InstallKind::MacBundle;
	const std::string target = t.root;
	const std::string parent = parent_of(target);
	if (!plain_path_text(target) || !plain_path_text(req.file) || !plain_path_text(req.work_dir) || !plain_path_text(req.log))
		return failed("a folder name cannot be used");
	std::error_code ec;
	if (!is_plain_dir(target))  // the plugin folder must be a folder of its own, not a link to somewhere else
		return failed("this plugin is not installed in a place that can be updated from here");
	if (!dir_writable(parent) || !dir_writable(target)) {
		ApplyResult r;
		r.outcome = ApplyOutcome::FolderShown;
		r.detail = parent;
		show_file(req);
		return r;
	}

	// 1. unpack next to the installed plugin (same file system, so the swap below is a rename), and look at it
	std::string templ = parent + "/.acw-update-XXXXXX";
	std::vector<char> buf(templ.begin(), templ.end());
	buf.push_back('\0');
	if (!mkdtemp(buf.data()))
		return failed("the update could not be unpacked here");
	const std::string stage_dir = buf.data();
	auto clean = [&] { fs::remove_all(fs::path(stage_dir), ec); };
	if (run_program({"tar", "-xf", req.file, "-C", stage_dir}, true) != 0) {
		clean();
		return failed("the update could not be unpacked");
	}
	const std::string staged = stage_dir + "/" + std::string(kPluginName) + (mac ? ".plugin" : "");
	bool good = is_plain_dir(staged);
	if (good && mac)
		good = is_plain_file(fs::path(staged) / "Contents" / "Info.plist") && is_plain_dir(fs::path(staged) / "Contents" / "MacOS");
	if (good && !mac)
		good = is_plain_file(fs::path(staged) / "bin" / "64bit" / (std::string(kPluginName) + ".so"));
	if (!good) {
		clean();
		return failed("the update does not contain the plugin");
	}

	// 2. the swap script
	PosixScriptArgs a;
	a.pid = req.immediate ? 0 : static_cast<long>(req.pid);
	a.target = target;
	a.staged = staged;
	a.stage_dir = stage_dir;
	a.work_dir = req.work_dir;
	a.log = req.log;
	const fs::path script = fs::path(req.work_dir) / "acw-update.sh";
	if (!write_text_file(script, posix_update_script(a), false)) {
		clean();
		return failed("the update could not be prepared");
	}
	chmod(script.c_str(), 0700);

	ApplyResult r;
	if (req.immediate) {
		if (run_program({"sh", script.string()}, true) != 0 || is_plain_dir(staged)) {
			clean();
			return failed("the plugin could not be replaced");
		}
		r.outcome = ApplyOutcome::InstalledNow;
		return r;
	}
	// Signals that end the terminal OBS was started from must not end the script.
	if (!run_detached("trap '' HUP INT TERM; sh \"$1\" </dev/null >/dev/null 2>&1 &", {script.string()})) {
		clean();
		return failed("the update could not be started");
	}
	r.outcome = ApplyOutcome::InstallsOnExit;
	return r;
}

#endif

}  // namespace

ApplyResult apply_update(const ApplyRequest &req)
{
	if (req.file.empty() || !is_hex_digest(req.file_sha512, 128) || req.work_dir.empty())
		return failed("the update could not be prepared");
	const InstallTarget &t = req.target;
	ApplyResult r;
	switch (t.kind) {
#if defined(_WIN32)
	case InstallKind::WindowsFolder:
		return apply_windows(req);
	default:
		r.outcome = ApplyOutcome::FolderShown;
		show_file(req);
		return r;
#else
	case InstallKind::LinuxPackage:
		// The system's own installer asks for the password and does the rest.
		if (req.open_ui)
			open_with_system(req.file);
		r.outcome = ApplyOutcome::InstallerOpened;
		return r;
	case InstallKind::MacBundle:
	case InstallKind::LinuxFolder:
		return apply_posix(req);
	default:
		r.outcome = ApplyOutcome::FolderShown;
		show_file(req);
		return r;
#endif
	}
}

}  // namespace acw
