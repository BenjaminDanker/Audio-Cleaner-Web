// SPDX-License-Identifier: GPL-2.0-or-later
// Offline tests of the update installation: where the plugin is installed, the text of the install scripts (and,
// on Linux/macOS, running them for real in a scratch folder), and the whole apply_update() flow with a real
// archive. On Windows the same flow runs against the real Windows PowerShell.
#include "acw/update_check.hpp"
#include "acw/update_install.hpp"
#include "acw/update_verify.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace acw;
namespace fs = std::filesystem;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                                   \
	do {                                                                          \
		++g_checks;                                                           \
		if (!(cond)) {                                                        \
			++g_failures;                                                 \
			std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		}                                                                     \
	} while (0)

static bool contains(const std::string &hay, const std::string &needle)
{
	return hay.find(needle) != std::string::npos;
}

static bool wait_until(const std::function<bool()> &pred, int ms)
{
	const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
	while (std::chrono::steady_clock::now() < end) {
		if (pred())
			return true;
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	return pred();
}

static std::string read_file(const fs::path &p)
{
	std::ifstream f(p, std::ios::binary);
	return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void write_file(const fs::path &p, const std::string &text)
{
	fs::create_directories(p.parent_path());
	std::ofstream f(p, std::ios::binary | std::ios::trunc);
	f << text;
}

static std::string u8(const fs::path &p)
{
	return p.u8string();
}

// A scratch folder that goes away with the object.
struct Scratch {
	fs::path dir;
	Scratch()
	{
		std::string d, err;
		if (!make_update_work_dir(d, err))
			std::printf("  cannot make a scratch folder: %s\n", err.c_str());
		dir = fs::u8path(d);
	}
	~Scratch()
	{
		std::error_code ec;
		fs::permissions(dir, fs::perms::owner_all, fs::perm_options::add, ec);
		fs::remove_all(dir, ec);
	}
};

// ---------------------------------------------------------------------------------------------------

static void test_classify()
{
	std::printf("install location from the module path\n");
	auto c = [](HostOs os, const char *p) { return classify_install(os, p); };

	InstallTarget t = c(HostOs::Windows, "C:\\ProgramData\\obs-studio\\plugins\\audio-cleaner\\bin\\64bit\\audio-cleaner.dll");
	CHECK(t.kind == InstallKind::WindowsFolder && t.key == kKeyWindows && t.root == "C:\\ProgramData\\obs-studio\\plugins\\audio-cleaner");
	t = c(HostOs::Windows, "C:/Users/Jörg Ünal/AppData/Roaming/obs-studio/plugins/my cleaner/bin/64bit/audio-cleaner.DLL");
	CHECK(t.kind == InstallKind::WindowsFolder && t.root == "C:\\Users\\Jörg Ünal\\AppData\\Roaming\\obs-studio\\plugins\\my cleaner");
	t = c(HostOs::Windows, "\\\\?\\D:\\Plugins\\audio-cleaner\\bin\\64bit\\audio-cleaner.dll");
	CHECK(t.kind == InstallKind::WindowsFolder && t.root == "D:\\Plugins\\audio-cleaner");
	t = c(HostOs::Windows, "\\\\server\\share\\obs\\audio-cleaner\\bin\\64bit\\audio-cleaner.dll");
	CHECK(t.kind == InstallKind::WindowsFolder && t.root == "\\\\server\\share\\obs\\audio-cleaner");
	for (const char *p : {"C:\\Program Files\\obs-studio\\obs-plugins\\64bit\\audio-cleaner.dll",  // the old layout
			      "C:\\bin\\64bit\\audio-cleaner.dll",                                      // the drive would be the plugin folder
			      "\\bin\\64bit\\audio-cleaner.dll", "bin\\64bit\\audio-cleaner.dll", "C:\\x\\bin\\32bit\\a.dll", "C:\\x\\bin\\64bit\\a.so",
			      "C:\\x\\\\bin\\64bit\\a.dll", "", "audio-cleaner.dll"})
		CHECK(c(HostOs::Windows, p).kind == InstallKind::Unknown);

	t = c(HostOs::MacOS, "/Users/me/Library/Application Support/obs-studio/plugins/audio-cleaner.plugin/Contents/MacOS/audio-cleaner");
	CHECK(t.kind == InstallKind::MacBundle && t.key == kKeyMac &&
	      t.root == "/Users/me/Library/Application Support/obs-studio/plugins/audio-cleaner.plugin");
	t = c(HostOs::MacOS, "/Library/Application Support/obs-studio/plugins/audio-cleaner.plugin/Contents/MacOS/audio-cleaner");
	CHECK(t.kind == InstallKind::MacBundle && t.root == "/Library/Application Support/obs-studio/plugins/audio-cleaner.plugin");
	for (const char *p : {"/x/audio-cleaner.plugin/Contents/Frameworks/audio-cleaner", "/x/audio-cleaner/Contents/MacOS/audio-cleaner",
			      "/x/.plugin/Contents/MacOS/a", "/x/a.plugin/Contents/MacOS/", "x/a.plugin/Contents/MacOS/a", "/x/a.plugin/MacOS/a", ""})
		CHECK(c(HostOs::MacOS, p).kind == InstallKind::Unknown);

	t = c(HostOs::Linux, "/usr/lib/x86_64-linux-gnu/obs-plugins/audio-cleaner.so");
	CHECK(t.kind == InstallKind::LinuxPackage && t.key == kKeyDebian);
	t = c(HostOs::Linux, "/home/me/.config/obs-studio/plugins/audio-cleaner/bin/64bit/audio-cleaner.so");
	CHECK(t.kind == InstallKind::LinuxFolder && t.key == kKeyLinuxOther && t.root == "/home/me/.config/obs-studio/plugins/audio-cleaner");
	t = c(HostOs::Linux, "/home/me/.var/app/com.obsproject.Studio/config/obs-studio/plugins/audio-cleaner/bin/64bit/audio-cleaner.so");
	CHECK(t.kind == InstallKind::LinuxFolder && t.root.size() > 20);
	t = c(HostOs::Linux, "/usr/local/lib/obs-plugins/audio-cleaner.so");
	CHECK(t.kind == InstallKind::Unknown);
	for (const char *p : {"/bin/64bit/audio-cleaner.so", "/home/me/audio-cleaner.so", "home/me/a/bin/64bit/a.so", "/home//me/a/bin/64bit/a.so", ""})
		CHECK(c(HostOs::Linux, p).kind == InstallKind::Unknown);
}

static void test_quoting_and_text()
{
	std::printf("script quoting and text\n");
	CHECK(ps_quote("a b") == "'a b'");
	CHECK(ps_quote("it's") == "'it''s'");
	CHECK(ps_quote("") == "''");
	CHECK(ps_quote("\xE2\x80\x98x\xE2\x80\x99\xE2\x80\x9A\xE2\x80\x9B") ==
	      "'\xE2\x80\x98\xE2\x80\x98x\xE2\x80\x99\xE2\x80\x99\xE2\x80\x9A\xE2\x80\x9A\xE2\x80\x9B\xE2\x80\x9B'");
	CHECK(ps_quote("\xE2\x80\x9C" "x" "\xE2\x80\x9D") == "'\xE2\x80\x9C" "x" "\xE2\x80\x9D'");  // double curly quotes are not quotes in '...'
	CHECK(ps_quote("$(evil) `x` $env:A ; & \"") == "'$(evil) `x` $env:A ; & \"'");  // nothing is expanded in '...'
	CHECK(sh_quote("a b") == "'a b'");
	CHECK(sh_quote("it's") == "'it'\\''s'");
	CHECK(sh_quote("''") == "''\\'''\\'''");
	CHECK(sh_quote("$(evil) `x` ; &") == "'$(evil) `x` ; &'");
	CHECK(plain_path_text("C:\\a b\\c") && plain_path_text("/tmp/ü"));
	CHECK(!plain_path_text("") && !plain_path_text("a\nb") && !plain_path_text("a\rb") && !plain_path_text(std::string("a\0b", 3)) &&
	      !plain_path_text("a\tb") && !plain_path_text(std::string(1001, 'a')));
	CHECK(utf16le_base64("A") == "QQA=");
	CHECK(utf16le_base64("hi") == "aABpAA==");
	CHECK(utf16le_base64("\xC3\xBC") == "/AA=");                    // u-umlaut
	CHECK(utf16le_base64("\xF0\x9F\x98\x80") == "PdgA3g==");        // U+1F600 as a surrogate pair

	WindowsScriptArgs w;
	w.pid = 4242;
	w.zip = "C:\\Users\\O'Brien\\AppData\\Local\\Temp\\acw-update-1\\p.zip";
	w.zip_sha512 = std::string(128, 'c');
	w.root = "C:\\ProgramData\\obs-studio\\plugins\\audio-cleaner";
	w.stage = "C:\\ProgramData\\obs-studio\\plugins\\.acw-update-4242";
	w.work_dir = "C:\\Users\\O'Brien\\AppData\\Local\\Temp\\acw-update-1";
	w.log = "C:\\Users\\O'Brien\\AppData\\Local\\Temp\\audio-cleaner-update.log";
	const std::string ws = windows_update_script(w);
	CHECK(contains(ws, "$ObsPid = 4242\n"));
	CHECK(contains(ws, "$Zip = 'C:\\Users\\O''Brien\\AppData\\Local\\Temp\\acw-update-1\\p.zip'\n"));
	CHECK(contains(ws, "$Root = 'C:\\ProgramData\\obs-studio\\plugins\\audio-cleaner'\n"));
	CHECK(contains(ws, "$ZipSha512 = '" + std::string(128, 'c') + "'\n"));
	CHECK(contains(ws, "Expand-Archive -LiteralPath") && contains(ws, "Move-Item -LiteralPath") && !contains(ws, "Invoke-Expression"));
	CHECK(!contains(ws, "Invoke-WebRequest") && !contains(ws, "DownloadString") && !contains(ws, "http"));  // nothing from the network
	// The text after the parameters is the same whatever the paths are.
	WindowsScriptArgs w2 = w;
	w2.pid = 7;
	w2.zip = "D:\\\xC3\xBC'\xE2\x80\x99 x\\p.zip";
	w2.root = "D:\\r";
	const std::string ws2 = windows_update_script(w2);
	const size_t b1 = ws.find("\nfunction Write-Log"), b2 = ws2.find("\nfunction Write-Log");
	CHECK(b1 != std::string::npos && b1 == ws.find("\nfunction") && ws.substr(b1) == ws2.substr(b2));
	CHECK(contains(ws2, "$Zip = 'D:\\\xC3\xBC''\xE2\x80\x99\xE2\x80\x99 x\\p.zip'\n"));
	const std::string g = windows_guarded_command("C:\\T'emp\\acw-update.ps1", std::string(128, 'd'));
	CHECK(contains(g, "ReadAllBytes('C:\\T''emp\\acw-update.ps1')") && contains(g, "-ne '" + std::string(128, 'd') + "'") &&
	      contains(g, "GetString($b, 3, $b.Length - 3)"));

	PosixScriptArgs p;
	p.pid = 99;
	p.target = "/home/me/it's here/audio-cleaner";
	p.staged = "/home/me/it's here/.acw-update-x/audio-cleaner";
	p.stage_dir = "/home/me/it's here/.acw-update-x";
	p.work_dir = "/tmp/acw-update-y";
	p.log = "/tmp/audio-cleaner-update.log";
	const std::string ps = posix_update_script(p);
	CHECK(ps.compare(0, 10, "#!/bin/sh\n") == 0);
	CHECK(contains(ps, "PID='99'\n"));
	CHECK(contains(ps, "TARGET='/home/me/it'\\''s here/audio-cleaner'\n"));
	CHECK(!contains(ps, "curl") && !contains(ps, "wget") && !contains(ps, "http"));
	PosixScriptArgs p2 = p;
	p2.pid = 5;
	p2.target = "/x y/z";
	const std::string ps2 = posix_update_script(p2);
	CHECK(ps.substr(ps.find("\nlog()")) == ps2.substr(ps2.find("\nlog()")));
}

#ifndef _WIN32

// Runs `printf %s <quoted>` in a real shell: whatever comes out is what the script would see.
static std::string shell_echo(const std::string &quoted)
{
	Scratch s;
	write_file(s.dir / "q.sh", "printf '%s' " + quoted + "\n");
	const std::string out = u8(s.dir / "out.txt");
	const int rc = std::system(("sh '" + u8(s.dir / "q.sh") + "' > '" + out + "'").c_str());
	return rc == 0 ? read_file(out) : std::string("<failed>");
}

static pid_t spawn_sleeper(int seconds)
{
	const pid_t pid = fork();
	if (pid == 0) {
		execlp("sleep", "sleep", std::to_string(seconds).c_str(), static_cast<char *>(nullptr));
		_exit(127);
	}
	return pid;
}

static const char *kNasty[] = {"plain", "with space", "it's", "''", "a\"b", "$HOME", "$(echo hi)", "`echo hi`", "a;b&c|d", "*", "~", "-n", "\\", "\\n",
			       "\xC3\xBC\xC3\xB1 \xE2\x80\x99", "\xE6\x97\xA5\xE6\x9C\xAC", "a  b", "!", "#x", "a=b"};

static void test_posix_scripts()
{
	std::printf("POSIX script: quoting in a real shell, swap, undo\n");
	for (const char *n : kNasty)
		CHECK(shell_echo(sh_quote(n)) == n);

	// A plugin folder in a path full of awkward characters; the swap happens only after the process is gone.
	{
		Scratch s;
		const fs::path parent = s.dir / fs::u8path("it's \xC3\xBC \"q\" $x `y`");
		const fs::path target = parent / "audio-cleaner", stage = parent / ".acw-update-t", work = s.dir / "work";
		write_file(target / "bin" / "64bit" / "audio-cleaner.so", "old");
		write_file(stage / "audio-cleaner" / "bin" / "64bit" / "audio-cleaner.so", "new");
		write_file(work / "keep.txt", "x");
		PosixScriptArgs a;
		const pid_t sleeper = spawn_sleeper(1);
		a.pid = sleeper;
		a.target = u8(target);
		a.staged = u8(stage / "audio-cleaner");
		a.stage_dir = u8(stage);
		a.work_dir = u8(work);
		a.log = u8(s.dir / "log.txt");
		write_file(work / "s.sh", posix_update_script(a));
		const int rc = std::system(("sh '" + u8(work / "s.sh") + "' >/dev/null 2>&1 &").c_str());
		CHECK(rc == 0);
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		CHECK(read_file(target / "bin" / "64bit" / "audio-cleaner.so") == "old");  // still waiting for the process
		int status = 0;
		waitpid(sleeper, &status, 0);
		CHECK(wait_until([&] { return read_file(target / "bin" / "64bit" / "audio-cleaner.so") == "new" && !fs::exists(stage); }, 15000));
		CHECK(wait_until([&] { return !fs::exists(work); }, 5000));
		CHECK(contains(read_file(s.dir / "log.txt"), "installed"));
		CHECK(!fs::exists(parent / "audio-cleaner" / "old"));
	}

	{  // the new files are gone: the old plugin stays
		Scratch s;
		const fs::path target = s.dir / "p" / "audio-cleaner", stage = s.dir / "p" / ".acw-update-t", work = s.dir / "work";
		write_file(target / "f", "old");
		fs::create_directories(stage);
		PosixScriptArgs a;
		a.pid = 0;
		a.target = u8(target);
		a.staged = u8(stage / "audio-cleaner");
		a.stage_dir = u8(stage);
		a.work_dir = u8(work);
		a.log = u8(s.dir / "log.txt");
		write_file(work / "s.sh", posix_update_script(a));
		CHECK(std::system(("sh '" + u8(work / "s.sh") + "'").c_str()) == 0);
		CHECK(read_file(target / "f") == "old" && !fs::exists(stage));
		CHECK(contains(read_file(s.dir / "log.txt"), "failed"));
	}
	{  // no old plugin any more (removed meanwhile): the new one is simply put in
		Scratch s;
		const fs::path target = s.dir / "p" / "audio-cleaner", stage = s.dir / "p" / ".acw-update-t", work = s.dir / "work";
		write_file(stage / "audio-cleaner" / "f", "new");
		PosixScriptArgs a;
		a.target = u8(target);
		a.staged = u8(stage / "audio-cleaner");
		a.stage_dir = u8(stage);
		a.work_dir = u8(work);
		a.log = u8(s.dir / "log.txt");
		write_file(work / "s.sh", posix_update_script(a));
		CHECK(std::system(("sh '" + u8(work / "s.sh") + "'").c_str()) == 0);
		CHECK(read_file(target / "f") == "new");
	}
#ifdef __linux__
	if (geteuid() != 0 && fs::is_directory("/dev/shm")) {
		// The second move fails half way (the new files are on another file system and one cannot be read): the old
		// plugin comes back and nothing half-made is left in its place.
		Scratch s;
		std::error_code ec;
		const fs::path shm = fs::path("/dev/shm") / ("acw-test-" + std::to_string(getpid()));
		fs::create_directories(shm / "audio-cleaner", ec);
		write_file(shm / "audio-cleaner" / "a", "new-a");
		write_file(shm / "audio-cleaner" / "z", "new-z");
		fs::permissions(shm / "audio-cleaner" / "z", fs::perms::none, ec);
		const fs::path target = s.dir / "p" / "audio-cleaner", stage = s.dir / "p" / ".acw-update-t", work = s.dir / "work";
		write_file(target / "f", "old");
		fs::create_directories(stage);
		PosixScriptArgs a;
		a.target = u8(target);
		a.staged = u8(shm / "audio-cleaner");
		a.stage_dir = u8(stage);
		a.work_dir = u8(work);
		a.log = u8(s.dir / "log.txt");
		write_file(work / "s.sh", posix_update_script(a));
		CHECK(std::system(("sh '" + u8(work / "s.sh") + "' 2>/dev/null").c_str()) == 0);
		CHECK(read_file(target / "f") == "old" && !fs::exists(target / "a") && !fs::exists(target / "old"));
		CHECK(contains(read_file(s.dir / "log.txt"), "put back"));
		fs::permissions(shm / "audio-cleaner" / "z", fs::perms::owner_all, ec);
		fs::remove_all(shm, ec);
	}
#endif
}

// ---------------------------------------------------------------------------------------------------
// apply_update() with real archives

static bool make_archive(const fs::path &src_root, const std::string &top, const fs::path &out)
{
	return std::system(("tar -cJf '" + u8(out) + "' -C '" + u8(src_root) + "' '" + top + "'").c_str()) == 0;
}

static ApplyRequest request_for(const fs::path &archive, const InstallTarget &t, const fs::path &work, const fs::path &log)
{
	ApplyRequest r;
	r.file = u8(archive);
	r.file_sha512 = sha512_hex(read_file(archive));
	r.target = t;
	r.work_dir = u8(work);
	r.log = u8(log);
	r.open_ui = false;
	return r;
}

static void test_apply_posix()
{
	std::printf("apply_update: unpack, swap on exit, undo\n");
	for (const bool mac : {false, true}) {
		const HostOs os = mac ? HostOs::MacOS : HostOs::Linux;
		const std::string top = mac ? "audio-cleaner.plugin" : "audio-cleaner";
		auto lay = [&](const fs::path &root, const std::string &payload) {
			if (mac) {
				write_file(root / "Contents" / "Info.plist", "plist");
				write_file(root / "Contents" / "MacOS" / "audio-cleaner", payload);
			} else {
				write_file(root / "bin" / "64bit" / "audio-cleaner.so", payload);
				write_file(root / "data" / "locale" / "en-US.ini", "texts " + payload);
			}
		};
		auto probe = [&](const fs::path &root) {
			return read_file(mac ? root / "Contents" / "MacOS" / "audio-cleaner" : root / "bin" / "64bit" / "audio-cleaner.so");
		};

		Scratch s;
		const fs::path plugins = s.dir / "my plugins" / "obs-studio" / "plugins";
		const fs::path installed = plugins / top;
		const std::string module_path = mac ? u8(installed / "Contents" / "MacOS" / "audio-cleaner") : u8(installed / "bin" / "64bit" / "audio-cleaner.so");
		lay(installed, "old");
		write_file(plugins / "other-plugin" / "keep", "other");
		fs::create_directories(s.dir / "pkg");
		lay(s.dir / "pkg" / top, "new");
		CHECK(make_archive(s.dir / "pkg", top, s.dir / "update.tar.xz"));

		const InstallTarget t = classify_install(os, module_path);
		CHECK(t.kind == (mac ? InstallKind::MacBundle : InstallKind::LinuxFolder) && t.root == u8(installed));
		std::string work, err;
		CHECK(make_update_work_dir(work, err));
		ApplyRequest req = request_for(s.dir / "update.tar.xz", t, fs::u8path(work), s.dir / "log.txt");
		const pid_t sleeper = spawn_sleeper(1);
		req.pid = static_cast<unsigned long>(sleeper);
		const ApplyResult r = apply_update(req);
		CHECK(r.outcome == ApplyOutcome::InstallsOnExit);
		if (r.outcome != ApplyOutcome::InstallsOnExit)
			std::printf("  error: %s\n", r.error.c_str());
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		CHECK(probe(installed) == "old");  // OBS is still running: nothing has been touched
		{
			bool stage_found = false;
			for (const auto &e : fs::directory_iterator(plugins))
				stage_found = stage_found || e.path().filename().u8string().compare(0, 12, ".acw-update-") == 0;
			CHECK(stage_found);  // the new files wait next to the old ones
		}
		int status = 0;
		waitpid(sleeper, &status, 0);  // "OBS exits"
		CHECK(wait_until([&] { return probe(installed) == "new"; }, 15000));
		CHECK(wait_until([&] { return !fs::exists(work); }, 5000));
		CHECK(read_file(plugins / "other-plugin" / "keep") == "other");
		CHECK(wait_until([&] {
			for (const auto &e : fs::directory_iterator(plugins))
				if (e.path().filename().u8string().compare(0, 12, ".acw-update-") == 0)
					return false;
			return true;
		}, 5000));
		CHECK(contains(read_file(s.dir / "log.txt"), "installed"));

		// immediate mode (OBS in a sandbox): swapped when the call returns
		fs::create_directories(s.dir / "pkg2");
		lay(s.dir / "pkg2" / top, "newer");
		CHECK(make_archive(s.dir / "pkg2", top, s.dir / "update2.tar.xz"));
		CHECK(make_update_work_dir(work, err));
		ApplyRequest req2 = request_for(s.dir / "update2.tar.xz", t, fs::u8path(work), s.dir / "log.txt");
		req2.immediate = true;
		req2.pid = static_cast<unsigned long>(getpid());
		const ApplyResult r2 = apply_update(req2);
		CHECK(r2.outcome == ApplyOutcome::InstalledNow && probe(installed) == "newer");

		// an archive that is not one, one without the plugin, one with the wrong layout: the plugin stays, nothing is left
		auto count_entries = [&] {
			int n = 0;
			for (const auto &e : fs::directory_iterator(plugins)) {
				(void)e;
				++n;
			}
			return n;
		};
		const int before = count_entries();
		write_file(s.dir / "junk.tar.xz", "this is not an archive");
		CHECK(make_update_work_dir(work, err));
		ApplyResult bad = apply_update(request_for(s.dir / "junk.tar.xz", t, fs::u8path(work), s.dir / "log.txt"));
		CHECK(bad.outcome == ApplyOutcome::Failed && contains(bad.error, "unpacked"));
		fs::create_directories(s.dir / "pkg3" / top);
		write_file(s.dir / "pkg3" / top / "readme", "no plugin in here");
		CHECK(make_archive(s.dir / "pkg3", top, s.dir / "update3.tar.xz"));
		bad = apply_update(request_for(s.dir / "update3.tar.xz", t, fs::u8path(work), s.dir / "log.txt"));
		CHECK(bad.outcome == ApplyOutcome::Failed && contains(bad.error, "does not contain"));
		fs::create_directories(s.dir / "pkg4" / "something-else");
		write_file(s.dir / "pkg4" / "something-else" / "x", "x");
		CHECK(make_archive(s.dir / "pkg4", "something-else", s.dir / "update4.tar.xz"));
		bad = apply_update(request_for(s.dir / "update4.tar.xz", t, fs::u8path(work), s.dir / "log.txt"));
		CHECK(bad.outcome == ApplyOutcome::Failed);
		CHECK(probe(installed) == "newer" && count_entries() == before);

		// the plugin folder is a link, or does not exist: refused
		fs::create_directory_symlink(installed, plugins / ("link-" + top));
		InstallTarget lt = t;
		lt.root = u8(plugins / ("link-" + top));
		bad = apply_update(request_for(s.dir / "update2.tar.xz", lt, fs::u8path(work), s.dir / "log.txt"));
		CHECK(bad.outcome == ApplyOutcome::Failed);
		lt.root = u8(plugins / ("missing-" + top));
		bad = apply_update(request_for(s.dir / "update2.tar.xz", lt, fs::u8path(work), s.dir / "log.txt"));
		CHECK(bad.outcome == ApplyOutcome::Failed);
		fs::remove(plugins / ("link-" + top));

		// not writable: nothing is changed, the user is told where the plugin goes
		if (geteuid() != 0) {
			std::error_code ec;
			fs::permissions(plugins, fs::perms::owner_read | fs::perms::owner_exec, ec);
			const ApplyResult ro = apply_update(request_for(s.dir / "update2.tar.xz", t, fs::u8path(work), s.dir / "log.txt"));
			fs::permissions(plugins, fs::perms::owner_all, ec);
			CHECK(ro.outcome == ApplyOutcome::FolderShown && ro.detail == u8(plugins));
			CHECK(probe(installed) == "newer");
		}
		// bad requests
		ApplyRequest nohash = request_for(s.dir / "update2.tar.xz", t, fs::u8path(work), s.dir / "log.txt");
		nohash.file_sha512 = "abc";
		CHECK(apply_update(nohash).outcome == ApplyOutcome::Failed);
		ApplyRequest evil = request_for(s.dir / "update2.tar.xz", t, fs::u8path(work), s.dir / "log.txt");
		evil.target.root = u8(installed) + "\nrm -rf /";
		CHECK(apply_update(evil).outcome == ApplyOutcome::Failed);
	}

	{  // a package install: handed to the system's installer, nothing else happens
		Scratch s;
		write_file(s.dir / "p.deb", "deb");
		const InstallTarget t = classify_install(HostOs::Linux, "/usr/lib/x86_64-linux-gnu/obs-plugins/audio-cleaner.so");
		ApplyRequest r = request_for(s.dir / "p.deb", t, s.dir, s.dir / "log.txt");
		CHECK(apply_update(r).outcome == ApplyOutcome::InstallerOpened);
	}
	{  // somewhere we do not know: the file is shown
		Scratch s;
		write_file(s.dir / "p.tar.xz", "x");
		ApplyRequest r = request_for(s.dir / "p.tar.xz", InstallTarget(), s.dir, s.dir / "log.txt");
		CHECK(apply_update(r).outcome == ApplyOutcome::FolderShown);
	}
	{  // work folders: private and different every time
		std::string a, b, err;
		CHECK(make_update_work_dir(a, err) && make_update_work_dir(b, err) && a != b);
		CHECK((fs::status(fs::u8path(a)).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
		std::error_code ec;
		fs::remove_all(fs::u8path(a), ec);
		fs::remove_all(fs::u8path(b), ec);
	}
}

#else  // _WIN32

static void test_apply_windows()
{
	std::printf("apply_update on Windows: the real PowerShell swaps the folders after the process ends\n");
	Scratch s;
	const fs::path plugins = s.dir / "my plugins O'Brien \xC3\xBC";
	const fs::path installed = plugins / "audio-cleaner";
	write_file(installed / "bin" / "64bit" / "audio-cleaner.dll", "old");
	write_file(plugins / "other" / "keep", "other");
	fs::create_directories(s.dir / "pkg" / "audio-cleaner" / "bin" / "64bit");
	write_file(s.dir / "pkg" / "audio-cleaner" / "bin" / "64bit" / "audio-cleaner.dll", "new");
	write_file(s.dir / "pkg" / "audio-cleaner" / "data" / "locale" / "en-US.ini", "texts");
	// Windows' own tar.exe (bsdtar, part of Windows 10 and later) writes zip files with -a. Named by its full path: in
	// a Git Bash (as on the CI runners) the first tar.exe on PATH is GNU tar, which takes the C: of the archive's path
	// for a remote host and writes no zip.
	const char *sysroot = std::getenv("SystemRoot");
	const std::string tar = std::string(sysroot ? sysroot : "C:\\Windows") + "\\System32\\tar.exe";
	const std::string zip = u8(s.dir / "update.zip");
	// cmd.exe /c drops the outer quotes of the whole line: the program and the paths keep theirs
	CHECK(std::system(("\"\"" + tar + "\" -a -c -f \"" + zip + "\" -C \"" + u8(s.dir / "pkg") + "\" audio-cleaner\"").c_str()) == 0);

	const InstallTarget t = classify_install(HostOs::Windows, u8(installed / "bin" / "64bit" / "audio-cleaner.dll"));
	CHECK(t.kind == InstallKind::WindowsFolder);
	std::string work, err;
	CHECK(make_update_work_dir(work, err));
	ApplyRequest req;
	req.file = zip;
	req.file_sha512 = sha512_hex(read_file(fs::u8path(zip)));
	req.target = t;
	req.work_dir = work;
	req.log = u8(s.dir / "log.txt");
	req.open_ui = false;

	// a process to wait for: lives three seconds
	STARTUPINFOA si{};
	si.cb = sizeof si;
	PROCESS_INFORMATION pi{};
	char cmd[] = "cmd.exe /c ping -n 4 127.0.0.1 > nul";
	CHECK(CreateProcessA(nullptr, cmd, nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi));
	req.pid = pi.dwProcessId;
	const ApplyResult r = apply_update(req);
	CHECK(r.outcome == ApplyOutcome::InstallsOnExit);
	if (r.outcome != ApplyOutcome::InstallsOnExit)
		std::printf("  error: %s\n", r.error.c_str());
	std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	CHECK(read_file(installed / "bin" / "64bit" / "audio-cleaner.dll") == "old");
	WaitForSingleObject(pi.hProcess, 20000);
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	const bool done = wait_until([&] { return read_file(installed / "bin" / "64bit" / "audio-cleaner.dll") == "new"; }, 90000);
	CHECK(done);
	if (!done)
		std::printf("  log: %s\n", read_file(s.dir / "log.txt").c_str());
	CHECK(read_file(installed / "data" / "locale" / "en-US.ini") == "texts");
	CHECK(read_file(plugins / "other" / "keep") == "other");
	CHECK(wait_until([&] { return !fs::exists(fs::u8path(work)); }, 20000));
	bool stage_left = false;
	for (const auto &e : fs::directory_iterator(plugins))
		stage_left = stage_left || e.path().filename().u8string().compare(0, 12, ".acw-update-") == 0;
	CHECK(!stage_left);
}

#endif

int main()
{
	test_classify();
	test_quoting_and_text();
#ifndef _WIN32
	test_posix_scripts();
	test_apply_posix();
#else
	test_apply_windows();
#endif
	std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
	return g_failures ? 1 : 0;
}
