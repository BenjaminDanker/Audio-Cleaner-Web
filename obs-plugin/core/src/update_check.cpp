// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/update_check.hpp"

#include "acw/update_verify.hpp"
#include "http_get.hpp"

#include <ixwebsocket/IXNetSystem.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>

namespace acw {

// ---------------------------------------------------------------------------------------------------
// versions

bool parse_version(const std::string &text, Version &out)
{
	Version v;
	int parts = 0, digits = 0;
	long cur = 0;
	for (size_t i = 0; i <= text.size(); ++i) {
		const char c = i < text.size() ? text[i] : '.';  // the end closes the last part
		if (c >= '0' && c <= '9') {
			if (++digits > 6)
				return false;
			cur = cur * 10 + (c - '0');
		} else if (c == '.' && digits > 0) {
			if (parts >= 4)
				return false;
			v.part[parts++] = static_cast<int>(cur);
			cur = 0;
			digits = 0;
		} else {
			return false;
		}
	}
	out = v;
	return parts >= 1;
}

int compare_versions(const Version &a, const Version &b)
{
	for (int i = 0; i < 4; ++i)
		if (a.part[i] != b.part[i])
			return a.part[i] < b.part[i] ? -1 : 1;
	return 0;
}

bool is_newer_version(const std::string &latest, const std::string &current)
{
	Version l, c;
	return parse_version(latest, l) && parse_version(current, c) && compare_versions(l, c) > 0;
}

bool valid_update_file_name(const std::string &name)
{
	if (name.empty() || name.size() > 120)
		return false;
	for (size_t i = 0; i < name.size(); ++i) {
		const char c = name[i];
		const bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
		if (!(alnum || c == '.' || c == '_' || c == '-'))
			return false;
		if (i == 0 && !alnum)
			return false;
		if (c == '.' && i + 1 < name.size() && name[i + 1] == '.')
			return false;
	}
	return true;
}

const char *update_file_suffix(const std::string &key)
{
	if (key == kKeyWindows)
		return ".zip";
	if (key == kKeyMac || key == kKeyLinuxOther)
		return ".tar.xz";
	if (key == kKeyDebian)
		return ".deb";
	return "";
}

namespace {

bool ends_with(const std::string &s, const std::string &suffix)
{
	return s.size() > suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool read_file_entry(const std::string &key, const nlohmann::json &j, UpdateFile &out)
{
	const std::string suffix = update_file_suffix(key);
	if (suffix.empty() || !j.is_object())
		return false;
	const auto name = j.find("name");
	const auto size = j.find("size");
	const auto s512 = j.find("sha512");
	const auto s256 = j.find("sha256");
	if (name == j.end() || !name->is_string() || size == j.end() || !size->is_number_unsigned() || s512 == j.end() ||
	    !s512->is_string())
		return false;
	UpdateFile f;
	f.name = name->get<std::string>();
	f.size = size->get<uint64_t>();
	f.sha512 = s512->get<std::string>();
	if (s256 != j.end()) {
		if (!s256->is_string() || !is_hex_digest(s256->get<std::string>(), 64))
			return false;
		f.sha256 = s256->get<std::string>();
	}
	if (!valid_update_file_name(f.name) || !ends_with(f.name, suffix) || f.size == 0 || f.size > kMaxUpdateFileBytes ||
	    !is_hex_digest(f.sha512, 128))
		return false;
	out = f;
	return true;
}

}  // namespace

bool parse_update_manifest(const std::string &body, UpdateManifest &out)
{
	const nlohmann::json j = nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
	if (!j.is_object())
		return false;
	const auto ver = j.find("version");
	Version unused;
	if (ver == j.end() || !ver->is_string() || !parse_version(ver->get<std::string>(), unused))
		return false;
	const auto files = j.find("files");
	if (files == j.end() || !files->is_object())
		return false;
	UpdateManifest m;
	m.version = ver->get<std::string>();
	for (auto it = files->begin(); it != files->end(); ++it) {
		UpdateFile f;
		if (read_file_entry(it.key(), it.value(), f))
			m.files[it.key()] = f;
	}
	out = std::move(m);
	return true;
}

// ---------------------------------------------------------------------------------------------------
// the check

struct UpdateCheck::Impl {
	detail::HttpGet http;

	bool fetch(const std::string &url, size_t max_bytes, const UpdateCheckOptions &opt, std::chrono::steady_clock::time_point deadline,
		   std::string &body, std::string &error)
	{
		detail::HttpGetOptions o;
		o.user_agent = opt.user_agent;
		o.accept = "application/json, application/octet-stream";
		o.ca_file = opt.ca_file;
		o.connect_timeout_s = opt.connect_timeout_s;
		o.transfer_timeout_s = opt.transfer_timeout_s;
		const auto left = std::chrono::duration_cast<std::chrono::seconds>(deadline - std::chrono::steady_clock::now()).count();
		o.total_timeout_s = static_cast<int>(std::max<long long>(1, left));
		o.max_redirects = opt.max_redirects;
		o.allow_local_http = opt.allow_local_http;
		std::string got;
		bool too_big = false;
		detail::HttpReceiver rx;
		rx.begin = [&] {
			got.clear();
			too_big = false;
		};
		rx.chunk = [&](const std::string &chunk) {
			got += chunk;
			if (got.size() > max_bytes) {
				too_big = true;
				return false;
			}
			return true;
		};
		if (!http.get(url, o, rx, error)) {
			if (too_big)
				error = "reply too large";
			return false;
		}
		body = std::move(got);
		return true;
	}
};

UpdateCheck::UpdateCheck() : impl_(std::make_shared<Impl>()) {}
UpdateCheck::~UpdateCheck() = default;

void UpdateCheck::cancel()
{
	impl_->http.cancel();
}

UpdateCheckResult UpdateCheck::run(const std::string &current_version, const UpdateCheckOptions &opt)
{
	UpdateCheckResult r;
	Version cur;
	if (!parse_version(current_version, cur)) {
		r.error = "own version number is not valid: " + current_version;
		return r;
	}
	if (opt.base_url.empty() || opt.base_url.back() != '/') {
		r.error = "bad update address";
		return r;
	}
	ix::initNetSystem();  // counted: the sessions' own initialisation is unaffected
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(std::max(1, opt.total_timeout_s));
	std::string manifest, sig;
	bool ok = impl_->fetch(opt.base_url + "manifest.json", opt.max_manifest_bytes, opt, deadline, manifest, r.error);
	if (ok)
		ok = impl_->fetch(opt.base_url + "manifest.json.sig", 64, opt, deadline, sig, r.error);
	ix::uninitNetSystem();
	if (!ok)
		return r;
	if (!verify_ed25519(manifest, sig, opt.public_key.empty() ? std::string(kUpdatePublicKeyBase64) : opt.public_key)) {
		r.error = "the signature of the update list does not match";
		return r;
	}
	if (!parse_update_manifest(manifest, r.manifest)) {
		r.error = "unreadable update list";
		return r;
	}
	Version latest;
	parse_version(r.manifest.version, latest);
	r.status = compare_versions(latest, cur) > 0 ? UpdateStatus::UpdateAvailable : UpdateStatus::UpToDate;
	return r;
}

}  // namespace acw
