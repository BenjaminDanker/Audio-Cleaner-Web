// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/update_check.hpp"

#include "acw/session.hpp"
#include "tls_options.hpp"

#include <ixwebsocket/IXHttp.h>
#include <ixwebsocket/IXHttpClient.h>
#include <ixwebsocket/IXNetSystem.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <mutex>

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

bool release_url_allowed(const std::string &url)
{
	const std::string prefix = kReleasePagePrefix;
	if (url.size() <= prefix.size() || url.size() > 300 || url.compare(0, prefix.size(), prefix) != 0)
		return false;
	for (char ch : url) {
		const unsigned char c = static_cast<unsigned char>(ch);
		if (c <= 0x20 || c >= 0x7f || c == '"' || c == '\'' || c == '<' || c == '>' || c == '`' || c == '\\' || c == '^' ||
		    c == '|' || c == '{' || c == '}')
			return false;
	}
	return true;
}

bool parse_update_json(const std::string &body, UpdateInfo &out)
{
	const nlohmann::json j = nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
	if (!j.is_object())
		return false;
	const auto ver = j.find("version");
	Version unused;
	if (ver == j.end() || !ver->is_string() || !parse_version(ver->get<std::string>(), unused))
		return false;
	UpdateInfo info;
	info.version = ver->get<std::string>();
	info.url = kReleasePageUrl;
	const auto url = j.find("url");
	if (url != j.end() && url->is_string() && release_url_allowed(url->get<std::string>()))
		info.url = url->get<std::string>();
	out = info;
	return true;
}

// ---------------------------------------------------------------------------------------------------
// the request

namespace {

// "https://host/path?query" -> "https://host" (empty when it is not an absolute address).
std::string origin_of(const std::string &url)
{
	const size_t scheme = url.find("://");
	if (scheme == std::string::npos)
		return {};
	const size_t slash = url.find_first_of("/?#", scheme + 3);
	return slash == std::string::npos ? url : url.substr(0, slash);
}

// A redirect target as an absolute address: the target itself, or a path on the same server.
std::string resolve_location(const std::string &base, const std::string &location)
{
	if (location.find("://") != std::string::npos)
		return location;
	if (!location.empty() && location[0] == '/' && !(location.size() > 1 && location[1] == '/'))
		return origin_of(base) + location;
	return {};  // "//host/x", "x/y": not worth supporting
}

// https:// anywhere, http:// only to this computer / the local network and only when the caller allows it.
bool url_allowed(const std::string &url, const UpdateCheckOptions &opt)
{
	if (url.size() > 2000)
		return false;
	if (url.compare(0, 8, "https://") == 0)
		return url.size() > 8;
	if (opt.allow_local_http && url.compare(0, 7, "http://") == 0)
		return server_url_allowed("ws://" + url.substr(7));
	return false;
}

const char *error_name(ix::HttpErrorCode c)
{
	switch (c) {
	case ix::HttpErrorCode::CannotConnect:
		return "cannot connect";
	case ix::HttpErrorCode::Timeout:
		return "timed out";
	case ix::HttpErrorCode::UrlMalformed:
		return "bad address";
	case ix::HttpErrorCode::CannotCreateSocket:
		return "no socket";
	case ix::HttpErrorCode::SendError:
		return "send failed";
	case ix::HttpErrorCode::ReadError:
	case ix::HttpErrorCode::CannotReadStatusLine:
	case ix::HttpErrorCode::MissingStatus:
	case ix::HttpErrorCode::HeaderParsingError:
	case ix::HttpErrorCode::ChunkReadError:
	case ix::HttpErrorCode::CannotReadBody:
		return "bad reply";
	case ix::HttpErrorCode::MissingLocation:
	case ix::HttpErrorCode::TooManyRedirects:
		return "bad redirect";
	case ix::HttpErrorCode::Cancelled:
		return "cancelled";
	default:
		return "error";
	}
}

}  // namespace

struct UpdateCheck::Impl {
	std::mutex mu;
	bool cancelled = false;
	std::shared_ptr<ix::HttpRequestArgs> current;  // the request in flight

	// Makes the request in flight (and every later one) give up.
	void cancel()
	{
		std::lock_guard<std::mutex> lk(mu);
		cancelled = true;
		if (current)
			current->cancel = true;
	}
	// False when cancel() was called; otherwise the request is registered so that cancel() reaches it.
	bool begin(const std::shared_ptr<ix::HttpRequestArgs> &args)
	{
		std::lock_guard<std::mutex> lk(mu);
		if (cancelled)
			return false;
		args->cancel = false;
		current = args;
		return true;
	}
	void end()
	{
		std::lock_guard<std::mutex> lk(mu);
		current.reset();
	}

	bool fetch(const UpdateCheckOptions &opt, std::string &body, std::string &error)
	{
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(std::max(1, opt.total_timeout_s));
		ix::HttpClient client;
		client.setTLSOptions(detail::make_tls_options(opt.ca_file));

		std::string url = opt.url;
		for (int hop = 0; hop <= opt.max_redirects; ++hop) {
			if (!url_allowed(url, opt)) {
				error = "address not allowed (only https)";
				return false;
			}
			if (std::chrono::steady_clock::now() > deadline) {
				error = "timed out";
				return false;
			}
			auto args = client.createRequest(url, ix::HttpClient::kGet);
			args->followRedirects = false;  // followed below, address by address
			args->connectTimeout = std::max(1, opt.connect_timeout_s);
			args->transferTimeout = std::max(1, opt.transfer_timeout_s);
			args->compress = false;
			args->extraHeaders["User-Agent"] = opt.user_agent;
			args->extraHeaders["Accept"] = "application/json";
			std::string got;
			bool too_big = false;
			args->onChunkCallback = [&](const std::string &chunk) {
				got += chunk;
				if (got.size() > opt.max_bytes) {
					too_big = true;
					args->cancel = true;  // stops the read; the rest of the reply is never fetched
				}
			};
			if (!begin(args)) {
				error = "cancelled";
				return false;
			}
			const ix::HttpResponsePtr resp = client.get(url, args);
			end();

			if (too_big) {
				error = "reply too large";
				return false;
			}
			if (!resp) {
				error = "no reply";
				return false;
			}
			const int code = resp->statusCode;
			if (code >= 301 && code <= 308 && code != 304 && code != 305 && code != 306) {
				const auto loc = resp->headers.find("Location");
				if (loc == resp->headers.end() || loc->second.empty()) {
					error = "redirect without a target";
					return false;
				}
				url = resolve_location(url, loc->second);
				if (url.empty()) {
					error = "unsupported redirect";
					return false;
				}
				continue;
			}
			if (resp->errorCode != ix::HttpErrorCode::Ok) {
				error = error_name(resp->errorCode);
				return false;
			}
			if (code != 200) {
				error = "HTTP " + std::to_string(code);
				return false;
			}
			body = std::move(got);
			return true;
		}
		error = "too many redirects";
		return false;
	}
};

UpdateCheck::UpdateCheck() : impl_(std::make_shared<Impl>()) {}
UpdateCheck::~UpdateCheck() = default;

void UpdateCheck::cancel()
{
	impl_->cancel();
}

UpdateCheckResult UpdateCheck::run(const std::string &current_version, const UpdateCheckOptions &opt)
{
	UpdateCheckResult r;
	Version cur;
	if (!parse_version(current_version, cur)) {
		r.error = "own version number is not valid: " + current_version;
		return r;
	}
	ix::initNetSystem();  // counted: the sessions' own initialisation is unaffected
	std::string body;
	const bool ok = impl_->fetch(opt, body, r.error);
	ix::uninitNetSystem();
	if (!ok)
		return r;
	if (!parse_update_json(body, r.info)) {
		r.error = "unreadable version file";
		return r;
	}
	Version latest;
	parse_version(r.info.version, latest);
	r.status = compare_versions(latest, cur) > 0 ? UpdateStatus::UpdateAvailable : UpdateStatus::UpToDate;
	return r;
}

}  // namespace acw
