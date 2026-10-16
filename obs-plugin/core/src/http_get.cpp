// SPDX-License-Identifier: GPL-2.0-or-later
#include "http_get.hpp"

#include "acw/session.hpp"
#include "tls_options.hpp"

#include <ixwebsocket/IXHttp.h>

#include <algorithm>
#include <cctype>
#include <chrono>

namespace acw {
namespace detail {

std::string origin_of(const std::string &url)
{
	const size_t scheme = url.find("://");
	if (scheme == std::string::npos)
		return {};
	const size_t slash = url.find_first_of("/?#", scheme + 3);
	return slash == std::string::npos ? url : url.substr(0, slash);
}

namespace {

std::string lower(std::string s)
{
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return s;
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
bool url_allowed(const std::string &url, const HttpGetOptions &opt)
{
	if (url.size() > 2000)
		return false;
	for (char ch : url)  // no spaces, control characters or non-ASCII in an address we send
		if (static_cast<unsigned char>(ch) <= 0x20 || static_cast<unsigned char>(ch) >= 0x7f)
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

struct HttpGet::Impl {
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
};

HttpGet::HttpGet() : impl_(std::make_shared<Impl>()) {}

void HttpGet::cancel()
{
	impl_->cancel();
}

bool HttpGet::get(const std::string &first_url, const HttpGetOptions &opt, const HttpReceiver &rx, std::string &error)
{
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(std::max(1, opt.total_timeout_s));
	ix::HttpClient client;
	client.setTLSOptions(make_tls_options(opt.ca_file));
	const std::string first_origin = lower(origin_of(first_url));

	std::string url = first_url;
	for (int hop = 0; hop <= opt.max_redirects; ++hop) {
		if (!url_allowed(url, opt)) {
			error = "address not allowed (only https)";
			return false;
		}
		if (opt.same_origin_redirects && hop > 0 && lower(origin_of(url)) != first_origin) {
			error = "redirect to another server not allowed";
			return false;
		}
		if (std::chrono::steady_clock::now() > deadline) {
			error = "timed out";
			return false;
		}
		auto args = client.createRequest(url, ix::HttpClient::kGet);
		args->followRedirects = false;  // followed here, address by address
		args->connectTimeout = std::max(1, opt.connect_timeout_s);
		args->transferTimeout = std::max(1, opt.transfer_timeout_s);
		args->compress = false;
		args->extraHeaders["User-Agent"] = opt.user_agent;
		args->extraHeaders["Accept"] = opt.accept;
		bool stopped = false;
		if (rx.begin)
			rx.begin();
		args->onChunkCallback = [&](const std::string &chunk) {
			if (stopped)
				return;
			if (rx.chunk && !rx.chunk(chunk)) {
				stopped = true;
				args->cancel = true;  // stops the read; the rest of the reply is never fetched
			}
		};
		if (!impl_->begin(args)) {
			error = "cancelled";
			return false;
		}
		const ix::HttpResponsePtr resp = client.get(url, args);
		impl_->end();

		if (stopped) {
			error = "stopped";
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
		return true;
	}
	error = "too many redirects";
	return false;
}

}  // namespace detail
}  // namespace acw
