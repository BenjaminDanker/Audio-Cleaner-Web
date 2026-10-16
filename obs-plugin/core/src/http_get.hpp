// SPDX-License-Identifier: GPL-2.0-or-later
// One HTTPS GET with redirects, shared by the update check (a small manifest) and the update download (the
// plugin file). Private to the library. Certificates and host names are always verified; only https:// is
// followed, plus http:// to this computer when a test allows it; by default redirects stay on the server of the
// first address (the plugin talks to one place only).
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <ixwebsocket/IXHttpClient.h>

namespace acw {
namespace detail {

struct HttpGetOptions {
	std::string user_agent = "obs-audio-cleaner";
	std::string accept = "*/*";
	std::string ca_file;  // "" = the system's trust store
	int connect_timeout_s = 5;
	int transfer_timeout_s = 8;  // per request, redirects not included
	int total_timeout_s = 20;    // for the whole GET, redirects included
	int max_redirects = 4;
	bool allow_local_http = false;    // tests: http:// to 127.x / the local network
	bool same_origin_redirects = true;  // a redirect to another server (scheme, host, port) is refused
};

// Receives the body. A reply of another status than 200 (or a redirect) also has a body, which this sees: begin()
// is called before every request, so a receiver starts over; only what follows the last begin() is the body of
// the final 200 reply. chunk() returns false to stop (get() then fails with error "stopped").
struct HttpReceiver {
	std::function<void()> begin;
	std::function<bool(const std::string &)> chunk;
};

class HttpGet {
public:
	HttpGet();
	// False after cancel(): every request gives up at once.
	bool get(const std::string &url, const HttpGetOptions &opt, const HttpReceiver &rx, std::string &error);
	void cancel();

private:
	struct Impl;
	std::shared_ptr<Impl> impl_;
};

// "https://host:port/path?query" -> "https://host:port" (empty when it is not an absolute address).
std::string origin_of(const std::string &url);

}  // namespace detail
}  // namespace acw
