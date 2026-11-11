// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/key_check.hpp"

#include "acw/session.hpp"
#include "tls_options.hpp"

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>

#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace acw {

namespace {

constexpr int kTimeoutS = 8;

using Pair = std::pair<std::string, std::string>;  // server address, key

// Shared with the threads, which may outlive the plugin's own data by a moment (see stop_key_checks).
struct Checks {
	std::mutex mu;
	std::condition_variable cv;
	std::map<Pair, KeyVerdict> answers;
	std::map<Pair, std::shared_ptr<ix::WebSocket>> running;
	int threads = 0;
	bool stopped = false;
};

std::shared_ptr<Checks> checks()
{
	static const std::shared_ptr<Checks> c = std::make_shared<Checks>();
	return c;
}

void run_check(std::shared_ptr<Checks> c, std::shared_ptr<ix::WebSocket> ws, const Pair id)
{
	KeyVerdict verdict = KeyVerdict::Unknown;
	try {
		// The upgrade to a WebSocket is where the service looks at the key: 401 for one it does not know, 403 for one
		// that may not do live audio. Nothing is started on a connection that was accepted: it is closed right away.
		const ix::WebSocketInitResult r = ws->connect(kTimeoutS);
		if (r.success) {
			verdict = KeyVerdict::Accepted;
			ws->close();
		} else if (r.http_status == 401) {
			verdict = KeyVerdict::Wrong;
		} else if (r.http_status == 403) {
			verdict = KeyVerdict::Refused;
		}
	} catch (...) {
		// no answer
	}
	{
		std::lock_guard<std::mutex> lk(c->mu);
		c->running.erase(id);
		if (verdict != KeyVerdict::Unknown && !c->stopped)
			c->answers[id] = verdict;  // in the same step: a question that has its answer is never seen without it
	}
	ws.reset();  // the connection goes here, before the thread counts as finished
	ix::uninitNetSystem();
	std::lock_guard<std::mutex> lk(c->mu);
	--c->threads;
	c->cv.notify_all();
}

}  // namespace

void ask_key(const std::string &url, const std::string &api_key, const std::string &client, bool again, const std::string &ca_file)
{
	if (api_key.empty() || !server_url_allowed(url))
		return;  // (an address that is not allowed never gets the key, here as in a session)
	const std::shared_ptr<Checks> c = checks();
	const Pair id(url, api_key);
	std::lock_guard<std::mutex> lk(c->mu);
	if (c->stopped || c->running.count(id))
		return;
	if (again)
		c->answers.erase(id);
	else if (c->answers.count(id))
		return;
	ix::initNetSystem();  // counted: the sessions' own initialisation is unaffected (the thread takes it back)
	bool started = false;
	try {
		auto ws = std::make_shared<ix::WebSocket>();
		ws->setUrl(url);
		ix::WebSocketHttpHeaders headers;
		headers["Authorization"] = "Bearer " + api_key;  // never logged
		headers["User-Agent"] = client;
		ws->setExtraHeaders(headers);
		ws->disableAutomaticReconnection();
		ws->disablePerMessageDeflate();
		ws->setTLSOptions(detail::make_tls_options(ca_file));
		ws->setOnMessageCallback([](const ix::WebSocketMessagePtr &) {});  // connect() reports the opening to it
		c->running[id] = ws;
		std::thread([c, ws, id]() mutable { run_check(std::move(c), std::move(ws), id); }).detach();
		started = true;
		++c->threads;
	} catch (...) {
		// no thread to be had: no answer
	}
	if (!started) {
		c->running.erase(id);
		ix::uninitNetSystem();
	}
}

KeyVerdict key_verdict(const std::string &url, const std::string &api_key)
{
	const std::shared_ptr<Checks> c = checks();
	std::lock_guard<std::mutex> lk(c->mu);
	const auto it = c->answers.find(Pair(url, api_key));
	return it == c->answers.end() ? KeyVerdict::Unknown : it->second;
}

bool key_check_running(const std::string &url, const std::string &api_key)
{
	const std::shared_ptr<Checks> c = checks();
	std::lock_guard<std::mutex> lk(c->mu);
	return c->running.count(Pair(url, api_key)) > 0;
}

bool stop_key_checks(int ms)
{
	const std::shared_ptr<Checks> c = checks();
	std::vector<std::shared_ptr<ix::WebSocket>> open;
	{
		std::lock_guard<std::mutex> lk(c->mu);
		c->stopped = true;
		for (const auto &r : c->running)
			open.push_back(r.second);
	}
	for (const auto &ws : open) {
		try {
			ws->close();  // gives up a connection that is still being made
		} catch (...) {
		}
	}
	std::unique_lock<std::mutex> lk(c->mu);
	return c->cv.wait_for(lk, std::chrono::milliseconds(ms > 0 ? ms : 0), [&c]() { return c->threads == 0; });
}

}  // namespace acw
