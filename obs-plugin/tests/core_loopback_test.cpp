// SPDX-License-Identifier: GPL-2.0-or-later
// Threaded end-to-end tests of acw_core against a fake server on the loopback interface (no OBS, no
// internet, no account). The real Session runs its network thread against an IXWebSocket server that
// plays the part of the service: it echoes frames, answers with errors, delays replies, sends garbage.
// Run under ASan/UBSan and TSan in CI.
#include "acw/reference.hpp"
#include "acw/session.hpp"
#include "acw/update_check.hpp"

#include <ixwebsocket/IXGetFreePort.h>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXSocket.h>
#include <ixwebsocket/IXSocketServer.h>
#include <ixwebsocket/IXWebSocketServer.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace acw;
using namespace std::chrono;

static std::atomic<int> g_failures{0};
static std::atomic<int> g_checks{0};

#define CHECK(cond)                                                                       \
	do {                                                                              \
		++g_checks;                                                               \
		if (!(cond)) {                                                            \
			++g_failures;                                                     \
			std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
		}                                                                         \
	} while (0)

// Polls `pred` every 20 ms; true when it became true within `ms`.
static bool wait_until(const std::function<bool()> &pred, int ms)
{
	const auto end = steady_clock::now() + milliseconds(ms);
	while (steady_clock::now() < end) {
		if (pred())
			return true;
		std::this_thread::sleep_for(milliseconds(20));
	}
	return pred();
}

static bool contains(const std::string &hay, const char *needle)
{
	return hay.find(needle) != std::string::npos;
}

// ---------------------------------------------------------------------------------------------------
// The fake service

enum class Mode { Echo, GarbageBinary, WildSeq, GarbageText, DropAfter100, Silent };

class FakeServer {
public:
	FakeServer() : port_(ix::getFreePort()), srv_(port_, "127.0.0.1")
	{
		srv_.setOnClientMessageCallback(
			[this](std::shared_ptr<ix::ConnectionState>, ix::WebSocket &ws, const ix::WebSocketMessagePtr &msg) {
				on_message(ws, msg);
			});
		const auto res = srv_.listenAndStart();
		if (!res)
			std::printf("fake server: cannot listen on port %d\n", port_);
		sender_ = std::thread([this] { sender_loop(); });
	}
	~FakeServer()
	{
		run_ = false;
		sender_.join();
		srv_.stop();
	}

	std::string url() const { return "ws://127.0.0.1:" + std::to_string(port_) + "/x"; }

	// Behaviour (set before the session starts; most are re-read per message).
	std::atomic<Mode> mode{Mode::Echo};
	std::atomic<int> reply_delay_ms{0};
	std::mutex mu;  // guards the strings below
	std::string ready = R"({"type":"ready","sessionId":"s","sampleRate":48000,"hopSamples":480,"modelLatencyMs":30.0,)"
			    R"("usdPerMinute":0.01,"maxSessionSeconds":43200,"echoReference":true})";
	std::string error;        // non-empty: answer `start` with this instead of `ready`, then close
	std::string after_ready;  // sent right after `ready`
	std::string last_start;

	std::atomic<int> opens{0}, starts{0}, mic{0}, refs{0};

	void reset_counts()
	{
		opens = starts = mic = refs = 0;
	}

private:
	struct Delayed {
		steady_clock::time_point due;
		std::string data;
		ix::WebSocket *ws;
	};

	// Sends the delayed replies. No lock is held while sending (the library has locks of its own, and taking ours
	// inside its callbacks would invert the order); a connection is looked up by address among the live ones,
	// which also keeps it alive during the send.
	void sender_loop()
	{
		while (run_) {
			std::vector<Delayed> due;
			{
				std::lock_guard<std::mutex> lk(q_mu_);
				while (!q_.empty() && q_.front().due <= steady_clock::now()) {
					due.push_back(std::move(q_.front()));
					q_.pop_front();
				}
			}
			if (!due.empty()) {
				const auto clients = srv_.getClients();
				for (const Delayed &d : due)
					for (const auto &c : clients)
						if (c.get() == d.ws)
							c->sendBinary(d.data);
			}
			std::this_thread::sleep_for(microseconds(500));
		}
	}

	void on_message(ix::WebSocket &ws, const ix::WebSocketMessagePtr &msg)
	{
		switch (msg->type) {
		case ix::WebSocketMessageType::Open:
			++opens;
			return;
		case ix::WebSocketMessageType::Close:
		case ix::WebSocketMessageType::Error:
			return;
		case ix::WebSocketMessageType::Message:
			break;
		default:
			return;
		}
		if (!msg->binary) {
			if (contains(msg->str, "\"start\"")) {
				++starts;
				std::string err, rdy, after;
				{
					std::lock_guard<std::mutex> lk(mu);
					last_start = msg->str;
					err = error;
					rdy = ready;
					after = after_ready;
				}
				if (!err.empty()) {
					ws.sendText(err);
					ws.close();
					return;
				}
				ws.sendText(rdy);
				if (!after.empty())
					ws.sendText(after);
				if (contains(after, "\"stopped\""))
					ws.close();  // like the service: the stopped message, then the end
				if (mode == Mode::GarbageText) {
					ws.sendText("not json");
					ws.sendText("[1,2,3]");
					ws.sendText(R"({"type":"caption","text":123,"final":"yes"})");
					ws.sendText(R"({"type":"warning","code":"low_balance","secondsLeft":1e300})");
					ws.sendText(R"({"type":"pong","t":18446744073709551615})");
					ws.sendText(std::string(200000, '['));
				}
			} else if (contains(msg->str, "\"ping\"")) {
				ws.sendText(R"({"type":"pong","t":1})");
			}
			return;
		}
		const std::string &d = msg->str;
		if (d.size() < 16)
			return;
		const uint16_t flags = static_cast<uint8_t>(d[2]) | (static_cast<uint8_t>(d[3]) << 8);
		if (flags & 1) {
			++refs;
			return;
		}
		const int n = ++mic;
		switch (mode.load()) {
		case Mode::Echo: {
			const int dl = reply_delay_ms.load();
			if (dl > 0) {
				std::lock_guard<std::mutex> lk(q_mu_);
				q_.push_back({steady_clock::now() + milliseconds(dl), d, &ws});
			} else {
				ws.sendBinary(d);
			}
			break;
		}
		case Mode::GarbageBinary: {
			std::string g = d;
			const size_t cnt = rng_() % (g.size() + 1);
			for (size_t i = 0; i < cnt; ++i)
				g[rng_() % g.size()] = static_cast<char>(rng_());
			g.resize(rng_() % (g.size() + 40));
			ws.sendBinary(g);
			break;
		}
		case Mode::WildSeq: {
			std::string g = d;
			const uint32_t s = static_cast<uint32_t>(rng_());
			std::memcpy(&g[4], &s, 4);
			ws.sendBinary(g);
			break;
		}
		case Mode::DropAfter100:
			ws.sendBinary(d);
			if (n == 100)
				ws.close();
			break;
		case Mode::GarbageText:
			ws.sendBinary(d);
			break;
		case Mode::Silent:
			break;
		}
	}

	int port_;
	ix::WebSocketServer srv_;
	std::thread sender_;
	std::atomic<bool> run_{true};
	std::mutex q_mu_;
	std::deque<Delayed> q_;
	std::mt19937 rng_{7};
};

// Answers every request (the WebSocket upgrade) with a plain HTTP status.
class FakeHttp : public ix::SocketServer {
public:
	FakeHttp() : ix::SocketServer(ix::getFreePort(), "127.0.0.1")
	{
		const auto res = listen();
		if (!res.first)
			std::printf("fake http: cannot listen: %s\n", res.second.c_str());
		start();
	}
	~FakeHttp() override { stop(); }
	std::string url() { return "ws://127.0.0.1:" + std::to_string(getPort()) + "/x"; }
	std::atomic<int> status{403};
	std::atomic<int> requests{0};

private:
	void handleConnection(std::unique_ptr<ix::Socket> sock, std::shared_ptr<ix::ConnectionState> state) override
	{
		++requests;
		const ix::CancellationRequest never = [] { return false; };
		for (int i = 0; i < 100; ++i) {  // the request headers, up to the blank line
			const auto line = sock->readLine(never);
			if (!line.first || line.second == "\r\n" || line.second == "\n" || line.second.empty())
				break;
		}
		const std::string r = "HTTP/1.1 " + std::to_string(status.load()) +
				      " Rejected\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
		sock->writeBytes(r, never);
		state->setTerminated();
	}
	size_t getConnectedClientsCount() override { return 0; }
};

// A web server for the update check: answers GET <path> with the raw HTTP reply the test has stored for it.
// "<hang>" never answers, "<endless>" promises a huge body and keeps sending.
class FakeWeb : public ix::SocketServer {
public:
	FakeWeb() : ix::SocketServer(ix::getFreePort(), "127.0.0.1")
	{
		const auto res = listen();
		if (!res.first)
			std::printf("fake web: cannot listen: %s\n", res.second.c_str());
		start();
	}
	~FakeWeb() override
	{
		stopping = true;
		stop();
	}
	std::string base() { return "http://127.0.0.1:" + std::to_string(getPort()); }

	static std::string ok(const std::string &body, const char *type = "application/json")
	{
		return "HTTP/1.1 200 OK\r\nContent-Type: " + std::string(type) + "\r\nContent-Length: " + std::to_string(body.size()) +
		       "\r\nConnection: close\r\n\r\n" + body;
	}
	static std::string redirect(const std::string &to, int code = 302)
	{
		return "HTTP/1.1 " + std::to_string(code) + " Found\r\nLocation: " + to + "\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
	}
	static std::string status(int code)
	{
		return "HTTP/1.1 " + std::to_string(code) + " X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
	}
	void set(const std::string &path, const std::string &reply)
	{
		std::lock_guard<std::mutex> lk(mu);
		replies[path] = reply;
	}

	std::mutex mu;
	std::map<std::string, std::string> replies;
	std::string last_user_agent;
	std::atomic<int> requests{0};
	std::atomic<bool> stopping{false};

private:
	void handleConnection(std::unique_ptr<ix::Socket> sock, std::shared_ptr<ix::ConnectionState> state) override
	{
		++requests;
		const ix::CancellationRequest never = [] { return false; };
		std::string path, ua;
		for (int i = 0; i < 100; ++i) {  // the request line and headers, up to the blank line
			const auto line = sock->readLine(never);
			if (!line.first || line.second == "\r\n" || line.second == "\n" || line.second.empty())
				break;
			if (i == 0) {
				const size_t a = line.second.find(' '), b = line.second.find(' ', a + 1);
				path = line.second.substr(a + 1, b - a - 1);
			} else if (line.second.compare(0, 11, "User-Agent:") == 0) {
				ua = line.second.substr(12);
				while (!ua.empty() && (ua.back() == '\r' || ua.back() == '\n'))
					ua.pop_back();
			}
		}
		std::string reply;
		{
			std::lock_guard<std::mutex> lk(mu);
			last_user_agent = ua;
			const auto it = replies.find(path);
			reply = it == replies.end() ? status(404) : it->second;
		}
		if (reply == "<hang>") {
			for (int i = 0; i < 300 && !stopping; ++i)
				std::this_thread::sleep_for(milliseconds(50));
		} else if (reply == "<endless>") {
			sock->writeBytes("HTTP/1.1 200 OK\r\nContent-Length: 100000000\r\nConnection: close\r\n\r\n", never);
			const std::string chunk(8192, 'x');
			for (int i = 0; i < 20000 && !stopping; ++i)
				if (!sock->writeBytes(chunk, never))
					break;
		} else {
			sock->writeBytes(reply, never);
		}
		state->setTerminated();
	}
	size_t getConnectedClientsCount() override { return 0; }
};

// ---------------------------------------------------------------------------------------------------
// The host side: feeds noise to the session in real time, like OBS does

class Feeder {
public:
	enum class Kind { Noise, Silence };
	Feeder(Session &s, std::shared_ptr<ReferenceBuffer> ref, Kind kind = Kind::Noise, int rate = 48000)
		: s_(s), ref_(std::move(ref)), kind_(kind), rate_(rate)
	{
		th_ = std::thread([this] { run(); });
		if (ref_)
			ref_th_ = std::thread([this] { run_ref(); });
	}
	~Feeder() { stop(); }
	void stop()
	{
		run_ = false;
		if (th_.joinable())
			th_.join();
		if (ref_th_.joinable())
			ref_th_.join();
	}
	int blocks() const { return blocks_; }

private:
	void run()
	{
		std::vector<float> b(static_cast<size_t>(rate_ / 100));
		std::mt19937 r(2);
		std::normal_distribution<float> nd(0, 0.1f);
		auto t = steady_clock::now();
		while (run_) {
			for (auto &v : b)
				v = kind_ == Kind::Noise ? nd(r) : 0.0f;
			s_.process(b.data(), b.size());
			++blocks_;
			t += milliseconds(10);
			std::this_thread::sleep_until(t);
		}
	}
	void run_ref()
	{
		std::vector<float> b(static_cast<size_t>(rate_ / 100));
		auto t = steady_clock::now();
		float ph = 0;
		while (run_) {
			for (auto &v : b) {
				v = 0.3f * std::sin(ph);
				ph += 0.05f;
			}
			ref_->push(b.data(), b.size());
			t += milliseconds(10);
			std::this_thread::sleep_until(t);
		}
	}
	Session &s_;
	std::shared_ptr<ReferenceBuffer> ref_;
	Kind kind_;
	int rate_;
	std::atomic<bool> run_{true};
	std::atomic<int> blocks_{0};
	std::thread th_, ref_th_;
};

static SessionConfig base_config(const std::string &url)
{
	SessionConfig c;
	c.url = url;
	c.api_key = "test-key";
	c.host_rate = 48000;
	c.delay_ms = 150;
	return c;
}

struct Logs {
	std::mutex mu;
	std::vector<std::string> lines;
	Session::LogCallback cb()
	{
		return [this](LogLevel, const std::string &m) {
			std::lock_guard<std::mutex> lk(mu);
			lines.push_back(m);
		};
	}
	bool any(const char *needle)
	{
		std::lock_guard<std::mutex> lk(mu);
		for (auto &l : lines)
			if (contains(l, needle))
				return true;
		return false;
	}
};

// ---------------------------------------------------------------------------------------------------

static void test_echo_and_keepalive()
{
	std::printf("echo server: processing works, reference and silent microphone frames are sent\n");
	FakeServer srv;
	Logs logs;
	auto ref = std::make_shared<ReferenceBuffer>(48000);
	SessionConfig cfg = base_config(srv.url());
	cfg.reference = ref;
	Session s(cfg, nullptr, logs.cb());
	s.start();
	{
		Feeder f(s, ref);
		CHECK(wait_until([&] { return s.snapshot().state == SessionState::Streaming; }, 5000));
		std::this_thread::sleep_for(seconds(4));
		const SessionSnapshot n = s.snapshot();
		CHECK(n.state == SessionState::Streaming);
		CHECK(n.echo_active);
		CHECK(n.sent > 120);
		CHECK(n.used > 100);
		CHECK(n.used > 3 * n.late);
		CHECK(!n.link_too_slow);
		CHECK(srv.refs.load() > 60);
		CHECK(contains(s.status_line(), "Connected"));
		CHECK(contains(s.status_line(), "Echo cancellation is on"));
		CHECK(!contains(s.status_line(), "ws://"));
		CHECK(s.status_line().find("test-key") == std::string::npos);
		std::lock_guard<std::mutex> lk(srv.mu);
		CHECK(contains(srv.last_start, "\"echoReference\":true"));
		CHECK(contains(srv.last_start, "\"denoise\":\"light\""));
		CHECK(contains(srv.last_start, "\"opusBitrate\":64000"));
		CHECK(contains(srv.last_start, "\"attenLimDb\":100"));
	}
	// Live strength change reaches the server as a config message, and the next start uses it.
	s.set_atten_lim_db(24);
	s.stop(0, 500);

	// A silent microphone (muted in OBS, or a quiet room) is still sent frame by frame: those frames keep the
	// session alive on the server, which closes sessions that get no microphone audio for a couple of minutes.
	FakeServer srv2;
	Session s2(base_config(srv2.url()), nullptr, nullptr);
	s2.start();
	const auto t0 = steady_clock::now();
	{
		Feeder f(s2, nullptr, Feeder::Kind::Silence);
		CHECK(wait_until([&] { return srv2.mic.load() > 10; }, 5000));
		srv2.mic = 0;
		std::this_thread::sleep_for(seconds(3));
		const int got = srv2.mic.load();
		std::printf("  silent input: %d microphone frames in 3 s (50 per second expected)\n", got);
		CHECK(got >= 135 && got <= 165);
	}
	(void)t0;
	s2.stop(0, 500);
}

static void test_slow_link()
{
	std::printf("replies later than the delay: the user is told\n");
	FakeServer srv;
	srv.reply_delay_ms = 400;  // far beyond the 150 ms budget
	Logs logs;
	SessionConfig cfg = base_config(srv.url());
	Session s(cfg, nullptr, logs.cb());
	s.start();
	Feeder f(s, nullptr);
	CHECK(wait_until([&] { return s.snapshot().link_too_slow; }, 12000));
	const SessionSnapshot n = s.snapshot();
	std::printf("  late %.0f%%: %s\n", n.late_percent, s.status_line().c_str());
	CHECK(n.link_too_slow && n.late_percent > 30.0);
	CHECK(contains(s.status_line(), "too slow"));
	CHECK(contains(s.status_line(), "Balanced (150 ms)"));
	CHECK(contains(s.status_line(), "longer delay"));
	CHECK(logs.any("too late"));
	CHECK(!contains(s.status_line(), "Your audio is being cleaned"));
	// The link recovers: the flag clears again.
	srv.reply_delay_ms = 0;
	CHECK(wait_until([&] { return !s.snapshot().link_too_slow; }, 15000));
	CHECK(contains(s.status_line(), "being cleaned"));
	f.stop();
	s.stop(0, 500);
}

struct ErrorCase {
	const char *code;
	const char *json;
	bool blocks;        // stops retrying until the user acts
	int min_backoff_s;  // otherwise: at least this long before the next try
	const char *text;   // part of the message the user sees
};

static void test_server_errors()
{
	std::printf("server errors: permanent ones stop the retries, limits back off, the message is plain\n");
	const ErrorCase cases[] = {
		{"bad_config",
		 R"({"type":"error","code":"bad_config","message":"Standard captions are English only. Extra-accurate captions work in other languages."})",
		 true, 0, "English only"},
		{"insufficient_funds",
		 R"({"type":"error","code":"insufficient_funds","message":"Live processing reserves 5 minutes up front ($0.05); available $0.01."})",
		 true, 0, "Add balance"},
		{"tos", R"({"type":"error","code":"tos","message":"Accept the terms"})", true, 0, "terms of service"},
		{"too_many_sessions", R"({"type":"error","code":"too_many_sessions","message":"Two live sessions at most per account."})",
		 false, 30, "two live sessions"},
		{"full", R"({"type":"error","code":"full","message":"Live noise removal is full right now. Try again in a few minutes."})",
		 false, 60, "full right now"},
		{"retry hint", R"({"type":"error","code":"unavailable","message":"busy","retryAfterSeconds":90})", false, 90, "busy or offline"},
	};
	for (const ErrorCase &c : cases) {
		std::printf("  %s\n", c.code);
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.error = c.json;
		}
		Logs logs;
		Session s(base_config(srv.url()), nullptr, logs.cb());
		s.start();
		Feeder f(s, nullptr);
		const SessionState want = c.blocks ? SessionState::Blocked : SessionState::Backoff;
		CHECK(wait_until([&] { return s.snapshot().state == want; }, 5000));
		const SessionSnapshot n = s.snapshot();
		CHECK(n.state == want);
		CHECK(contains(s.status_line(), c.text));
		CHECK(!contains(s.status_line(), "premium") && !contains(s.status_line(), "Premium"));
		if (c.blocks) {
			// No retry while the user has not acted (a normal backoff would have been back within 2 s).
			std::this_thread::sleep_for(milliseconds(2200));
			CHECK(srv.starts.load() == 1);
			CHECK(s.snapshot().state == SessionState::Blocked);
			CHECK(contains(s.status_line(), "Try again") || contains(s.status_line(), "Change") ||
			      contains(s.status_line(), "terms"));
			s.retry_now();
			CHECK(wait_until([&] { return srv.starts.load() >= 2; }, 3000));
		} else {
			CHECK(n.backoff_seconds >= c.min_backoff_s);
			std::this_thread::sleep_for(milliseconds(1500));
			CHECK(srv.starts.load() == 1);  // still waiting
			s.retry_now();                  // no effect unless blocked
			std::this_thread::sleep_for(milliseconds(300));
			CHECK(srv.starts.load() == 1);
		}
		f.stop();
		s.stop(0, 500);
	}

	// The same error arriving after the session ran: a stopped event for lack of balance.
	std::printf("  balance runs out while streaming\n");
	FakeServer srv;
	{
		std::lock_guard<std::mutex> lk(srv.mu);
		srv.after_ready = R"({"type":"stopped","reason":"insufficient_funds"})";
	}
	Session s(base_config(srv.url()), nullptr, nullptr);
	s.start();
	Feeder f(s, nullptr);
	CHECK(wait_until([&] { return s.snapshot().state == SessionState::Blocked; }, 5000));
	CHECK(contains(s.status_line(), "balance ran out"));
	std::this_thread::sleep_for(milliseconds(1500));
	CHECK(srv.starts.load() == 1);
	f.stop();
	s.stop(0, 500);
}

static void test_http_refusals()
{
	std::printf("refused upgrades (HTTP status)\n");
	struct C {
		int status;
		bool blocks;
		int min_backoff_s;
		const char *text;
	};
	const C cases[] = {
		{401, true, 0, "API key"},
		{403, false, 120, "refused the connection"},
		{429, false, 120, "Too many connection attempts"},
		{404, true, 0, "server address"},
		{502, false, 5, "having trouble"},
	};
	FakeHttp srv;
	for (const C &c : cases) {
		std::printf("  HTTP %d\n", c.status);
		srv.status = c.status;
		srv.requests = 0;
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Feeder f(s, nullptr);
		const SessionState want = c.blocks ? SessionState::Blocked : SessionState::Backoff;
		CHECK(wait_until([&] { return s.snapshot().state == want; }, 5000));
		const SessionSnapshot n = s.snapshot();
		CHECK(n.state == want);
		CHECK(contains(s.status_line(), c.text));
		if (!c.blocks)
			CHECK(n.backoff_seconds >= c.min_backoff_s);
		std::this_thread::sleep_for(milliseconds(1800));
		CHECK(srv.requests.load() == 1);
		f.stop();
		s.stop(0, 500);
	}
}

static void test_ready_validation()
{
	std::printf("ready message: invalid numbers are refused, not played\n");
	struct C {
		const char *ready;
		SessionState state;
		const char *text;
	};
	const C cases[] = {
		{R"({"type":"ready","sampleRate":48000,"modelLatencyMs":-5000})", SessionState::Backoff, "does not understand"},
		{R"({"type":"ready","sampleRate":48000,"modelLatencyMs":99999})", SessionState::Backoff, "does not understand"},
		{R"({"type":"ready","sampleRate":48000,"modelLatencyMs":400})", SessionState::Blocked, "too short"},
		{R"({"type":"ready","sampleRate":44100,"modelLatencyMs":30})", SessionState::Blocked, "update the plugin"},
	};
	for (const C &c : cases) {
		std::printf("  %s\n", c.ready);
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.ready = c.ready;
		}
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Feeder f(s, nullptr);
		CHECK(wait_until([&] { return s.snapshot().state == c.state; }, 5000));
		CHECK(s.snapshot().state == c.state);
		CHECK(contains(s.status_line(), c.text));
		CHECK(s.snapshot().sent == 0);
		f.stop();
		s.stop(0, 500);
	}
}

static void test_captions_only()
{
	std::printf("captions only: smallest return stream, captions delivered, audio untouched\n");
	FakeServer srv;
	{
		std::lock_guard<std::mutex> lk(srv.mu);
		srv.after_ready = R"({"type":"caption","text":"hello world","final":true,"start":1.0,"end":2.0})";
	}
	std::mutex mu;
	std::vector<std::string> caps;
	SessionConfig cfg = base_config(srv.url());
	cfg.denoise = false;
	cfg.captions = "local";
	Session s(cfg,
		  [&](const CaptionEvent &e) {
			  std::lock_guard<std::mutex> lk(mu);
			  caps.push_back(e.text);
		  },
		  nullptr);
	s.start();
	Feeder f(s, nullptr);
	CHECK(wait_until([&] { return s.snapshot().state == SessionState::Streaming; }, 5000));
	CHECK(wait_until([&] { std::lock_guard<std::mutex> lk(mu); return !caps.empty(); }, 3000));
	{
		std::lock_guard<std::mutex> lk(mu);
		CHECK(!caps.empty() && caps[0] == "hello world");
	}
	CHECK(s.delay_seconds() == 0.0);
	CHECK(contains(s.status_line(), "Captions are running"));
	{
		std::lock_guard<std::mutex> lk(srv.mu);
		CHECK(contains(srv.last_start, "\"denoise\":null"));
		CHECK(contains(srv.last_start, "\"captions\":\"local\""));
		CHECK(contains(srv.last_start, "\"opusBitrate\":24000"));
	}
	std::this_thread::sleep_for(seconds(2));
	CHECK(s.snapshot().received > 50);
	CHECK(s.snapshot().bad_frames == 0);
	f.stop();
	s.stop(0, 500);
}

static void test_robustness()
{
	std::printf("garbage from the server, drops and reconnects\n");
	struct C {
		const char *name;
		Mode mode;
		int secs;
	};
	const C cases[] = {
		{"garbage binary frames", Mode::GarbageBinary, 3},
		{"wild sequence numbers", Mode::WildSeq, 3},
		{"garbage text events", Mode::GarbageText, 3},
		{"server drops the connection", Mode::DropAfter100, 6},
		{"server never answers", Mode::Silent, 2},
	};
	for (const C &c : cases) {
		std::printf("  %s\n", c.name);
		FakeServer srv;
		srv.mode = c.mode;
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		{
			Feeder f(s, nullptr);
			std::this_thread::sleep_for(seconds(c.secs));
			const SessionSnapshot n = s.snapshot();
			CHECK(n.sent > 20);
			if (c.mode == Mode::DropAfter100) {
				CHECK(n.reconnects >= 1);
				CHECK(srv.opens.load() >= 2);
			}
			(void)s.status_line();
			(void)s.status_details();
		}
		const auto t0 = steady_clock::now();
		s.stop(0, 500);
		CHECK(duration_cast<milliseconds>(steady_clock::now() - t0).count() < 700);
	}
}

static void test_stop_paths()
{
	std::printf("stopping: at once, from any state\n");
	{
		// Refused address, unreachable port: stop() must come back quickly.
		SessionConfig cfg = base_config("ws://127.0.0.1:1/x");
		Session s(cfg, nullptr, nullptr);
		s.start();
		Feeder f(s, nullptr);
		// Windows takes about two seconds to give up on a refused port (it retries the SYN), Linux and macOS answer at once.
		CHECK(wait_until(
			[&] {
				return s.snapshot().state == SessionState::Backoff &&
				       contains(s.status_line(), "Could not reach the server");
			},
			10000));
		f.stop();
		const auto t0 = steady_clock::now();
		s.stop(0, 500);
		CHECK(duration_cast<milliseconds>(steady_clock::now() - t0).count() < 700);
	}
	{
		// request_stop() ends the connection without waiting; the object can be freed afterwards.
		FakeServer srv;
		auto s = std::make_unique<Session>(base_config(srv.url()), nullptr, nullptr);
		s->start();
		Feeder f(*s, nullptr);
		CHECK(wait_until([&] { return s->snapshot().state == SessionState::Streaming; }, 5000));
		s->request_stop();
		CHECK(wait_until([&] { return s->snapshot().state == SessionState::Stopped; }, 2000));
		f.stop();
		s->stop(0, 500);
		s.reset();
		CHECK(wait_for_background_threads(1000));
	}
	{
		// A public ws:// address is refused before anything is sent.
		SessionConfig cfg = base_config("ws://example.com/x");
		Session s(cfg, nullptr, nullptr);
		s.start();
		CHECK(s.snapshot().state == SessionState::Stopped);
		CHECK(contains(s.status_line(), "local network"));
	}
}

static void test_update_check()
{
	std::printf("update check against a local web server\n");
	FakeWeb web;
	const std::string json_new = R"({"version": "0.1.57", "url": ")" + std::string(kReleasePageUrl) + R"("})";
	web.set("/new.json", FakeWeb::ok(json_new));
	web.set("/same.json", FakeWeb::ok(R"({"version":"0.1.9"})"));
	web.set("/old.json", FakeWeb::ok(R"({"version":"0.0.3"})"));
	web.set("/garbage.json", FakeWeb::ok("<html>not json</html>", "text/html"));
	web.set("/noversion.json", FakeWeb::ok(R"({"url":"x"})"));
	web.set("/badversion.json", FakeWeb::ok(R"({"version":"latest"})"));
	web.set("/big.json", FakeWeb::ok(R"({"version":"9.9.9","pad":")" + std::string(6000, 'a') + R"("})"));
	web.set("/err500", FakeWeb::status(500));
	web.set("/hop1", FakeWeb::redirect("/hop2", 301));
	web.set("/hop2", FakeWeb::redirect(web.base() + "/new.json", 307));
	web.set("/loop", FakeWeb::redirect("/loop"));
	web.set("/to-plain-http", FakeWeb::redirect("http://example.com/version.json"));
	web.set("/to-ftp", FakeWeb::redirect("ftp://example.com/version.json"));
	web.set("/to-relative", FakeWeb::redirect("new.json"));
	web.set("/to-network-path", FakeWeb::redirect("//example.com/new.json"));
	web.set("/hang", "<hang>");
	web.set("/endless", "<endless>");

	auto opts = [&](const char *path) {
		UpdateCheckOptions o;
		o.url = web.base() + path;
		o.allow_local_http = true;
		o.user_agent = "obs-audio-cleaner/0.1.9";
		o.connect_timeout_s = 2;
		o.transfer_timeout_s = 2;
		o.total_timeout_s = 6;
		return o;
	};
	auto run = [&](const char *current, const char *path) {
		UpdateCheck c;
		return c.run(current, opts(path));
	};

	{
		const auto r = run("0.1.9", "/new.json");
		CHECK(r.status == UpdateStatus::UpdateAvailable);
		CHECK(r.info.version == "0.1.57");
		CHECK(r.info.url == kReleasePageUrl);
		std::lock_guard<std::mutex> lk(web.mu);
		CHECK(web.last_user_agent == "obs-audio-cleaner/0.1.9");
	}
	CHECK(run("0.1.9", "/same.json").status == UpdateStatus::UpToDate);
	CHECK(run("0.1.9", "/old.json").status == UpdateStatus::UpToDate);
	CHECK(run("0.1.57", "/new.json").status == UpdateStatus::UpToDate);
	CHECK(run("0.1.56", "/new.json").status == UpdateStatus::UpdateAvailable);
	CHECK(run("0.1.100", "/new.json").status == UpdateStatus::UpToDate);  // numbers, not text

	// Redirects: followed (relative and absolute, 301/307), but only to places that are allowed.
	CHECK(run("0.1.9", "/hop1").status == UpdateStatus::UpdateAvailable);
	CHECK(run("0.1.9", "/to-relative").status == UpdateStatus::Failed);  // relative to a directory: not supported
	for (const char *p : {"/loop", "/to-plain-http", "/to-ftp", "/to-network-path"}) {
		const auto r = run("0.1.9", p);
		CHECK(r.status == UpdateStatus::Failed && !r.error.empty());
	}
	{
		const auto r = run("0.1.9", "/to-plain-http");
		CHECK(contains(r.error, "not allowed"));
	}

	// Things that are not a version file.
	for (const char *p : {"/garbage.json", "/noversion.json", "/badversion.json", "/err500", "/missing"}) {
		const auto r = run("0.1.9", p);
		CHECK(r.status == UpdateStatus::Failed && !r.error.empty());
	}
	CHECK(contains(run("0.1.9", "/err500").error, "500"));
	CHECK(contains(run("0.1.9", "/missing").error, "404"));
	CHECK(run("not a version", "/new.json").status == UpdateStatus::Failed);

	// Size limit: a body over the limit is refused (declared long, or really long).
	{
		const auto r = run("0.1.9", "/big.json");
		CHECK(r.status == UpdateStatus::Failed && contains(r.error, "large"));
	}
	{
		const auto t0 = steady_clock::now();
		const auto r = run("0.1.9", "/endless");
		CHECK(r.status == UpdateStatus::Failed && contains(r.error, "large"));
		CHECK(steady_clock::now() - t0 < seconds(3));
	}

	// A server that never answers: the transfer timeout ends it.
	{
		const auto t0 = steady_clock::now();
		auto o = opts("/hang");
		o.transfer_timeout_s = 1;
		UpdateCheck c;
		const auto r = c.run("0.1.9", o);
		CHECK(r.status == UpdateStatus::Failed);
		CHECK(steady_clock::now() - t0 < seconds(4));
	}
	// ... and cancel() ends it at once, from another thread (OBS unloading the plugin).
	{
		UpdateCheck c;
		auto o = opts("/hang");
		o.transfer_timeout_s = 30;
		o.total_timeout_s = 60;
		UpdateCheckResult r;
		std::atomic<bool> done{false};
		std::thread th([&] {
			r = c.run("0.1.9", o);
			done = true;
		});
		std::this_thread::sleep_for(milliseconds(400));
		CHECK(!done);
		const auto t0 = steady_clock::now();
		c.cancel();
		CHECK(wait_until([&] { return done.load(); }, 3000));
		th.join();
		CHECK(steady_clock::now() - t0 < seconds(3));
		CHECK(r.status == UpdateStatus::Failed);
		// A cancelled check stays cancelled.
		CHECK(c.run("0.1.9", opts("/new.json")).status == UpdateStatus::Failed);
	}

	// Addresses that are never used: plain http:// (unless a test allows the local one), other schemes.
	{
		auto o = opts("/new.json");
		o.allow_local_http = false;
		const int before = web.requests.load();
		UpdateCheck c;
		const auto r = c.run("0.1.9", o);
		CHECK(r.status == UpdateStatus::Failed && contains(r.error, "not allowed"));
		CHECK(web.requests.load() == before);  // nothing was sent
	}
	for (const char *u : {"http://example.com/version.json", "ftp://127.0.0.1/x", "file:///etc/passwd", "", "https://"}) {
		auto o = opts("/new.json");
		o.url = u;
		UpdateCheck c;
		const auto r = c.run("0.1.9", o);
		CHECK(r.status == UpdateStatus::Failed);
	}
	// https:// to a port where nobody listens: a plain failure, quickly.
	{
		auto o = opts("/new.json");
		o.url = "https://127.0.0.1:" + std::to_string(ix::getFreePort()) + "/version.json";
		const auto t0 = steady_clock::now();
		UpdateCheck c;
		const auto r = c.run("0.1.9", o);
		CHECK(r.status == UpdateStatus::Failed);
		CHECK(steady_clock::now() - t0 < seconds(5));
	}
	// https:// to a server that only speaks plain HTTP: the TLS handshake fails, nothing is accepted.
	{
		auto o = opts("/new.json");
		o.url = "https://127.0.0.1:" + std::to_string(web.getPort()) + "/new.json";
		UpdateCheck c;
		const auto r = c.run("0.1.9", o);
		CHECK(r.status == UpdateStatus::Failed);
	}
}

int main(int argc, char **argv)
{
	ix::initNetSystem();
	if (argc >= 2 && std::string(argv[1]) == "--serve") {
		// Not a test: the fake service on its own (echo, 30 ms model latency), e.g. for the obs_smoke host.
		FakeServer srv;
		std::printf("%s\n", srv.url().c_str());
		std::fflush(stdout);
		for (;;) {
			std::this_thread::sleep_for(seconds(1));
			std::printf("mic=%d ref=%d opens=%d\n", srv.mic.load(), srv.refs.load(), srv.opens.load());
			std::fflush(stdout);
		}
	}
	test_echo_and_keepalive();
	test_slow_link();
	test_server_errors();
	test_http_refusals();
	test_ready_validation();
	test_captions_only();
	test_robustness();
	test_stop_paths();
	test_update_check();
	ix::uninitNetSystem();
	std::printf("\n%d checks, %d failures\n", g_checks.load(), g_failures.load());
	return g_failures ? 1 : 0;
}
