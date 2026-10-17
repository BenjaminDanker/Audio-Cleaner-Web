// SPDX-License-Identifier: GPL-2.0-or-later
// Threaded end-to-end tests of acw_core against a fake server on the loopback interface (no OBS, no
// internet, no account). The real Session runs its network thread against an IXWebSocket server that
// plays the part of the service: it echoes frames, answers with errors, delays replies, sends garbage.
// Run under ASan/UBSan and TSan in CI.
#include "acw/reference.hpp"
#include "acw/session.hpp"
#include "acw/update_check.hpp"
#include "acw/update_download.hpp"
#include "acw/update_verify.hpp"
#include "update_test_util.hpp"

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
#include <filesystem>
#include <fstream>
#include <iterator>
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

// A folder of our own for the downloaded files.
namespace testing_detail {
inline std::string tmp_dir()
{
	return (std::filesystem::temp_directory_path() / ("acw-dl-test-" + std::to_string(std::random_device{}()))).string();
}
}  // namespace testing_detail

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

// ReplySilence: answers every microphone frame with a silence frame (as the service does for silence); BadSilence: with a
// silence frame whose sample count is wrong.
enum class Mode { Echo, GarbageBinary, WildSeq, GarbageText, DropAfter100, Silent, ReplySilence, BadSilence };

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
	std::string after_ready;  // sent right after `ready` (several messages: one per line)
	std::string close_reason; // non-empty: close the socket with this reason (code 1008) after `after_ready`
	std::string last_start;
	std::string last_change;  // the last `config` message with an id
	// The answer to a change: "" = ok, with the session's settings as asked; otherwise this refusal's message.
	std::string refuse_change;
	std::atomic<int> changes{0};

	std::atomic<int> opens{0}, starts{0}, mic{0}, refs{0};
	// By kind: microphone frames that were silence frames (codec 4) or Opus (codec 3), reference frames that were silence frames.
	std::atomic<int> silent_mic{0}, opus_mic{0}, silent_refs{0};
	// true: honour "returnAudio": false (send no audio back), as the service does. Off by default: the other tests want the echo.
	std::atomic<bool> honor_return_audio{false};

	void reset_counts()
	{
		opens = starts = mic = refs = silent_mic = opus_mic = silent_refs = 0;
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
				no_audio_back_ = honor_return_audio && contains(msg->str, "\"returnAudio\":false");
				std::string err, rdy, after, creason;
				{
					std::lock_guard<std::mutex> lk(mu);
					last_start = msg->str;
					err = error;
					rdy = ready;
					after = after_ready;
					creason = close_reason;
				}
				if (!err.empty()) {
					ws.sendText(err);
					ws.close();
					return;
				}
				ws.sendText(rdy);
				for (size_t pos = 0; pos <= after.size();) {
					const size_t nl = after.find('\n', pos);
					const std::string part = after.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
					if (!part.empty())
						ws.sendText(part);
					if (nl == std::string::npos)
						break;
					pos = nl + 1;
				}
				if (!creason.empty())
					ws.close(1008, creason);
				else if (contains(after, "\"stopped\""))
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
			} else if (contains(msg->str, "\"config\"") && contains(msg->str, "\"id\"")) {
				++changes;
				const std::string &m = msg->str;
				// the parts of the message the answer repeats: its id and the translations (an array of codes)
				auto after = [&](const char *key, char end) {
					const size_t at = m.find(key);
					if (at == std::string::npos)
						return std::string();
					const size_t from = at + std::strlen(key);
					return m.substr(from, m.find(end, from) - from + (end == ']' ? 1 : 0));
				};
				std::string refuse;
				{
					std::lock_guard<std::mutex> lk(mu);
					last_change = m;
					refuse = refuse_change;
				}
				std::string a = "{\"type\":\"config\",\"id\":\"" + after("\"id\":\"", '"') + "\",";
				if (refuse.empty())
					a += std::string("\"ok\":true,\"usdPerMinute\":") + (contains(m, "\"premium\"") ? "0.05" : "0.02") +
					     ",\"translate\":" + after("\"translate\":", ']') +
					     ",\"echoReference\":" + (contains(m, "\"echoReference\":true") ? "true" : "false") + "}";
				else
					a += "\"ok\":false,\"code\":\"full\",\"message\":\"" + refuse + "\"}";
				ws.sendText(a);
			}
			return;
		}
		const std::string &d = msg->str;
		if (d.size() < 16)
			return;
		const uint16_t flags = static_cast<uint8_t>(d[2]) | (static_cast<uint8_t>(d[3]) << 8);
		const uint8_t codec = static_cast<uint8_t>(d[1]);
		if (flags & 1) {
			++refs;
			if (codec == 4)
				++silent_refs;
			return;
		}
		const int n = ++mic;
		if (codec == 4)
			++silent_mic;
		else if (codec == 3)
			++opus_mic;
		if (no_audio_back_)
			return;
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
		case Mode::ReplySilence:
		case Mode::BadSilence: {
			std::string g = d.substr(0, 16);
			g[1] = 4;
			const uint16_t count = mode.load() == Mode::ReplySilence ? 960 : 7;
			g.push_back(static_cast<char>(count & 0xff));
			g.push_back(static_cast<char>(count >> 8));
			ws.sendBinary(g);
			break;
		}
		}
	}

	std::atomic<bool> no_audio_back_{false};
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
	cfg.echo = true;
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

static void test_changes_while_live()
{
	std::printf("changes while live: captions, language, translations and echo removal change without a new session\n");
	FakeServer srv;
	auto ref = std::make_shared<ReferenceBuffer>(48000);
	SessionConfig cfg = base_config(srv.url());
	cfg.captions = "local";
	cfg.reference = ref;  // the speakers' buffer is there; echo removal is off for now
	{
		std::lock_guard<std::mutex> lk(srv.mu);  // the service says echoReference only to a session that asked for it
		srv.ready = R"({"type":"ready","sessionId":"s","sampleRate":48000,"hopSamples":480,"modelLatencyMs":39.0,)"
			    R"("usdPerMinute":0.01,"maxSessionSeconds":43200,"silenceFrames":true,"changeable":true})";
	}
	Session s(cfg, nullptr, nullptr);
	s.start();
	{
		Feeder f(s, ref);
		CHECK(wait_until([&] { return s.snapshot().state == SessionState::Streaming; }, 5000));
		CHECK(!s.snapshot().echo_active);
		std::this_thread::sleep_for(milliseconds(500));
		const int refs_before = srv.refs.load();
		CHECK(refs_before == 0);  // no reference frames while echo removal is off

		SessionConfig next = cfg;
		next.captions = "premium";
		next.language = "es";
		next.translate = {"de"};
		next.echo = true;
		CHECK(s.change(next));
		CHECK(wait_until([&] { return s.snapshot().usd_per_minute > 0.04; }, 5000));
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			CHECK(contains(srv.last_change, "\"captions\":\"premium\""));
			CHECK(contains(srv.last_change, "\"language\":\"es\""));
			CHECK(contains(srv.last_change, "\"echoReference\":true"));
			CHECK(contains(srv.last_change, "\"denoise\":\"light\""));
		}
		CHECK(srv.opens.load() == 1);  // the same connection
		const SessionSnapshot n = s.snapshot();
		CHECK(n.echo_active && n.translate_active == std::vector<std::string>{"de"} && n.change_refused.empty());
		CHECK(contains(s.status_line(), "Translating into German."));
		CHECK(wait_until([&] { return srv.refs.load() > refs_before + 20; }, 5000));  // the speakers' sound goes out now

		// Noise removal off, in a session that started with it: the delay stays, nothing new is needed.
		next.denoise = false;
		CHECK(s.change(next));
		CHECK(wait_until([&] { return srv.changes.load() == 2; }, 5000));
		CHECK(wait_until([&] { return !contains(s.status_line(), "being cleaned"); }, 5000));

		// A change the service refuses: the session goes on as it was and says why.
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.refuse_change = "Live is full right now.";
		}
		next.captions = "local";
		next.language = "";
		next.translate = {};
		CHECK(s.change(next));
		CHECK(wait_until([&] { return !s.snapshot().change_refused.empty(); }, 5000));
		CHECK(contains(s.status_line(), "Live is full right now."));
		CHECK(contains(s.status_line(), "Translating into German."));  // still the earlier settings
		CHECK(srv.opens.load() == 1);
		f.stop();
		s.stop(0, 500);
	}
	// A session that started captions only has no delay to put cleaned audio in: noise removal needs a new one.
	SessionConfig captions_only = base_config(srv.url());
	captions_only.denoise = false;
	captions_only.captions = "local";
	Session c(captions_only, nullptr, nullptr);
	SessionConfig with_noise = captions_only;
	with_noise.denoise = true;
	CHECK(!c.change(with_noise));
	SessionConfig other = captions_only;
	other.delay_ms = 250;
	CHECK(!c.change(other));  // not something a running session changes
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
		{"too_many_sessions", R"({"type":"error","code":"too_many_sessions","message":"Four live sessions at most per account."})",
		 false, 30, "four live sessions"},
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

// Records what the session does to the audio: pushes noise through process() in real time (10 ms blocks) and keeps
// both sides. `max_call_us` is the longest single process() call.
struct Capture {
	std::vector<float> in, out;
	long long max_call_us = 0;
};

static Capture run_audio(Session &s, double secs)
{
	Capture c;
	std::mt19937 rng(11);
	std::normal_distribution<float> nd(0.0f, 0.1f);
	std::vector<float> blk(480), proc;
	auto t = steady_clock::now();
	const int blocks = static_cast<int>(secs * 100);
	for (int b = 0; b < blocks; ++b) {
		for (float &v : blk)
			v = nd(rng) + 0.001f;  // never exactly zero: a hole of silence would show
		proc = blk;
		const auto t0 = steady_clock::now();
		s.process(proc.data(), proc.size());
		c.max_call_us = std::max<long long>(c.max_call_us, duration_cast<microseconds>(steady_clock::now() - t0).count());
		c.in.insert(c.in.end(), blk.begin(), blk.end());
		c.out.insert(c.out.end(), proc.begin(), proc.end());
		t += milliseconds(10);
		std::this_thread::sleep_until(t);
	}
	return c;
}

// The audio is the delayed original, sample for sample: nothing lost, nothing silenced, nothing added.
static bool is_plain_delayed_copy(const Capture &c, int delay_samples)
{
	if (c.in.size() != c.out.size() || c.in.size() < static_cast<size_t>(delay_samples) + 1000)
		return false;
	for (size_t i = static_cast<size_t>(delay_samples); i < c.out.size(); ++i)
		if (c.out[i] != c.in[i - static_cast<size_t>(delay_samples)])
			return false;
	return true;
}

static void test_never_breaks_the_stream()
{
	std::printf("whatever the service does, the microphone audio passes through unprocessed, unbroken, without silence\n");
	struct Case {
		const char *name;
		const char *error;        // answer `start` with this
		const char *after_ready;  // ... or this after `ready`
		const char *close_reason;
	};
	const Case cases[] = {
		{"service never sends processed audio", "", "", ""},
		{"service refuses (busy)", R"({"type":"error","code":"unavailable","message":"busy"})", "", ""},
		{"service stops the session: no_audio", "", R"({"type":"stopped","reason":"no_audio"})", ""},
		{"service stops the session: trial_over", "", R"({"type":"stopped","reason":"trial_over"})", ""},
		{"service closes: too_many_messages", "", "", "too_many_messages"},
		{"service says the captions broke", "", R"({"type":"stopped","reason":"captions_unavailable"})", ""},
	};
	for (const Case &c : cases) {
		std::printf("  %s\n", c.name);
		FakeServer srv;
		srv.mode = Mode::Silent;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.error = c.error;
			srv.after_ready = c.after_ready;
			srv.close_reason = c.close_reason;
		}
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		const Capture cap = run_audio(s, 4.0);
		const int d = static_cast<int>(std::lround(s.delay_seconds() * 48000));
		CHECK(d == 7200);
		CHECK(is_plain_delayed_copy(cap, d));
		CHECK(cap.max_call_us < 30000);
		CHECK(!s.status_line().empty());
		s.stop(0, 500);
	}
	{
		std::printf("  service unreachable\n");
		Session s(base_config("ws://127.0.0.1:1/x"), nullptr, nullptr);
		s.start();
		const Capture cap = run_audio(s, 3.0);
		CHECK(is_plain_delayed_copy(cap, static_cast<int>(std::lround(s.delay_seconds() * 48000))));
		CHECK(cap.max_call_us < 30000);
		s.stop(0, 500);
	}
	{
		std::printf("  service ends mid-session (server dropped after 100 frames)\n");
		FakeServer srv;
		srv.mode = Mode::DropAfter100;
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		const Capture cap = run_audio(s, 5.0);
		// Processed frames came back for a while (the echo), so the audio is not a plain copy; what matters is that it
		// never went silent and never jumped: every block carries a signal as strong as the input's.
		double min_rms = 1e9;
		const size_t blk = 480;
		for (size_t i = 8000; i + blk <= cap.out.size(); i += blk) {
			double e = 0;
			for (size_t k = 0; k < blk; ++k)
				e += static_cast<double>(cap.out[i + k]) * cap.out[i + k];
			min_rms = std::min(min_rms, std::sqrt(e / blk));
		}
		std::printf("    quietest 10 ms block: rms %.4f (input 0.1)\n", min_rms);
		CHECK(min_rms > 0.05);
		CHECK(cap.max_call_us < 30000);
		s.stop(0, 500);
	}
}

static void test_stop_reasons()
{
	std::printf("stop reasons: plain words, reconnect at once when audio flows again after no_audio\n");
	{
		std::printf("  no_audio\n");
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.after_ready = R"({"type":"stopped","reason":"no_audio"})";
		}
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Feeder f(s, nullptr);
		CHECK(wait_until([&] { return srv.starts.load() >= 2; }, 4000));  // not the 2 minutes of an unknown reason
		CHECK(wait_until([&] { return s.snapshot().state == SessionState::Backoff; }, 3000));
		const SessionSnapshot n = s.snapshot();
		CHECK(n.reason_code == "no_audio");
		CHECK(n.backoff_seconds <= 8);
		CHECK(contains(s.status_line(), "microphone audio"));
		CHECK(contains(s.status_line(), "Your audio keeps playing without cleaning"));
		f.stop();
		s.stop(0, 500);
	}
	{
		std::printf("  no audio from OBS at all: nothing connects, nothing is billed\n");
		FakeServer srv;
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		std::this_thread::sleep_for(milliseconds(800));
		CHECK(srv.opens.load() == 0);
		s.stop(0, 500);
	}
	struct Case {
		const char *name;
		const char *error;
		const char *after_ready;
		const char *close_reason;
		bool blocks;
		int min_backoff_s;
		const char *text;
		const char *code;
	};
	const Case cases[] = {
		{"trial_over (stopped)", "", R"({"type":"stopped","reason":"trial_over"})", "", true, 0, "free trial", "trial_over"},
		{"trial_over (error)", R"({"type":"error","code":"trial_over","message":"x"})", "", "", true, 0, "free trial", "trial_over"},
		{"needs_topup", R"({"type":"error","code":"needs_topup","message":"Top up first"})", "", "", true, 0, "first top-up", "needs_topup"},
		{"trial_busy", R"({"type":"error","code":"trial_busy","message":"busy"})", "", "", false, 60, "free trials is busy", "trial_busy"},
		{"too_many_messages (close frame)", "", "", "too_many_messages", false, 20, "too many messages", "too_many_messages"},
		{"too_many_messages (error)", R"({"type":"error","code":"rate_limited","message":"Too many messages."})", "", "", false, 20,
		 "too many messages", "too_many_messages"},
		{"captions_unavailable (stopped)", "", R"({"type":"stopped","reason":"captions_unavailable"})", "", false, 300, "caption service",
		 "captions_unavailable"},
		{"insufficient_funds (stopped)", "", R"({"type":"stopped","reason":"insufficient_funds"})", "", true, 0, "balance ran out",
		 "insufficient_funds"},
	};
	for (const Case &c : cases) {
		std::printf("  %s\n", c.name);
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.error = c.error;
			srv.after_ready = c.after_ready;
			srv.close_reason = c.close_reason;
		}
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Feeder f(s, nullptr);
		const SessionState want = c.blocks ? SessionState::Blocked : SessionState::Backoff;
		CHECK(wait_until([&] { return s.snapshot().state == want; }, 5000));
		const SessionSnapshot n = s.snapshot();
		CHECK(n.reason_code == c.code);
		CHECK(contains(s.status_line(), c.text));
		if (!c.blocks)
			CHECK(n.backoff_seconds >= c.min_backoff_s);
		// The plugin's own wording wins when it has one.
		const std::string want_id = std::string("reason.") + c.code;
		const std::string loc = s.status_line(
			[&](const std::string &id) { return id == want_id ? std::string("LOCALISED[%1]") : std::string(); });
		CHECK(contains(loc, "LOCALISED["));
		CHECK(!contains(loc, c.text));
		std::this_thread::sleep_for(milliseconds(1500));
		CHECK(srv.starts.load() == 1);  // none of these is retried at once
		f.stop();
		s.stop(0, 500);
	}
}

// Translations: what is asked for, what the service switches on, what arrives, what goes wrong.
static void test_translations()
{
	std::printf("translations: start request, ready echo, events with times, warnings\n");
	const std::string ready_es_de =
		R"({"type":"ready","sessionId":"s","sampleRate":48000,"hopSamples":480,"modelLatencyMs":30.0,"usdPerMinute":0.01,)"
		R"("maxSessionSeconds":43200,"translate":["es","de"]})";
	{
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.ready = ready_es_de;
			srv.after_ready =
				"{\"type\":\"caption\",\"id\":\"l1\",\"text\":\"hello there\",\"final\":true,\"start\":1.0,\"end\":2.5}\n"
				"{\"type\":\"translation\",\"lang\":\"es\",\"text\":\"hola\",\"of\":\"l1\",\"start\":1.0,\"end\":2.5}\n"
				"{\"type\":\"translation\",\"lang\":\"de\",\"text\":\"hallo\",\"of\":\"l1\",\"start\":1.0,\"end\":2.5}\n"
				"{\"type\":\"translation\",\"lang\":\"fr\",\"text\":\"\",\"of\":\"l1\"}";
		}
		std::mutex mu;
		std::vector<CaptionEvent> caps;
		std::vector<TranslationEvent> trs;
		SessionConfig cfg = base_config(srv.url());
		cfg.denoise = false;
		cfg.captions = "local";
		cfg.translate = {"es", "de"};
		Session s(
			cfg,
			[&](const CaptionEvent &e) {
				std::lock_guard<std::mutex> lk(mu);
				caps.push_back(e);
			},
			nullptr);
		s.set_translation_callback([&](const TranslationEvent &e) {
			std::lock_guard<std::mutex> lk(mu);
			trs.push_back(e);
		});
		s.start();
		Feeder f(s, nullptr);
		CHECK(wait_until(
			[&] {
				std::lock_guard<std::mutex> lk(mu);
				return caps.size() >= 1 && trs.size() >= 2;
			},
			5000));
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			CHECK(contains(srv.last_start, "\"translate\":[\"es\",\"de\"]"));
			CHECK(contains(srv.last_start, "\"captions\":\"local\""));
		}
		std::this_thread::sleep_for(milliseconds(300));
		{
			std::lock_guard<std::mutex> lk(mu);
			CHECK(caps.size() == 1 && caps[0].id == "l1" && caps[0].text == "hello there");
			CHECK(caps[0].origin != steady_clock::time_point{});
			CHECK(trs.size() == 2);
			if (trs.size() >= 2) {
				CHECK(trs[0].lang == "es" && trs[0].text == "hola" && trs[0].of == "l1");
				CHECK(trs[0].start == 1.0 && trs[0].end == 2.5);
				CHECK(trs[0].origin == caps[0].origin);
				CHECK(trs[1].lang == "de" && trs[1].text == "hallo");
			}
		}
		const SessionSnapshot n = s.snapshot();
		CHECK(n.translate_active_known && n.translate_active.size() == 2);
		CHECK(contains(s.status_line(), "Captions are running. Translating into Spanish and German."));
		CHECK(contains(s.status_line([](const std::string &id) {
			       return id == "status.translating" ? std::string("T[%1]") : std::string();
		       }),
			       "T[Spanish and German]"));
		f.stop();
		s.stop(0, 500);
	}
	{
		std::printf("  an older service leaves the translations out of `ready`\n");
		FakeServer srv;
		SessionConfig cfg = base_config(srv.url());
		cfg.denoise = false;
		cfg.captions = "local";
		cfg.translate = {"es"};
		Session s(cfg, nullptr, nullptr);
		s.start();
		Feeder f(s, nullptr);
		CHECK(wait_until([&] { return s.snapshot().state == SessionState::Streaming; }, 5000));
		CHECK(contains(s.status_line(), "did not switch on translation into Spanish"));
		CHECK(!contains(s.status_line(), "Translating into"));
		f.stop();
		s.stop(0, 500);
	}
	{
		std::printf("  translation_unavailable: that language stops, the captions go on\n");
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.ready = ready_es_de;
			srv.after_ready = R"({"type":"warning","code":"translation_unavailable","lang":"es","message":"engine down"})";
		}
		SessionConfig cfg = base_config(srv.url());
		cfg.denoise = false;
		cfg.captions = "local";
		cfg.translate = {"es", "de"};
		Session s(cfg, nullptr, nullptr);
		s.start();
		Feeder f(s, nullptr);
		CHECK(wait_until([&] { return !s.snapshot().translate_failed.empty(); }, 5000));
		const std::string line = s.status_line();
		CHECK(contains(line, "Translation into Spanish stopped"));
		CHECK(contains(line, "Translating into German."));
		CHECK(contains(line, "Captions are running"));
		CHECK(s.snapshot().state == SessionState::Streaming);
		f.stop();
		s.stop(0, 500);
	}
	{
		std::printf("  captions_unavailable warning: noise removal goes on\n");
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.after_ready = R"({"type":"warning","code":"captions_unavailable","message":"x"})";
		}
		SessionConfig cfg = base_config(srv.url());
		cfg.captions = "local";
		Session s(cfg, nullptr, nullptr);
		s.start();
		Feeder f(s, nullptr);
		CHECK(wait_until([&] { return s.snapshot().captions_failed; }, 5000));
		CHECK(contains(s.status_line(), "captions have stopped"));
		CHECK(!contains(s.status_line(), "Captions are running"));
		CHECK(s.snapshot().state == SessionState::Streaming);
		f.stop();
		s.stop(0, 500);
	}
	{
		std::printf("  no captions: no translations are asked for\n");
		FakeServer srv;
		SessionConfig cfg = base_config(srv.url());
		cfg.translate = {"es"};
		Session s(cfg, nullptr, nullptr);
		s.start();
		Feeder f(s, nullptr);
		CHECK(wait_until([&] { return s.snapshot().state == SessionState::Streaming; }, 5000));
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			CHECK(!contains(srv.last_start, "translate"));
		}
		f.stop();
		s.stop(0, 500);
	}
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

// A signed release on the fake web server: /m/manifest.json, /m/manifest.json.sig and the files.
struct FakeRelease {
	std::string payload_win = std::string(300000, 'w') + "windows payload";
	std::string payload_mac = "mac payload";
	std::string manifest;
	std::string sig;
	updtest::KeyPair key = updtest::make_key(1);

	static std::string manifest_for(const std::string &version, const std::string &win_payload, const std::string &mac_payload)
	{
		return "{\"files\":{\"macos\":" + updtest::file_entry("plugin-" + version + "-mac.tar.xz", mac_payload) +
		       ",\"windows-x64\":" + updtest::file_entry("plugin-" + version + "-windows.zip", win_payload) + "},\"version\":\"" + version + "\"}";
	}
	void publish(FakeWeb &web, const std::string &version, const std::string &prefix = "/m/")
	{
		manifest = manifest_for(version, payload_win, payload_mac);
		sig = updtest::sign(key, manifest);
		web.set(prefix + "manifest.json", FakeWeb::ok(manifest));
		web.set(prefix + "manifest.json.sig", FakeWeb::ok(sig, "application/octet-stream"));
		web.set(prefix + "plugin-" + version + "-windows.zip", FakeWeb::ok(payload_win, "application/zip"));
		web.set(prefix + "plugin-" + version + "-mac.tar.xz", FakeWeb::ok(payload_mac, "application/octet-stream"));
	}
};

static void test_update_check()
{
	std::printf("update check against a local web server (signed manifest)\n");
	FakeWeb web;
	FakeRelease rel;
	rel.publish(web, "0.1.57");
	const updtest::KeyPair other = updtest::make_key(2);

	auto opts = [&](const std::string &folder = "/m/") {
		UpdateCheckOptions o;
		o.base_url = web.base() + folder;
		o.public_key = rel.key.pub_b64();
		o.allow_local_http = true;
		o.user_agent = "obs-audio-cleaner/0.1.9";
		o.connect_timeout_s = 2;
		o.transfer_timeout_s = 2;
		o.total_timeout_s = 6;
		return o;
	};
	auto run = [&](const char *current, const std::string &folder = "/m/") {
		UpdateCheck c;
		return c.run(current, opts(folder));
	};

	{
		const auto r = run("0.1.9");
		CHECK(r.status == UpdateStatus::UpdateAvailable);
		CHECK(r.manifest.version == "0.1.57" && r.manifest.files.size() == 2);
		CHECK(r.manifest.files.at(kKeyWindows).name == "plugin-0.1.57-windows.zip");
		CHECK(r.manifest.files.at(kKeyWindows).size == rel.payload_win.size());
		std::lock_guard<std::mutex> lk(web.mu);
		CHECK(web.last_user_agent == "obs-audio-cleaner/0.1.9");
	}
	CHECK(run("0.1.57").status == UpdateStatus::UpToDate);
	CHECK(run("0.1.100").status == UpdateStatus::UpToDate);  // numbers, not text
	CHECK(run("0.1.56").status == UpdateStatus::UpdateAvailable);
	CHECK(run("0.2.0").status == UpdateStatus::UpToDate);    // an old signed manifest never downgrades
	CHECK(run("not a version").status == UpdateStatus::Failed);

	// The signature decides.
	{
		FakeRelease bad;
		bad.publish(web, "0.1.58", "/tampered/");
		std::string m = bad.manifest;
		m[m.find("0.1.58")] = '9';  // one changed character, the old signature
		web.set("/tampered/manifest.json", FakeWeb::ok(m));
		auto r = run("0.1.9", "/tampered/");
		CHECK(r.status == UpdateStatus::Failed && contains(r.error, "signature"));

		bad.key = other;  // signed by somebody else
		bad.publish(web, "0.1.58", "/wrongkey/");
		r = run("0.1.9", "/wrongkey/");
		CHECK(r.status == UpdateStatus::Failed && contains(r.error, "signature"));

		bad.key = rel.key;
		bad.publish(web, "0.1.58", "/shortsig/");
		web.set("/shortsig/manifest.json.sig", FakeWeb::ok(bad.sig.substr(0, 63), "application/octet-stream"));
		CHECK(run("0.1.9", "/shortsig/").status == UpdateStatus::Failed);
		bad.publish(web, "0.1.58", "/longsig/");
		web.set("/longsig/manifest.json.sig", FakeWeb::ok(bad.sig + "x", "application/octet-stream"));
		CHECK(run("0.1.9", "/longsig/").status == UpdateStatus::Failed);
		bad.publish(web, "0.1.58", "/nosig/");
		web.set("/nosig/manifest.json.sig", FakeWeb::status(404));
		auto r2 = run("0.1.9", "/nosig/");
		CHECK(r2.status == UpdateStatus::Failed && contains(r2.error, "404"));
		bad.publish(web, "0.1.58", "/swapped/");  // the signature of another version
		FakeRelease older;
		older.publish(web, "0.1.50", "/older/");
		web.set("/swapped/manifest.json.sig", FakeWeb::ok(older.sig, "application/octet-stream"));
		CHECK(run("0.1.9", "/swapped/").status == UpdateStatus::Failed);
		bad.publish(web, "0.1.58", "/hex/");  // a hex or base64 text signature is not the raw one
		web.set("/hex/manifest.json.sig", FakeWeb::ok(acw::hex_encode(reinterpret_cast<const uint8_t *>(bad.sig.data()), 64)));
		CHECK(run("0.1.9", "/hex/").status == UpdateStatus::Failed);

		// the default key is the real one: a test-signed manifest is refused with it
		UpdateCheckOptions o = opts();
		o.public_key.clear();
		UpdateCheck c;
		CHECK(c.run("0.1.9", o).status == UpdateStatus::Failed);
	}

	// A correctly signed file that is not a manifest, a missing manifest, errors.
	{
		auto publish_text = [&](const std::string &folder, const std::string &text) {
			web.set(folder + "manifest.json", FakeWeb::ok(text));
			web.set(folder + "manifest.json.sig", FakeWeb::ok(updtest::sign(rel.key, text), "application/octet-stream"));
		};
		publish_text("/garbage/", "<html>not json</html>");
		publish_text("/nover/", "{\"files\":{}}");
		publish_text("/badver/", "{\"version\":\"latest\",\"files\":{}}");
		for (const char *f : {"/garbage/", "/nover/", "/badver/"}) {
			const auto r = run("0.1.9", f);
			CHECK(r.status == UpdateStatus::Failed && contains(r.error, "unreadable"));
		}
		web.set("/err500/manifest.json", FakeWeb::status(500));
		CHECK(contains(run("0.1.9", "/err500/").error, "500"));
		CHECK(contains(run("0.1.9", "/nothing-here/").error, "404"));
	}

	// Size limit: a manifest over the limit is refused (declared long, or really long).
	{
		web.set("/big/manifest.json", FakeWeb::ok(std::string(20000, ' ') + rel.manifest));
		web.set("/big/manifest.json.sig", FakeWeb::ok(updtest::sign(rel.key, std::string(20000, ' ') + rel.manifest), "application/octet-stream"));
		const auto r = run("0.1.9", "/big/");
		CHECK(r.status == UpdateStatus::Failed && contains(r.error, "large"));
		web.set("/endless/manifest.json", "<endless>");
		const auto t0 = steady_clock::now();
		const auto r2 = run("0.1.9", "/endless/");
		CHECK(r2.status == UpdateStatus::Failed && contains(r2.error, "large"));
		CHECK(steady_clock::now() - t0 < seconds(3));
	}

	// Redirects: followed on the same server (relative and absolute), refused anywhere else.
	{
		web.set("/r1/manifest.json", FakeWeb::redirect("/r2/manifest.json", 301));
		web.set("/r2/manifest.json", FakeWeb::redirect(web.base() + "/m/manifest.json", 307));
		web.set("/r1/manifest.json.sig", FakeWeb::redirect("/m/manifest.json.sig", 302));
		CHECK(run("0.1.9", "/r1/").status == UpdateStatus::UpdateAvailable);
		web.set("/loop/manifest.json", FakeWeb::redirect("/loop/manifest.json"));
		CHECK(run("0.1.9", "/loop/").status == UpdateStatus::Failed);
		web.set("/plain/manifest.json", FakeWeb::redirect("http://example.com/manifest.json"));
		CHECK(contains(run("0.1.9", "/plain/").error, "not allowed"));
		web.set("/ftp/manifest.json", FakeWeb::redirect("ftp://example.com/manifest.json"));
		CHECK(run("0.1.9", "/ftp/").status == UpdateStatus::Failed);
		web.set("/rel/manifest.json", FakeWeb::redirect("manifest2.json"));
		CHECK(run("0.1.9", "/rel/").status == UpdateStatus::Failed);
		web.set("/net/manifest.json", FakeWeb::redirect("//example.com/manifest.json"));
		CHECK(run("0.1.9", "/net/").status == UpdateStatus::Failed);
		// another server, even a local one: refused (the plugin talks to one place only)
		FakeWeb elsewhere;
		FakeRelease rel2;
		rel2.publish(elsewhere, "0.1.99");
		web.set("/away/manifest.json", FakeWeb::redirect(elsewhere.base() + "/m/manifest.json"));
		const int asked = elsewhere.requests.load();
		const auto away = run("0.1.9", "/away/");
		CHECK(away.status == UpdateStatus::Failed && contains(away.error, "another server"));
		CHECK(elsewhere.requests.load() == asked);
	}

	// A server that never answers: the transfer timeout ends it; cancel() ends it at once.
	{
		web.set("/hang/manifest.json", "<hang>");
		const auto t0 = steady_clock::now();
		auto o = opts("/hang/");
		o.transfer_timeout_s = 1;
		UpdateCheck c;
		CHECK(c.run("0.1.9", o).status == UpdateStatus::Failed);
		CHECK(steady_clock::now() - t0 < seconds(4));
	}
	{
		UpdateCheck c;
		auto o = opts("/hang/");
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
		CHECK(c.run("0.1.9", opts()).status == UpdateStatus::Failed);  // a cancelled check stays cancelled
	}

	// Addresses that are never used: plain http:// (unless a test allows the local one), other schemes, no folder.
	{
		auto o = opts();
		o.allow_local_http = false;
		const int before = web.requests.load();
		UpdateCheck c;
		const auto r = c.run("0.1.9", o);
		CHECK(r.status == UpdateStatus::Failed && contains(r.error, "not allowed"));
		CHECK(web.requests.load() == before);  // nothing was sent
	}
	for (const char *u : {"http://example.com/p/", "ftp://127.0.0.1/p/", "file:///etc/", "", "https://", "https://127.0.0.1/no-slash"}) {
		auto o = opts();
		o.base_url = u;
		UpdateCheck c;
		CHECK(c.run("0.1.9", o).status == UpdateStatus::Failed);
	}
	// https:// to a port where nobody listens, and to a server that only speaks plain HTTP: plain failures.
	{
		auto o = opts();
		o.base_url = "https://127.0.0.1:" + std::to_string(ix::getFreePort()) + "/m/";
		const auto t0 = steady_clock::now();
		UpdateCheck c;
		CHECK(c.run("0.1.9", o).status == UpdateStatus::Failed);
		CHECK(steady_clock::now() - t0 < seconds(5));
	}
	{
		auto o = opts();
		o.base_url = "https://127.0.0.1:" + std::to_string(web.getPort()) + "/m/";
		UpdateCheck c;
		CHECK(c.run("0.1.9", o).status == UpdateStatus::Failed);
	}
	// The address the plugin ships with: one place, no GitHub.
	{
		const std::string base = kUpdateBaseUrl;
		CHECK(base == "https://mc-multi-world.space/audio/api/downloads/obs-plugin/");
		CHECK(base.find("github") == std::string::npos && base.find("Benjamin") == std::string::npos);
	}
}

static void test_update_download()
{
	std::printf("update download against a local web server\n");
	FakeWeb web;
	FakeRelease rel;
	rel.publish(web, "0.1.57");
	const UpdateFile win = [&] {
		UpdateManifest m;
		parse_update_manifest(rel.manifest, m);
		return m.files.at(kKeyWindows);
	}();
	const UpdateFile mac = [&] {
		UpdateManifest m;
		parse_update_manifest(rel.manifest, m);
		return m.files.at(kKeyMac);
	}();
	auto tmp = std::filesystem::path(testing_detail::tmp_dir());
	std::filesystem::create_directories(tmp);
	auto listing = [&] {
		std::string names;
		for (const auto &e : std::filesystem::directory_iterator(tmp))
			names += e.path().filename().string() + ";";
		return names;
	};
	auto opts = [&](const std::string &folder = "/m/") {
		DownloadOptions o;
		o.base_url = web.base() + folder;
		o.allow_local_http = true;
		o.connect_timeout_s = 2;
		o.transfer_timeout_s = 5;
		o.user_agent = "obs-audio-cleaner/0.1.9";
		return o;
	};
	auto read = [](const std::string &path) {
		std::ifstream f(path, std::ios::binary);
		return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	};

	{
		UpdateDownload d;
		const auto r = d.run(win, tmp.string(), opts());
		CHECK(r.ok && r.error.empty());
		CHECK(read(r.path) == rel.payload_win);
		CHECK(std::filesystem::path(r.path).filename() == "plugin-0.1.57-windows.zip");
		CHECK(d.bytes_done() == rel.payload_win.size());
		CHECK(listing() == "plugin-0.1.57-windows.zip;");  // no .part left
		std::filesystem::remove(r.path);
	}
	{
		UpdateDownload d;
		const auto r = d.run(mac, tmp.string(), opts());
		CHECK(r.ok && read(r.path) == rel.payload_mac);
		std::filesystem::remove(r.path);
	}
	auto expect_fail = [&](const char *what, const UpdateFile &f, const std::string &folder, const char *why) {
		UpdateDownload d;
		const auto r = d.run(f, tmp.string(), opts(folder));
		const bool ok = !r.ok && r.path.empty() && contains(r.error, why) && listing().empty();
		CHECK(ok);
		if (!ok)
			std::printf("  (%s) ok=%d error='%s' dir='%s'\n", what, r.ok, r.error.c_str(), listing().c_str());
	};
	{  // the server sends other bytes than the signed list names
		FakeRelease evil;
		evil.payload_win = std::string(rel.payload_win.size(), 'E');  // same size, other content
		evil.publish(web, "0.1.57", "/evil/");
		web.set("/evil/plugin-0.1.57-windows.zip", FakeWeb::ok(evil.payload_win));
		expect_fail("same size, other bytes", win, "/evil/", "not the one");
		web.set("/evil/plugin-0.1.57-windows.zip", FakeWeb::ok(rel.payload_win + "x"));
		expect_fail("longer", win, "/evil/", "not the one");
		web.set("/evil/plugin-0.1.57-windows.zip", FakeWeb::ok(rel.payload_win.substr(0, rel.payload_win.size() - 1)));
		expect_fail("shorter", win, "/evil/", "incomplete");
		// the connection ends before the announced length
		web.set("/evil/plugin-0.1.57-windows.zip", "HTTP/1.1 200 OK\r\nContent-Length: 400000\r\nConnection: close\r\n\r\nabc");
		expect_fail("cut off", win, "/evil/", "failed");
		web.set("/evil/plugin-0.1.57-windows.zip", FakeWeb::ok(""));
		expect_fail("empty", win, "/evil/", "incomplete");
	}
	expect_fail("404", win, "/nothing/", "404");
	{
		web.set("/e500/plugin-0.1.57-windows.zip", FakeWeb::status(500));
		expect_fail("500", win, "/e500/", "500");
	}
	{  // redirects: followed on the same server; the body of a redirect or an error never ends up in the file
		web.set("/redir/plugin-0.1.57-windows.zip", "HTTP/1.1 302 Found\r\nLocation: /m/plugin-0.1.57-windows.zip\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello");
		UpdateDownload d;
		const auto r = d.run(win, tmp.string(), opts("/redir/"));
		CHECK(r.ok && read(r.path) == rel.payload_win);
		std::filesystem::remove(r.path);
		FakeWeb elsewhere;
		FakeRelease rel2;
		rel2.publish(elsewhere, "0.1.57");
		web.set("/away/plugin-0.1.57-windows.zip", FakeWeb::redirect(elsewhere.base() + "/m/plugin-0.1.57-windows.zip"));
		const int asked = elsewhere.requests.load();
		expect_fail("other server", win, "/away/", "failed");
		CHECK(elsewhere.requests.load() == asked);
		web.set("/plain/plugin-0.1.57-windows.zip", FakeWeb::redirect("http://example.com/x.zip"));
		expect_fail("plain http", win, "/plain/", "failed");
	}
	{  // names that are not plain names are refused before anything is sent
		const int before = web.requests.load();
		for (const char *bad : {"../x.zip", "a/b.zip", "..", "", "x y.zip", "C:\\x.zip", ".hidden"}) {
			UpdateFile f = win;
			f.name = bad;
			UpdateDownload d;
			const auto r = d.run(f, tmp.string(), opts());
			CHECK(!r.ok && r.path.empty());
		}
		UpdateFile f = win;
		f.size = 0;
		UpdateDownload d0;
		CHECK(!d0.run(f, tmp.string(), opts()).ok);
		f = win;
		f.sha512 = "abc";
		UpdateDownload d1;
		CHECK(!d1.run(f, tmp.string(), opts()).ok);
		f = win;
		f.size = kMaxUpdateFileBytes + 1;
		UpdateDownload d2;
		CHECK(!d2.run(f, tmp.string(), opts()).ok);
		CHECK(web.requests.load() == before && listing().empty());
	}
	{  // a server that sends far more than listed is stopped early
		UpdateFile f = win;
		web.set("/m/endless.zip", "<endless>");
		f.name = "endless.zip";
		UpdateDownload d;
		const auto t0 = steady_clock::now();
		const auto r = d.run(f, tmp.string(), opts());
		CHECK(!r.ok && contains(r.error, "not the one") && listing().empty());
		CHECK(steady_clock::now() - t0 < seconds(5));
	}
	{  // cancel() ends a download that has stalled
		UpdateFile f = win;
		f.name = "hang.zip";
		web.set("/m/hang.zip", "<hang>");
		UpdateDownload d;
		auto o = opts();
		o.transfer_timeout_s = 30;
		DownloadResult r;
		std::atomic<bool> done{false};
		std::thread th([&] {
			r = d.run(f, tmp.string(), o);
			done = true;
		});
		std::this_thread::sleep_for(milliseconds(400));
		CHECK(!done);
		d.cancel();
		CHECK(wait_until([&] { return done.load(); }, 3000));
		th.join();
		CHECK(!r.ok && r.cancelled && listing().empty());
	}
	{  // a folder that does not exist: a plain failure
		UpdateDownload d;
		const auto r = d.run(win, (tmp / "nope").string(), opts());
		CHECK(!r.ok && contains(r.error, "saved"));
	}
	std::error_code ec;
	std::filesystem::remove_all(tmp, ec);
}

// ---------------------------------------------------------------------------------------------------
// Silence frames, returnAudio, the network thread's wake-ups

static const char *kReadySilence =
	R"({"type":"ready","sessionId":"s","sampleRate":48000,"hopSamples":480,"modelLatencyMs":30.0,)"
	R"("usdPerMinute":0.01,"maxSessionSeconds":43200,"echoReference":true,"silenceFrames":true})";

// A host that feeds the session in real time, in blocks of `chunk` samples (OBS hands over 480 or 1024 at a time); what it feeds
// (noise, or a constant level) and whether the source is muted can change while it runs. Optionally keeps what went in and came out.
class Source {
public:
	explicit Source(Session &s, size_t chunk = 480, bool keep = false) : s_(s), chunk_(chunk), keep_(keep)
	{
		th_ = std::thread([this] { run(); });
	}
	~Source() { stop(); }
	void stop()
	{
		run_ = false;
		if (th_.joinable())
			th_.join();
	}
	std::atomic<bool> noise{true}, muted{false};
	std::atomic<float> level{0.0f};  // fed when `noise` is off
	Capture cap;                     // read after stop()

private:
	void run()
	{
		std::mt19937 rng(5);
		std::normal_distribution<float> nd(0.0f, 0.1f);
		std::vector<float> blk(chunk_), proc;
		const auto period = microseconds(static_cast<long long>(static_cast<double>(chunk_) * 1e6 / 48000.0));
		auto t = steady_clock::now();
		while (run_) {
			const bool nz = noise.load();
			const float lv = level.load();
			for (float &v : blk)
				v = nz ? nd(rng) + 0.001f : lv;
			proc = blk;
			s_.process(proc.data(), proc.size(), muted.load());
			if (keep_) {
				cap.in.insert(cap.in.end(), blk.begin(), blk.end());
				cap.out.insert(cap.out.end(), proc.begin(), proc.end());
			}
			t += period;
			std::this_thread::sleep_until(t);
		}
	}
	Session &s_;
	size_t chunk_;
	bool keep_;
	std::atomic<bool> run_{true};
	std::thread th_;
};

static bool streaming(Session &s)
{
	return s.snapshot().state == SessionState::Streaming;
}

static void test_silence_frames()
{
	std::printf("silence frames: only for a service that takes them; silence, a muted source and a quiet block are sent as such\n");
	// 1. A service that takes them: digital silence is not encoded.
	{
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.ready = kReadySilence;
		}
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Source src(s);
		src.noise = false;
		CHECK(wait_until([&] { return streaming(s); }, 5000));
		CHECK(wait_until([&] { return srv.silent_mic.load() > 100; }, 6000));
		const SessionSnapshot n = s.snapshot();
		CHECK(n.silence_active);
		CHECK(srv.opus_mic.load() == 0);
		CHECK(srv.silent_mic.load() >= srv.mic.load() - 2);
		CHECK(n.silent_sent >= 100 && n.sent - n.silent_sent <= 2);
		CHECK(n.tx_bytes <= 20 * n.sent);  // 18 bytes each
		CHECK(wait_until([&] { return s.snapshot().used > 60; }, 3000));  // the silence replies are played
		CHECK(s.snapshot().bad_frames == 0);
		CHECK(contains(s.status_details(), "silent frames"));
		CHECK(contains(s.status_line(), "Connected"));
		src.stop();
		s.stop(0, 500);
	}
	// 2. A service that does not mention them gets ordinary frames, silent or not.
	{
		FakeServer srv;
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Source src(s);
		src.noise = false;
		CHECK(wait_until([&] { return srv.opus_mic.load() > 100; }, 8000));
		CHECK(srv.silent_mic.load() == 0);
		CHECK(!s.snapshot().silence_active && s.snapshot().silent_sent == 0);
		src.muted = true;  // muted: still ordinary frames
		const int before = srv.opus_mic.load();
		CHECK(wait_until([&] { return srv.opus_mic.load() > before + 40; }, 3000));
		CHECK(srv.silent_mic.load() == 0);
		src.stop();
		s.stop(0, 500);
	}
	// 3. A muted source is silence (even though the audio is loud); unmuted, the audio is sent again.
	{
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.ready = kReadySilence;
		}
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Source src(s);
		src.muted = true;
		CHECK(wait_until([&] { return srv.silent_mic.load() > 60; }, 8000));
		CHECK(srv.opus_mic.load() <= 1);
		src.muted = false;
		CHECK(wait_until([&] { return srv.opus_mic.load() > 40; }, 4000));
		const int silent_then = srv.silent_mic.load();
		std::this_thread::sleep_for(milliseconds(800));
		CHECK(srv.silent_mic.load() - silent_then <= 2);  // loud audio is not mistaken for silence
		src.muted = true;
		const int opus_then = srv.opus_mic.load();
		CHECK(wait_until([&] { return srv.silent_mic.load() > silent_then + 40; }, 3000));
		CHECK(srv.opus_mic.load() - opus_then <= 3);
		src.stop();
		s.stop(0, 500);
	}
	// 4. "Silence" is every sample below 1e-5: a little above that is audio.
	{
		FakeServer srv;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.ready = kReadySilence;
		}
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Source src(s);
		src.noise = false;
		src.level = -6e-6f;
		CHECK(wait_until([&] { return srv.silent_mic.load() > 50; }, 8000));
		CHECK(srv.opus_mic.load() == 0);
		src.level = 3e-5f;
		CHECK(wait_until([&] { return srv.opus_mic.load() > 40; }, 4000));
		const int silent_then = srv.silent_mic.load();
		std::this_thread::sleep_for(milliseconds(600));
		CHECK(srv.silent_mic.load() - silent_then <= 2);
		src.stop();
		s.stop(0, 500);
	}
	// 5. The echo reference: a silent block is a silence frame (a service without them gets nothing, as before).
	for (int takes = 0; takes < 2; ++takes) {
		FakeServer srv;
		if (takes) {
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.ready = kReadySilence;
		}
		auto ref = std::make_shared<ReferenceBuffer>(48000);
		SessionConfig cfg = base_config(srv.url());
		cfg.reference = ref;
		cfg.echo = true;
		Session s(cfg, nullptr, nullptr);
		s.start();
		std::atomic<bool> run{true}, loud{false};
		std::thread rth([&] {
			std::vector<float> b(480);
			float ph = 0;
			auto t = steady_clock::now();
			while (run) {
				for (float &v : b) {
					v = loud ? 0.3f * std::sin(ph) : 0.0f;
					ph += 0.05f;
				}
				ref->push(b.data(), b.size());
				t += milliseconds(10);
				std::this_thread::sleep_until(t);
			}
		});
		Source src(s);
		CHECK(wait_until([&] { return srv.mic.load() > 150; }, 8000));
		if (takes) {
			CHECK(srv.refs.load() > 100);
			CHECK(srv.silent_refs.load() >= srv.refs.load() - 2);
			CHECK(srv.opus_mic.load() > 100);  // the microphone is loud: encoded
			loud = true;
			const int refs_then = srv.refs.load();
			CHECK(wait_until([&] { return srv.refs.load() > refs_then + 80; }, 4000));
			const int silent_then = srv.silent_refs.load();
			std::this_thread::sleep_for(milliseconds(800));
			CHECK(srv.silent_refs.load() - silent_then <= 2);
		} else {
			CHECK(srv.refs.load() == 0);
			CHECK(srv.silent_refs.load() == 0);
		}
		run = false;
		rth.join();
		src.stop();
		s.stop(0, 500);
	}
	// 6. The service's silence frames are zeros in the output, on the usual deadlines.
	{
		FakeServer srv;
		srv.mode = Mode::ReplySilence;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.ready = kReadySilence;
		}
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Source src(s, 480, true);
		CHECK(wait_until([&] { return s.snapshot().used > 150; }, 10000));
		src.stop();
		const Capture &c = src.cap;
		double peak = 0;
		for (size_t i = c.out.size() - 24000; i < c.out.size(); ++i)  // the last half second
			peak = std::max<double>(peak, std::fabs(c.out[i]));
		std::printf("  loud microphone, silence replies: output peak %.2g\n", peak);
		CHECK(peak == 0.0);
		CHECK(s.snapshot().bad_frames == 0);
		s.stop(0, 500);
	}
	// 7. A silence reply that is not valid is a bad frame: never played, the audio goes on unprocessed.
	{
		FakeServer srv;
		srv.mode = Mode::BadSilence;
		{
			std::lock_guard<std::mutex> lk(srv.mu);
			srv.ready = kReadySilence;
		}
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Source src(s, 480, true);
		CHECK(wait_until([&] { return s.snapshot().bad_frames > 50; }, 10000));
		src.stop();
		CHECK(s.snapshot().used == 0);
		CHECK(is_plain_delayed_copy(src.cap, static_cast<int>(std::lround(s.delay_seconds() * 48000.0))));
		s.stop(0, 500);
	}
}

static void test_return_audio()
{
	std::printf("captions only: the service is asked not to send audio back, and nothing waits for it\n");
	FakeServer srv;
	srv.honor_return_audio = true;
	{
		std::lock_guard<std::mutex> lk(srv.mu);
		srv.after_ready = R"({"type":"caption","text":"no audio back","final":true,"start":1.0,"end":2.0})";
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
	Source src(s, 480, true);
	CHECK(wait_until([&] { return streaming(s); }, 5000));
	CHECK(wait_until([&] { std::lock_guard<std::mutex> lk(mu); return !caps.empty(); }, 3000));
	{
		std::lock_guard<std::mutex> lk(srv.mu);
		CHECK(contains(srv.last_start, "\"returnAudio\":false"));
		CHECK(contains(srv.last_start, "\"denoise\":null"));
	}
	std::this_thread::sleep_for(seconds(3));
	const SessionSnapshot n = s.snapshot();
	CHECK(n.state == SessionState::Streaming && n.reconnects == 0);
	CHECK(n.sent > 120);
	CHECK(n.received == 0 && n.bad_frames == 0);  // none came back
	CHECK(contains(s.status_line(), "Captions are running"));
	src.stop();
	CHECK(s.delay_seconds() == 0.0);
	CHECK(is_plain_delayed_copy(src.cap, 0));  // the microphone is passed through at once, untouched
	s.stop(0, 500);

	// With noise removal the audio is wanted: the start message does not say otherwise.
	FakeServer srv2;
	srv2.honor_return_audio = true;
	Session s2(base_config(srv2.url()), nullptr, nullptr);
	s2.start();
	Source src2(s2);
	CHECK(wait_until([&] { return s2.snapshot().used > 30; }, 8000));
	{
		std::lock_guard<std::mutex> lk(srv2.mu);
		CHECK(!contains(srv2.last_start, "returnAudio"));
	}
	src2.stop();
	s2.stop(0, 500);
}

static void test_network_thread_sleeps()
{
	std::printf("network thread: sleeps until the next frame is due instead of polling\n");
	for (size_t chunk : {size_t(480), size_t(1024)}) {
		FakeServer srv;
		Session s(base_config(srv.url()), nullptr, nullptr);
		s.start();
		Source src(s, chunk);
		CHECK(wait_until([&] { return s.snapshot().used > 20; }, 8000));
		const SessionSnapshot a = s.snapshot();
		std::this_thread::sleep_for(seconds(3));
		const SessionSnapshot b = s.snapshot();
		const double loops_per_s = static_cast<double>(b.net_loops - a.net_loops) / 3.0;
		const double frames_per_s = static_cast<double>(b.sent - a.sent) / 3.0;
		std::printf("  chunks of %zu samples: %.0f loop rounds/s (was 500 with the 2 ms poll), %.1f frames/s, round trip %.2f ms, late %llu\n",
			    chunk, loops_per_s, frames_per_s, b.rtt_avg_ms, static_cast<unsigned long long>(b.late - a.late));
		CHECK(loops_per_s < 130.0);
		CHECK(frames_per_s > 0.93 * 48000.0 / 960.0);  // every frame goes out, none is held back
		CHECK(b.rtt_avg_ms < 30.0);
		CHECK(b.late - a.late <= 2 && b.dropped == 0);
		src.stop();
		s.stop(0, 500);
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
	test_changes_while_live();
	test_slow_link();
	test_server_errors();
	test_stop_reasons();
	test_translations();
	test_never_breaks_the_stream();
	test_http_refusals();
	test_ready_validation();
	test_captions_only();
	test_return_audio();
	test_silence_frames();
	test_network_thread_sleeps();
	test_robustness();
	test_stop_paths();
	test_update_check();
	test_update_download();
	ix::uninitNetSystem();
	std::printf("\n%d checks, %d failures\n", g_checks.load(), g_failures.load());
	return g_failures ? 1 : 0;
}
