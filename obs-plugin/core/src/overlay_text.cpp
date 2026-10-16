// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/overlay_text.hpp"

#include "acw/caption_pacer.hpp"

#include <algorithm>
#include <chrono>

namespace acw {

// ---------------------------------------------------------------------------
// OverlayText

double OverlayText::hold_for(const std::string &line) const
{
	const double secs = static_cast<double>(utf8_count(line)) / std::max(1.0, opt_.chars_per_second);
	return std::min(opt_.max_line_s, std::max(opt_.min_line_s, secs));
}

void OverlayText::rebuild()
{
	text_.clear();
	for (size_t i = 0; i < shown_.size(); ++i) {
		if (i > 0)
			text_ += '\n';
		text_ += shown_[i].text;
	}
}

void OverlayText::push(const std::string &text, double now)
{
	for (std::string &l : split_caption_lines(text, std::max<size_t>(8, opt_.line_chars)))
		pending_.push_back(Line{std::move(l), now});
	while (pending_.size() > std::max<size_t>(1, opt_.max_pending))
		pending_.pop_front();
}

// When the front waiting line may go up: at once while there is room, otherwise when the oldest line on screen has
// been up long enough to be read.
static double reveal_time(double arrived, size_t shown, size_t max_lines, double oldest_ready, double last_reveal)
{
	double t = arrived;
	if (shown >= max_lines)
		t = std::max(t, oldest_ready);
	return std::max(t, last_reveal);
}

bool OverlayText::update(double now)
{
	const std::string before = text_;
	const size_t max_lines = std::max<size_t>(1, opt_.max_lines);
	while (!pending_.empty()) {
		const Line &l = pending_.front();
		const double t = reveal_time(l.arrived, shown_.size(), max_lines, shown_.empty() ? 0.0 : shown_.front().ready_at, last_reveal_);
		if (t > now)
			break;
		const double hold = hold_for(l.text);
		shown_.push_back(Shown{l.text, t + hold});
		while (shown_.size() > max_lines)
			shown_.pop_front();
		last_reveal_ = t;
		clear_at_ = t + std::max(opt_.clear_after_s, hold);
		pending_.pop_front();
	}
	if (pending_.empty() && !shown_.empty() && now >= clear_at_)
		shown_.clear();
	rebuild();
	return text_ != before;
}

double OverlayText::seconds_until_change(double now) const
{
	double best = -1.0;
	auto consider = [&](double dt) {
		dt = std::max(0.0, dt);
		if (best < 0.0 || dt < best)
			best = dt;
	};
	if (!pending_.empty()) {
		const Line &l = pending_.front();
		const size_t max_lines = std::max<size_t>(1, opt_.max_lines);
		consider(reveal_time(l.arrived, shown_.size(), max_lines, shown_.empty() ? 0.0 : shown_.front().ready_at, last_reveal_) - now);
	} else if (!shown_.empty()) {
		consider(clear_at_ - now);
	}
	return best;
}

void OverlayText::clear()
{
	pending_.clear();
	shown_.clear();
	text_.clear();
}

// ---------------------------------------------------------------------------
// OverlayDisplay

struct OverlayDisplay::State {
	struct Entry {
		OverlayText text;
		std::string sent;  // what the sink got last
	};
	using Clock = std::chrono::steady_clock;

	SinkFn sink;
	std::mutex mu;
	std::condition_variable cv;
	std::map<std::string, Entry> entries;
	std::deque<std::string> to_clear;  // targets that were dropped: they get an empty text
	bool stop = false;
	bool done = false;
	std::mutex done_mu;
	std::condition_variable done_cv;
	std::thread th;
	Clock::time_point epoch = Clock::now();

	double now_s() const { return std::chrono::duration<double>(Clock::now() - epoch).count(); }

	void run()
	{
		std::unique_lock<std::mutex> lk(mu);
		for (;;) {
			std::vector<std::pair<std::string, std::string>> out;
			const double now = now_s();
			double wait = -1.0;
			for (auto &kv : entries) {
				Entry &e = kv.second;
				e.text.update(now);
				if (e.text.text() != e.sent) {
					e.sent = e.text.text();
					out.emplace_back(kv.first, e.sent);
				}
				const double w = e.text.seconds_until_change(now);
				if (w >= 0.0 && (wait < 0.0 || w < wait))
					wait = w;
			}
			for (std::string &name : to_clear)
				out.emplace_back(std::move(name), std::string());
			to_clear.clear();
			const bool stopping = stop;
			lk.unlock();
			for (auto &o : out) {
				try {
					if (sink)
						sink(o.first, o.second);
				} catch (...) {
				}
			}
			lk.lock();
			if (stopping)
				break;
			if (stop || !to_clear.empty())
				continue;
			if (wait < 0.0)
				cv.wait(lk);
			else
				cv.wait_for(lk, std::chrono::duration<double>(wait + 0.002));
		}
		{
			std::lock_guard<std::mutex> dl(done_mu);
			done = true;
		}
		done_cv.notify_all();
	}
};

OverlayDisplay::OverlayDisplay(SinkFn sink) : st_(std::make_shared<State>())
{
	st_->sink = std::move(sink);
	auto st = st_;
	st_->th = std::thread([st] { st->run(); });
}

OverlayDisplay::~OverlayDisplay()
{
	stop(300);
}

void OverlayDisplay::set_targets(const std::map<std::string, OverlayText::Options> &targets)
{
	{
		std::lock_guard<std::mutex> lk(st_->mu);
		for (auto it = st_->entries.begin(); it != st_->entries.end();) {
			if (targets.count(it->first)) {
				++it;
				continue;
			}
			if (!it->second.sent.empty())
				st_->to_clear.push_back(it->first);
			it = st_->entries.erase(it);
		}
		for (const auto &t : targets) {
			auto it = st_->entries.find(t.first);
			if (it == st_->entries.end())
				st_->entries[t.first].text.set_options(t.second);
			else
				it->second.text.set_options(t.second);
		}
	}
	st_->cv.notify_all();
}

void OverlayDisplay::push(const std::string &target, const std::string &text)
{
	{
		std::lock_guard<std::mutex> lk(st_->mu);
		auto it = st_->entries.find(target);
		if (it == st_->entries.end() || st_->stop)
			return;
		it->second.text.push(text, st_->now_s());
	}
	st_->cv.notify_all();
}

void OverlayDisplay::clear_all()
{
	{
		std::lock_guard<std::mutex> lk(st_->mu);
		for (auto &kv : st_->entries)
			kv.second.text.clear();
	}
	st_->cv.notify_all();
}

void OverlayDisplay::stop(int timeout_ms)
{
	State &s = *st_;
	if (!s.th.joinable())
		return;
	{
		std::lock_guard<std::mutex> lk(s.mu);
		for (auto &kv : s.entries)
			kv.second.text.clear();  // the thread sends the empty texts before it ends
		s.stop = true;
	}
	s.cv.notify_all();
	bool ok;
	{
		std::unique_lock<std::mutex> lk(s.done_mu);
		ok = s.done_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return s.done; });
	}
	if (ok)
		s.th.join();
	else
		s.th.detach();
}

}  // namespace acw
