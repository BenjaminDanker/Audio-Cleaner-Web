// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/caption_pacer.hpp"

#include <algorithm>
#include <atomic>
#include <memory>

namespace acw {

namespace {

// Number of bytes in the UTF-8 sequence starting with `lead`.
size_t utf8_len(unsigned char lead)
{
	if (lead < 0x80)
		return 1;
	if ((lead >> 5) == 0x6)
		return 2;
	if ((lead >> 4) == 0xE)
		return 3;
	if ((lead >> 3) == 0x1E)
		return 4;
	return 1;  // invalid lead byte: treat as one unit
}

// A word as a list of byte offsets of each code point (plus the end).
struct Word {
	std::string s;
	size_t cps = 0;
};

}  // namespace

std::vector<std::string> split_caption_lines(const std::string &text, size_t max_chars)
{
	std::vector<Word> words;
	{
		Word cur;
		for (size_t i = 0; i < text.size();) {
			const unsigned char c = static_cast<unsigned char>(text[i]);
			if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
				if (cur.cps) {
					words.push_back(std::move(cur));
					cur = Word{};
				}
				++i;
				continue;
			}
			const size_t l = std::min(utf8_len(c), text.size() - i);
			cur.s.append(text, i, l);
			++cur.cps;
			i += l;
		}
		if (cur.cps)
			words.push_back(std::move(cur));
	}

	std::vector<std::string> lines;
	Word line;
	auto flush = [&] {
		if (line.cps) {
			lines.push_back(std::move(line.s));
			line = Word{};
		}
	};
	for (Word &w : words) {
		// Hard-split words longer than a line.
		while (w.cps > max_chars) {
			flush();
			size_t bytes = 0, cps = 0;
			while (cps < max_chars && bytes < w.s.size()) {
				bytes += std::min(utf8_len(static_cast<unsigned char>(w.s[bytes])), w.s.size() - bytes);
				++cps;
			}
			lines.push_back(w.s.substr(0, bytes));
			w.s.erase(0, bytes);
			w.cps -= cps;
		}
		if (line.cps && line.cps + 1 + w.cps > max_chars)
			flush();
		if (line.cps) {
			line.s += ' ';
			++line.cps;
		}
		line.s += w.s;
		line.cps += w.cps;
	}
	flush();
	return lines;
}

struct CaptionPacer::State {
	SendFn send;
	Options opt;
	std::mutex mu;
	std::condition_variable cv;
	std::deque<Item> q;
	bool enabled = true;
	bool stop = false;
	bool done = false;
	std::thread th;
	std::atomic<uint64_t> sent{0}, dropped{0};
	std::mutex done_mu;
	std::condition_variable done_cv;

	void run()
	{
		using Clock = std::chrono::steady_clock;
		Clock::time_point not_before = Clock::now();
		std::unique_lock<std::mutex> lk(mu);
		while (!stop) {
			if (q.empty()) {
				cv.wait(lk);
				continue;
			}
			const auto now = Clock::now();
			if (now < not_before) {
				cv.wait_until(lk, not_before);
				continue;
			}
			Item it = std::move(q.front());
			q.pop_front();
			if (std::chrono::duration<double>(now - it.queued).count() > opt.stale_seconds) {
				dropped.fetch_add(1);
				continue;
			}
			const bool en = enabled;
			lk.unlock();
			const size_t chars = it.line.size();
			if (en) {
				const double dur = std::min(4.0, std::max(1.5, static_cast<double>(chars) / 12.0));
				send(it.line, dur);
				sent.fetch_add(1);
			}
			lk.lock();
			const double gap = std::max(opt.min_gap_seconds, static_cast<double>(chars) / opt.chars_per_second);
			not_before = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(gap));
		}
		{
			std::lock_guard<std::mutex> dl(done_mu);
			done = true;
		}
		done_cv.notify_all();
	}
};

CaptionPacer::CaptionPacer(SendFn send) : CaptionPacer(std::move(send), Options{}) {}

CaptionPacer::CaptionPacer(SendFn send, const Options &opt) : st_(std::make_shared<State>())
{
	st_->send = std::move(send);
	st_->opt = opt;
	auto st = st_;
	st_->th = std::thread([st] { st->run(); });
}

CaptionPacer::~CaptionPacer()
{
	stop(300);
}

void CaptionPacer::push(const std::string &text)
{
	const std::vector<std::string> lines = split_caption_lines(text, st_->opt.max_chars);
	if (lines.empty())
		return;
	const auto now = std::chrono::steady_clock::now();
	{
		std::lock_guard<std::mutex> lk(st_->mu);
		if (!st_->enabled || st_->stop)
			return;
		for (const std::string &l : lines) {
			// No backlog: the newest captions win.
			while (st_->q.size() >= st_->opt.max_queue_lines) {
				st_->q.pop_front();
				st_->dropped.fetch_add(1);
			}
			st_->q.push_back(Item{l, now});
		}
	}
	st_->cv.notify_all();
}

void CaptionPacer::set_enabled(bool on)
{
	{
		std::lock_guard<std::mutex> lk(st_->mu);
		st_->enabled = on;
		if (!on)
			st_->q.clear();
	}
	st_->cv.notify_all();
}

void CaptionPacer::stop(int timeout_ms)
{
	State &s = *st_;
	if (!s.th.joinable())
		return;
	{
		std::lock_guard<std::mutex> lk(s.mu);
		s.stop = true;
		s.enabled = false;
		s.q.clear();
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
		s.th.detach();  // a send callback is stuck: let the thread end on its own
}

uint64_t CaptionPacer::sent_lines() const
{
	return st_->sent.load();
}
uint64_t CaptionPacer::dropped_lines() const
{
	return st_->dropped.load();
}

}  // namespace acw
