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

// Decodes the code point at s[i..]; advances i. Invalid bytes come back as themselves (>= 0x80 never
// matches the allowed sets below except by accident of Latin-1, so they are mapped to U+FFFD).
uint32_t next_cp(const std::string &s, size_t &i)
{
	const unsigned char c = static_cast<unsigned char>(s[i]);
	size_t l = utf8_len(c);
	if (l == 1 || i + l > s.size()) {
		++i;
		return c < 0x80 ? c : 0xFFFD;
	}
	uint32_t cp = c & (0xFF >> (l + 1));
	for (size_t k = 1; k < l; ++k) {
		const unsigned char cc = static_cast<unsigned char>(s[i + k]);
		if ((cc & 0xC0) != 0x80) {
			++i;
			return 0xFFFD;
		}
		cp = (cp << 6) | (cc & 0x3F);
	}
	i += l;
	return cp;
}

void put_cp(std::string &out, uint32_t cp)
{
	if (cp < 0x80) {
		out += static_cast<char>(cp);
	} else if (cp < 0x800) {
		out += static_cast<char>(0xC0 | (cp >> 6));
		out += static_cast<char>(0x80 | (cp & 0x3F));
	} else if (cp < 0x10000) {
		out += static_cast<char>(0xE0 | (cp >> 12));
		out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
		out += static_cast<char>(0x80 | (cp & 0x3F));
	} else {
		out += static_cast<char>(0xF0 | (cp >> 18));
		out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
		out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
		out += static_cast<char>(0x80 | (cp & 0x3F));
	}
}

// Latin-1 letters and signs the CEA-608 character sets contain.
bool cea608_latin1(uint32_t cp)
{
	if (cp >= 0xC0 && cp <= 0xFF) {
		switch (cp) {
		case 0xC6: case 0xD0: case 0xD7: case 0xDD: case 0xDE:  // AE, Eth, multiplication, Y acute, Thorn
		case 0xE6: case 0xF0: case 0xFD: case 0xFE: case 0xFF:  // ae, eth, y acute, thorn, y diaeresis
			return false;
		default:
			return true;
		}
	}
	switch (cp) {
	case 0xA1: case 0xA2: case 0xA3: case 0xA5: case 0xA9: case 0xAB: case 0xAE:
	case 0xB0: case 0xBB: case 0xBD: case 0xBF:
		return true;
	default:
		return false;
	}
}

// Letters the 608 character sets lack, written with plain ones (Polish, Czech, Turkish, Romanian, Nordic ...).
// `buf` holds the result (at most 2 characters + NUL).
const char *fold_latin(uint32_t cp, char (&buf)[3])
{
	static const char kExtA[] = "AaAaAaCcCcCcCcDd"  // U+0100
				    "DdEeEeEeEeEeGgGg"  // U+0110
				    "GgGgHhHhIiIiIiIi"  // U+0120
				    "IiIiJjKkkLlLlLlL"  // U+0130
				    "lLlNnNnNnnNnOoOo"  // U+0140
				    "OoOoRrRrRrSsSsSs"  // U+0150 (U+0152/3 are OE/oe)
				    "SsTtTtTtUuUuUuUu"  // U+0160
				    "UuUuWwYyYZzZzZzs"; // U+0170
	buf[0] = buf[1] = buf[2] = 0;
	if (cp >= 0x100 && cp <= 0x17F) {
		buf[0] = kExtA[cp - 0x100];
		if (cp == 0x152)
			buf[1] = 'E';
		else if (cp == 0x153)
			buf[1] = 'e';
		return buf;
	}
	switch (cp) {
	case 0xC6: return "AE";
	case 0xE6: return "ae";
	case 0xD0: return "D";
	case 0xF0: return "d";
	case 0xDD: return "Y";
	case 0xFD: case 0xFF: return "y";
	case 0xDE: return "Th";
	case 0xFE: return "th";
	case 0xD7: return "x";
	case 0x218: return "S";
	case 0x219: return "s";
	case 0x21A: return "T";
	case 0x21B: return "t";
	default: return nullptr;
	}
}

}  // namespace

size_t utf8_count(const std::string &s)
{
	size_t n = 0;
	for (unsigned char c : s)
		if ((c & 0xC0) != 0x80)
			++n;
	return n;
}

std::string to_cea608(const std::string &text, double *lost)
{
	std::string out;
	out.reserve(text.size());
	size_t visible = 0, dropped = 0;
	for (size_t i = 0; i < text.size();) {
		uint32_t cp = next_cp(text, i);
		const char *plain = nullptr;
		switch (cp) {
		case 0x2018: case 0x2019: case 0x201A: case 0x02BC: case 0x2032: plain = "'"; break;
		case 0x201C: case 0x201D: case 0x201E: case 0x2033: plain = "\""; break;
		case 0x2010: case 0x2011: case 0x2012: case 0x2013: case 0x2014: case 0x2015: case 0x2212: plain = "-"; break;
		case 0x2026: plain = "..."; break;
		case 0x00A0: case 0x2002: case 0x2003: case 0x2009: case 0x200A: case 0x202F: case 0x3000: plain = " "; break;
		case 0x200B: case 0x200C: case 0x200D: case 0xFEFF: plain = ""; break;  // zero width: not visible
		default: break;
		}
		if (plain) {
			out += plain;
			continue;
		}
		if (cp == '\t' || cp == '\n' || cp == '\r') {
			out += ' ';
			continue;
		}
		if (cp == ' ') {
			out += ' ';
			continue;
		}
		++visible;
		char fb[3];
		if ((cp >= 0x21 && cp <= 0x7E) || cea608_latin1(cp))
			put_cp(out, cp);
		else if (const char *f = fold_latin(cp, fb))
			out += f;
		else
			++dropped;
	}
	if (lost)
		*lost = visible ? static_cast<double>(dropped) / static_cast<double>(visible) : 0.0;
	return out;
}

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

	// Time the channel is kept busy after a line of `chars` characters.
	double gap_for(size_t chars) const
	{
		return std::max(opt.min_gap_seconds, static_cast<double>(chars) / opt.chars_per_second);
	}

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
			if (now > it.fresh_until) {
				dropped.fetch_add(1);
				continue;
			}
			const bool en = enabled;
			lk.unlock();
			const size_t chars = utf8_count(it.line);
			if (en) {
				const double dur = std::min(4.0, std::max(1.5, static_cast<double>(chars) / 12.0));
				send(it.line, dur);
				sent.fetch_add(1);
			}
			lk.lock();
			const double gap = gap_for(chars);
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

void CaptionPacer::push(const std::string &text_in)
{
	State &st = *st_;
	std::string text = text_in;
	if (st.opt.cea608) {
		double lost = 0.0;
		text = to_cea608(text_in, &lost);
		if (lost > 0.5) {  // a language the caption channel cannot show: better nothing than gibberish
			st.dropped.fetch_add(1);
			return;
		}
	}
	std::vector<std::string> lines = split_caption_lines(text, st.opt.max_chars);
	if (lines.empty())
		return;
	// A caption longer than the whole queue keeps its beginning (what was said first is read first).
	if (lines.size() > st.opt.max_queue_lines) {
		st.dropped.fetch_add(lines.size() - st.opt.max_queue_lines);
		lines.resize(st.opt.max_queue_lines);
	}
	using Clock = std::chrono::steady_clock;
	const auto now = Clock::now();
	{
		std::lock_guard<std::mutex> lk(st.mu);
		if (!st.enabled || st.stop)
			return;
		// No backlog: make room by dropping the oldest lines of earlier captions, never lines of this one.
		while (!st.q.empty() && st.q.size() + lines.size() > st.opt.max_queue_lines) {
			st.q.pop_front();
			st.dropped.fetch_add(1);
		}
		// Stale = waited longer than stale_seconds beyond the time the lines before it in this caption need.
		double ahead = 0.0;
		for (const std::string &l : lines) {
			const auto fresh = now + std::chrono::duration_cast<Clock::duration>(
							 std::chrono::duration<double>(st.opt.stale_seconds + ahead));
			st.q.push_back(Item{l, now, fresh});
			ahead += st.gap_for(utf8_count(l));
		}
	}
	st.cv.notify_all();
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
