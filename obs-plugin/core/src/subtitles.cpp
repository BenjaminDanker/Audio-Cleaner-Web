// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/subtitles.hpp"

#include "acw/caption_pacer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace acw {

std::string srt_time(double seconds)
{
	if (!(seconds > 0.0))
		seconds = 0.0;
	const long long ms_total = std::llround(seconds * 1000.0);
	const long long h = ms_total / 3600000, m = (ms_total / 60000) % 60, s = (ms_total / 1000) % 60, ms = ms_total % 1000;
	char buf[40];
	std::snprintf(buf, sizeof buf, "%02lld:%02lld:%02lld,%03lld", h, m, s, ms);
	return buf;
}

std::string make_srt(std::vector<SrtCue> cues, size_t line_chars)
{
	std::stable_sort(cues.begin(), cues.end(), [](const SrtCue &a, const SrtCue &b) { return a.start < b.start; });
	// Every cue is on screen for a readable time, then cut where the next one starts.
	for (size_t i = 0; i < cues.size(); ++i) {
		SrtCue &c = cues[i];
		if (!(c.end > c.start))
			c.end = c.start + std::min(6.0, std::max(1.0, static_cast<double>(utf8_count(c.text)) / 15.0));
		if (i + 1 < cues.size())
			c.end = std::max(std::min(c.end, cues[i + 1].start), c.start + 0.2);
	}
	std::string out;
	int n = 0;
	for (const SrtCue &c : cues) {
		const std::vector<std::string> lines = split_caption_lines(c.text, std::max<size_t>(8, line_chars));
		if (lines.empty())
			continue;
		out += std::to_string(++n) + "\n" + srt_time(c.start) + " --> " + srt_time(c.end) + "\n";
		for (const std::string &l : lines)
			out += l + "\n";
		out += "\n";
	}
	return out;
}

std::string subtitle_path(const std::string &recording_path, const std::string &lang)
{
	size_t dot = recording_path.rfind('.');
	const size_t slash = recording_path.find_last_of("/\\");
	if (dot == std::string::npos || (slash != std::string::npos && dot < slash) || dot == 0)
		dot = recording_path.size();
	return recording_path.substr(0, dot) + (lang.empty() ? "" : "." + lang) + ".srt";
}

bool write_text_file(const std::string &path, const std::string &content, std::string *error)
{
	namespace fs = std::filesystem;
	try {
		const fs::path target = fs::u8path(path);
		fs::path tmp = target;
		tmp += ".part";
		{
			std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
			if (!f) {
				if (error)
					*error = "cannot create " + tmp.u8string();
				return false;
			}
			f.write(content.data(), static_cast<std::streamsize>(content.size()));
			f.flush();
			if (!f) {
				if (error)
					*error = "cannot write " + tmp.u8string();
				std::error_code ec;
				fs::remove(tmp, ec);
				return false;
			}
		}
		std::error_code ec;
		fs::rename(tmp, target, ec);
		if (ec) {  // Windows does not replace an existing file with rename in every case
			fs::remove(target, ec);
			fs::rename(tmp, target, ec);
		}
		if (ec) {
			if (error)
				*error = ec.message();
			fs::remove(tmp, ec);
			return false;
		}
		return true;
	} catch (const std::exception &e) {
		if (error)
			*error = e.what();
		return false;
	}
}

void SubtitleRecorder::begin(std::chrono::steady_clock::time_point start)
{
	std::lock_guard<std::mutex> lk(mu_);
	active_ = true;
	t0_ = start;
	paused_ = false;
	pauses_.clear();
	cues_.clear();
	total_ = 0;
}

bool SubtitleRecorder::active() const
{
	std::lock_guard<std::mutex> lk(mu_);
	return active_;
}

double SubtitleRecorder::position(std::chrono::steady_clock::time_point t) const
{
	double s = std::chrono::duration<double>(t - t0_).count();
	for (const auto &p : pauses_) {
		if (t >= p.second)
			s -= std::chrono::duration<double>(p.second - p.first).count();
		else if (t > p.first)
			s -= std::chrono::duration<double>(t - p.first).count();
	}
	return s;
}

void SubtitleRecorder::pause(std::chrono::steady_clock::time_point now)
{
	std::lock_guard<std::mutex> lk(mu_);
	if (!active_ || paused_)
		return;
	paused_ = true;
	pause_start_ = now;
}

void SubtitleRecorder::resume(std::chrono::steady_clock::time_point now)
{
	std::lock_guard<std::mutex> lk(mu_);
	if (!active_ || !paused_)
		return;
	paused_ = false;
	if (now > pause_start_ && pauses_.size() < 1000)
		pauses_.emplace_back(pause_start_, now);
}

void SubtitleRecorder::add(const SubtitleCue &cue)
{
	if (cue.text.empty())
		return;
	std::lock_guard<std::mutex> lk(mu_);
	if (!active_ || total_ >= kMaxCues)
		return;
	std::chrono::steady_clock::time_point cue_end = cue.end;
	if (paused_) {  // speech during the pause is not in the recording
		if (cue.start >= pause_start_)
			return;
		cue_end = std::min(cue_end, pause_start_);
	}
	SrtCue c;
	c.start = position(cue.start);
	c.end = position(cue_end);
	c.text = cue.text;
	if (std::max(c.start, c.end) <= 0.0)
		return;  // spoken before the recording began
	c.start = std::max(0.0, c.start);
	cues_[cue.lang].push_back(std::move(c));
	++total_;
}

std::vector<SubtitleRecorder::File> SubtitleRecorder::finish()
{
	std::map<std::string, std::vector<SrtCue>> cues;
	{
		std::lock_guard<std::mutex> lk(mu_);
		if (!active_)
			return {};
		active_ = false;
		cues.swap(cues_);
		total_ = 0;
	}
	std::vector<File> out;
	for (auto &kv : cues) {
		File f;
		f.lang = kv.first;
		f.cues = kv.second.size();
		f.srt = make_srt(std::move(kv.second));
		if (!f.srt.empty())
			out.push_back(std::move(f));
	}
	return out;
}

void SubtitleRecorder::cancel()
{
	std::lock_guard<std::mutex> lk(mu_);
	active_ = false;
	cues_.clear();
	total_ = 0;
}

}  // namespace acw
