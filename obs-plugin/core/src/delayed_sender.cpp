// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/delayed_sender.hpp"

#include <algorithm>

namespace acw {

struct DelayedSender::State {
	struct Item {
		std::string line;
		double seconds;
		std::chrono::steady_clock::time_point due;
	};
	SendFn send;
	size_t max_lines;
	mutable std::mutex mu;
	std::condition_variable cv;
	std::deque<Item> q;
	bool stopping = false;
	bool done = false;
	std::condition_variable done_cv;
	std::thread th;

	void run()
	{
		std::unique_lock<std::mutex> lk(mu);
		while (!stopping) {
			if (q.empty()) {
				cv.wait(lk, [this] { return stopping || !q.empty(); });
				continue;
			}
			const auto due = q.front().due;
			if (std::chrono::steady_clock::now() < due) {
				cv.wait_until(lk, due);
				continue;
			}
			Item it = std::move(q.front());
			q.pop_front();
			lk.unlock();
			send(it.line, it.seconds);
			lk.lock();
		}
		done = true;
		done_cv.notify_all();
	}
};

DelayedSender::DelayedSender(SendFn send, size_t max_lines) : st_(std::make_shared<State>())
{
	st_->send = std::move(send);
	st_->max_lines = std::max<size_t>(1, max_lines);
	auto st = st_;  // the thread keeps the state alive if it is detached
	st_->th = std::thread([st] { st->run(); });
}

DelayedSender::~DelayedSender()
{
	stop();
}

void DelayedSender::later(const std::string &line, double display_seconds, std::chrono::steady_clock::time_point due)
{
	std::lock_guard<std::mutex> lk(st_->mu);
	if (st_->stopping)
		return;
	if (!st_->q.empty() && due < st_->q.back().due)
		due = st_->q.back().due;  // keep the order even if the delay was shortened meanwhile
	st_->q.push_back({line, display_seconds, due});
	while (st_->q.size() > st_->max_lines)
		st_->q.pop_front();
	st_->cv.notify_all();
}

void DelayedSender::clear()
{
	std::lock_guard<std::mutex> lk(st_->mu);
	st_->q.clear();
}

void DelayedSender::stop(int timeout_ms)
{
	if (!st_ || !st_->th.joinable())
		return;
	{
		std::lock_guard<std::mutex> lk(st_->mu);
		st_->stopping = true;
		st_->q.clear();
		st_->cv.notify_all();
	}
	bool finished;
	{
		std::unique_lock<std::mutex> lk(st_->mu);
		finished = st_->done_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] { return st_->done; });
	}
	if (finished)
		st_->th.join();
	else
		st_->th.detach();  // a send that is stuck inside OBS: never hold up shutdown
}

size_t DelayedSender::waiting() const
{
	std::lock_guard<std::mutex> lk(st_->mu);
	return st_->q.size();
}

}  // namespace acw
