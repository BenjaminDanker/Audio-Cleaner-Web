// SPDX-License-Identifier: GPL-2.0-or-later
// Wait-free single-producer / single-consumer queue of fixed-size elements.
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace acw {

template<typename T>
class SpscQueue {
public:
	// capacity is rounded up to a power of two.
	explicit SpscQueue(size_t min_capacity)
	{
		size_t cap = 2;
		while (cap < min_capacity)
			cap <<= 1;
		cap_ = cap;
		mask_ = cap - 1;
		buf_.reset(new T[cap]());
	}

	size_t capacity() const { return cap_; }

	// --- producer ---------------------------------------------------------
	// Returns a pointer to the next free slot, or nullptr when full.
	T *claim()
	{
		const size_t h = head_.load(std::memory_order_relaxed);
		const size_t t = tail_.load(std::memory_order_acquire);
		if (h - t >= cap_)
			return nullptr;
		return &buf_[h & mask_];
	}
	void commit() { head_.store(head_.load(std::memory_order_relaxed) + 1, std::memory_order_release); }

	bool push(const T &v)
	{
		T *slot = claim();
		if (!slot)
			return false;
		*slot = v;
		commit();
		return true;
	}

	// --- consumer ---------------------------------------------------------
	// Returns the oldest element or nullptr when empty.
	T *front()
	{
		const size_t t = tail_.load(std::memory_order_relaxed);
		const size_t h = head_.load(std::memory_order_acquire);
		if (t == h)
			return nullptr;
		return &buf_[t & mask_];
	}
	void pop() { tail_.store(tail_.load(std::memory_order_relaxed) + 1, std::memory_order_release); }

	size_t size_approx() const
	{
		return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
	}

private:
	size_t cap_ = 0;
	size_t mask_ = 0;
	std::unique_ptr<T[]> buf_;
	alignas(64) std::atomic<size_t> head_{0};
	alignas(64) std::atomic<size_t> tail_{0};
};

}  // namespace acw
