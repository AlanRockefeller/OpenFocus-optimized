#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace openfocus {

struct CancelRequested : public std::runtime_error {
    CancelRequested() : std::runtime_error("OpenFocus native operation cancelled") {}
};

inline int clamp_threads(int num_threads, int64_t num_tasks) {
    if (num_tasks <= 1) {
        return 1;
    }
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) {
        hw = 4;
    }
    int requested = num_threads <= 0 ? hw : num_threads;
    int capped = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(requested, num_tasks)));
    return capped;
}

// Runs fn(start, stop, state) over [0, total) in bands. Each worker builds one
// state with make_state() and reuses it for every band it takes, so scratch
// buffers are allocated once per worker instead of once per band.
template <typename MakeState, typename Fn>
inline void parallel_bands_stateful(
    int64_t total,
    int64_t band,
    int num_threads,
    MakeState&& make_state,
    Fn&& fn,
    const std::atomic<bool>* cancel_flag = nullptr
) {
    if (total <= 0) {
        return;
    }
    const int64_t step = std::max<int64_t>(1, band);
    const int64_t num_tasks = (total + step - 1) / step;
    const int workers = clamp_threads(num_threads, num_tasks);

    auto check_cancel = [cancel_flag]() {
        if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_relaxed)) {
            throw CancelRequested();
        }
    };

    if (workers <= 1) {
        auto state = make_state();
        for (int64_t start = 0; start < total; start += step) {
            check_cancel();
            const int64_t stop = std::min<int64_t>(total, start + step);
            fn(start, stop, state);
        }
        check_cancel();
        return;
    }

    std::atomic<int64_t> next_task{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_error = nullptr;
    std::mutex err_mutex;

    auto worker_loop = [&]() {
        try {
            auto state = make_state();
            while (!failed.load(std::memory_order_relaxed)) {
                if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_relaxed)) {
                    throw CancelRequested();
                }
                const int64_t task_idx = next_task.fetch_add(1, std::memory_order_relaxed);
                if (task_idx >= num_tasks) {
                    break;
                }
                const int64_t start = task_idx * step;
                const int64_t stop = std::min<int64_t>(total, start + step);
                fn(start, stop, state);
            }
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(err_mutex);
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(workers - 1));
    for (int i = 0; i < workers - 1; ++i) {
        threads.emplace_back(worker_loop);
    }
    worker_loop();
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
    check_cancel();
}

template <typename Fn>
inline void parallel_bands(
    int64_t total,
    int64_t band,
    int num_threads,
    Fn&& fn,
    const std::atomic<bool>* cancel_flag = nullptr
) {
    parallel_bands_stateful(
        total, band, num_threads,
        []() { return 0; },
        [&fn](int64_t start, int64_t stop, int&) { fn(start, stop); },
        cancel_flag);
}

}  // namespace openfocus
