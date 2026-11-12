#pragma once
/// @file parallel_for.hpp
/// @brief Runs one piece of work per index on a pool of threads.

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief Runs work(index) for every index in [0, count) on a pool of threads.
class ParallelFor {
public:
    /// @brief Runs the work; the threads take the indices in increasing order.
    /// @param count the number of indices
    /// @param threads the number of threads, 1 or more
    /// @param work called once per index; it must be safe to call from several threads at once
    /// @throws std::invalid_argument when threads is below 1
    /// @throws the first error that a thread caught (the other threads stop taking new indices), after all threads stopped
    static void run(const std::size_t count, const int threads, const std::function<void(std::size_t)>& work)
    {
        if (threads < 1) {
            throw std::invalid_argument("ParallelFor: threads " + std::to_string(threads) + " (need 1 or more)");
        }
        std::atomic<std::size_t> next{ 0 };
        std::atomic<bool> failed{ false };
        std::exception_ptr first_error{};
        std::mutex lock{};
        const auto worker{ [&]() {
            for (std::size_t n{ next.fetch_add(1) }; n < count && !failed.load(); n = next.fetch_add(1)) {
                try {
                    work(n);
                } catch (...) { // a thread boundary: kept and rethrown after the join
                    const std::lock_guard<std::mutex> guard{ lock };
                    if (!first_error) {
                        first_error = std::current_exception();
                    }
                    failed.store(true);
                    return;
                }
            }
        } };
        std::vector<std::thread> pool{};
        for (int t{ 0 }; t < threads; ++t) {
            pool.emplace_back(worker);
        }
        for (std::thread& thread : pool) {
            thread.join();
        }
        if (first_error) {
            std::rethrow_exception(first_error);
        }
    }
};

} // namespace fluidx3d::experimental::wall_builder
