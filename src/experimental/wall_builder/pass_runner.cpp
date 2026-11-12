/// @file pass_runner.cpp
/// @brief Runs the passes on many threads; the results stay in the serial order of terrain_container.py run().

#include "pass_runner.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fluidx3d::experimental::wall_builder {

namespace {

/// @brief Every unit of the passes in the serial order: facets (moving directions), then cells (all directions).
class SerialUnits {
public:
    SerialUnits(const std::vector<std::vector<UnitResult>>& facets, const std::vector<std::vector<UnitResult>>& cells)
    {
        for (const std::vector<UnitResult>& task : facets) {
            for (const UnitResult& unit : task) {
                units.push_back(&unit);
            }
        }
        for (const std::vector<UnitResult>& task : cells) {
            for (const UnitResult& unit : task) {
                units.push_back(&unit);
            }
        }
    }

    std::vector<const UnitResult*> units{}; ///< Every unit, serial order.
};

} // namespace

PassRunner::PassRunner(const TerrainGeometry& geometry, const int threads)
    : geometry_{ geometry }
    , threads_{ threads }
{
    if (threads < 1) {
        throw std::invalid_argument("PassRunner: threads " + std::to_string(threads) + " (need 1 or more)");
    }
}

void PassRunner::work(const std::size_t task)
{
    const PassEngine engine{ geometry_ };
    const std::size_t facet_count{ geometry_.facets().size() };
    if (task < facet_count) {
        facet_results_[task] = engine.producer(geometry_.facets()[task]);
        return;
    }
    const std::size_t cell{ task - facet_count };
    cell_results_[cell] = engine.donor(geometry_.cells()[cell]);
}

void PassRunner::run()
{
    const std::size_t facet_count{ geometry_.facets().size() };
    const std::size_t cell_count{ geometry_.cells().size() };
    const std::size_t total{ facet_count + cell_count };
    facet_results_.assign(facet_count, {});
    cell_results_.assign(cell_count, {});
    std::atomic<std::size_t> next{ 0 };
    std::atomic<std::size_t> done{ 0 };
    std::atomic<bool> failed{ false };
    std::exception_ptr first_error{};
    std::mutex lock{};
    const auto start{ std::chrono::steady_clock::now() };
    const std::size_t step{ total / 20 > 0 ? total / 20 : 1 };
    const auto worker{ [&]() {
        for (std::size_t task{ next.fetch_add(1) }; task < total && !failed.load(); task = next.fetch_add(1)) {
            try {
                work(task);
            } catch (...) { // a thread boundary: kept and rethrown by run()
                const std::lock_guard<std::mutex> guard{ lock };
                if (!first_error) {
                    first_error = std::current_exception();
                }
                failed.store(true);
                return;
            }
            const std::size_t finished{ done.fetch_add(1) + 1 };
            if (finished % step == 0 || finished == total) {
                const double seconds{ std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() };
                const std::lock_guard<std::mutex> guard{ lock };
                std::cout << "  passes " << finished << "/" << total << " " << static_cast<int>(seconds) << " s" << std::endl;
            }
        }
    } };
    std::vector<std::thread> pool{};
    for (int t{ 0 }; t < threads_; ++t) {
        pool.emplace_back(worker);
    }
    for (std::thread& thread : pool) {
        thread.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

DiagTotals PassRunner::diag() const
{
    // the statements of builder3d.split / note_small and of run()'s max_unit_sum, in the serial unit order
    DiagTotals totals{};
    const SerialUnits serial{ facet_results_, cell_results_ };
    for (const UnitResult* unit : serial.units) {
        const double before{ totals.sum_defect };
        for (const DiagEvent& event : unit->events) {
            if (event.is_defect) {
                totals.max_defect = std::max(totals.max_defect, event.value);
                totals.sum_defect += event.value;
            } else {
                totals.small_count += 1;
                totals.small_total += event.value;
            }
        }
        totals.max_unit_sum = std::max(totals.max_unit_sum, totals.sum_defect - before);
    }
    return totals;
}

} // namespace fluidx3d::experimental::wall_builder
