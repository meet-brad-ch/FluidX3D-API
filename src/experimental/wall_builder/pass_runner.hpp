#pragma once
/// @file pass_runner.hpp
/// @brief Runs the passes of all facets and cells on many threads and merges them in the serial order of
/// terrain_container.py run(), so the result is the same as one Python process.

#include "passes.hpp"
#include "terrain.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief builder3d.py Diag: the split-defect and small-piece diagnostics.
struct DiagTotals {
    double max_defect{};       ///< Largest split defect.
    double sum_defect{};       ///< Sum of split defects, in serial order.
    std::int64_t small_count{}; ///< Pieces with volume <= SMALL.
    double small_total{};      ///< Their total volume.
    double max_unit_sum{};     ///< Largest defect sum of one row or one donor state.
};

/// @brief The passes of the whole terrain.
class PassRunner {
public:
    /// @brief Uses the geometry; it must outlive the runner.
    /// @param threads the number of worker threads, 1 or more
    PassRunner(const TerrainGeometry& geometry, int threads);

    /// @brief Runs every producer and donor pass; prints progress lines.
    /// @throws the first error of any thread (the builder's message), after all threads have stopped
    void run();

    /// @brief The diagnostics, replayed in the serial order.
    [[nodiscard]] DiagTotals diag() const;

    /// @brief Per canonical facet (sorted), one result per moving direction in direction order.
    [[nodiscard]] const std::vector<std::vector<UnitResult>>& facet_results() const { return facet_results_; }

    /// @brief Per cell (sorted), one result per direction.
    [[nodiscard]] const std::vector<std::vector<UnitResult>>& cell_results() const { return cell_results_; }

private:
    const TerrainGeometry& geometry_;
    int threads_{ 1 };
    std::vector<std::vector<UnitResult>> facet_results_{}; ///< Per canonical facet, its moving directions.
    std::vector<std::vector<UnitResult>> cell_results_{};  ///< Per cell, all directions.

    void work(std::size_t task) ;
};

} // namespace fluidx3d::experimental::wall_builder
