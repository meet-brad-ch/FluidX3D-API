#pragma once
/// @file pass_runner.hpp
/// @brief Runs the producer passes of all facets, and the donor passes of the cells one window of x-columns at a time,
/// on many threads. The results keep the serial order of terrain_container.py run(), so they are the same as one
/// Python process.

#include "passes.hpp"
#include "terrain.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief builder3d.py Diag: the split-defect and small-piece diagnostics, added unit by unit in the serial order.
struct DiagTotals {
    double max_defect{};       ///< Largest split defect.
    double sum_defect{};       ///< Sum of split defects, in serial order.
    std::int64_t small_count{}; ///< Pieces with volume <= SMALL.
    double small_total{};      ///< Their total volume.
    double max_unit_sum{};     ///< Largest defect sum of one row or one donor state.

    /// @brief Adds the events of one unit (a producer row or a donor state), as builder3d.split, note_small and run()'s
    /// max_unit_sum do.
    void add_unit(std::span<const DiagEvent> events);
};

/// @brief h of one producer row in one cell: the wrapped cell and the volume (the image is not needed after the pass).
struct HCell {
    std::array<int, 3> cell{}; ///< Wrapped cell.
    double value{};            ///< Volume.
};

/// @brief W of one producer row: the receiver facet and the volume.
struct WFacet {
    FacetKey gkey{}; ///< Receiver facet.
    double value{};  ///< Volume.
};

/// @brief h_moment of one producer row in one cell.
struct HMoment {
    std::array<int, 3> cell{};     ///< Wrapped cell.
    std::array<double, 4> value{}; ///< Volume, then moment x, y, z.
};

/// @brief One producer row (facet, moving direction) that has content: where its entries lie in the facet's lists.
struct ProducerRow {
    int slot{};                 ///< Position among the moving directions.
    bool has_vi{};              ///< The row has a prism volume (it is a wall row).
    double vi{};                ///< The prism volume c . a.
    std::uint32_t h_begin{};    ///< h entries [h_begin, h_end) of ProducerFacet::h.
    std::uint32_t h_end{};
    std::uint32_t w_begin{};    ///< W entries of ProducerFacet::w.
    std::uint32_t w_end{};
    std::uint32_t m_begin{};    ///< h_moment entries of ProducerFacet::moments.
    std::uint32_t m_end{};
    std::uint32_t e_begin{};    ///< Diagnostic events of ProducerFacet::events.
    std::uint32_t e_end{};
};

/// @brief The producer results of one facet, compact: the rows with content and their entries in insertion order.
struct ProducerFacet {
    std::vector<ProducerRow> rows{};     ///< By increasing slot; the rows without any content are left out.
    std::vector<HCell> h{};
    std::vector<WFacet> w{};
    std::vector<HMoment> moments{};
    std::vector<DiagEvent> events{};
};

/// @brief One producer row as the maps read it; exists is false for the rest direction and for a row without content.
struct ProducerRowView {
    bool exists{};
    bool has_vi{};
    double vi{};
    std::span<const HCell> h{};
    std::span<const WFacet> w{};
    std::span<const HMoment> moments{};
};

/// @brief The producer passes of the whole terrain.
class PassRunner {
public:
    /// @brief Uses the geometry; it must outlive the runner.
    /// @param threads the number of worker threads, 1 or more
    /// @throws std::invalid_argument when threads is below 1
    PassRunner(const TerrainGeometry& geometry, int threads);

    /// @brief Runs every producer pass; prints progress lines.
    /// @throws the first error of any thread (the builder's message), after all threads have stopped
    void run();

    /// @brief The diagnostics of the facets, replayed in the serial order (the cells follow them).
    [[nodiscard]] DiagTotals diag() const;

    /// @brief Producer row (facet, slot): facet in the sorted facets, slot among the moving directions.
    [[nodiscard]] ProducerRowView row(std::size_t facet, int slot) const;

private:
    const TerrainGeometry& geometry_;
    int threads_{ 1 };
    std::vector<ProducerFacet> facets_{}; ///< Per canonical facet.

    /// @brief The compact form of one facet's results (one UnitResult per moving direction, in direction order).
    /// @throws std::length_error when a list does not fit its uint32 offsets
    [[nodiscard]] static ProducerFacet compact(const std::vector<UnitResult>& results);
};

/// @brief What a donor state keeps (builder3d.py direct and direct_moment of one (cell, d)).
struct DonorState {
    double volume{};                 ///< direct: the kept volume.
    std::array<double, 4> moment{};  ///< direct_moment: volume, then the moment about the cell corner.
};

/// @brief The donor results of the cells of one x-column.
struct DonorColumn {
    std::size_t first{};                          ///< Index of the column's first cell.
    std::vector<DonorState> states{};             ///< Per cell, per direction: (cell - first) * q + d.
    std::vector<std::vector<DiagEvent>> events{}; ///< Per cell, the events of its states in direction order.
    std::vector<std::uint32_t> event_end{};       ///< Per state, the end of its events in its cell's list.
};

/// @brief The donor passes of the cells, per x-column, computed on demand and freed when no slab needs them.
class DonorColumns {
public:
    /// @brief Uses the geometry; it must outlive this object.
    /// @param threads the number of worker threads, 1 or more
    /// @throws std::invalid_argument when threads is below 1
    DonorColumns(const TerrainGeometry& geometry, int threads);

    /// @brief Computes the x-columns of the list that are not held yet, all their cells on all threads.
    /// @param columns x-column indices inside the box
    /// @throws the first error of any thread (GeometryInvalid of the builder), after all threads have stopped
    void compute(const std::vector<int>& columns);

    /// @brief The results of x-column i. @throws std::logic_error when it is not computed
    [[nodiscard]] const DonorColumn& column(int i) const;

    /// @brief Frees the results of x-column i (nothing when it is not held).
    void drop(int i);

    /// @brief The bytes held now (an estimate: the vectors' capacities).
    [[nodiscard]] std::size_t bytes() const;

private:
    const TerrainGeometry& geometry_;
    int threads_{ 1 };
    std::vector<std::unique_ptr<DonorColumn>> columns_{}; ///< Per x-column, null when not held.
};

} // namespace fluidx3d::experimental::wall_builder
