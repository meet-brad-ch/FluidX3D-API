#pragma once
/// @file wall_tables.hpp
/// @brief The wall tables of terrain_container.py: arms, wall_list, sample_support_closure and gradient_map.

#include "maps.hpp"
#include "slab_stream.hpp"

#include <cstdint>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief arms(): the part centroid minus the cell's fluid centroid, per permuted G entry (the D part, direct_arm, is
/// made by the slab stream).
struct ArmTables {
    std::vector<double> prism_arm{};           ///< 3 per G entry.
    std::vector<std::int64_t> wall_cells{};    ///< Sorted unique gather donor cells.
};

/// @brief The wall list after rule #253 and the sample-support closure.
struct WallList {
    std::vector<std::int64_t> cells{};         ///< Sorted.
    std::vector<std::int64_t> added_253{};     ///< Added by rule #253, sorted.
    std::vector<std::int64_t> added_support{}; ///< Added by the sample-support closure, sorted.
    double support_arm_max{};                  ///< Largest |direct arm| of an added support cell.
};

/// @brief gradient_map(): the per-axis gradient stencils of the wall cells and its report.
struct GradientMap {
    std::vector<std::int64_t> cells{};    ///< gradient_cells.
    std::vector<std::int64_t> offsets{};  ///< gradient_offsets.
    std::vector<std::int64_t> indices{};  ///< gradient_indices.
    std::vector<double> weights{};        ///< gradient_weights.
    double reproduction_max{};
    double condition_max{};
    std::int64_t stencil_size_min{};
    std::int64_t stencil_size_max{};
    std::int64_t ties{};                  ///< stencils_with_a_distance_tie_at_the_cut.
    double smallest_gap{};                ///< smallest_relative_gap_of_squared_distance_after_the_stencil.
};

/// @brief The wall tables of one terrain.
class WallTables {
public:
    WallTables(const TerrainGeometry& geometry, const VelocityTable& velocities, const MapAssembly& assembly);

    /// @brief The G part of arms(vs, b, m, p).
    /// @throws std::runtime_error for a gather entry that is not the opposite direction of its row, or without h_moment
    [[nodiscard]] ArmTables arms(const AssembledMaps& m, const PermutedMaps& p) const;

    /// @brief wall_list and sample_support_closure, from the stream's rule #253 flags.
    /// @param wall per cell: 1 for a gather donor cell
    /// @param stream the slab stream's results (added cells, direct arms of the support cells)
    /// @param s the permuted S
    /// @param support the S stencil cells, sorted unique (the order of stream.support_arm)
    /// @throws std::runtime_error when an added support cell has a direct arm above 1e-12
    [[nodiscard]] WallList wall_list(const std::vector<char>& wall, const SlabResults& stream, const SparseMatrix& s,
                                     const std::vector<std::int64_t>& support) const;

    /// @brief gradient_map on all threads. @throws std::runtime_error with the builder's message for a bad stencil
    [[nodiscard]] GradientMap gradient(const std::vector<std::int64_t>& wall_cells, int threads) const;

private:
    static constexpr int QUERY{ 40 };
    static constexpr int OTHERS{ 11 };
    static constexpr double TIE_RELATIVE{ 1e-9 };
    static constexpr double ARM_ROUNDING{ 1e-12 };

    /// @brief The stencil of one wall cell.
    struct CellStencil {
        std::vector<std::int64_t> cells{}; ///< Own cell first, then the tied shell.
        std::vector<double> weights{};     ///< 3 rows of size cells.size(): d/dx, d/dy, d/dz.
        double reproduction{};
        double condition{};
        bool tie{};
        double gap{};
    };

    /// @brief One point of the 9 periodic images: a cell and its shift index (sa + 1) * 3 + (sb + 1).
    struct ImagePoint {
        double d2{};
        std::int64_t cell{};
        int shift{};
    };

    const TerrainGeometry& geometry_;
    const VelocityTable& velocities_;
    const MapAssembly& assembly_;
    int kmin_{};
    int kmax_{};

    [[nodiscard]] CellStencil stencil(std::int64_t a) const;
    /// @brief sample_support_closure: per cell, 1 for an S stencil cell outside the wall list; sets out.support_arm_max.
    /// @throws std::runtime_error when such a cell has a direct arm above 1e-12
    [[nodiscard]] std::vector<char> support_closure(const std::vector<char>& all, const SlabResults& stream, const SparseMatrix& s,
                                                    const std::vector<std::int64_t>& support, WallList& out) const;
    [[nodiscard]] std::vector<ImagePoint> nearest_points(std::int64_t a) const;
    void window_points(std::int64_t a, int m, std::vector<ImagePoint>& out) const;
    [[nodiscard]] std::array<double, 3> image_delta(std::int64_t cell, int shift, std::int64_t a) const;
};

} // namespace fluidx3d::experimental::wall_builder
