#pragma once
/// @file sampler.hpp
/// @brief The candidate search of the facet-state sampler: builder3d.py build_sampler with SamplerCandidates
/// (exact window, visible prefix) and the visibility test segment_free / surface_height, ported in Python's operation
/// order. The stencil weights (a 4 x 4 least-squares solve, numpy and LAPACK) stay in Python; this part gives it the
/// candidates and a visibility flag per candidate for its R3 check.

#include "terrain.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief One kept candidate: a cell centroid at an x, y image.
struct SamplerCandidate {
    double d2{};      ///< Squared distance from the sample point (numpy delta @ delta).
    int index{};      ///< Cell index in the sorted cell list.
    int sx{};         ///< x image shift (-Lx, 0 or Lx).
    int sy{};         ///< y image shift.
    bool visible{};   ///< segment_free from the sample point to the shifted centroid.
};

/// @brief The sampler row of one facet.
struct FacetSample {
    Vec3 point{};                            ///< x_s = area centroid + 0.5 unit normal.
    std::vector<SamplerCandidate> kept{};    ///< The visible prefix, in the sorted order.
    bool least_squares{};                    ///< The stencil branch (false: direct_centroid).
    std::vector<int> cells{};                ///< The stencil cells (S columns), in order.
    std::vector<double> weights{};           ///< The S weights, one per stencil cell.
    double condition{};                      ///< cond of the normal matrix (least squares only).
    double reproduction{};                   ///< Reproduction error (least squares only).
    std::string error{};                     ///< The builder's rejection (R2 or R3), empty when none.
};

/// @brief The counters of SamplerCandidates (logged by terrain_container.py).
struct SamplerCounters {
    std::int64_t window{};          ///< Facets served by a window.
    std::int64_t full{};            ///< Facets served by the full list.
    std::int64_t largest_m{};       ///< Largest accepted window half width.
    std::int64_t hidden_skipped{};  ///< Hidden candidates skipped.
    bool contained{};               ///< Every centroid lies in its unit cell.
};

/// @brief The terrain surface of builder3d.py surface_height and segment_free.
class TerrainSurface {
public:
    /// @brief Tabulates the binary64 triangles and planes of the unwrapped columns -Lx-2 .. 2Lx+1 (y likewise).
    explicit TerrainSurface(const TerrainGeometry& geometry);

    /// @brief The height of the terrain at (x, y) (surface_height).
    /// @throws std::runtime_error when no triangle contains the point or the column is outside the table
    [[nodiscard]] double height(double x, double y) const;

    /// @brief True when z - h >= VIS_MARGIN along p -> q and both ends lie below the lid (segment_free).
    [[nodiscard]] bool segment_free(const Vec3& p, const Vec3& q) const;

private:
    static constexpr double VIS_MARGIN{ 1e-12 }; ///< builder3d.VIS_MARGIN.

    int lx_{};
    int ly_{};
    int z_top_{};
    int i0_{};
    int j0_{};
    int ni_{};
    int nj_{};
    std::vector<std::array<ColumnTrianglePlane, 2>> columns_{};

    /// @brief point_in_triangle of builder3d.py (on the triangle's footprint).
    [[nodiscard]] static bool contains(const ColumnTrianglePlane& t, double x, double y);
    /// @brief Adds a value to the parameter set unless an equal value is in it (Python's set).
    static void add_unique(std::vector<double>& values, double value);
};

/// @brief The sampler candidates of every facet, on many threads.
class SamplerSearch {
public:
    /// @brief Uses the geometry; it must outlive the search.
    /// @param neighbours the stencil size (terrain_container.NEIGHBOURS)
    SamplerSearch(const TerrainGeometry& geometry, int neighbours);

    /// @brief Searches every facet (the canonical facets in their sorted order).
    void run(int threads);

    /// @brief The results, one per canonical facet.
    [[nodiscard]] const std::vector<FacetSample>& samples() const { return samples_; }

    /// @brief The summed counters.
    [[nodiscard]] SamplerCounters counters() const;

    /// @brief Throws the first rejection in facet order (R2, R3 or a stencil error), if any.
    /// @throws std::runtime_error with the builder's message
    void check() const;

private:
    /// @brief The candidates and the counters of one facet.
    struct FacetCounters {
        std::int64_t window{};
        std::int64_t full{};
        std::int64_t largest_m{};
        std::int64_t hidden_skipped{};
    };

    const TerrainGeometry& geometry_;
    TerrainSurface surface_;
    int neighbours_{};
    bool contained_{};
    int limit_{};
    int kmin_{};
    int nk_{};
    std::vector<FacetSample> samples_{};
    std::vector<FacetCounters> facet_counters_{};

    void search(std::size_t facet, FacetSample& sample, FacetCounters& counters) const;
    /// @brief True when every cell centroid lies in its unit cell (then the index windows of the search are exact).
    [[nodiscard]] bool centroids_contained() const;
    /// @brief cell <= value <= cell + 1.
    [[nodiscard]] static bool inside(double value, int cell);
    /// @brief True when every unwrapped index in [first, last] lies in one of the images -n, 0, n of a periodic axis of n.
    [[nodiscard]] static bool three_images(int first, int last, int n);
    [[nodiscard]] static Vec3 area_centroid(const Facet& f);
    [[nodiscard]] SamplerCandidate candidate(const Vec3& xs, int index, int sx, int sy) const;
    [[nodiscard]] bool window(const Vec3& xs, int m, std::vector<SamplerCandidate>& out) const;
    void full_list(const Vec3& xs, std::vector<SamplerCandidate>& out) const;
    [[nodiscard]] std::vector<SamplerCandidate> visible_prefix(const Vec3& xs, const std::vector<SamplerCandidate>& sorted,
                                                               FacetCounters& counters) const;
    [[nodiscard]] Vec3 shifted_centroid(const SamplerCandidate& c) const;
    /// @brief The stencil weights of the kept candidates and the R3 visibility check of the chosen cells (build_sampler).
    void weights(const Facet& f, FacetSample& sample) const;
};

} // namespace fluidx3d::experimental::wall_builder
