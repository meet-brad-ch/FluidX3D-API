#pragma once
/// @file stencil.hpp
/// @brief The sampler stencil weights of terrain_container.py stencil_weights, on the OpenBLAS numpy uses.

#include "dense_algebra.hpp"

#include <array>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief One sorted sampler candidate with its offset from the sample point (the tuple (d2, idx, 0, sx, sy, delta)).
struct StencilCandidate {
    double d2{};                  ///< delta . delta.
    int index{};                  ///< Cell index.
    int sx{};                     ///< x image shift.
    int sy{};                     ///< y image shift.
    std::array<double, 3> delta{}; ///< Shifted centroid minus sample point.
};

/// @brief The weights of one facet.
struct StencilResult {
    bool least_squares{};                     ///< false: direct_centroid (one exact centroid, weight 1).
    std::vector<StencilCandidate> chosen{};   ///< The cells used, in order.
    std::vector<double> coefficients{};       ///< One weight per chosen cell.
    double condition{};                       ///< np.linalg.cond of the normal matrix (least squares only).
    double reproduction{};                    ///< max |coeff @ B - (1, 0, 0, 0)| (least squares only).
};

/// @brief stencil_weights(cand, Ny=1) of terrain_container.py.
class StencilWeights {
public:
    /// @brief The weights of the sorted candidates.
    /// @param candidates the visible prefix, sorted
    /// @param neighbours the stencil size (NEIGHBOURS)
    /// @throws std::runtime_error with the R3 message for an ambiguous exact centroid or a rank below 4
    /// @throws std::domain_error from the algebra (singular system, SVD failure)
    [[nodiscard]] static StencilResult compute(const std::vector<StencilCandidate>& candidates, int neighbours);
};

} // namespace fluidx3d::experimental::wall_builder
