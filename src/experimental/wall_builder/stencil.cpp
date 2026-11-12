/// @file stencil.cpp
/// @brief The sampler stencil weights of terrain_container.py stencil_weights, on the OpenBLAS numpy uses.
/// Built with FLUIDX3D_PRECISE_FLOAT: the elementwise operations below round as numpy's do.

#include "stencil.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fluidx3d::experimental::wall_builder {

StencilResult StencilWeights::compute(const std::vector<StencilCandidate>& candidates, const int neighbours)
{
    // exact = {(z[1], z[2] % Ny): z for z in cand if z[0] == 0.0}: one entry per cell, the last candidate wins
    std::vector<int> exact_cells{};
    std::vector<StencilCandidate> exact_last{};
    for (const StencilCandidate& z : candidates) {
        if (z.d2 != 0.0) {
            continue;
        }
        const auto found{ std::find(exact_cells.begin(), exact_cells.end(), z.index) };
        if (found == exact_cells.end()) {
            exact_cells.push_back(z.index);
            exact_last.push_back(z);
        } else {
            exact_last[static_cast<std::size_t>(found - exact_cells.begin())] = z;
        }
    }
    if (exact_cells.size() > 1) {
        throw std::runtime_error("R3: sample coincides with " + std::to_string(exact_cells.size()) + " distinct cell centroids (ambiguous)");
    }
    StencilResult result{};
    if (exact_cells.size() == 1) {
        result.chosen = { exact_last[0] };
        result.coefficients = { 1.0 };
        return result;
    }
    const std::size_t count{ std::min(candidates.size(), static_cast<std::size_t>(neighbours)) };
    result.least_squares = true;
    result.chosen.assign(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(count));
    const int n{ static_cast<int>(count) };
    Matrix basis{ n, 4, std::vector<double>(4 * count, 0.0) };   // Bm = c_[ones, delta]
    Matrix weighted{ n, 4, std::vector<double>(4 * count, 0.0) }; // w[:, None] * Bm
    Matrix rooted{ n, 4, std::vector<double>(4 * count, 0.0) };   // sqrt(w)[:, None] * Bm
    for (std::size_t r{ 0 }; r < count; ++r) {
        const StencilCandidate& z{ result.chosen[r] };
        const double w{ 1 / z.d2 };
        const double root{ std::sqrt(w) };
        const std::array<double, 4> row{ 1.0, z.delta[0], z.delta[1], z.delta[2] };
        for (std::size_t c{ 0 }; c < 4; ++c) {
            basis.data[4 * r + c] = row[c];
            weighted.data[4 * r + c] = w * row[c];
            rooted.data[4 * r + c] = root * row[c];
        }
    }
    const Matrix normal{ DenseAlgebra::transpose_times(basis, weighted) };
    if (DenseAlgebra::rank(rooted) < 4) {
        throw std::runtime_error("R3: stencil rank < 4");
    }
    const std::vector<double> unit{ 1.0, 0.0, 0.0, 0.0 };
    const std::vector<double> solution{ DenseAlgebra::solve(normal, unit) };
    result.coefficients = DenseAlgebra::times_vector(weighted, solution);
    result.condition = DenseAlgebra::condition(normal);
    const std::vector<double> reproduced{ DenseAlgebra::vector_times(result.coefficients, basis) };
    double worst{ 0.0 };
    for (std::size_t c{ 0 }; c < 4; ++c) {
        const double error{ std::fabs(reproduced[c] - unit[c]) };
        worst = c == 0 ? error : std::max(worst, error);
    }
    result.reproduction = worst;
    return result;
}

} // namespace fluidx3d::experimental::wall_builder
