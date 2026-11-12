#pragma once
/// @file stl_heights.hpp
/// @brief The node heights of the builder from a terrain STL (research/wall_model/v2/hill_demo/e2/make_hill_heights.py,
/// generalised to any cell size), with a cache: the same STL bytes and recipe give the cached CSV without a new sampling.
///
/// The steps, in the Python operation order (numpy 2.2.6, scipy 1.15.3):
/// 1. A 4 m raster of vertical lines offset from the mesh vertices; per raster point the highest triangle hit (zmax).
/// 2. Bilinear interpolation of zmax (scipy RegularGridInterpolator, linear, extrapolated) at the lattice nodes:
///    lattice x = STL +y (both footprint edges, n + 1 nodes), lattice y = x_hi - STL x (n periodic nodes).
/// 3. h_cells = 1 + h_m / cell, rounded to 2^-10 cell (half to even), at least 1.
/// 4. Flat apron columns (height exactly 1) added at both x ends: Lx = apron + n + apron, Ly = n.

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief How an STL becomes node heights.
struct HeightsRecipe {
    double cell_m{};                          ///< Cell size in metres; the footprint side must be a whole number of cells.
    int apron{ 8 };                           ///< Flat columns added at each x end, 1 or more.
    double quantum{ 1.0 / 1024.0 };           ///< Height rounding in cells (2^-10).
    double raster_m{ 4.0 };                   ///< Raster spacing in metres.
};

/// @brief The triangles of an STL, vertices in float64 (from float32).
struct StlMesh {
    std::vector<std::array<double, 9>> triangles{}; ///< x0 y0 z0 x1 y1 z1 x2 y2 z2.
};

/// @brief The heights CSV of a terrain STL, made once per STL and recipe.
class StlHeights {
public:
    /// @brief The cached CSV of the STL and recipe; samples the STL and writes the CSV (and its .json) when it is missing.
    /// @param cache_dir the cache folder (created when missing)
    /// @param threads the threads of the raster, 1 or more
    /// @return the CSV path
    /// @throws std::runtime_error when the STL cannot be read, its footprint is not a square of whole cells, a node has no
    /// height, or a file cannot be written
    [[nodiscard]] static std::filesystem::path cached(const std::filesystem::path& stl, const HeightsRecipe& recipe,
                                                      const std::filesystem::path& cache_dir, int threads);

    /// @brief The CSV text (Ly lines of Lx node heights, Python repr) of the STL and recipe.
    /// @throws std::runtime_error as cached()
    [[nodiscard]] static std::string csv_text(const StlMesh& mesh, const HeightsRecipe& recipe, int threads);

    /// @brief Reads a binary or ASCII STL (binary when the size is 84 + 50 n with n from the header).
    /// @throws std::runtime_error when it cannot be read or has no triangle
    [[nodiscard]] static StlMesh read(const std::filesystem::path& stl);

private:
    /// @brief The raster of builder3d's E1 inventory: points and the highest hit per point.
    struct Raster {
        std::vector<double> x{};    ///< Raster x (numpy arange).
        std::vector<double> y{};    ///< Raster y.
        std::vector<double> zmax{}; ///< Per (y, x), row-major by y; -inf without a hit.
    };

    [[nodiscard]] static std::vector<double> arange(double start, double stop, double step);
    [[nodiscard]] static Raster raster(const StlMesh& mesh, double spacing, int threads);
    static void rasterize(const std::array<double, 9>& t, const Raster& r, double spacing, std::vector<double>& zmax);
    [[nodiscard]] static std::size_t interval(const std::vector<double>& grid, double value);
    [[nodiscard]] static double interpolate(const Raster& r, double y, double x);
    [[nodiscard]] static std::string cache_key(const std::string& stl_sha, const HeightsRecipe& recipe);
};

} // namespace fluidx3d::experimental::wall_builder
