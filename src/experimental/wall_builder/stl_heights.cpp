/// @file stl_heights.cpp
/// @brief The node heights of the builder from a terrain STL, ported from make_hill_heights.py (E1 inventory raster,
/// scipy RegularGridInterpolator) in Python's operation order.
/// Built with FLUIDX3D_PRECISE_FLOAT: every value keeps numpy's and scipy's operation order (no fused multiply-add).

#include "stl_heights.hpp"

#include "parallel_for.hpp"
#include "python_text.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace fluidx3d::experimental::wall_builder {

namespace {

/// @brief Reads a whole file. @throws std::runtime_error when it cannot be read
std::vector<char> file_bytes(const std::filesystem::path& path)
{
    std::ifstream in{ path, std::ios::binary };
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    const auto size{ static_cast<std::size_t>(std::filesystem::file_size(path)) };
    std::vector<char> bytes(size, 0);
    in.read(bytes.data(), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return bytes;
}

/// @brief Writes a file through a temporary name, so a stopped run leaves no broken cache entry.
/// @throws std::runtime_error when it cannot be written or renamed
void write_atomic(const std::filesystem::path& path, const std::string& text)
{
    const std::filesystem::path part{ path.string() + ".part" };
    {
        std::ofstream out{ part, std::ios::binary | std::ios::trunc };
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out) {
            throw std::runtime_error("cannot write " + part.string());
        }
    }
    std::filesystem::rename(part, path);
}

} // namespace

StlMesh StlHeights::read(const std::filesystem::path& stl)
{
    const std::vector<char> data{ file_bytes(stl) };
    StlMesh mesh{};
    std::uint32_t count{ 0 };
    if (data.size() >= 84) {
        std::memcpy(&count, data.data() + 80, 4);
    }
    const bool binary{ data.size() >= 84 && data.size() == 84 + 50 * static_cast<std::size_t>(count) };
    if (binary) {
        for (std::size_t t{ 0 }; t < count; ++t) {
            const char* const record{ data.data() + 84 + 50 * t + 12 }; // skip the normal
            std::array<float, 9> v{};
            std::memcpy(v.data(), record, 36);
            std::array<double, 9> tri{};
            for (std::size_t k{ 0 }; k < 9; ++k) {
                tri[k] = static_cast<double>(v[k]);
            }
            mesh.triangles.push_back(tri);
        }
    } else {
        // ASCII: every "vertex x y z" line, three per triangle
        std::istringstream text{ std::string{ data.begin(), data.end() } };
        std::string line{};
        std::vector<double> values{};
        while (std::getline(text, line)) {
            std::istringstream words{ line };
            std::string word{};
            words >> word;
            if (word != "vertex") {
                continue;
            }
            std::array<double, 3> p{};
            words >> p[0] >> p[1] >> p[2];
            if (!words) {
                throw std::runtime_error("bad vertex line in " + stl.string() + ": " + line);
            }
            values.insert(values.end(), p.begin(), p.end());
        }
        if (values.size() % 9 != 0) {
            throw std::runtime_error(stl.string() + ": the vertex count is not a multiple of 3");
        }
        for (std::size_t t{ 0 }; t < values.size(); t += 9) {
            std::array<double, 9> tri{};
            std::copy(values.begin() + static_cast<std::ptrdiff_t>(t), values.begin() + static_cast<std::ptrdiff_t>(t + 9), tri.begin());
            mesh.triangles.push_back(tri);
        }
    }
    if (mesh.triangles.empty()) {
        throw std::runtime_error(stl.string() + " has no triangle");
    }
    return mesh;
}

std::vector<double> StlHeights::arange(const double start, const double stop, const double step)
{
    // numpy.arange: length ceil((stop - start) / step); element i = start + i * delta, delta = (start + step) - start
    const double span{ (stop - start) / step };
    const auto length{ static_cast<std::size_t>(std::max(0.0, std::ceil(span))) };
    std::vector<double> out(length, 0.0);
    if (length == 0) {
        return out;
    }
    out[0] = start;
    const double second{ start + step };
    const double delta{ second - start };
    for (std::size_t i{ 1 }; i < length; ++i) {
        out[i] = start + static_cast<double>(i) * delta;
    }
    return out;
}

void StlHeights::rasterize(const std::array<double, 9>& t, const Raster& r, const double spacing, std::vector<double>& zmax)
{
    const double x0{ r.x[0] };
    const double y0{ r.y[0] };
    const double tx_min{ std::min({ t[0], t[3], t[6] }) };
    const double tx_max{ std::max({ t[0], t[3], t[6] }) };
    const double ty_min{ std::min({ t[1], t[4], t[7] }) };
    const double ty_max{ std::max({ t[1], t[4], t[7] }) };
    const long long ix0{ std::max(0LL, static_cast<long long>(std::ceil((tx_min - x0) / spacing))) };
    const long long ix1{ std::min(static_cast<long long>(r.x.size()) - 1, static_cast<long long>(std::floor((tx_max - x0) / spacing))) };
    const long long iy0{ std::max(0LL, static_cast<long long>(std::ceil((ty_min - y0) / spacing))) };
    const long long iy1{ std::min(static_cast<long long>(r.y.size()) - 1, static_cast<long long>(std::floor((ty_max - y0) / spacing))) };
    if (ix1 < ix0 || iy1 < iy0) {
        return;
    }
    const double ax{ t[0] };
    const double ay{ t[1] };
    const double az{ t[2] };
    const double bx{ t[3] };
    const double by{ t[4] };
    const double bz{ t[5] };
    const double cx{ t[6] };
    const double cy{ t[7] };
    const double cz{ t[8] };
    const double det{ (by - cy) * (ax - cx) + (cx - bx) * (ay - cy) };
    if (det == 0.0) {
        return;
    }
    const std::size_t width{ r.x.size() };
    for (long long iy{ iy0 }; iy <= iy1; ++iy) {
        const double py{ r.y[static_cast<std::size_t>(iy)] };
        for (long long ix{ ix0 }; ix <= ix1; ++ix) {
            const double px{ r.x[static_cast<std::size_t>(ix)] };
            const double l1{ ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det };
            const double l2{ ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det };
            const double l3{ 1.0 - l1 - l2 };
            const bool inside{ l1 >= 0 && l2 >= 0 && l3 >= 0 };
            if (!inside) {
                continue;
            }
            const double z{ l1 * az + l2 * bz + l3 * cz };
            double& top{ zmax[static_cast<std::size_t>(iy) * width + static_cast<std::size_t>(ix)] };
            top = std::max(top, z);
        }
    }
}

StlHeights::Raster StlHeights::raster(const StlMesh& mesh, const double spacing, const int threads)
{
    std::array<double, 3> lo{ std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity() };
    std::array<double, 3> hi{ -lo[0], -lo[1], -lo[2] };
    for (const std::array<double, 9>& t : mesh.triangles) {
        for (std::size_t k{ 0 }; k < 9; ++k) {
            lo[k % 3] = std::min(lo[k % 3], t[k]);
            hi[k % 3] = std::max(hi[k % 3], t[k]);
        }
    }
    Raster r{};
    r.x = arange(lo[0] + 0.37 * spacing, hi[0], spacing);
    r.y = arange(lo[1] + 0.41 * spacing, hi[1], spacing);
    const double none{ -std::numeric_limits<double>::infinity() };
    r.zmax.assign(r.x.size() * r.y.size(), none);
    // the highest hit does not depend on the triangle order: each thread keeps its own maxima, then they are merged
    const auto parts{ static_cast<std::size_t>(threads) };
    std::vector<std::vector<double>> partial(parts);
    ParallelFor::run(parts, threads, [&](const std::size_t part) {
        std::vector<double> zmax(r.zmax.size(), none);
        for (std::size_t t{ part }; t < mesh.triangles.size(); t += parts) {
            rasterize(mesh.triangles[t], r, spacing, zmax);
        }
        partial[part] = std::move(zmax);
    });
    for (const std::vector<double>& zmax : partial) {
        for (std::size_t n{ 0 }; n < zmax.size(); ++n) {
            r.zmax[n] = std::max(r.zmax[n], zmax[n]);
        }
    }
    return r;
}

std::size_t StlHeights::interval(const std::vector<double>& grid, const double value)
{
    // scipy find_interval_ascending with extrapolation: grid[i] <= value < grid[i + 1], clamped to the end intervals
    const std::size_t last{ grid.size() - 2 };
    if (value < grid[0]) {
        return 0;
    }
    if (value >= grid.back()) {
        return last;
    }
    const auto above{ std::upper_bound(grid.begin(), grid.end(), value) };
    return static_cast<std::size_t>(above - grid.begin()) - 1u;
}

double StlHeights::interpolate(const Raster& r, const double y, const double x)
{
    // scipy evaluate_linear_2d: values[i0, i1] (1 - y0)(1 - y1) + values[i0, i1 + 1] (1 - y0) y1 + values[i0 + 1, i1] y0 (1 - y1)
    // + values[i0 + 1, i1 + 1] y0 y1, with y0, y1 the normalised distances (value - grid[i]) / (grid[i + 1] - grid[i])
    const std::size_t i0{ interval(r.y, y) };
    const std::size_t i1{ interval(r.x, x) };
    const double t0{ (y - r.y[i0]) / (r.y[i0 + 1] - r.y[i0]) };
    const double t1{ (x - r.x[i1]) / (r.x[i1 + 1] - r.x[i1]) };
    const std::size_t width{ r.x.size() };
    const double v00{ r.zmax[i0 * width + i1] };
    const double v01{ r.zmax[i0 * width + i1 + 1] };
    const double v10{ r.zmax[(i0 + 1) * width + i1] };
    const double v11{ r.zmax[(i0 + 1) * width + i1 + 1] };
    const double value{ v00 * (1 - t0) * (1 - t1) + v01 * (1 - t0) * t1 + v10 * t0 * (1 - t1) + v11 * t0 * t1 };
    if (!std::isfinite(value)) {
        throw std::runtime_error("the STL has no height near node (" + PythonFloat::repr(x) + ", " + PythonFloat::repr(y) + ") m");
    }
    return value;
}

std::string StlHeights::csv_text(const StlMesh& mesh, const HeightsRecipe& recipe, const int threads)
{
    if (recipe.apron < 1 || !(recipe.cell_m > 0.0)) {
        throw std::runtime_error("StlHeights: apron " + std::to_string(recipe.apron) + " (1 or more) and cell " + PythonFloat::repr(recipe.cell_m) + " m (above 0)");
    }
    const Raster r{ raster(mesh, recipe.raster_m, threads) };
    double lo_x{ std::numeric_limits<double>::infinity() };
    double lo_y{ lo_x };
    double hi_x{ -lo_x };
    double hi_y{ -lo_x };
    for (const std::array<double, 9>& t : mesh.triangles) {
        for (std::size_t k{ 0 }; k < 9; k += 3) {
            lo_x = std::min(lo_x, t[k]);
            hi_x = std::max(hi_x, t[k]);
            lo_y = std::min(lo_y, t[k + 1]);
            hi_y = std::max(hi_y, t[k + 1]);
        }
    }
    // the footprint side in whole metres (the STL stores float32 coordinates): make_hill_heights.py declares 4000.0
    const double side{ std::round(hi_x - lo_x) };
    const double cells_wide{ side / recipe.cell_m };
    const auto n{ static_cast<int>(std::lround(cells_wide)) };
    const bool square{ std::fabs((hi_x - lo_x) - side) <= 1e-3 && std::fabs((hi_y - lo_y) - side) <= 1e-3 };
    if (!square || static_cast<double>(n) * recipe.cell_m != side || n < 2) {
        throw std::runtime_error("the STL footprint " + PythonFloat::repr(hi_x - lo_x) + " x " + PythonFloat::repr(hi_y - lo_y)
                                 + " m is not a square of whole " + PythonFloat::repr(recipe.cell_m) + " m cells");
    }
    const double dx{ side / static_cast<double>(n) };
    const int lx{ recipe.apron + n + recipe.apron };
    const int ly{ n };
    // grid[i, j]: lattice x node i (STL y = lo_y + dx * (i - apron)), lattice y node j (STL x = hi_x - dx * j)
    std::vector<double> grid(static_cast<std::size_t>(lx) * static_cast<std::size_t>(ly), 1.0);
    for (int a{ 0 }; a <= n; ++a) {
        const double stl_y{ lo_y + dx * static_cast<double>(a) };
        for (int j{ 0 }; j < ly; ++j) {
            const double stl_x{ hi_x - dx * static_cast<double>(j) };
            const double h_m{ interpolate(r, stl_y, stl_x) };
            const double raw{ 1.0 + h_m / dx };
            const double q{ std::nearbyint(raw / recipe.quantum) * recipe.quantum }; // np.round: half to even
            const double hill{ std::max(q, 1.0) };
            grid[static_cast<std::size_t>(recipe.apron + a) * static_cast<std::size_t>(ly) + static_cast<std::size_t>(j)] = hill;
        }
    }
    std::string text{};
    for (int j{ 0 }; j < ly; ++j) {
        for (int i{ 0 }; i < lx; ++i) {
            if (i > 0) {
                text += ",";
            }
            text += PythonFloat::repr(grid[static_cast<std::size_t>(i) * static_cast<std::size_t>(ly) + static_cast<std::size_t>(j)]);
        }
        text += "\n";
    }
    return text;
}

std::string StlHeights::cache_key(const std::string& stl_sha, const HeightsRecipe& recipe)
{
    // every input of the CSV: the STL bytes, the recipe and the method version
    const std::string recipe_text{ "stl_heights v1; stl sha256 " + stl_sha + "; cell_m " + PythonFloat::repr(recipe.cell_m) + "; apron "
                                   + std::to_string(recipe.apron) + "; quantum " + PythonFloat::repr(recipe.quantum) + "; raster_m "
                                   + PythonFloat::repr(recipe.raster_m) + "; lattice x = STL +y, lattice y = x_hi - STL x" };
    return Sha256::of(recipe_text);
}

std::filesystem::path StlHeights::cached(const std::filesystem::path& stl, const HeightsRecipe& recipe, const std::filesystem::path& cache_dir,
                                         const int threads)
{
    const std::string stl_sha{ Sha256::of(file_bytes(stl)) };
    const std::string key{ cache_key(stl_sha, recipe) };
    const std::string stem{ stl.stem().string() + "_cell" + PythonFloat::repr(recipe.cell_m) + "m_apron" + std::to_string(recipe.apron) + "_" + key.substr(0, 16) };
    const std::filesystem::path csv{ cache_dir / (stem + ".csv") };
    if (std::filesystem::exists(csv)) {
        std::cout << "heights: cached " << csv.string() << std::endl;
        return csv;
    }
    std::filesystem::create_directories(cache_dir);
    const StlMesh mesh{ read(stl) };
    const std::string text{ csv_text(mesh, recipe, threads) };
    const std::string record{ "{\n \"tool\": \"volumetric_wall_builder --stl (stl_heights v1)\",\n \"stl\": \"" + stl.generic_string() + "\",\n \"stl_sha256\": \""
                              + stl_sha + "\",\n \"cell_m\": " + PythonFloat::repr(recipe.cell_m) + ",\n \"apron\": " + std::to_string(recipe.apron)
                              + ",\n \"quantum\": " + PythonFloat::repr(recipe.quantum) + ",\n \"raster_m\": " + PythonFloat::repr(recipe.raster_m)
                              + ",\n \"key\": \"" + key + "\",\n \"csv_sha256\": \"" + Sha256::of(text) + "\"\n}\n" };
    write_atomic(cache_dir / (stem + ".json"), record);
    write_atomic(csv, text);
    std::cout << "heights: sampled " << csv.string() << std::endl;
    return csv;
}

} // namespace fluidx3d::experimental::wall_builder
