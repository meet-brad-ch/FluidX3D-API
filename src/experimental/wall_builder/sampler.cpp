/// @file sampler.cpp
/// @brief The candidate search and stencil weights of the facet-state sampler, ported from builder3d.py in Python's operation
/// order. Built with FLUIDX3D_PRECISE_FLOAT: the operation order below is the order of the Python code, and must stay so.

#include "sampler.hpp"

#include "stencil.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fluidx3d::experimental::wall_builder {

namespace {

/// @brief Python's tuple order (d2, cell index, x shift, y shift) of SamplerCandidates._sort.
bool candidate_before(const SamplerCandidate& a, const SamplerCandidate& b)
{
    if (a.d2 != b.d2) {
        return a.d2 < b.d2;
    }
    if (a.index != b.index) {
        return a.index < b.index;
    }
    if (a.sx != b.sx) {
        return a.sx < b.sx;
    }
    return a.sy < b.sy;
}

std::string point_text(const Vec3& p)
{
    std::ostringstream text{};
    text.precision(17);
    text << "[" << p.x << ", " << p.y << ", " << p.z << "]";
    return text.str();
}

} // namespace

// ------------------------------------------------------------------------------------------------ TerrainSurface

TerrainSurface::TerrainSurface(const TerrainGeometry& geometry)
    : lx_{ geometry.record().lx }
    , ly_{ geometry.record().ly }
    , z_top_{ geometry.record().z_top }
    , i0_{ -geometry.record().lx - 2 }
    , j0_{ -geometry.record().ly - 2 }
    , ni_{ 3 * geometry.record().lx + 4 }
    , nj_{ 3 * geometry.record().ly + 4 }
{
    columns_.resize(static_cast<std::size_t>(ni_) * static_cast<std::size_t>(nj_));
    for (int a{ 0 }; a < ni_; ++a) {
        for (int b{ 0 }; b < nj_; ++b) {
            columns_[static_cast<std::size_t>(a) * static_cast<std::size_t>(nj_) + static_cast<std::size_t>(b)] = geometry.column_planes(a + i0_, b + j0_);
        }
    }
}

bool TerrainSurface::contains(const ColumnTrianglePlane& t, const double x, const double y)
{
    const std::array<double, 6>& f{ t.foot };
    // s(a, b, c) = (b0 - a0) (c1 - a1) - (b1 - a1) (c0 - a0)
    const double d1{ (f[2] - f[0]) * (y - f[1]) - (f[3] - f[1]) * (x - f[0]) };
    const double d2{ (f[4] - f[2]) * (y - f[3]) - (f[5] - f[3]) * (x - f[2]) };
    const double d3{ (f[0] - f[4]) * (y - f[5]) - (f[1] - f[5]) * (x - f[4]) };
    const bool negative{ d1 < 0 || d2 < 0 || d3 < 0 };
    const bool positive{ d1 > 0 || d2 > 0 || d3 > 0 };
    return !(negative && positive);
}

double TerrainSurface::height(const double x, const double y) const
{
    const auto i{ static_cast<int>(std::floor(x)) };
    const auto j{ static_cast<int>(std::floor(y)) };
    const int a{ i - i0_ };
    const int b{ j - j0_ };
    if (a < 0 || a >= ni_ || b < 0 || b >= nj_) {
        throw std::runtime_error("surface_height: column (" + std::to_string(i) + ", " + std::to_string(j) + ") is outside the table");
    }
    const std::array<ColumnTrianglePlane, 2>& column{ columns_[static_cast<std::size_t>(a) * static_cast<std::size_t>(nj_) + static_cast<std::size_t>(b)] };
    for (const ColumnTrianglePlane& t : column) {
        if (contains(t, x, y)) {
            return t.a * x + t.b * y + t.g;
        }
    }
    throw std::runtime_error("no triangle contains (" + std::to_string(x) + ", " + std::to_string(y) + ")");
}

void TerrainSurface::add_unique(std::vector<double>& values, const double value)
{
    for (const double v : values) {
        if (v == value) {
            return;
        }
    }
    values.push_back(value);
}

bool TerrainSurface::segment_free(const Vec3& p, const Vec3& q) const
{
    std::vector<double> ts{ 0.0, 1.0 };
    const double dx{ q.x - p.x };
    const double dy{ q.y - p.y };
    const double lo_x{ std::min(p.x, q.x) };
    const double hi_x{ std::max(p.x, q.x) };
    const double lo_y{ std::min(p.y, q.y) };
    const double hi_y{ std::max(p.y, q.y) };
    if (dx != 0) {
        for (auto n{ static_cast<int>(std::floor(lo_x)) }; n <= static_cast<int>(std::ceil(hi_x)); ++n) {
            add_unique(ts, (static_cast<double>(n) - p.x) / dx);
        }
    }
    if (dy != 0) {
        for (auto m{ static_cast<int>(std::floor(lo_y)) }; m <= static_cast<int>(std::ceil(hi_y)); ++m) {
            add_unique(ts, (static_cast<double>(m) - p.y) / dy);
        }
    }
    if (dx + dy != 0) {
        for (auto s{ static_cast<int>(std::floor(lo_x + lo_y)) - 1 }; s < static_cast<int>(std::ceil(hi_x + hi_y)) + 2; ++s) {
            add_unique(ts, (static_cast<double>(s) - p.x - p.y) / (dx + dy));
        }
    }
    if (dx - dy != 0) {
        for (auto s{ static_cast<int>(std::floor(lo_x - hi_y)) - 1 }; s < static_cast<int>(std::ceil(hi_x - lo_y)) + 2; ++s) {
            add_unique(ts, (static_cast<double>(s) - p.x + p.y) / (dx - dy));
        }
    }
    // the result does not depend on the order; the sorted order keeps the first failing point of Python
    std::sort(ts.begin(), ts.end());
    for (const double t : ts) {
        if (!(0.0 <= t && t <= 1.0)) {
            continue;
        }
        const double x{ p.x + t * (q.x - p.x) };
        const double y{ p.y + t * (q.y - p.y) };
        const double z{ p.z + t * (q.z - p.z) };
        if (z - height(x, y) < VIS_MARGIN || z > static_cast<double>(z_top_)) {
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ SamplerSearch

SamplerSearch::SamplerSearch(const TerrainGeometry& geometry, const int neighbours)
    : geometry_{ geometry }
    , surface_{ geometry }
    , neighbours_{ neighbours }
{
    const TerrainRecord& record{ geometry.record() };
    const std::vector<TerrainCell>& cells{ geometry.cells() };
    int kmax{ std::numeric_limits<int>::min() };
    kmin_ = std::numeric_limits<int>::max();
    contained_ = true;
    for (const TerrainCell& cell : cells) {
        kmin_ = std::min(kmin_, cell.ijk[2]);
        kmax = std::max(kmax, cell.ijk[2]);
        const std::array<double, 3> ce{ cell.centroid.x, cell.centroid.y, cell.centroid.z };
        for (std::size_t a{ 0 }; a < 3; ++a) {
            const auto k{ static_cast<double>(cell.ijk[a]) };
            contained_ = contained_ && k <= ce[a] && ce[a] <= k + 1;
        }
    }
    const int layers{ cells.empty() ? 1 : kmax - kmin_ + 1 };
    limit_ = std::max({ record.lx, record.ly, layers });
    nk_ = layers;
    cell_index_.assign(static_cast<std::size_t>(record.lx) * static_cast<std::size_t>(record.ly) * static_cast<std::size_t>(nk_), -1);
    for (std::size_t n{ 0 }; n < cells.size(); ++n) {
        const std::array<int, 3>& c{ cells[n].ijk };
        const std::size_t slot{ (static_cast<std::size_t>(c[0]) * static_cast<std::size_t>(record.ly) + static_cast<std::size_t>(c[1])) * static_cast<std::size_t>(nk_)
                                + static_cast<std::size_t>(c[2] - kmin_) };
        cell_index_[slot] = static_cast<int>(n);
    }
}

Vec3 SamplerSearch::area_centroid(const Facet& f)
{
    const Vec3& p0{ f.poly[0] };
    double wsum{ 0.0 };
    std::array<double, 3> acc{ 0.0, 0.0, 0.0 };
    for (std::size_t q{ 1 }; q + 1 < f.poly.size(); ++q) {
        const Vec3& a{ f.poly[q] };
        const Vec3& b{ f.poly[q + 1] };
        const Vec3 cr{ VectorOps::cross(VectorOps::subtract(a, p0), VectorOps::subtract(b, p0)) };
        const double w{ 0.5 * std::sqrt(VectorOps::dot(cr, cr)) };
        wsum += w;
        acc[0] += w * (p0.x + a.x + b.x) / 3.0;
        acc[1] += w * (p0.y + a.y + b.y) / 3.0;
        acc[2] += w * (p0.z + a.z + b.z) / 3.0;
    }
    return Vec3{ acc[0] / wsum, acc[1] / wsum, acc[2] / wsum };
}

SamplerCandidate SamplerSearch::candidate(const Vec3& xs, const int index, const int sx, const int sy) const
{
    const Vec3& ce{ geometry_.cells()[static_cast<std::size_t>(index)].centroid };
    const double dx{ ce.x + static_cast<double>(sx) - xs.x };
    const double dy{ ce.y + static_cast<double>(sy) - xs.y };
    const double dz{ ce.z - xs.z };
    const double d2{ dx * dx + dy * dy + dz * dz }; // numpy delta @ delta: the plain sum (measured equal on 300000 cases)
    return SamplerCandidate{ d2, index, sx, sy, false };
}

bool SamplerSearch::window(const Vec3& xs, const int m, std::vector<SamplerCandidate>& out) const
{
    const TerrainRecord& record{ geometry_.record() };
    const auto fx{ static_cast<int>(std::floor(xs.x)) };
    const auto fy{ static_cast<int>(std::floor(xs.y)) };
    const auto fz{ static_cast<int>(std::floor(xs.z)) };
    for (int big_i{ fx - m }; big_i <= fx + m; ++big_i) {
        const int sx{ big_i - TerrainGeometry::wrap(big_i, record.lx) };
        if (sx != -record.lx && sx != 0 && sx != record.lx) {
            return false;
        }
    }
    for (int big_j{ fy - m }; big_j <= fy + m; ++big_j) {
        const int sy{ big_j - TerrainGeometry::wrap(big_j, record.ly) };
        if (sy != -record.ly && sy != 0 && sy != record.ly) {
            return false;
        }
    }
    out.clear();
    for (int big_i{ fx - m }; big_i <= fx + m; ++big_i) {
        const int i{ TerrainGeometry::wrap(big_i, record.lx) };
        for (int big_j{ fy - m }; big_j <= fy + m; ++big_j) {
            const int j{ TerrainGeometry::wrap(big_j, record.ly) };
            const int k_first{ std::max(fz - m, kmin_) };
            const int k_last{ std::min(fz + m, kmin_ + nk_ - 1) };
            for (int k{ k_first }; k <= k_last; ++k) {
                const std::size_t slot{ (static_cast<std::size_t>(i) * static_cast<std::size_t>(record.ly) + static_cast<std::size_t>(j)) * static_cast<std::size_t>(nk_)
                                        + static_cast<std::size_t>(k - kmin_) };
                const int index{ cell_index_[slot] };
                if (index >= 0) {
                    out.push_back(candidate(xs, index, big_i - i, big_j - j));
                }
            }
        }
    }
    std::sort(out.begin(), out.end(), candidate_before);
    return true;
}

void SamplerSearch::full_list(const Vec3& xs, std::vector<SamplerCandidate>& out) const
{
    const TerrainRecord& record{ geometry_.record() };
    const std::array<int, 3> shifts_x{ -record.lx, 0, record.lx };
    const std::array<int, 3> shifts_y{ -record.ly, 0, record.ly };
    const auto count{ static_cast<int>(geometry_.cells().size()) };
    out.clear();
    for (const int sx : shifts_x) {
        for (const int sy : shifts_y) {
            for (int index{ 0 }; index < count; ++index) {
                out.push_back(candidate(xs, index, sx, sy));
            }
        }
    }
    std::sort(out.begin(), out.end(), candidate_before);
}

Vec3 SamplerSearch::shifted_centroid(const SamplerCandidate& c) const
{
    const Vec3& ce{ geometry_.cells()[static_cast<std::size_t>(c.index)].centroid };
    return Vec3{ ce.x + static_cast<double>(c.sx), ce.y + static_cast<double>(c.sy), ce.z };
}

std::vector<SamplerCandidate> SamplerSearch::visible_prefix(const Vec3& xs, const std::vector<SamplerCandidate>& sorted, FacetCounters& counters) const
{
    std::vector<SamplerCandidate> kept{};
    for (const SamplerCandidate& z : sorted) {
        if (static_cast<int>(kept.size()) == neighbours_) {
            break;
        }
        if (z.d2 == 0.0) {
            SamplerCandidate exact{ z };
            exact.visible = surface_.segment_free(xs, shifted_centroid(z)); // for the R3 check of build_sampler
            kept.push_back(exact);
            continue;
        }
        if (surface_.segment_free(xs, shifted_centroid(z))) {
            SamplerCandidate visible{ z };
            visible.visible = true;
            kept.push_back(visible);
        } else {
            counters.hidden_skipped += 1;
        }
    }
    return kept;
}

void SamplerSearch::search(const std::size_t facet, FacetSample& sample, FacetCounters& counters) const
{
    const Facet& f{ geometry_.facets()[facet] };
    const Vec3 cen{ area_centroid(f) };
    const Vec3 xs{ cen.x + 0.5 * f.unit.x, cen.y + 0.5 * f.unit.y, cen.z + 0.5 * f.unit.z };
    sample.point = xs;
    const Vec3 start{ f.key.lid ? cen : Vec3{ cen.x + 1e-9 * f.unit.x, cen.y + 1e-9 * f.unit.y, cen.z + 1e-9 * f.unit.z } };
    if (!surface_.segment_free(start, xs)) {
        sample.error = "R2: sample of facet " + f.key.python_text() + " not visible";
        return;
    }
    std::vector<SamplerCandidate> sorted{};
    if (contained_) {
        for (int m{ 3 }; m <= limit_; m *= 2) {
            if (!window(xs, m, sorted)) {
                break;
            }
            std::vector<SamplerCandidate> prefix{ visible_prefix(xs, sorted, counters) };
            const bool complete{ static_cast<int>(prefix.size()) == neighbours_ };
            if (complete && prefix.back().d2 < static_cast<double>(m * m)) {
                counters.window += 1;
                counters.largest_m = std::max<std::int64_t>(counters.largest_m, m);
                sample.kept = std::move(prefix);
                weights(f, sample);
                return;
            }
        }
    }
    counters.full += 1;
    full_list(xs, sorted);
    sample.kept = visible_prefix(xs, sorted, counters);
    if (static_cast<int>(sample.kept.size()) < neighbours_) {
        sample.error = "R3: only " + std::to_string(sample.kept.size()) + " visible stencil candidates (need " + std::to_string(neighbours_)
                       + ") for the sample " + point_text(xs) + " at facet " + f.key.python_text();
        return;
    }
    weights(f, sample);
}

void SamplerSearch::weights(const Facet& f, FacetSample& sample) const
{
    const Vec3& xs{ sample.point };
    std::vector<StencilCandidate> candidates{};
    for (const SamplerCandidate& z : sample.kept) {
        const Vec3& ce{ geometry_.cells()[static_cast<std::size_t>(z.index)].centroid };
        // delta = (ce[0] + sx - xs[0], ce[1] + sy - xs[1], ce[2] - xs[2]), the expression of build_sampler
        const std::array<double, 3> delta{ ce.x + static_cast<double>(z.sx) - xs.x, ce.y + static_cast<double>(z.sy) - xs.y, ce.z - xs.z };
        candidates.push_back(StencilCandidate{ z.d2, z.index, z.sx, z.sy, delta });
    }
    StencilResult result{};
    try {
        result = StencilWeights::compute(candidates, neighbours_);
    } catch (const std::exception& error) { // the rejection gets the facet, as build_sampler adds it
        std::ostringstream text{};
        text << error.what() << " at facet " << f.key.python_text();
        sample.error = text.str();
        return;
    }
    for (const StencilCandidate& z : result.chosen) {
        const SamplerCandidate* kept{ nullptr };
        for (const SamplerCandidate& k : sample.kept) {
            if (k.index == z.index && k.sx == z.sx && k.sy == z.sy) {
                kept = &k;
            }
        }
        if (kept == nullptr || !kept->visible) {
            const std::array<int, 3>& cell{ geometry_.cells()[static_cast<std::size_t>(z.index)].ijk };
            sample.error = "R3: stencil cell (" + std::to_string(cell[0]) + ", " + std::to_string(cell[1]) + ", " + std::to_string(cell[2])
                           + ") not visible from the sample of facet " + f.key.python_text();
            return;
        }
        sample.cells.push_back(z.index);
    }
    sample.least_squares = result.least_squares;
    sample.weights = result.coefficients;
    sample.condition = result.condition;
    sample.reproduction = result.reproduction;
}

void SamplerSearch::run(const int threads)
{
    const std::size_t count{ geometry_.facets().size() };
    samples_.assign(count, {});
    facet_counters_.assign(count, {});
    std::atomic<std::size_t> next{ 0 };
    const auto worker{ [&]() {
        for (std::size_t n{ next.fetch_add(1) }; n < count; n = next.fetch_add(1)) {
            try {
                search(n, samples_[n], facet_counters_[n]);
            } catch (const std::exception& error) { // kept per facet; write() reports the first in facet order
                samples_[n].error = error.what();
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
}

SamplerCounters SamplerSearch::counters() const
{
    SamplerCounters total{};
    total.contained = contained_;
    for (const FacetCounters& c : facet_counters_) {
        total.window += c.window;
        total.full += c.full;
        total.largest_m = std::max(total.largest_m, c.largest_m);
        total.hidden_skipped += c.hidden_skipped;
    }
    return total;
}

void SamplerSearch::check() const
{
    for (const FacetSample& sample : samples_) {
        if (!sample.error.empty()) {
            throw std::runtime_error(sample.error);
        }
    }
}

} // namespace fluidx3d::experimental::wall_builder
