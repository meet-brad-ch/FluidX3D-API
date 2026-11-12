/// @file terrain.cpp
/// @brief The terrain case, its cells and facets, ported from builder3d.py in Python's operation order.
/// Built with FLUIDX3D_PRECISE_FLOAT: the operation order below is the order of the Python code, and must stay so.

#include "terrain.hpp"

#include "parallel_for.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace fluidx3d::experimental::wall_builder {

namespace {

constexpr int TABLE_MARGIN{ 2 }; // columns and layers around the box that candidates() can reach

/// @brief A triangle vertex: integer x, y and the exact height.
struct ExactPoint {
    int x{};   ///< Unwrapped x.
    int y{};   ///< Unwrapped y.
    Dyadic z{}; ///< Exact height.
};

/// @brief One column triangle (builder3d.py Case.triangles).
struct ColumnTriangle {
    int tag{};                           ///< 0 for "T0", 1 for "T1".
    std::array<ExactPoint, 3> points{};  ///< The vertices.
};

/// @brief The exact plane z = a x + b y + g of a column triangle (builder3d.py exact_plane).
struct ExactPlaneValues {
    Dyadic a{}; ///< Slope in x.
    Dyadic b{}; ///< Slope in y.
    Dyadic g{}; ///< Offset at the unwrapped position.
};

/// @brief The 2D polygon operations of builder3d.py: polygon_clip2 and translated_fan_area.
class Polygon2 {
public:
    /// @brief Signed area by a fan from the first vertex (translated_fan_area).
    [[nodiscard]] static double fan_area(const std::vector<Vec3>& poly)
    {
        const double x0{ poly[0].x };
        const double y0{ poly[0].y };
        double sum{ 0.0 };
        for (std::size_t q{ 1 }; q + 1 < poly.size(); ++q) {
            const double term{ (poly[q].x - x0) * (poly[q + 1].y - y0) - (poly[q + 1].x - x0) * (poly[q].y - y0) };
            sum += term;
        }
        return 0.5 * sum;
    }

    /// @brief poly cut to nx x + ny y <= d; empty below 3 vertices or for zero area (polygon_clip2).
    [[nodiscard]] static std::optional<std::vector<Vec3>> clip(const std::vector<Vec3>& poly, const double nx, const double ny, const double d)
    {
        std::vector<Vec3> out{};
        const std::size_t m{ poly.size() };
        const double tol{ PolyClipper::EPS_ON * std::max(1.0, std::fabs(d)) };
        for (std::size_t q{ 0 }; q < m; ++q) {
            const Vec3& p{ poly[q] };
            const Vec3& r{ poly[(q + 1) % m] };
            const double fp{ nx * p.x + ny * p.y - d };
            const double fr{ nx * r.x + ny * r.y - d };
            const int cp{ std::fabs(fp) <= tol ? 0 : (fp > 0 ? 1 : -1) };
            const int cr{ std::fabs(fr) <= tol ? 0 : (fr > 0 ? 1 : -1) };
            if (cp <= 0) {
                out.push_back(p);
            }
            if (cp * cr < 0) {
                const double s{ fp / (fp - fr) };
                out.push_back(Vec3{ p.x + s * (r.x - p.x), p.y + s * (r.y - p.y), 0.0 });
            }
        }
        if (out.size() < 3) {
            return std::nullopt;
        }
        const double area{ fan_area(out) };
        if (std::fabs(area) > 0.0) {
            return out;
        }
        return std::nullopt;
    }
};

/// @brief The column geometry of one record: heights, triangles and exact planes.
class ColumnGeometry {
public:
    explicit ColumnGeometry(const TerrainRecord& source)
        : record{ source }
    {
    }

    /// @brief The exact height of unwrapped node (i, j).
    [[nodiscard]] Dyadic height(const int i, const int j) const
    {
        const int wi{ TerrainGeometry::wrap(i, record.lx) };
        const int wj{ TerrainGeometry::wrap(j, record.ly) };
        const double value{ record.heights[static_cast<std::size_t>(wi * record.ly + wj)] };
        return Dyadic::from_double(value);
    }

    /// @brief The two triangles of unwrapped column (i, j), in Case.triangles order.
    [[nodiscard]] std::array<ColumnTriangle, 2> triangles(const int i, const int j) const
    {
        const ExactPoint p00{ i, j, height(i, j) };
        const ExactPoint p10{ i + 1, j, height(i + 1, j) };
        const ExactPoint p11{ i + 1, j + 1, height(i + 1, j + 1) };
        const ExactPoint p01{ i, j + 1, height(i, j + 1) };
        const int wi{ TerrainGeometry::wrap(i, record.lx) };
        const int wj{ TerrainGeometry::wrap(j, record.ly) };
        const bool main{ record.main_diagonal[static_cast<std::size_t>(wi * record.ly + wj)] };
        if (main) {
            return { ColumnTriangle{ 0, { p00, p10, p11 } }, ColumnTriangle{ 1, { p00, p11, p01 } } };
        }
        return { ColumnTriangle{ 0, { p00, p10, p01 } }, ColumnTriangle{ 1, { p10, p11, p01 } } };
    }

    /// @brief The exact plane of a column triangle; the determinant must be +1 or -1.
    /// @throws std::domain_error for another determinant
    [[nodiscard]] static ExactPlaneValues exact_plane(const ColumnTriangle& tri)
    {
        const ExactPoint& p0{ tri.points[0] };
        const ExactPoint& p1{ tri.points[1] };
        const ExactPoint& p2{ tri.points[2] };
        const int dx1{ p1.x - p0.x };
        const int dy1{ p1.y - p0.y };
        const int dx2{ p2.x - p0.x };
        const int dy2{ p2.y - p0.y };
        const int det{ dx1 * dy2 - dx2 * dy1 };
        if (det != 1 && det != -1) {
            throw std::domain_error("exact_plane: determinant " + std::to_string(det) + " is not +1 or -1");
        }
        const Dyadic dz1{ p1.z - p0.z };
        const Dyadic dz2{ p2.z - p0.z };
        const Dyadic sign{ Dyadic::from_integer(det) };
        const Dyadic a{ (dz1 * Dyadic::from_integer(dy2) - dz2 * Dyadic::from_integer(dy1)) * sign };
        const Dyadic b{ (Dyadic::from_integer(dx1) * dz2 - Dyadic::from_integer(dx2) * dz1) * sign };
        const Dyadic g{ p0.z - a * Dyadic::from_integer(p0.x) - b * Dyadic::from_integer(p0.y) };
        return ExactPlaneValues{ a, b, g };
    }

    /// @brief g of the plane in the frame of an integer origin: g + a ox + b oy - oz (the same rational number as
    /// exact_plane of the translated triangle).
    [[nodiscard]] static Dyadic shifted_offset(const ExactPlaneValues& plane, const int ox, const int oy, const int oz)
    {
        return plane.g + plane.a * Dyadic::from_integer(ox) + plane.b * Dyadic::from_integer(oy) - Dyadic::from_integer(oz);
    }

    const TerrainRecord& record; ///< The input.
};

/// @brief Small facet helpers of builder3d.py: the exact direction signs (Facet.sign) and the triangle footprint.
class FacetTools {
public:
    /// @brief The exact signs of c . n_raw for every direction of the table (Facet.sign).
    [[nodiscard]] static std::array<std::int8_t, 27> direction_signs(const std::vector<Direction>& directions, const Dyadic& nx, const Dyadic& ny,
                                                                    const Dyadic& nz)
    {
        std::array<std::int8_t, 27> signs{};
        for (std::size_t d{ 0 }; d < directions.size(); ++d) {
            const Direction& c{ directions[d] };
            const Dyadic s{ Dyadic::from_integer(c.x) * nx + Dyadic::from_integer(c.y) * ny + Dyadic::from_integer(c.z) * nz };
            signs[d] = static_cast<std::int8_t>(s.sign());
        }
        return signs;
    }

    /// @brief The footprint of a triangle in the frame (ox, oy), reversed when its signed area is negative.
    [[nodiscard]] static std::vector<Vec3> footprint(const ColumnTriangle& tri, const int ox, const int oy)
    {
        std::vector<Vec3> foot{};
        for (const ExactPoint& p : tri.points) {
            foot.push_back(Vec3{ static_cast<double>(p.x - ox), static_cast<double>(p.y - oy), 0.0 });
        }
        const double area{ 0.5 * ((foot[1].x - foot[0].x) * (foot[2].y - foot[0].y) - (foot[2].x - foot[0].x) * (foot[1].y - foot[0].y)) };
        if (area < 0) {
            std::reverse(foot.begin(), foot.end());
        }
        return foot;
    }
};

} // namespace

// ------------------------------------------------------------------------------------------------ FacetKey

bool FacetKey::operator==(const FacetKey& other) const
{
    if (lid != other.lid || i != other.i || j != other.j) {
        return false;
    }
    return lid || (tag == other.tag && k == other.k);
}

std::string FacetKey::python_text() const
{
    if (lid) {
        return "('L', " + std::to_string(i) + ", " + std::to_string(j) + ")";
    }
    return "('T', " + std::to_string(i) + ", " + std::to_string(j) + ", 'T" + std::to_string(tag) + "', " + std::to_string(k) + ")";
}

// ------------------------------------------------------------------------------------------------ TerrainGeometry

int TerrainGeometry::wrap(const int i, const int n)
{
    const int r{ i % n };
    return r < 0 ? r + n : r;
}

TerrainGeometry::TerrainGeometry(TerrainRecord record, const int threads)
    : record_{ std::move(record) }
    , threads_{ threads }
{
    if (threads < 1) {
        throw std::invalid_argument("TerrainGeometry: threads " + std::to_string(threads) + " (need 1 or more)");
    }
    if (record_.lx < 1 || record_.ly < 1 || record_.heights.size() != static_cast<std::size_t>(record_.lx * record_.ly)) {
        throw std::domain_error("TerrainGeometry: " + std::to_string(record_.heights.size()) + " heights for a " + std::to_string(record_.lx) + " x "
                                + std::to_string(record_.ly) + " box");
    }
    if (record_.directions.size() > 27) {
        throw std::domain_error("TerrainGeometry: " + std::to_string(record_.directions.size()) + " directions (at most 27)");
    }
    const double lowest{ *std::min_element(record_.heights.begin(), record_.heights.end()) };
    zmin_ = static_cast<int>(std::floor(lowest));
    build_base_table();
    build_column_index();
    build_cells();
    build_facets();
}

double TerrainGeometry::height_value(const int i, const int j) const
{
    const int wi{ wrap(i, record_.lx) };
    const int wj{ wrap(j, record_.ly) };
    return record_.heights[static_cast<std::size_t>(wi * record_.ly + wj)];
}

void TerrainGeometry::build_base_table()
{
    base_i0_ = -TABLE_MARGIN;
    base_j0_ = -TABLE_MARGIN;
    base_k0_ = zmin_ - TABLE_MARGIN;
    base_ni_ = record_.lx + 2 * TABLE_MARGIN;
    base_nj_ = record_.ly + 2 * TABLE_MARGIN;
    base_nk_ = record_.z_top - zmin_ + 2 * TABLE_MARGIN;
    bases_.resize(static_cast<std::size_t>(base_ni_) * static_cast<std::size_t>(base_nj_) * static_cast<std::size_t>(base_nk_));
    const auto layers{ static_cast<std::size_t>(base_nk_) };
    const auto columns_y{ static_cast<std::size_t>(base_nj_) };
    ParallelFor::run(bases_.size(), threads_, [&](const std::size_t slot) {
        const auto a{ static_cast<int>(slot / (columns_y * layers)) };
        const auto b{ static_cast<int>((slot / layers) % columns_y) };
        const auto c{ static_cast<int>(slot % layers) };
        bases_[slot] = compute_bases(a + base_i0_, b + base_j0_, c + base_k0_);
    });
}

const std::vector<FacetBase>& TerrainGeometry::bases_at(const int i, const int j, const int k) const
{
    const int a{ i - base_i0_ };
    const int b{ j - base_j0_ };
    const int c{ k - base_k0_ };
    if (a < 0 || a >= base_ni_ || b < 0 || b >= base_nj_ || c < 0 || c >= base_nk_) {
        throw std::out_of_range("terrain_facets: column (" + std::to_string(i) + ", " + std::to_string(j) + ") layer " + std::to_string(k)
                                + " is outside the precomputed table");
    }
    const std::size_t slot{ (static_cast<std::size_t>(a) * static_cast<std::size_t>(base_nj_) + static_cast<std::size_t>(b)) * static_cast<std::size_t>(base_nk_)
                            + static_cast<std::size_t>(c) };
    return bases_[slot];
}

std::vector<FacetBase> TerrainGeometry::compute_bases(const int i, const int j, const int k) const
{
    const ColumnGeometry columns{ record_ };
    std::vector<FacetBase> out{};
    for (const ColumnTriangle& tri : columns.triangles(i, j)) {
        const ExactPlaneValues plane{ ColumnGeometry::exact_plane(tri) };
        if (plane.a.sign() == 0 && plane.b.sign() == 0 && plane.g.floor() != k) {
            continue;
        }
        const Dyadic g_own_exact{ ColumnGeometry::shifted_offset(plane, i, j, k) };
        const double a{ plane.a.to_double() };
        const double b{ plane.b.to_double() };
        const double g_own{ g_own_exact.to_double() };
        const std::vector<Vec3> foot{ FacetTools::footprint(tri, i, j) };
        std::optional<std::vector<Vec3>> part{ Polygon2::clip(foot, a, b, 1.0 - g_own) };
        if (part) {
            part = Polygon2::clip(*part, -a, -b, g_own);
        }
        if (!part) {
            continue;
        }
        const double axy{ Polygon2::fan_area(*part) };
        const Vec3 n_raw{ -a, -b, 1.0 };
        const double ln{ std::sqrt(VectorOps::dot(n_raw, n_raw)) };
        FacetBase base{};
        base.key = FacetKey{ false, wrap(i, record_.lx), wrap(j, record_.ly), tri.tag, k };
        base.ae = plane.a;
        base.be = plane.b;
        base.ge = plane.g;
        base.part = std::move(*part);
        base.g_own = g_own;
        base.a = a;
        base.b = b;
        base.area = Vec3{ axy * n_raw.x, axy * n_raw.y, axy * n_raw.z };
        base.unit = Vec3{ n_raw.x / ln, n_raw.y / ln, n_raw.z / ln };
        base.length = ln;
        base.cell = { base.key.i, base.key.j, k };
        base.signs = FacetTools::direction_signs(record_.directions, -plane.a, -plane.b, Dyadic::from_integer(1));
        out.push_back(std::move(base));
    }
    return out;
}

Facet TerrainGeometry::facet_from_base(const FacetBase& base, const int i, const int j, const int k, const std::array<int, 3>& origin) const
{
    const double dx{ static_cast<double>(i - origin[0]) };
    const double dy{ static_cast<double>(j - origin[1]) };
    const double dz{ static_cast<double>(k - origin[2]) };
    Facet f{};
    f.key = base.key;
    for (const Vec3& p : base.part) {
        const double z{ base.a * p.x + base.b * p.y + base.g_own + dz };
        f.poly.push_back(Vec3{ p.x + dx, p.y + dy, z });
    }
    const ExactPlaneValues plane{ base.ae, base.be, base.ge };
    const Dyadic g_frame{ ColumnGeometry::shifted_offset(plane, origin[0], origin[1], origin[2]) };
    f.n_raw = { -base.ae, -base.be, Dyadic::from_integer(1) };
    f.area = base.area;
    f.unit = base.unit;
    f.cell = base.cell;
    f.plane = Plane{ base.unit, g_frame.to_double() / base.length };
    f.d_raw = base.ge;
    f.signs = base.signs;
    return f;
}

void TerrainGeometry::terrain_facets(const int i, const int j, const int k, const std::array<int, 3>& origin, std::vector<Facet>& out) const
{
    for (const FacetBase& base : bases_at(i, j, k)) {
        out.push_back(facet_from_base(base, i, j, k, origin));
    }
}

Facet TerrainGeometry::lid_facet(const int i, const int j, const std::array<int, 3>& origin) const
{
    const double z{ static_cast<double>(record_.z_top - origin[2]) };
    const double x0{ static_cast<double>(i - origin[0]) };
    const double y0{ static_cast<double>(j - origin[1]) };
    Facet f{};
    f.key = FacetKey{ true, wrap(i, record_.lx), wrap(j, record_.ly), 0, 0 };
    f.poly = { Vec3{ x0, y0, z }, Vec3{ x0 + 1.0, y0, z }, Vec3{ x0 + 1.0, y0 + 1.0, z }, Vec3{ x0, y0 + 1.0, z } };
    f.n_raw = { Dyadic{}, Dyadic{}, Dyadic::from_integer(-1) };
    f.area = Vec3{ 0.0, 0.0, -1.0 };
    f.unit = Vec3{ 0.0, 0.0, -1.0 };
    f.cell = { f.key.i, f.key.j, record_.z_top - 1 };
    f.plane = Plane{ Vec3{ 0.0, 0.0, -1.0 }, -z };
    f.d_raw = Dyadic::from_integer(-record_.z_top);
    f.signs = FacetTools::direction_signs(record_.directions, f.n_raw[0], f.n_raw[1], f.n_raw[2]);
    return f;
}

Facet TerrainGeometry::facet_in_frame(const Facet& f, const std::array<int, 3>& origin) const
{
    if (f.key.lid) {
        return lid_facet(f.key.i, f.key.j, origin);
    }
    std::vector<Facet> rebuilt{};
    terrain_facets(f.key.i, f.key.j, f.key.k, origin, rebuilt);
    for (Facet& g : rebuilt) {
        if (g.key == f.key) {
            return std::move(g);
        }
    }
    throw std::runtime_error("facet " + f.key.python_text() + " is not rebuilt in the frame");
}

void TerrainGeometry::candidates(const int i, const int j, const int k, const FacetKey* exclude, const std::array<int, 3>& origin,
                                 std::vector<Candidate>& out) const
{
    const bool lid_layer{ record_.z_top - 1 >= k - 1 && record_.z_top - 1 <= k + 1 };
    std::vector<Facet> column{};
    for (int di{ -1 }; di <= 1; ++di) {
        for (int dj{ -1 }; dj <= 1; ++dj) {
            const int ii{ i + di };
            const int jj{ j + dj };
            const std::array<int, 2> image{ ii - wrap(ii, record_.lx), jj - wrap(jj, record_.ly) };
            column.clear();
            for (int kk{ k - 1 }; kk <= k + 1; ++kk) {
                terrain_facets(ii, jj, kk, origin, column);
            }
            if (lid_layer) {
                column.push_back(lid_facet(ii, jj, origin));
            }
            for (Facet& f : column) {
                const bool excluded{ exclude != nullptr && f.key == *exclude };
                if (!excluded) {
                    out.push_back(Candidate{ std::move(f), image });
                }
            }
        }
    }
}

std::vector<Poly> TerrainGeometry::cell_pieces(const int i, const int j, const int k) const
{
    const ColumnGeometry columns{ record_ };
    std::vector<Poly> out{};
    for (const ColumnTriangle& tri : columns.triangles(i, j)) {
        const ExactPlaneValues plane{ ColumnGeometry::exact_plane(tri) };
        const double a{ plane.a.to_double() };
        const double b{ plane.b.to_double() };
        const double g{ ColumnGeometry::shifted_offset(plane, i, j, k).to_double() };
        std::optional<Poly> piece{ Poly::box(Vec3{ 0.0, 0.0, 0.0 }, Vec3{ 1.0, 1.0, 1.0 }) };
        const double ln{ std::sqrt(a * a + b * b + 1.0) };
        std::vector<Plane> planes{ Plane{ Vec3{ a / ln, b / ln, -1.0 / ln }, -g / ln } };
        const std::vector<Vec3> foot{ FacetTools::footprint(tri, i, j) };
        for (std::size_t q{ 0 }; q < 3; ++q) {
            const Vec3& p0{ foot[q] };
            const Vec3& p1{ foot[(q + 1) % 3] };
            const double nx{ p1.y - p0.y };
            const double ny{ -(p1.x - p0.x) };
            // math.hypot is Python's own algorithm; it equals std::hypot only on these edge vectors (wall_builder_test.cpp)
            if (std::fabs(nx) > 1.0 || std::fabs(ny) > 1.0) {
                throw std::domain_error("cell_pieces: edge vector component outside -1..1");
            }
            const double l2{ std::hypot(nx, ny) };
            planes.push_back(Plane{ Vec3{ nx / l2, ny / l2, 0.0 }, (nx * p0.x + ny * p0.y) / l2 });
        }
        for (const Plane& p : planes) {
            if (piece) {
                piece = PolyClipper::clip(*piece, p);
            }
        }
        if (piece) {
            out.push_back(std::move(*piece));
        }
    }
    return out;
}

std::array<ColumnTrianglePlane, 2> TerrainGeometry::column_planes(const int i, const int j) const
{
    const ColumnGeometry columns{ record_ };
    const std::array<ColumnTriangle, 2> triangles{ columns.triangles(i, j) };
    std::array<ColumnTrianglePlane, 2> out{};
    for (std::size_t t{ 0 }; t < 2; ++t) {
        const ColumnTriangle& tri{ triangles[t] };
        const ExactPlaneValues plane{ ColumnGeometry::exact_plane(tri) };
        for (std::size_t v{ 0 }; v < 3; ++v) {
            out[t].foot[2 * v] = static_cast<double>(tri.points[v].x);
            out[t].foot[2 * v + 1] = static_cast<double>(tri.points[v].y);
        }
        out[t].a = plane.a.to_double();
        out[t].b = plane.b.to_double();
        out[t].g = plane.g.to_double();
    }
    return out;
}

void TerrainGeometry::cell_moments(TerrainCell& cell, const bool full)
{
    const double i{ static_cast<double>(cell.ijk[0]) };
    const double j{ static_cast<double>(cell.ijk[1]) };
    const double k{ static_cast<double>(cell.ijk[2]) };
    // the exact volume and moment of every piece are cached here, once: the donor pass copies the pieces for each of the
    // q directions, and the copies share the caches (the shared geometry is only read after this)
    for (const Poly& piece : cell.pieces) {
        static_cast<void>(piece.moment());
    }
    if (full) {
        cell.volume = 1.0; // #558: the exact cube
        cell.centroid = Vec3{ i + 0.5, j + 0.5, k + 0.5 };
        cell.centroid_local = Vec3{ 0.5, 0.5, 0.5 };
        return;
    }
    double volume{ 0.0 };
    Vec3 moment{ 0.0, 0.0, 0.0 };
    for (const Poly& piece : cell.pieces) {
        const VolumeMoment vm{ piece.moment() };
        volume += vm.volume;
        moment = Vec3{ moment.x + vm.moment.x, moment.y + vm.moment.y, moment.z + vm.moment.z };
    }
    // task 106: the centroid of the local-frame moment, shifted back by the cell origin
    const Vec3 local{ moment.x / volume, moment.y / volume, moment.z / volume };
    cell.volume = volume;
    cell.centroid_local = local;
    cell.centroid = Vec3{ local.x + i, local.y + j, local.z + k };
}

void TerrainGeometry::build_column_index()
{
    const auto columns{ static_cast<std::size_t>(record_.lx) * static_cast<std::size_t>(record_.ly) };
    kfirst_.assign(columns, 0);
    first_.assign(columns + 1, 0);
    for (std::size_t c{ 0 }; c < columns; ++c) {
        const auto i{ static_cast<int>(c / static_cast<std::size_t>(record_.ly)) };
        const auto j{ static_cast<int>(c % static_cast<std::size_t>(record_.ly)) };
        const double lowest{ std::min({ height_value(i, j), height_value(i, j + 1), height_value(i + 1, j), height_value(i + 1, j + 1) }) };
        int k{ zmin_ };
        while (k < record_.z_top && static_cast<double>(k + 1) <= lowest) { // bounded by z_top; the empty layers of Builder.__init__
            ++k;
        }
        kfirst_[c] = k;
        first_[c + 1] = first_[c] + (record_.z_top - k);
        if (k < record_.z_top) {
            lowest_layer_ = std::min(lowest_layer_, k);
        }
    }
    if (first_.back() == 0) {
        throw std::domain_error("TerrainGeometry: no fluid cell below z_top " + std::to_string(record_.z_top));
    }
}

std::array<int, 3> TerrainGeometry::cell_ijk(const std::size_t n) const
{
    const auto cell{ static_cast<std::int64_t>(n) };
    const auto above{ std::upper_bound(first_.begin(), first_.end(), cell) };
    const auto column{ static_cast<std::size_t>(above - first_.begin()) - 1u };
    const auto i{ static_cast<int>(column / static_cast<std::size_t>(record_.ly)) };
    const auto j{ static_cast<int>(column % static_cast<std::size_t>(record_.ly)) };
    const auto k{ kfirst_[column] + static_cast<int>(cell - first_[column]) };
    return { i, j, k };
}

std::int64_t TerrainGeometry::cell_index(const int i, const int j, const int k) const
{
    const int wi{ wrap(i, record_.lx) };
    const int wj{ wrap(j, record_.ly) };
    const auto column{ static_cast<std::size_t>(wi) * static_cast<std::size_t>(record_.ly) + static_cast<std::size_t>(wj) };
    if (k < kfirst_[column] || k >= record_.z_top) {
        return -1;
    }
    return first_[column] + (k - kfirst_[column]);
}

TerrainCell TerrainGeometry::donor_cell(const std::size_t n) const
{
    TerrainCell cell{};
    cell.ijk = cell_ijk(n);
    cell.pieces = cell_pieces(cell.ijk[0], cell.ijk[1], cell.ijk[2]);
    if (cell.pieces.empty()) {
        throw std::runtime_error("cell (" + std::to_string(cell.ijk[0]) + ", " + std::to_string(cell.ijk[1]) + ", " + std::to_string(cell.ijk[2])
                                 + ") classified fluid but has no piece");
    }
    // the donor pass copies the pieces for each direction; the copies share these cached volumes and moments
    for (const Poly& piece : cell.pieces) {
        static_cast<void>(piece.moment());
    }
    return cell;
}

void TerrainGeometry::build_cells()
{
    const std::size_t count{ static_cast<std::size_t>(first_.back()) };
    volume_.assign(count, 0.0);
    centroid_.assign(count, Vec3{});
    local_.assign(count, Vec3{});
    const auto columns{ static_cast<std::size_t>(record_.lx) * static_cast<std::size_t>(record_.ly) };
    ParallelFor::run(columns, threads_, [&](const std::size_t c) {
        const auto i{ static_cast<int>(c / static_cast<std::size_t>(record_.ly)) };
        const auto j{ static_cast<int>(c % static_cast<std::size_t>(record_.ly)) };
        const double highest{ std::max({ height_value(i, j), height_value(i, j + 1), height_value(i + 1, j), height_value(i + 1, j + 1) }) };
        for (int k{ kfirst_[c] }; k < record_.z_top; ++k) {
            const auto n{ static_cast<std::size_t>(first_[c] + (k - kfirst_[c])) };
            TerrainCell cell{ donor_cell(n) };
            const bool full{ static_cast<double>(k) >= highest };
            cell_moments(cell, full);
            volume_[n] = cell.volume;
            centroid_[n] = cell.centroid;
            local_[n] = cell.centroid_local;
        }
    });
}

void TerrainGeometry::build_facets()
{
    const auto columns{ static_cast<std::size_t>(record_.lx) * static_cast<std::size_t>(record_.ly) };
    std::vector<std::vector<Facet>> per_column(columns);
    ParallelFor::run(columns, threads_, [&](const std::size_t c) {
        const auto i{ static_cast<int>(c / static_cast<std::size_t>(record_.ly)) };
        const auto j{ static_cast<int>(c % static_cast<std::size_t>(record_.ly)) };
        for (int k{ kfirst_[c] }; k < record_.z_top; ++k) {
            terrain_facets(i, j, k, { 0, 0, 0 }, per_column[c]);
        }
        per_column[c].push_back(lid_facet(i, j, { 0, 0, 0 }));
    });
    std::vector<Facet> facet_list{};
    for (std::vector<Facet>& column : per_column) {
        for (Facet& facet : column) {
            facet_list.push_back(std::move(facet));
        }
    }
    // sorted by Python's str() of the key; the texts are made once
    std::vector<std::string> texts{};
    for (const Facet& f : facet_list) {
        texts.push_back(f.key.python_text());
    }
    std::vector<std::size_t> order(facet_list.size());
    std::iota(order.begin(), order.end(), std::size_t{ 0 });
    std::stable_sort(order.begin(), order.end(), [&texts](const std::size_t x, const std::size_t y) { return texts[x] < texts[y]; });
    for (const std::size_t n : order) {
        facets_.push_back(std::move(facet_list[n]));
    }
}

} // namespace fluidx3d::experimental::wall_builder
