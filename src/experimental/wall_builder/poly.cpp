/// @file poly.cpp
/// @brief The binary64 convex polyhedra of the wall builder, ported operation for operation from builder3d.py.
/// Built with FLUIDX3D_PRECISE_FLOAT: the operation order below is the order of the Python code, and must stay so.

#include "poly.hpp"

#include "correct_rounding.hpp"
#include "dyadic.hpp"
#include "wide_int.hpp"

#include <intrin.h>

#include <algorithm>
#include <span>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace fluidx3d::experimental::wall_builder {

/// @brief A signed 256-bit two's-complement accumulator for the fan determinants whose scaled coordinates fit in 62 bits:
/// differences then fit in int64, cross-product terms in 128 bits and a determinant in 194 bits. It gives the same exact
/// integers as WideInt with fixed-size branch-free arithmetic (the general WideInt path was most of the volume cost).
class Int256 {
public:
    std::array<std::uint64_t, 4> limbs{}; ///< Two's complement, least significant first.

    /// @brief this += a * b for an int64 a and a signed 128-bit b (hi:lo).
    void add_product(const std::int64_t a, const std::int64_t b_high, const std::uint64_t b_low)
    {
        // the magnitudes, their 192-bit product, then the sign
        const bool negative{ (a < 0) != (b_high < 0) };
        const std::uint64_t ua{ a < 0 ? 0u - static_cast<std::uint64_t>(a) : static_cast<std::uint64_t>(a) };
        std::uint64_t ub_low{ b_low };
        std::uint64_t ub_high{ static_cast<std::uint64_t>(b_high) };
        if (b_high < 0) {
            const unsigned char borrow{ _subborrow_u64(0, 0u, ub_low, &ub_low) };
            _subborrow_u64(borrow, 0u, ub_high, &ub_high);
        }
        std::uint64_t p1{ 0u };
        const std::uint64_t p0{ _umul128(ua, ub_low, &p1) };
        std::uint64_t p2{ 0u };
        const std::uint64_t q1{ _umul128(ua, ub_high, &p2) };
        std::array<std::uint64_t, 4> term{ p0, 0u, 0u, 0u };
        const unsigned char c{ _addcarry_u64(0, p1, q1, &term[1]) };
        term[2] = p2 + c;
        if (negative) {
            unsigned char borrow{ 0 };
            for (std::size_t n{ 0 }; n < 4; ++n) {
                borrow = _subborrow_u64(borrow, 0u, term[n], &term[n]);
            }
        }
        unsigned char carry{ 0 };
        for (std::size_t n{ 0 }; n < 4; ++n) {
            carry = _addcarry_u64(carry, limbs[n], term[n], &limbs[n]);
        }
    }

    /// @brief The absolute value as a WideInt.
    [[nodiscard]] WideInt magnitude() const
    {
        std::array<std::uint64_t, 4> m{ limbs };
        if (static_cast<std::int64_t>(limbs[3]) < 0) {
            unsigned char borrow{ 0 };
            for (std::size_t n{ 0 }; n < 4; ++n) {
                borrow = _subborrow_u64(borrow, 0u, m[n], &m[n]);
            }
        }
        return WideInt::from_limbs(m.data(), 4);
    }
};

/// @brief The signed 128-bit value a * b - c * d of int64 operands, as hi:lo (|value| < 2^127 for |operands| < 2^63).
struct Wide128 {
    std::int64_t high{};
    std::uint64_t low{};

    static Wide128 cross(const std::int64_t a, const std::int64_t b, const std::int64_t c, const std::int64_t d)
    {
        std::int64_t h1{ 0 };
        const std::uint64_t l1{ static_cast<std::uint64_t>(_mul128(a, b, &h1)) };
        std::int64_t h2{ 0 };
        const std::uint64_t l2{ static_cast<std::uint64_t>(_mul128(c, d, &h2)) };
        std::uint64_t low{ 0u };
        const unsigned char borrow{ _subborrow_u64(0, l1, l2, &low) };
        std::uint64_t high{ 0u };
        _subborrow_u64(borrow, static_cast<std::uint64_t>(h1), static_cast<std::uint64_t>(h2), &high);
        return Wide128{ static_cast<std::int64_t>(high), low };
    }
};

/// @brief The exact fan determinants of one polyhedron (Poly.fan_tetrahedra, scaled_integers, tetra_determinants).
/// Every coordinate is a binary64 m * 2^e; with D = 2^power the largest denominator, each is the exact integer
/// m * 2^(e + power), and the determinant of the scaled points is the exact integer D^3 det.
class FanDeterminants {
public:
    /// @brief One fan triangle: vertex indices a, b, c (a = the face's first vertex).
    struct Triangle {
        int a{};
        int b{};
        int c{};
    };

    /// @brief Computes |det| of every fan tetrahedron (g, a, b, c).
    /// @throws std::domain_error when a coordinate is not finite
    /// @throws std::overflow_error when a determinant needs more than WideInt::BITS (1024) bits
    explicit FanDeterminants(const Poly& poly)
        : centre{ poly.interior() }
    {
        collect_triangles(poly);
        find_power(poly);
        compute(poly);
    }

    Vec3 centre{};                       ///< g: the vertex mean.
    std::vector<Triangle> triangles{};   ///< The fan triangles in face order.
    std::vector<WideInt> determinants{}; ///< |det| scaled by D^3, one per triangle.
    WideInt total{};                     ///< The sum of determinants.
    int power{};                         ///< D = 2^power: the largest denominator of all coordinates.

    /// @brief An exact determinant (or sum) divided by 6 D^3, rounded once.
    [[nodiscard]] double weight(const WideInt& value) const
    {
        const int exponent{ -(1 + 3 * power) };
        return CorrectRounding::round_quotient(value, 3u, exponent);
    }

private:
    /// @brief A binary64 as m * 2^e with m odd (or zero), from its bits (Python's as_integer_ratio).
    struct Parts {
        std::int64_t mantissa{};
        int exponent{};
    };

    std::vector<char> used_{}; ///< Per vertex: in some fan triangle.

    static Parts parts(const double x)
    {
        if (!std::isfinite(x)) {
            throw std::domain_error("a polyhedron coordinate is not finite: " + std::to_string(x));
        }
        const std::uint64_t bits{ std::bit_cast<std::uint64_t>(x) };
        const auto biased{ static_cast<int>((bits >> 52) & 0x7FFu) };
        const std::uint64_t fraction{ bits & ((std::uint64_t{ 1 } << 52) - 1u) };
        if (biased == 0 && fraction == 0u) {
            return Parts{ 0, 0 };
        }
        std::uint64_t m{ biased == 0 ? fraction : (fraction | (std::uint64_t{ 1 } << 52)) };
        int e{ biased == 0 ? -1074 : biased - 1075 };
        const int trailing{ std::countr_zero(m) };
        m >>= trailing;
        e += trailing;
        const auto signed_m{ static_cast<std::int64_t>(m) };
        return Parts{ (bits >> 63) != 0u ? -signed_m : signed_m, e };
    }

    void raise_power(const double coordinate)
    {
        const Parts p{ parts(coordinate) };
        if (p.mantissa != 0 && -p.exponent > power) {
            power = -p.exponent;
        }
    }

    [[nodiscard]] WideInt scaled(const double coordinate) const
    {
        const Parts p{ parts(coordinate) };
        return p.mantissa == 0 ? WideInt{} : WideInt::from_shifted(p.mantissa, p.exponent + power);
    }

    void collect_triangles(const Poly& poly)
    {
        used_.assign(poly.vertices.size(), 0);
        for (std::size_t f{ 0 }; f < poly.faces.count(); ++f) {
            const std::span<const int> face{ poly.faces.face(f) };
            const std::size_t count{ face.size() };
            for (std::size_t q{ 1 }; q + 1 < count; ++q) {
                triangles.push_back(Triangle{ face[0], face[q], face[q + 1] });
                used_[static_cast<std::size_t>(face[0])] = 1;
                used_[static_cast<std::size_t>(face[q])] = 1;
                used_[static_cast<std::size_t>(face[q + 1])] = 1;
            }
        }
    }

    void find_power(const Poly& poly)
    {
        power = 0;
        raise_power(centre.x);
        raise_power(centre.y);
        raise_power(centre.z);
        for (std::size_t v{ 0 }; v < poly.vertices.size(); ++v) {
            if (used_[v] != 0) {
                raise_power(poly.vertices[v].x);
                raise_power(poly.vertices[v].y);
                raise_power(poly.vertices[v].z);
            }
        }
    }

    /// @brief The scaled coordinate as an int64 when it is below 2^62 in magnitude.
    [[nodiscard]] bool small(const double coordinate, std::int64_t& out) const
    {
        const Parts p{ parts(coordinate) };
        if (p.mantissa == 0) {
            out = 0;
            return true;
        }
        const std::uint64_t magnitude{ static_cast<std::uint64_t>(p.mantissa < 0 ? -p.mantissa : p.mantissa) };
        const int shift{ p.exponent + power };
        if (static_cast<int>(std::bit_width(magnitude)) + shift > 62) {
            return false;
        }
        out = p.mantissa * (std::int64_t{ 1 } << shift);
        return true;
    }

    /// @brief The determinants with Int256 when every scaled coordinate is below 2^62. @return false when one is not
    bool compute_small(const Poly& poly)
    {
        std::array<std::int64_t, 3> g{};
        if (!small(centre.x, g[0]) || !small(centre.y, g[1]) || !small(centre.z, g[2])) {
            return false;
        }
        std::vector<std::array<std::int64_t, 3>> points(poly.vertices.size());
        for (std::size_t v{ 0 }; v < poly.vertices.size(); ++v) {
            if (used_[v] == 0) {
                continue;
            }
            const Vec3& q{ poly.vertices[v] };
            if (!small(q.x, points[v][0]) || !small(q.y, points[v][1]) || !small(q.z, points[v][2])) {
                return false;
            }
        }
        for (const Triangle& t : triangles) {
            const std::array<std::int64_t, 3>& a{ points[static_cast<std::size_t>(t.a)] };
            const std::array<std::int64_t, 3>& b{ points[static_cast<std::size_t>(t.b)] };
            const std::array<std::int64_t, 3>& c{ points[static_cast<std::size_t>(t.c)] };
            // x = a - g, y = b - a, z = c - a; det = x . (y x z): the same integers as compute_wide()
            const std::int64_t x0{ a[0] - g[0] };
            const std::int64_t x1{ a[1] - g[1] };
            const std::int64_t x2{ a[2] - g[2] };
            const std::int64_t y0{ b[0] - a[0] };
            const std::int64_t y1{ b[1] - a[1] };
            const std::int64_t y2{ b[2] - a[2] };
            const std::int64_t z0{ c[0] - a[0] };
            const std::int64_t z1{ c[1] - a[1] };
            const std::int64_t z2{ c[2] - a[2] };
            const Wide128 cross0{ Wide128::cross(y1, z2, y2, z1) };
            const Wide128 cross1{ Wide128::cross(y2, z0, y0, z2) };
            const Wide128 cross2{ Wide128::cross(y0, z1, y1, z0) };
            Int256 det{};
            det.add_product(x0, cross0.high, cross0.low);
            det.add_product(x1, cross1.high, cross1.low);
            det.add_product(x2, cross2.high, cross2.low);
            const WideInt magnitude{ det.magnitude() };
            determinants.push_back(magnitude);
            total = total + magnitude;
        }
        return true;
    }

    void compute(const Poly& poly)
    {
        if (compute_small(poly)) {
            return;
        }
        determinants.clear();
        total = WideInt{};
        compute_wide(poly);
    }

    void compute_wide(const Poly& poly)
    {
        const WideInt gx{ scaled(centre.x) };
        const WideInt gy{ scaled(centre.y) };
        const WideInt gz{ scaled(centre.z) };
        std::vector<std::array<WideInt, 3>> points(poly.vertices.size());
        for (std::size_t v{ 0 }; v < poly.vertices.size(); ++v) {
            if (used_[v] != 0) {
                points[v] = { scaled(poly.vertices[v].x), scaled(poly.vertices[v].y), scaled(poly.vertices[v].z) };
            }
        }
        for (const Triangle& t : triangles) {
            const std::array<WideInt, 3>& a{ points[static_cast<std::size_t>(t.a)] };
            const std::array<WideInt, 3>& b{ points[static_cast<std::size_t>(t.b)] };
            const std::array<WideInt, 3>& c{ points[static_cast<std::size_t>(t.c)] };
            // x = a - g, y = b - a, z = c - a; det = x . (y x z)
            const WideInt x0{ a[0] - gx };
            const WideInt x1{ a[1] - gy };
            const WideInt x2{ a[2] - gz };
            const WideInt y0{ b[0] - a[0] };
            const WideInt y1{ b[1] - a[1] };
            const WideInt y2{ b[2] - a[2] };
            const WideInt z0{ c[0] - a[0] };
            const WideInt z1{ c[1] - a[1] };
            const WideInt z2{ c[2] - a[2] };
            const WideInt cross0{ y1 * z2 - y2 * z1 };
            const WideInt cross1{ y2 * z0 - y0 * z2 };
            const WideInt cross2{ y0 * z1 - y1 * z0 };
            const WideInt det{ x0 * cross0 + x1 * cross1 + x2 * cross2 };
            const WideInt magnitude{ det.magnitude() };
            determinants.push_back(magnitude);
            total = total + magnitude;
        }
    }
};

namespace {

/// @brief Per-thread buffers of clip(), reused from call to call: allocating them per call dominated the cost of a cut
/// (measured 2026-11-12 on L24: about 5 us per cut before the reuse).
struct ClipScratch {
    std::vector<double> values{};      ///< normal . v - offset per old vertex.
    std::vector<int> classes{};        ///< -1 inside, 0 ON, +1 outside per old vertex.
    std::vector<int> kept{};           ///< New index per old vertex, -1 when not kept yet.
    std::vector<int> edges{};          ///< New index per edge (low * count + high), -1 when not cut yet.
    std::vector<Vec3> new_vertices{};  ///< The vertices of the result, in creation order.
    std::vector<char> on_flags{};      ///< Per new vertex: a cut point or a snapped ON vertex.
    std::vector<int> raw{};            ///< The kept and cut indices of one face.
    std::vector<int> cycle{};          ///< One face without consecutive repeats.
    std::vector<int> on_vertices{};    ///< The ON vertices of one face.
    std::vector<char> in_cap{};        ///< Per new vertex: on the cap polygon.
    std::vector<int> cap{};            ///< The cap vertices.
    std::vector<int> remap{};          ///< Result index per new vertex.
    FaceList faces{};                  ///< The faces of the result in new indices.
};

/// @brief The working state of one clip (builder3d.clip and its closures keep and cut).
class ClipWork {
public:
    ClipWork(const Poly& source, const Plane& clip_plane, ClipScratch& buffers)
        : poly{ source }
        , plane{ clip_plane }
        , scratch{ buffers }
    {
        const std::size_t count{ source.vertices.size() };
        scratch.kept.assign(count, -1);
        scratch.edges.assign(count * count, -1);
        scratch.new_vertices.clear();
        scratch.on_flags.clear();
    }

    /// @brief The new index of an old vertex; an ON vertex is snapped onto the plane (keep).
    int keep(const int old_index)
    {
        const auto old{ static_cast<std::size_t>(old_index) };
        if (scratch.kept[old] >= 0) {
            return scratch.kept[old];
        }
        Vec3 point{ poly.vertices[old] };
        const bool on_plane{ scratch.classes[old] == 0 };
        if (on_plane) {
            const double value{ scratch.values[old] };
            point = Vec3{ point.x - value * plane.normal.x, point.y - value * plane.normal.y, point.z - value * plane.normal.z };
        }
        scratch.kept[old] = add_vertex(point, on_plane);
        return scratch.kept[old];
    }

    /// @brief The new index of the cut point of edge (i, j); the first caller's direction decides the point (cut).
    int cut(const int i, const int j)
    {
        const std::size_t count{ poly.vertices.size() };
        const auto low{ static_cast<std::size_t>(std::min(i, j)) };
        const auto high{ static_cast<std::size_t>(std::max(i, j)) };
        const std::size_t key{ low * count + high };
        if (scratch.edges[key] >= 0) {
            return scratch.edges[key];
        }
        const auto ii{ static_cast<std::size_t>(i) };
        const auto jj{ static_cast<std::size_t>(j) };
        const double s{ scratch.values[ii] / (scratch.values[ii] - scratch.values[jj]) };
        const Vec3& a{ poly.vertices[ii] };
        const Vec3& b{ poly.vertices[jj] };
        const Vec3 point{ a.x + s * (b.x - a.x), a.y + s * (b.y - a.y), a.z + s * (b.z - a.z) };
        scratch.edges[key] = add_vertex(point, true);
        return scratch.edges[key];
    }

    /// @brief The kept part of one face into scratch.cycle (consecutive repeats removed), its ON vertices into
    /// scratch.on_vertices.
    void cut_face(const std::span<const int> face)
    {
        scratch.raw.clear();
        const std::size_t m{ face.size() };
        for (std::size_t q{ 0 }; q < m; ++q) {
            const int i{ face[q] };
            const int j{ face[(q + 1) % m] };
            const int class_i{ scratch.classes[static_cast<std::size_t>(i)] };
            const int class_j{ scratch.classes[static_cast<std::size_t>(j)] };
            if (class_i <= 0) {
                scratch.raw.push_back(keep(i));
            }
            if (class_i * class_j < 0) {
                scratch.raw.push_back(cut(i, j));
            }
        }
        scratch.cycle.clear();
        for (const int x : scratch.raw) {
            if (scratch.cycle.empty() || scratch.cycle.back() != x) {
                scratch.cycle.push_back(x);
            }
        }
        while (scratch.cycle.size() > 1 && scratch.cycle.front() == scratch.cycle.back()) {
            scratch.cycle.pop_back();
        }
        scratch.on_vertices.clear();
        for (const int x : scratch.cycle) {
            if (scratch.on_flags[static_cast<std::size_t>(x)] != 0) {
                scratch.on_vertices.push_back(x);
            }
        }
    }

private:
    const Poly& poly;     ///< The polyhedron being clipped.
    const Plane& plane;   ///< The clip plane.
    ClipScratch& scratch; ///< The buffers of this thread.

    int add_vertex(const Vec3& point, const bool on_plane)
    {
        scratch.new_vertices.push_back(point);
        scratch.on_flags.push_back(on_plane ? 1 : 0);
        return static_cast<int>(scratch.new_vertices.size()) - 1;
    }
};

} // namespace

// ------------------------------------------------------------------------------------------------ VectorOps

double VectorOps::dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 VectorOps::cross(const Vec3& a, const Vec3& b)
{
    return Vec3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

Vec3 VectorOps::subtract(const Vec3& a, const Vec3& b)
{
    return Vec3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

// ------------------------------------------------------------------------------------------------ Poly

Poly Poly::box(const Vec3& lo, const Vec3& hi)
{
    Poly result{};
    result.vertices = { { lo.x, lo.y, lo.z }, { hi.x, lo.y, lo.z }, { hi.x, hi.y, lo.z }, { lo.x, hi.y, lo.z },
                        { lo.x, lo.y, hi.z }, { hi.x, lo.y, hi.z }, { hi.x, hi.y, hi.z }, { lo.x, hi.y, hi.z } };
    result.faces.add({ 0, 3, 2, 1 });
    result.faces.add({ 4, 5, 6, 7 });
    result.faces.add({ 0, 1, 5, 4 });
    result.faces.add({ 2, 3, 7, 6 });
    result.faces.add({ 1, 2, 6, 5 });
    result.faces.add({ 0, 4, 7, 3 });
    return result;
}

Poly Poly::prism(const std::vector<Vec3>& polygon, const Vec3& vec)
{
    const int k{ static_cast<int>(polygon.size()) };
    Poly result{};
    result.vertices = polygon;
    for (const Vec3& p : polygon) {
        result.vertices.push_back(Vec3{ p.x + vec.x, p.y + vec.y, p.z + vec.z });
    }
    std::vector<int> bottom(static_cast<std::size_t>(k));
    std::iota(bottom.begin(), bottom.end(), 0);
    std::vector<int> top(static_cast<std::size_t>(k));
    std::iota(top.begin(), top.end(), k);
    result.faces.add(bottom);
    result.faces.add(top);
    for (int q{ 0 }; q < k; ++q) {
        const int next{ (q + 1) % k };
        result.faces.add({ q, next, k + next, k + q });
    }
    return result;
}

Vec3 Poly::interior() const
{
    double sx{ 0.0 };
    double sy{ 0.0 };
    double sz{ 0.0 };
    for (const Vec3& p : vertices) {
        sx += p.x;
        sy += p.y;
        sz += p.z;
    }
    const auto n{ static_cast<double>(vertices.size()) };
    return Vec3{ sx / n, sy / n, sz / n };
}

const FanDeterminants& Poly::fan() const
{
    if (!fan_) {
        fan_ = std::make_shared<const FanDeterminants>(*this);
    }
    return *fan_;
}

double Poly::volume() const
{
    if (volume_) {
        return *volume_;
    }
    const FanDeterminants& f{ fan() };
    volume_ = f.weight(f.total);
    return *volume_;
}

VolumeMoment Poly::moment() const
{
    if (moment_) {
        return *moment_;
    }
    const FanDeterminants& fan_data{ fan() };
    const Vec3& g{ fan_data.centre };
    Vec3 m{ 0.0, 0.0, 0.0 };
    for (std::size_t t{ 0 }; t < fan_data.triangles.size(); ++t) {
        const double w{ fan_data.weight(fan_data.determinants[t]) };
        const Vec3& a{ vertices[static_cast<std::size_t>(fan_data.triangles[t].a)] };
        const Vec3& b{ vertices[static_cast<std::size_t>(fan_data.triangles[t].b)] };
        const Vec3& c{ vertices[static_cast<std::size_t>(fan_data.triangles[t].c)] };
        // m[k] += w * (g[k] + a[k] + b[k] + c[k]) / 4.0, in Python's order
        m.x += w * (g.x + a.x + b.x + c.x) / 4.0;
        m.y += w * (g.y + a.y + b.y + c.y) / 4.0;
        m.z += w * (g.z + a.z + b.z + c.z) / 4.0;
    }
    const double total_volume{ fan_data.weight(fan_data.total) };
    moment_ = VolumeMoment{ total_volume, m };
    volume_ = total_volume; // the same rounded sum as volume()
    return *moment_;
}

std::vector<Plane> Poly::halfspaces() const
{
    const Vec3 g{ interior() };
    std::vector<Plane> out{};
    for (std::size_t f{ 0 }; f < faces.count(); ++f) {
        const std::span<const int> face{ faces.face(f) };
        const Vec3& p0{ vertices[static_cast<std::size_t>(face[0])] };
        Vec3 n{ 0.0, 0.0, 0.0 };
        for (std::size_t q{ 1 }; q + 1 < face.size(); ++q) {
            const Vec3 edge_a{ VectorOps::subtract(vertices[static_cast<std::size_t>(face[q])], p0) };
            const Vec3 edge_b{ VectorOps::subtract(vertices[static_cast<std::size_t>(face[q + 1])], p0) };
            const Vec3 c{ VectorOps::cross(edge_a, edge_b) };
            n = Vec3{ n.x + c.x, n.y + c.y, n.z + c.z };
        }
        const double ln{ std::sqrt(VectorOps::dot(n, n)) };
        if (ln == 0.0) {
            continue;
        }
        n = Vec3{ n.x / ln, n.y / ln, n.z / ln };
        const Vec3 to_centre{ VectorOps::subtract(g, p0) };
        if (VectorOps::dot(n, to_centre) > 0) {
            n = Vec3{ -n.x, -n.y, -n.z };
        }
        out.push_back(Plane{ n, VectorOps::dot(n, p0) });
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ PolyClipper

std::optional<Poly> PolyClipper::clip(const Poly& poly, const Plane& plane)
{
    thread_local ClipScratch scratch{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables): per-thread reuse, see ClipScratch
    const double tol{ EPS_ON * std::max(1.0, std::fabs(plane.offset)) };
    const std::size_t count{ poly.vertices.size() };
    scratch.values.resize(count);
    scratch.classes.resize(count);
    bool all_not_inside{ true };
    bool all_not_outside{ true };
    for (std::size_t i{ 0 }; i < count; ++i) {
        const double value{ VectorOps::dot(plane.normal, poly.vertices[i]) - plane.offset };
        const int cls{ std::fabs(value) <= tol ? 0 : (value > 0 ? 1 : -1) };
        scratch.values[i] = value;
        scratch.classes[i] = cls;
        all_not_inside = all_not_inside && cls >= 0;
        all_not_outside = all_not_outside && cls <= 0;
    }
    if (all_not_inside) {
        return std::nullopt;
    }
    if (all_not_outside) {
        return poly;
    }
    ClipWork work{ poly, plane, scratch };
    scratch.faces.clear();
    scratch.in_cap.clear();
    for (std::size_t f{ 0 }; f < poly.faces.count(); ++f) {
        work.cut_face(poly.faces.face(f));
        if (scratch.cycle.size() >= 3) {
            scratch.faces.add(scratch.cycle);
        }
        if (scratch.on_vertices.size() >= 2) {
            scratch.in_cap.resize(scratch.new_vertices.size(), 0);
            for (const int x : scratch.on_vertices) {
                scratch.in_cap[static_cast<std::size_t>(x)] = 1;
            }
        }
    }
    scratch.cap.clear();
    for (std::size_t x{ 0 }; x < scratch.in_cap.size(); ++x) {
        if (scratch.in_cap[x] != 0) {
            scratch.cap.push_back(static_cast<int>(x));
        }
    }
    if (scratch.cap.size() >= 3) {
        const std::vector<int> ordered{ order_cycle(scratch.new_vertices, scratch.cap, plane.normal) };
        scratch.faces.add(ordered);
    }
    scratch.remap.assign(scratch.new_vertices.size(), -1);
    for (const int x : scratch.faces.all()) {
        scratch.remap[static_cast<std::size_t>(x)] = 0;
    }
    Poly result{};
    for (std::size_t x{ 0 }; x < scratch.remap.size(); ++x) {
        if (scratch.remap[x] == 0) {
            scratch.remap[x] = static_cast<int>(result.vertices.size());
            result.vertices.push_back(scratch.new_vertices[x]);
        }
    }
    for (std::size_t f{ 0 }; f < scratch.faces.count(); ++f) {
        scratch.cycle.clear();
        for (const int x : scratch.faces.face(f)) {
            scratch.cycle.push_back(scratch.remap[static_cast<std::size_t>(x)]);
        }
        result.faces.add(scratch.cycle);
    }
    return result;
}

std::vector<int> PolyClipper::order_cycle(const std::vector<Vec3>& vertices, const std::vector<int>& indices, const Vec3& normal)
{
    const std::size_t count{ indices.size() };
    double sx{ 0.0 };
    double sy{ 0.0 };
    double sz{ 0.0 };
    for (const int index : indices) {
        const Vec3& p{ vertices[static_cast<std::size_t>(index)] };
        sx += p.x;
        sy += p.y;
        sz += p.z;
    }
    const auto n{ static_cast<double>(count) };
    const Vec3 g{ sx / n, sy / n, sz / n };
    const Vec3 axis{ std::fabs(normal.x) < 0.9 ? Vec3{ 1.0, 0.0, 0.0 } : Vec3{ 0.0, 1.0, 0.0 } };
    const Vec3 u{ VectorOps::cross(normal, axis) };
    const Vec3 w{ VectorOps::cross(normal, u) };
    std::vector<double> angles(count);
    for (std::size_t k{ 0 }; k < count; ++k) {
        const Vec3 offset{ VectorOps::subtract(vertices[static_cast<std::size_t>(indices[k])], g) };
        angles[k] = std::atan2(VectorOps::dot(offset, w), VectorOps::dot(offset, u));
    }
    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), std::size_t{ 0 });
    // Python's sorted() is stable: equal angles keep their order
    std::stable_sort(order.begin(), order.end(), [&angles](const std::size_t a, const std::size_t b) { return angles[a] < angles[b]; });
    std::vector<int> result{};
    for (const std::size_t k : order) {
        result.push_back(indices[k]);
    }
    return result;
}

PolyClipper::Side PolyClipper::side(const Poly& poly, const Plane& plane)
{
    // the classes of clip(); the flipped plane gives exactly the negated values (IEEE rounding is symmetric)
    const double tol{ EPS_ON * std::max(1.0, std::fabs(plane.offset)) };
    bool all_not_inside{ true };
    bool all_not_outside{ true };
    for (const Vec3& p : poly.vertices) {
        const double value{ VectorOps::dot(plane.normal, p) - plane.offset };
        const int cls{ std::fabs(value) <= tol ? 0 : (value > 0 ? 1 : -1) };
        all_not_inside = all_not_inside && cls >= 0;
        all_not_outside = all_not_outside && cls <= 0;
    }
    if (all_not_inside && all_not_outside) {
        return Side::on; // clip gives nothing for the plane and for the flipped plane
    }
    if (all_not_inside) {
        return Side::outside;
    }
    if (all_not_outside) {
        return Side::inside;
    }
    return Side::cut;
}

void PolyClipper::split(const Poly& poly, const Plane& plane, std::vector<DiagEvent>& events, std::optional<Poly>& inside,
                        std::optional<Poly>& outside)
{
    const Plane flipped{ Vec3{ -plane.normal.x, -plane.normal.y, -plane.normal.z }, -plane.offset };
    inside = clip(poly, plane);
    outside = clip(poly, flipped);
    const double v{ poly.volume() };
    const double vl{ inside ? inside->volume() : 0.0 };
    const double vh{ outside ? outside->volume() : 0.0 };
    const double defect{ std::fabs(v - vl - vh) };
    events.push_back(DiagEvent{ true, defect });
}

void PolyClipper::subtract_convex(const std::vector<Poly>& pieces, const std::vector<Plane>& planes, std::vector<DiagEvent>& events,
                                  std::vector<Poly>& inside, std::vector<Poly>& outside)
{
    for (const Poly& piece : pieces) {
        std::optional<Poly> current{ piece };
        for (const Plane& plane : planes) {
            const Side where{ side(*current, plane) };
            if (where == Side::cut) {
                std::optional<Poly> low{};
                std::optional<Poly> high{};
                split(*current, plane, events, low, high);
                if (high) {
                    outside.push_back(std::move(*high));
                }
                current = std::move(low);
            } else {
                // split() without the copies: the piece is kept whole on one side, or dropped when it lies on the plane;
                // the defect |(v - vl) - vh| has the same operands as in split()
                const double v{ current->volume() };
                const double vl{ where == Side::inside ? v : 0.0 };
                const double vh{ where == Side::outside ? v : 0.0 };
                events.push_back(DiagEvent{ true, std::fabs(v - vl - vh) });
                if (where == Side::outside) {
                    outside.push_back(std::move(*current));
                }
                if (where != Side::inside) {
                    current.reset();
                }
            }
            if (!current) {
                break;
            }
        }
        if (current) {
            inside.push_back(std::move(*current));
        }
    }
}

double PolyClipper::checked_volume(const Poly& poly)
{
    const double v{ poly.volume() };
    if (!std::isfinite(v) || v < 0.0) {
        throw std::domain_error("non-finite or negative volume " + std::to_string(v));
    }
    return v;
}

void PolyClipper::note_small(const double volume, std::vector<DiagEvent>& events)
{
    if (volume <= SMALL) {
        events.push_back(DiagEvent{ false, volume });
    }
}

} // namespace fluidx3d::experimental::wall_builder
