/// @file passes.cpp
/// @brief The producer and donor passes, ported from terrain_container.py and builder3d.py in Python's operation order.
/// Built with FLUIDX3D_PRECISE_FLOAT: the operation order below is the order of the Python code, and must stay so.

#include "passes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fluidx3d::experimental::wall_builder {

namespace {

/// @brief An integer box [lo, hi) of cells (terrain_container.py facet_box, swept, positive_overlap).
struct IntBox {
    std::array<int, 3> lo{}; ///< Lower corner.
    std::array<int, 3> hi{}; ///< Upper corner.
};

/// @brief The task-103 integer box test of terrain_container.py.
class BoxTest {
public:
    /// @brief The cell box of a facet at an image (facet_box).
    [[nodiscard]] static IntBox facet_box(const Facet& g, const std::array<int, 2>& image)
    {
        const std::array<int, 3> lo{ g.cell[0] + image[0], g.cell[1] + image[1], g.cell[2] };
        return IntBox{ lo, { lo[0] + 1, lo[1] + 1, lo[2] + 1 } };
    }

    /// @brief The box swept by an integer vector (swept).
    [[nodiscard]] static IntBox swept(const IntBox& box, const std::array<int, 3>& v)
    {
        IntBox out{};
        for (std::size_t a{ 0 }; a < 3; ++a) {
            out.lo[a] = box.lo[a] + std::min(0, v[a]);
            out.hi[a] = box.hi[a] + std::max(0, v[a]);
        }
        return out;
    }

    /// @brief True when the boxes share a positive volume (positive_overlap).
    [[nodiscard]] static bool positive_overlap(const IntBox& a, const IntBox& b)
    {
        for (std::size_t k{ 0 }; k < 3; ++k) {
            if (!(a.lo[k] < b.hi[k] && b.lo[k] < a.hi[k])) {
                return false;
            }
        }
        return true;
    }
};

std::array<int, 3> as_array(const Direction& c)
{
    return { c.x, c.y, c.z };
}

Vec3 as_vec(const Direction& c)
{
    return Vec3{ static_cast<double>(c.x), static_cast<double>(c.y), static_cast<double>(c.z) };
}

Dyadic dot_exact(const Direction& c, const std::array<Dyadic, 3>& n)
{
    return Dyadic::from_integer(c.x) * n[0] + Dyadic::from_integer(c.y) * n[1] + Dyadic::from_integer(c.z) * n[2];
}

/// @brief The order plane sigma_a <= sigma_b (or its mirror) of two non-parallel planes, as builder3d.py builds it.
Plane order_plane(const Plane& first, const Plane& second, const Vec3& c)
{
    const double c_first{ VectorOps::dot(first.normal, c) };
    const double c_second{ VectorOps::dot(second.normal, c) };
    const Vec3 n{ first.normal.x / c_first - second.normal.x / c_second, first.normal.y / c_first - second.normal.y / c_second,
                  first.normal.z / c_first - second.normal.z / c_second };
    const double ln{ std::sqrt(VectorOps::dot(n, n)) };
    const double offset{ (first.offset / c_first - second.offset / c_second) / ln };
    return Plane{ Vec3{ n.x / ln, n.y / ln, n.z / ln }, offset };
}

} // namespace

// ------------------------------------------------------------------------------------------------ ExactOrder

bool ExactOrder::parallel(const Facet& a, const Facet& b)
{
    const std::array<Dyadic, 3>& n1{ a.n_raw };
    const std::array<Dyadic, 3>& n2{ b.n_raw };
    const Dyadic x{ n1[1] * n2[2] - n1[2] * n2[1] };
    const Dyadic y{ n1[2] * n2[0] - n1[0] * n2[2] };
    const Dyadic z{ n1[0] * n2[1] - n1[1] * n2[0] };
    return x.sign() == 0 && y.sign() == 0 && z.sign() == 0;
}

int ExactOrder::sigma_difference(const Facet& g1, const Facet& g2, const Direction& c)
{
    const Dyadic c1{ dot_exact(c, g1.n_raw) };
    const Dyadic c2{ dot_exact(c, g2.n_raw) };
    if (c1.sign() == 0 || c2.sign() == 0) {
        throw std::domain_error("exact_sigma_difference: n . c is zero for " + g1.key.python_text() + " or " + g2.key.python_text());
    }
    // d2/c2 - d1/c1 = (d2 c1 - d1 c2) / (c1 c2)
    const Dyadic numerator{ g2.d_raw * c1 - g1.d_raw * c2 };
    return numerator.sign() * c1.sign() * c2.sign();
}

// ------------------------------------------------------------------------------------------------ PassEngine

PassEngine::PassEngine(const TerrainGeometry& geometry)
    : geometry_{ geometry }
{
}

std::vector<UnitResult> PassEngine::producer(const Facet& facet) const
{
    const std::vector<Direction>& directions{ geometry_.record().directions };
    double min_x{ facet.poly[0].x };
    double min_y{ facet.poly[0].y };
    for (const Vec3& p : facet.poly) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
    }
    const auto i0{ static_cast<int>(std::floor(min_x)) };
    const auto j0{ static_cast<int>(std::floor(min_y)) };
    const std::array<int, 3> origin{ facet.cell };
    std::vector<Candidate> all{};
    geometry_.candidates(i0, j0, facet.cell[2], &facet.key, origin, all);
    const Facet in_frame{ geometry_.facet_in_frame(facet, origin) };
    std::vector<UnitResult> results{};
    for (std::size_t d{ 0 }; d < directions.size(); ++d) {
        const Direction& c{ directions[d] };
        if (c.x == 0 && c.y == 0 && c.z == 0) {
            continue;
        }
        UnitResult result{};
        producer_row(facet, in_frame, all, static_cast<int>(d), result);
        results.push_back(std::move(result));
    }
    return results;
}

void PassEngine::producer_row(const Facet& f, const Facet& in_frame, const std::vector<Candidate>& all, const int d, UnitResult& out) const
{
    const Direction& c{ geometry_.record().directions[static_cast<std::size_t>(d)] };
    if (f.signs[static_cast<std::size_t>(d)] <= 0) {
        return;
    }
    const double vi{ VectorOps::dot(f.area, as_vec(c)) };
    if (!(std::isfinite(vi) && vi > 0.0)) {
        throw std::domain_error("exact-positive row " + f.key.python_text() + " direction " + std::to_string(d) + " has FP64 vi " + std::to_string(vi));
    }
    out.vi.push_back(ViEntry{ f.key, d, vi });
    const IntBox outgoing{ BoxTest::swept(BoxTest::facet_box(f, { 0, 0 }), as_array(c)) };
    std::vector<const Candidate*> receivers{};
    for (const Candidate& candidate : all) {
        if (candidate.facet.signs[static_cast<std::size_t>(d)] >= 0) {
            continue;
        }
        const IntBox incoming{ BoxTest::swept(BoxTest::facet_box(candidate.facet, candidate.image), as_array(c)) };
        if (BoxTest::positive_overlap(outgoing, incoming)) {
            receivers.push_back(&candidate);
        }
    }
    std::vector<Poly> free_pieces{};
    std::vector<FirstOwner> owned{};
    partition(in_frame, c, receivers, out, free_pieces, owned);
    for (const Poly& piece : free_pieces) {
        add_h(f, d, piece, f.cell, out);
    }
    for (const FirstOwner& item : owned) {
        add_w(f, d, *receivers[static_cast<std::size_t>(item.owner)], item.piece, out);
    }
}

void PassEngine::partition(const Facet& f, const Direction& c, const std::vector<const Candidate*>& receivers, UnitResult& out,
                           std::vector<Poly>& free_pieces, std::vector<FirstOwner>& owned) const
{
    const Vec3 cv{ as_vec(c) };
    std::vector<OwnedPiece> state{ OwnedPiece{ Poly::prism(f.poly, cv), {} } };
    for (std::size_t r{ 0 }; r < receivers.size(); ++r) {
        const Facet& g{ receivers[r]->facet };
        std::vector<Plane> planes{ Poly::prism(g.poly, cv).halfspaces() };
        if (ExactOrder::parallel(g, f)) {
            if (ExactOrder::sigma_difference(g, f, c) >= 0) {
                continue; // sigma_a >= sigma_b everywhere: never after the producer
            }
        } else {
            planes.push_back(order_plane(g.plane, f.plane, cv));
        }
        std::vector<OwnedPiece> next{};
        for (const OwnedPiece& item : state) {
            std::vector<Poly> inside{};
            std::vector<Poly> outside{};
            PolyClipper::subtract_convex({ item.piece }, planes, out.events, inside, outside);
            for (Poly& q : inside) {
                std::vector<int> owners{ item.owners };
                owners.push_back(static_cast<int>(r));
                next.push_back(OwnedPiece{ std::move(q), std::move(owners) });
            }
            for (Poly& q : outside) {
                next.push_back(OwnedPiece{ std::move(q), item.owners });
            }
        }
        state = std::move(next);
    }
    for (const OwnedPiece& item : state) {
        if (item.owners.empty()) {
            free_pieces.push_back(item.piece);
        } else {
            first_owner(item, c, receivers, out, owned);
        }
    }
}

void PassEngine::first_owner(const OwnedPiece& start, const Direction& c, const std::vector<const Candidate*>& receivers, UnitResult& out,
                             std::vector<FirstOwner>& owned) const
{
    // builder3d.py recurses on owners[1:]; a stack with the losing pieces pushed in reverse gives the same depth-first order
    const Vec3 cv{ as_vec(c) };
    std::vector<OwnedPiece> stack{ start };
    while (!stack.empty()) { // ends: every pop removes one owner or settles the piece
        const OwnedPiece item{ std::move(stack.back()) };
        stack.pop_back();
        if (item.owners.size() == 1) {
            owned.push_back(FirstOwner{ item.piece, item.owners[0] });
            continue;
        }
        const Facet& g{ receivers[static_cast<std::size_t>(item.owners[0])]->facet };
        const std::vector<int> rest(item.owners.begin() + 1, item.owners.end());
        std::vector<Plane> planes{};
        bool g_never_first{ false };
        for (const int other : rest) {
            const Facet& g2{ receivers[static_cast<std::size_t>(other)]->facet };
            if (!ExactOrder::parallel(g, g2)) {
                planes.push_back(order_plane(g2.plane, g.plane, cv));
                continue;
            }
            const int s{ ExactOrder::sigma_difference(g, g2, c) };
            if (s == 0) {
                throw std::runtime_error("R4: coincident receiver planes " + g.key.python_text() + " " + g2.key.python_text() + " over an overlap");
            }
            if (s < 0) {
                g_never_first = true; // g2 is nearer the producer on every ray
                break;
            }
        }
        if (g_never_first) {
            stack.push_back(OwnedPiece{ item.piece, rest });
            continue;
        }
        std::vector<Poly> win{};
        std::vector<Poly> lose{};
        if (planes.empty()) {
            win.push_back(item.piece);
        } else {
            PolyClipper::subtract_convex({ item.piece }, planes, out.events, win, lose);
        }
        for (Poly& q : win) {
            owned.push_back(FirstOwner{ std::move(q), item.owners[0] });
        }
        for (auto it{ lose.rbegin() }; it != lose.rend(); ++it) {
            stack.push_back(OwnedPiece{ std::move(*it), rest });
        }
    }
}

void PassEngine::add_h(const Facet& f, const int d, const Poly& piece, const std::array<int, 3>& origin, UnitResult& out) const
{
    double min_x{ piece.vertices[0].x };
    double max_x{ min_x };
    double min_y{ piece.vertices[0].y };
    double max_y{ min_y };
    double min_z{ piece.vertices[0].z };
    double max_z{ min_z };
    for (const Vec3& q : piece.vertices) {
        min_x = std::min(min_x, q.x);
        max_x = std::max(max_x, q.x);
        min_y = std::min(min_y, q.y);
        max_y = std::max(max_y, q.y);
        min_z = std::min(min_z, q.z);
        max_z = std::max(max_z, q.z);
    }
    const auto x0{ static_cast<int>(std::floor(min_x)) };
    const auto x1{ static_cast<int>(std::ceil(max_x)) };
    const auto y0{ static_cast<int>(std::floor(min_y)) };
    const auto y1{ static_cast<int>(std::ceil(max_y)) };
    const auto z0{ static_cast<int>(std::floor(min_z)) };
    const auto z1{ static_cast<int>(std::ceil(max_z)) };
    for (int li{ x0 }; li < x1; ++li) {
        for (int lj{ y0 }; lj < y1; ++lj) {
            for (int lk{ z0 }; lk < z1; ++lk) {
                const Vec3 lo{ static_cast<double>(li), static_cast<double>(lj), static_cast<double>(lk) };
                const Vec3 hi{ static_cast<double>(li + 1), static_cast<double>(lj + 1), static_cast<double>(lk + 1) };
                std::optional<Poly> q{ piece };
                for (const Plane& plane : Poly::box(lo, hi).halfspaces()) {
                    if (q) {
                        q = PolyClipper::clip(*q, plane);
                    }
                }
                if (q) {
                    add_h_cell(f, d, *q, { li, lj, lk }, origin, out);
                }
            }
        }
    }
}

void PassEngine::add_h_cell(const Facet& f, const int d, const Poly& clipped, const std::array<int, 3>& corner, const std::array<int, 3>& origin,
                            UnitResult& out) const
{
    const TerrainRecord& record{ geometry_.record() };
    const int ci{ corner[0] + origin[0] };
    const int cj{ corner[1] + origin[1] };
    const int ck{ corner[2] + origin[2] };
    const double v{ PolyClipper::checked_volume(clipped) };
    PolyClipper::note_small(v, out.events);
    const std::array<int, 3> cell{ TerrainGeometry::wrap(ci, record.lx), TerrainGeometry::wrap(cj, record.ly), ck };
    const std::array<int, 2> image{ ci - cell[0], cj - cell[1] };
    auto h_entry{ std::find_if(out.h.begin(), out.h.end(), [&](const HEntry& e) { return e.cell == cell && e.image == image; }) };
    if (h_entry == out.h.end()) {
        out.h.push_back(HEntry{ f.key, d, cell, image, 0.0 + v });
    } else {
        h_entry->value = h_entry->value + v;
    }
    const VolumeMoment vm{ clipped.moment() };
    auto part{ std::find_if(out.h_moment.begin(), out.h_moment.end(), [&](const HMomentEntry& e) { return e.cell == cell; }) };
    if (part == out.h_moment.end()) {
        out.h_moment.push_back(HMomentEntry{ f.key, d, cell, { 0.0, 0.0, 0.0, 0.0 } });
        part = out.h_moment.end() - 1;
    }
    if (out.row_moment.empty()) {
        out.row_moment.push_back(RowMomentEntry{ f.key, d, { 0.0, 0.0, 0.0, 0.0 } });
    }
    std::array<double, 4>& row{ out.row_moment[0].value };
    part->value[0] += vm.volume;
    row[0] += vm.volume;
    const std::array<double, 3> moment{ vm.moment.x, vm.moment.y, vm.moment.z };
    for (std::size_t k{ 0 }; k < 3; ++k) {
        part->value[1 + k] += moment[k] - vm.volume * static_cast<double>(corner[k]);
        row[1 + k] += moment[k] + vm.volume * static_cast<double>(origin[k]); // the row moment stays global
    }
}

void PassEngine::add_w(const Facet& f, const int d, const Candidate& receiver, const Poly& piece, UnitResult& out) const
{
    const double v{ PolyClipper::checked_volume(piece) };
    PolyClipper::note_small(v, out.events);
    const FacetKey& gkey{ receiver.facet.key };
    auto entry{ std::find_if(out.w.begin(), out.w.end(), [&](const WEntry& e) { return e.gkey == gkey && e.image == receiver.image; }) };
    if (entry == out.w.end()) {
        out.w.push_back(WEntry{ f.key, d, gkey, receiver.image, 0.0 + v });
    } else {
        entry->value = entry->value + v;
    }
}

std::vector<UnitResult> PassEngine::donor(const TerrainCell& cell) const
{
    const std::vector<Direction>& directions{ geometry_.record().directions };
    std::vector<Candidate> all{};
    geometry_.candidates(cell.ijk[0], cell.ijk[1], cell.ijk[2], nullptr, cell.ijk, all);
    std::vector<UnitResult> results(directions.size());
    for (std::size_t d{ 0 }; d < directions.size(); ++d) {
        donor_state(cell, all, static_cast<int>(d), results[d]);
    }
    return results;
}

void PassEngine::donor_state(const TerrainCell& cell, const std::vector<Candidate>& all, const int d, UnitResult& out) const
{
    const Direction& c{ geometry_.record().directions[static_cast<std::size_t>(d)] };
    const std::array<int, 3> back{ -c.x, -c.y, -c.z };
    const Vec3 back_vec{ static_cast<double>(back[0]), static_cast<double>(back[1]), static_cast<double>(back[2]) };
    const IntBox cell_box{ cell.ijk, { cell.ijk[0] + 1, cell.ijk[1] + 1, cell.ijk[2] + 1 } };
    std::vector<Poly> pieces{ cell.pieces };
    for (const Candidate& candidate : all) {
        if (candidate.facet.signs[static_cast<std::size_t>(d)] >= 0) {
            continue;
        }
        const IntBox swept_box{ BoxTest::swept(BoxTest::facet_box(candidate.facet, candidate.image), back) };
        if (!BoxTest::positive_overlap(cell_box, swept_box)) {
            continue; // task 103: zero-measure contact, no volume
        }
        const Poly sweep{ Poly::prism(candidate.facet.poly, back_vec) };
        std::vector<Poly> inside{};
        std::vector<Poly> outside{};
        PolyClipper::subtract_convex(pieces, sweep.halfspaces(), out.events, inside, outside);
        pieces = std::move(outside);
        if (pieces.empty()) {
            break;
        }
    }
    double v{ 0.0 };
    std::array<double, 4> part{ 0.0, 0.0, 0.0, 0.0 };
    for (const Poly& p : pieces) {
        const double pv{ PolyClipper::checked_volume(p) };
        PolyClipper::note_small(pv, out.events);
        v += pv;
        const VolumeMoment vm{ p.moment() };
        part[0] += vm.volume;
        part[1] += vm.moment.x;
        part[2] += vm.moment.y;
        part[3] += vm.moment.z;
    }
    out.direct.push_back(DirectEntry{ cell.ijk, d, v, part });
}

} // namespace fluidx3d::experimental::wall_builder
