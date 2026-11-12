/// @file wall_tables.cpp
/// @brief The wall tables of terrain_container.py: arms, wall_list, sample_support_closure and gradient_map.
/// Built with FLUIDX3D_PRECISE_FLOAT: every expression keeps numpy's operation order; the dense algebra is numpy's OpenBLAS.

#include "wall_tables.hpp"

#include "dense_algebra.hpp"
#include "python_text.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace fluidx3d::experimental::wall_builder {

namespace {

constexpr int OWN_SHIFT{ 4 }; // shift index of (0, 0)

/// @brief One distinct neighbour cell of the gradient stencil: (d2, cell, shift, delta).
struct Neighbour {
    double d2{};
    std::int64_t cell{};
    int shift{};
    std::array<double, 3> delta{};
};

std::string cell_text(const std::array<int, 3>& cell)
{
    return "(" + std::to_string(cell[0]) + ", " + std::to_string(cell[1]) + ", " + std::to_string(cell[2]) + ")";
}

} // namespace

WallTables::WallTables(const TerrainGeometry& geometry, const VelocityTable& velocities, const MapAssembly& assembly)
    : geometry_{ geometry }
    , velocities_{ velocities }
    , assembly_{ assembly }
{
    kmin_ = std::numeric_limits<int>::max();
    kmax_ = std::numeric_limits<int>::min();
    for (const TerrainCell& cell : geometry.cells()) {
        kmin_ = std::min(kmin_, cell.ijk[2]);
        kmax_ = std::max(kmax_, cell.ijk[2]);
    }
}

// ------------------------------------------------------------------------------------------------ arms

ArmTables WallTables::arms(const AssembledMaps& m, const PermutedMaps& p) const
{
    const auto q{ static_cast<std::int64_t>(velocities_.directions.size()) };
    const std::vector<TerrainCell>& cells{ geometry_.cells() };
    ArmTables out{};
    // direct_arm: np.array(part[1:]) / part[0] - centroid_local[cell], per D entry in CSR order
    for (const std::int64_t column : p.d.indices) {
        const auto cell{ static_cast<std::size_t>(column / q) };
        const auto d{ static_cast<std::size_t>(column % q) };
        const std::array<double, 4>& part{ assembly_.passes().cell_results()[cell][d].direct[0].moment };
        const Vec3& local{ cells[cell].centroid_local };
        out.direct_arm.push_back(part[1] / part[0] - local.x);
        out.direct_arm.push_back(part[2] / part[0] - local.y);
        out.direct_arm.push_back(part[3] / part[0] - local.z);
    }
    std::vector<char> gather_cells(cells.size(), 0);
    for (std::int64_t row{ 0 }; row < p.g.rows; ++row) {
        const WallRow& origin{ m.rows[static_cast<std::size_t>(p.order[static_cast<std::size_t>(row)])] };
        const UnitResult* unit{ assembly_.producer_unit(origin.facet, origin.direction) };
        for (std::int64_t e{ p.g.offsets[static_cast<std::size_t>(row)] }; e < p.g.offsets[static_cast<std::size_t>(row) + 1u]; ++e) {
            const std::int64_t column{ p.g.indices[static_cast<std::size_t>(e)] };
            const auto cell{ static_cast<std::size_t>(column / q) };
            if (column % q != velocities_.opposites[static_cast<std::size_t>(origin.direction)]) {
                throw std::runtime_error("INVALID: a gather entry is not the opposite direction of its row");
            }
            const auto found{ std::find_if(unit->h_moment.begin(), unit->h_moment.end(), [&](const HMomentEntry& h) { return h.cell == cells[cell].ijk; }) };
            if (found == unit->h_moment.end()) {
                throw std::runtime_error("INVALID: no h_moment for a gather entry of cell " + cell_text(cells[cell].ijk));
            }
            const Vec3& local{ cells[cell].centroid_local };
            out.prism_arm.push_back(found->value[1] / found->value[0] - local.x);
            out.prism_arm.push_back(found->value[2] / found->value[0] - local.y);
            out.prism_arm.push_back(found->value[3] / found->value[0] - local.z);
            gather_cells[cell] = 1;
        }
    }
    for (std::size_t cell{ 0 }; cell < gather_cells.size(); ++cell) {
        if (gather_cells[cell] != 0) {
            out.wall_cells.push_back(static_cast<std::int64_t>(cell));
        }
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ wall list

WallList WallTables::wall_list(const PermutedMaps& p, const ArmTables& arms) const
{
    const TerrainRecord& record{ geometry_.record() };
    const auto q{ static_cast<std::int64_t>(velocities_.directions.size()) };
    const std::size_t cells{ geometry_.cells().size() };
    // per-cell flags in place of Python's sets: their sorted lists are the flagged indices in order
    std::vector<char> wall(cells, 0);
    for (const std::int64_t cell : arms.wall_cells) {
        wall[static_cast<std::size_t>(cell)] = 1;
    }
    std::vector<char> added(cells, 0);
    for (std::int64_t state{ 0 }; state < p.d.rows; ++state) {
        const std::int64_t cell{ state / q };
        const auto d{ static_cast<std::size_t>(state % q) };
        const std::int64_t a{ p.d.offsets[static_cast<std::size_t>(state)] };
        const std::int64_t b{ p.d.offsets[static_cast<std::size_t>(state) + 1u] };
        const Direction& c{ velocities_.directions[d] };
        const std::array<int, 3>& k{ geometry_.cells()[static_cast<std::size_t>(cell)].ijk };
        const std::int64_t expected{ assembly_.cell_index(TerrainGeometry::wrap(k[0] - c.x, record.lx), TerrainGeometry::wrap(k[1] - c.y, record.ly), k[2] - c.z) };
        const bool no_return{ p.r.offsets[static_cast<std::size_t>(state) + 1u] == p.r.offsets[static_cast<std::size_t>(state)] };
        const bool pure{ b - a == 1 && p.d.values[static_cast<std::size_t>(a)] == 1.0 && expected >= 0
                         && p.d.indices[static_cast<std::size_t>(a)] == expected * q + static_cast<std::int64_t>(d) && no_return };
        if (pure) {
            continue;
        }
        if (wall[static_cast<std::size_t>(cell)] == 0) {
            added[static_cast<std::size_t>(cell)] = 1;
        }
        for (std::int64_t e{ a }; e < b; ++e) {
            const auto donor{ static_cast<std::size_t>(p.d.indices[static_cast<std::size_t>(e)] / q) };
            if (wall[donor] == 0) {
                added[donor] = 1;
            }
        }
    }
    WallList out{};
    std::vector<char> all(cells, 0);
    for (std::size_t cell{ 0 }; cell < cells; ++cell) {
        all[cell] = static_cast<char>(wall[cell] != 0 || added[cell] != 0);
        if (added[cell] != 0) {
            out.added_253.push_back(static_cast<std::int64_t>(cell));
        }
    }
    // sample_support_closure: every S stencil cell needs a gradient
    std::vector<char> extra(cells, 0);
    bool any_extra{ false };
    for (const std::int64_t cell : p.s.indices) {
        if (all[static_cast<std::size_t>(cell)] == 0) {
            extra[static_cast<std::size_t>(cell)] = 1;
            any_extra = true;
        }
    }
    if (any_extra) {
        for (std::size_t e{ 0 }; e < p.d.indices.size(); ++e) {
            if (extra[static_cast<std::size_t>(p.d.indices[e] / q)] == 0) {
                continue;
            }
            for (std::size_t k{ 0 }; k < 3; ++k) {
                out.support_arm_max = std::max(out.support_arm_max, std::fabs(arms.direct_arm[3 * e + k]));
            }
        }
        if (!(out.support_arm_max <= ARM_ROUNDING)) {
            throw std::runtime_error("INVALID: a sample-support cell outside rule #253 has a direct arm " + PythonFloat::repr(out.support_arm_max)
                                     + " > 1e-12 (it is not a whole-cell donor)");
        }
    }
    for (std::size_t cell{ 0 }; cell < cells; ++cell) {
        if (extra[cell] != 0) {
            out.added_support.push_back(static_cast<std::int64_t>(cell));
        }
        if (all[cell] != 0 || extra[cell] != 0) {
            out.cells.push_back(static_cast<std::int64_t>(cell));
        }
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ gradient map

std::array<double, 3> WallTables::image_delta(const std::int64_t cell, const int shift, const std::int64_t a) const
{
    const TerrainRecord& record{ geometry_.record() };
    const Vec3& c{ geometry_.cells()[static_cast<std::size_t>(cell)].centroid };
    const Vec3& own{ geometry_.cells()[static_cast<std::size_t>(a)].centroid };
    const double sx{ static_cast<double>((shift / 3 - 1) * record.lx) };
    const double sy{ static_cast<double>((shift % 3 - 1) * record.ly) };
    // centroid[cell] + shifts[shift] - centroid[a]
    return { c.x + sx - own.x, c.y + sy - own.y, c.z + 0.0 - own.z };
}

void WallTables::window_points(const std::int64_t a, const int m, std::vector<ImagePoint>& out) const
{
    const TerrainRecord& record{ geometry_.record() };
    const std::array<int, 3>& own{ geometry_.cells()[static_cast<std::size_t>(a)].ijk };
    out.clear();
    for (int big_i{ own[0] - m }; big_i <= own[0] + m; ++big_i) {
        const int i{ TerrainGeometry::wrap(big_i, record.lx) };
        const int sa{ (big_i - i) / record.lx };
        if (sa < -1 || sa > 1) {
            continue; // only the 9 images of gradient_map exist
        }
        for (int big_j{ own[1] - m }; big_j <= own[1] + m; ++big_j) {
            const int j{ TerrainGeometry::wrap(big_j, record.ly) };
            const int sb{ (big_j - j) / record.ly };
            if (sb < -1 || sb > 1) {
                continue;
            }
            const int k_first{ std::max(own[2] - m, kmin_) };
            const int k_last{ std::min(own[2] + m, kmax_) };
            for (int k{ k_first }; k <= k_last; ++k) {
                const std::int64_t cell{ assembly_.cell_index(i, j, k) };
                if (cell < 0) {
                    continue;
                }
                const int shift{ (sa + 1) * 3 + (sb + 1) };
                const std::array<double, 3> delta{ image_delta(cell, shift, a) };
                out.push_back(ImagePoint{ delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2], cell, shift });
            }
        }
    }
}

std::vector<WallTables::ImagePoint> WallTables::nearest_points(const std::int64_t a) const
{
    // cKDTree.query(k=QUERY) over the 9 images: an index window of half width m holds every point with d2 < m^2 (each
    // centroid lies in its unit cell), so its QUERY nearest are exact once the QUERY-th has d2 < m^2
    const TerrainRecord& record{ geometry_.record() };
    const auto points{ static_cast<std::int64_t>(geometry_.cells().size()) * 9 };
    const auto wanted{ static_cast<std::size_t>(std::min<std::int64_t>(QUERY, points)) };
    const int whole{ std::max({ 3 * record.lx, 3 * record.ly, kmax_ - kmin_ + 1 }) };
    std::vector<ImagePoint> found{};
    for (int m{ 3 };; m *= 2) { // ends: the window covers every point once m >= whole
        window_points(a, m, found);
        const bool complete{ m >= whole };
        if (found.size() >= wanted) {
            std::nth_element(found.begin(), found.begin() + static_cast<std::ptrdiff_t>(wanted - 1), found.end(),
                             [](const ImagePoint& x, const ImagePoint& y) { return x.d2 < y.d2; });
            if (complete || found[wanted - 1].d2 < static_cast<double>(m) * static_cast<double>(m)) {
                found.resize(wanted);
                return found;
            }
        } else if (complete) {
            return found;
        }
    }
}

WallTables::CellStencil WallTables::stencil(const std::int64_t a) const
{
    const double inf{ std::numeric_limits<double>::infinity() };
    std::vector<Neighbour> nearest{};
    std::vector<std::int64_t> second_cell{};
    std::vector<double> second_d2{};
    const auto second_min{ [&](const std::int64_t cell, const double d2) {
        const auto at{ std::find(second_cell.begin(), second_cell.end(), cell) };
        if (at == second_cell.end()) {
            second_cell.push_back(cell);
            second_d2.push_back(d2);
            return;
        }
        double& current{ second_d2[static_cast<std::size_t>(at - second_cell.begin())] };
        current = std::min(current, d2);
    } };
    const auto second_of{ [&](const std::int64_t cell) {
        const auto at{ std::find(second_cell.begin(), second_cell.end(), cell) };
        return at == second_cell.end() ? inf : second_d2[static_cast<std::size_t>(at - second_cell.begin())];
    } };
    for (const ImagePoint& point : nearest_points(a)) {
        const std::array<double, 3> delta{ image_delta(point.cell, point.shift, a) };
        const double d2{ delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] };
        if (point.cell == a) {
            if (point.shift != OWN_SHIFT) {
                second_min(a, d2);
            }
            continue;
        }
        const auto known{ std::find_if(nearest.begin(), nearest.end(), [&](const Neighbour& n) { return n.cell == point.cell; }) };
        if (known == nearest.end()) {
            nearest.push_back(Neighbour{ d2, point.cell, point.shift, delta });
        } else if (d2 < known->d2 || (d2 == known->d2 && point.shift < known->shift)) {
            second_min(point.cell, known->d2);
            *known = Neighbour{ d2, point.cell, point.shift, delta };
        } else {
            second_min(point.cell, d2);
        }
    }
    std::sort(nearest.begin(), nearest.end(), [](const Neighbour& x, const Neighbour& y) {
        if (x.d2 != y.d2) {
            return x.d2 < y.d2;
        }
        return x.cell != y.cell ? x.cell < y.cell : x.shift < y.shift;
    });
    if (nearest.size() <= static_cast<std::size_t>(OTHERS)) {
        throw std::runtime_error("INVALID: the neighbour query is too short");
    }
    const double limit{ nearest[OTHERS - 1].d2 * (1.0 + TIE_RELATIVE) };
    std::size_t shell{ 0 };
    while (shell < nearest.size() && nearest[shell].d2 <= limit) {
        ++shell;
    }
    if (shell == nearest.size()) {
        throw std::runtime_error("INVALID: the neighbour query is too short for the tied shell");
    }
    bool second_inside{ second_of(a) <= limit };
    for (std::size_t n{ 0 }; n < shell; ++n) {
        second_inside = second_inside || second_of(nearest[n].cell) <= limit;
    }
    if (second_inside) {
        throw std::runtime_error("INVALID: a second periodic image of a stencil cell lies inside the stencil limit");
    }
    CellStencil out{};
    out.tie = shell > static_cast<std::size_t>(OTHERS);
    out.gap = (nearest[shell].d2 - nearest[shell - 1].d2) / nearest[shell - 1].d2;
    const int count{ static_cast<int>(shell) + 1 };
    Matrix basis{ count, 4, std::vector<double>(static_cast<std::size_t>(count) * 4u, 0.0) };
    basis.data[0] = 1.0; // the own cell: delta 0
    out.cells.push_back(a);
    for (std::size_t n{ 0 }; n < shell; ++n) {
        const std::size_t row{ n + 1 };
        basis.data[4 * row] = 1.0;
        basis.data[4 * row + 1] = nearest[n].delta[0];
        basis.data[4 * row + 2] = nearest[n].delta[1];
        basis.data[4 * row + 3] = nearest[n].delta[2];
        out.cells.push_back(nearest[n].cell);
    }
    const Matrix normal{ DenseAlgebra::gram(basis) };
    if (DenseAlgebra::rank(basis) < 4) {
        throw std::runtime_error("gradient stencil of cell " + cell_text(geometry_.cells()[static_cast<std::size_t>(a)].ijk) + ": rank < 4");
    }
    Matrix transposed{ 4, count, std::vector<double>(basis.data.size(), 0.0) };
    for (int r{ 0 }; r < count; ++r) {
        for (int c{ 0 }; c < 4; ++c) {
            transposed.data[static_cast<std::size_t>(c) * static_cast<std::size_t>(count) + static_cast<std::size_t>(r)] = basis.at(r, c);
        }
    }
    const Matrix solved{ DenseAlgebra::solve(normal, transposed) };
    Matrix coefficient{ 3, count, std::vector<double>(solved.data.begin() + count, solved.data.end()) };
    const Matrix reproduced{ DenseAlgebra::times(coefficient, basis) };
    double worst{ 0.0 };
    for (int axis{ 0 }; axis < 3; ++axis) {
        for (int c{ 0 }; c < 4; ++c) {
            const double expected{ c == axis + 1 ? 1.0 : 0.0 };
            worst = std::max(worst, std::fabs(reproduced.at(axis, c) - expected));
        }
    }
    out.reproduction = worst;
    out.condition = DenseAlgebra::condition(normal);
    out.weights = std::move(coefficient.data);
    return out;
}

GradientMap WallTables::gradient(const std::vector<std::int64_t>& wall_cells, const int threads) const
{
    const std::size_t count{ wall_cells.size() };
    std::vector<CellStencil> stencils(count);
    std::vector<std::string> errors(count);
    std::atomic<std::size_t> next{ 0 };
    const auto worker{ [&]() {
        for (std::size_t n{ next.fetch_add(1) }; n < count; n = next.fetch_add(1)) {
            try {
                stencils[n] = stencil(wall_cells[n]);
            } catch (const std::exception& error) { // kept per cell; the first in cell order is reported
                errors[n] = error.what();
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
    for (const std::string& error : errors) {
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
    }
    GradientMap out{};
    out.cells = wall_cells;
    out.offsets.push_back(0);
    out.stencil_size_min = std::numeric_limits<std::int64_t>::max();
    out.smallest_gap = std::numeric_limits<double>::infinity();
    out.reproduction_max = -1.0;
    out.condition_max = -1.0;
    for (const CellStencil& s : stencils) {
        const std::size_t size{ s.cells.size() };
        for (std::size_t axis{ 0 }; axis < 3; ++axis) {
            out.indices.insert(out.indices.end(), s.cells.begin(), s.cells.end());
            out.weights.insert(out.weights.end(), s.weights.begin() + static_cast<std::ptrdiff_t>(axis * size),
                               s.weights.begin() + static_cast<std::ptrdiff_t>((axis + 1) * size));
            out.offsets.push_back(static_cast<std::int64_t>(out.indices.size()));
        }
        out.reproduction_max = std::max(out.reproduction_max, s.reproduction);
        out.condition_max = std::max(out.condition_max, s.condition);
        out.stencil_size_min = std::min(out.stencil_size_min, static_cast<std::int64_t>(size));
        out.stencil_size_max = std::max(out.stencil_size_max, static_cast<std::int64_t>(size));
        out.ties += s.tie ? 1 : 0;
        out.smallest_gap = std::min(out.smallest_gap, s.gap);
    }
    return out;
}

} // namespace fluidx3d::experimental::wall_builder
