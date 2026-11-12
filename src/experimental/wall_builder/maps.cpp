/// @file maps.cpp
/// @brief The transport maps of the container, with scipy.sparse's construction and product rules.
/// Built with FLUIDX3D_PRECISE_FLOAT: every sum keeps the order of scipy's loops and numpy's elementwise operations.

#include "maps.hpp"

#include "parallel_for.hpp"
#include "python_text.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace fluidx3d::experimental::wall_builder {

namespace {

/// @brief One value and its first-insertion position, for Python's dict.get(key, 0.0) + v accumulation.
struct Accumulated {
    std::int64_t key{}; ///< The key (a cell index or a facet index).
    double value{};     ///< The sum, in insertion order.
};

/// @brief dict[key] = dict.get(key, 0.0) + v with first-insertion order (small lists, linear search).
void accumulate(std::vector<Accumulated>& sums, const std::int64_t key, const double v)
{
    for (Accumulated& entry : sums) {
        if (entry.key == key) {
            entry.value = entry.value + v;
            return;
        }
    }
    sums.push_back(Accumulated{ key, 0.0 + v });
}

std::string cell_text(const std::array<int, 3>& cell)
{
    return "(" + std::to_string(cell[0]) + ", " + std::to_string(cell[1]) + ", " + std::to_string(cell[2]) + ")";
}

} // namespace

// ------------------------------------------------------------------------------------------------ SparseOps

SparseMatrix SparseOps::from_triplets(const std::int64_t rows, const std::int64_t columns, const std::vector<Triplet>& entries)
{
    // coo_tocsr: the entries of each row in input order
    std::vector<std::int64_t> counts(static_cast<std::size_t>(rows) + 1u, 0);
    for (const Triplet& t : entries) {
        counts[static_cast<std::size_t>(t.row) + 1u] += 1;
    }
    std::partial_sum(counts.begin(), counts.end(), counts.begin());
    std::vector<Triplet> placed(entries.size());
    std::vector<std::int64_t> cursor(counts.begin(), counts.end() - 1);
    for (const Triplet& t : entries) {
        placed[static_cast<std::size_t>(cursor[static_cast<std::size_t>(t.row)]++)] = t;
    }
    // sum_duplicates: sort each row by column, then add equal columns in that order
    SparseMatrix m{ rows, columns, { 0 }, {}, {} };
    for (std::int64_t r{ 0 }; r < rows; ++r) {
        const auto first{ placed.begin() + counts[static_cast<std::size_t>(r)] };
        const auto last{ placed.begin() + counts[static_cast<std::size_t>(r) + 1u] };
        std::stable_sort(first, last, [](const Triplet& a, const Triplet& b) { return a.column < b.column; });
        for (auto it{ first }; it != last;) {
            double sum{ it->value };
            const std::int64_t column{ it->column };
            int count{ 1 };
            for (++it; it != last && it->column == column; ++it) {
                sum += it->value;
                ++count;
            }
            if (count > 2) {
                throw std::runtime_error("from_triplets: " + std::to_string(count) + " entries at (" + std::to_string(r) + ", " + std::to_string(column)
                                         + "): scipy's order of their sum is not reproducible");
            }
            m.indices.push_back(column);
            m.values.push_back(sum);
        }
        m.offsets.push_back(static_cast<std::int64_t>(m.indices.size()));
    }
    return m;
}

StateRows SparseOps::state_rows(const std::int64_t states, const std::int64_t columns, const std::vector<Triplet>& entries)
{
    // coo_tocsr keeps the entries of each row in input order: a stable sort by row does the same without offsets for
    // every state
    std::vector<Triplet> placed{ entries };
    std::stable_sort(std::execution::par, placed.begin(), placed.end(), [](const Triplet& a, const Triplet& b) { return a.row < b.row; });
    StateRows m{ states, columns, {}, { 0 }, {}, {} };
    auto first{ placed.begin() };
    while (first != placed.end()) {
        const std::int64_t row{ first->row };
        if (row < 0 || row >= states) {
            throw std::runtime_error("state_rows: row " + std::to_string(row) + " outside " + std::to_string(states) + " states");
        }
        const auto last{ std::find_if(first, placed.end(), [row](const Triplet& t) { return t.row != row; }) };
        // sum_duplicates: sort the row by column, then add equal columns in that order
        std::stable_sort(first, last, [](const Triplet& a, const Triplet& b) { return a.column < b.column; });
        for (auto it{ first }; it != last;) {
            double sum{ it->value };
            const std::int64_t column{ it->column };
            int count{ 1 };
            for (++it; it != last && it->column == column; ++it) {
                sum += it->value;
                ++count;
            }
            if (count > 2) {
                throw std::runtime_error("state_rows: " + std::to_string(count) + " entries at (" + std::to_string(row) + ", " + std::to_string(column)
                                         + "): scipy's order of their sum is not reproducible");
            }
            m.indices.push_back(column);
            m.values.push_back(sum);
        }
        m.state.push_back(row);
        m.offsets.push_back(static_cast<std::int64_t>(m.indices.size()));
        first = last;
    }
    return m;
}

StateValues SparseOps::times(const StateRows& m, const std::vector<double>& x)
{
    StateValues y{};
    for (std::size_t r{ 0 }; r < m.state.size(); ++r) {
        double sum{ 0.0 };
        for (std::int64_t e{ m.offsets[r] }; e < m.offsets[r + 1u]; ++e) {
            sum += m.values[static_cast<std::size_t>(e)] * x[static_cast<std::size_t>(m.indices[static_cast<std::size_t>(e)])];
        }
        y.state.push_back(m.state[r]);
        y.value.push_back(sum);
    }
    return y;
}

StateValues SparseOps::column_sums(const SparseMatrix& m)
{
    // y[column] += value * 1.0 with the rows in order: per column, its entries in row order, from 0.0
    std::vector<Triplet> entries{};
    for (std::int64_t r{ 0 }; r < m.rows; ++r) {
        for (std::int64_t e{ m.offsets[static_cast<std::size_t>(r)] }; e < m.offsets[static_cast<std::size_t>(r) + 1u]; ++e) {
            entries.push_back(Triplet{ r, m.indices[static_cast<std::size_t>(e)], m.values[static_cast<std::size_t>(e)] });
        }
    }
    std::stable_sort(std::execution::par, entries.begin(), entries.end(), [](const Triplet& a, const Triplet& b) { return a.column < b.column; });
    StateValues y{};
    for (const Triplet& t : entries) {
        const bool new_column{ y.state.empty() || y.state.back() != t.column };
        if (new_column) {
            y.state.push_back(t.column);
            y.value.push_back(0.0);
        }
        y.value.back() += t.value * 1.0;
    }
    return y;
}

StateRows SparseOps::permuted_columns(const StateRows& m, const std::vector<std::int64_t>& column_order)
{
    std::vector<std::int64_t> new_column(static_cast<std::size_t>(m.columns), 0);
    std::iota(new_column.begin(), new_column.end(), std::int64_t{ 0 });
    for (std::size_t i{ 0 }; i < column_order.size(); ++i) {
        new_column[static_cast<std::size_t>(column_order[i])] = static_cast<std::int64_t>(i);
    }
    StateRows out{ m.states, m.columns, {}, { 0 }, {}, {} };
    std::vector<Triplet> row{};
    for (std::size_t r{ 0 }; r < m.state.size(); ++r) {
        row.clear();
        for (std::int64_t e{ m.offsets[r] }; e < m.offsets[r + 1u]; ++e) {
            const double value{ m.values[static_cast<std::size_t>(e)] };
            if (value != 0.0) { // csr_matmat keeps only nonzero sums
                row.push_back(Triplet{ m.state[r], new_column[static_cast<std::size_t>(m.indices[static_cast<std::size_t>(e)])], value });
            }
        }
        if (row.empty()) {
            continue;
        }
        std::sort(row.begin(), row.end(), [](const Triplet& a, const Triplet& b) { return a.column < b.column; });
        for (const Triplet& t : row) {
            out.indices.push_back(t.column);
            out.values.push_back(t.value);
        }
        out.state.push_back(m.state[r]);
        out.offsets.push_back(static_cast<std::int64_t>(out.indices.size()));
    }
    return out;
}

std::vector<double> SparseOps::row_sums(const SparseMatrix& m)
{
    std::vector<double> y(static_cast<std::size_t>(m.rows), 0.0);
    for (std::int64_t r{ 0 }; r < m.rows; ++r) {
        double sum{ 0.0 };
        for (std::int64_t e{ m.offsets[static_cast<std::size_t>(r)] }; e < m.offsets[static_cast<std::size_t>(r) + 1u]; ++e) {
            sum += m.values[static_cast<std::size_t>(e)] * 1.0;
        }
        y[static_cast<std::size_t>(r)] = sum;
    }
    return y;
}

// ------------------------------------------------------------------------------------------------ IdentityMax

double IdentityMax::largest(const double current, const double value)
{
    const double magnitude{ std::fabs(value) };
    if (std::isnan(current) || std::isnan(magnitude)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::max(current, magnitude);
}

double IdentityMax::combine(const double a, const double b)
{
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::max(a, b);
}

std::vector<double> SparseOps::times(const SparseMatrix& m, const std::vector<double>& x)
{
    std::vector<double> y(static_cast<std::size_t>(m.rows), 0.0);
    for (std::int64_t r{ 0 }; r < m.rows; ++r) {
        double sum{ 0.0 };
        for (std::int64_t e{ m.offsets[static_cast<std::size_t>(r)] }; e < m.offsets[static_cast<std::size_t>(r) + 1u]; ++e) {
            sum += m.values[static_cast<std::size_t>(e)] * x[static_cast<std::size_t>(m.indices[static_cast<std::size_t>(e)])];
        }
        y[static_cast<std::size_t>(r)] = sum;
    }
    return y;
}

std::vector<double> SparseOps::left_times(const std::vector<double>& x, const SparseMatrix& m)
{
    std::vector<double> y(static_cast<std::size_t>(m.columns), 0.0);
    for (std::int64_t r{ 0 }; r < m.rows; ++r) {
        const double xr{ x[static_cast<std::size_t>(r)] };
        for (std::int64_t e{ m.offsets[static_cast<std::size_t>(r)] }; e < m.offsets[static_cast<std::size_t>(r) + 1u]; ++e) {
            const auto column{ static_cast<std::size_t>(m.indices[static_cast<std::size_t>(e)]) };
            y[column] += m.values[static_cast<std::size_t>(e)] * xr;
        }
    }
    return y;
}

SparseMatrix SparseOps::permuted(const SparseMatrix& m, const std::vector<std::int64_t>& row_order, const std::vector<std::int64_t>& column_order)
{
    std::vector<std::int64_t> new_column(static_cast<std::size_t>(m.columns), 0);
    std::iota(new_column.begin(), new_column.end(), std::int64_t{ 0 });
    for (std::size_t i{ 0 }; i < column_order.size(); ++i) {
        new_column[static_cast<std::size_t>(column_order[i])] = static_cast<std::int64_t>(i);
    }
    SparseMatrix out{ m.rows, m.columns, { 0 }, {}, {} };
    std::vector<Triplet> row{};
    for (std::int64_t i{ 0 }; i < m.rows; ++i) {
        const std::int64_t old{ row_order.empty() ? i : row_order[static_cast<std::size_t>(i)] };
        row.clear();
        for (std::int64_t e{ m.offsets[static_cast<std::size_t>(old)] }; e < m.offsets[static_cast<std::size_t>(old) + 1u]; ++e) {
            const double value{ m.values[static_cast<std::size_t>(e)] };
            if (value != 0.0) { // csr_matmat keeps only nonzero sums
                row.push_back(Triplet{ i, new_column[static_cast<std::size_t>(m.indices[static_cast<std::size_t>(e)])], value });
            }
        }
        std::sort(row.begin(), row.end(), [](const Triplet& a, const Triplet& b) { return a.column < b.column; });
        for (const Triplet& t : row) {
            out.indices.push_back(t.column);
            out.values.push_back(t.value);
        }
        out.offsets.push_back(static_cast<std::int64_t>(out.indices.size()));
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ MapAssembly

MapAssembly::MapAssembly(const TerrainGeometry& geometry, const VelocityTable& velocities, const PassRunner& passes, const SamplerSearch& sampler)
    : geometry_{ geometry }
    , velocities_{ velocities }
    , passes_{ passes }
    , sampler_{ sampler }
{
    int moving{ 0 };
    for (const Direction& c : velocities.directions) {
        const bool rest{ c.x == 0 && c.y == 0 && c.z == 0 };
        moving_slot_.push_back(rest ? -1 : moving);
        moving += rest ? 0 : 1;
    }
    std::vector<std::size_t> order(geometry.facets().size());
    std::iota(order.begin(), order.end(), std::size_t{ 0 });
    std::sort(order.begin(), order.end(), [&geometry](const std::size_t a, const std::size_t b) {
        return pack(geometry.facets()[a].key) < pack(geometry.facets()[b].key);
    });
    for (const std::size_t n : order) {
        facet_codes_.push_back(pack(geometry.facets()[n].key));
        facet_by_code_.push_back(static_cast<int>(n));
    }
}

std::uint64_t MapAssembly::pack(const FacetKey& key)
{
    const std::uint64_t lid{ key.lid ? 1u : 0u };
    const auto i{ static_cast<std::uint64_t>(key.i) & 0xFFFFFu };
    const auto j{ static_cast<std::uint64_t>(key.j) & 0xFFFFFu };
    const std::uint64_t tag{ key.lid ? 0u : static_cast<std::uint64_t>(key.tag) & 1u };
    const std::uint64_t k{ key.lid ? 0u : static_cast<std::uint64_t>(key.k + 0x40000) & 0x7FFFFu };
    return (lid << 63) | (i << 40) | (j << 20) | (tag << 19) | k;
}

std::int64_t MapAssembly::cell_index(const int i, const int j, const int k) const
{
    const TerrainRecord& record{ geometry_.record() };
    if (i < 0 || i >= record.lx || j < 0 || j >= record.ly) {
        return -1;
    }
    return geometry_.cell_index(i, j, k);
}

int MapAssembly::facet_index(const FacetKey& key) const
{
    const std::uint64_t code{ pack(key) };
    const auto found{ std::lower_bound(facet_codes_.begin(), facet_codes_.end(), code) };
    if (found == facet_codes_.end() || *found != code) {
        throw std::runtime_error("unknown facet " + key.python_text());
    }
    return facet_by_code_[static_cast<std::size_t>(found - facet_codes_.begin())];
}

ProducerRowView MapAssembly::producer_unit(const int facet, const int direction) const
{
    const int slot{ moving_slot_[static_cast<std::size_t>(direction)] };
    if (slot < 0) {
        return ProducerRowView{};
    }
    return passes_.row(static_cast<std::size_t>(facet), slot);
}

void MapAssembly::gather_entries(const FacetRange& range, const std::vector<std::int64_t>& row_of, std::vector<Triplet>& g, std::vector<Triplet>& r) const
{
    const auto q{ static_cast<std::int64_t>(velocities_.directions.size()) };
    std::vector<Accumulated> sums{};
    for (std::size_t f{ range.first }; f < range.last; ++f) {
        for (std::int64_t d{ 0 }; d < q; ++d) {
            const ProducerRowView unit{ producer_unit(static_cast<int>(f), static_cast<int>(d)) };
            if (unit.h.empty()) {
                continue;
            }
            if (!unit.has_vi) {
                throw std::logic_error("a producer row with h but without vi");
            }
            sums.clear();
            for (const HCell& e : unit.h) {
                const std::int64_t cell{ cell_index(e.cell[0], e.cell[1], e.cell[2]) };
                if (cell < 0) {
                    throw std::runtime_error("h of a non-fluid cell " + cell_text(e.cell));
                }
                accumulate(sums, cell, e.value);
            }
            const std::int64_t row{ row_of[f * static_cast<std::size_t>(q) + static_cast<std::size_t>(d)] };
            const double qv{ unit.vi };
            const auto opposite{ static_cast<std::int64_t>(velocities_.opposites[static_cast<std::size_t>(d)]) };
            for (const Accumulated& s : sums) {
                g.push_back(Triplet{ row, s.key * q + opposite, s.value });
                const double scaled{ qv * geometry_.cell_volume(static_cast<std::size_t>(s.key)) };
                r.push_back(Triplet{ s.key * q + d, row, s.value / scaled });
            }
        }
    }
}

void MapAssembly::pending_entries(const FacetRange& range, const std::vector<std::int64_t>& row_of, std::vector<Triplet>& out) const
{
    const auto q{ static_cast<std::int64_t>(velocities_.directions.size()) };
    std::vector<Accumulated> sums{};
    for (std::size_t f{ range.first }; f < range.last; ++f) {
        for (std::int64_t d{ 0 }; d < q; ++d) {
            const ProducerRowView unit{ producer_unit(static_cast<int>(f), static_cast<int>(d)) };
            if (unit.w.empty()) {
                continue;
            }
            sums.clear();
            for (const WFacet& e : unit.w) {
                accumulate(sums, facet_index(e.gkey), e.value);
            }
            const auto opposite{ static_cast<std::size_t>(velocities_.opposites[static_cast<std::size_t>(d)]) };
            const std::int64_t producer{ row_of[f * static_cast<std::size_t>(q) + static_cast<std::size_t>(d)] };
            for (const Accumulated& s : sums) {
                const std::int64_t receiver{ row_of[static_cast<std::size_t>(s.key) * static_cast<std::size_t>(q) + opposite] };
                if (receiver < 0) {
                    throw std::runtime_error("W receiver row (" + geometry_.facets()[static_cast<std::size_t>(s.key)].key.python_text() + ", "
                                             + std::to_string(opposite) + ") does not exist");
                }
                out.push_back(Triplet{ receiver, producer, s.value });
            }
        }
    }
}

void MapAssembly::entries(const std::vector<std::int64_t>& row_of, const int threads, std::vector<Triplet>& g, std::vector<Triplet>& r,
                          std::vector<Triplet>& w) const
{
    // blocks of facets on all threads; the blocks are joined in facet order, so the entries keep the serial order
    const std::size_t facets{ geometry_.facets().size() };
    const std::size_t block{ 4096 };
    const std::size_t blocks{ (facets + block - 1) / block };
    std::vector<std::vector<Triplet>> g_parts(blocks);
    std::vector<std::vector<Triplet>> r_parts(blocks);
    std::vector<std::vector<Triplet>> w_parts(blocks);
    ParallelFor::run(blocks, threads, [&](const std::size_t b) {
        const FacetRange range{ b * block, std::min(facets, (b + 1) * block) };
        gather_entries(range, row_of, g_parts[b], r_parts[b]);
        pending_entries(range, row_of, w_parts[b]);
    });
    for (std::size_t b{ 0 }; b < blocks; ++b) {
        g.insert(g.end(), g_parts[b].begin(), g_parts[b].end());
        r.insert(r.end(), r_parts[b].begin(), r_parts[b].end());
        w.insert(w.end(), w_parts[b].begin(), w_parts[b].end());
    }
}

AssembledMaps MapAssembly::assemble(const int threads) const
{
    const auto q{ static_cast<std::int64_t>(velocities_.directions.size()) };
    const auto cells{ static_cast<std::int64_t>(geometry_.cell_count()) };
    const auto facets{ static_cast<std::int64_t>(geometry_.facets().size()) };
    AssembledMaps m{};
    std::vector<std::int64_t> row_of(static_cast<std::size_t>(facets * q), -1);
    for (std::int64_t f{ 0 }; f < facets; ++f) {
        for (std::int64_t d{ 0 }; d < q; ++d) {
            const ProducerRowView unit{ producer_unit(static_cast<int>(f), static_cast<int>(d)) };
            if (unit.has_vi) {
                row_of[static_cast<std::size_t>(f * q + d)] = static_cast<std::int64_t>(m.rows.size());
                m.rows.push_back(WallRow{ static_cast<int>(f), static_cast<int>(d) });
                m.vi.push_back(unit.vi);
            }
        }
    }
    const auto rows{ static_cast<std::int64_t>(m.rows.size()) };
    const std::int64_t states{ cells * q };
    std::vector<Triplet> g_entries{};
    std::vector<Triplet> r_entries{};
    std::vector<Triplet> w_entries{};
    std::vector<Triplet> s_entries{};
    entries(row_of, threads, g_entries, r_entries, w_entries);
    for (std::int64_t f{ 0 }; f < facets; ++f) {
        const FacetSample& sample{ sampler_.samples()[static_cast<std::size_t>(f)] };
        for (std::size_t n{ 0 }; n < sample.cells.size(); ++n) {
            s_entries.push_back(Triplet{ f, sample.cells[n], sample.weights[n] });
        }
    }
    m.g = SparseOps::from_triplets(rows, states, g_entries);
    m.r = SparseOps::state_rows(states, rows, r_entries);
    m.w = SparseOps::from_triplets(rows, rows, w_entries);
    m.s = SparseOps::from_triplets(facets, cells, s_entries);
    return m;
}

StateProducts MapAssembly::state_products(const AssembledMaps& m)
{
    StateProducts p{};
    p.one_g = SparseOps::column_sums(m.g);
    p.r_vi = SparseOps::times(m.r, m.vi);
    return p;
}

std::vector<double> MapAssembly::volume_times_r(const StateRows& r) const
{
    // Vs @ R: y[row] += value * Vs[state], the states in order; Vs[state] = V[state // q]
    const std::size_t q{ velocities_.directions.size() };
    std::vector<double> y(static_cast<std::size_t>(r.columns), 0.0);
    for (std::size_t n{ 0 }; n < r.state.size(); ++n) {
        const double vs{ geometry_.cell_volume(static_cast<std::size_t>(r.state[n]) / q) };
        for (std::int64_t e{ r.offsets[n] }; e < r.offsets[n + 1u]; ++e) {
            const auto column{ static_cast<std::size_t>(r.indices[static_cast<std::size_t>(e)]) };
            y[column] += r.values[static_cast<std::size_t>(e)] * vs;
        }
    }
    return y;
}

RowProducts MapAssembly::row_products(const AssembledMaps& m) const
{
    const std::size_t rows{ m.vi.size() };
    RowProducts p{};
    const std::vector<double> ones_rows(rows, 1.0);
    // four independent products; each keeps scipy's summation order
    std::vector<std::thread> jobs{};
    jobs.emplace_back([&]() { p.g_one = SparseOps::row_sums(m.g); });
    jobs.emplace_back([&]() { p.w_one = SparseOps::times(m.w, ones_rows); });
    jobs.emplace_back([&]() { p.vs_r = volume_times_r(m.r); });
    jobs.emplace_back([&]() { p.one_w = SparseOps::left_times(ones_rows, m.w); });
    for (std::thread& job : jobs) {
        job.join();
    }
    return p;
}

IdentityValues MapAssembly::row_identities(const AssembledMaps& m, const RowProducts& p)
{
    const std::size_t rows{ m.vi.size() };
    IdentityValues out{};
    out.prism_coverage = -1.0;
    out.return_delivery = -1.0;
    for (std::size_t r{ 0 }; r < rows; ++r) {
        out.prism_coverage = IdentityMax::largest(out.prism_coverage, p.g_one[r] + p.w_one[r] - m.vi[r]);
        out.return_delivery = IdentityMax::largest(out.return_delivery, p.vs_r[r] + p.one_w[r] / m.vi[r] - 1.0);
    }
    return out;
}

GateFailure MapAssembly::failure(const std::string& name, const bool per_state, const std::vector<double>& value)
{
    GateFailure out{ name, per_state, {}, {} };
    std::vector<std::int64_t> bad{};
    for (std::size_t n{ 0 }; n < value.size(); ++n) {
        if (!std::isfinite(value[n]) || value[n] > GateReport::TOLERANCE) {
            bad.push_back(static_cast<std::int64_t>(n));
        }
    }
    const auto key{ [&value](const std::int64_t n) { return std::isnan(value[static_cast<std::size_t>(n)]) ? std::numeric_limits<double>::infinity() : value[static_cast<std::size_t>(n)]; } };
    std::stable_sort(bad.begin(), bad.end(), [&key](const std::int64_t a, const std::int64_t b) { return key(a) > key(b); });
    for (std::size_t n{ 0 }; n < bad.size() && n < 50; ++n) {
        out.items.push_back(bad[n]);
        out.residuals.push_back(value[static_cast<std::size_t>(bad[n])]);
    }
    return out;
}

RowGate MapAssembly::row_gate(const AssembledMaps& m, const RowProducts& p)
{
    const std::size_t rows{ m.vi.size() };
    std::vector<double> prism(rows, 0.0);
    std::vector<double> ret(rows, 0.0);
    for (std::size_t r{ 0 }; r < rows; ++r) {
        prism[r] = std::fabs(p.g_one[r] + p.w_one[r] - m.vi[r]) / m.vi[r];
        ret[r] = std::fabs(p.vs_r[r] + p.one_w[r] / m.vi[r] - 1.0);
    }
    const auto all_finite{ [](const std::vector<double>& v) { return std::all_of(v.begin(), v.end(), [](const double x) { return std::isfinite(x); }); } };
    const auto worst{ [&all_finite](const std::vector<double>& v) {
        return (!v.empty() && all_finite(v)) ? *std::max_element(v.begin(), v.end()) : std::numeric_limits<double>::quiet_NaN();
    } };
    RowGate gate{};
    gate.finite = all_finite(m.vi) && all_finite(m.g.values) && all_finite(m.r.values) && all_finite(m.w.values);
    gate.prism_worst = worst(prism);
    gate.return_worst = worst(ret);
    gate.prism = failure("prism_coverage", false, prism);
    gate.ret = failure("return_delivery", false, ret);
    return gate;
}

PermutedMaps MapAssembly::permute(const AssembledMaps& m) const
{
    const std::size_t rows{ m.rows.size() };
    PermutedMaps p{};
    std::vector<std::int64_t> rf(rows, 0);
    std::vector<std::int64_t> rd(rows, 0);
    for (std::size_t r{ 0 }; r < rows; ++r) {
        rf[r] = m.rows[r].facet;
        rd[r] = m.rows[r].direction;
    }
    p.order.resize(rows);
    std::iota(p.order.begin(), p.order.end(), std::int64_t{ 0 });
    std::stable_sort(p.order.begin(), p.order.end(), [&rf](const std::int64_t a, const std::int64_t b) { return rf[static_cast<std::size_t>(a)] < rf[static_cast<std::size_t>(b)]; });
    std::vector<double> weight_of_row(rows, 0.0);
    for (std::size_t r{ 0 }; r < rows; ++r) {
        weight_of_row[r] = velocities_.weights[static_cast<std::size_t>(rd[r])];
    }
    const std::vector<double> s0{ SparseOps::times(m.w, weight_of_row) };
    p.s = m.s;
    p.g = SparseOps::permuted(m.g, p.order, {});
    p.r = SparseOps::permuted_columns(m.r, p.order);
    p.w = SparseOps::permuted(m.w, p.order, p.order);
    for (const std::int64_t o : p.order) {
        p.row_facet.push_back(rf[static_cast<std::size_t>(o)]);
        p.row_direction.push_back(rd[static_cast<std::size_t>(o)]);
        p.vi.push_back(m.vi[static_cast<std::size_t>(o)]);
        p.s0.push_back(s0[static_cast<std::size_t>(o)]);
    }
    return p;
}

} // namespace fluidx3d::experimental::wall_builder
