/// @file maps.cpp
/// @brief The transport maps of the container, with scipy.sparse's construction and product rules.
/// Built with FLUIDX3D_PRECISE_FLOAT: every sum keeps the order of scipy's loops and numpy's elementwise operations.

#include "maps.hpp"

#include "python_text.hpp"

#include <algorithm>
#include <cmath>
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
    const TerrainRecord& record{ geometry.record() };
    int kmax{ std::numeric_limits<int>::min() };
    kmin_ = std::numeric_limits<int>::max();
    for (const TerrainCell& cell : geometry.cells()) {
        kmin_ = std::min(kmin_, cell.ijk[2]);
        kmax = std::max(kmax, cell.ijk[2]);
    }
    layers_ = kmax - kmin_ + 1;
    cell_slots_.assign(static_cast<std::size_t>(record.lx) * static_cast<std::size_t>(record.ly) * static_cast<std::size_t>(layers_), -1);
    for (std::size_t n{ 0 }; n < geometry.cells().size(); ++n) {
        const std::array<int, 3>& c{ geometry.cells()[n].ijk };
        cell_slots_[(static_cast<std::size_t>(c[0]) * static_cast<std::size_t>(record.ly) + static_cast<std::size_t>(c[1])) * static_cast<std::size_t>(layers_)
                    + static_cast<std::size_t>(c[2] - kmin_)]
            = static_cast<std::int64_t>(n);
    }
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
    if (i < 0 || i >= record.lx || j < 0 || j >= record.ly || k < kmin_ || k >= kmin_ + layers_) {
        return -1;
    }
    return cell_slots_[(static_cast<std::size_t>(i) * static_cast<std::size_t>(record.ly) + static_cast<std::size_t>(j)) * static_cast<std::size_t>(layers_)
                       + static_cast<std::size_t>(k - kmin_)];
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

const UnitResult* MapAssembly::producer_unit(const int facet, const int direction) const
{
    const int slot{ moving_slot_[static_cast<std::size_t>(direction)] };
    if (slot < 0) {
        return nullptr;
    }
    return &passes_.facet_results()[static_cast<std::size_t>(facet)][static_cast<std::size_t>(slot)];
}

void MapAssembly::direct_entries(const std::vector<double>& volume, std::vector<Triplet>& out) const
{
    const TerrainRecord& record{ geometry_.record() };
    const auto q{ static_cast<std::int64_t>(velocities_.directions.size()) };
    for (std::size_t n{ 0 }; n < geometry_.cells().size(); ++n) {
        const std::array<int, 3>& cell{ geometry_.cells()[n].ijk };
        for (std::int64_t d{ 0 }; d < q; ++d) {
            const double vol{ passes_.cell_results()[n][static_cast<std::size_t>(d)].direct[0].value };
            if (vol == 0.0) {
                continue;
            }
            const Direction& c{ velocities_.directions[static_cast<std::size_t>(d)] };
            const std::array<int, 3> recv{ TerrainGeometry::wrap(cell[0] + c.x, record.lx), TerrainGeometry::wrap(cell[1] + c.y, record.ly), cell[2] + c.z };
            const std::int64_t receiver{ cell_index(recv[0], recv[1], recv[2]) };
            if (receiver < 0) {
                throw std::runtime_error("direct " + PythonFloat::repr(vol) + " from " + cell_text(cell) + " along direction " + std::to_string(d)
                                         + " into a non-fluid cell " + cell_text(recv));
            }
            const double ratio{ vol / volume[static_cast<std::size_t>(receiver)] };
            out.push_back(Triplet{ receiver * q + d, static_cast<std::int64_t>(n) * q + d, ratio });
        }
    }
}

void MapAssembly::gather_entries(const std::vector<double>& volume, const std::vector<std::int64_t>& row_of, std::vector<Triplet>& g,
                                 std::vector<Triplet>& r) const
{
    const auto q{ static_cast<std::int64_t>(velocities_.directions.size()) };
    std::vector<Accumulated> sums{};
    for (std::size_t f{ 0 }; f < geometry_.facets().size(); ++f) {
        for (std::int64_t d{ 0 }; d < q; ++d) {
            const UnitResult* unit{ producer_unit(static_cast<int>(f), static_cast<int>(d)) };
            if (unit == nullptr || unit->h.empty()) {
                continue;
            }
            sums.clear();
            for (const HEntry& e : unit->h) {
                const std::int64_t cell{ cell_index(e.cell[0], e.cell[1], e.cell[2]) };
                if (cell < 0) {
                    throw std::runtime_error("h of a non-fluid cell " + cell_text(e.cell));
                }
                accumulate(sums, cell, e.value);
            }
            const std::int64_t row{ row_of[f * static_cast<std::size_t>(q) + static_cast<std::size_t>(d)] };
            const double qv{ unit->vi[0].value };
            const auto opposite{ static_cast<std::int64_t>(velocities_.opposites[static_cast<std::size_t>(d)]) };
            for (const Accumulated& s : sums) {
                g.push_back(Triplet{ row, s.key * q + opposite, s.value });
                const double scaled{ qv * volume[static_cast<std::size_t>(s.key)] };
                r.push_back(Triplet{ s.key * q + d, row, s.value / scaled });
            }
        }
    }
}

void MapAssembly::pending_entries(const std::vector<std::int64_t>& row_of, std::vector<Triplet>& out) const
{
    const auto q{ static_cast<std::int64_t>(velocities_.directions.size()) };
    std::vector<Accumulated> sums{};
    for (std::size_t f{ 0 }; f < geometry_.facets().size(); ++f) {
        for (std::int64_t d{ 0 }; d < q; ++d) {
            const UnitResult* unit{ producer_unit(static_cast<int>(f), static_cast<int>(d)) };
            if (unit == nullptr || unit->w.empty()) {
                continue;
            }
            sums.clear();
            for (const WEntry& e : unit->w) {
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

AssembledMaps MapAssembly::assemble() const
{
    const auto q{ static_cast<std::int64_t>(velocities_.directions.size()) };
    const auto cells{ static_cast<std::int64_t>(geometry_.cells().size()) };
    const auto facets{ static_cast<std::int64_t>(geometry_.facets().size()) };
    AssembledMaps m{};
    for (const TerrainCell& cell : geometry_.cells()) {
        m.volume.push_back(cell.volume);
    }
    std::vector<std::int64_t> row_of(static_cast<std::size_t>(facets * q), -1);
    for (std::int64_t f{ 0 }; f < facets; ++f) {
        for (std::int64_t d{ 0 }; d < q; ++d) {
            const UnitResult* unit{ producer_unit(static_cast<int>(f), static_cast<int>(d)) };
            if (unit != nullptr && !unit->vi.empty()) {
                row_of[static_cast<std::size_t>(f * q + d)] = static_cast<std::int64_t>(m.rows.size());
                m.rows.push_back(WallRow{ static_cast<int>(f), static_cast<int>(d) });
                m.vi.push_back(unit->vi[0].value);
            }
        }
    }
    const auto rows{ static_cast<std::int64_t>(m.rows.size()) };
    const std::int64_t states{ cells * q };
    std::vector<Triplet> d_entries{};
    std::vector<Triplet> g_entries{};
    std::vector<Triplet> r_entries{};
    std::vector<Triplet> w_entries{};
    std::vector<Triplet> s_entries{};
    direct_entries(m.volume, d_entries);
    gather_entries(m.volume, row_of, g_entries, r_entries);
    pending_entries(row_of, w_entries);
    for (std::int64_t f{ 0 }; f < facets; ++f) {
        const FacetSample& sample{ sampler_.samples()[static_cast<std::size_t>(f)] };
        for (std::size_t n{ 0 }; n < sample.cells.size(); ++n) {
            s_entries.push_back(Triplet{ f, sample.cells[n], sample.weights[n] });
        }
    }
    m.d = SparseOps::from_triplets(states, states, d_entries);
    m.g = SparseOps::from_triplets(rows, states, g_entries);
    m.r = SparseOps::from_triplets(states, rows, r_entries);
    m.w = SparseOps::from_triplets(rows, rows, w_entries);
    m.s = SparseOps::from_triplets(facets, cells, s_entries);
    return m;
}

MapProducts MapAssembly::products(const AssembledMaps& m) const
{
    const std::size_t q{ velocities_.directions.size() };
    const std::size_t states{ m.volume.size() * q };
    const std::size_t rows{ m.vi.size() };
    MapProducts p{};
    p.vs.assign(states, 0.0);
    for (std::size_t s{ 0 }; s < states; ++s) {
        p.vs[s] = m.volume[s / q];
    }
    const std::vector<double> ones_rows(rows, 1.0);
    const std::vector<double> ones_states(states, 1.0);
    // eight independent products; each keeps scipy's summation order
    std::vector<std::thread> jobs{};
    jobs.emplace_back([&]() { p.vs_d = SparseOps::left_times(p.vs, m.d); });
    jobs.emplace_back([&]() { p.one_g = SparseOps::left_times(ones_rows, m.g); });
    jobs.emplace_back([&]() { p.d_one = SparseOps::times(m.d, ones_states); });
    jobs.emplace_back([&]() { p.r_vi = SparseOps::times(m.r, m.vi); });
    jobs.emplace_back([&]() { p.g_one = SparseOps::times(m.g, ones_states); });
    jobs.emplace_back([&]() { p.w_one = SparseOps::times(m.w, ones_rows); });
    jobs.emplace_back([&]() { p.vs_r = SparseOps::left_times(p.vs, m.r); });
    jobs.emplace_back([&]() { p.one_w = SparseOps::left_times(ones_rows, m.w); });
    for (std::thread& job : jobs) {
        job.join();
    }
    return p;
}

IdentityValues MapAssembly::identities(const AssembledMaps& m, const MapProducts& p) const
{
    const std::size_t states{ p.vs.size() };
    const std::size_t rows{ m.vi.size() };
    IdentityValues out{};
    // float(np.max(np.abs(x))): NaN propagates
    const auto largest{ [](double current, const double value) {
        const double magnitude{ std::fabs(value) };
        return (std::isnan(current) || std::isnan(magnitude)) ? std::numeric_limits<double>::quiet_NaN() : std::max(current, magnitude);
    } };
    out.donor_partition = -1.0;
    out.receiver_coverage = -1.0;
    for (std::size_t s{ 0 }; s < states; ++s) {
        out.donor_partition = largest(out.donor_partition, p.vs_d[s] + p.one_g[s] - p.vs[s]);
        out.receiver_coverage = largest(out.receiver_coverage, p.d_one[s] + p.r_vi[s] - 1.0);
    }
    out.prism_coverage = -1.0;
    out.return_delivery = -1.0;
    for (std::size_t r{ 0 }; r < rows; ++r) {
        out.prism_coverage = largest(out.prism_coverage, p.g_one[r] + p.w_one[r] - m.vi[r]);
        out.return_delivery = largest(out.return_delivery, p.vs_r[r] + p.one_w[r] / m.vi[r] - 1.0);
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

GateReport MapAssembly::gate(const AssembledMaps& m, const MapProducts& p) const
{
    const std::size_t states{ p.vs.size() };
    const std::size_t rows{ m.vi.size() };
    const std::vector<double>& vs{ p.vs };
    const std::vector<double>& vs_d{ p.vs_d };
    const std::vector<double>& one_g{ p.one_g };
    const std::vector<double>& d_one{ p.d_one };
    const std::vector<double>& r_vi{ p.r_vi };
    const std::vector<double>& g_one{ p.g_one };
    const std::vector<double>& w_one{ p.w_one };
    const std::vector<double>& vs_r{ p.vs_r };
    const std::vector<double>& one_w{ p.one_w };
    std::vector<double> donor(states, 0.0);
    std::vector<double> recv(states, 0.0);
    for (std::size_t s{ 0 }; s < states; ++s) {
        donor[s] = std::fabs(vs_d[s] + one_g[s] - vs[s]) / vs[s];
        recv[s] = std::fabs(d_one[s] + r_vi[s] - 1.0);
    }
    std::vector<double> prism(rows, 0.0);
    std::vector<double> ret(rows, 0.0);
    for (std::size_t r{ 0 }; r < rows; ++r) {
        prism[r] = std::fabs(g_one[r] + w_one[r] - m.vi[r]) / m.vi[r];
        ret[r] = std::fabs(vs_r[r] + one_w[r] / m.vi[r] - 1.0);
    }
    GateReport report{};
    const auto all_finite{ [](const std::vector<double>& v) { return std::all_of(v.begin(), v.end(), [](const double x) { return std::isfinite(x); }); } };
    report.finite = all_finite(m.volume) && all_finite(m.vi) && all_finite(m.d.values) && all_finite(m.g.values) && all_finite(m.r.values)
                    && all_finite(m.w.values);
    const auto worst{ [&all_finite](const std::vector<double>& v) {
        return (!v.empty() && all_finite(v)) ? *std::max_element(v.begin(), v.end()) : std::numeric_limits<double>::quiet_NaN();
    } };
    report.worst = IdentityValues{ worst(donor), worst(recv), worst(prism), worst(ret) };
    const std::vector<GateFailure> candidates{ failure("donor_partition", true, donor), failure("receiver_coverage", true, recv),
                                               failure("prism_coverage", false, prism), failure("return_delivery", false, ret) };
    for (const GateFailure& f : candidates) {
        if (!f.items.empty()) {
            report.failing.push_back(f);
        }
    }
    report.passed = report.finite && report.failing.empty();
    return report;
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
    p.d = m.d;
    p.s = m.s;
    p.g = SparseOps::permuted(m.g, p.order, {});
    p.r = SparseOps::permuted(m.r, {}, p.order);
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
