/// @file slab_stream.cpp
/// @brief The volume parts of the container, one slab of x-columns at a time.
/// Built with FLUIDX3D_PRECISE_FLOAT: every value keeps the operation order of scipy's loops and numpy's elementwise
/// operations (terrain_container.py assemble, arms, MapIdentities, NativeIdentityGate, wall_list).

#include "slab_stream.hpp"

#include "python_text.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace fluidx3d::experimental::wall_builder {

namespace {

std::string cell_text(const std::array<int, 3>& cell)
{
    return "(" + std::to_string(cell[0]) + ", " + std::to_string(cell[1]) + ", " + std::to_string(cell[2]) + ")";
}

} // namespace

// ------------------------------------------------------------------------------------------------ SpoolFile

SpoolFile::SpoolFile(const std::filesystem::path& path, std::string section)
    : path_{ path }
    , section_{ std::move(section) }
    , file_{ path, std::ios::binary | std::ios::trunc }
{
    if (!file_) {
        throw std::runtime_error("cannot create the temporary file " + path.string());
    }
    buffer_.reserve(BUFFER);
}

void SpoolFile::u32(const std::int64_t value)
{
    if (value < 0 || value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("INVALID: " + section_ + " does not fit uint32");
    }
    const auto narrow{ static_cast<std::uint32_t>(value) };
    append(&narrow, 4);
}

void SpoolFile::f64(const double value)
{
    append(&value, 8);
}

void SpoolFile::append(const void* const data, const std::size_t size)
{
    if (buffer_.size() + size > BUFFER) {
        flush();
    }
    const auto* const bytes{ static_cast<const char*>(data) };
    buffer_.insert(buffer_.end(), bytes, bytes + size);
    count_ += 1;
}

void SpoolFile::flush()
{
    file_.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
    if (!file_) {
        throw std::runtime_error("cannot write the temporary file " + path_.string());
    }
    buffer_.clear();
}

void SpoolFile::close()
{
    flush();
    file_.close();
    if (!file_) {
        throw std::runtime_error("cannot close the temporary file " + path_.string());
    }
}

// ------------------------------------------------------------------------------------------------ FailureTop

void FailureTop::add(const std::int64_t item, const double residual)
{
    if (std::isfinite(residual) && !(residual > GateReport::TOLERANCE)) {
        return;
    }
    items_.push_back(Item{ item, residual });
    if (items_.size() >= 4096) {
        prune();
    }
}

void FailureTop::prune()
{
    // MapAssembly::failure: a stable sort by the residual (NaN as infinity), largest first, of the items in index order
    const auto key{ [](const Item& x) { return std::isnan(x.residual) ? std::numeric_limits<double>::infinity() : x.residual; } };
    std::stable_sort(items_.begin(), items_.end(), [&key](const Item& a, const Item& b) {
        const double ka{ key(a) };
        const double kb{ key(b) };
        if (ka != kb) {
            return ka > kb;
        }
        return a.index < b.index;
    });
    if (items_.size() > 50) {
        items_.resize(50);
    }
}

GateFailure FailureTop::result(const std::string& name) const
{
    FailureTop sorted{ *this };
    sorted.prune();
    GateFailure out{ name, true, {}, {} };
    for (const Item& item : sorted.items_) {
        out.items.push_back(item.index);
        out.residuals.push_back(item.residual);
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ WorstValue

void WorstValue::add(const double value)
{
    if (!std::isfinite(value)) {
        finite_ = false;
        return;
    }
    if (!any_ || value > largest_) {
        largest_ = value;
    }
    any_ = true;
}

double WorstValue::result() const
{
    if (!any_ || !finite_) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return largest_;
}

// ------------------------------------------------------------------------------------------------ SlabStream

SlabStream::SlabStream(const SlabInputs& inputs, const std::filesystem::path& folder, const int threads)
    : in_{ inputs }
    , donors_{ *inputs.geometry, threads }
    , d_offsets_{ folder / "D_offsets.u32", "D_offsets" }
    , d_indices_{ folder / "D_indices.u32", "D_indices" }
    , d_weights_{ folder / "D_weights.f64", "D_weights" }
    , direct_arm_{ folder / "direct_arm.f64", "direct_arm" }
{
    const std::size_t cells{ in_.geometry->cell_count() };
    results_.added.assign(cells, 0);
    support_flag_.assign(cells, 0);
    for (const std::int64_t cell : *in_.support) {
        support_flag_[static_cast<std::size_t>(cell)] = 1;
    }
    results_.support_arm.assign(in_.support->size(), 0.0);
}

const SpoolFile& SlabStream::spool(const std::string& name) const
{
    if (name == "D_offsets") {
        return d_offsets_;
    }
    if (name == "D_indices") {
        return d_indices_;
    }
    if (name == "D_weights") {
        return d_weights_;
    }
    if (name == "direct_arm") {
        return direct_arm_;
    }
    throw std::invalid_argument("no streamed section " + name);
}

int SlabStream::slab_end(const int first, const std::size_t slab_cells, const int slab_columns) const
{
    const int lx{ in_.geometry->record().lx };
    if (slab_columns > 0) {
        return std::min(lx, first + slab_columns);
    }
    const std::size_t start{ in_.geometry->column_first(first) };
    int last{ first + 1 };
    while (last < lx && in_.geometry->column_first(last + 1) - start <= slab_cells) {
        ++last;
    }
    return last;
}

void SlabStream::run(const std::size_t slab_cells, const int slab_columns, DiagTotals& diag)
{
    const int lx{ in_.geometry->record().lx };
    const auto start{ std::chrono::steady_clock::now() };
    d_offsets_.u32(0);
    int slabs{ 0 };
    for (int first{ 0 }; first < lx;) {
        const int last{ slab_end(first, slab_cells, slab_columns) };
        // the donors of a receiver lie in its own x-column or the next one on each side (periodic in x)
        std::vector<int> needed{ TerrainGeometry::wrap(first - 1, lx) };
        for (int i{ first }; i <= last; ++i) {
            needed.push_back(TerrainGeometry::wrap(i, lx));
        }
        donors_.compute(needed);
        states(first, last);
        replay(first, last, diag);
        // keep x-column last - 1 (the next slab's left neighbour), the next slab's first, and the two periodic ends
        for (int i{ std::max(first - 1, 0) }; i < last - 1; ++i) {
            if (i != 0 && i != lx - 1) {
                donors_.drop(i);
            }
        }
        ++slabs;
        const double seconds{ std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() };
        std::cout << "  slab " << slabs << ": x-columns " << first << ".." << last - 1 << " of " << lx << ", " << static_cast<int>(seconds) << " s" << std::endl;
        first = last;
    }
    d_offsets_.close();
    d_indices_.close();
    d_weights_.close();
    direct_arm_.close();
}

double SlabStream::value_at(const StateValues& v, const std::int64_t s, std::size_t& at)
{
    while (at < v.state.size() && v.state[at] < s) {
        ++at;
    }
    if (at < v.state.size() && v.state[at] == s) {
        return v.value[at];
    }
    return 0.0;
}

bool SlabStream::has_return(const std::int64_t s)
{
    const std::vector<std::int64_t>& rows{ in_.permuted_r->state };
    while (return_at_ < rows.size() && rows[return_at_] < s) {
        ++return_at_;
    }
    return return_at_ < rows.size() && rows[return_at_] == s;
}

const DonorState& SlabStream::donor_state(const int i, const std::int64_t n, const std::size_t d) const
{
    const std::size_t q{ in_.velocities->directions.size() };
    const DonorColumn& column{ donors_.column(i) };
    return column.states[(static_cast<std::size_t>(n) - column.first) * q + d];
}

void SlabStream::states(const int first, const int last)
{
    const TerrainGeometry& geometry{ *in_.geometry };
    const int ly{ geometry.record().ly };
    const int z_top{ geometry.record().z_top };
    for (int i{ first }; i < last; ++i) {
        for (int j{ 0 }; j < ly; ++j) {
            const int k_first{ geometry.first_layer(i, j) };
            const auto n_first{ static_cast<std::int64_t>(geometry.column_cell(i, j)) };
            for (int k{ k_first }; k < z_top; ++k) {
                cell_states(i, j, k, n_first + (k - k_first));
            }
        }
    }
}

void SlabStream::cell_states(const int i, const int j, const int k, const std::int64_t n)
{
    const std::vector<Direction>& directions{ in_.velocities->directions };
    const auto q{ static_cast<std::int64_t>(directions.size()) };
    const double volume{ in_.geometry->cell_volume(static_cast<std::size_t>(n)) };
    results_.finite = results_.finite && std::isfinite(volume);
    for (std::int64_t d{ 0 }; d < q; ++d) {
        const StatePlace at{ { i, j, k }, n, d, n * q + d, volume };
        donor_role(at);
        const ReceiverEntry entry{ receiver_role(at) };
        rule_253(at, entry);
    }
}

void SlabStream::donor_role(const StatePlace& at)
{
    // D column s: assemble's D entry of this donor state and (Vs @ D)[s]
    const TerrainGeometry& geometry{ *in_.geometry };
    const Direction& c{ in_.velocities->directions[static_cast<std::size_t>(at.d)] };
    const std::array<int, 3>& cell{ at.cell };
    const DonorState& own{ donor_state(cell[0], at.n, static_cast<std::size_t>(at.d)) };
    double vs_d{ 0.0 };
    if (own.volume != 0.0) {
        const std::int64_t receiver{ geometry.cell_index(cell[0] + c.x, cell[1] + c.y, cell[2] + c.z) };
        if (receiver < 0) {
            const int recv_i{ TerrainGeometry::wrap(cell[0] + c.x, geometry.record().lx) };
            const int recv_j{ TerrainGeometry::wrap(cell[1] + c.y, geometry.record().ly) };
            const std::array<int, 3> recv{ recv_i, recv_j, cell[2] + c.z };
            throw std::runtime_error("direct " + PythonFloat::repr(own.volume) + " from " + cell_text(cell) + " along direction " + std::to_string(at.d)
                                     + " into a non-fluid cell " + cell_text(recv));
        }
        const double receiver_volume{ geometry.cell_volume(static_cast<std::size_t>(receiver)) };
        const double ratio{ own.volume / receiver_volume };
        vs_d += ratio * receiver_volume;
    }
    const double one_g{ value_at(in_.products->one_g, at.s, one_g_at_) };
    const double residual{ vs_d + one_g - at.volume };
    results_.donor_partition = IdentityMax::largest(results_.donor_partition, residual);
    const double relative{ std::fabs(vs_d + one_g - at.volume) / at.volume };
    results_.donor_worst.add(relative);
    results_.donor_failures.add(at.s, relative);
}

SlabStream::ReceiverEntry SlabStream::receiver_role(const StatePlace& at)
{
    // D row s: its only possible donor is the cell at -c (the streaming is a bijection on the periodic box)
    const TerrainGeometry& geometry{ *in_.geometry };
    const Direction& c{ in_.velocities->directions[static_cast<std::size_t>(at.d)] };
    const std::array<int, 3>& cell{ at.cell };
    ReceiverEntry entry{};
    entry.donor = geometry.cell_index(cell[0] - c.x, cell[1] - c.y, cell[2] - c.z);
    if (entry.donor >= 0) {
        const int donor_i{ TerrainGeometry::wrap(cell[0] - c.x, geometry.record().lx) };
        const DonorState& from{ donor_state(donor_i, entry.donor, static_cast<std::size_t>(at.d)) };
        if (from.volume != 0.0) {
            entry.has_entry = true;
            entry.ratio = from.volume / at.volume;
            write_entry(at, entry, from);
        }
    }
    d_offsets_.u32(static_cast<std::int64_t>(results_.d_entries));
    double d_one{ 0.0 };
    if (entry.has_entry) {
        d_one += entry.ratio * 1.0;
    }
    const double r_vi{ value_at(in_.products->r_vi, at.s, r_vi_at_) };
    results_.receiver_coverage = IdentityMax::largest(results_.receiver_coverage, d_one + r_vi - 1.0);
    const double relative{ std::fabs(d_one + r_vi - 1.0) };
    results_.receiver_worst.add(relative);
    results_.receiver_failures.add(at.s, relative);
    return entry;
}

void SlabStream::write_entry(const StatePlace& at, const ReceiverEntry& entry, const DonorState& from)
{
    // the D entry and its direct_arm: np.array(part[1:]) / part[0] - centroid_local[donor]
    const auto q{ static_cast<std::int64_t>(in_.velocities->directions.size()) };
    const Vec3& local{ in_.geometry->cell_local(static_cast<std::size_t>(entry.donor)) };
    const double arm_x{ from.moment[1] / from.moment[0] - local.x };
    const double arm_y{ from.moment[2] / from.moment[0] - local.y };
    const double arm_z{ from.moment[3] / from.moment[0] - local.z };
    d_indices_.u32(entry.donor * q + at.d);
    d_weights_.f64(entry.ratio);
    direct_arm_.f64(arm_x);
    direct_arm_.f64(arm_y);
    direct_arm_.f64(arm_z);
    results_.d_entries += 1;
    results_.finite = results_.finite && std::isfinite(entry.ratio);
    if (support_flag_[static_cast<std::size_t>(entry.donor)] == 0) {
        return;
    }
    // sample_support_closure needs the largest |direct arm| of each support cell's D entries
    const auto position{ std::lower_bound(in_.support->begin(), in_.support->end(), entry.donor) - in_.support->begin() };
    double& largest{ results_.support_arm[static_cast<std::size_t>(position)] };
    largest = std::max(largest, std::fabs(arm_x));
    largest = std::max(largest, std::fabs(arm_y));
    largest = std::max(largest, std::fabs(arm_z));
}

void SlabStream::rule_253(const StatePlace& at, const ReceiverEntry& entry)
{
    // wall_list: a state is pure when its one D entry is 1.0 from the cell at -c and R has no entry in its row
    const bool pure{ entry.has_entry && entry.ratio == 1.0 && !has_return(at.s) };
    if (pure) {
        return;
    }
    const std::vector<char>& wall{ *in_.wall };
    if (wall[static_cast<std::size_t>(at.n)] == 0) {
        results_.added[static_cast<std::size_t>(at.n)] = 1;
    }
    if (entry.has_entry && wall[static_cast<std::size_t>(entry.donor)] == 0) {
        results_.added[static_cast<std::size_t>(entry.donor)] = 1;
    }
}

void SlabStream::replay(const int first, const int last, DiagTotals& diag) const
{
    const std::size_t q{ in_.velocities->directions.size() };
    for (int i{ first }; i < last; ++i) {
        const DonorColumn& column{ donors_.column(i) };
        const std::size_t cells{ column.events.size() };
        for (std::size_t cell{ 0 }; cell < cells; ++cell) {
            const std::vector<DiagEvent>& events{ column.events[cell] };
            std::size_t begin{ 0 };
            for (std::size_t d{ 0 }; d < q; ++d) {
                const std::size_t end{ column.event_end[cell * q + d] };
                diag.add_unit(std::span<const DiagEvent>{ events.data() + begin, end - begin });
                begin = end;
            }
        }
    }
}

} // namespace fluidx3d::experimental::wall_builder
