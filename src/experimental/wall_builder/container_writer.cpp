/// @file container_writer.cpp
/// @brief The VWNAT01 container of terrain_container.py, section for section, and its provenance text.
/// Built with FLUIDX3D_PRECISE_FLOAT: every value keeps numpy's operation order.

#include "container_writer.hpp"

#include "dense_algebra.hpp"
#include "python_text.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fluidx3d::experimental::wall_builder {

namespace {

constexpr std::uint32_t DTYPE_U32{ 1 };
constexpr std::uint32_t DTYPE_I32{ 2 };
constexpr std::uint32_t DTYPE_F64{ 3 };
constexpr std::uint32_t DTYPE_TEXT{ 4 };

/// @brief The sections of a VWNAT01 file (Writer): name[32], dtype u32, count u64, payload.
class SectionBuffer {
public:
    void f64(const std::string& name, const std::vector<double>& values) { add(name, DTYPE_F64, values.data(), values.size(), 8); }

    void i32(const std::string& name, const std::vector<std::int64_t>& values)
    {
        std::vector<std::int32_t> narrow{};
        for (const std::int64_t v : values) {
            if (v < std::numeric_limits<std::int32_t>::min() || v > std::numeric_limits<std::int32_t>::max()) {
                throw std::runtime_error("INVALID: " + name + " does not fit int32");
            }
            narrow.push_back(static_cast<std::int32_t>(v));
        }
        add(name, DTYPE_I32, narrow.data(), narrow.size(), 4);
    }

    void u32(const std::string& name, const std::vector<std::int64_t>& values)
    {
        std::vector<std::uint32_t> narrow{};
        for (const std::int64_t v : values) {
            if (v < 0 || v > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("INVALID: " + name + " does not fit uint32");
            }
            narrow.push_back(static_cast<std::uint32_t>(v));
        }
        add(name, DTYPE_U32, narrow.data(), narrow.size(), 4);
    }

    void text(const std::string& name, const std::string& value) { add(name, DTYPE_TEXT, value.data(), value.size(), 1); }

    void csr(const std::string& name, const SparseMatrix& m)
    {
        u32(name + "_shape", { m.rows, m.columns });
        u32(name + "_offsets", m.offsets);
        u32(name + "_indices", m.indices);
        f64(name + "_weights", m.values);
    }

    /// @brief The file as chunks: magic and section count, then one chunk per section (no copy into one buffer).
    [[nodiscard]] std::vector<std::vector<char>> file()
    {
        std::vector<char> head(12, 0);
        const char magic[]{ "VWNAT01\n" };
        std::memcpy(head.data(), magic, 8);
        const auto count{ static_cast<std::uint32_t>(chunks_.size()) };
        std::memcpy(head.data() + 8, &count, 4);
        std::vector<std::vector<char>> out{};
        out.push_back(std::move(head));
        for (std::vector<char>& chunk : chunks_) {
            out.push_back(std::move(chunk));
        }
        chunks_.clear();
        return out;
    }

private:
    std::vector<std::vector<char>> chunks_{}; ///< One per section: name[32], dtype u32, count u64, payload.

    void add(const std::string& name, const std::uint32_t dtype, const void* data, const std::size_t count, const std::size_t element)
    {
        if (name.size() > 32) {
            throw std::runtime_error("section name longer than 32: " + name);
        }
        const std::size_t payload{ count * element };
        std::vector<char> chunk(44 + payload, 0);
        std::copy(name.begin(), name.end(), chunk.begin());
        std::memcpy(chunk.data() + 32, &dtype, 4);
        const std::uint64_t elements{ count };
        std::memcpy(chunk.data() + 36, &elements, 8);
        if (payload > 0) {
            std::memcpy(chunk.data() + 44, data, payload);
        }
        chunks_.push_back(std::move(chunk));
    }
};

std::string velocity_list(const std::array<double, 3>& u)
{
    return "[" + PythonFloat::repr(u[0]) + ", " + PythonFloat::repr(u[1]) + ", " + PythonFloat::repr(u[2]) + "]";
}

} // namespace

// ------------------------------------------------------------------------------------------------ RoughLaw

RoughLaw::RoughLaw(const double z0_value, const double k0_value)
    : z0{ z0_value }
    , k0{ k0_value }
{
    if (!(std::isfinite(z0) && z0 > 0.0)) {
        throw std::invalid_argument("--rough-z0 must be finite and > 0 (it is " + PythonFloat::repr(z0) + ")");
    }
    if (!(SAMPLE_DISTANCE > z0)) {
        throw std::invalid_argument("--rough-z0 " + PythonFloat::repr(z0) + " >= the sample distance 0.5: the log law needs the sample above the roughness");
    }
    if (!(std::isfinite(k0) && k0 >= 0.0)) {
        throw std::invalid_argument("--rough-K0 must be finite and >= 0 (it is " + PythonFloat::repr(k0) + ")");
    }
    const double ratio{ KAPPA / std::log(SAMPLE_DISTANCE / z0) };
    c = std::pow(ratio, 2.0); // Python's float ** 2 calls the C library pow
    if (!(std::isfinite(c) && c > 0.0)) {
        throw std::invalid_argument("--rough-z0 " + PythonFloat::repr(z0) + " gives C = " + PythonFloat::repr(c) + ", not a finite positive coefficient");
    }
}

// ------------------------------------------------------------------------------------------------ ContainerWriter

ContainerWriter::ContainerWriter(const ContainerParts& parts, const ContainerOptions& options)
    : parts_{ parts }
    , options_{ options }
{
    int lowest{ std::numeric_limits<int>::max() };
    for (const TerrainCell& cell : parts.geometry->cells()) {
        lowest = std::min(lowest, cell.ijk[2]);
    }
    z_lo_ = lowest - 1;
}

std::array<double, 3> ContainerWriter::parse_velocity(const std::string& text)
{
    std::vector<double> values{};
    std::istringstream fields{ text };
    std::string field{};
    while (std::getline(fields, field, ',')) {
        char* end{ nullptr };
        const double value{ std::strtod(field.c_str(), &end) };
        if (field.empty() || *end != '\0') {
            throw std::invalid_argument("--initial-velocity needs three comma-separated numbers (it is '" + text + "')");
        }
        values.push_back(value);
    }
    if (values.size() != 3) {
        throw std::invalid_argument("--initial-velocity needs three comma-separated numbers (it is '" + text + "')");
    }
    const double sum{ 0.0 + values[0] * values[0] + values[1] * values[1] + values[2] * values[2] };
    const bool finite{ std::isfinite(values[0]) && std::isfinite(values[1]) && std::isfinite(values[2]) };
    if (!finite || !(std::sqrt(sum) < 1.0 / std::sqrt(3.0))) {
        throw std::invalid_argument("--initial-velocity '" + text + "' must be finite with |u| < c = 1/sqrt(3)");
    }
    return { values[0], values[1], values[2] };
}

std::vector<double> ContainerWriter::initial_state() const
{
    const std::vector<double>& w{ parts_.velocities->weights };
    const std::size_t cells{ parts_.geometry->cells().size() };
    const std::size_t q{ w.size() };
    std::vector<double> f0(cells * q, 0.0);
    if (!options_.initial_velocity) {
        for (std::size_t n{ 0 }; n < cells; ++n) {
            std::copy(w.begin(), w.end(), f0.begin() + static_cast<std::ptrdiff_t>(n * q));
        }
        return f0;
    }
    // UniformStart.f0: cu = u @ c.T (numpy's dgemm), uu = (u * u).sum(axis=1), the bracket in numpy's order
    const std::array<double, 3>& velocity{ *options_.initial_velocity };
    Matrix u{ static_cast<int>(cells), 3, std::vector<double>(cells * 3, 0.0) };
    for (std::size_t n{ 0 }; n < cells; ++n) {
        std::copy(velocity.begin(), velocity.end(), u.data.begin() + static_cast<std::ptrdiff_t>(3 * n));
    }
    Matrix c{ static_cast<int>(q), 3, std::vector<double>(q * 3, 0.0) };
    for (std::size_t d{ 0 }; d < q; ++d) {
        const Direction& dir{ parts_.velocities->directions[d] };
        c.data[3 * d] = static_cast<double>(dir.x);
        c.data[3 * d + 1] = static_cast<double>(dir.y);
        c.data[3 * d + 2] = static_cast<double>(dir.z);
    }
    const Matrix cu{ DenseAlgebra::times_transposed(u, c) };
    const double uu{ velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2] };
    for (std::size_t n{ 0 }; n < cells; ++n) {
        for (std::size_t d{ 0 }; d < q; ++d) {
            const double x{ cu.data[n * q + d] };
            const double bracket{ 1.0 + 3.0 * x + 4.5 * x * x - 1.5 * uu + 4.5 * (x * x * x - x * uu) };
            const double rho{ 1.0 };
            f0[n * q + d] = w[d] * rho * bracket;
        }
    }
    return f0;
}

std::vector<double> ContainerWriter::base_rows(const std::vector<double>& area, const std::vector<double>& normal) const
{
    // MapIdentities.base_rows: phi0 = (6 w vi^2 / A) (c - (c . n) n), in numpy's order
    const PermutedMaps& p{ *parts_.permuted };
    std::vector<double> out{};
    for (std::size_t r{ 0 }; r < p.vi.size(); ++r) {
        const auto facet{ static_cast<std::size_t>(p.row_facet[r]) };
        const auto d{ static_cast<std::size_t>(p.row_direction[r]) };
        const Direction& dir{ parts_.velocities->directions[d] };
        const std::array<double, 3> c{ static_cast<double>(dir.x), static_cast<double>(dir.y), static_cast<double>(dir.z) };
        const std::array<double, 3> n{ normal[3 * facet], normal[3 * facet + 1], normal[3 * facet + 2] };
        const double cn{ c[0] * n[0] + c[1] * n[1] + c[2] * n[2] };
        const double prism{ p.vi[r] };
        const double scale{ 6.0 * parts_.velocities->weights[d] * (prism * prism) / area[facet] };
        for (std::size_t k{ 0 }; k < 3; ++k) {
            const double tangent{ c[k] - cn * n[k] };
            const double value{ scale * tangent };
            if (!std::isfinite(value)) {
                throw std::runtime_error("INVALID: bad base rows");
            }
            out.push_back(value);
        }
    }
    return out;
}

std::vector<std::vector<char>> ContainerWriter::chunks() const
{
    const TerrainInput& input{ *parts_.input };
    const TerrainGeometry& geometry{ *parts_.geometry };
    const PermutedMaps& p{ *parts_.permuted };
    const std::vector<TerrainCell>& cells{ geometry.cells() };
    const std::vector<Facet>& facets{ geometry.facets() };
    const auto z_lo{ static_cast<double>(z_lo_) };
    SectionBuffer w{};
    w.i32("box", { input.lx(), input.ly(), input.z_top() - z_lo_ + 1 });
    w.u32("counts", { static_cast<std::int64_t>(cells.size()), static_cast<std::int64_t>(facets.size()), static_cast<std::int64_t>(p.vi.size()) });
    std::vector<std::int64_t> cell_xyz{};
    std::vector<double> centroid{};
    for (const TerrainCell& cell : cells) {
        cell_xyz.insert(cell_xyz.end(), { cell.ijk[0], cell.ijk[1], cell.ijk[2] - z_lo_ });
        centroid.insert(centroid.end(), { cell.centroid.x, cell.centroid.y, cell.centroid.z - z_lo });
    }
    w.i32("cell_xyz", cell_xyz);
    w.f64("volume", parts_.maps->volume);
    w.f64("centroid", centroid);
    std::vector<double> samples{};
    for (const FacetSample& s : parts_.sampler->samples()) {
        samples.insert(samples.end(), { s.point.x, s.point.y, s.point.z - z_lo });
    }
    w.f64("sample_point", samples);
    w.f64("mode_phi", std::vector<double>(cells.size(), 1.0));
    std::vector<double> normal{};
    std::vector<double> area{};
    for (const Facet& f : facets) {
        normal.insert(normal.end(), { f.unit.x, f.unit.y, f.unit.z });
        const std::vector<double> a{ f.area.x, f.area.y, f.area.z };
        area.push_back(std::sqrt(DenseAlgebra::dot(a, a)));
    }
    w.f64("normal", normal);
    w.f64("area", area);
    std::vector<std::int64_t> facet_offsets{};
    for (std::int64_t f{ 0 }; f <= static_cast<std::int64_t>(facets.size()); ++f) {
        facet_offsets.push_back(static_cast<std::int64_t>(std::lower_bound(p.row_facet.begin(), p.row_facet.end(), f) - p.row_facet.begin()));
    }
    w.u32("facet_offsets", facet_offsets);
    w.u32("row_direction", p.row_direction);
    w.f64("prism_volume", p.vi);
    w.u32("row_origin", p.order);
    w.csr("D", p.d);
    w.csr("G", p.g);
    w.csr("R", p.r);
    w.csr("S", p.s);
    w.csr("W", p.w);
    w.f64("scalars", { options_.nu, 1.0 / (3.0 * options_.nu + 0.5), options_.friction });
    w.f64("f0", initial_state());
    w.f64("s0", p.s0);
    w.f64("friction_rows", base_rows(area, normal));
    w.f64("direct_arm", parts_.arms->direct_arm);
    w.f64("prism_arm", parts_.arms->prism_arm);
    w.u32("gradient_cells", parts_.gradient->cells);
    w.u32("gradient_offsets", parts_.gradient->offsets);
    w.u32("gradient_indices", parts_.gradient->indices);
    w.f64("gradient_weights", parts_.gradient->weights);
    if (options_.rough_z0) {
        const RoughLaw law{ *options_.rough_z0, options_.rough_k0 };
        std::vector<double> friction{};
        std::vector<std::int64_t> kind{};
        for (const Facet& f : facets) {
            friction.insert(friction.end(), { f.key.lid ? 0.0 : law.k0, f.key.lid ? 0.0 : law.c });
            kind.push_back(f.key.lid ? 1 : 0);
        }
        w.f64("facet_friction", friction);
        w.u32("facet_kind", kind);
    }
    w.text("provenance", provenance());
    return w.file();
}

std::string ContainerWriter::provenance() const
{
    const TerrainInput& input{ *parts_.input };
    const GradientMap& g{ *parts_.gradient };
    const DiagTotals diag{ parts_.passes->diag() };
    std::int64_t least_squares{ 0 };
    for (const FacetSample& s : parts_.sampler->samples()) {
        least_squares += s.least_squares ? 1 : 0;
    }
    const auto direct_centroid{ static_cast<std::int64_t>(parts_.sampler->samples().size()) - least_squares };
    JsonWriter json{ true };
    json.begin_object();
    json.key("format");
    json.value("VWNAT01");
    json.key("case");
    json.value(input.name());
    json.key("velocity_set");
    json.value(parts_.velocities->name);
    json.key("Lx");
    json.value(input.lx());
    json.key("Ly");
    json.value(input.ly());
    json.key("Z_top");
    json.value(input.z_top());
    json.key("z_lo");
    json.value(z_lo_);
    json.key("box");
    json.begin_list();
    json.value(input.lx());
    json.value(input.ly());
    json.value(input.z_top() - z_lo_ + 1);
    json.end_list();
    json.key("nu");
    json.value(options_.nu);
    json.key("friction_K");
    json.value(options_.friction);
    json.key("f0");
    json.value(options_.initial_velocity ? "feq3(1, " + velocity_list(*options_.initial_velocity) + ") uniform" : std::string{ "rest w_i (rho0 = 1)" });
    json.key("s0");
    json.value("W w_rd (rest prehistory)");
    json.key("identities");
    json.begin_object();
    json.key("donor_partition");
    json.value(parts_.identities.donor_partition);
    json.key("receiver_coverage");
    json.value(parts_.identities.receiver_coverage);
    json.key("prism_coverage");
    json.value(parts_.identities.prism_coverage);
    json.key("return_delivery");
    json.value(parts_.identities.return_delivery);
    json.end_object();
    json.key("gradient_report");
    json.begin_object();
    json.key("reproduction_max");
    json.value(g.reproduction_max);
    json.key("condition_max");
    json.value(g.condition_max);
    json.key("stencil_size_min");
    json.value(g.stencil_size_min);
    json.key("stencil_size_max");
    json.value(g.stencil_size_max);
    json.key("stencils_with_a_distance_tie_at_the_cut");
    json.value(g.ties);
    json.key("smallest_relative_gap_of_squared_distance_after_the_stencil");
    json.value(g.smallest_gap);
    json.end_object();
    json.key("wall_list_cells_added_by_rule_253");
    json.value(static_cast<std::int64_t>(parts_.wall->added_253.size()));
    json.key("builder_diagnostics");
    json.begin_object();
    json.key("max_split_defect");
    json.value(diag.max_defect);
    json.key("sum_split_defect");
    json.value(diag.sum_defect);
    json.key("max_row_or_cell_defect_sum");
    json.value(diag.max_unit_sum);
    json.key("small_pieces");
    json.value(diag.small_count);
    json.key("small_total");
    json.value(diag.small_total);
    json.end_object();
    json.key("sampler_branches");
    json.begin_object();
    json.key("direct_centroid");
    json.value(direct_centroid);
    json.key("least_squares");
    json.value(least_squares);
    json.end_object();
    // the exporter fields name this tool (the Python exporter writes "exporter", "python", "numpy" and "scipy" here)
    json.key("exporter");
    json.value("src/experimental/wall_builder (volumetric_wall_builder)");
    json.key("openblas");
    json.value(OPENBLAS);
    json.key("manifest_sha256");
    json.value(Sha256::of(input.record_json()));
    if (!parts_.wall->added_support.empty()) {
        json.key("wall_list_cells_added_by_sample_support");
        json.value(static_cast<std::int64_t>(parts_.wall->added_support.size()));
        json.key("sample_support_direct_arm_max_abs");
        json.value(parts_.wall->support_arm_max);
    }
    if (options_.rough_z0) {
        const RoughLaw law{ *options_.rough_z0, options_.rough_k0 };
        json.key("rough");
        json.begin_object();
        json.key("z0");
        json.value(law.z0);
        json.key("z0_hex");
        json.value(PythonFloat::hex(law.z0));
        json.key("K0_terrain");
        json.value(law.k0);
        json.key("C_terrain");
        json.value(law.c);
        json.key("C_terrain_hex");
        json.value(PythonFloat::hex(law.c));
        json.key("kappa");
        json.value(RoughLaw::KAPPA);
        json.key("sample_distance");
        json.value(RoughLaw::SAMPLE_DISTANCE);
        json.key("lid");
        json.value("K0 = C = 0");
        json.end_object();
    }
    json.end_object();
    return json.text();
}

} // namespace fluidx3d::experimental::wall_builder
