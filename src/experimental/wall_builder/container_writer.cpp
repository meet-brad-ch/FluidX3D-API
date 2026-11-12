/// @file container_writer.cpp
/// @brief The VWNAT01 container of terrain_container.py, section for section, and its provenance text.
/// Built with FLUIDX3D_PRECISE_FLOAT: every value keeps numpy's operation order.

#include "container_writer.hpp"

#include "dense_algebra.hpp"
#include "python_text.hpp"

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

/// @brief Values of one section collected in a local buffer and handed to the sink in large parts.
class ValueBuffer {
public:
    /// @param sink the file
    /// @param section the section name, for the range errors
    ValueBuffer(ContainerSink& sink, std::string section)
        : sink_{ sink }
        , section_{ std::move(section) }
    {
        bytes_.reserve(PART);
    }

    ValueBuffer(const ValueBuffer&) = delete;
    ValueBuffer& operator=(const ValueBuffer&) = delete;

    /// @brief Appends a float64.
    void f64(const double value) { append(&value, 8); }

    /// @brief Appends an int32. @throws std::runtime_error when it does not fit
    void i32(const std::int64_t value)
    {
        if (value < std::numeric_limits<std::int32_t>::min() || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error("INVALID: " + section_ + " does not fit int32");
        }
        const auto narrow{ static_cast<std::int32_t>(value) };
        append(&narrow, 4);
    }

    /// @brief Appends a uint32. @throws std::runtime_error when it does not fit
    void u32(const std::int64_t value)
    {
        if (value < 0 || value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("INVALID: " + section_ + " does not fit uint32");
        }
        const auto narrow{ static_cast<std::uint32_t>(value) };
        append(&narrow, 4);
    }

    /// @brief Hands the rest to the sink.
    void flush()
    {
        sink_.put(bytes_.data(), bytes_.size());
        bytes_.clear();
    }

private:
    static constexpr std::size_t PART{ std::size_t{ 1 } << 20 };

    ContainerSink& sink_;
    std::string section_{};
    std::vector<char> bytes_{};

    void append(const void* const data, const std::size_t size)
    {
        if (bytes_.size() + size > PART) {
            flush();
        }
        const auto* const raw{ static_cast<const char*>(data) };
        bytes_.insert(bytes_.end(), raw, raw + size);
    }
};

/// @brief The sections of a VWNAT01 file (Writer): name[32], dtype u32, count u64, payload; counts the sections.
class SectionOut {
public:
    explicit SectionOut(ContainerSink& sink)
        : sink_{ sink }
    {
    }

    /// @brief The section header. @throws std::runtime_error for a name longer than 32
    void header(const std::string& name, const std::uint32_t dtype, const std::uint64_t count)
    {
        if (name.size() > 32) {
            throw std::runtime_error("section name longer than 32: " + name);
        }
        std::array<char, 44> head{};
        std::copy(name.begin(), name.end(), head.begin());
        std::memcpy(head.data() + 32, &dtype, 4);
        std::memcpy(head.data() + 36, &count, 8);
        sink_.put(head.data(), head.size());
        written += 1;
    }

    void f64(const std::string& name, const std::vector<double>& values)
    {
        header(name, DTYPE_F64, values.size());
        sink_.put(values.data(), values.size() * 8);
    }

    void i32(const std::string& name, const std::vector<std::int64_t>& values)
    {
        header(name, DTYPE_I32, values.size());
        ValueBuffer out{ sink_, name };
        for (const std::int64_t v : values) {
            out.i32(v);
        }
        out.flush();
    }

    void u32(const std::string& name, const std::vector<std::int64_t>& values)
    {
        header(name, DTYPE_U32, values.size());
        ValueBuffer out{ sink_, name };
        for (const std::int64_t v : values) {
            out.u32(v);
        }
        out.flush();
    }

    void text(const std::string& name, const std::string& value)
    {
        header(name, DTYPE_TEXT, value.size());
        sink_.put(value.data(), value.size());
    }

    void csr(const std::string& name, const SparseMatrix& m)
    {
        u32(name + "_shape", { m.rows, m.columns });
        u32(name + "_offsets", m.offsets);
        u32(name + "_indices", m.indices);
        f64(name + "_weights", m.values);
    }

    /// @brief A section copied from a spool file of the slab stream.
    void spool(const std::string& name, const std::uint32_t dtype, const SpoolFile& file, const std::uint64_t element)
    {
        header(name, dtype, file.count());
        sink_.copy_file(file.path(), file.count() * element);
    }

    std::uint32_t written{}; ///< Sections written.

private:
    ContainerSink& sink_;
};

std::string velocity_list(const std::array<double, 3>& u)
{
    return "[" + PythonFloat::repr(u[0]) + ", " + PythonFloat::repr(u[1]) + ", " + PythonFloat::repr(u[2]) + "]";
}

} // namespace

// ------------------------------------------------------------------------------------------------ ContainerSink

ContainerSink::ContainerSink(const std::filesystem::path& path)
    : path_{ path }
    , file_{ path, std::ios::binary | std::ios::trunc }
{
    if (!file_) {
        throw std::runtime_error("cannot create " + path.string());
    }
    filling_.reserve(BLOCK);
    pending_.reserve(BLOCK);
}

ContainerSink::~ContainerSink()
{
    if (writer_.joinable()) {
        writer_.join();
    }
    if (hasher_.joinable()) {
        hasher_.join();
    }
}

void ContainerSink::wait()
{
    if (writer_.joinable()) {
        writer_.join();
    }
    if (hasher_.joinable()) {
        hasher_.join();
    }
    if (write_error_) {
        std::rethrow_exception(write_error_);
    }
    if (hash_error_) {
        std::rethrow_exception(hash_error_);
    }
}

void ContainerSink::hand_over()
{
    wait();
    std::swap(filling_, pending_);
    filling_.clear();
    writer_ = std::thread{ [this]() {
        try {
            file_.write(pending_.data(), static_cast<std::streamsize>(pending_.size()));
            if (!file_) {
                throw std::runtime_error("cannot write " + path_.string());
            }
        } catch (...) { // a thread boundary: rethrown by wait()
            write_error_ = std::current_exception();
        }
    } };
    hasher_ = std::thread{ [this]() {
        try {
            hash_.update(pending_.data(), pending_.size());
        } catch (...) { // a thread boundary: rethrown by wait()
            hash_error_ = std::current_exception();
        }
    } };
}

void ContainerSink::put(const void* const data, const std::size_t size)
{
    const auto* raw{ static_cast<const char*>(data) };
    std::size_t left{ size };
    while (left > 0) {
        const std::size_t room{ BLOCK - filling_.size() };
        const std::size_t part{ std::min(room, left) };
        filling_.insert(filling_.end(), raw, raw + part);
        raw += part;
        left -= part;
        if (filling_.size() == BLOCK) {
            hand_over();
        }
    }
    bytes_ += size;
}

void ContainerSink::copy_file(const std::filesystem::path& path, const std::uint64_t bytes)
{
    std::ifstream in{ path, std::ios::binary };
    if (!in) {
        throw std::runtime_error("cannot open the temporary file " + path.string());
    }
    std::vector<char> part(BLOCK, 0);
    std::uint64_t left{ bytes };
    while (left > 0) {
        const auto size{ static_cast<std::size_t>(std::min<std::uint64_t>(left, BLOCK)) };
        in.read(part.data(), static_cast<std::streamsize>(size));
        if (!in) {
            throw std::runtime_error("cannot read the temporary file " + path.string());
        }
        put(part.data(), size);
        left -= size;
    }
    const auto at_end{ in.peek() };
    if (at_end != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("the temporary file " + path.string() + " is longer than its section");
    }
}

std::string ContainerSink::finish()
{
    hand_over();
    wait();
    file_.close();
    if (!file_) {
        throw std::runtime_error("cannot close " + path_.string());
    }
    return hash_.finish();
}

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
    z_lo_ = parts.geometry->lowest_layer() - 1;
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

std::vector<double> ContainerWriter::initial_row() const
{
    const std::vector<double>& w{ parts_.velocities->weights };
    const std::size_t q{ w.size() };
    if (!options_.initial_velocity) {
        return w;
    }
    // UniformStart.f0: cu = u @ c.T (numpy's dgemm), uu = (u * u).sum(axis=1), the bracket in numpy's order. Every row of
    // u is the same, so every row of f0 is: a block of rows shows that the product gives them all the same values.
    const std::size_t rows{ std::min<std::size_t>(parts_.geometry->cell_count(), 64) };
    const std::array<double, 3>& velocity{ *options_.initial_velocity };
    Matrix u{ static_cast<int>(rows), 3, std::vector<double>(rows * 3, 0.0) };
    for (std::size_t n{ 0 }; n < rows; ++n) {
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
    for (std::size_t n{ 1 }; n < rows; ++n) {
        const bool same{ std::equal(cu.data.begin(), cu.data.begin() + static_cast<std::ptrdiff_t>(q), cu.data.begin() + static_cast<std::ptrdiff_t>(n * q)) };
        if (!same) {
            throw std::runtime_error("initial_row: the rows of u @ c.T differ; one row does not stand for every cell");
        }
    }
    const double uu{ velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2] };
    std::vector<double> row(q, 0.0);
    for (std::size_t d{ 0 }; d < q; ++d) {
        const double x{ cu.data[d] };
        const double bracket{ 1.0 + 3.0 * x + 4.5 * x * x - 1.5 * uu + 4.5 * (x * x * x - x * uu) };
        const double rho{ 1.0 };
        row[d] = w[d] * rho * bracket;
    }
    return row;
}

std::uint32_t ContainerWriter::section_count() const
{
    // 13 cell and surface sections, 5 maps of 4, scalars f0 s0 friction_rows direct_arm prism_arm, 4 gradient, provenance
    const std::uint32_t base{ 13 + 20 + 6 + 4 + 1 };
    return options_.rough_z0 ? base + 2 : base;
}

std::uint32_t ContainerWriter::cell_sections(ContainerSink& sink) const
{
    const TerrainInput& input{ *parts_.input };
    const TerrainGeometry& geometry{ *parts_.geometry };
    const PermutedMaps& p{ *parts_.permuted };
    const std::size_t cells{ geometry.cell_count() };
    const auto z_lo{ static_cast<double>(z_lo_) };
    SectionOut w{ sink };
    w.i32("box", { input.lx(), input.ly(), input.z_top() - z_lo_ + 1 });
    w.u32("counts", { static_cast<std::int64_t>(cells), static_cast<std::int64_t>(geometry.facets().size()), static_cast<std::int64_t>(p.vi.size()) });
    const int lx{ geometry.record().lx };
    const int ly{ geometry.record().ly };
    const int z_top{ geometry.record().z_top };
    w.header("cell_xyz", DTYPE_I32, cells * 3);
    ValueBuffer xyz{ sink, "cell_xyz" };
    for (int i{ 0 }; i < lx; ++i) {
        for (int j{ 0 }; j < ly; ++j) {
            for (int k{ geometry.first_layer(i, j) }; k < z_top; ++k) {
                xyz.i32(i);
                xyz.i32(j);
                xyz.i32(k - z_lo_);
            }
        }
    }
    xyz.flush();
    w.header("volume", DTYPE_F64, cells);
    ValueBuffer volume{ sink, "volume" };
    for (std::size_t n{ 0 }; n < cells; ++n) {
        volume.f64(geometry.cell_volume(n));
    }
    volume.flush();
    w.header("centroid", DTYPE_F64, cells * 3);
    ValueBuffer centroid{ sink, "centroid" };
    for (std::size_t n{ 0 }; n < cells; ++n) {
        const Vec3& c{ geometry.cell_centroid(n) };
        centroid.f64(c.x);
        centroid.f64(c.y);
        centroid.f64(c.z - z_lo);
    }
    centroid.flush();
    std::vector<double> samples{};
    for (const FacetSample& s : parts_.sampler->samples()) {
        samples.insert(samples.end(), { s.point.x, s.point.y, s.point.z - z_lo });
    }
    w.f64("sample_point", samples);
    w.header("mode_phi", DTYPE_F64, cells);
    ValueBuffer phi{ sink, "mode_phi" };
    for (std::size_t n{ 0 }; n < cells; ++n) {
        phi.f64(1.0);
    }
    phi.flush();
    return w.written;
}

std::uint32_t ContainerWriter::surface_sections(ContainerSink& sink, const std::vector<double>& normal, const std::vector<double>& area) const
{
    const PermutedMaps& p{ *parts_.permuted };
    const std::size_t facets{ parts_.geometry->facets().size() };
    SectionOut w{ sink };
    w.f64("normal", normal);
    w.f64("area", area);
    std::vector<std::int64_t> facet_offsets{};
    for (std::int64_t f{ 0 }; f <= static_cast<std::int64_t>(facets); ++f) {
        facet_offsets.push_back(static_cast<std::int64_t>(std::lower_bound(p.row_facet.begin(), p.row_facet.end(), f) - p.row_facet.begin()));
    }
    w.u32("facet_offsets", facet_offsets);
    w.u32("row_direction", p.row_direction);
    w.f64("prism_volume", p.vi);
    w.u32("row_origin", p.order);
    return w.written;
}

std::uint32_t ContainerWriter::map_sections(ContainerSink& sink) const
{
    const PermutedMaps& p{ *parts_.permuted };
    const SlabStream& stream{ *parts_.stream };
    const auto states{ static_cast<std::int64_t>(parts_.geometry->cell_count() * parts_.velocities->directions.size()) };
    SectionOut w{ sink };
    w.u32("D_shape", { states, states });
    w.spool("D_offsets", DTYPE_U32, stream.spool("D_offsets"), 4);
    w.spool("D_indices", DTYPE_U32, stream.spool("D_indices"), 4);
    w.spool("D_weights", DTYPE_F64, stream.spool("D_weights"), 8);
    w.csr("G", p.g);
    w.u32("R_shape", { p.r.states, p.r.columns });
    w.header("R_offsets", DTYPE_U32, static_cast<std::uint64_t>(states) + 1u);
    ValueBuffer offsets{ sink, "R_offsets" };
    offsets.u32(0);
    std::size_t listed{ 0 };
    std::int64_t end{ 0 };
    for (std::int64_t s{ 0 }; s < states; ++s) {
        if (listed < p.r.state.size() && p.r.state[listed] == s) {
            end = p.r.offsets[listed + 1u];
            ++listed;
        }
        offsets.u32(end);
    }
    offsets.flush();
    w.u32("R_indices", p.r.indices);
    w.f64("R_weights", p.r.values);
    w.csr("S", p.s);
    w.csr("W", p.w);
    return w.written;
}

std::uint32_t ContainerWriter::state_sections(ContainerSink& sink, const std::vector<double>& normal, const std::vector<double>& area) const
{
    const PermutedMaps& p{ *parts_.permuted };
    const std::size_t cells{ parts_.geometry->cell_count() };
    SectionOut w{ sink };
    w.f64("scalars", { options_.nu, 1.0 / (3.0 * options_.nu + 0.5), options_.friction });
    const std::vector<double> row{ initial_row() };
    w.header("f0", DTYPE_F64, cells * row.size());
    for (std::size_t n{ 0 }; n < cells; ++n) {
        sink.put(row.data(), row.size() * 8);
    }
    w.f64("s0", p.s0);
    w.f64("friction_rows", base_rows(area, normal));
    w.spool("direct_arm", DTYPE_F64, parts_.stream->spool("direct_arm"), 8);
    w.f64("prism_arm", parts_.arms->prism_arm);
    return w.written;
}

std::uint32_t ContainerWriter::tail_sections(ContainerSink& sink) const
{
    SectionOut w{ sink };
    w.u32("gradient_cells", parts_.gradient->cells);
    w.u32("gradient_offsets", parts_.gradient->offsets);
    w.u32("gradient_indices", parts_.gradient->indices);
    w.f64("gradient_weights", parts_.gradient->weights);
    if (options_.rough_z0) {
        const RoughLaw law{ *options_.rough_z0, options_.rough_k0 };
        std::vector<double> friction{};
        std::vector<std::int64_t> kind{};
        for (const Facet& f : parts_.geometry->facets()) {
            friction.insert(friction.end(), { f.key.lid ? 0.0 : law.k0, f.key.lid ? 0.0 : law.c });
            kind.push_back(f.key.lid ? 1 : 0);
        }
        w.f64("facet_friction", friction);
        w.u32("facet_kind", kind);
    }
    w.text("provenance", provenance());
    return w.written;
}

std::string ContainerWriter::write(ContainerSink& sink) const
{
    std::array<char, 12> head{};
    const char magic[]{ "VWNAT01\n" };
    std::memcpy(head.data(), magic, 8);
    const std::uint32_t count{ section_count() };
    std::memcpy(head.data() + 8, &count, 4);
    sink.put(head.data(), head.size());
    std::vector<double> normal{};
    std::vector<double> area{};
    for (const Facet& f : parts_.geometry->facets()) {
        normal.insert(normal.end(), { f.unit.x, f.unit.y, f.unit.z });
        const std::vector<double> a{ f.area.x, f.area.y, f.area.z };
        area.push_back(std::sqrt(DenseAlgebra::dot(a, a)));
    }
    std::uint32_t written{ cell_sections(sink) };
    written += surface_sections(sink, normal, area);
    written += map_sections(sink);
    written += state_sections(sink, normal, area);
    written += tail_sections(sink);
    if (written != count) {
        throw std::logic_error("ContainerWriter: " + std::to_string(written) + " sections written, the header says " + std::to_string(count));
    }
    return sink.finish();
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

std::string ContainerWriter::provenance() const
{
    JsonWriter json{ true };
    json.begin_object();
    case_json(json);
    reports_json(json);
    diagnostics_json(json);
    tool_json(json);
    json.end_object();
    return json.text();
}

void ContainerWriter::case_json(JsonWriter& json) const
{
    const TerrainInput& input{ *parts_.input };
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
}

void ContainerWriter::reports_json(JsonWriter& json) const
{
    const GradientMap& g{ *parts_.gradient };
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
}

void ContainerWriter::diagnostics_json(JsonWriter& json) const
{
    const DiagTotals& diag{ parts_.diag };
    std::int64_t least_squares{ 0 };
    for (const FacetSample& s : parts_.sampler->samples()) {
        if (s.least_squares) {
            least_squares += 1;
        }
    }
    const auto direct_centroid{ static_cast<std::int64_t>(parts_.sampler->samples().size()) - least_squares };
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
}

void ContainerWriter::tool_json(JsonWriter& json) const
{
    const TerrainInput& input{ *parts_.input };
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
}

} // namespace fluidx3d::experimental::wall_builder
