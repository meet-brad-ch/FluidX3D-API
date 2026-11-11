// Task 19: host side of the native volumetric wall. See volumetric_wall_native.hpp.
#include "volumetric_wall_native.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fluidx3d::experimental {

namespace {

constexpr char magic[8]{ 'V', 'W', 'N', 'A', 'T', '0', '1', '\n' };
constexpr std::uint32_t dtype_u32{ 1u }, dtype_i32{ 2u }, dtype_f64{ 3u }, dtype_text{ 4u };
constexpr double near_unit_tolerance{ 1e-12 };   // REPORT only; classification is exact (#253)

/// @brief Throw std::invalid_argument(message) when ok is false.
void require(const bool ok, const std::string& message) {
    if(!ok) {
        throw std::invalid_argument("VolumetricWallNative: "+message);
    }
}

}  // namespace

double minimal_image(const double delta, const double period, const std::uint32_t axis) {
    // #399: IEEE remainder is exact (no rounding of delta/period + 1/2 onto the tie); every non-finite input or result
    // is rejected; the result must lie strictly inside (-period/2, +period/2), an exact tie is ambiguous and rejected
    require(std::isfinite(period) && period>0.0, "minimal_image: non-finite or non-positive period on axis "+std::to_string(axis));
    require(std::isfinite(delta), "minimal_image: non-finite displacement on axis "+std::to_string(axis));
    const double wrapped{ std::remainder(delta, period) };
    require(std::isfinite(wrapped) && std::fabs(wrapped)<0.5*period, "sampler stencil displacement at half the box on axis "+std::to_string(axis)+" (ambiguous periodic image)");
    return wrapped;
}

namespace {

/// @brief One section of the container on disk.
struct Section {
    std::string name{};
    std::uint32_t dtype{};
    std::vector<char> bytes{};
};

void write_section(std::ofstream& out, const std::string& name, const std::uint32_t dtype, const void* data, const std::uint64_t count, const std::uint64_t element) {
    char name_bytes[32]{};
    require(name.size()<32u, "section name too long");
    std::memcpy(name_bytes, name.data(), name.size());
    out.write(name_bytes, 32);
    out.write(reinterpret_cast<const char*>(&dtype), 4);
    out.write(reinterpret_cast<const char*>(&count), 8);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count*element));
}

template<class T> void write_vector(std::ofstream& out, const std::string& name, const std::uint32_t dtype, const std::vector<T>& values) {
    write_section(out, name, dtype, values.data(), values.size(), sizeof(T));
}

void write_map(std::ofstream& out, const std::string& name, const NativeSparseMap& map) {
    const std::vector<std::uint32_t> shape{ map.rows, map.columns };
    write_vector(out, name+"_shape", dtype_u32, shape);
    write_vector(out, name+"_offsets", dtype_u32, map.offsets);
    write_vector(out, name+"_indices", dtype_u32, map.indices);
    write_vector(out, name+"_weights", dtype_f64, map.weights);
}

std::map<std::string, Section> read_sections(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if(!in) {
        throw std::runtime_error("VolumetricWallNative: cannot open "+path);
    }
    char head[8]{};
    in.read(head, 8);
    if(std::memcmp(head, magic, 8)!=0) {
        throw std::runtime_error("VolumetricWallNative: not a VWNAT01 file: "+path);
    }
    in.seekg(0, std::ios::end);
    const std::uint64_t file_size{ static_cast<std::uint64_t>(in.tellg()) };
    in.seekg(12, std::ios::beg);
    std::uint64_t consumed{ 12ull };
    in.seekg(8, std::ios::beg);
    std::uint32_t count{};
    in.read(reinterpret_cast<char*>(&count), 4);
    if(count==0u || count>1024u) {
        throw std::runtime_error("VolumetricWallNative: impossible section count");
    }
    std::map<std::string, Section> sections{};
    for(std::uint32_t i=0u; i<count; i++) {
        char name_bytes[33]{};
        in.read(name_bytes, 32);
        Section section{};
        section.name = std::string(name_bytes);
        in.read(reinterpret_cast<char*>(&section.dtype), 4);
        std::uint64_t elements{};
        in.read(reinterpret_cast<char*>(&elements), 8);
        consumed += 44ull;
        if(!in || section.name.empty()) {
            throw std::runtime_error("VolumetricWallNative: truncated section header");
        }
        if(section.dtype!=dtype_u32 && section.dtype!=dtype_i32 && section.dtype!=dtype_f64 && section.dtype!=dtype_text) {
            throw std::runtime_error("VolumetricWallNative: unknown dtype of section "+section.name);
        }
        const std::uint64_t element_size{ section.dtype==dtype_f64 ? 8ull : (section.dtype==dtype_text ? 1ull : 4ull) };
        if(elements>(file_size-consumed)/element_size) {   // impossible length: refused before any allocation
            throw std::runtime_error("VolumetricWallNative: impossible length of section "+section.name);
        }
        section.bytes.resize(static_cast<size_t>(elements*element_size));
        in.read(section.bytes.data(), static_cast<std::streamsize>(section.bytes.size()));
        consumed += elements*element_size;
        if(!in) {
            throw std::runtime_error("VolumetricWallNative: truncated section "+section.name);
        }
        if(sections.count(section.name)) {
            throw std::runtime_error("VolumetricWallNative: duplicate section "+section.name);
        }
        sections[section.name] = std::move(section);
    }
    if(consumed!=file_size) {
        throw std::runtime_error("VolumetricWallNative: trailing bytes after the last section");
    }
    return sections;
}

template<class T> std::vector<T> take(std::map<std::string, Section>& sections, const std::string& name, const std::uint32_t dtype) {
    const auto it{ sections.find(name) };
    if(it==sections.end()) {
        throw std::runtime_error("VolumetricWallNative: missing section "+name);
    }
    if(it->second.dtype!=dtype) {
        throw std::runtime_error("VolumetricWallNative: wrong dtype of section "+name);
    }
    std::vector<T> values(it->second.bytes.size()/sizeof(T));
    std::memcpy(values.data(), it->second.bytes.data(), it->second.bytes.size());
    sections.erase(it);
    return values;
}

NativeSparseMap take_map(std::map<std::string, Section>& sections, const std::string& name) {
    NativeSparseMap map{};
    const std::vector<std::uint32_t> shape{ take<std::uint32_t>(sections, name+"_shape", dtype_u32) };
    if(shape.size()!=2u) {
        throw std::runtime_error("VolumetricWallNative: bad shape of map "+name);
    }
    map.rows = shape[0];
    map.columns = shape[1];
    map.offsets = take<std::uint32_t>(sections, name+"_offsets", dtype_u32);
    map.indices = take<std::uint32_t>(sections, name+"_indices", dtype_u32);
    map.weights = take<double>(sections, name+"_weights", dtype_f64);
    return map;
}

/// @brief Shape, offsets, index range and finite weights of one map.
void check_map(const NativeSparseMap& map, const std::uint32_t rows, const std::uint32_t columns, const std::string& name) {
    require(map.rows==rows && map.columns==columns, "map "+name+" dimensions");
    require(map.offsets.size()==static_cast<size_t>(rows)+1u && map.offsets.front()==0u, "map "+name+" offsets size");
    require(map.offsets.back()==map.indices.size() && map.indices.size()==map.weights.size(), "map "+name+" entry count");
    for(std::uint32_t i=0u; i<rows; i++) {
        require(map.offsets[i]<=map.offsets[i+1u], "map "+name+" offsets order");
    }
    for(size_t z=0u; z<map.weights.size(); z++) {
        require(map.indices[z]<columns && std::isfinite(map.weights[z]), "map "+name+" entry "+std::to_string(z));
    }
}

}  // namespace

// ---------------------------------------------------------------- VolumetricWallNativeInput

VolumetricWallNativeInput VolumetricWallNativeInput::read(const std::string& path) {
    std::map<std::string, Section> sections{ read_sections(path) };
    VolumetricWallNativeInput in{};
    const std::vector<std::int32_t> box{ take<std::int32_t>(sections, "box", dtype_i32) };
    if(box.size()!=3u) {
        throw std::runtime_error("VolumetricWallNative: box must have 3 entries");
    }
    in.box = { box[0], box[1], box[2] };
    const std::vector<std::uint32_t> counts{ take<std::uint32_t>(sections, "counts", dtype_u32) };
    if(counts.size()!=3u) {
        throw std::runtime_error("VolumetricWallNative: counts must have 3 entries");
    }
    in.cells = counts[0];
    in.facets = counts[1];
    in.rows = counts[2];
    in.cell_xyz = take<std::int32_t>(sections, "cell_xyz", dtype_i32);
    in.volume = take<double>(sections, "volume", dtype_f64);
    in.centroid = take<double>(sections, "centroid", dtype_f64);
    in.sample_point = take<double>(sections, "sample_point", dtype_f64);
    in.mode_phi = take<double>(sections, "mode_phi", dtype_f64);
    in.normal = take<double>(sections, "normal", dtype_f64);
    in.area = take<double>(sections, "area", dtype_f64);
    in.facet_offsets = take<std::uint32_t>(sections, "facet_offsets", dtype_u32);
    in.row_direction = take<std::uint32_t>(sections, "row_direction", dtype_u32);
    in.prism_volume = take<double>(sections, "prism_volume", dtype_f64);
    in.row_origin = take<std::uint32_t>(sections, "row_origin", dtype_u32);
    in.D = take_map(sections, "D");
    in.G = take_map(sections, "G");
    in.R = take_map(sections, "R");
    in.S = take_map(sections, "S");
    in.W = take_map(sections, "W");
    const std::vector<double> scalars{ take<double>(sections, "scalars", dtype_f64) };
    if(scalars.size()!=3u) {
        throw std::runtime_error("VolumetricWallNative: scalars must be viscosity, omega, friction");
    }
    in.viscosity = scalars[0];
    in.omega = scalars[1];
    in.friction = scalars[2];
    in.f0 = take<double>(sections, "f0", dtype_f64);
    in.s0 = take<double>(sections, "s0", dtype_f64);
    // task 98: q from the D shape (states = cells*q); only D3Q19 and D3Q27 are supported
    if(in.cells==0u || in.D.rows%in.cells!=0u || (in.D.rows/in.cells!=19u && in.D.rows/in.cells!=27u)) {
        throw std::runtime_error("VolumetricWallNative: D has "+std::to_string(in.D.rows)+" states for "+std::to_string(in.cells)+" cells: not D3Q19 or D3Q27");
    }
    in.velocity_set = in.D.rows/in.cells;
    if(sections.count("friction_rows")) {   // task 43: optional; a container without it stays a valid VWNAT01 file
        in.friction_rows = take<double>(sections, "friction_rows", dtype_f64);
        if(in.friction_rows.empty()) {
            throw std::runtime_error("VolumetricWallNative: empty section friction_rows");
        }
    }
    {   // task 55: the six optional sections of the affine candidate, all or none
        const std::array<const char*, 6> names{ "direct_arm", "prism_arm", "gradient_cells", "gradient_offsets", "gradient_indices", "gradient_weights" };
        std::uint32_t present{ 0u };
        for(const char* name : names) {
            present += sections.count(name) ? 1u : 0u;
        }
        if(present!=0u && present!=6u) {
            throw std::runtime_error("VolumetricWallNative: the affine sections are incomplete (all six or none)");
        }
        if(present==6u) {
            in.direct_arm = take<double>(sections, "direct_arm", dtype_f64);
            in.prism_arm = take<double>(sections, "prism_arm", dtype_f64);
            in.gradient_cells = take<std::uint32_t>(sections, "gradient_cells", dtype_u32);
            in.gradient_offsets = take<std::uint32_t>(sections, "gradient_offsets", dtype_u32);
            in.gradient_indices = take<std::uint32_t>(sections, "gradient_indices", dtype_u32);
            in.gradient_weights = take<double>(sections, "gradient_weights", dtype_f64);
            if(in.gradient_cells.empty()) {
                throw std::runtime_error("VolumetricWallNative: empty section gradient_cells");
            }
        }
    }
    const std::vector<char> text{ take<char>(sections, "provenance", dtype_text) };
    in.provenance.assign(text.begin(), text.end());
    if(!sections.empty()) {
        throw std::runtime_error("VolumetricWallNative: unknown section "+sections.begin()->first);
    }
    return in;
}

void VolumetricWallNativeInput::write(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if(!out) {
        throw std::runtime_error("VolumetricWallNative: cannot write "+path);
    }
    out.write(magic, 8);
    const std::uint32_t count{ (friction_rows.empty() ? 37u : 38u)+(affine() ? 6u : 0u) };   // 38 with the optional friction_rows (task 43); box, counts, cell_xyz, volume, centroid, sample_point, mode_phi, normal, area, facet_offsets, row_direction, prism_volume, row_origin (13) + 5 maps x 4 + scalars, f0, s0, provenance (4); 6 more with the affine sections (task 55)
    out.write(reinterpret_cast<const char*>(&count), 4);
    const std::vector<std::int32_t> box{ this->box[0], this->box[1], this->box[2] };
    write_vector(out, "box", dtype_i32, box);
    const std::vector<std::uint32_t> counts{ cells, facets, rows };
    write_vector(out, "counts", dtype_u32, counts);
    write_vector(out, "cell_xyz", dtype_i32, cell_xyz);
    write_vector(out, "volume", dtype_f64, volume);
    write_vector(out, "centroid", dtype_f64, centroid);
    write_vector(out, "sample_point", dtype_f64, sample_point);
    write_vector(out, "mode_phi", dtype_f64, mode_phi);
    write_vector(out, "normal", dtype_f64, normal);
    write_vector(out, "area", dtype_f64, area);
    write_vector(out, "facet_offsets", dtype_u32, facet_offsets);
    write_vector(out, "row_direction", dtype_u32, row_direction);
    write_vector(out, "prism_volume", dtype_f64, prism_volume);
    write_vector(out, "row_origin", dtype_u32, row_origin);
    write_map(out, "D", D);
    write_map(out, "G", G);
    write_map(out, "R", R);
    write_map(out, "S", S);
    write_map(out, "W", W);
    const std::vector<double> scalars{ viscosity, omega, friction };
    write_vector(out, "scalars", dtype_f64, scalars);
    write_vector(out, "f0", dtype_f64, f0);
    write_vector(out, "s0", dtype_f64, s0);
    if(!friction_rows.empty()) {
        write_vector(out, "friction_rows", dtype_f64, friction_rows);
    }
    if(affine()) {   // task 55
        write_vector(out, "direct_arm", dtype_f64, direct_arm);
        write_vector(out, "prism_arm", dtype_f64, prism_arm);
        write_vector(out, "gradient_cells", dtype_u32, gradient_cells);
        write_vector(out, "gradient_offsets", dtype_u32, gradient_offsets);
        write_vector(out, "gradient_indices", dtype_u32, gradient_indices);
        write_vector(out, "gradient_weights", dtype_f64, gradient_weights);
    }
    write_section(out, "provenance", dtype_text, provenance.data(), provenance.size(), 1u);
    if(!out) {
        throw std::runtime_error("VolumetricWallNative: write failed "+path);
    }
}

void VolumetricWallNativeInput::validate() const {
    require(box[0]>0 && box[1]>0 && box[2]>0, "box dimensions");
    require(cells>0u && facets>0u && rows>0u && cells<0xFFFFFFFFu/velocity_set, "counts");
    require(cell_xyz.size()==static_cast<size_t>(cells)*3u, "cell_xyz size");
    for(std::uint32_t c=0u; c<cells; c++) {
        for(std::uint32_t a=0u; a<3u; a++) {
            const std::int32_t v{ cell_xyz[3u*c+a] };
            require(v>=0 && v<box[a], "cell "+std::to_string(c)+" outside the box");
        }
    }
    require(volume.size()==cells, "volume size");
    for(std::uint32_t c=0u; c<cells; c++) {
        require(std::isfinite(volume[c]) && volume[c]>0.0 && volume[c]<=1.0, "volume of cell "+std::to_string(c)+" not in (0, 1]");
    }
    require(normal.size()==static_cast<size_t>(facets)*3u && area.size()==facets, "facet geometry sizes");
    for(std::uint32_t a=0u; a<facets; a++) {
        const double nx{ normal[3u*a] }, ny{ normal[3u*a+1u] }, nz{ normal[3u*a+2u] };
        const double length{ std::sqrt(nx*nx+ny*ny+nz*nz) };
        require(std::isfinite(length) && std::fabs(length-1.0)<=1e-12, "normal of facet "+std::to_string(a)+" not unit");
        require(std::isfinite(area[a]) && area[a]>0.0, "area of facet "+std::to_string(a));
    }
    require(facet_offsets.size()==static_cast<size_t>(facets)+1u && facet_offsets.front()==0u && facet_offsets.back()==rows, "facet_offsets");
    require(row_direction.size()==rows && prism_volume.size()==rows && row_origin.size()==rows, "row arrays");
    {
        std::vector<std::uint32_t> origin_sorted{ row_origin };
        std::sort(origin_sorted.begin(), origin_sorted.end());
        for(std::uint32_t r=0u; r<rows; r++) {
            require(origin_sorted[r]==r, "row_origin is not a permutation");
        }
    }
    for(std::uint32_t r=0u; r<rows; r++) {
        require(row_direction[r]<velocity_set && row_direction[r]!=rest_direction(), "row direction "+std::to_string(r));
        require(std::isfinite(prism_volume[r]) && prism_volume[r]>0.0, "prism volume of row "+std::to_string(r));
    }
    check_map(D, states(), states(), "D");
    check_map(G, rows, states(), "G");
    check_map(R, states(), rows, "R");
    check_map(S, facets, cells, "S");
    check_map(W, rows, rows, "W");
    require(std::isfinite(viscosity) && viscosity>0.0, "viscosity");
    require(std::isfinite(omega) && omega>0.0 && omega<2.0, "omega");
    require(std::isfinite(friction) && friction>=0.0, "friction");
    require(f0.size()==states(), "f0 size");
    for(const double v : f0) {
        require(std::isfinite(v) && v>=0.0, "f0 value");
    }
    for(std::uint32_t c=0u; c<cells; c++) {
        double density{ 0.0 };
        for(std::uint32_t d=0u; d<velocity_set; d++) {
            density += f0[c*velocity_set+d];
        }
        require(density>0.0, "initial density of cell "+std::to_string(c)+" not positive");
    }
    require(s0.size()==rows, "s0 size");
    for(const double v : s0) {
        require(std::isfinite(v) && v>=0.0, "s0 value negative or not finite");
    }
    if(!friction_rows.empty()) {   // task 43: structural part; the identities are validate_friction_rows()
        require(friction_rows.size()==static_cast<size_t>(rows)*3u, "friction_rows size is not rows*3");
        for(const double v : friction_rows) {
            require(std::isfinite(v), "friction_rows value not finite");
        }
    }
    require(centroid.size()==static_cast<size_t>(cells)*3u && sample_point.size()==static_cast<size_t>(facets)*3u, "centroid/sample_point sizes");
    for(const double v : centroid) {
        require(std::isfinite(v), "centroid value");
    }
    for(const double v : sample_point) {
        require(std::isfinite(v), "sample_point value");
    }
    require(mode_phi.size()==cells, "mode_phi size");
    double mode_norm{ 0.0 };
    for(std::uint32_t c=0u; c<cells; c++) {
        require(std::isfinite(mode_phi[c]), "mode_phi value");
        mode_norm += volume[c]*mode_phi[c]*mode_phi[c];
    }
    require(mode_norm>0.0, "mode_phi has zero norm");
    validate_physical();
}

namespace {

constexpr double identity_tolerance{ 1e-10 };     // CF10 GATES static_routing / static_quadrature (cases.py)
constexpr double sampler_tolerance{ 1e-10 };      // CF10 GATES sampler_reproduction
constexpr double orientation_tolerance{ 1e-12 };  // vi = A (c.n), the registered float64 product
constexpr double omega_tolerance{ 1e-12 };        // omega = 1/(3 nu + 1/2)
constexpr double friction_row_tolerance{ 1e-10 }; // task 43: coefficient identities of friction_rows, residual / facet area (the tolerance of the other container identities)

}  // namespace

void VolumetricWallNativeInput::validate_friction_rows() const {
    check_friction_rows(true);
}

void VolumetricWallNativeInput::validate_affine() const {
    require(affine(), "the container has no affine sections");
    require(direct_arm.size()==D.indices.size()*3u, "direct_arm size is not 3 per D entry");
    require(prism_arm.size()==G.indices.size()*3u, "prism_arm size is not 3 per G entry");
    const double arm_limit{ 1.7320508075688773+1e-9 };   // a part centroid and the cell centroid lie in one unit cell
    for(const double v : direct_arm) {
        require(std::isfinite(v) && std::fabs(v)<=arm_limit, "direct_arm value not finite or longer than the cell diagonal");
    }
    for(const double v : prism_arm) {
        require(std::isfinite(v) && std::fabs(v)<=arm_limit, "prism_arm value not finite or longer than the cell diagonal");
    }
    const size_t gradient_count{ gradient_cells.size() };
    require(gradient_offsets.size()==gradient_count*3u+1u && gradient_offsets.front()==0u, "gradient_offsets size");
    require(gradient_offsets.back()==gradient_indices.size() && gradient_indices.size()==gradient_weights.size(), "gradient entry count");
    std::vector<std::uint8_t> seen(cells, 0u);
    for(size_t g=0u; g<gradient_count; g++) {
        const std::uint32_t cell{ gradient_cells[g] };
        require(cell<cells && seen[cell]==0u, "gradient_cells entry outside the mesh or twice");
        seen[cell] = 1u;
        const std::uint32_t begin{ gradient_offsets[3u*g] };
        require(gradient_offsets[3u*g+1u]>begin, "empty gradient row");
        const std::uint32_t length{ gradient_offsets[3u*g+1u]-begin };
        for(std::uint32_t axis=0u; axis<3u; axis++) {
            const std::uint32_t row_begin{ gradient_offsets[3u*g+axis] };
            require(gradient_offsets[3u*g+axis+1u]>=row_begin && gradient_offsets[3u*g+axis+1u]-row_begin==length, "the three gradient rows of a cell differ in length");
            require(row_begin+length<=gradient_indices.size(), "gradient row outside the entry list");
            double sum{ 0.0 }, magnitude{ 0.0 };
            for(std::uint32_t k=0u; k<length; k++) {
                require(gradient_indices[row_begin+k]<cells && gradient_indices[row_begin+k]==gradient_indices[begin+k], "gradient stencil cell outside the mesh, or the three rows of a cell have different stencils");
                require(std::isfinite(gradient_weights[row_begin+k]), "gradient weight not finite");
                sum += gradient_weights[row_begin+k];
                magnitude += std::fabs(gradient_weights[row_begin+k]);
            }
            require(std::fabs(sum)<=identity_tolerance*std::max(1.0, magnitude), "a gradient row does not sum to zero");
        }
        require(gradient_indices[begin]==cell, "the first stencil entry of a gradient row is not the own cell");
    }
    // task 98 (#1991): every row needs a fluid gather part. A fully occluded row (sum of its G volumes = 0) has no prism mean
    // of the centred scatter (0/0 in the kernels); such geometry is unsupported and refused here, before any kernel runs.
    for(std::uint32_t row=0u; row<rows; row++) {
        double gathered{ 0.0 };
        for(std::uint32_t z=G.offsets[row]; z<G.offsets[row+1u]; z++) gathered += G.weights[z];
        require(gathered>0.0, "row "+std::to_string(row)+" has no fluid gather part (fully occluded prism): unsupported geometry");
    }
    check_friction_rows(false);
}

void VolumetricWallNativeInput::check_friction_rows(const bool zero_shell_sum) const {
    require(!friction_rows.empty(), "the container has no friction_rows section (the supported path needs geometry-bound friction rows; see vwnat_rows.py)");
    require(friction_rows.size()==static_cast<size_t>(rows)*3u, "friction_rows size is not rows*3");
    const NativeDirections dir(velocity_set);
    FrictionRowResiduals worst{};
    for(std::uint32_t a=0u; a<facets; a++) {
        const std::array<double, 3> n{ normal[3u*a], normal[3u*a+1u], normal[3u*a+2u] };
        const double facet_area{ area[a] };
        std::array<std::array<double, 3>, 3> shell_sum{};      // [shell - 1][component] (task 98: shells 1, 2, 3)
        std::array<std::array<double, 3>, 3> tensor{};         // [i][j] = sum_r c_r,i phi_r,j
        double normal_component{ 0.0 };
        for(std::uint32_t row=facet_offsets[a]; row<facet_offsets[a+1u]; row++) {
            const std::uint32_t d{ row_direction[row] };
            const std::array<double, 3> c{ static_cast<double>(dir.lex_c(d, 0u)), static_cast<double>(dir.lex_c(d, 1u)), static_cast<double>(dir.lex_c(d, 2u)) };
            const std::array<double, 3> phi{ friction_rows[3u*row], friction_rows[3u*row+1u], friction_rows[3u*row+2u] };
            const double speed2{ c[0]*c[0]+c[1]*c[1]+c[2]*c[2] };
            const std::uint32_t shell{ speed2==1.0 ? 0u : (speed2==2.0 ? 1u : 2u) };
            double phi_n{ 0.0 };
            for(std::uint32_t j=0u; j<3u; j++) {
                require(std::isfinite(phi[j]), "friction_rows value not finite");
                shell_sum[shell][j] += phi[j];
                phi_n += phi[j]*n[j];
                for(std::uint32_t i=0u; i<3u; i++) {
                    tensor[i][j] += c[i]*phi[j];
                }
            }
            normal_component = std::max(normal_component, std::fabs(phi_n));
        }
        double shell_residual{ 0.0 }, tensor_residual{ 0.0 };
        for(std::uint32_t j=0u; j<3u; j++) {
            shell_residual = std::max(shell_residual, std::max(std::fabs(shell_sum[0][j]), std::max(std::fabs(shell_sum[1][j]), std::fabs(shell_sum[2][j]))));
            for(std::uint32_t i=0u; i<3u; i++) {
                const double target{ facet_area/3.0*((i==j ? 1.0 : 0.0)-n[i]*n[j]) };
                tensor_residual = std::max(tensor_residual, std::fabs(tensor[i][j]-target));
            }
        }
        const std::string facet_text{ "friction_rows of facet "+std::to_string(a) };
        require(!zero_shell_sum || shell_residual<=friction_row_tolerance*facet_area, facet_text+": nonzero shell sum (rows with shell mass, or another geometry / row order)");
        require(tensor_residual<=friction_row_tolerance*facet_area, facet_text+": sum c (x) phi differs from (A/3)(I - n n^T) (another geometry / row order)");
        require(normal_component<=friction_row_tolerance*facet_area, facet_text+": a row has a normal component");
        worst.shell_sum = std::max(worst.shell_sum, shell_residual/facet_area);
        worst.impulse_tensor = std::max(worst.impulse_tensor, tensor_residual/facet_area);
        worst.normal_component = std::max(worst.normal_component, normal_component/facet_area);
    }
    friction_row_residuals = worst;
}

void VolumetricWallNativeInput::validate_physical() const {
    const NativeDirections dir(velocity_set);
    const std::uint32_t Q{ velocity_set };
    // nonnegative transport maps (the sampler S is signed by construction)
    for(const double w : D.weights) { require(w>=0.0, "D weight negative"); }
    for(const double w : G.weights) { require(w>=0.0, "G weight negative"); }
    for(const double w : R.weights) { require(w>=0.0, "R weight negative"); }
    for(const double w : W.weights) { require(w>=0.0, "W weight negative"); }
    // facets: nonempty bounded row sets, unique outgoing directions, orientation vi = A (c.n) > 0, both shells present
    std::vector<std::uint32_t> facet_of_row(rows);
    double orientation{ 0.0 };
    for(std::uint32_t a=0u; a<facets; a++) {
        const std::uint32_t begin{ facet_offsets[a] }, end{ facet_offsets[a+1u] };
        require(begin<=end && end<=rows, "facet "+std::to_string(a)+" offsets outside [0, rows] or decreasing");   // before any indexing (#269)
        require(end>begin && end-begin<=Q, "facet "+std::to_string(a)+" has an empty or oversized row set");
        bool shell1{ false }, shell2{ false };   // per facet: each shell balance needs a nonempty denominator (#269)
        std::set<std::uint32_t> seen{};
        for(std::uint32_t row=begin; row<end; row++) {
            facet_of_row[row] = a;
            const std::uint32_t d{ row_direction[row] };
            require(seen.insert(d).second, "facet "+std::to_string(a)+" repeats outgoing direction "+std::to_string(d));
            const double cn{ dir.lex_c(d, 0)*normal[3u*a]+dir.lex_c(d, 1)*normal[3u*a+1u]+dir.lex_c(d, 2)*normal[3u*a+2u] };
            require(cn>0.0, "row "+std::to_string(row)+" is not outgoing (c.n <= 0)");
            const double expected{ area[a]*cn };
            orientation = std::max(orientation, std::fabs(prism_volume[row]-expected)/expected);
            const int shell{ dir.lex_c(d, 0)*dir.lex_c(d, 0)+dir.lex_c(d, 1)*dir.lex_c(d, 1)+dir.lex_c(d, 2)*dir.lex_c(d, 2) };
            shell1 = shell1 || shell==1;
            shell2 = shell2 || shell==2;
        }
        require(shell1 && shell2, "facet "+std::to_string(a)+" rows do not cover both speed shells");
    }
    residuals.orientation = orientation;
    require(orientation<=orientation_tolerance, "prism volumes differ from A (c.n)");
    // direction ownership: D keeps the direction; G gathers OPP(rd) of the donor; R delivers rd; W: producer outgoing = OPP(receiver outgoing)
    for(std::uint32_t state=0u; state<states(); state++) {
        for(std::uint32_t z=D.offsets[state]; z<D.offsets[state+1u]; z++) {
            require(D.indices[z]%Q==state%Q, "D entry changes the direction");
        }
        for(std::uint32_t z=R.offsets[state]; z<R.offsets[state+1u]; z++) {
            require(row_direction[R.indices[z]]==state%Q, "R entry delivers a row of another direction");
        }
    }
    for(std::uint32_t row=0u; row<rows; row++) {
        for(std::uint32_t z=G.offsets[row]; z<G.offsets[row+1u]; z++) {
            require(G.indices[z]%Q==dir.lex_opposite(row_direction[row]), "G entry gathers a direction other than OPP(outgoing)");
        }
        for(std::uint32_t z=W.offsets[row]; z<W.offsets[row+1u]; z++) {
            require(row_direction[W.indices[z]]==dir.lex_opposite(row_direction[row]), "W entry couples rows that are not opposite");
        }
    }
    // donor partition v^T D + 1^T G = v^T   (amount leaving every donor state)
    std::vector<double> leaving(states(), 0.0);
    for(std::uint32_t receiver=0u; receiver<states(); receiver++) {
        for(std::uint32_t z=D.offsets[receiver]; z<D.offsets[receiver+1u]; z++) {
            leaving[D.indices[z]] += volume[receiver/Q]*D.weights[z];
        }
    }
    for(std::uint32_t row=0u; row<rows; row++) {
        for(std::uint32_t z=G.offsets[row]; z<G.offsets[row+1u]; z++) {
            leaving[G.indices[z]] += G.weights[z];
        }
    }
    double donor_partition{ 0.0 };
    for(std::uint32_t state=0u; state<states(); state++) {
        donor_partition = std::max(donor_partition, std::fabs(leaving[state]-volume[state/Q])/volume[state/Q]);
    }
    residuals.donor_partition = donor_partition;
    require(donor_partition<=identity_tolerance, "donor partition v^T D + 1^T G = v^T violated");
    // receiver coverage D 1 + R vi = 1 per state
    double receiver_coverage{ 0.0 };
    for(std::uint32_t state=0u; state<states(); state++) {
        double covered{ 0.0 };
        for(std::uint32_t z=D.offsets[state]; z<D.offsets[state+1u]; z++) { covered += D.weights[z]; }
        for(std::uint32_t z=R.offsets[state]; z<R.offsets[state+1u]; z++) { covered += R.weights[z]*prism_volume[R.indices[z]]; }
        receiver_coverage = std::max(receiver_coverage, std::fabs(covered-1.0));
    }
    residuals.receiver_coverage = receiver_coverage;
    require(receiver_coverage<=identity_tolerance, "receiver coverage D 1 + R vi = 1 violated");
    // prism coverage G 1 + W 1 = vi per row, return delivery v^T R + 1^T W diag(1/vi) = 1 per row
    std::vector<double> delivered(rows, 0.0);
    for(std::uint32_t state=0u; state<states(); state++) {
        for(std::uint32_t z=R.offsets[state]; z<R.offsets[state+1u]; z++) {
            delivered[R.indices[z]] += volume[state/Q]*R.weights[z];
        }
    }
    double prism_coverage{ 0.0 }, return_delivery{ 0.0 };
    for(std::uint32_t row=0u; row<rows; row++) {
        double gathered{ 0.0 };
        for(std::uint32_t z=G.offsets[row]; z<G.offsets[row+1u]; z++) { gathered += G.weights[z]; }
        for(std::uint32_t z=W.offsets[row]; z<W.offsets[row+1u]; z++) {
            gathered += W.weights[z];
            delivered[W.indices[z]] += W.weights[z]/prism_volume[W.indices[z]];
        }
        prism_coverage = std::max(prism_coverage, std::fabs(gathered-prism_volume[row])/prism_volume[row]);
    }
    for(std::uint32_t row=0u; row<rows; row++) {
        return_delivery = std::max(return_delivery, std::fabs(delivered[row]-1.0));
    }
    residuals.prism_coverage = prism_coverage;
    residuals.return_delivery = return_delivery;
    require(prism_coverage<=identity_tolerance, "prism coverage G 1 + W 1 = vi violated");
    require(return_delivery<=identity_tolerance, "return delivery v^T R + 1^T W diag(1/vi) = 1 violated");
    // sampler: constant reproduction and affine target consistency per facet
    double sampler_constant{ 0.0 }, sampler_affine{ 0.0 };
    for(std::uint32_t a=0u; a<facets; a++) {
        require(S.offsets[a+1u]>S.offsets[a], "facet "+std::to_string(a)+" has no sample weights");
        double sum{ 0.0 };
        std::array<double, 3> moment{};
        for(std::uint32_t z=S.offsets[a]; z<S.offsets[a+1u]; z++) {
            sum += S.weights[z];
            for(std::uint32_t axis=0u; axis<3u; axis++) {
                // task 22 (NATIVE_PERIODICITY_CONTRACT): the lattice box is periodic on every axis, so the displacement is
                // the half-up minimal image on every axis; an actual half-period ambiguity is rejected, never resolved
                const double delta{ minimal_image(centroid[3u*S.indices[z]+axis]-sample_point[3u*a+axis], static_cast<double>(box[axis]), axis) };
                moment[axis] += S.weights[z]*delta;
            }
        }
        require(std::isfinite(sum), "facet "+std::to_string(a)+" sampler weight sum not finite");
        sampler_constant = std::max(sampler_constant, std::fabs(sum-1.0));
        for(const double m : moment) {
            require(std::isfinite(m), "facet "+std::to_string(a)+" sampler affine moment not finite");   // #399: std::max would hide a NaN
            sampler_affine = std::max(sampler_affine, std::fabs(m));
        }
    }
    residuals.sampler_constant = sampler_constant;
    residuals.sampler_affine = sampler_affine;
    require(sampler_constant<=sampler_tolerance, "sampler does not reproduce constants");
    require(sampler_affine<=sampler_tolerance, "sampler does not reproduce the affine target");
    // scalars: omega = 1/(3 nu + 1/2); K == 0 is the attachment scope (checked by the LBM, not here)
    residuals.omega = std::fabs(omega-1.0/(3.0*viscosity+0.5));
    require(residuals.omega<=omega_tolerance, "omega inconsistent with nu");
}

// ---------------------------------------------------------------- NativeDirections

NativeDirections::NativeDirections(const std::uint32_t q) : q_(q) {
    require(q==19u || q==27u, "NativeDirections: velocity set "+std::to_string(q)+" is not D3Q19 or D3Q27");
    const int limit{ q==27u ? 3 : 2 };
    std::uint32_t d{ 0u };
    for(int x=-1; x<=1; x++) {
        for(int y=-1; y<=1; y++) {
            for(int z=-1; z<=1; z++) {
                if(x*x+y*y+z*z<=limit) {
                    lex_[d] = { x, y, z };
                    d++;
                }
            }
        }
    }
    require(d==q, "NativeDirections: lexicographic table size");
    // src/kernel.cpp c_data, D3Q27 branch (rows x, y, z); its first 19 entries are the D3Q19 branch, entry by entry
    const int fx_x[27]{ 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1 };
    const int fx_y[27]{ 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1 };
    const int fx_z[27]{ 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1 };
    for(std::uint32_t k=0u; k<q_; k++) {
        fx_[k] = { fx_x[k], fx_y[k], fx_z[k] };
    }
    for(std::uint32_t k=0u; k<q_; k++) {
        std::uint32_t matches{ 0u };
        for(std::uint32_t e=0u; e<q_; e++) {
            if(lex_[e]==fx_[k]) {
                fx_to_lex_[k] = e;
                lex_to_fx_[e] = k;
                matches++;
            }
        }
        require(matches==1u, "direction permutation: FluidX3D direction "+std::to_string(k)+" has "+std::to_string(matches)+" matches");
    }
    for(std::uint32_t e=0u; e<q_; e++) {
        const std::array<int, 3> opposite{ -lex_[e][0], -lex_[e][1], -lex_[e][2] };
        std::uint32_t matches{ 0u };
        for(std::uint32_t o=0u; o<q_; o++) {
            if(lex_[o]==opposite) {
                lex_opp_[e] = o;
                matches++;
            }
        }
        require(matches==1u, "lexicographic opposite not unique");
    }
    for(std::uint32_t i=1u; i<q_; i+=2u) {   // FluidX3D opposite pairs (i, i+1) for odd i, as load_f/store_f assume
        const std::array<int, 3> opposite{ -fx_[i][0], -fx_[i][1], -fx_[i][2] };
        require(fx_[i+1u]==opposite, "FluidX3D pair ("+std::to_string(i)+", "+std::to_string(i+1u)+") is not opposite");
        require(lex_opp_[fx_to_lex_[i]]==fx_to_lex_[i+1u], "opposites disagree between the tables");
    }
}

double NativeDirections::rest_weight_lex(const std::uint32_t d) {
    // task 20: the three rest weights computed once from the tables (the same expressions 1.0/3.0, 1.0/18.0, 1.0/36.0
    // as before); the former per-call construction of a NativeDirections dominated the host diagnostics
    static const std::array<double, 19> table{ [] {
        NativeDirections tables{};
        std::array<double, 19> w{};
        for(std::uint32_t i=0u; i<19u; i++) {
            const int shell{ tables.lex_c(i, 0)*tables.lex_c(i, 0)+tables.lex_c(i, 1)*tables.lex_c(i, 1)+tables.lex_c(i, 2)*tables.lex_c(i, 2) };
            w[i] = shell==0 ? 1.0/3.0 : (shell==1 ? 1.0/18.0 : 1.0/36.0);
        }
        return w;
    }() };
    return table[d];
}

double NativeDirections::weight_lex(const std::uint32_t d) const {
    const int shell{ lex_[d][0]*lex_[d][0]+lex_[d][1]*lex_[d][1]+lex_[d][2]*lex_[d][2] };
    if(q_==27u) {   // the doubles of vw_weight (kernel.cpp, D3Q27 branch)
        return shell==0 ? 8.0/27.0 : (shell==1 ? 2.0/27.0 : (shell==2 ? 1.0/54.0 : 1.0/216.0));
    }
    return shell==0 ? 1.0/3.0 : (shell==1 ? 1.0/18.0 : 1.0/36.0);
}

std::vector<int> NativeDirections::fx_table_flat() const {
    std::vector<int> flat(3u*q_, 0);
    for(std::uint32_t axis=0u; axis<3u; axis++) {
        for(std::uint32_t k=0u; k<q_; k++) {
            flat[axis*q_+k] = fx_[k][axis];
        }
    }
    return flat;
}

// ---------------------------------------------------------------- NativeAddresses

NativeAddresses::NativeAddresses(const std::array<std::int32_t, 3>& box, const NativeDirections& directions) : dir_(directions) {   // copied
    box_ = { box[0], box[1], box[2] };
    N_ = static_cast<std::uint64_t>(box_[0])*static_cast<std::uint64_t>(box_[1])*static_cast<std::uint64_t>(box_[2]);
}

std::uint64_t NativeAddresses::cell(std::int64_t x, std::int64_t y, std::int64_t z) const {
    x = ((x%box_[0])+box_[0])%box_[0];
    y = ((y%box_[1])+box_[1])%box_[1];
    z = ((z%box_[2])+box_[2])%box_[2];
    return static_cast<std::uint64_t>(x+box_[0]*(y+box_[1]*z));   // n = x + (y + z*Ny)*Nx, as kernel.cpp index()
}

std::array<std::int64_t, 3> NativeAddresses::xyz(const std::uint64_t n) const {
    const std::int64_t x{ static_cast<std::int64_t>(n%static_cast<std::uint64_t>(box_[0])) };
    const std::int64_t y{ static_cast<std::int64_t>((n/static_cast<std::uint64_t>(box_[0]))%static_cast<std::uint64_t>(box_[1])) };
    const std::int64_t z{ static_cast<std::int64_t>(n/(static_cast<std::uint64_t>(box_[0])*static_cast<std::uint64_t>(box_[1]))) };
    return { x, y, z };
}

std::uint64_t NativeAddresses::neighbour(const std::uint64_t n, const std::uint32_t k) const {
    const std::array<std::int64_t, 3> p{ xyz(n) };
    return cell(p[0]+dir_.fx_c(k, 0), p[1]+dir_.fx_c(k, 1), p[2]+dir_.fx_c(k, 2));
}

std::vector<std::uint64_t> NativeAddresses::own_set(const std::uint64_t n, const std::uint32_t parity) const {
    std::vector<std::uint64_t> out(dir_.q(), 0u);
    out[0] = address(n, 0u);
    for(std::uint32_t i=1u; i<dir_.q(); i+=2u) {
        out[i] = address(n, parity ? i : i+1u);
        out[i+1u] = address(neighbour(n, i), parity ? i+1u : i);
    }
    return out;
}

std::uint64_t NativeAddresses::next_state_address(const std::uint64_t n, const std::uint32_t k, const std::uint32_t parity) const {
    if(k==0u) {
        return address(n, 0u);
    }
    if(k%2u==1u) {
        return address(n, parity ? k+1u : k);
    }
    return address(neighbour(n, k-1u), parity ? k-1u : k);
}

std::uint64_t NativeAddresses::native_writer(const std::uint64_t addr, const std::uint32_t parity) const {
    const std::uint32_t slot{ static_cast<std::uint32_t>(addr/N_) };
    const std::uint64_t cell_index{ addr%N_ };
    if(slot==0u) {
        return cell_index;
    }
    if(slot%2u==1u) {
        return parity ? cell_index : neighbour(cell_index, slot+1u);        // -c_slot = c_{slot+1}
    }
    return parity ? neighbour(cell_index, slot) : cell_index;             // -c_{slot-1} = c_slot
}

// ---------------------------------------------------------------- NativeOwnershipReport

bool NativeOwnershipReport::pass() const {
    return conflict_active_ordinary_writer==0u && donor_outside_R==0u && native_row_writer_mismatch==0u
           && duplicate_r_address==0u && ordinary_non_pure_rows==0u && missing_direct_row==0u
           && algebra_writer_violations==0u && algebra_stream_violations==0u;
}

std::string NativeOwnershipReport::text() const {
    std::ostringstream s{};
    s << "rows_r_owned=" << rows_r_owned << " rows_native=" << rows_native << " rows_r_to_ordinary=" << rows_r_to_ordinary
      << " inactive_old_owner_permitted=" << inactive_old_owner_permitted
      << " conflict_active_ordinary_writer=" << conflict_active_ordinary_writer << " donor_outside_R=" << donor_outside_R
      << " native_row_writer_mismatch=" << native_row_writer_mismatch << " duplicate_r_address=" << duplicate_r_address
      << " ordinary_non_pure_rows=" << ordinary_non_pure_rows << " missing_direct_row=" << missing_direct_row
      << " near_unit_rows=" << near_unit_rows << " near_unit_row_max_deviation=" << near_unit_row_max_deviation
      << " near_unit_volumes=" << near_unit_volumes << " algebra_writer_violations=" << algebra_writer_violations
      << " algebra_stream_violations=" << algebra_stream_violations << " pass=" << (pass() ? "true" : "false");
    return s.str();
}

// ---------------------------------------------------------------- VolumetricWallEmbedding

VolumetricWallEmbedding::VolumetricWallEmbedding(const VolumetricWallNativeInput& input, const std::vector<std::uint8_t>& solid)
    : in_(input), solid_(solid), directions_(input.velocity_set), addresses_(input.box, directions_) {
    in_.validate();
    require(solid_.size()==addresses_.cells(), "solid mask size differs from the box");
    lattice_of_cell_.resize(in_.cells);
    for(std::uint32_t c=0u; c<in_.cells; c++) {
        const std::uint64_t n{ addresses_.cell(in_.cell_xyz[3u*c], in_.cell_xyz[3u*c+1u], in_.cell_xyz[3u*c+2u]) };
        require(!cell_of_lattice_.count(n), "two mesh cells share lattice cell "+std::to_string(n));
        require(solid_[n]==0u, "mesh cell "+std::to_string(c)+" is flagged solid");
        cell_of_lattice_[n] = c;
        lattice_of_cell_[c] = n;
    }
    verify_algebra();
    derive_regions();
    verify_ownership();
    require(report_.pass(), "ownership verification failed: "+report_.text());
}

VolumetricWallEmbedding::RowEntries VolumetricWallEmbedding::row_entries(const std::uint32_t cell, const std::uint32_t d) const {
    RowEntries out{};
    const std::uint32_t row{ cell*in_.velocity_set+d };
    for(std::uint32_t z=in_.D.offsets[row]; z<in_.D.offsets[row+1u]; z++) {
        out.donors.emplace_back(in_.D.indices[z]/in_.velocity_set, in_.D.indices[z]%in_.velocity_set);
        out.weights.push_back(in_.D.weights[z]);
    }
    out.scatter_entries = in_.R.offsets[row+1u]-in_.R.offsets[row];
    return out;
}

bool VolumetricWallEmbedding::pure(const std::uint32_t cell, const std::uint32_t d) {
    const std::uint64_t key{ static_cast<std::uint64_t>(cell)*in_.velocity_set+d };
    const auto cached{ pure_cache_.find(key) };
    if(cached!=pure_cache_.end()) {
        return cached->second;
    }
    const RowEntries e{ row_entries(cell, d) };
    bool ok{ false };
    if(e.donors.size()==1u && e.scatter_entries==0u) {
        const double w{ e.weights[0] };
        if(w!=1.0 && std::fabs(w-1.0)<=near_unit_tolerance) {
            report_.near_unit_rows++;
            report_.near_unit_row_max_deviation = std::max(report_.near_unit_row_max_deviation, std::fabs(w-1.0));
        }
        if(w==1.0) {   // EXACT unit weight
            const std::uint32_t k{ directions_.lex_to_fx(d) };
            const std::uint32_t back{ directions_.lex_to_fx(directions_.lex_opposite(d)) };
            const std::uint64_t expected{ addresses_.neighbour(lattice_of_cell_[cell], back) };   // receiver - c_d
            const auto donor_lattice{ cell_of_lattice_.find(expected) };
            ok = e.donors[0].second==d && donor_lattice!=cell_of_lattice_.end() && donor_lattice->second==e.donors[0].first;
            (void)k;
        }
    }
    pure_cache_[key] = ok;
    return ok;
}

void VolumetricWallEmbedding::derive_regions() {
    std::set<std::uint32_t> T{}, S{};
    for(std::uint32_t c=0u; c<in_.cells; c++) {
        if(in_.volume[c]!=1.0) {   // EXACT unit volume is ordinary
            T.insert(c);
            cut_cells_++;
            if(in_.volume[c]>=1.0-near_unit_tolerance) {
                report_.near_unit_volumes++;
            }
        }
        for(std::uint32_t d=0u; d<in_.velocity_set; d++) {
            if(!pure(c, d)) {
                T.insert(c);
                for(const auto& donor : row_entries(c, d).donors) {
                    T.insert(donor.first);
                }
            }
        }
    }
    for(const std::uint32_t state : in_.G.indices) {
        T.insert(state/in_.velocity_set);
    }
    for(std::uint32_t state=0u; state<in_.states(); state++) {
        if(in_.R.offsets[state+1u]>in_.R.offsets[state]) {
            T.insert(state/in_.velocity_set);
        }
    }
    for(const std::uint32_t c : in_.S.indices) {
        S.insert(c);
    }
    t_cells_ = T.size();
    s_cells_ = S.size();
    r_set_ = T;
    r_set_.insert(S.begin(), S.end());
    for(;;) {   // closure: donors of non-pure R rows, receivers of non-pure rows fed from R
        std::set<std::uint32_t> grow{};
        for(const std::uint32_t c : r_set_) {
            for(std::uint32_t d=0u; d<in_.velocity_set; d++) {
                if(!pure(c, d)) {
                    for(const auto& donor : row_entries(c, d).donors) {
                        if(!r_set_.count(donor.first)) {
                            grow.insert(donor.first);
                        }
                    }
                }
            }
        }
        for(std::uint32_t c=0u; c<in_.cells; c++) {
            if(r_set_.count(c)) {
                continue;
            }
            for(std::uint32_t d=0u; d<in_.velocity_set; d++) {
                const RowEntries e{ row_entries(c, d) };
                bool donor_in_r{ false };
                for(const auto& donor : e.donors) {
                    donor_in_r = donor_in_r || r_set_.count(donor.first)>0u;
                }
                if(donor_in_r && !pure(c, d)) {
                    grow.insert(c);
                }
            }
        }
        if(grow.empty()) {
            break;
        }
        r_set_.insert(grow.begin(), grow.end());
        closure_added_ += grow.size();
    }
    r_cells_.assign(r_set_.begin(), r_set_.end());
    r_index_of_cell_.assign(in_.cells, -1);
    for(size_t i=0u; i<r_cells_.size(); i++) {
        r_index_of_cell_[static_cast<size_t>(r_cells_[i])] = static_cast<std::int64_t>(i);
    }
}

void VolumetricWallEmbedding::verify_algebra() {
    for(std::uint64_t n=0u; n<addresses_.cells(); n++) {
        for(std::uint32_t parity=0u; parity<2u; parity++) {
            const std::vector<std::uint64_t> own{ addresses_.own_set(n, parity) };
            for(const std::uint64_t a : own) {
                if(addresses_.native_writer(a, parity)!=n) {
                    report_.algebra_writer_violations++;
                }
            }
            for(std::uint32_t k=1u; k<in_.velocity_set; k++) {
                const std::uint32_t back{ k%2u==1u ? k+1u : k-1u };
                const std::uint64_t donor{ addresses_.neighbour(n, back) };
                const std::vector<std::uint64_t> donor_set{ addresses_.own_set(donor, parity) };
                const std::uint64_t a{ addresses_.next_state_address(n, k, parity) };
                if(std::find(donor_set.begin(), donor_set.end(), a)==donor_set.end()) {
                    report_.algebra_stream_violations++;
                }
            }
        }
    }
}

void VolumetricWallEmbedding::verify_ownership() {
    std::set<std::uint64_t> written0{}, written1{};
    const auto classify_writer = [&](const std::uint64_t a, const std::uint32_t parity) {
        const std::uint64_t writer{ addresses_.native_writer(a, parity) };
        if(solid_[writer]!=0u) {
            report_.inactive_old_owner_permitted++;
            return;
        }
        const auto mesh{ cell_of_lattice_.find(writer) };
        const bool in_r{ mesh!=cell_of_lattice_.end() && r_set_.count(mesh->second)>0u };
        if(!in_r) {
            report_.conflict_active_ordinary_writer++;
        }
    };
    const auto r_write = [&](const std::uint32_t cell, const std::uint32_t d) {
        const RowEntries e{ row_entries(cell, d) };
        if(e.donors.empty() && e.scatter_entries==0u) {
            report_.missing_direct_row++;
        }
        for(const auto& donor : e.donors) {
            if(!r_set_.count(donor.first)) {
                report_.donor_outside_R++;
            }
        }
        const std::uint32_t k{ directions_.lex_to_fx(d) };
        const std::uint64_t n{ lattice_of_cell_[cell] };
        NativeOwnedRow row{};
        row.receiver_cell = cell;
        row.direction = d;
        row.address_parity0 = addresses_.next_state_address(n, k, 0u);
        row.address_parity1 = addresses_.next_state_address(n, k, 1u);
        if(!written0.insert(row.address_parity0).second || !written1.insert(row.address_parity1).second) {
            report_.duplicate_r_address++;
        }
        classify_writer(row.address_parity0, 0u);
        classify_writer(row.address_parity1, 1u);
        owned_.push_back(row);
        r_rows_.push_back(static_cast<std::int64_t>(cell)*in_.velocity_set+d);
        report_.rows_r_owned++;
    };
    for(std::uint32_t cell=0u; cell<in_.cells; cell++) {
        const bool cell_in_r{ r_set_.count(cell)>0u };
        for(std::uint32_t d=0u; d<in_.velocity_set; d++) {
            const RowEntries e{ row_entries(cell, d) };
            const bool is_pure{ pure(cell, d) };
            bool donor_in_r{ false };
            for(const auto& donor : e.donors) {
                donor_in_r = donor_in_r || r_set_.count(donor.first)>0u;
            }
            if(cell_in_r) {
                if(is_pure && !donor_in_r) {
                    report_.rows_native++;
                    native_rows_.push_back(static_cast<std::int64_t>(cell)*in_.velocity_set+d);
                    const std::uint32_t k{ directions_.lex_to_fx(d) };
                    const std::uint64_t n{ lattice_of_cell_[cell] };
                    for(std::uint32_t parity=0u; parity<2u; parity++) {
                        const std::uint64_t writer{ addresses_.native_writer(addresses_.next_state_address(n, k, parity), parity) };
                        if(writer!=lattice_of_cell_[e.donors[0].first]) {
                            report_.native_row_writer_mismatch++;
                        }
                    }
                } else {
                    r_write(cell, d);
                }
            } else {
                if(!is_pure) {
                    report_.ordinary_non_pure_rows++;
                }
                if(donor_in_r) {
                    report_.rows_r_to_ordinary++;
                    r_write(cell, d);
                } else {
                    report_.rows_native++;
                    native_rows_.push_back(static_cast<std::int64_t>(cell)*in_.velocity_set+d);
                }
            }
        }
    }
    std::sort(native_rows_.begin(), native_rows_.end());
    std::sort(r_rows_.begin(), r_rows_.end());   // owned_ keeps construction order (cell-major, direction-minor) = sorted too
}

void VolumetricWallEmbedding::write_lists(const std::string& prefix) const {
    const auto dump = [&](const std::string& name, const std::vector<std::int64_t>& values) {
        std::ofstream out(prefix+"_"+name+".i64", std::ios::binary);
        if(!out) {
            throw std::runtime_error("VolumetricWallNative: cannot write "+prefix+"_"+name+".i64");
        }
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()*sizeof(std::int64_t)));
    };
    dump("R_cells", r_cells_);
    dump("R_rows", r_rows_);
    dump("native_rows", native_rows_);
}

}  // namespace fluidx3d::experimental
