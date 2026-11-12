// Task 19: the native volumetric wall fixture and instrument driver (VOLUMETRIC_WALL build). It is the usable example
// of the API (flags -> set_volumetric_wall -> run) and the registered engineering instruments of Stage C. It is NOT a
// physical acceptance test: physical runs need the registered integration protocol and the CF13 acceptance.
//
//   VW_MODE=attach   VW_INPUT=INPUT.vwnat VW_OUT=DIR              attach, run(0): the stored initial R state vs fl32(f0 - w), raw fi dump
//   VW_MODE=sentinel VW_INPUT=INPUT.vwnat VW_OUT=DIR              A3: fi filled with a sentinel, R cells flagged, ONE native step at even and
//                                                                   at odd t; every R-owned address must keep the sentinel (write-count 0 by
//                                                                   native), every active ordinary own-set address must change
//   VW_MODE=step     ... VW_N=N                                 N registered steps with the wall; after every step: raw fi, t, the capture
//                                                                   (gpre, f, post, fields, incoming, Q, impulse, pendings, prestore, clamp)
//   VW_MODE=poison   ... (RR27)                              #2034: a synthetic negative-density sampler state; run(1) must stop with
//                                                            the engine error "invalid wall states" (exit 1); returning is a FAILURE
//   VW_MODE=resume   ... VW_M=M VW_RESUME_FI=F.f32 VW_RESUME_PENDING=P.f64 VW_RESUME_T=t VW_RESUME_CONTAINER_SHA256=hex
//                                                            task 98: a FRESH process loads the saved raw state of level t (container
//                                                            identity pinned), runs M steps, writes resume_end_fi/pending + summary
//   VW_MODE=restart  ... VW_N=N VW_M=M
//   VW_MODE=protocol ... VW_N=steps VW_CHECKPOINTS=0,1,2,3,...  per-step host series (series.f64 + header), rolling SHA-256 of fi, raw checkpoints
//                                                            (fi, pending, t) at the listed steps and at the first failure (non-finite, negative population, clamp)  run N steps, snapshot (fi, t, pending), run M more; restore and run M; the
//                                                                   two end states must be bitwise equal (A6, both parities when N is odd/even)
// Uniform force (task 43), builds with VOLUME_FORCE only, where it is REQUIRED:
//   VW_FORCE=fx,fy,fz                 the force density of the LBM constructor, three float32 values as C hexadecimal
//                                     floating-point text (for example 0x1.2214eap-24), so the registered value is exact;
//                                     0x0p+0,0x0p+0,0x0p+0 is the force-zero regression. Refused in a build without VOLUME_FORCE.
// Warm-up (task 45), protocol mode only, optional:
//   VW_WARMUP=N                       N steps (even, > 0) at engine speed with NO readback, ledger or hash, after the initial state is
//                                     saved (warmup_initial_fi.f32, warmup_initial_pending.f64); the instrumented protocol then starts
//                                     from level N: row 0 and checkpoint 0 are level N, t_after is absolute. The clamp counter is
//                                     cumulative over the warm-up. Nothing else is observed during the warm-up.
//   VW_WARMUP_CHECKPOINT=M            with VW_WARMUP: M even, 0 < M < N; the raw state of level M is saved once
//                                     (warmup_checkpoint_fi.f32, warmup_checkpoint_pending.f64, warmup_checkpoint_meta.txt)
// Friction rows (task 43), REQUIRED in every mode that attaches the wall:
//   VW_ROWS=container                 the supported path LBM::set_volumetric_wall: the container's friction_rows section, coefficient identities checked
//   VW_ROWS=legacy [VW_PHI=FILE]      the research path LBM::set_volumetric_wall_research: the original kernel expression, or the rows
//                                     file of the archived runs (tasks 39, 41, 42); reproduces every archived fixture
// SGS closure (V2), builds with VOLUMETRIC_WALL_SGS only, where it is REQUIRED; refused in other builds:
//   VW_SGS_K=k                        the constant k = 18 sqrt(2) (C Delta)^2 given to LBM::set_volumetric_wall_sgs before the
//                                     attachment (hexadecimal text recommended; the qualified value is 0x1.8746d2p-1; 0 = the
//                                     V1 rate). After the attachment the stored k (getter) is printed and written to VW_OUT/sgs_k.txt.
//   VW_MODE=sgs_refusal VW_SGS_REFUSAL=value|after_attach|after_run   one isolated API refusal; the engine must stop (exit 1)
// Open x (task 105), builds with VOLUMETRIC_WALL_OPEN_X only, where it is REQUIRED (no defaults); refused in other builds:
//   VW_OPEN=x VW_OPEN_INLET_U=ux,uy,uz VW_OPEN_INLET_MODE=hard|relaxed,K_u VW_OPEN_SIGMA VW_OPEN_MACH VW_OPEN_LENGTH
//   VW_OPEN_RHO_INF VW_OPEN_E_INITIAL=rho,ux,uy,uz   given to LBM::set_volumetric_wall_open_x before the attachment. Records:
//   VW_OUT/open_declaration.txt (stored values and K_p, hex), VW_OUT/open_layout.txt (E cells, exterior facets and rows),
//   and next to every step dump and checkpoint <prefix>_open_state.f64 (the E state of that level, 4 doubles per slot
//   side*(Ny*Nz) + y + Ny*z) and <prefix>_open_meta.txt (t, open_failures). Modes: attach, step, protocol (the protocol
//   series columns are the closed-box quantities; the open budgets are formed offline from the checkpoints).
// Every output is raw (non-finite values kept); analysis is the research tree's job.
#include "lbm.hpp"
#include "physics_check.hpp"
#include "volumetric_wall_native_protocol.hpp"
#include "provenance_axis.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

std::string env(const char* name);   // global; defined with main_setup below

namespace {

/// @brief Task 98: the velocity set of this build (the container must have the same q; checked in Fixture).
#ifdef D3Q27
constexpr std::uint32_t kQ{ 27u };
#else
constexpr std::uint32_t kQ{ 19u };
#endif

using fluidx3d::experimental::NativeAddresses;
using fluidx3d::experimental::NativeDirections;
using fluidx3d::experimental::VolumetricWallEmbedding;
using fluidx3d::experimental::VolumetricWallNativeCapture;
using fluidx3d::experimental::VolumetricWallNativeInput;

/// @brief Host-side finiteness by bit pattern (the device program is built with -cl-finite-math-only, so no device
/// isfinite is trusted; the host is compiled with precise float, but the bit test does not depend on it).
class BitPattern {
public:
    static std::uint64_t non_finite(const std::vector<float>& values) {
        std::uint64_t count{ 0u };
        for(const float v : values) { std::uint32_t bits{}; std::memcpy(&bits, &v, 4u); if(((bits>>23u)&0xFFu)==0xFFu) count++; }
        return count;
    }
    static std::uint64_t non_finite(const std::vector<double>& values) {
        std::uint64_t count{ 0u };
        for(const double v : values) { std::uint64_t bits{}; std::memcpy(&bits, &v, 8u); if(((bits>>52u)&0x7FFull)==0x7FFull) count++; }
        return count;
    }
};

/// @brief Binary dump helpers (little-endian, raw); every dump reports its non-finite count by bit pattern.
class Dump {
public:
    static std::uint64_t non_finite_total;
    template<class T> static void vector(const std::string& path, const std::vector<T>& values) {
        std::ofstream out(path, std::ios::binary);
        if(!out) throw std::runtime_error("cannot write "+path);
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()*sizeof(T)));
        const std::uint64_t bad{ BitPattern::non_finite(values) };
        non_finite_total += bad;
        if(bad) std::printf("NONFINITE %llu values in %s\n", static_cast<unsigned long long>(bad), path.c_str());
    }
    static void capture(const std::string& prefix, const VolumetricWallNativeCapture& c) {
        if(!c.available) { std::ofstream meta(prefix+"_meta.txt"); meta << "t " << c.t << "\ncapture UNAVAILABLE (no wall step yet)\nclamp_count " << c.clamp_count << "\n"; return; }
        vector(prefix+"_gpre.f32", c.gpre); vector(prefix+"_f.f64", c.f); vector(prefix+"_post.f64", c.post); vector(prefix+"_fields.f64", c.fields);
        vector(prefix+"_incoming.f64", c.incoming); vector(prefix+"_Q.f64", c.Q); vector(prefix+"_impulse.f64", c.impulse);
        vector(prefix+"_pending_old.f64", c.pending_old); vector(prefix+"_pending_new.f64", c.pending_new); vector(prefix+"_prestore.f64", c.prestore);
        if(!c.returned.empty()) { // task 55, affine candidate only: the accepted path writes no new file
            vector(prefix+"_support_post.f64", c.support_post); vector(prefix+"_returned.f64", c.returned); vector(prefix+"_direct_increment.f64", c.direct_increment);
        }
        if(!c.wall_state.empty()) { // task 98, RR27 only: rho_w, u_w (4 per facet), the one-per-facet delta, the rho_w failure count
            vector(prefix+"_wall_state.f64", c.wall_state); vector(prefix+"_delta.f64", c.delta);
            std::ofstream failures(prefix+"_wall_state_failures.txt"); failures << c.wall_state_failures << "\n";
            if(!c.facet_law.empty()) vector(prefix+"_facet_law.f64", c.facet_law); // V2 ROUGH: (K0_a, C_a) per facet as uploaded
        }
        if(!c.t119_out.empty()) { // task 119, T119 builds only: B0, a (3), L/S per row, and the PR2-trigger count
            vector(prefix+"_t119_out.f64", c.t119_out);
            std::ofstream negative(prefix+"_t119_negative_q.txt"); negative << c.t119_negative_q << "\n";
            std::ofstream p1(prefix+"_t119_p1.txt"); p1 << c.t119_p1 << "\n";
        }
        if(c.srd_stats.size()==6u) { // task 121, SRD builds only: the six cumulative counters of the redistribution
            std::ofstream srd(prefix+"_srd_stats.txt");
            srd << "cumulative_since_attachment 1\nnonpositive_mean " << c.srd_stats[0] << "\nsingular_B " << c.srd_stats[1] << "\nnonfinite_recon " << c.srd_stats[2] << "\nlimiter_entries " << c.srd_stats[3] << "\nnegative_fprime " << c.srd_stats[4] << "\nnonfinite_fprime " << c.srd_stats[5] << "\n";
        }
        std::ofstream meta(prefix+"_meta.txt");
        meta << "t " << c.t << "\nclamp_count " << c.clamp_count << "\n";
    }
};
std::uint64_t Dump::non_finite_total{ 0u };

/// @brief The uniform force density of a run (task 43), from VW_FORCE.
struct ForceInput {
    std::array<float, 3> value{};      ///< fx, fy, fz, lattice units (momentum per volume per step).
    std::array<std::string, 3> text{}; ///< The three hexadecimal texts as given (recorded in the summary).

    /// @brief Reads VW_FORCE. A build with VOLUME_FORCE requires it; a build without refuses it.
    /// @return the force (zero and empty texts in a build without VOLUME_FORCE)
    /// @throws std::invalid_argument on a missing, extra, malformed, non-finite or not exactly representable value
    static ForceInput from_environment() {
        ForceInput force{};
        const std::string list{ env("VW_FORCE") };
#ifndef VOLUME_FORCE
        if(!list.empty()) {
            throw std::invalid_argument("VW_FORCE is set, but this executable is built without VOLUME_FORCE");
        }
#else
        if(list.empty()) {
            throw std::invalid_argument("VW_FORCE=fx,fy,fz (three hexadecimal float32 values) is required in a VOLUME_FORCE build");
        }
        std::stringstream stream(list);
        std::string item{};
        std::uint32_t count{ 0u };
        while(std::getline(stream, item, ',')) {
            if(count>=3u) {
                throw std::invalid_argument("VW_FORCE has more than three values");
            }
            const bool hexadecimal{ item.rfind("0x", 0u)==0u || item.rfind("-0x", 0u)==0u };
            if(!hexadecimal) {
                throw std::invalid_argument("VW_FORCE value '"+item+"' is not hexadecimal floating-point text");
            }
            char* end{ nullptr };
            const double parsed{ std::strtod(item.c_str(), &end) };
            const float narrowed{ static_cast<float>(parsed) };
            if(end!=item.c_str()+item.size() || !std::isfinite(parsed) || static_cast<double>(narrowed)!=parsed) {
                throw std::invalid_argument("VW_FORCE value '"+item+"' is malformed, not finite or not exactly a float32");
            }
            force.value[count] = narrowed;
            force.text[count] = item;
            count++;
        }
        if(count!=3u) {
            throw std::invalid_argument("VW_FORCE needs exactly three values");
        }
        std::printf("preflight: force %s %s %s (%.9g %.9g %.9g) bound to VW_FORCE\n", force.text[0].c_str(), force.text[1].c_str(), force.text[2].c_str(),
                    static_cast<double>(force.value[0]), static_cast<double>(force.value[1]), static_cast<double>(force.value[2]));
#endif
        return force;
    }
};

/// @brief V2: the SGS closure constant k of a run, from VW_SGS_K. A VOLUMETRIC_WALL_SGS build requires it; another build
/// refuses it. The text is parsed as one number (hexadecimal floating-point text recommended, so the registered value is
/// exact) and given to LBM::set_volumetric_wall_sgs unchanged: the engine, not the fixture, refuses a negative or
/// non-finite k (the refusal tests of VW_MODE=sgs_refusal rely on this).
struct SgsInput {
    double value{ 0.0 };   ///< the parsed number (0 in a build without VOLUMETRIC_WALL_SGS)
    std::string text{};    ///< the VW_SGS_K text
    /// @brief Reads VW_SGS_K.
    /// @throws std::invalid_argument when it is missing in an SGS build, set in another build, or not exactly one number
    static SgsInput from_environment() {
        SgsInput sgs{};
        sgs.text = env("VW_SGS_K");
#ifndef VOLUMETRIC_WALL_SGS
        if(!sgs.text.empty()) {
            throw std::invalid_argument("VW_SGS_K is set, but this executable is built without VOLUMETRIC_WALL_SGS");
        }
#else
        if(sgs.text.empty()) {
            throw std::invalid_argument("VW_SGS_K (one number, for example 0x1.8746d2p-1) is required in a VOLUMETRIC_WALL_SGS build");
        }
        char* end{ nullptr };
        sgs.value = std::strtod(sgs.text.c_str(), &end);
        if(end==sgs.text.c_str() || *end!='\0') {
            throw std::invalid_argument("VW_SGS_K '"+sgs.text+"' is not exactly one number");
        }
#endif // VOLUMETRIC_WALL_SGS
        return sgs;
    }
};

/// @brief Task 105: the open-x declaration of a run (VOLUMETRIC_WALL_OPEN_X builds), from the environment. Every value is
/// REQUIRED (no default: the registration declares each one; hexadecimal floating-point text recommended):
///   VW_OPEN=x, VW_OPEN_INLET_U=ux,uy,uz, VW_OPEN_INLET_MODE=hard | relaxed,K_u, VW_OPEN_SIGMA, VW_OPEN_MACH, VW_OPEN_LENGTH (L_x),
///   VW_OPEN_RHO_INF, VW_OPEN_E_INITIAL=rho,ux,uy,uz.
/// A build without the macro refuses VW_OPEN. The engine (VolumetricWallOpenX::validate) refuses invalid values.
struct OpenInput {
    /// @brief One number of a list item.
    /// @throws std::invalid_argument when the text is not exactly one number
    static double number(const std::string& text, const std::string& name) {
        char* end{ nullptr };
        const double value{ std::strtod(text.c_str(), &end) };
        if(text.empty() || end==text.c_str() || *end!='\0') throw std::invalid_argument(name+" '"+text+"' is not exactly one number");
        return value;
    }
    /// @brief A comma list of exactly `count` numbers.
    static std::vector<double> list(const std::string& text, const std::size_t count, const std::string& name) {
        std::vector<double> values{};
        std::stringstream ss(text);
        std::string item{};
        while(std::getline(ss, item, ',')) values.push_back(number(item, name));
        if(values.size()!=count) throw std::invalid_argument(name+" needs exactly "+std::to_string(count)+" comma-separated numbers");
        return values;
    }
    /// @brief Reads the declaration and gives it to the engine before the attachment (open builds), or refuses VW_OPEN.
    /// @throws std::invalid_argument when a value is missing or malformed, or VW_OPEN is set in a closed build
    static void apply(LBM& lbm) {
        const std::string open{ env("VW_OPEN") };
#ifndef VOLUMETRIC_WALL_OPEN_X
        (void)lbm;
        if(!open.empty()) throw std::invalid_argument("VW_OPEN is set, but this executable is built without VOLUMETRIC_WALL_OPEN_X (a closed build)");
#else
        if(open!="x") throw std::invalid_argument("VW_OPEN=x and the full open declaration are required in a VOLUMETRIC_WALL_OPEN_X build");
        fluidx3d::experimental::VolumetricWallOpenX d{};
        const std::vector<double> u{ list(env("VW_OPEN_INLET_U"), 3u, "VW_OPEN_INLET_U") };
        d.inlet_velocity = { u[0], u[1], u[2] };
        const std::string mode{ env("VW_OPEN_INLET_MODE") };
        if(mode=="hard") {
            d.hard_inlet = true;
        } else if(mode.rfind("relaxed,", 0u)==0u) {
            d.hard_inlet = false;
            d.inlet_relaxation = number(mode.substr(8u), "VW_OPEN_INLET_MODE K_u");
        } else {
            throw std::invalid_argument("VW_OPEN_INLET_MODE must be hard or relaxed,K_u");
        }
        d.outlet_sigma = number(env("VW_OPEN_SIGMA"), "VW_OPEN_SIGMA");
        d.outlet_mach = number(env("VW_OPEN_MACH"), "VW_OPEN_MACH");
        d.outlet_length = number(env("VW_OPEN_LENGTH"), "VW_OPEN_LENGTH");
        d.outlet_rho_inf = number(env("VW_OPEN_RHO_INF"), "VW_OPEN_RHO_INF");
        const std::vector<double> e{ list(env("VW_OPEN_E_INITIAL"), 4u, "VW_OPEN_E_INITIAL") };
        d.initial_rho = e[0];
        d.initial_velocity = { e[1], e[2], e[3] };
        lbm.set_volumetric_wall_open_x(d);
#endif // VOLUMETRIC_WALL_OPEN_X
    }
    /// @brief After the attachment: VW_OUT/open_declaration.txt (the stored declaration and K_p as hexadecimal text) and
    /// VW_OUT/open_layout.txt (E cells: mesh index, slot, outward sign, b-1, b-2; exterior facets with their E cell; exterior rows).
    static void record(LBM& lbm, const std::string& out) {
#ifdef VOLUMETRIC_WALL_OPEN_X
        const fluidx3d::experimental::VolumetricWallOpenX d{ lbm.volumetric_wall_open_declaration() };
        char text[768]{};
        std::snprintf(text, sizeof(text), "inlet_velocity %a %a %a\nhard_inlet %d\ninlet_relaxation %a\nsigma %a\nmach %a\nlength %a\nrho_inf %a\ninitial %a %a %a %a\nK_p %a\n",
            d.inlet_velocity[0], d.inlet_velocity[1], d.inlet_velocity[2], d.hard_inlet ? 1 : 0, d.inlet_relaxation, d.outlet_sigma, d.outlet_mach,
            d.outlet_length, d.outlet_rho_inf, d.initial_rho, d.initial_velocity[0], d.initial_velocity[1], d.initial_velocity[2], d.outlet_relaxation());
        std::ofstream declaration(out+"/open_declaration.txt");
        declaration << text;
        const std::array<uint, 3> paths{ lbm.volumetric_wall_open_e_paths() };
        declaration << "e_cells_R " << paths[0] << "\ne_cells_support " << paths[1] << "\ne_cells_ordinary " << paths[2] << "\n";
        const fluidx3d::experimental::VolumetricWallOpenLayout& layout{ lbm.volumetric_wall_open_layout() };
        std::ofstream masks(out+"/open_layout.txt");
        masks << "# " << layout.text() << "\n# e mesh slot sign b1 b2 | facet a e_cell | row r\n";
        for(size_t k=0u; k<layout.e_cells().size(); k++) masks << "e " << layout.e_cells()[k] << ' ' << layout.e_slot()[k] << ' ' << layout.e_side()[k] << ' ' << layout.e_inward()[2u*k] << ' ' << layout.e_inward()[2u*k+1u] << '\n';
        for(size_t a=0u; a<layout.exterior_facet().size(); a++) if(layout.exterior_facet()[a]!=0u) masks << "facet " << a << ' ' << layout.e_cell_of_facet()[a] << '\n';
        for(size_t r=0u; r<layout.exterior_row().size(); r++) if(layout.exterior_row()[r]!=0u) masks << "row " << r << '\n';
        if(!declaration || !masks) throw std::runtime_error("cannot write the open records in "+out);
#else
        (void)lbm; (void)out;
#endif // VOLUMETRIC_WALL_OPEN_X
    }
    /// @brief The current E state and the refusal count next to a dump: <prefix>_open_state.f64 (4 doubles per slot
    /// side*(Ny*Nz) + y + Ny*z) and <prefix>_open_meta.txt (t, open_failures). Nothing in a closed build.
    static void dump(LBM& lbm, const std::string& prefix) {
#ifdef VOLUMETRIC_WALL_OPEN_X
        std::vector<double> state{};
        lbm.volumetric_wall_read_open_state(state);
        std::ofstream bin(prefix+"_open_state.f64", std::ios::binary);
        bin.write(reinterpret_cast<const char*>(state.data()), static_cast<std::streamsize>(state.size()*sizeof(double)));
        std::ofstream meta(prefix+"_open_meta.txt");
        meta << "t " << lbm.get_t() << "\nopen_failures " << lbm.volumetric_wall_open_failures() << "\n";
        if(!bin || !meta) throw std::runtime_error("cannot write "+prefix+"_open_state.f64");
#else
        (void)lbm; (void)prefix;
#endif // VOLUMETRIC_WALL_OPEN_X
    }
};

/// @brief The lattice of a container: the box, the solid flags (every non-mesh cell solid), the cell moments of f0.
/// @brief Task 98: the direction tables of this build, made once.
const fluidx3d::experimental::NativeDirections& directions_of_build() {
    static const fluidx3d::experimental::NativeDirections tables(kQ);
    return tables;
}

class Fixture {
public:
    explicit Fixture(const std::string& path) : input(VolumetricWallNativeInput::read(path)), addresses(input.box, NativeDirections(kQ)) {
        input.validate();
        if(input.velocity_set!=kQ) {
            throw std::invalid_argument("the container is D3Q"+std::to_string(input.velocity_set)+", this fixture build is D3Q"+std::to_string(kQ));
        }
        solid.assign(static_cast<size_t>(addresses.cells()), 1u);
        for(std::uint32_t c=0u; c<input.cells; c++) {
            solid[static_cast<size_t>(lattice(c))] = 0u;
        }
    }
    std::uint64_t lattice(const std::uint32_t c) const { return addresses.cell(input.cell_xyz[3u*c], input.cell_xyz[3u*c+1u], input.cell_xyz[3u*c+2u]); }
    /// @brief The LBM with the flags and the f0 moments as the ordinary initial fields (rho, u); nu from the container.
    LBM* make_lbm() const {
        const std::string nu_text{ env("VW_NU") };   // task 45 (#1015), input-validation test only: an LBM viscosity that differs from the container's must be refused at the attachment
        const float lbm_nu{ nu_text.empty() ? static_cast<float>(input.viscosity) : std::stof(nu_text) };
#ifdef VOLUME_FORCE
        LBM* lbm = new LBM(static_cast<uint>(input.box[0]), static_cast<uint>(input.box[1]), static_cast<uint>(input.box[2]), lbm_nu, force.value[0], force.value[1], force.value[2]);   // task 43: the uniform force density
#else
        LBM* lbm = new LBM(static_cast<uint>(input.box[0]), static_cast<uint>(input.box[1]), static_cast<uint>(input.box[2]), lbm_nu);
#endif
        const NativeDirections dir(kQ);
        for(ulong n=0ull; n<lbm->get_N(); n++) {
            lbm->flags[n] = solid[static_cast<size_t>(n)] ? TYPE_S : 0u;
        }
        for(std::uint32_t c=0u; c<input.cells; c++) {
            double rho=0.0, jx=0.0, jy=0.0, jz=0.0;
            for(std::uint32_t d=0u; d<kQ; d++) {
                const double v{ input.f0[c*kQ+d] };
                rho += v; jx += v*dir.lex_c(d, 0); jy += v*dir.lex_c(d, 1); jz += v*dir.lex_c(d, 2);
            }
            const ulong n{ static_cast<ulong>(lattice(c)) };
            lbm->rho[n] = static_cast<float>(rho);
            lbm->u.x[n] = static_cast<float>(jx/rho); lbm->u.y[n] = static_cast<float>(jy/rho); lbm->u.z[n] = static_cast<float>(jz/rho);
        }
        return lbm;
    }
    VolumetricWallNativeInput input;
    NativeAddresses addresses;
    std::vector<std::uint8_t> solid{};
    ForceInput force{ ForceInput::from_environment() };   ///< task 43: zero in a build without VOLUME_FORCE
};

/// @brief Task 105: the R cells by declaration of the attachment, so the fixture's embedding (the R order of every capture
/// array, embedding_*.i64) equals the engine's: the E cells of the open-x layout in an open build, none otherwise.
/// @param fx the fixture (its container)
/// @return the mesh cells the engine forces into R
std::vector<std::uint32_t> forced_r_cells(const Fixture& fx) {
#ifdef VOLUMETRIC_WALL_OPEN_X
    return fluidx3d::experimental::VolumetricWallOpenLayout(fx.input).e_cells();
#else
    (void)fx;
    return {};
#endif // VOLUMETRIC_WALL_OPEN_X
}

/// @brief Attach the wall as VW_ROWS says (task 43). The fixture reads the environment; the engine does not.
/// @param lbm the lattice with its flags set
/// @param path the container
/// @throws std::invalid_argument when VW_ROWS is not "container" or "legacy", or VW_PHI is set with VW_ROWS=container
void attach_wall(LBM& lbm, const std::string& path) {
    const std::string rows_mode{ env("VW_ROWS") };
    const std::string rows_file{ env("VW_PHI") };
    const SgsInput sgs{ SgsInput::from_environment() };   // V2: checked in every build (refused where the macro is off)
#ifdef VOLUMETRIC_WALL_SGS
    lbm.set_volumetric_wall_sgs(sgs.value);                // before the attachment, which gives k to the kernels
#else
    (void)sgs;
#endif // VOLUMETRIC_WALL_SGS
#ifdef VOLUMETRIC_WALL_SRD
    if(env("VW_SRD_CAPTURE")=="1") lbm.set_volumetric_wall_srd_capture(true);   // task 124: diagnostic capture of the pre-SRD stored state
#endif // VOLUMETRIC_WALL_SRD
    OpenInput::apply(lbm);                                 // task 105: the open declaration before the attachment (open builds), or the refusal of VW_OPEN
    if(rows_mode=="container") {
        if(!rows_file.empty()) {
            throw std::invalid_argument("VW_PHI is set together with VW_ROWS=container (the supported path takes its rows from the container only)");
        }
        lbm.set_volumetric_wall(path);
    } else if(rows_mode=="legacy") {
        lbm.set_volumetric_wall_research(path, rows_file);
    } else {
        throw std::invalid_argument("VW_ROWS must be container (supported path) or legacy (research path); it is '"+rows_mode+"'");
    }
    OpenInput::record(lbm, env("VW_OUT"));                 // task 105: the stored declaration and the classification (open builds)
#ifdef VOLUMETRIC_WALL_SGS
    { // V2: the record of the k the engine stored (read back with the getter, not the input text)
        char line[160]{};
        std::snprintf(line, sizeof(line), "%a", lbm.get_volumetric_wall_sgs());
        std::printf("preflight: sgs_k %s stored (VW_SGS_K '%s')\n", line, sgs.text.c_str());
        const std::string out{ env("VW_OUT") };
        std::ofstream record(out+"/sgs_k.txt");
        record << "sgs_k_input_text " << sgs.text << "\nsgs_k_stored_hex " << line << "\n";
        if(!record) throw std::runtime_error("cannot write "+out+"/sgs_k.txt");
    }
#endif // VOLUMETRIC_WALL_SGS
}

/// @brief V2 (VOLUMETRIC_WALL_SGS builds): one isolated refusal of LBM::set_volumetric_wall_sgs, chosen by VW_SGS_REFUSAL.
/// The engine must stop the program (print_error, exit 1); a return is a FAILURE.
///   value         the call with the VW_SGS_K value before any attachment (for a negative or non-finite VW_SGS_K)
///   after_attach  the attachment with VW_SGS_K, then a second call (refused: the kernels already have k)
///   after_run     the attachment and run(0), then a second call (refused: the simulation is initialized)
/// @param fx the fixture
/// @param path the container
/// @return 1 (reached only when the engine did not refuse)
int run_sgs_refusal(const Fixture& fx, const std::string& path) {
#ifndef VOLUMETRIC_WALL_SGS
    (void)fx; (void)path;
    throw std::invalid_argument("VW_MODE=sgs_refusal needs a VOLUMETRIC_WALL_SGS build");
#else
    const std::string which{ env("VW_SGS_REFUSAL") };
    LBM* lbm = fx.make_lbm();
    if(which=="value") {
        const SgsInput sgs{ SgsInput::from_environment() };
        std::printf("sgs_refusal value: set_volumetric_wall_sgs(%a) must stop\n", sgs.value);
        std::fflush(stdout);
        lbm->set_volumetric_wall_sgs(sgs.value);
    } else if(which=="after_attach") {
        attach_wall(*lbm, path);
        std::printf("sgs_refusal after_attach: a second set_volumetric_wall_sgs must stop\n");
        std::fflush(stdout);
        lbm->set_volumetric_wall_sgs(lbm->get_volumetric_wall_sgs());
    } else if(which=="after_run") {
        attach_wall(*lbm, path);
        lbm->run(0u);
        std::printf("sgs_refusal after_run: set_volumetric_wall_sgs after run(0) must stop\n");
        std::fflush(stdout);
        lbm->set_volumetric_wall_sgs(lbm->get_volumetric_wall_sgs());
    } else {
        delete lbm;
        throw std::invalid_argument("VW_SGS_REFUSAL must be value, after_attach or after_run; it is '"+which+"'");
    }
    std::printf("PHYSICS volumetric_wall_native: set_volumetric_wall_sgs (%s) was NOT refused FAILED\n", which.c_str());
    delete lbm;
    return 1;
#endif // VOLUMETRIC_WALL_SGS
}

int run_attach(const Fixture& fx, const std::string& path, const std::string& out) {
    LBM* lbm = fx.make_lbm();
    attach_wall(*lbm, path);
    lbm->run(0u);
    std::vector<fpxx> fi{};
    lbm->volumetric_wall_read_fi(fi);
    Dump::vector(out+"/attach_fi.f32", fi);
    // A1 (lossless part), whole domain (#273): every mesh cell's stored g equals fl32(f0 - w) exactly, read back through the
    // parity-0 load sets; the decoded level 0 fl64(w + g) differs from f0 by the rounding point R0 only (reported)
    const VolumetricWallEmbedding embedding(fx.input, fx.solid, forced_r_cells(fx));
    const NativeDirections dir(kQ);
    std::uint64_t mismatches{ 0u };
    double max_abs{ 0.0 }, max_r0{ 0.0 };
    for(std::uint32_t cell=0u; cell<fx.input.cells; cell++) {
        const std::uint64_t n{ embedding.lattice_of_cell()[cell] };
        const auto own{ fx.addresses.own_set(n, 0u) };    // level 0 states of n: load_f(n, t=0)
        for(std::uint32_t k=0u; k<kQ; k++) {
            const std::uint32_t d{ dir.fx_to_lex(k) };
            const double f0{ fx.input.f0[cell*kQ+d] };
            const float expected{ static_cast<float>(f0-directions_of_build().weight_lex(d)) };
            const float stored{ fi[static_cast<size_t>(own[k])] };
            if(stored!=expected) mismatches++;
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(stored-expected)));
            max_r0 = std::max(max_r0, std::fabs(directions_of_build().weight_lex(d)+static_cast<double>(stored)-f0));
        }
    }
    std::printf("attach: mesh cells %u (R %zu), stored-vs-fl32(f0-w) mismatches %llu, max |diff| %.3e, decoded level 0 vs f0 max |fl64(w+g)-f0| %.3e (R0)\n",
                fx.input.cells, embedding.r_cells().size(), static_cast<unsigned long long>(mismatches), max_abs, max_r0);
    const VolumetricWallNativeCapture c{ lbm->volumetric_wall_capture() };
    std::printf("attach: capture available before the first step: %s (expected no)\n", c.available ? "yes" : "no");
    Dump::capture(out+"/attach", c);
    const bool ok{ PhysicsCheck::report("volumetric_wall_native_attach_initial_state", static_cast<double>(mismatches+(c.available ? 1u : 0u)), 0.5) };
    delete lbm;
    return ok ? 0 : 1;
}

/// @brief Task 43 (#979): a second attachment on an attached LBM must be refused by the engine (print_error ends the
/// program with the reason). Reaching the end of this function is the failure.
int run_attach_twice(const Fixture& fx, const std::string& path) {
    LBM* lbm = fx.make_lbm();
    attach_wall(*lbm, path);
    attach_wall(*lbm, path);
    std::printf("PHYSICS volumetric_wall_native: the second attachment was NOT refused FAILED\n");
    delete lbm;
    return 1;
}

int run_sentinel(const Fixture& fx, const std::string& out) {
    // The R cells are flagged TYPE_V but no wall is attached: do_time_step runs ONLY the native stream_collide.
    const VolumetricWallEmbedding embedding(fx.input, fx.solid, forced_r_cells(fx));
    LBM* lbm = fx.make_lbm();
    for(const std::int64_t cell : embedding.r_cells()) lbm->flags[embedding.lattice_of_cell()[static_cast<size_t>(cell)]] |= TYPE_V;
    lbm->run(0u);
    const float sentinel{ -12345.5f };
    bool all_ok{ true };
    for(std::uint32_t parity=0u; parity<2u; parity++) {
        std::vector<fpxx> fi(static_cast<size_t>(lbm->get_N())*kQ, sentinel);
        lbm->volumetric_wall_write_fi(fi);
        lbm->volumetric_wall_set_time_step(parity);        // the step's parity
        lbm->run(1u);
        lbm->volumetric_wall_read_fi(fi);
        std::uint64_t owned_touched{ 0u }, ordinary_untouched{ 0u }, checked_owned{ 0u }, checked_ordinary{ 0u };
        for(const auto& row : embedding.owned()) {
            const std::uint64_t a{ parity ? row.address_parity1 : row.address_parity0 };
            checked_owned++;
            if(fi[static_cast<size_t>(a)]!=sentinel) owned_touched++;
        }
        for(std::uint64_t n=0u; n<fx.addresses.cells(); n++) {
            if(fx.solid[static_cast<size_t>(n)] || (lbm->flags[n]&TYPE_V)) continue;   // active ordinary cells only
            for(const std::uint64_t a : fx.addresses.own_set(n, parity)) {
                checked_ordinary++;
                if(fi[static_cast<size_t>(a)]==sentinel) ordinary_untouched++;
            }
        }
        std::printf("sentinel parity %u: owned addresses %llu, touched by native %llu; ordinary own-set addresses %llu, untouched %llu\n",
                    parity, static_cast<unsigned long long>(checked_owned), static_cast<unsigned long long>(owned_touched),
                    static_cast<unsigned long long>(checked_ordinary), static_cast<unsigned long long>(ordinary_untouched));
        Dump::vector(out+"/sentinel_parity"+std::to_string(parity)+"_fi.f32", fi);
        all_ok = PhysicsCheck::report(parity ? "volumetric_wall_native_sentinel_odd" : "volumetric_wall_native_sentinel_even", static_cast<double>(owned_touched+ordinary_untouched), 0.5) && all_ok;
    }
    delete lbm;
    return all_ok ? 0 : 1;
}

int run_steps(const Fixture& fx, const std::string& path, const std::string& out, const std::uint32_t steps) {
    LBM* lbm = fx.make_lbm();
    attach_wall(*lbm, path);
    lbm->run(0u);
    std::vector<fpxx> fi{};
    lbm->volumetric_wall_read_fi(fi);
    Dump::vector(out+"/step0000_fi.f32", fi);
    OpenInput::dump(*lbm, out+"/step0000");                // task 105: E^0 (open builds)
    for(std::uint32_t s=1u; s<=steps; s++) {
        lbm->run(1u);
        char name[32]{};
        std::snprintf(name, sizeof(name), "/step%04u", s);
        lbm->volumetric_wall_read_fi(fi);
        Dump::vector(out+name+"_fi.f32", fi);
        Dump::capture(out+name, lbm->volumetric_wall_capture());
        OpenInput::dump(*lbm, out+name);                   // task 105: E^s, the state of level s (open builds)
    }
    const VolumetricWallNativeCapture last{ lbm->volumetric_wall_capture() };
    std::printf("steps: %u done, t = %llu, native clamp activations %u, non-finite values in all dumps (host bit pattern) %llu\n", steps, static_cast<unsigned long long>(last.t), last.clamp_count, static_cast<unsigned long long>(Dump::non_finite_total));
    const bool ok{ PhysicsCheck::report("volumetric_wall_native_steps_completed", static_cast<double>(Dump::non_finite_total+last.clamp_count), 0.5) };
    delete lbm;
    return ok ? 0 : 1;
}


/// @brief Host decoding of a raw fi dump with the address algebra: the level-t pre-collision populations of a cell
/// (its load_f(n, t) set) and the post-collision populations a cell stored at step t (its store_f(n, t) set, #260).
/// Lexicographic order, densities (w + g).
class Decoder {
public:
    explicit Decoder(const Fixture& fx) : fx_(fx) {
        const NativeDirections dir(kQ);
        for(std::uint32_t k=0u; k<kQ; k++) {
            fx2lex_[k] = dir.fx_to_lex(k);
            const int cx{ dir.fx_c(k, 0) }, cy{ dir.fx_c(k, 1) }, cz{ dir.fx_c(k, 2) };
            const int shell{ cx*cx+cy*cy+cz*cz };
            w_fx_[k] = dir.weight_lex(fx2lex_[k]);   // task 98: the double weight of the set (D3Q19: 1.0/3.0, 1.0/18.0, 1.0/36.0 as before)
            (void)shell;
            c_lex_[fx2lex_[k]] = { static_cast<double>(cx), static_cast<double>(cy), static_cast<double>(cz) };
        }
        // task 20: both-parity load and store address tables per mesh cell, computed once with the same algebra (integers only)
        const std::size_t cells{ fx.input.cells };
        for(std::uint32_t parity=0u; parity<2u; parity++) {
            load_[parity].resize(cells*kQ);
            store_[parity].resize(cells*kQ);
            for(std::uint32_t cell=0u; cell<cells; cell++) {
                const std::uint64_t n{ fx.lattice(cell) };
                const auto own{ fx.addresses.own_set(n, parity) };
                for(std::uint32_t k=0u; k<kQ; k++) load_[parity][cell*kQ+k] = own[k];
                store_[parity][cell*kQ] = fx.addresses.address(n, 0u);
                for(std::uint32_t i=1u; i<kQ; i+=2u) {
                    store_[parity][cell*kQ+i] = fx.addresses.address(fx.addresses.neighbour(n, i), parity ? i+1u : i);
                    store_[parity][cell*kQ+i+1u] = fx.addresses.address(n, parity ? i : i+1u);
                }
            }
        }
    }
    void load(const std::vector<fpxx>& fi, const std::uint32_t cell, const std::uint32_t parity, std::array<double, kQ>& out) const {
        const std::uint64_t* own{ load_[parity].data()+static_cast<std::size_t>(cell)*kQ };
        for(std::uint32_t k=0u; k<kQ; k++) out[fx2lex_[k]] = w_fx_[k]+static_cast<double>(fi[static_cast<size_t>(own[k])]);
    }
    /// @brief The post-collision populations cell n stored at a step of the given parity: fhn[i] (odd i) at (j[i], parity ? i+1 : i),
    /// fhn[i+1] at (n, parity ? i : i+1), fhn[0] at (n, 0) (store_f of src/kernel.cpp); addresses from the precomputed table.
    void post(const std::vector<fpxx>& fi, const std::uint32_t cell, const std::uint32_t parity, std::array<double, kQ>& out) const {
        const std::uint64_t* st{ store_[parity].data()+static_cast<std::size_t>(cell)*kQ };
        for(std::uint32_t k=0u; k<kQ; k++) out[fx2lex_[k]] = w_fx_[k]+static_cast<double>(fi[static_cast<size_t>(st[k])]);
    }
    const std::array<std::array<double, 3>, kQ>& c_lex() const { return c_lex_; }
private:
    const Fixture& fx_;
    std::array<std::uint32_t, kQ> fx2lex_{};
    std::array<double, kQ> w_fx_{};
    std::array<std::array<double, 3>, kQ> c_lex_{};
    std::array<std::vector<std::uint64_t>, 2> load_{}, store_{};   // [parity][cell*19 + k]
};

/// @brief The static coefficient residual vectors of the supplement (NATIVE_LEDGER_SUPPLEMENT_2026-11-08.md), computed once
/// on the host from the container maps: e_cell = D^T v + G^T 1 - v (states), e_out = R^T v + diag(1/vi) W^T 1 - 1 (rows), and
/// their momentum forms e_pcell_j = D^T a_j - a_j - G^T c_j, e_pout_j = R^T a_j + diag(1/vi) W^T b_j - c_j with b = -c.
class StaticResiduals {
public:
    StaticResiduals(const VolumetricWallNativeInput& in, const std::array<std::array<double, 3>, kQ>& c_lex) {
        const std::uint32_t states{ in.states() }, rows{ in.rows };
        std::vector<double> v(states);
        std::array<std::vector<double>, 3> a{};
        for(std::uint32_t j=0u; j<3u; j++) a[j].assign(states, 0.0);
        for(std::uint32_t st=0u; st<states; st++) { v[st] = in.volume[st/kQ]; for(std::uint32_t j=0u; j<3u; j++) a[j][st] = v[st]*c_lex[st%kQ][j]; }
        e_cell.assign(states, 0.0); e_out.assign(rows, 0.0);
        for(std::uint32_t j=0u; j<3u; j++) { e_pcell[j].assign(states, 0.0); e_pout[j].assign(rows, 0.0); }
        for(std::uint32_t receiver=0u; receiver<states; receiver++) {       // D^T v, D^T a_j
            for(std::uint32_t z=in.D.offsets[receiver]; z<in.D.offsets[receiver+1u]; z++) {
                e_cell[in.D.indices[z]] += v[receiver]*in.D.weights[z];
                for(std::uint32_t j=0u; j<3u; j++) e_pcell[j][in.D.indices[z]] += a[j][receiver]*in.D.weights[z];
            }
            for(std::uint32_t z=in.R.offsets[receiver]; z<in.R.offsets[receiver+1u]; z++) {   // R^T v, R^T a_j
                e_out[in.R.indices[z]] += v[receiver]*in.R.weights[z];
                for(std::uint32_t j=0u; j<3u; j++) e_pout[j][in.R.indices[z]] += a[j][receiver]*in.R.weights[z];
            }
        }
        for(std::uint32_t row=0u; row<rows; row++) {
            const std::array<double, 3>& c_row{ c_lex[in.row_direction[row]] };
            for(std::uint32_t z=in.G.offsets[row]; z<in.G.offsets[row+1u]; z++) {   // G^T 1, G^T c_j
                e_cell[in.G.indices[z]] += in.G.weights[z];
                for(std::uint32_t j=0u; j<3u; j++) e_pcell[j][in.G.indices[z]] -= c_row[j]*in.G.weights[z];
            }
            for(std::uint32_t z=in.W.offsets[row]; z<in.W.offsets[row+1u]; z++) {   // diag(1/vi) W^T 1, diag(1/vi) W^T b_j (b = -c of the receiving row)
                const std::uint32_t producer{ in.W.indices[z] };
                e_out[producer] += in.W.weights[z]/in.prism_volume[producer];
                for(std::uint32_t j=0u; j<3u; j++) e_pout[j][producer] += (-c_row[j])*in.W.weights[z]/in.prism_volume[producer];
            }
        }
        for(std::uint32_t st=0u; st<states; st++) { e_cell[st] -= v[st]; for(std::uint32_t j=0u; j<3u; j++) e_pcell[j][st] -= a[j][st]; }
        for(std::uint32_t row=0u; row<rows; row++) { e_out[row] -= 1.0; for(std::uint32_t j=0u; j<3u; j++) e_pout[j][row] -= c_lex[in.row_direction[row]][j]; }
    }
    std::vector<double> e_cell{}, e_out{};
    std::array<std::vector<double>, 3> e_pcell{}, e_pout{};
};

/// @brief Level inventories, extrema and minima of one decoded fi (used for row 0 and for every step's "after" state).
struct LevelState {
    double mass{}, min_f{ 1e300 }, max_dep{}, max_speed{}, max_uniform{}, max_transverse{}, amplitude{};
    std::array<double, 3> mom{};
    double mass_abs{}; std::array<double, 3> mom_abs{}; std::uint64_t ops{};   // actual absolute sums and additions of this level's reductions (#287)
};

/// @brief The protocol run: VW_N steps with the wall; per-step host series (schema v3, header from the names list),
/// checkpoints at VW_CHECKPOINTS (comma list) and at the first failure (non-finite, negative population, clamp): each
/// checkpoint t > 0 holds the current fi and pending, the previous-level fi (checkpoint_%06u_before_fi.f32) and the
/// capture of the last step; rolling SHA-256 of the raw fi after every step.
/// @brief Wall-clock phase timers of the protocol loop (diagnostic only; no effect on any output value).
struct PhaseTimers {
    double gpu_step{}, readback_fi{}, capture{}, rows{}, hash{}, io{};
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    std::string text() const {
        char buffer[256]{};
        std::snprintf(buffer, sizeof(buffer), "phase seconds: gpu_step %.3f readback_fi %.3f capture %.3f rows %.3f hash %.3f io %.3f", gpu_step, readback_fi, capture, rows, hash, io);
        return buffer;
    }
};

/// @brief The series row computation of one step, factored out of the protocol loop (task 20). Every floating-point
/// expression and the Neumaier accumulation order are those of the task-19 batch executable (fb03759); the inputs are
/// the level-(s-1) raw fi, the level-s raw fi, the step's capture, the pending consumed by the step and the cumulative
/// impulse BEFORE the step. The replay mode feeds it from retained checkpoint files; the protocol mode from the device.
class RowComputer {
public:
    RowComputer(const Fixture& fx, const double u0, const std::uint32_t axis) : fx_(fx), u0_(u0), axis_(axis), embedding_(fx.input, fx.solid, forced_r_cells(fx)), decoder_(fx), statics_(fx.input, decoder_.c_lex()) {
        for(std::uint32_t cell=0u; cell<fx_.input.cells; cell++) mode_norm_ += fx_.input.volume[cell]*fx_.input.mode_phi[cell]*fx_.input.mode_phi[cell];
        capacity_.assign(fx_.input.rows, 0.0);   // W row sums (CF10 BoundedOperator.capacity)
        for(std::uint32_t row=0u; row<fx_.input.rows; row++) for(std::uint32_t z=fx_.input.W.offsets[row]; z<fx_.input.W.offsets[row+1u]; z++) capacity_[row] += fx_.input.W.weights[z];
        exact_zero_.assign(fx_.input.rows, 0u);   // exact zero (no W entry: pending must be exactly 0) vs tiny positive (excluded from the ratio, pending allowed) (#287)
        for(std::uint32_t row=0u; row<fx_.input.rows; row++) {
            if(fx_.input.W.offsets[row+1u]==fx_.input.W.offsets[row]) { exact_zero_[row] = 1u; zero_capacity_rows++; }
            else if(!(capacity_[row]>1e-14*fx_.input.prism_volume[row])) low_capacity_rows++;
        }
        // task 21: the structural operation count k of the traction identity proxy (FRICTION_SCREEN_SPECIFICATION section 5):
        // per facet 4 adds per sampler entry, per row (gather entries + 8), 3 per row for the shell sums and Q, 12 for the
        // projection and the requested value, plus the two facet sums. An instrument comparison proxy, not an error theorem.
        K_ = fx_.input.friction;
        traction_ops_ = 2u*static_cast<std::uint64_t>(fx_.input.facets);
        for(std::uint32_t a=0u; a<fx_.input.facets; a++) {
            const std::uint32_t rows_f{ fx_.input.facet_offsets[a+1u]-fx_.input.facet_offsets[a] };
            std::uint64_t ops{ 4u*static_cast<std::uint64_t>(fx_.input.S.offsets[a+1u]-fx_.input.S.offsets[a])+3u*static_cast<std::uint64_t>(rows_f)+12u };
            for(std::uint32_t r=fx_.input.facet_offsets[a]; r<fx_.input.facet_offsets[a+1u]; r++) ops += static_cast<std::uint64_t>(fx_.input.G.offsets[r+1u]-fx_.input.G.offsets[r])+8u;
            traction_ops_ += ops;
        }
    }
    const VolumetricWallEmbedding& embedding() const { return embedding_; }
    double mode_norm() const { return mode_norm_; }
    std::uint64_t traction_ops() const { return traction_ops_; }
    std::uint32_t zero_capacity_rows{ 0u }, low_capacity_rows{ 0u };

    /// @brief Row 0, the baseline row: the state the instrumented protocol starts from, from its raw fi and pending.
    /// @param fi0 raw populations of the baseline level
    /// @param pending0 pending amounts of the baseline level
    /// @param t0 total time step of the baseline level: 0 without a warm-up, the warm-up length with one (even, so the
    /// decode parity is 0). Task 46: written to t_after; before, the baseline row carried t_after 0 after a warm-up too.
    native_protocol::SeriesRow row_zero(const std::vector<fpxx>& fi0, const std::vector<double>& pending0, const std::uint64_t t0) {
        native_protocol::SeriesRow row{};
        LevelState L{}; level_state(fi0, 0u, L);
        double pm{}, pmin{}, pratio{}; std::array<double, 3> pmom{}; pending_state(pending0, pm, pmom, pmin, pratio);
        row.at("step") = 0.0; row.at("t_after") = static_cast<double>(t0); row.at("mass_fluid") = L.mass; row.at("mass_pending") = pm;
        row.at("momx_fluid") = L.mom[0]; row.at("momy_fluid") = L.mom[1]; row.at("momz_fluid") = L.mom[2]; row.at("momx_pending") = pmom[0]; row.at("momy_pending") = pmom[1]; row.at("momz_pending") = pmom[2];
        row.at("min_f") = L.min_f; row.at("min_pending") = pmin; row.at("min_pending_ratio") = pratio; row.at("zero_capacity_rows") = zero_capacity_rows; row.at("low_capacity_rows") = low_capacity_rows;
        row.at("max_density_departure") = L.max_dep; row.at("max_speed") = L.max_speed; row.at("max_uniform_error") = L.max_uniform; row.at("max_transverse") = L.max_transverse; row.at("amplitude") = L.amplitude;
        row.at("nonfinite_fi") = static_cast<double>(BitPattern::non_finite(fi0));
        row.at("max_abs_pending_zero_capacity") = zero_capacity_max_abs_;
        return row;
    }

    /// @brief The row of step s. J_cum is the cumulative impulse before the step and is advanced by the step's impulse.
    /// `failure` reports the executable's stop condition; `summary` a short printable state.
    native_protocol::SeriesRow row_step(const std::uint32_t s, const std::vector<fpxx>& fi_previous, const std::vector<fpxx>& fi, const VolumetricWallNativeCapture& cap,
                                        const std::vector<double>& pending_previous, std::array<double, 3>& J_cum, std::array<double, 3>& requested_cum, std::array<double, 3>& delivered_cum,
                                        bool& failure, std::string& summary) {
        using native_protocol::SeriesRow;
        const auto& owned{ embedding_.owned() };
        const std::vector<std::int64_t>& rindex{ embedding_.r_index_of_cell() };
        std::array<double, kQ> f{}, p{};
        const std::uint32_t parity_before{ static_cast<std::uint32_t>((cap.t-1u)%2u) }, parity_after{ static_cast<std::uint32_t>(cap.t%2u) };
        SeriesRow row{};
        row.at("step") = s; row.at("t_after") = static_cast<double>(cap.t);
        LevelState L{}; level_state(fi, parity_after, L);
        double min_post_all{ 1e300 };
        native_protocol::Neumaier mass_start_acc{}, defect_mass_acc{}, coeff_cell_mass_acc{};
        std::array<native_protocol::Neumaier, 3> mom_start_acc{}, defect_mom_acc{}, coeff_cell_mom_acc{};
        for(std::uint32_t cell=0u; cell<fx_.input.cells; cell++) {
            const double V{ fx_.input.volume[cell] };
            decoder_.load(fi_previous, cell, parity_before, f);
            if(rindex[cell]>=0) { for(std::uint32_t d=0u; d<kQ; d++) p[d] = cap.post[static_cast<size_t>(rindex[cell])*kQ+d]; }
            else decoder_.post(fi, cell, parity_before, p);
            double rho_start{ 0.0 }, defect_cell{ 0.0 }; std::array<double, 3> j_start{}, defect_cell_mom{};
            for(std::uint32_t d=0u; d<kQ; d++) {
                min_post_all = std::min(min_post_all, p[d]);
                rho_start += f[d]; defect_cell += p[d]-f[d];
                const std::uint32_t st{ cell*kQ+d };
                coeff_cell_mass_acc.add(statics_.e_cell[st]*p[d]);
                for(std::uint32_t a=0u; a<3u; a++) { const double c{ decoder_.c_lex()[d][a] }; j_start[a] += f[d]*c; defect_cell_mom[a] += (p[d]-f[d])*c; coeff_cell_mom_acc[a].add(statics_.e_pcell[a][st]*p[d]); }
            }
            mass_start_acc.add(V*rho_start); defect_mass_acc.add(V*defect_cell);
            for(std::uint32_t a=0u; a<3u; a++) { mom_start_acc[a].add(V*j_start[a]); defect_mom_acc[a].add(V*defect_cell_mom[a]); }
        }
        const double mass_start{ mass_start_acc.value() }, defect_mass{ defect_mass_acc.value() }, coeff_cell_mass{ coeff_cell_mass_acc.value() };
        std::array<double, 3> mom_start{}, defect_mom{}, coeff_cell_mom{};
        for(std::uint32_t a=0u; a<3u; a++) { mom_start[a] = mom_start_acc[a].value(); defect_mom[a] = defect_mom_acc[a].value(); coeff_cell_mom[a] = coeff_cell_mom_acc[a].value(); }
        double pend_start{}, pend_next{}, pmin_start{}, pmin_next{}, pratio_start{}, pratio_next{};
        std::array<double, 3> pmom_start{}, pmom_next{};
        pending_state(pending_previous, pend_start, pmom_start, pmin_start, pratio_start);
        const double pend_start_abs{ pend_abs_acc_.abs_sum }; const std::uint64_t pend_start_ops{ pend_abs_acc_.count };
        pending_state(cap.pending_new, pend_next, pmom_next, pmin_next, pratio_next);
        const double zero_cap_new{ zero_capacity_max_abs_ };
        const double pend_next_abs{ pend_abs_acc_.abs_sum }; const std::uint64_t pend_next_ops{ pend_abs_acc_.count };
        double min_post_R{ 1e300 }, min_q_over_vi{ 1e300 };
        native_protocol::Neumaier shell_acc{}, incoming_acc{}, coeff_out_mass_acc{}; std::array<native_protocol::Neumaier, 3> coeff_out_mom_acc{};
        for(std::uint32_t r=0u; r<fx_.input.rows; r++) {
            shell_acc.add(cap.Q[r]-cap.incoming[r]); incoming_acc.add(cap.incoming[r]); coeff_out_mass_acc.add(statics_.e_out[r]*cap.Q[r]);
            min_q_over_vi = std::min(min_q_over_vi, cap.Q[r]/fx_.input.prism_volume[r]);
            for(std::uint32_t a=0u; a<3u; a++) coeff_out_mom_acc[a].add(statics_.e_pout[a][r]*cap.Q[r]);
        }
        const double shell{ shell_acc.value() }, incoming_sum{ incoming_acc.value() }, coeff_out_mass{ coeff_out_mass_acc.value() };
        std::array<double, 3> coeff_out_mom{};
        for(std::uint32_t a=0u; a<3u; a++) coeff_out_mom[a] = coeff_out_mom_acc[a].value();
        for(const double v : cap.post) min_post_R = std::min(min_post_R, v);
        std::array<native_protocol::Neumaier, 3> J_acc{};
        for(std::uint32_t a=0u; a<fx_.input.facets; a++) for(std::uint32_t k=0u; k<3u; k++) J_acc[k].add(cap.impulse[3u*a+k]);
        std::array<double, 3> J{};
        for(std::uint32_t k=0u; k<3u; k++) { J[k] = J_acc[k].value(); J_cum[k] += J[k]; }
        // task 21: requested tangential impulse (the friction law -K rho_s cs^2 A u_t at the sampled state of the captured R fields through S')
        // and delivered tangential impulse (the captured per-facet impulse minus its normal part), with the identity proxy operand sum
        // (componentwise L1 over x, y, z of every addend of both sides). Section 2.3 of the contract: equal in exact arithmetic.
        std::array<native_protocol::Neumaier, 3> req_acc{}, del_acc{};
        double traction_abs{ 0.0 };
        const double cs2{ 1.0/3.0 };
        for(std::uint32_t a=0u; a<fx_.input.facets; a++) {
            const std::array<double, 3> n{ fx_.input.normal[3u*a], fx_.input.normal[3u*a+1u], fx_.input.normal[3u*a+2u] };
            double rho_s{ 0.0 }, abs_rho_terms{ 0.0 }; std::array<double, 3> u_s{}, abs_u_terms{};
            for(std::uint32_t z=fx_.input.S.offsets[a]; z<fx_.input.S.offsets[a+1u]; z++) {
                const std::int64_t rb{ rindex[fx_.input.S.indices[z]] };
                if(rb<0) throw std::runtime_error("sampler support cell outside R (structural contract violated)");
                const double sw{ fx_.input.S.weights[z] }, rho_b{ cap.fields[4u*static_cast<size_t>(rb)] };
                rho_s += sw*rho_b; abs_rho_terms += std::fabs(sw)*std::fabs(rho_b);
                for(std::uint32_t k=0u; k<3u; k++) { const double ub{ cap.fields[4u*static_cast<size_t>(rb)+1u+k] }; u_s[k] += sw*ub; abs_u_terms[k] += std::fabs(sw)*std::fabs(ub); }
            }
            const double un{ n[0]*u_s[0]+n[1]*u_s[1]+n[2]*u_s[2] };
            const double Jn{ n[0]*cap.impulse[3u*a]+n[1]*cap.impulse[3u*a+1u]+n[2]*cap.impulse[3u*a+2u] };
            const double factor{ K_*cs2*fx_.input.area[a] };
            for(std::uint32_t k=0u; k<3u; k++) {
                const double u_t{ u_s[k]-n[k]*un };
                req_acc[k].add(-factor*rho_s*u_t);
                del_acc[k].add(cap.impulse[3u*a+k]-n[k]*Jn);
                traction_abs += factor*(std::fabs(rho_s)*std::fabs(u_t)+abs_rho_terms*std::fabs(u_t)+std::fabs(rho_s)*abs_u_terms[k]);
            }
            for(std::uint32_t r=fx_.input.facet_offsets[a]; r<fx_.input.facet_offsets[a+1u]; r++) {
                const std::array<double, 3>& c{ decoder_.c_lex()[fx_.input.row_direction[r]] };
                const double amount{ std::fabs(cap.Q[r])+std::fabs(cap.incoming[r]) };
                for(std::uint32_t k=0u; k<3u; k++) traction_abs += std::fabs(c[k])*amount;
            }
        }
        std::array<double, 3> requested{}, delivered{};
        for(std::uint32_t k=0u; k<3u; k++) { requested[k] = req_acc[k].value(); delivered[k] = del_acc[k].value(); requested_cum[k] += requested[k]; delivered_cum[k] += delivered[k]; }
        const double traction_bound{ native_protocol::ledger_bound(traction_ops_, traction_abs) };
        native_protocol::Neumaier inc_mass_acc{}, bound_mass_acc{}; std::array<native_protocol::Neumaier, 3> inc_mom_acc{}, bound_mom_acc{};
        for(std::size_t o=0u; o<owned.size(); o++) {
            const std::uint64_t address{ parity_before==0u ? owned[o].address_parity0 : owned[o].address_parity1 };
            const double w{ directions_of_build().weight_lex(owned[o].direction) };
            const double g_pre{ cap.prestore[o]-w };
            const double stored{ static_cast<double>(fi[static_cast<size_t>(address)]) };
            const double V{ fx_.input.volume[owned[o].receiver_cell] };
            const double half{ native_protocol::half_ulp32_conservative(g_pre) };
            inc_mass_acc.add(V*(stored-g_pre)); bound_mass_acc.add(V*half);
            for(std::uint32_t a=0u; a<3u; a++) { const double c{ decoder_.c_lex()[owned[o].direction][a] }; inc_mom_acc[a].add(V*(stored-g_pre)*c); bound_mom_acc[a].add(V*half*std::fabs(c)); }
        }
        const double inc_mass{ inc_mass_acc.value() }, bound_mass{ bound_mass_acc.value() };
        std::array<double, 3> inc_mom{}, bound_mom{};
        for(std::uint32_t a=0u; a<3u; a++) { inc_mom[a] = inc_mom_acc[a].value(); bound_mom[a] = bound_mom_acc[a].value(); }
        const double mass_identity_residual{ (L.mass+pend_next)-(mass_start+pend_start)-(defect_mass+coeff_cell_mass+coeff_out_mass+shell+inc_mass) };
        const double ledger_abs_sum_mass{ mass_start_acc.abs_sum+L.mass_abs+pend_start_abs+pend_next_abs+defect_mass_acc.abs_sum+coeff_cell_mass_acc.abs_sum+coeff_out_mass_acc.abs_sum+shell_acc.abs_sum+inc_mass_acc.abs_sum };
        const std::uint64_t ledger_ops_mass{ mass_start_acc.count+L.ops+pend_start_ops+pend_next_ops+defect_mass_acc.count+coeff_cell_mass_acc.count+coeff_out_mass_acc.count+shell_acc.count+inc_mass_acc.count+8u };
        const double ledger_bound_mass{ native_protocol::ledger_bound(ledger_ops_mass, ledger_abs_sum_mass) };
        std::array<double, 3> ledger_abs_sum_mom{}, ledger_bound_mom{}, storage_allowance_mom{};
        std::uint64_t ledger_ops_mom{ 0u };
        for(std::uint32_t a=0u; a<3u; a++) {
            ledger_abs_sum_mom[a] = mom_start_acc[a].abs_sum+L.mom_abs[a]+pend_start_abs+pend_next_abs+J_acc[a].abs_sum+defect_mom_acc[a].abs_sum+coeff_cell_mom_acc[a].abs_sum+coeff_out_mom_acc[a].abs_sum+inc_mom_acc[a].abs_sum;
            ledger_ops_mom = std::max(ledger_ops_mom, mom_start_acc[a].count+L.ops+pend_start_ops+pend_next_ops+J_acc[a].count+defect_mom_acc[a].count+coeff_cell_mom_acc[a].count+coeff_out_mom_acc[a].count+inc_mom_acc[a].count+8u);
            ledger_bound_mom[a] = native_protocol::ledger_bound(ledger_ops_mom, ledger_abs_sum_mom[a]);
            storage_allowance_mom[a] = native_protocol::ledger_bound(inc_mom_acc[a].count, inc_mom_acc[a].abs_sum);
        }
        const double storage_allowance_mass{ native_protocol::ledger_bound(inc_mass_acc.count, inc_mass_acc.abs_sum) };
        std::array<double, 3> mom_identity_residual{};
        for(std::uint32_t a=0u; a<3u; a++) mom_identity_residual[a] = (L.mom[a]+pmom_next[a])-(mom_start[a]+pmom_start[a])-J[a]-(defect_mom[a]+coeff_cell_mom[a]+coeff_out_mom[a]+inc_mom[a]);
        const std::uint64_t nonfinite_fi{ BitPattern::non_finite(fi) };
        const std::uint64_t nonfinite_cap{ BitPattern::non_finite(cap.post)+BitPattern::non_finite(cap.Q)+BitPattern::non_finite(cap.incoming)+BitPattern::non_finite(cap.pending_new)+BitPattern::non_finite(cap.impulse)+BitPattern::non_finite(cap.prestore)+BitPattern::non_finite(cap.fields) };
        row.at("mass_fluid") = L.mass; row.at("mass_pending") = pend_next;
        row.at("momx_fluid") = L.mom[0]; row.at("momy_fluid") = L.mom[1]; row.at("momz_fluid") = L.mom[2]; row.at("momx_pending") = pmom_next[0]; row.at("momy_pending") = pmom_next[1]; row.at("momz_pending") = pmom_next[2];
        row.at("Jx") = J[0]; row.at("Jy") = J[1]; row.at("Jz") = J[2]; row.at("Jx_cum") = J_cum[0]; row.at("Jy_cum") = J_cum[1]; row.at("Jz_cum") = J_cum[2];
        row.at("min_f") = L.min_f; row.at("min_post_R") = min_post_R; row.at("min_post_all") = min_post_all; row.at("min_Q_over_vi") = min_q_over_vi; row.at("min_pending") = pmin_next; row.at("min_pending_ratio") = pratio_next; row.at("zero_capacity_rows") = zero_capacity_rows;
        row.at("max_density_departure") = L.max_dep; row.at("max_speed") = L.max_speed; row.at("max_uniform_error") = L.max_uniform; row.at("max_transverse") = L.max_transverse; row.at("amplitude") = L.amplitude;
        row.at("clamp_count") = cap.clamp_count; row.at("nonfinite_fi") = static_cast<double>(nonfinite_fi); row.at("nonfinite_capture") = static_cast<double>(nonfinite_cap);
        row.at("storage_increment_mass") = inc_mass; row.at("storage_increment_momx") = inc_mom[0]; row.at("storage_increment_momy") = inc_mom[1]; row.at("storage_increment_momz") = inc_mom[2];
        row.at("storage_bound_mass") = bound_mass; row.at("storage_bound_momx") = bound_mom[0]; row.at("storage_bound_momy") = bound_mom[1]; row.at("storage_bound_momz") = bound_mom[2];
        row.at("collision_defect_mass") = defect_mass; row.at("collision_defect_momx") = defect_mom[0]; row.at("collision_defect_momy") = defect_mom[1]; row.at("collision_defect_momz") = defect_mom[2];
        row.at("coeff_cell_mass") = coeff_cell_mass; row.at("coeff_out_mass") = coeff_out_mass; row.at("coeff_cell_momx") = coeff_cell_mom[0]; row.at("coeff_cell_momy") = coeff_cell_mom[1]; row.at("coeff_cell_momz") = coeff_cell_mom[2];
        row.at("coeff_out_momx") = coeff_out_mom[0]; row.at("coeff_out_momy") = coeff_out_mom[1]; row.at("coeff_out_momz") = coeff_out_mom[2];
        row.at("shell_residual") = shell; row.at("mass_identity_residual") = mass_identity_residual; row.at("mom_identity_residual_x") = mom_identity_residual[0]; row.at("mom_identity_residual_y") = mom_identity_residual[1]; row.at("mom_identity_residual_z") = mom_identity_residual[2]; row.at("incoming_sum") = incoming_sum;
        row.at("max_abs_pending_zero_capacity") = zero_cap_new; row.at("low_capacity_rows") = low_capacity_rows;
        row.at("ledger_abs_sum_mass") = ledger_abs_sum_mass; row.at("ledger_ops_mass") = static_cast<double>(ledger_ops_mass); row.at("ledger_bound_mass") = ledger_bound_mass;
        row.at("ledger_abs_sum_momx") = ledger_abs_sum_mom[0]; row.at("ledger_abs_sum_momy") = ledger_abs_sum_mom[1]; row.at("ledger_abs_sum_momz") = ledger_abs_sum_mom[2]; row.at("ledger_ops_mom") = static_cast<double>(ledger_ops_mom);
        row.at("ledger_bound_momx") = ledger_bound_mom[0]; row.at("ledger_bound_momy") = ledger_bound_mom[1]; row.at("ledger_bound_momz") = ledger_bound_mom[2];
        row.at("storage_allowance_mass") = storage_allowance_mass; row.at("storage_allowance_momx") = storage_allowance_mom[0]; row.at("storage_allowance_momy") = storage_allowance_mom[1]; row.at("storage_allowance_momz") = storage_allowance_mom[2];
        row.at("requested_tangential_impulse_x") = requested[0]; row.at("requested_tangential_impulse_y") = requested[1]; row.at("requested_tangential_impulse_z") = requested[2];
        row.at("requested_tangential_impulse_x_cum") = requested_cum[0]; row.at("requested_tangential_impulse_y_cum") = requested_cum[1]; row.at("requested_tangential_impulse_z_cum") = requested_cum[2];
        row.at("delivered_tangential_impulse_x") = delivered[0]; row.at("delivered_tangential_impulse_y") = delivered[1]; row.at("delivered_tangential_impulse_z") = delivered[2];
        row.at("delivered_tangential_impulse_x_cum") = delivered_cum[0]; row.at("delivered_tangential_impulse_y_cum") = delivered_cum[1]; row.at("delivered_tangential_impulse_z_cum") = delivered_cum[2];
        row.at("traction_identity_abs_sum") = traction_abs; row.at("traction_identity_ops") = static_cast<double>(traction_ops_); row.at("traction_identity_bound") = traction_bound;
        failure = nonfinite_fi>0u || nonfinite_cap>0u || L.min_f<-1e-12 || min_post_all<-1e-12 || min_q_over_vi<-1e-12 || pmin_next<-1e-12 || pratio_next<-1e-12 || zero_cap_new!=0.0 || cap.clamp_count>0u;
#ifdef VOLUMETRIC_WALL_T119
        // task 119 (restricted fail-fast test): the T115 screens (density departure 0.05, speed 0.3) and the explicit projection-required refusal (raw Q < 0, or the mass/tangential screen outside 1e-13): stop, keep the raw states
        const bool t119_density{ L.max_dep>0.05 }, t119_speed{ L.max_speed>0.3 }, t119_projection{ cap.t119_negative_q>0u||cap.t119_p1>0u };
        failure = failure || t119_density || t119_speed || t119_projection;
        const std::string t119_reason{ std::string(t119_density ? " DENSITY_SCREEN" : "")+(t119_speed ? " SPEED_SCREEN" : "")+(t119_projection ? " PR2_PROJECTION_REQUIRED(raw Q<0 count "+std::to_string(cap.t119_negative_q)+", feasibility count "+std::to_string(cap.t119_p1)+")" : "") };
#endif // VOLUMETRIC_WALL_T119
#ifdef VOLUMETRIC_WALL_SRD
        // task 121: the six device counters of the redistribution are CUMULATIVE since the attachment. A non-positive or non-finite neighbourhood mean (0), a singular B (1),
        // a non-finite reconstruction (2) or a non-finite redistributed population (5) stop the run (the CPU raises); nothing is repaired. Counter 3 (limiter entries) is a report; counter 4 (a negative redistributed population) is covered by the min_f rule.
        const bool srd_invalid{ cap.srd_stats.size()!=6u || cap.srd_stats[0]>0u || cap.srd_stats[1]>0u || cap.srd_stats[2]>0u || cap.srd_stats[5]>0u };
        failure = failure || srd_invalid;
        const std::string srd_reason{ srd_invalid ? " SRD_INVALID" : "" };
#endif // VOLUMETRIC_WALL_SRD
        char buffer[512]{};
        std::snprintf(buffer, sizeof(buffer), "mass %.12f pending %.6e max|rho-1| %.3e min_f %.4f A %.6e clamp %u identity %.2e | nonfinite fi %llu cap %llu min_post_all %.3e min_Q/vi %.3e min_pending %.3e min_pending_ratio %.3e zero_capacity_pending %.3e",
                      L.mass, pend_next, L.max_dep, L.min_f, L.amplitude, cap.clamp_count, mass_identity_residual, static_cast<unsigned long long>(nonfinite_fi), static_cast<unsigned long long>(nonfinite_cap), min_post_all, min_q_over_vi, pmin_next, pratio_next, zero_cap_new);
        summary = buffer;
#ifdef VOLUMETRIC_WALL_T119
        summary += " | T119"+t119_reason;
#endif // VOLUMETRIC_WALL_T119
#ifdef VOLUMETRIC_WALL_SRD
        if(cap.srd_stats.size()==6u) {
            summary += " | SRD"+srd_reason+" counters(cumulative) mean<=0:"+std::to_string(cap.srd_stats[0])+" singularB:"+std::to_string(cap.srd_stats[1])+" nonfinite_recon:"+std::to_string(cap.srd_stats[2])+" limiter_entries:"+std::to_string(cap.srd_stats[3])+" negative_fprime:"+std::to_string(cap.srd_stats[4])+" nonfinite_fprime:"+std::to_string(cap.srd_stats[5]);
        }
#endif // VOLUMETRIC_WALL_SRD
        return row;
    }

private:
    void level_state(const std::vector<fpxx>& raw, const std::uint32_t parity, LevelState& L) const {
        std::array<double, kQ> f{};
        L = LevelState{};
        native_protocol::Neumaier mass_acc{}, amp_acc{}; std::array<native_protocol::Neumaier, 3> mom_acc{};
        for(std::uint32_t cell=0u; cell<fx_.input.cells; cell++) {
            decoder_.load(raw, cell, parity, f);
            const double V{ fx_.input.volume[cell] };
            double rho{ 0.0 }; std::array<double, 3> j{};
            for(std::uint32_t d=0u; d<kQ; d++) { rho += f[d]; L.min_f = std::min(L.min_f, f[d]); for(std::uint32_t a=0u; a<3u; a++) j[a] += f[d]*decoder_.c_lex()[d][a]; }
            mass_acc.add(V*rho); for(std::uint32_t a=0u; a<3u; a++) mom_acc[a].add(V*j[a]);
            L.max_dep = std::max(L.max_dep, std::fabs(rho-1.0));
            const double ux{ j[0]/rho }, uy{ j[1]/rho }, uz{ j[2]/rho };
#ifdef VOLUMETRIC_WALL_SRD
            // task 121: the speed screen uses the PHYSICAL velocity (sum c f + F/2)/rho of the half-force convention (the earlier fixtures omitted F/2); the other diagnostics keep ux, uy, uz
            const double sx{ (j[0]+0.5*static_cast<double>(fx_.force.value[0]))/rho }, sy{ (j[1]+0.5*static_cast<double>(fx_.force.value[1]))/rho }, sz{ (j[2]+0.5*static_cast<double>(fx_.force.value[2]))/rho };
            L.max_speed = std::max(L.max_speed, std::sqrt(sx*sx+sy*sy+sz*sz));
#else
            L.max_speed = std::max(L.max_speed, std::sqrt(ux*ux+uy*uy+uz*uz));
#endif // VOLUMETRIC_WALL_SRD
            // task 23: the reference axis a_ (z by default). For a_ = 2 these are the task-19 expressions term for term.
            const std::array<double, 3> u{ ux, uy, uz };
            const double t0{ axis_==0u ? (u[0]-u0_)*(u[0]-u0_) : u[0]*u[0] };
            const double t1{ axis_==1u ? (u[1]-u0_)*(u[1]-u0_) : u[1]*u[1] };
            const double t2{ axis_==2u ? (u[2]-u0_)*(u[2]-u0_) : u[2]*u[2] };
            L.max_uniform = std::max(L.max_uniform, std::sqrt(t0+t1+t2));
            const double transverse{ axis_==2u ? u[0]*u[0]+u[1]*u[1] : (axis_==1u ? u[0]*u[0]+u[2]*u[2] : u[1]*u[1]+u[2]*u[2]) };
            L.max_transverse = std::max(L.max_transverse, std::sqrt(transverse));
            amp_acc.add(V*u[axis_]*fx_.input.mode_phi[cell]);
        }
        L.mass = mass_acc.value(); for(std::uint32_t a=0u; a<3u; a++) { L.mom[a] = mom_acc[a].value(); L.mom_abs[a] = mom_acc[a].abs_sum; }
        L.mass_abs = mass_acc.abs_sum; L.ops = mass_acc.count;
        L.amplitude = amp_acc.value()/mode_norm_;
    }
    void pending_state(const std::vector<double>& s_now, double& mass, std::array<double, 3>& mom, double& minimum, double& min_ratio) {
        native_protocol::Neumaier mass_acc{}; std::array<native_protocol::Neumaier, 3> mom_acc{};
        minimum = 1e300; min_ratio = 1e300; zero_capacity_max_abs_ = 0.0;
        for(std::uint32_t r=0u; r<fx_.input.rows; r++) {
            const std::array<double, 3>& c{ decoder_.c_lex()[fx_.input.row_direction[r]] };
            mass_acc.add(s_now[r]); minimum = std::min(minimum, s_now[r]);
            for(std::uint32_t a=0u; a<3u; a++) mom_acc[a].add(-s_now[r]*c[a]);
            if(capacity_[r]>1e-14*fx_.input.prism_volume[r]) min_ratio = std::min(min_ratio, s_now[r]/capacity_[r]);
            if(exact_zero_[r]) zero_capacity_max_abs_ = std::max(zero_capacity_max_abs_, std::fabs(s_now[r]));
        }
        pend_abs_acc_ = mass_acc;
        mass = mass_acc.value(); for(std::uint32_t a=0u; a<3u; a++) mom[a] = mom_acc[a].value();
        if(min_ratio==1e300) min_ratio = 0.0;   // no row with capacity (CF10 convention: 0)
    }
    const Fixture& fx_;
    const double u0_;
    const std::uint32_t axis_;            ///< task 23: lexicographic index of the reference axis (1 y, 2 z)
    const VolumetricWallEmbedding embedding_;
    const Decoder decoder_;
    const StaticResiduals statics_;
    std::vector<double> capacity_{};
    std::vector<std::uint8_t> exact_zero_{};
    double mode_norm_{ 0.0 }, zero_capacity_max_abs_{ 0.0 };
    native_protocol::Neumaier pend_abs_acc_{};
    double K_{ 0.0 };                     ///< the container's friction coefficient (task 21)
    std::uint64_t traction_ops_{ 0u };    ///< structural operation count of the traction identity proxy
};

/// @brief Task 21 preflight: the container's friction K must equal VW_K bitwise when VW_K is set (registration binding).
void preflight_k(const Fixture& fx) {
    const std::string declared{ env("VW_K") };
    if(declared.empty()) { std::printf("preflight: K %.17g (VW_K unset)\n", fx.input.friction); return; }
    const double expected{ std::stod(declared) };
    if(std::memcmp(&expected, &fx.input.friction, sizeof(double))!=0) throw std::runtime_error("container friction K differs bitwise from VW_K "+declared);
    std::printf("preflight: K %.17g bound to VW_K\n", fx.input.friction);
}

/// @brief SHA-256 hex of a double array (little-endian IEEE-754 bytes), for the per-step pending chain (task 21).
std::string sha256_hex_of(const std::vector<double>& values) {
    native_protocol::Sha256 h{};
    h.update(values.data(), values.size()*sizeof(double));
    return h.hex();
}

/// @brief Task 23: the reference axis of the uniform-flow diagnostics (sign binding, max_uniform_error, max_transverse,
/// mode amplitude), from the TOP-LEVEL provenance member "reference_axis", read with the structural JSON reader of
/// provenance_axis.hpp (#442: any JSON whitespace; nested or string content never counts). Absent member = z (every
/// accepted container, so the existing default is unchanged). Accepted values: "y" (1) and "z" (2). Rejected: any other
/// value, a non-string value, a duplicated top-level member, invalid JSON. When VW_AXIS is set it must equal the axis.
/// @param fx The fixture whose provenance is read.
/// @return The lexicographic component index of the reference axis (1 for y, 2 for z).
/// @throws std::runtime_error On an unsupported, malformed or duplicated axis, or on a VW_AXIS mismatch.
std::uint32_t reference_axis(const Fixture& fx) {
    const std::string bound{ env("VW_AXIS") };
    const std::uint32_t axis{ provenance_axis::resolve_axis(fx.input.provenance, bound) };   // #442: structural top-level JSON reader
    std::printf("preflight: reference axis %s%s\n", axis==1u ? "y" : "z", bound.empty() ? " (VW_AXIS unset)" : " bound to VW_AXIS");
    return axis;
}

/// @brief The signed u0 of the container (top-level provenance field; the nested notes follow it) and the preflight (#290).
/// Task 23: the sign binding uses the reference axis component (z by default, unchanged expressions for z).
double preflight_u0(const Fixture& fx, const std::uint32_t axis) {
    const double u0{ provenance_axis::resolve_u0(fx.input.provenance) };   // #442: structural top-level number, finite
    const std::string expected_case{ env("VW_CASE") };
    const std::string key{ "\"case\": \"" }; const size_t pos{ fx.input.provenance.find(key) };
    const std::string declared{ pos==std::string::npos ? std::string() : fx.input.provenance.substr(pos+key.size(), fx.input.provenance.find('"', pos+key.size())-(pos+key.size())) };
    if(!expected_case.empty() && declared!=expected_case) throw std::runtime_error("container case '"+declared+"' differs from VW_CASE '"+expected_case+"'");
    const NativeDirections dir(kQ);
    double num{ 0.0 }, den{ 0.0 }, amp_num{ 0.0 }, amp_den{ 0.0 };
    for(std::uint32_t cell=0u; cell<fx.input.cells; cell++) {
        double rho{ 0.0 }, ja{ 0.0 };
        for(std::uint32_t d=0u; d<kQ; d++) { rho += fx.input.f0[cell*kQ+d]; ja += fx.input.f0[cell*kQ+d]*dir.lex_c(d, axis); }
        const double V{ fx.input.volume[cell] }, ua{ ja/rho }, phi{ fx.input.mode_phi[cell] };
        num += V*ua; den += V; amp_num += V*ua*phi; amp_den += V*phi*phi;
    }
    const double mean_ua{ num/den }, amplitude_ua{ amp_num/amp_den };
    const double measure{ std::fabs(mean_ua)>1e-12*std::fabs(u0) ? mean_ua : amplitude_ua };   // the (1,1) mode has zero mean: decay cases bind through the amplitude
    const char axis_letter{ axis==1u ? 'y' : 'z' };
    if(u0!=0.0 && ((measure>0.0)!=(u0>0.0))) throw std::runtime_error(std::string("decoded f0 (mean u_")+axis_letter+" or mode amplitude) has not the sign of the declared u0");
    std::printf("preflight: case '%s' u0 %.6e mean u_%c(f0) %.6e amplitude(f0) %.6e\n", declared.c_str(), u0, axis_letter, mean_ua, amplitude_ua);
    return u0;
}

/// @brief The protocol run (as the task-19 batch executable, with the row computation in RowComputer and phase timers).
int run_protocol(const Fixture& fx, const std::string& path, const std::string& out, const std::uint32_t steps, const std::string& checkpoint_list) {
    using native_protocol::SeriesRow;
    std::vector<std::uint32_t> checkpoints{};
    { std::stringstream ss(checkpoint_list); std::string item{}; while(std::getline(ss, item, ',')) { if(!item.empty()) checkpoints.push_back(static_cast<std::uint32_t>(std::stoul(item))); } }
    const std::uint32_t axis{ reference_axis(fx) };
    const double u0{ preflight_u0(fx, axis) };
    preflight_k(fx);
    RowComputer rows(fx, u0, axis);
    PhaseTimers timers{};
    LBM* lbm = fx.make_lbm();
    attach_wall(*lbm, path);
    lbm->run(0u);
    native_protocol::Sha256 rolling{};
    std::vector<fpxx> fi{}, fi_previous{};
    std::vector<double> pending{};
    std::vector<double> open_before{};   // task 105 (#2421): the authoritative FP64 E state before the current update (open builds; empty otherwise)
    std::ofstream series_bin(out+"/series.f64", std::ios::binary);
    { std::ofstream header(out+"/series_header.txt"); header << SeriesRow::header() << "\n"; }
    std::ofstream hashes(out+"/rolling_sha256.txt");
    const auto write_row = [&](SeriesRow& row) { series_bin.write(reinterpret_cast<const char*>(row.v.data()), static_cast<std::streamsize>(row.v.size()*sizeof(double))); };
    const auto checkpoint = [&](const std::uint32_t step, const char* why, const std::vector<fpxx>& before, const VolumetricWallNativeCapture* cap) {
        const double t0{ PhaseTimers::now() };
        char name[48]{};
        std::snprintf(name, sizeof(name), "/checkpoint_%06u", step);
        lbm->volumetric_wall_read_fi(fi);
        lbm->volumetric_wall_read_pending(pending);
        Dump::vector(out+name+"_fi.f32", fi);
        Dump::vector(out+name+"_pending.f64", pending);
#if defined(VOLUMETRIC_WALL_SRD) && defined(VOLUMETRIC_WALL_OPEN_X)
        if(step>0u && env("VW_SRD_CAPTURE")=="1") { std::vector<fpxx> pe{}; lbm->volumetric_wall_pre_wall_fi(pe); Dump::vector(out+name+"_pre_wall_fi.f32", pe); }   // task 125: for the oracle budget decode
#endif
#ifdef VOLUMETRIC_WALL_SRD
        if(step>0u && env("VW_SRD_CAPTURE")=="1") { std::vector<fpxx> pre{}; lbm->volumetric_wall_srd_pre_fi(pre); Dump::vector(out+name+"_pre_srd_fi.f32", pre); }   // task 124: the stored state the SRD stages of this update consumed (diagnostic copy)
#endif // VOLUMETRIC_WALL_SRD
        if(step>0u) { Dump::vector(out+name+"_before_fi.f32", before); if(cap) Dump::capture(out+name+"_capture", *cap); }
        OpenInput::dump(*lbm, out+name);                   // task 105: the E state of this checkpoint's level (open builds)
#ifdef VOLUMETRIC_WALL_OPEN_X
        if(step>0u) Dump::vector(out+name+"_before_open_state.f64", open_before); // #2421: E^(n-1), the state the captured update read
#endif // VOLUMETRIC_WALL_OPEN_X
        std::ofstream meta(out+name+"_meta.txt");
        meta << "step " << step << "\nt " << lbm->get_t() << "\nreason " << why << "\nbefore_fi " << (step>0u ? "yes" : "no") << "\ncapture " << (step>0u && cap ? "yes" : "no") << "\n";
        timers.io += PhaseTimers::now()-t0;
    };
    std::array<double, 3> J_cum{}, requested_cum{}, delivered_cum{};
    bool failed{ false };
    std::uint32_t first_failure{ 0u };
    const std::string warmup_text{ env("VW_WARMUP") };
    const std::uint32_t warmup{ warmup_text.empty() ? 0u : static_cast<std::uint32_t>(std::stoul(warmup_text)) };
    if(!warmup_text.empty()) {   // task 45: unobserved steps at engine speed before the instrumented window
        if(warmup==0u || warmup%2u!=0u) {
            throw std::invalid_argument("VW_WARMUP must be an even number > 0 (the window's step parity must equal the parity of t)");
        }
        lbm->volumetric_wall_read_fi(fi);
        lbm->volumetric_wall_read_pending(pending);
        Dump::vector(out+"/warmup_initial_fi.f32", fi);
        Dump::vector(out+"/warmup_initial_pending.f64", pending);
        const double w0{ PhaseTimers::now() };
        const std::string middle_text{ env("VW_WARMUP_CHECKPOINT") };
        std::uint32_t remaining{ warmup };
        if(!middle_text.empty()) {   // #1012: one separated state for the stationarity comparison; two readbacks, no per-step instrument
            const std::uint32_t middle{ static_cast<std::uint32_t>(std::stoul(middle_text)) };
            if(middle==0u || middle>=warmup || middle%2u!=0u) {
                throw std::invalid_argument("VW_WARMUP_CHECKPOINT must be even and inside (0, VW_WARMUP)");
            }
            lbm->run(static_cast<ulong>(middle));
            lbm->volumetric_wall_read_fi(fi);
            lbm->volumetric_wall_read_pending(pending);
            Dump::vector(out+"/warmup_checkpoint_fi.f32", fi);
            Dump::vector(out+"/warmup_checkpoint_pending.f64", pending);
            std::ofstream meta(out+"/warmup_checkpoint_meta.txt");
            meta << "t " << lbm->get_t() << "\nclamp_count " << lbm->volumetric_wall_capture().clamp_count << "\n";
            remaining = warmup-middle;
        }
        lbm->run(static_cast<ulong>(remaining));
        std::printf("warm-up: %u steps without readback in %.3f s; the instrumented window starts at t = %llu\n", warmup, PhaseTimers::now()-w0, static_cast<unsigned long long>(lbm->get_t()));
    }
    if(std::find(checkpoints.begin(), checkpoints.end(), 0u)!=checkpoints.end()) checkpoint(0u, "registered", fi, nullptr);
#ifdef VOLUMETRIC_WALL_OPEN_X
    // task 105 (#2410): the OPEN inventory of every level, labelled and separate from the closed series (whose mass_fluid and
    // momentum columns keep their closed meaning: every fluid cell and every pending row). Omega = the mesh cells that are not E
    // cells; pending: the interior rows only (exterior rows hold no pending, O4). Columns: step, t, M_open = sum_Omega V sum f +
    // sum_interior s, Px, Py, Pz = sum_Omega V sum c f - sum_interior c_r s_r, mass_pending_interior, open_failures (cumulative).
    std::ofstream open_inventory_bin(out+"/open_inventory.f64", std::ios::binary);
    { std::ofstream header(out+"/open_inventory_header.txt"); header << "step,t,M_open,Px_open,Py_open,Pz_open,mass_pending_interior,open_failures\n# Omega = mesh cells except E cells; pending = interior rows (contract Section 4); the closed series.f64 columns are all-cell quantities\n"; }
    { std::ofstream note(out+"/series_header_note.txt"); note << "VOLUMETRIC_WALL_OPEN_X build: series.f64 mass/momentum columns include the E cells and every row (the closed-box quantities); the open inventory is open_inventory.f64\n"; }
    const Decoder open_decoder(fx);
    const fluidx3d::experimental::VolumetricWallOpenLayout& open_layout{ lbm->volumetric_wall_open_layout() };
    const auto open_row = [&](const std::uint32_t step, const std::vector<fpxx>& state, const std::vector<double>& pend) {
        const std::uint32_t parity{ static_cast<std::uint32_t>(lbm->get_t()%2ull) };
        native_protocol::Neumaier mass{}, pend_mass{};
        std::array<native_protocol::Neumaier, 3> mom{};
        std::array<double, kQ> f{};
        for(std::uint32_t cell=0u; cell<fx.input.cells; cell++) {
            if(open_layout.slot_of_cell()[cell]>=0) continue;
            open_decoder.load(state, cell, parity, f);
            const double V{ fx.input.volume[cell] };
            for(std::uint32_t d=0u; d<kQ; d++) { mass.add(V*f[d]); for(std::uint32_t a=0u; a<3u; a++) mom[a].add(V*f[d]*open_decoder.c_lex()[d][a]); }
        }
        for(std::uint32_t r=0u; r<fx.input.rows; r++) {
            if(open_layout.exterior_row()[r]!=0u) continue;
            pend_mass.add(pend[r]);
            for(std::uint32_t a=0u; a<3u; a++) mom[a].add(-open_decoder.c_lex()[fx.input.row_direction[r]][a]*pend[r]);
        }
        const std::array<double, 8> v{ static_cast<double>(step), static_cast<double>(lbm->get_t()), mass.value()+pend_mass.value(), mom[0].value(), mom[1].value(), mom[2].value(),
                                       pend_mass.value(), static_cast<double>(lbm->volumetric_wall_open_failures()) };
        open_inventory_bin.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()*sizeof(double)));
    };
    // #2422: the four boundary transfers of every update (contract Section 4, the reference OpenPass), formed on the host from
    // the per-step capture and the container tables, one row per step in open_flux.f64: step, t, then mass and momentum
    // (x, y, z) of (1) E -> Omega, (2) exterior return, (3) Omega -> E, (4) exterior gather, then the interior impulse sum J
    // (x, y, z): 2 + 16 + 3 = 21 doubles. Sources: post of R cells (cap.post; E cells are R cells and hold feq3 of E^n, no
    // arm), the stored post of ordinary donors (decoded store set), cap.Q, cap.incoming, cap.returned and cap.direct_increment
    // (owned states of R receivers), cap.impulse (exterior facets are 0). (3) = direct from Omega donors + their arm increment +
    // the returns into E minus the exterior-row part; (4) = incoming of exterior rows minus their E-donor gathers (the Omega part
    // with its arms); exterior rows hold no pending (O4).
    std::ofstream open_flux_bin(out+"/open_flux.f64", std::ios::binary);
    { std::ofstream header(out+"/open_flux_header.txt");
      header << "step,t,m1,p1x,p1y,p1z,m2,p2x,p2y,p2z,m3,p3x,p3y,p3z,m4,p4x,p4y,p4z,Jx,Jy,Jz\n# (1) E->Omega (2) exterior return (3) Omega->E (4) exterior gather; Phi_in = m1+m2, Phi_out = m3+m4 (contract Section 4)\n"; }
    const std::vector<std::int64_t>& open_rindex{ rows.embedding().r_index_of_cell() };
    const auto is_e = [&](const std::uint32_t cell) { return open_layout.slot_of_cell()[cell]>=0; };
    const auto open_flux = [&](const std::uint32_t step, const std::vector<fpxx>& state, const VolumetricWallNativeCapture& cap) {
        const std::uint32_t q{ kQ };
        const std::uint32_t parity_before{ static_cast<std::uint32_t>((cap.t-1u)%2u) };
        std::array<double, kQ> stored{};
        const auto post = [&](const std::uint32_t st) -> double {   // the post-collision value of a donor state as the transport used it
            const std::uint32_t cell{ st/q }, d{ st%q };
            const std::int64_t r{ open_rindex[cell] };
            if(r>=0) return cap.post[static_cast<size_t>(r)*q+d];
            open_decoder.post(state, cell, parity_before, stored);
            return stored[d];
        };
        const auto& c{ open_decoder.c_lex() };
        std::array<native_protocol::Neumaier, 4> m{};
        std::array<std::array<native_protocol::Neumaier, 3>, 4> p{};
        const auto add = [&](const std::uint32_t k, const double amount, const std::uint32_t dir) { m[k].add(amount); for(std::uint32_t a=0u; a<3u; a++) p[k][a].add(amount*c[dir][a]); };
        const fluidx3d::experimental::NativeSparseMap& D{ fx.input.D };
        for(std::uint32_t rs=0u; rs<D.rows; rs++) {   // (1) and (3) direct
            const std::uint32_t rc{ rs/q }, rd{ rs%q };
            const double V{ fx.input.volume[rc] };
            for(std::uint32_t z=D.offsets[rs]; z<D.offsets[rs+1u]; z++) {
                const std::uint32_t dc{ D.indices[z]/q };
                if(is_e(dc) && !is_e(rc)) add(0u, V*D.weights[z]*post(D.indices[z]), rd);
                if(!is_e(dc) && is_e(rc)) add(2u, V*D.weights[z]*post(D.indices[z]), rd);
            }
            if(is_e(rc)) {   // (3) the arm increment and the interior returns into E receivers (owned states of R cells)
                const std::int64_t r{ open_rindex[rc] };
                const double dinc{ cap.direct_increment[static_cast<size_t>(r)*q+rd] };
                double back{ cap.returned[static_cast<size_t>(r)*q+rd] };
                for(std::uint32_t z=fx.input.R.offsets[rs]; z<fx.input.R.offsets[rs+1u]; z++) if(open_layout.exterior_row()[fx.input.R.indices[z]]!=0u) back -= fx.input.R.weights[z]*cap.Q[fx.input.R.indices[z]];
                add(2u, V*(dinc+back), rd);
            } else {   // (2) the exterior returns into Omega receivers
                for(std::uint32_t z=fx.input.R.offsets[rs]; z<fx.input.R.offsets[rs+1u]; z++) if(open_layout.exterior_row()[fx.input.R.indices[z]]!=0u) add(1u, V*fx.input.R.weights[z]*cap.Q[fx.input.R.indices[z]], rd);
            }
        }
        for(std::uint32_t r=0u; r<fx.input.rows; r++) {   // (1) interior gathers from E donors, (4) exterior gathers from Omega donors
            const bool exterior{ open_layout.exterior_row()[r]!=0u };
            double e_part{ 0.0 };
            std::uint32_t dir{ 0u };
            for(std::uint32_t z=fx.input.G.offsets[r]; z<fx.input.G.offsets[r+1u]; z++) {
                dir = fx.input.G.indices[z]%q;
                if(!is_e(fx.input.G.indices[z]/q)) continue;
                const double amount{ fx.input.G.weights[z]*post(fx.input.G.indices[z]) };
                if(exterior) e_part += amount; else add(0u, amount, dir);
            }
            if(exterior && fx.input.G.offsets[r+1u]>fx.input.G.offsets[r]) add(3u, cap.incoming[r]-cap.pending_old[r]-e_part, dir);
        }
        std::array<native_protocol::Neumaier, 3> J{};
        for(std::uint32_t a=0u; a<fx.input.facets; a++) for(std::uint32_t k=0u; k<3u; k++) J[k].add(cap.impulse[3u*a+k]);
        std::array<double, 21> v{};
        v[0] = static_cast<double>(step); v[1] = static_cast<double>(cap.t);
        for(std::uint32_t k=0u; k<4u; k++) { v[2u+4u*k] = m[k].value(); for(std::uint32_t a=0u; a<3u; a++) v[3u+4u*k+a] = p[k][a].value(); }
        for(std::uint32_t k=0u; k<3u; k++) v[18u+k] = J[k].value();
        open_flux_bin.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()*sizeof(double)));
    };
#endif // VOLUMETRIC_WALL_OPEN_X
    lbm->volumetric_wall_read_fi(fi_previous);
    lbm->volumetric_wall_read_pending(pending);
    rolling.update(fi_previous.data(), fi_previous.size()*sizeof(fpxx));
    hashes << "0 " << rolling.hex() << "\n";
    std::ofstream pending_hashes(out+"/pending_sha256.txt");   // task 21: per-step SHA-256 of the pending array at level s (the array step s+1 consumes)
    std::string last_pending_hex{ sha256_hex_of(pending) };
    pending_hashes << "0 " << last_pending_hex << "\n";
    std::vector<double> pending_previous{ pending };
    { SeriesRow row{ rows.row_zero(fi_previous, pending_previous, static_cast<std::uint64_t>(lbm->get_t())) }; write_row(row); }
#ifdef VOLUMETRIC_WALL_OPEN_X
    open_row(0u, fi_previous, pending_previous);
#endif // VOLUMETRIC_WALL_OPEN_X
    for(std::uint32_t s=1u; s<=steps; s++) {
#ifdef VOLUMETRIC_WALL_OPEN_X
        lbm->volumetric_wall_read_open_state(open_before); // #2421: E^(s-1), read before the update s (saved at a checkpoint of step s)
#endif // VOLUMETRIC_WALL_OPEN_X
        double t0{ PhaseTimers::now() };
        lbm->run(1u);
        double t1{ PhaseTimers::now() }; timers.gpu_step += t1-t0;
        lbm->volumetric_wall_read_fi(fi);
        double t2{ PhaseTimers::now() }; timers.readback_fi += t2-t1;
        const VolumetricWallNativeCapture cap{ lbm->volumetric_wall_capture() };
        double t3{ PhaseTimers::now() }; timers.capture += t3-t2;
        rolling.update(fi.data(), fi.size()*sizeof(fpxx));
        double t4{ PhaseTimers::now() }; timers.hash += t4-t3;
        bool failure{ false };
        std::string summary{};
        SeriesRow row{ rows.row_step(s, fi_previous, fi, cap, pending_previous, J_cum, requested_cum, delivered_cum, failure, summary) };
        double t5{ PhaseTimers::now() }; timers.rows += t5-t4;
        write_row(row);
#ifdef VOLUMETRIC_WALL_OPEN_X
        open_row(s, fi, cap.pending_new);
        if(lbm->volumetric_wall_open_failures()!=0u) { failure = true; summary += " | OPEN_FAILURE(cumulative count "+std::to_string(lbm->volumetric_wall_open_failures())+")"; }   // task 125 (scientist #3110): a refused E state stops the run at the first failing update (test control only)
#if defined(VOLUMETRIC_WALL_SRD) && defined(VOLUMETRIC_WALL_OPEN_X)
        if(env("VW_SRD_CAPTURE")=="1") { std::vector<fpxx> pre_wall{}; lbm->volumetric_wall_pre_wall_fi(pre_wall); open_flux(s, pre_wall, cap); }   // task 125: the donors' stored post BEFORE the E store (the final state aliases the new E populations in the shared slots)
        else open_flux(s, fi, cap);
#else
        open_flux(s, fi, cap);
#endif
#endif // VOLUMETRIC_WALL_OPEN_X
        hashes << s << " " << rolling.hex() << "\n";
        last_pending_hex = sha256_hex_of(cap.pending_new);
        pending_hashes << s << " " << last_pending_hex << "\n";
        timers.io += PhaseTimers::now()-t5;
        if(failure && !failed) { failed = true; first_failure = s; checkpoint(s, "first_failure", fi_previous, &cap); std::printf("FIRST FAILURE at step %u: %s\n", s, summary.c_str()); break; }
        if(std::find(checkpoints.begin(), checkpoints.end(), s)!=checkpoints.end()) checkpoint(s, "registered", fi_previous, &cap);
        if(s%100u==0u || s==steps) std::printf("protocol step %u: %s hash %s\n", s, summary.substr(0, summary.find(" | ")).c_str(), rolling.hex().substr(0, 16).c_str());
        fi_previous.swap(fi);
        pending_previous = cap.pending_new;
    }
    series_bin.close();
    hashes.close();
    pending_hashes.close();
    std::printf("%s\n", timers.text().c_str());
    char k_text[64]{}; std::snprintf(k_text, sizeof(k_text), "%.17g", fx.input.friction);
    { std::ofstream summary(out+"/summary.txt"); summary << "steps_requested " << steps << "\nfailed " << (failed ? 1 : 0) << "\nfirst_failure " << first_failure << "\nfinal_rolling_sha256 " << rolling.hex() << "\nu0_lattice " << u0 << "\nreference_axis " << (axis==1u ? "y" : "z") << "\ncolumns " << SeriesRow::names().size() << "\nrows " << (failed ? first_failure : steps)+1u << "\nzero_capacity_rows " << rows.zero_capacity_rows << "\nmode_norm " << rows.mode_norm() << "\nschema v5\nfriction_K " << k_text << "\nfinal_pending_sha256 " << last_pending_hex << "\ntraction_identity_ops " << rows.traction_ops() << "\n"; }
    if(warmup>0u) {   // task 45: appended only for a warm-up run
        std::ofstream summary(out+"/summary.txt", std::ios::app);
        summary << "warmup_steps " << warmup << "\n";
    }
#ifdef VOLUME_FORCE
    { // task 43: appended lines; a build without VOLUME_FORCE writes the archived summary format unchanged
        std::ofstream summary(out+"/summary.txt", std::ios::app);
        summary << "volume_force 1\nforce_x " << fx.force.text[0] << "\nforce_y " << fx.force.text[1] << "\nforce_z " << fx.force.text[2] << "\n";
    }
#endif
    { std::ofstream phases(out+"/phase_timers.txt"); phases << timers.text() << "\n"; }
    const bool ok{ PhysicsCheck::report("volumetric_wall_native_protocol_completed", failed ? 1.0 : 0.0, 0.5) };
    delete lbm;
    return ok ? 0 : 1;
}

namespace {
template<class T> std::vector<T> read_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary|std::ios::ate);
    if(!in) throw std::runtime_error("cannot read "+path);
    const std::streamsize size{ in.tellg() };
    in.seekg(0);
    std::vector<T> values(static_cast<size_t>(size)/sizeof(T));
    in.read(reinterpret_cast<char*>(values.data()), size);
    return values;
}
}

/// @brief Task 20 replay: recompute the series rows of every retained checkpoint of a batch folder from the retained raw data
/// (before_fi, fi, capture, pending) and the PREVIOUS recorded row's cumulative impulse, and compare them bitwise with the
/// recorded series. Reads the run folder only; writes replay_rows.f64 and replay_compare.txt to VW_OUT.
int run_replay(const Fixture& fx, const std::string& run, const std::string& out) {
    using native_protocol::SeriesRow;
    const std::uint32_t axis{ reference_axis(fx) };
    const double u0{ preflight_u0(fx, axis) };
    RowComputer rows(fx, u0, axis);
    const std::vector<std::string> names{ SeriesRow::names() };
    const std::vector<double> recorded{ read_raw<double>(run+"/series.f64") };
    if(recorded.size()%names.size()) throw std::runtime_error("recorded series width mismatch");
    const std::size_t n_rows{ recorded.size()/names.size() };
    const auto recorded_row = [&](const std::size_t r, const char* name) { for(std::size_t i=0u; i<names.size(); i++) if(names[i]==name) return recorded[r*names.size()+i]; throw std::runtime_error("name"); };
    std::ofstream rows_bin(out+"/replay_rows.f64", std::ios::binary);
    std::ofstream compare(out+"/replay_compare.txt");
    std::uint64_t compared{ 0u }, differing{ 0u }, rows_checked{ 0u };
    std::string first_difference{};
    const auto check = [&](const std::size_t r, SeriesRow& row) {
        rows_bin.write(reinterpret_cast<const char*>(row.v.data()), static_cast<std::streamsize>(row.v.size()*sizeof(double)));
        rows_checked++;
        for(std::size_t i=0u; i<names.size(); i++) {
            const double a{ row.v[i] }, b{ recorded[r*names.size()+i] };
            compared++;
            if(std::memcmp(&a, &b, sizeof(double))!=0) {
                differing++;
                char buffer[256]{};
                std::snprintf(buffer, sizeof(buffer), "row %zu column %s: replay %.17g recorded %.17g", r, names[i].c_str(), a, b);
                compare << buffer << "\n";
                if(first_difference.empty()) first_difference = buffer;
            }
        }
    };
    { // row 0 from checkpoint 0
        const std::vector<fpxx> fi0{ read_raw<fpxx>(run+"/checkpoint_000000_fi.f32") };
        const std::vector<double> pending0{ read_raw<double>(run+"/checkpoint_000000_pending.f64") };
        SeriesRow row{ rows.row_zero(fi0, pending0, 0u) };
        check(0u, row);
    }
    for(std::size_t r=1u; r<n_rows; r++) {
        char name[48]{};
        std::snprintf(name, sizeof(name), "/checkpoint_%06zu", r);
        const std::string prefix{ run+name };
        std::ifstream meta(prefix+"_meta.txt");
        if(!meta) continue;   // not a retained checkpoint
        std::vector<fpxx> before{ read_raw<fpxx>(prefix+"_before_fi.f32") }, fi{ read_raw<fpxx>(prefix+"_fi.f32") };
        VolumetricWallNativeCapture cap{};
        cap.available = true;
        cap.gpre = read_raw<float>(prefix+"_capture_gpre.f32"); cap.f = read_raw<double>(prefix+"_capture_f.f64"); cap.post = read_raw<double>(prefix+"_capture_post.f64");
        cap.fields = read_raw<double>(prefix+"_capture_fields.f64"); cap.incoming = read_raw<double>(prefix+"_capture_incoming.f64"); cap.Q = read_raw<double>(prefix+"_capture_Q.f64");
        cap.impulse = read_raw<double>(prefix+"_capture_impulse.f64"); cap.pending_old = read_raw<double>(prefix+"_capture_pending_old.f64"); cap.pending_new = read_raw<double>(prefix+"_capture_pending_new.f64");
        cap.prestore = read_raw<double>(prefix+"_capture_prestore.f64");
        { std::ifstream cm(prefix+"_capture_meta.txt"); std::string key{}; while(cm >> key) { if(key=="t") cm >> cap.t; else if(key=="clamp_count") cm >> cap.clamp_count; else { std::string rest{}; std::getline(cm, rest); } } }
        std::array<double, 3> J_cum{ recorded_row(r-1u, "Jx_cum"), recorded_row(r-1u, "Jy_cum"), recorded_row(r-1u, "Jz_cum") };   // the PREVIOUS recorded row, never the target
        std::array<double, 3> requested_cum{ recorded_row(r-1u, "requested_tangential_impulse_x_cum"), recorded_row(r-1u, "requested_tangential_impulse_y_cum"), recorded_row(r-1u, "requested_tangential_impulse_z_cum") };
        std::array<double, 3> delivered_cum{ recorded_row(r-1u, "delivered_tangential_impulse_x_cum"), recorded_row(r-1u, "delivered_tangential_impulse_y_cum"), recorded_row(r-1u, "delivered_tangential_impulse_z_cum") };
        bool failure{ false };
        std::string summary{};
        SeriesRow row{ rows.row_step(static_cast<std::uint32_t>(r), before, fi, cap, cap.pending_old, J_cum, requested_cum, delivered_cum, failure, summary) };
        check(r, row);
    }
    compare << "rows_checked " << rows_checked << "\nvalues_compared " << compared << "\nvalues_differing " << differing << "\nfirst_difference " << (first_difference.empty() ? "none" : first_difference) << "\n";
    std::printf("replay: rows %llu, values compared %llu, differing %llu, first difference: %s\n", static_cast<unsigned long long>(rows_checked), static_cast<unsigned long long>(compared), static_cast<unsigned long long>(differing), first_difference.empty() ? "none" : first_difference.c_str());
    return PhysicsCheck::report("volumetric_wall_native_replay_bitwise", static_cast<double>(differing), 0.5) ? 0 : 1;
}

int run_restart(const Fixture& fx, const std::string& path, const std::string& out, const std::uint32_t first, const std::uint32_t second) {
    preflight_k(fx);
    LBM* a = fx.make_lbm();
    attach_wall(*a, path);
    a->run(static_cast<ulong>(first));
    std::vector<fpxx> fi_snapshot{};
    std::vector<double> pending_snapshot{};
    a->volumetric_wall_read_fi(fi_snapshot);
    a->volumetric_wall_read_pending(pending_snapshot);
    const ulong t_snapshot{ a->get_t() };
    a->run(static_cast<ulong>(second));
    std::vector<fpxx> fi_end{};
    std::vector<double> pending_end{};
    a->volumetric_wall_read_fi(fi_end);
    a->volumetric_wall_read_pending(pending_end);
    delete a;
    LBM* b = fx.make_lbm();
    attach_wall(*b, path);
    b->run(0u);
    b->volumetric_wall_write_fi(fi_snapshot);
    b->volumetric_wall_write_pending(pending_snapshot);
    b->volumetric_wall_set_time_step(t_snapshot);
    b->run(static_cast<ulong>(second));
    std::vector<fpxx> fi_b{};
    std::vector<double> pending_b{};
    b->volumetric_wall_read_fi(fi_b);
    b->volumetric_wall_read_pending(pending_b);
    delete b;
    std::uint64_t fi_diff{ 0u }, pending_diff{ 0u };
    for(size_t i=0u; i<fi_end.size(); i++) if(std::memcmp(&fi_end[i], &fi_b[i], sizeof(fpxx))!=0) fi_diff++;
    for(size_t i=0u; i<pending_end.size(); i++) if(std::memcmp(&pending_end[i], &pending_b[i], sizeof(double))!=0) pending_diff++;
    std::printf("restart at t = %llu (+%u): fi differing slots %llu of %zu, pending differing rows %llu of %zu\n", static_cast<unsigned long long>(t_snapshot), second,
                static_cast<unsigned long long>(fi_diff), fi_end.size(), static_cast<unsigned long long>(pending_diff), pending_end.size());
    Dump::vector(out+"/restart_end_a_fi.f32", fi_end);
    Dump::vector(out+"/restart_end_b_fi.f32", fi_b);
    Dump::vector(out+"/restart_end_a_pending.f64", pending_end);   // task 21: the pending end states are retained for an independent comparison
    Dump::vector(out+"/restart_end_b_pending.f64", pending_b);
    std::printf("restart pending sha256: a %s b %s\n", sha256_hex_of(pending_end).c_str(), sha256_hex_of(pending_b).c_str());
    return PhysicsCheck::report("volumetric_wall_native_restart_bitwise", static_cast<double>(fi_diff+pending_diff), 0.5) ? 0 : 1;
}

/// @brief #2034 (RR27 builds): a synthetic invalid wall state must end run() with the engine's error. After attach and run(0),
/// every cell of the sampler stencil of facet 0 gets the stored value g = -1 in all its level-0 slots (density about 1 - q, so
/// the sampled density rho_s and the wall density rho_w are negative). run(1) must then stop the program (print_error, exit 1)
/// with "invalid wall states". Reaching the end of this function is the failure.
int run_poison(const Fixture& fx, const std::string& path) {
    LBM* lbm = fx.make_lbm();
    attach_wall(*lbm, path);
    lbm->run(0u);
    std::vector<fpxx> fi{};
    lbm->volumetric_wall_read_fi(fi);
    std::uint64_t written{ 0u };
    for(std::uint32_t z=fx.input.S.offsets[0]; z<fx.input.S.offsets[1]; z++) {
        for(const std::uint64_t a : fx.addresses.own_set(fx.lattice(fx.input.S.indices[z]), 0u)) {
            fi[static_cast<size_t>(a)] = -1.0f;
            written++;
        }
    }
    lbm->volumetric_wall_write_fi(fi);
    std::printf("poison: %llu stored values of the facet-0 sampler cells set to g = -1; run(1) must stop with invalid wall states\n", static_cast<unsigned long long>(written));
    std::fflush(stdout);
    lbm->run(1u);
    std::printf("PHYSICS volumetric_wall_native: the invalid wall state was NOT reported by run() FAILED\n");
    delete lbm;
    return 1;
}

#ifdef VOLUMETRIC_WALL_OPEN_X
/// @brief Task 125 control: the ACTUAL host admission predicate vw_open_refusal on the registered values (outlet s = +1, inlet s = -1; u_n = s ux): the closed endpoints +-eps, +-eps/2, 0, and wrong-sign 2 eps and 1e-6.
/// Writes open_admission.json (text per case); returns 0 when every expectation holds, else 1.
int run_open_admission(const std::string& out) {
    const double eps{ 1e-13 };
    struct Case { const char* name; int s; double un; bool admitted; };
    const std::vector<Case> cases{ { "outlet un=0", 1, 0.0, true }, { "outlet un=+eps/2", 1, eps/2, true }, { "outlet un=-eps/2", 1, -eps/2, true }, { "outlet un=-eps (closed endpoint)", 1, -eps, true },
        { "outlet un=-2 eps (wrong sign)", 1, -2.0*eps, false }, { "outlet un=-1e-6 (wrong sign)", 1, -1e-6, false }, { "outlet un=+1e-6", 1, 1e-6, true },
        { "inlet un=0", -1, 0.0, true }, { "inlet un=+eps/2", -1, eps/2, true }, { "inlet un=-eps/2", -1, -eps/2, true }, { "inlet un=+eps (closed endpoint)", -1, eps, true },
        { "inlet un=+2 eps (wrong sign)", -1, 2.0*eps, false }, { "inlet un=+1e-6 (wrong sign)", -1, 1e-6, false }, { "inlet un=-1e-6", -1, -1e-6, true } };
    std::ofstream json(out+"/open_admission.json");
    json << "{\n";
    bool all_ok{ true };
    for(std::size_t i=0u; i<cases.size(); i++) {
        const Case& k{ cases[i] };
        const std::string text{ vw_open_refusal(1.0, static_cast<double>(k.s)*k.un, 0.0, 0.0, k.s) };   // ux = s un, so that un = s ux
        const bool admitted{ text.empty() };
        const bool ok{ admitted==k.admitted };
        all_ok = all_ok && ok;
        json << " \"" << k.name << "\": {\"admitted\": " << (admitted ? "true" : "false") << ", \"expected\": " << (k.admitted ? "true" : "false") << ", \"text\": \"" << text << "\"}" << (i+1u<cases.size() ? "," : "") << "\n";
        std::printf("open_admission: %-40s admitted %d expected %d %s\n", k.name, admitted ? 1 : 0, k.admitted ? 1 : 0, ok ? "ok" : "MISMATCH");
    }
    json << "}\n";
    return PhysicsCheck::report("volumetric_wall_native_open_admission_controls", all_ok ? 0.0 : 100.0, 50.0) ? 0 : 1;
}
#endif // VOLUMETRIC_WALL_OPEN_X

#ifdef VOLUMETRIC_WALL_SRD
/// @brief Task 121 operator check: the state redistribution alone, on a given decoded level. VW_STATE is a raw float64 file of cells x 27 populations
/// (lexicographic, the level to redistribute); VW_PARITY (0 or 1) is the parity of the level-t1 load sets it is encoded into. The fixture attaches the
/// wall (which uploads the tables), encodes fl32_RTE(f - w) into the load sets of every mesh cell (all other words are zero), keeps the pending as the
/// container's s0 or VW_PENDING (raw float64, rows), applies the three stages once, and writes the raw fi before and after, the pending before and after,
/// the neighbourhood means, the reconstructed values and the six counters. Then it repeats the three stages VW_TIMING_REPEAT times (default 200) for the
/// kernel time per application (the repeated applications change the state; the written outputs are those of the first application only).
/// @param fx the fixture
/// @param path the container (with the t121 sections)
/// @param out the output folder
/// @return 0 when every written value is finite, else 1
int run_srd_operator(const Fixture& fx, const std::string& path, const std::string& out) {
    const std::string state_path{ env("VW_STATE") }, parity_text{ env("VW_PARITY") }, pending_path{ env("VW_PENDING") }, repeat_text{ env("VW_TIMING_REPEAT") };
    if(state_path.empty() || parity_text.empty()) {
        throw std::invalid_argument("VW_MODE=srd_operator needs VW_STATE (float64 cells x 27) and VW_PARITY (0 or 1)");
    }
    const std::uint32_t parity{ static_cast<std::uint32_t>(std::stoul(parity_text)) };
    if(parity>1u) {
        throw std::invalid_argument("VW_PARITY must be 0 or 1");
    }
    const std::uint32_t repeat{ repeat_text.empty() ? 200u : static_cast<std::uint32_t>(std::stoul(repeat_text)) };
    const std::vector<double> state{ read_raw<double>(state_path) };
    if(state.size()!=static_cast<size_t>(fx.input.cells)*kQ) {
        throw std::invalid_argument("VW_STATE has "+std::to_string(state.size())+" values, expected cells x 27 = "+std::to_string(static_cast<size_t>(fx.input.cells)*kQ));
    }
    preflight_k(fx);
    LBM* lbm = fx.make_lbm();
    attach_wall(*lbm, path);
    lbm->run(0u);
    const NativeDirections& dir{ directions_of_build() };
    std::vector<fpxx> fi(static_cast<size_t>(lbm->get_N())*kQ, 0.0f);
    for(std::uint32_t cell=0u; cell<fx.input.cells; cell++) {
        const auto own{ fx.addresses.own_set(fx.lattice(cell), parity) };
        for(std::uint32_t k=0u; k<kQ; k++) {
            const std::uint32_t d{ dir.fx_to_lex(k) };
            const double g{ state[static_cast<size_t>(cell)*kQ+d]-dir.weight_lex(d) };
            fi[static_cast<size_t>(own[k])] = static_cast<float>(g);
        }
    }
    lbm->volumetric_wall_write_fi(fi);
    if(!pending_path.empty()) {
        lbm->volumetric_wall_write_pending(read_raw<double>(pending_path));
    }
    std::vector<double> pending_before{};
    lbm->volumetric_wall_read_pending(pending_before);
    std::vector<fpxx> fi_before{};
    lbm->volumetric_wall_read_fi(fi_before);
    lbm->volumetric_wall_srd_apply(static_cast<ulong>(parity));
    std::vector<fpxx> fi_after{};
    std::vector<double> pending_after{}, qhat{}, recon{};
    std::vector<uint> stats{};
    lbm->volumetric_wall_read_fi(fi_after);
    lbm->volumetric_wall_read_pending(pending_after);
    lbm->volumetric_wall_srd_read(qhat, recon, stats);
    // word accounting: every changed raw word must lie in the load set of an active cell (the stages write nothing else)
    std::vector<std::uint8_t> active_word(fi_before.size(), 0u);
    for(const std::uint32_t cell : fx.input.srd.active_cells) {
        const auto own{ fx.addresses.own_set(fx.lattice(cell), parity) };
        for(std::uint32_t k=0u; k<kQ; k++) {
            active_word[static_cast<size_t>(own[k])] = 1u;
        }
    }
    std::uint64_t changed_in_active{ 0u }, changed_outside{ 0u };
    for(size_t a=0u; a<fi_before.size(); a++) {
        if(std::memcmp(&fi_before[a], &fi_after[a], sizeof(fpxx))!=0) {
            (active_word[a] ? changed_in_active : changed_outside)++;
        }
    }
    std::vector<fpxx> g_before(static_cast<size_t>(fx.input.cells)*kQ), g_after(static_cast<size_t>(fx.input.cells)*kQ);   // stored g per cell, lexicographic
    for(std::uint32_t cell=0u; cell<fx.input.cells; cell++) {
        const auto own{ fx.addresses.own_set(fx.lattice(cell), parity) };
        for(std::uint32_t k=0u; k<kQ; k++) {
            const size_t slot{ static_cast<size_t>(cell)*kQ+dir.fx_to_lex(k) };
            g_before[slot] = fi_before[static_cast<size_t>(own[k])];
            g_after[slot] = fi_after[static_cast<size_t>(own[k])];
        }
    }
    Dump::vector(out+"/operator_before_g.f32", g_before);
    Dump::vector(out+"/operator_after_g.f32", g_after);
    Dump::vector(out+"/operator_before_fi.f32", fi_before);
    Dump::vector(out+"/operator_after_fi.f32", fi_after);
    Dump::vector(out+"/operator_pending_before.f64", pending_before);
    Dump::vector(out+"/operator_pending_after.f64", pending_after);
    Dump::vector(out+"/operator_qhat.f64", qhat);
    Dump::vector(out+"/operator_recon.f64", recon);
    const auto first = std::chrono::steady_clock::now();
    for(std::uint32_t i=0u; i<repeat; i++) {
        lbm->volumetric_wall_srd_apply(static_cast<ulong>(parity));
    }
    const double seconds{ std::chrono::duration<double>(std::chrono::steady_clock::now()-first).count() };
    {
        std::ofstream summary(out+"/operator_summary.txt");
        summary << "counters_cumulative_since_attachment 1\nchanged_words_in_active_load_sets " << changed_in_active << "\nchanged_words_outside_active_load_sets " << changed_outside << "\nparity " << parity << "\ncells " << fx.input.cells << "\nstats_nonpositive_mean " << stats[0] << "\nstats_singular_B " << stats[1]
                << "\nstats_nonfinite_recon " << stats[2] << "\nstats_limiter_entries " << stats[3] << "\nstats_negative_after " << stats[4]
                << "\nstats_nonfinite_after " << stats[5] << "\ntiming_repeat " << repeat << "\ntiming_seconds_per_application " << (repeat>0u ? seconds/repeat : 0.0) << "\n";
    }
    std::printf("srd_operator: parity %u, counters %u %u %u %u %u %u, %.6f ms per application (%u repeats)\n", parity, stats[0], stats[1], stats[2], stats[3], stats[4], stats[5],
                repeat>0u ? 1000.0*seconds/repeat : 0.0, repeat);
    delete lbm;
    return PhysicsCheck::report("volumetric_wall_native_srd_operator_completed", static_cast<double>(Dump::non_finite_total), 0.5) ? 0 : 1;
}
#endif // VOLUMETRIC_WALL_SRD

/// @brief Task 98 (#1973): resume in a FRESH process from saved raw state files. The container identity is pinned: the
/// SHA-256 of the VW_INPUT bytes must equal VW_RESUME_CONTAINER_SHA256. The new LBM attaches the wall, runs run(0), writes the
/// saved raw fi (FluidX3D esoteric layout, level t) and pending (FP64, level t), sets the time step t and runs VW_M steps.
/// Writes resume_end_fi.f32, resume_end_pending.f64 and resume_summary.txt (t start/end, hashes). The comparison with an
/// uninterrupted run (its checkpoint of level t + M) is the caller's; this mode only produces the resumed end state.
int run_resume(const Fixture& fx, const std::string& path, const std::string& out, const std::uint32_t steps) {
    const std::string fi_path{ env("VW_RESUME_FI") }, pending_path{ env("VW_RESUME_PENDING") }, t_text{ env("VW_RESUME_T") }, pinned{ env("VW_RESUME_CONTAINER_SHA256") };
    if(fi_path.empty() || pending_path.empty() || t_text.empty() || pinned.empty()) {
        throw std::invalid_argument("VW_MODE=resume needs VW_RESUME_FI, VW_RESUME_PENDING, VW_RESUME_T and VW_RESUME_CONTAINER_SHA256");
    }
    std::string container_hex{};
    {
        std::ifstream in(path, std::ios::binary);
        if(!in) throw std::invalid_argument("resume: cannot read the container");
        const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        native_protocol::Sha256 h{};
        h.update(bytes.data(), bytes.size());
        container_hex = h.hex();
    }
    if(container_hex!=pinned) throw std::invalid_argument("resume: the container SHA-256 "+container_hex+" differs from the pinned "+pinned);
    const ulong t_start{ static_cast<ulong>(std::stoull(t_text)) };
    const std::vector<fpxx> fi_saved{ read_raw<fpxx>(fi_path) };
    const std::vector<double> pending_saved{ read_raw<double>(pending_path) };
    preflight_k(fx);
    LBM* lbm = fx.make_lbm();
    attach_wall(*lbm, path);
    lbm->run(0u);
    lbm->volumetric_wall_write_fi(fi_saved);
    lbm->volumetric_wall_write_pending(pending_saved);
    lbm->volumetric_wall_set_time_step(t_start);
    lbm->run(static_cast<ulong>(steps));
    std::vector<fpxx> fi_end{};
    std::vector<double> pending_end{};
    lbm->volumetric_wall_read_fi(fi_end);
    lbm->volumetric_wall_read_pending(pending_end);
    const ulong t_end{ lbm->get_t() };
    delete lbm;
    Dump::vector(out+"/resume_end_fi.f32", fi_end);
    Dump::vector(out+"/resume_end_pending.f64", pending_end);
    native_protocol::Sha256 hf{};
    hf.update(fi_end.data(), fi_end.size()*sizeof(fpxx));
    {
        std::ofstream summary(out+"/resume_summary.txt");
        summary << "container_sha256 " << container_hex << "\nt_start " << t_start << "\nsteps " << steps << "\nt_end " << t_end
                << "\nend_fi_sha256 " << hf.hex() << "\nend_pending_sha256 " << sha256_hex_of(pending_end) << "\n";
    }
    std::printf("resume: t %llu -> %llu, end fi sha256 %s, end pending sha256 %s\n", static_cast<unsigned long long>(t_start), static_cast<unsigned long long>(t_end),
                hf.hex().c_str(), sha256_hex_of(pending_end).c_str());
    return PhysicsCheck::report("volumetric_wall_native_resume_completed", static_cast<double>(Dump::non_finite_total), 0.5) ? 0 : 1;
}

}  // namespace

/// @brief The environment variable, "" if unset (the physics executables take no command line: main.cpp owns main()).
std::string env(const char* name) {
#ifdef _MSC_VER
    char* value = nullptr;
    size_t length = 0u;
    std::string result{};
    if(_dupenv_s(&value, &length, name)==0 && value!=nullptr) {
        result = value;
        free(value);
    }
    return result;
#else
    const char* value = std::getenv(name);
    return value!=nullptr ? std::string(value) : std::string();
#endif
}

void main_setup() { // VW_MODE attach|sentinel|step|restart, VW_INPUT container, VW_OUT folder, VW_N / VW_M step counts
    int code{ 2 };
    try {
        const std::string mode{ env("VW_MODE") }, path{ env("VW_INPUT") }, out{ env("VW_OUT") };
        if(mode.empty()||path.empty()||out.empty()) {
            std::printf("usage: VW_MODE=attach|sentinel|step|restart VW_INPUT=INPUT.vwnat VW_OUT=DIR [VW_N=n VW_M=m] physics_volumetric_wall_native\n");
        } else {
            const std::uint32_t n{ env("VW_N").empty() ? 3u : static_cast<std::uint32_t>(std::stoul(env("VW_N"))) };
            const std::uint32_t m{ env("VW_M").empty() ? 2u : static_cast<std::uint32_t>(std::stoul(env("VW_M"))) };
            const Fixture fx(path);
            { // task 98: the direction map of the raw fi dumps, for the decoder (FluidX3D slot, lexicographic index, c, double weight)
                std::ofstream table(out+"/directions_fx_to_lex.txt");
                if(!table) throw std::runtime_error("cannot write "+out+"/directions_fx_to_lex.txt");
                const NativeDirections& dir{ directions_of_build() };
                table << "# fx_slot lex_index cx cy cz weight_double_hex\n";
                char hex[64]{};
                for(std::uint32_t k=0u; k<kQ; k++) {
                    const std::uint32_t d{ dir.fx_to_lex(k) };
                    std::snprintf(hex, sizeof(hex), "%a", dir.weight_lex(d));
                    table << k << " " << d << " " << dir.fx_c(k, 0u) << " " << dir.fx_c(k, 1u) << " " << dir.fx_c(k, 2u) << " " << hex << "\n";
                }
            }
            { // task 98 (#1983): the embedding lists of this container, in every mode: embedding_R_cells.i64 (the R-cell order of every
              // capture array), embedding_R_rows.i64 and embedding_native_rows.i64 (state = cell*q + d), little-endian int64
                const VolumetricWallEmbedding lists(fx.input, fx.solid, forced_r_cells(fx));
                lists.write_lists(out+"/embedding");
            }
#ifdef VOLUMETRIC_WALL_OPEN_X
            if(mode=="restart" || mode=="resume" || mode=="sentinel" || mode=="poison" || mode=="sgs_refusal") { // task 105: these modes do not save or restore the E state (or do not attach with the declaration); refused instead of inexact
                throw std::invalid_argument("VW_MODE="+mode+" is not supported in a VOLUMETRIC_WALL_OPEN_X build (use attach, step or protocol)");
            }
#endif // VOLUMETRIC_WALL_OPEN_X
            if(mode=="attach") code = run_attach(fx, path, out);
            else if(mode=="attach_twice") code = run_attach_twice(fx, path);
            else if(mode=="sentinel") code = run_sentinel(fx, out);
            else if(mode=="step") code = run_steps(fx, path, out, n);
            else if(mode=="restart") code = run_restart(fx, path, out, n, m);
            else if(mode=="resume") code = run_resume(fx, path, out, m);   // task 98: fresh-process resume from saved raw files
            else if(mode=="poison") code = run_poison(fx, path);   // #2034: a synthetic invalid wall state must stop run() (RR27 builds)
            else if(mode=="sgs_refusal") code = run_sgs_refusal(fx, path);   // V2: one isolated refusal of set_volumetric_wall_sgs (SGS builds)
            else if(mode=="protocol") code = run_protocol(fx, path, out, n, env("VW_CHECKPOINTS"));
            else if(mode=="replay") code = run_replay(fx, env("VW_RUN"), out);
#ifdef VOLUMETRIC_WALL_OPEN_X
            else if(mode=="open_admission") code = run_open_admission(out);   // task 125: the actual host admission predicate on the registered control values
#endif // VOLUMETRIC_WALL_OPEN_X
#ifdef VOLUMETRIC_WALL_SRD
            else if(mode=="srd_operator") code = run_srd_operator(fx, path, out);   // task 121: the redistribution alone on a given decoded level
#endif // VOLUMETRIC_WALL_SRD
            else std::printf("unknown VW_MODE %s\n", mode.c_str());
        }
    } catch(const std::exception& error) {
        std::printf("PHYSICS volumetric_wall_native: exception %s FAILED\n", error.what());
        code = 1;
    }
    std::fflush(stdout);
    std::exit(code);
}
