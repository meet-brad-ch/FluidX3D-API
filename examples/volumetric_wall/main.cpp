/// @file main.cpp
/// @brief The ordinary setup of the opt-in native volumetric slip wall (extension VOLUMETRIC_WALL): a periodic box whose
/// fluid cells, cut volumes, wall facets, friction coefficient K and geometry-bound friction rows come from one VWNAT01
/// container. The setup marks the solid cells, attaches the wall with LBM::set_volumetric_wall() and runs. The flow is
/// driven by a uniform force density (VOLUME_FORCE), the only supported driver; without a force the flow only decays.
///
/// Targets (CMakeLists.txt): volumetric_wall and volumetric_wall_update_fields (the earlier D3Q19 wall);
/// volumetric_wall_rr27 (the D3Q27 revision VOLUMETRIC_WALL_RR27, research/slip_wall/README.md); volumetric_wall_noslip
/// (the baseline: the same container cells as plain voxels, FluidX3D's bounce-back and FP32 SRT collision, no wall: the
/// geometry AND the collision differ from the wall build); and the *_visual variants with GRAPHICS frames.
/// Scope: one domain on one GPU, FP32 populations, FP64 wall arithmetic, constant K >= 0, static periodic geometry,
/// laminar flow, a uniform constant force (zero allowed). Not supported (the build or the attachment refuses them):
/// SUBGRID, FP16, inlet/outlet and moving boundaries, FORCE_FIELD and the other forcing types, more than one domain.
/// The container's velocity set must be the build's (D3Q19 or D3Q27). The D3Q27 container is made offline by
/// scripts/slip_wall/terrain_container.py; GRAPHICS with the wall is a demonstration output, not a qualified diagnostic.
///
/// Input (this file reads the environment; the engine does not):
///   VOLUMETRIC_WALL_CONTAINER   path of the container (required)
///   VOLUMETRIC_WALL_STEPS       number of time steps (default 1000)
///   VOLUMETRIC_WALL_FORCE       fx,fy,fz: the uniform force density, lattice units (momentum per volume per step);
///                               default 0,0,0. The velocity the fields hold is the physical one, (j + F/2)/rho.
///   VOLUMETRIC_WALL_OUT         existing folder for the output after the run (optional): rho.f32, u.f32 (the fields),
///                               fi.f32 and pending.f64 (the complete raw state, wall builds only),
///                               meta.txt (steps, force bits, the time level of the fields)
///   VOLUMETRIC_WALL_FRAMES      GRAPHICS builds only: ascending steps for a frame (frame_<step>-<timestep>.png in the out folder),
///                               the same fixed view in every build (y slice through the middle, fixed camera and colour scale)
///   VOLUMETRIC_WALL_SGS_K       VOLUMETRIC_WALL_SGS builds only (V2): the static Smagorinsky constant k = 18 sqrt(2) (C Delta)^2
///                               of the RR collision (one number >= 0; the qualified value is 0x1.8746d2p-1 = (double)0.76421222f).
///                               Unset: k = 0, the V1 rate. Set in another build: refused. meta.txt records the stored k.
///   VOLUMETRIC_WALL_OPEN        VOLUMETRIC_WALL_OPEN_X builds only (task 105): "x" (required there; refused in every other build),
///                               with VOLUMETRIC_WALL_INLET_U, _INLET_MODE, _OUTLET_SIGMA, _OUTLET_MACH, _OUTLET_LENGTH,
///                               _OUTLET_RHO_INF, _E_INITIAL (see open_declaration()). The out folder then also gets open_state.f64
///                               (the E state per slot), open_layout.txt (E cells, exterior facets and rows) and the declaration
///                               in meta.txt; an E state that fails a refusal 7.6 ends the program with exit code 1.
///   VOLUMETRIC_WALL_VIEW        GRAPHICS and INTERACTIVE_GRAPHICS builds only (task 107): "slice" (default: the fixed y-slice
///                               view above) or "hill": the terrain (the solid-cell surface of the container's cells) with
///                               velocity-coloured streamlines, a fixed camera comparable to examples/hill, and the mode label
///                               drawn on each frame (frame_<step>.png). Set in a build without graphics: refused. A live
///                               (INTERACTIVE_GRAPHICS) build refuses VOLUMETRIC_WALL_FRAMES and shows the mode in its
///                               "LBM Type" line.
///
/// VOLUMETRIC_WALL_VIEW_CENTER_X  GRAPHICS and INTERACTIVE_GRAPHICS builds, view hill only (task 128): the x coordinate in cells (0 < x < Nx) that the hill
///                               camera is centred on; default Nx/2. The extended hill box (160 x 64 x 25) is centred on the hill with 40; the camera
///                               then frames the hill exactly as in the 80-wide box.
///
/// The actual-hill example (V2, task 107; extended downstream in task 128): the declared 80 x 64 terrain examples/volumetric_wall/hill_64_apron8_heights.csv
/// (Z_top 24), its container built with scripts/slip_wall/terrain_container.py --allow-steep --rough-z0, and two modes with
/// the same terrain cells, inflow speed, viscosity, camera and colour scale: the V2 wall (targets
/// volumetric_wall_rr27_sgs_rough_open_visual and _live) and the no-slip voxel baseline (volumetric_wall_noslip_open_visual
/// and _live: the container cells as voxels with FluidX3D's bounce-back, its FP32 SRT collision with SUBGRID, and
/// EQUILIBRIUM_BOUNDARIES cells at x = 0 and x = Nx - 1 that hold rho 1 and the inlet velocity). The baseline reads
/// VOLUMETRIC_WALL_OPEN = x and VOLUMETRIC_WALL_INLET_U only; it refuses the LODI inputs, which it would not use. The two modes
/// differ in geometry, wall, collision, subgrid closure and outlet rule, so their difference is not a wall-law-only
/// experiment. research/slip_wall/README.md has the build and launch commands.
///
/// Time level of the fields. The populations and the pending amounts after run(N) are level N. The target
/// `volumetric_wall` (no UPDATE_FIELDS) computes rho and u from them with LBM::update_fields(): level N, the physical
/// velocity (j + F/2)/rho. The target `volumetric_wall_update_fields` (UPDATE_FIELDS, the configuration of the research
/// fixture) writes the fields inside each step from the level the step starts with: after run(N) they are level N-1,
/// and update_fields() does nothing. Both give the same populations; meta.txt names the level of the fields.

#include "lbm.hpp"
#include "experimental/volumetric_wall_native.hpp" // the container reader: in the core library of every build (also the baseline without VOLUMETRIC_WALL)

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

/// @brief The velocity set of this build (the container must have the same q).
#ifdef D3Q27
constexpr std::uint32_t kQ{ 27u };
#else
constexpr std::uint32_t kQ{ 19u };
#endif

using fluidx3d::experimental::NativeAddresses;
using fluidx3d::experimental::NativeDirections;
using fluidx3d::experimental::VolumetricWallNativeInput;

/// @brief Means of the fluid after a run: volume-weighted density, mass-weighted velocity.
struct FluidMeans {
	double density{};                 ///< Mean density, lattice units.
	std::array<double, 3> velocity{}; ///< Mean velocity x, y, z, lattice units.
	double fluid_volume{};            ///< Sum of the cut-cell volumes, lattice cells.
};

/// @brief The environment variable, "" if it is not set.
/// @param name the variable's name
/// @return its value
std::string environment(const char* name) {
#ifdef _MSC_VER
	char* value{ nullptr };
	size_t length{ 0u };
	std::string result{};
	if(_dupenv_s(&value, &length, name)==0 && value!=nullptr) {
		result = value;
		free(value);
	}
	return result;
#else
	const char* value{ std::getenv(name) };
	return value!=nullptr ? std::string(value) : std::string();
#endif
}

/// @brief Reads three comma-separated numbers.
/// @param text "fx,fy,fz", or "" for zero
/// @return the three values
/// @throws std::invalid_argument when the text is not exactly three finite numbers
std::array<float, 3> parse_force(const std::string& text) {
	std::array<float, 3> force{};
	if(text.empty()) {
		return force;
	}
	size_t begin{ 0u };
	for(std::uint32_t axis=0u; axis<3u; axis++) {
		const size_t comma{ text.find(',', begin) };
		const bool last{ axis==2u };
		if((comma==std::string::npos)!=last) {
			throw std::invalid_argument("VOLUMETRIC_WALL_FORCE needs exactly three comma-separated numbers");
		}
		const std::string item{ text.substr(begin, last ? std::string::npos : comma-begin) };
		size_t used{ 0u };
		const float value{ std::stof(item, &used) };
		if(used!=item.size() || !std::isfinite(value)) {
			throw std::invalid_argument("VOLUMETRIC_WALL_FORCE value '"+item+"' is not a finite number");
		}
		force[axis] = value;
		begin = comma+1u;
	}
	return force;
}

/// @brief Reads the SGS closure constant k of VOLUMETRIC_WALL_SGS_K (V2).
/// @param text one number, or "" for no call (the engine keeps k = 0)
/// @return the value
/// @throws std::invalid_argument when the text is not exactly one finite number >= 0, or when it is set in a build
/// without VOLUMETRIC_WALL_SGS (the value would be ignored)
double parse_sgs_k(const std::string& text) {
#ifndef VOLUMETRIC_WALL_SGS
	if(!text.empty()) {
		throw std::invalid_argument("VOLUMETRIC_WALL_SGS_K is set, but this build has no VOLUMETRIC_WALL_SGS");
	}
	return 0.0;
#else
	if(text.empty()) {
		return 0.0;
	}
	size_t used{ 0u };
	const double value{ std::stod(text, &used) };
	if(used!=text.size() || !std::isfinite(value) || value<0.0) {
		throw std::invalid_argument("VOLUMETRIC_WALL_SGS_K '"+text+"' is not one finite number >= 0");
	}
	return value;
#endif // VOLUMETRIC_WALL_SGS
}

/// @brief The view of a graphics build (task 107).
enum class View {
	slice, ///< the fixed y-slice view of the velocity field (the default, unchanged)
	hill   ///< the terrain surface with streamlines, the camera comparable to examples/hill, and the mode label
};

/// @brief Reads VOLUMETRIC_WALL_VIEW (task 107).
/// @param text "slice", "hill", or "" for the default "slice"
/// @return the view
/// @throws std::invalid_argument when the text is another value, or when it is set in a build without graphics
View parse_view(const std::string& text) {
#ifndef GRAPHICS
	if(!text.empty()) {
		throw std::invalid_argument("VOLUMETRIC_WALL_VIEW is set, but this build has no GRAPHICS or INTERACTIVE_GRAPHICS");
	}
	return View::slice;
#else
	if(text.empty() || text=="slice") {
		return View::slice;
	}
	if(text=="hill") {
		return View::hill;
	}
	throw std::invalid_argument("VOLUMETRIC_WALL_VIEW '"+text+"' must be 'slice' or 'hill'");
#endif // GRAPHICS
}

/// @brief Reads VOLUMETRIC_WALL_VIEW_CENTER_X (task 128): the x coordinate, in cells, that the hill camera is centred on.
/// @param text a number with 0 < x < nx (cells), or "" for the box centre nx/2
/// @param nx the width of the box in cells (> 0)
/// @return the x coordinate of the view centre in cells
/// @throws std::invalid_argument when the text is not one number inside the box, or when it is set in a build without graphics
float parse_view_center_x(const std::string& text, const uint nx) {
#ifndef GRAPHICS
	if(!text.empty()) {
		throw std::invalid_argument("VOLUMETRIC_WALL_VIEW_CENTER_X is set, but this build has no GRAPHICS or INTERACTIVE_GRAPHICS");
	}
	(void)nx;
	return 0.0f;
#else
	if(text.empty()) {
		return 0.5f*static_cast<float>(nx);
	}
	std::size_t used{ 0u };
	double value{ 0.0 };
	try {
		value = std::stod(text, &used);
	} catch(const std::exception&) {
		used = 0u; // not a number: refused below
	}
	if(used!=text.size() || !(value>0.0 && value<static_cast<double>(nx))) {
		throw std::invalid_argument("VOLUMETRIC_WALL_VIEW_CENTER_X '"+text+"' must be one number with 0 < x < Nx (cells)");
	}
	return static_cast<float>(value);
#endif // GRAPHICS
}

/// @brief The full name of this build's mode (task 107): drawn on every frame of the hill view and written to meta.txt.
/// @return the label
const char* mode_label() {
#if defined(VOLUMETRIC_WALL_OPEN_X) && defined(VOLUMETRIC_WALL_T119) && defined(VOLUMETRIC_WALL_SRD)
	return "V2 VOLUMETRIC SLIP WALL: cut cells, rough wall law, static SGS, FP64 RR27, whole-prism correction + state redistribution, open x (outflow-only outlet)";
#elif defined(VOLUMETRIC_WALL_OPEN_X)
	return "V2 VOLUMETRIC SLIP WALL: cut cells, rough wall law, static SGS, FP64 RR27, open x (hard inlet, LODI outlet)";
#elif defined(VOLUMETRIC_WALL)
	return "VOLUMETRIC SLIP WALL: cut cells, FP64 RR27, periodic box";
#elif defined(EQUILIBRIUM_BOUNDARIES)
	return "NO-SLIP VOXEL BASELINE: bounce-back voxels, FP32 SRT + SUBGRID, fixed-equilibrium inlet and outlet";
#else
	return "NO-SLIP VOXEL BASELINE: bounce-back voxels, FP32 SRT, periodic box";
#endif
}

/// @brief The short name of this build's mode for the "LBM Type" line of the live window (at most 22 characters, task 107).
/// @return the label
const char* mode_short_label() {
#if defined(VOLUMETRIC_WALL_OPEN_X)
	return "RR27 V2 SLIP WALL";
#elif defined(VOLUMETRIC_WALL)
	return "RR27 SLIP WALL";
#elif defined(SUBGRID)
	return "SRT+SGS NO-SLIP VOXEL";
#else
	return "SRT NO-SLIP VOXEL";
#endif
}

#if defined(VOLUMETRIC_WALL_OPEN_X) || (!defined(VOLUMETRIC_WALL) && defined(EQUILIBRIUM_BOUNDARIES))
/// @brief Reads a comma-separated list of exactly `count` finite doubles (task 105).
/// @param text the list
/// @param count the number of values
/// @param name the variable name for the message
/// @return the values
/// @throws std::invalid_argument when the text is not exactly `count` finite numbers
std::vector<double> parse_doubles(const std::string& text, const std::uint32_t count, const std::string& name) {
	std::vector<double> values{};
	size_t begin{ 0u };
	for(std::uint32_t k=0u; k<count; k++) {
		const size_t comma{ text.find(',', begin) };
		const bool last{ k+1u==count };
		if((comma==std::string::npos)!=last) {
			throw std::invalid_argument(name+" needs exactly "+std::to_string(count)+" comma-separated numbers");
		}
		const std::string item{ text.substr(begin, last ? std::string::npos : comma-begin) };
		size_t used{ 0u };
		const double value{ std::stod(item, &used) };
		if(used!=item.size() || !std::isfinite(value)) {
			throw std::invalid_argument(name+" value '"+item+"' is not a finite number");
		}
		values.push_back(value);
		begin = comma+1u;
	}
	return values;
}
#endif // VOLUMETRIC_WALL_OPEN_X || the open-x baseline

#if !defined(VOLUMETRIC_WALL) && defined(EQUILIBRIUM_BOUNDARIES)
/// @brief The inlet velocity of the open-x no-slip baseline (task 107). Inputs: VOLUMETRIC_WALL_OPEN = "x" (required) and
/// VOLUMETRIC_WALL_INLET_U = ux,uy,uz (required; 0 < u_x and |u| < c = 1/sqrt(3)). VOLUMETRIC_WALL_INLET_MODE may be "hard";
/// every other open input of the wall build (relaxed inlet, outlet sigma, Mach, length, rho_inf, E_INITIAL) is refused,
/// because the baseline's fixed-equilibrium cells would not use it.
/// @return the inlet velocity
/// @throws std::invalid_argument when an input is missing, refused or out of range
std::array<float, 3> baseline_inlet_velocity() {
	if(environment("VOLUMETRIC_WALL_OPEN")!="x") {
		throw std::invalid_argument("this build is the open-x no-slip baseline: set VOLUMETRIC_WALL_OPEN=x and VOLUMETRIC_WALL_INLET_U");
	}
	const std::string mode{ environment("VOLUMETRIC_WALL_INLET_MODE") };
	if(!mode.empty() && mode!="hard") {
		throw std::invalid_argument("the no-slip baseline has only the fixed-equilibrium inlet: VOLUMETRIC_WALL_INLET_MODE must be unset or 'hard'");
	}
	const char* unused[]{ "VOLUMETRIC_WALL_OUTLET_SIGMA", "VOLUMETRIC_WALL_OUTLET_MACH", "VOLUMETRIC_WALL_OUTLET_LENGTH",
	                      "VOLUMETRIC_WALL_OUTLET_RHO_INF", "VOLUMETRIC_WALL_E_INITIAL" };
	for(const char* name : unused) {
		if(!environment(name).empty()) {
			throw std::invalid_argument(std::string(name)+" is set, but the no-slip baseline has no LODI outlet (its outlet cells hold rho 1 and the inlet velocity)");
		}
	}
	const std::vector<double> inlet{ parse_doubles(environment("VOLUMETRIC_WALL_INLET_U"), 3u, "VOLUMETRIC_WALL_INLET_U") };
	const double speed{ std::sqrt(inlet[0]*inlet[0]+inlet[1]*inlet[1]+inlet[2]*inlet[2]) };
	if(!(inlet[0]>0.0) || !(speed<1.0/std::sqrt(3.0))) {
		throw std::invalid_argument("VOLUMETRIC_WALL_INLET_U needs u_x > 0 and |u| < c = 1/sqrt(3)");
	}
	return { static_cast<float>(inlet[0]), static_cast<float>(inlet[1]), static_cast<float>(inlet[2]) };
}
#endif // the open-x baseline

#ifdef VOLUMETRIC_WALL_OPEN_X
using fluidx3d::experimental::VolumetricWallOpenLayout;
using fluidx3d::experimental::VolumetricWallOpenX;

/// @brief The open-x declaration from the environment (task 105, VOLUMETRIC_WALL_OPEN_X builds). Inputs:
///   VOLUMETRIC_WALL_OPEN            "x" (required in an open build; refused in a closed build)
///   VOLUMETRIC_WALL_INLET_U         ux,uy,uz: the inlet target (required; u_x > 0)
///   VOLUMETRIC_WALL_INLET_MODE      "hard" (default) or "relaxed,K_u"
///   VOLUMETRIC_WALL_OUTLET_SIGMA    sigma > 0 (required)
///   VOLUMETRIC_WALL_OUTLET_MACH     Ma of K_p (default: U_x c^-1 = U_x sqrt(3), computed in double)
///   VOLUMETRIC_WALL_OUTLET_LENGTH   L_x of K_p (default: Nx - 2, the cut length)
///   VOLUMETRIC_WALL_OUTLET_RHO_INF  rho_inf (default 1)
///   VOLUMETRIC_WALL_E_INITIAL       rho,ux,uy,uz of every E cell at the attachment (default: rho_inf and the inlet target)
/// meta.txt records the stored declaration (hexadecimal) and K_p.
/// @param nx the lattice size in x
/// @return the declaration (validated by set_volumetric_wall_open_x)
VolumetricWallOpenX open_declaration(const uint nx) {
	VolumetricWallOpenX open{};
	const std::vector<double> inlet{ parse_doubles(environment("VOLUMETRIC_WALL_INLET_U"), 3u, "VOLUMETRIC_WALL_INLET_U") };
	open.inlet_velocity = { inlet[0], inlet[1], inlet[2] };
	const std::string mode{ environment("VOLUMETRIC_WALL_INLET_MODE") };
	if(mode.empty() || mode=="hard") {
		open.hard_inlet = true;
	} else if(mode.rfind("relaxed,", 0u)==0u) {
		open.hard_inlet = false;
		open.inlet_relaxation = parse_doubles(mode.substr(8u), 1u, "VOLUMETRIC_WALL_INLET_MODE relaxed,K_u")[0];
	} else {
		throw std::invalid_argument("VOLUMETRIC_WALL_INLET_MODE must be 'hard' or 'relaxed,K_u'");
	}
	open.outlet_sigma = parse_doubles(environment("VOLUMETRIC_WALL_OUTLET_SIGMA"), 1u, "VOLUMETRIC_WALL_OUTLET_SIGMA")[0];
	const std::string mach{ environment("VOLUMETRIC_WALL_OUTLET_MACH") };
	open.outlet_mach = mach.empty() ? inlet[0]*std::sqrt(3.0) : parse_doubles(mach, 1u, "VOLUMETRIC_WALL_OUTLET_MACH")[0];
	const std::string length{ environment("VOLUMETRIC_WALL_OUTLET_LENGTH") };
	open.outlet_length = length.empty() ? static_cast<double>(nx)-2.0 : parse_doubles(length, 1u, "VOLUMETRIC_WALL_OUTLET_LENGTH")[0];
	const std::string rho_inf{ environment("VOLUMETRIC_WALL_OUTLET_RHO_INF") };
	open.outlet_rho_inf = rho_inf.empty() ? 1.0 : parse_doubles(rho_inf, 1u, "VOLUMETRIC_WALL_OUTLET_RHO_INF")[0];
	const std::string initial{ environment("VOLUMETRIC_WALL_E_INITIAL") };
	if(initial.empty()) {
		open.initial_rho = open.outlet_rho_inf;
		open.initial_velocity = open.inlet_velocity;
	} else {
		const std::vector<double> e{ parse_doubles(initial, 4u, "VOLUMETRIC_WALL_E_INITIAL") };
		open.initial_rho = e[0];
		open.initial_velocity = { e[1], e[2], e[3] };
	}
	return open;
}
#endif // VOLUMETRIC_WALL_OPEN_X

/// @brief One wall-bounded periodic box, read from a container.
class WallBox {
public:
	/// @param container_path path of the VWNAT01 container
	/// @throws std::runtime_error when the file is not a readable container
	/// @throws std::invalid_argument when a container identity fails
	explicit WallBox(const std::string& container_path)
		: path_(container_path), input_(VolumetricWallNativeInput::read(container_path)), addresses_(input_.box, NativeDirections(kQ)) {
		input_.validate();
		if(input_.velocity_set!=kQ) {
			throw std::invalid_argument("the container is D3Q"+std::to_string(input_.velocity_set)+", this build is D3Q"+std::to_string(kQ));
		}
	}

	/// @brief Lattice size along an axis.
	/// @param axis 0, 1 or 2
	/// @return the number of cells
	uint size(const std::uint32_t axis) const {
		return static_cast<uint>(input_.box[axis]);
	}

	/// @brief The container's kinematic viscosity, lattice units.
	float viscosity() const {
		return static_cast<float>(input_.viscosity);
	}

	/// @brief Linear lattice index of a mesh (fluid) cell.
	/// @param cell mesh cell index, 0 .. cells-1
	/// @return the index n = x + (y + z Ny) Nx
	ulong lattice_index(const std::uint32_t cell) const {
		const std::uint64_t n{ addresses_.cell(input_.cell_xyz[3u*cell], input_.cell_xyz[3u*cell+1u], input_.cell_xyz[3u*cell+2u]) };
		return static_cast<ulong>(n);
	}

	/// @brief Marks every cell without a mesh cell solid and gives the fluid cells the moments of the container's
	/// initial populations. The attachment then writes the exact initial populations and the pending amounts.
	/// @param lbm the lattice, of the container's box size, before run()
	void set_flags_and_fields(LBM& lbm) const {
		const NativeDirections directions(kQ);
		for(ulong n=0ull; n<lbm.get_N(); n++) {
			lbm.flags[n] = TYPE_S;
		}
		for(std::uint32_t cell=0u; cell<input_.cells; cell++) {
			double density{ 0.0 };
			std::array<double, 3> momentum{};
			for(std::uint32_t d=0u; d<kQ; d++) {
				const double population{ input_.f0[cell*kQ+d] };
				density += population;
				for(std::uint32_t axis=0u; axis<3u; axis++) {
					momentum[axis] += population*directions.lex_c(d, axis);
				}
			}
			const ulong n{ lattice_index(cell) };
			lbm.flags[n] = 0u;
			lbm.rho[n] = static_cast<float>(density);
			lbm.u.x[n] = static_cast<float>(momentum[0]/density);
			lbm.u.y[n] = static_cast<float>(momentum[1]/density);
			lbm.u.z[n] = static_cast<float>(momentum[2]/density);
		}
	}

#if !defined(VOLUMETRIC_WALL) && defined(EQUILIBRIUM_BOUNDARIES)
	/// @brief The open-x no-slip baseline (task 107): the container cells at x = 0 and x = Nx - 1 (the cells that are the E
	/// cells of the open wall build) become FluidX3D equilibrium-boundary cells (TYPE_E) that hold rho 1 and the inlet velocity.
	/// @param lbm the lattice after set_flags_and_fields()
	/// @param velocity the inlet velocity
	void set_open_ends(LBM& lbm, const std::array<float, 3>& velocity) const {
		const std::int32_t last_x{ input_.box[0]-1 };
		for(std::uint32_t cell=0u; cell<input_.cells; cell++) {
			const std::int32_t x{ input_.cell_xyz[3u*cell] };
			if(x!=0 && x!=last_x) {
				continue;
			}
			const ulong n{ lattice_index(cell) };
			lbm.flags[n] = TYPE_E;
			lbm.rho[n] = 1.0f;
			lbm.u.x[n] = velocity[0];
			lbm.u.y[n] = velocity[1];
			lbm.u.z[n] = velocity[2];
		}
	}
#endif // the open-x baseline

	/// @brief Attaches the wall on the supported path: the friction rows come from the container and are checked
	/// against its facets; a container without valid rows stops the program with the reason.
	/// @param lbm the lattice with its flags set
	void attach(LBM& lbm) const {
#ifdef VOLUMETRIC_WALL
		lbm.set_volumetric_wall(path_);
#else
		(void)lbm; // the baseline: the container cells as plain voxels with FluidX3D's bounce-back on the voxel faces, no wall operator
#endif // VOLUMETRIC_WALL
	}

	/// @brief Volume-weighted mean density and mass-weighted mean velocity of the lattice fields (read from the device first).
	/// @param lbm the lattice after a run, rho and u read from the device
	/// @return the means
	FluidMeans means(LBM& lbm) const {
		FluidMeans result{};
		double mass{ 0.0 };
		std::array<double, 3> momentum{};
		for(std::uint32_t cell=0u; cell<input_.cells; cell++) {
			const ulong n{ lattice_index(cell) };
			const double volume{ input_.volume[cell] };
			const double density{ static_cast<double>(lbm.rho[n]) };
			result.fluid_volume += volume;
			mass += volume*density;
			momentum[0] += volume*density*static_cast<double>(lbm.u.x[n]);
			momentum[1] += volume*density*static_cast<double>(lbm.u.y[n]);
			momentum[2] += volume*density*static_cast<double>(lbm.u.z[n]);
		}
		result.density = mass/result.fluid_volume;
		for(std::uint32_t axis=0u; axis<3u; axis++) {
			result.velocity[axis] = momentum[axis]/mass;
		}
		return result;
	}

private:
	std::string path_{};                ///< Path of the container.
	VolumetricWallNativeInput input_{}; ///< The container.
	NativeAddresses addresses_;         ///< Lattice index algebra of the container's box.
};

/// @brief Writes a float field as raw little-endian values.
/// @param path the file
/// @param values the values
/// @throws std::runtime_error when the file cannot be written
void write_raw(const std::string& path, const std::vector<float>& values) {
	std::ofstream out(path, std::ios::binary);
	out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()*sizeof(float)));
	if(!out) {
		throw std::runtime_error("cannot write "+path);
	}
}

/// @brief Ends the program with a plain one-line message and exit code 1. The engine's print_error() draws a text
/// box that fails on a long word such as a file path, so the setup's own failures do not use it.
/// @param message what failed and what to do
[[noreturn]] void fail(const std::string& message) {
	std::printf("volumetric wall setup FAILED: %s\n", message.c_str());
	std::fflush(stdout);
	std::exit(1);
}

/// @brief Writes a double array as raw little-endian values.
/// @param path the file
/// @param values the values
/// @throws std::runtime_error when the file cannot be written
void write_raw_double(const std::string& path, const std::vector<double>& values) {
	std::ofstream out(path, std::ios::binary);
	out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()*sizeof(double)));
	if(!out) {
		throw std::runtime_error("cannot write "+path);
	}
}

/// @brief Writes the complete raw state after a run: fi.f32 (the stored populations, engine layout, level t) and
/// pending.f64 (the pending amount of every wall row), and meta.txt.
/// @param lbm the lattice after run()
/// @param folder an existing folder
/// @param steps the number of steps run
/// @param force the uniform force density given to the LBM
void write_state(LBM& lbm, const std::string& folder, const ulong steps, const std::array<float, 3>& force) {
#ifdef VOLUMETRIC_WALL
	std::vector<fpxx> populations{};
	lbm.volumetric_wall_read_fi(populations);
	write_raw(folder+"/fi.f32", populations);
	std::vector<double> pending{};
	lbm.volumetric_wall_read_pending(pending);
	write_raw_double(folder+"/pending.f64", pending);
#endif // VOLUMETRIC_WALL (the baseline has no wall state; fields and meta only)
#ifdef UPDATE_FIELDS
	const ulong field_level{ steps>0ull ? steps-1ull : 0ull }; // written inside the last step from the level it started with
	const char* schedule{ "UPDATE_FIELDS (inside each step)" };
#else
	const ulong field_level{ steps };                          // LBM::update_fields() from the final populations
	const char* schedule{ "update_fields() after the run" };
#endif
	std::ofstream meta(folder+"/meta.txt");
	char line[256]{};
	std::snprintf(line, sizeof(line), "force_hex %a %a %a\n", static_cast<double>(force[0]), static_cast<double>(force[1]), static_cast<double>(force[2]));
	meta << "steps " << steps << "\nt " << lbm.get_t() << "\nstate_level " << lbm.get_t() << "\nfield_level " << field_level
	     << "\nfield_schedule " << schedule << "\n" << line;
#if defined(VOLUMETRIC_WALL_OPEN_X) || (!defined(VOLUMETRIC_WALL) && defined(EQUILIBRIUM_BOUNDARIES))
	meta << "mode " << mode_label() << "\n"; // task 107: the mode of the hill example builds (the closed V1 meta.txt is unchanged)
#endif
#ifdef VOLUMETRIC_WALL_SGS
	char sgs_line[96]{};
	std::snprintf(sgs_line, sizeof(sgs_line), "sgs_k_hex %a\n", lbm.get_volumetric_wall_sgs()); // V2: the value the engine stored (read back, not the input text)
	meta << sgs_line;
#endif // VOLUMETRIC_WALL_SGS
#ifdef VOLUMETRIC_WALL_OPEN_X
	{ // task 105: the stored declaration (read back), K_p, the refusal count, the E state and the classification
		const VolumetricWallOpenX open{ lbm.volumetric_wall_open_declaration() };
		char open_line[768]{};
		std::snprintf(open_line, sizeof(open_line),
			"open_x 1\nopen_inlet_velocity_hex %a %a %a\nopen_hard_inlet %d\nopen_inlet_relaxation_hex %a\nopen_outlet_sigma_hex %a\nopen_outlet_mach_hex %a\n"
			"open_outlet_length_hex %a\nopen_outlet_rho_inf_hex %a\nopen_initial_hex %a %a %a %a\nopen_K_p_hex %a\nopen_failures %u\n",
			open.inlet_velocity[0], open.inlet_velocity[1], open.inlet_velocity[2], open.hard_inlet ? 1 : 0, open.inlet_relaxation, open.outlet_sigma,
			open.outlet_mach, open.outlet_length, open.outlet_rho_inf, open.initial_rho, open.initial_velocity[0], open.initial_velocity[1],
			open.initial_velocity[2], open.outlet_relaxation(), lbm.volumetric_wall_open_failures());
		meta << open_line;
		std::vector<double> state{};
		lbm.volumetric_wall_read_open_state(state);
		write_raw_double(folder+"/open_state.f64", state);
		const VolumetricWallOpenLayout& layout{ lbm.volumetric_wall_open_layout() };
		std::ofstream masks(folder+"/open_layout.txt");
		masks << "# E cells (mesh index, slot, outward sign, b-1, b-2), then exterior facets, then exterior rows\n";
		for(size_t k=0u; k<layout.e_cells().size(); k++) {
			masks << "e " << layout.e_cells()[k] << ' ' << layout.e_slot()[k] << ' ' << layout.e_side()[k] << ' ' << layout.e_inward()[2u*k] << ' ' << layout.e_inward()[2u*k+1u] << '\n';
		}
		for(size_t a=0u; a<layout.exterior_facet().size(); a++) {
			if(layout.exterior_facet()[a]!=0u) masks << "facet " << a << ' ' << layout.e_cell_of_facet()[a] << '\n';
		}
		for(size_t r=0u; r<layout.exterior_row().size(); r++) {
			if(layout.exterior_row()[r]!=0u) masks << "row " << r << '\n';
		}
		if(!masks) {
			throw std::runtime_error("cannot write "+folder+"/open_layout.txt");
		}
	}
#endif // VOLUMETRIC_WALL_OPEN_X
	if(!meta) {
		throw std::runtime_error("cannot write "+folder+"/meta.txt");
	}
}

/// @brief Writes rho.f32 (N values) and u.f32 (x, then y, then z; 3 N values) into a folder.
/// @param lbm the lattice, rho and u read from the device
/// @param folder an existing folder
void write_fields(LBM& lbm, const std::string& folder) {
	const size_t cells{ static_cast<size_t>(lbm.get_N()) };
	std::vector<float> density(cells);
	std::vector<float> velocity(3u*cells);
	for(ulong n=0ull; n<lbm.get_N(); n++) {
		const size_t index{ static_cast<size_t>(n) };
		density[index] = lbm.rho[n];
		velocity[index] = lbm.u.x[n];
		velocity[cells+index] = lbm.u.y[n];
		velocity[2u*cells+index] = lbm.u.z[n];
	}
	write_raw(folder+"/rho.f32", density);
	write_raw(folder+"/u.f32", velocity);
}

/// @brief Runs the steps. With GRAPHICS and an output folder, writes a frame at each step of VOLUMETRIC_WALL_FRAMES (a
/// comma-separated list, for example 0,1000,5000,20000) with the same fixed view in every build: the velocity field on the
/// y slice through the middle of the box, the solid cells drawn, a fixed camera, and the fixed colour scale of the build
/// setting GRAPHICS_U_MAX. The frames are frame_<step>-<timestep>.png in the output folder. Without GRAPHICS: run(steps).
/// @param lbm the lattice with the wall attached (or the baseline)
/// @param steps the number of steps
/// @param frames_text the frame steps, or "" for none
/// @param out_folder the output folder, or "" for none
/// @param view the view of a graphics build (task 107)
void run_with_frames(LBM& lbm, const ulong steps, const std::string& frames_text, const std::string& out_folder, const View view, const float view_center_x);

#ifdef VOLUMETRIC_WALL_OPEN_X
/// @brief Task 105: stops the program when an E state failed a refusal 7.6 (no invalid run continues as if it were valid flow).
/// @param lbm the lattice after a run chunk
/// @param done the steps run so far
void check_open_states(LBM& lbm, const ulong done) {
	const uint failures{ lbm.volumetric_wall_open_failures() };
	if(failures!=0u) {
		fail("open x: "+std::to_string(failures)+" E state(s) failed a refusal 7.6 by step "+std::to_string(done)+" (backflow, wrong-sign inflow, |u_n| >= c, rho <= 0 or not finite); the run is invalid and stopped");
	}
}
#endif // VOLUMETRIC_WALL_OPEN_X

/// @brief Runs `steps` steps; an open build runs them in chunks of at most kOpenChunk and checks the E states after each.
/// @param lbm the lattice
/// @param steps the steps of this call
/// @param done the steps run before this call (for the message)
void run_checked(LBM& lbm, const ulong steps, const ulong done) {
#ifdef VOLUMETRIC_WALL_OPEN_X
	constexpr ulong kOpenChunk{ 100ull };
	ulong left{ steps };
	while(left>0ull) {
		const ulong chunk{ left<kOpenChunk ? left : kOpenChunk };
		lbm.run(chunk);
		left -= chunk;
		check_open_states(lbm, done+steps-left);
	}
	if(steps==0ull) lbm.run(0ull);
#else
	(void)done;
	lbm.run(steps);
#endif // VOLUMETRIC_WALL_OPEN_X
}

#ifdef GRAPHICS
/// @brief The hill view (task 107): the terrain as the surface of the solid cells (the cells without a container cell, so the
/// same surface in the wall build and in the baseline), velocity-coloured streamlines (colour scale GRAPHICS_U_MAX), no slice,
/// and a free camera that looks along the flow over the whole hill (see the body). This box has a solid lid, so the camera is
/// inside the box below the lid.
/// @param lbm the lattice
/// @param view_center_x the x coordinate in cells that the camera is centred on (task 128; Nx/2 gives the original camera)
void set_hill_view(LBM& lbm, const float view_center_x) {
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_STREAMLINES;
	lbm.graphics.field_mode = 0;   // the velocity magnitude colours the streamlines
	lbm.graphics.slice_mode = 0;   // no slice: 3D streamlines
	// The container's lid is solid cells at the top of the box, and the solid surface hides the terrain from an orbit above
	// the box. So the camera is a free camera BELOW the lid. Task 128 (presentation): the camera looks ALONG the flow (+x) from
	// the upstream end of the hill, at the middle of the box width, 0.6 box widths (Ny) upstream of the view centre, at 86 % of the
	// box height, looking down by kHillElevation with a wide field of view, so that the whole raised hill, the approach and the wake
	// are in the picture (the view centre is the x of the hill centre, VOLUMETRIC_WALL_VIEW_CENTER_X). Both modes use this function.
	constexpr float kHillAzimuth{ 180.0f };    // degrees: the camera looks along +x (the view direction is the azimuth + 180 degrees)
	constexpr float kHillElevation{ 24.0f };   // degrees below the horizontal
	constexpr float kHillFieldOfView{ 84.0f }; // degrees
	const float radius{ 0.6f*static_cast<float>(lbm.get_Ny()) };
	const float height{ 0.86f*static_cast<float>(lbm.get_Nz())-0.5f*static_cast<float>(lbm.get_Nz()) }; // from the box centre
	const float shift_x{ view_center_x-0.5f*static_cast<float>(lbm.get_Nx()) }; // task 128: the camera moves with the view centre along x (0 for the default Nx/2)
	lbm.graphics.set_camera_free(float3(shift_x-radius, 0.0f, height), kHillAzimuth, kHillElevation, kHillFieldOfView);
}
#endif // GRAPHICS

#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)

/// @brief Writes one frame of the hill view with the mode label drawn on it (task 107): folder/name.png. The engine's
/// write_frame() has no label, so this renders the frame, draws the labels into the rendered pixels with the engine's font
/// (draw_label() draws into camera.bitmap, which points at the frame while the labels are drawn), and writes the png.
/// GRAPHICS builds only (no live window that renders at the same time).
/// @param lbm the lattice
/// @param folder an existing folder
/// @param name the file name without the extension
void write_hill_frame(LBM& lbm, const std::string& folder, const std::string& name) {
	camera.key_update = true; // render a new frame
	int* frame{ lbm.graphics.draw_frame() };
	const int color{ invert(GRAPHICS_BACKGROUND_COLOR) };
	char step_text[160]{};
	std::snprintf(step_text, sizeof(step_text), "step %llu, lattice %u x %u x %u, nu %.6g, colour: |u| from 0 to %.3g (lattice units)",
		static_cast<unsigned long long>(lbm.get_t()), lbm.get_Nx(), lbm.get_Ny(), lbm.get_Nz(), static_cast<double>(lbm.get_nu()),
		static_cast<double>(GRAPHICS_U_MAX)/std::sqrt(3.0)); // GRAPHICS_U_MAX is in units of c = 1/sqrt(3)
	std::swap(camera.bitmap, frame);
	// task 128: a plain background strip behind the three label lines, so that the streamlines never cover the text
	for(uint y=0u; y<4u*FONT_HEIGHT && y<camera.height; y++) {
		for(uint x=0u; x<camera.width; x++) {
			camera.bitmap[camera.width*y+x] = GRAPHICS_BACKGROUND_COLOR;
		}
	}
	draw_label(4, 4, mode_label(), color);
	draw_label(4, 4+FONT_HEIGHT, std::string(step_text), color);
	draw_label(4, 4+2*FONT_HEIGHT, "terrain: the surface of the solid cells of the container (cell resolution; the same in both modes)", color);
	std::swap(camera.bitmap, frame);
	Image image(camera.width, camera.height);
	for(uint y=0u; y<camera.height; y++) {
		for(uint x=0u; x<camera.width; x++) {
			image.set_color(x, y, frame[camera.width*y+x]);
		}
	}
	write_png(folder+"/"+name+".png", &image);
}
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS

void run_with_frames(LBM& lbm, const ulong steps, const std::string& frames_text, const std::string& out_folder, const View view, const float view_center_x) {
#ifdef GRAPHICS
	if(view==View::hill) {
		set_hill_view(lbm, view_center_x);
	}
#endif // GRAPHICS
#ifdef INTERACTIVE_GRAPHICS
	// the live window (task 107): no frame files; P starts the run, the window shows the mode in its "LBM Type" line
	if(!frames_text.empty()) {
		throw std::invalid_argument("VOLUMETRIC_WALL_FRAMES is for GRAPHICS builds; a live INTERACTIVE_GRAPHICS build shows the run in its window");
	}
	(void)out_folder;
	run_checked(lbm, 0ull, 0ull);           // initializes the lattice; the window's labels exist from here
	info.collision = mode_short_label();    // replaces the engine's collision name ("SRT (FP32/FP32)") in the "LBM Type" line
#elif defined(GRAPHICS)
	std::vector<ulong> frames{};
	size_t begin{ 0u };
	while(begin<frames_text.size()) {
		const size_t comma{ frames_text.find(',', begin) };
		const std::string item{ frames_text.substr(begin, comma==std::string::npos ? std::string::npos : comma-begin) };
		if(!item.empty()) frames.push_back(static_cast<ulong>(std::stoull(item)));
		begin = comma==std::string::npos ? frames_text.size() : comma+1u;
	}
	if(!frames.empty() && !out_folder.empty()) {
		if(view==View::slice) {
			lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_FIELD;
			lbm.graphics.field_mode = 0;                                 // the velocity magnitude
			lbm.graphics.slice_mode = 2;                                 // one y slice
			lbm.graphics.slice_y = static_cast<int>(lbm.get_Ny()/2u);
			lbm.graphics.set_camera_centered(-90.0f, 0.0f, 60.0f, 1.0f);  // fixed: looking along -y at the y slice (x right, z up)
		}
		ulong done{ 0ull };
		for(const ulong frame : frames) {
			if(frame<done || frame>steps) throw std::invalid_argument("VOLUMETRIC_WALL_FRAMES must be ascending and <= the steps");
			run_checked(lbm, frame-done, done);
			done = frame;
			char name[32]{};
			std::snprintf(name, sizeof(name), "frame_%06llu", static_cast<unsigned long long>(frame));
			if(view==View::hill) {
				write_hill_frame(lbm, out_folder, std::string(name));
			} else {
				lbm.graphics.write_frame(out_folder+"/", std::string(name), ".png");
			}
		}
		run_checked(lbm, steps-done, done);
		return;
	}
#else
	(void)frames_text; (void)out_folder; (void)view; (void)view_center_x;
#endif // INTERACTIVE_GRAPHICS, GRAPHICS
	run_checked(lbm, steps, 0ull);
}

}  // namespace

void main_setup() { // the native volumetric slip wall; required extensions in defines.hpp: VOLUMETRIC_WALL, VOLUME_FORCE (UPDATE_FIELDS optional, see the file comment)
	const std::string container_path{ environment("VOLUMETRIC_WALL_CONTAINER") };
	const std::string steps_text{ environment("VOLUMETRIC_WALL_STEPS") };
	const std::string out_folder{ environment("VOLUMETRIC_WALL_OUT") };
	const std::string force_text{ environment("VOLUMETRIC_WALL_FORCE") };
	const std::string frames_text{ environment("VOLUMETRIC_WALL_FRAMES") };
	const std::string sgs_text{ environment("VOLUMETRIC_WALL_SGS_K") };
	if(container_path.empty()) {
		fail("set VOLUMETRIC_WALL_CONTAINER to the path of a VWNAT01 container with friction rows");
	}
	{ // a missing file is named here: the container reader reports it only as a failed read
		const std::ifstream probe(container_path, std::ios::binary);
		if(!probe) {
			fail("cannot open the container file "+container_path);
		}
	}
	try {
		const ulong steps{ steps_text.empty() ? 1000ull : static_cast<ulong>(std::stoull(steps_text)) };
		const WallBox box(container_path);
		const std::array<float, 3> force{ parse_force(force_text) };
		const double sgs_k{ parse_sgs_k(sgs_text) };
		const View view{ parse_view(environment("VOLUMETRIC_WALL_VIEW")) };
		const float view_center_x{ parse_view_center_x(environment("VOLUMETRIC_WALL_VIEW_CENTER_X"), box.size(0u)) };
		LBM lbm(box.size(0u), box.size(1u), box.size(2u), box.viscosity(), force[0], force[1], force[2]);
		box.set_flags_and_fields(lbm);
#ifdef VOLUMETRIC_WALL_SGS
		if(!sgs_text.empty()) {
			lbm.set_volumetric_wall_sgs(sgs_k); // V2: before the attachment, which gives k to the kernels
		}
#else
		(void)sgs_k;
#endif // VOLUMETRIC_WALL_SGS
		const std::string open_text{ environment("VOLUMETRIC_WALL_OPEN") };
#ifdef VOLUMETRIC_WALL_OPEN_X
		if(open_text!="x") {
			fail("this build is VOLUMETRIC_WALL_OPEN_X: set VOLUMETRIC_WALL_OPEN=x and the open declaration (VOLUMETRIC_WALL_INLET_U, VOLUMETRIC_WALL_OUTLET_SIGMA, ...)");
		}
		lbm.set_volumetric_wall_open_x(open_declaration(box.size(0u))); // task 105: before the attachment, which classifies the container with it
#elif !defined(VOLUMETRIC_WALL) && defined(EQUILIBRIUM_BOUNDARIES)
		(void)open_text;
		if(force[0]!=0.0f || force[1]!=0.0f || force[2]!=0.0f) {
			fail("the open-x no-slip baseline refuses a nonzero VOLUMETRIC_WALL_FORCE (its inlet and outlet cells drive the flow, as in the open wall build)");
		}
		box.set_open_ends(lbm, baseline_inlet_velocity()); // task 107: TYPE_E cells at x = 0 and x = Nx - 1
#else
		if(!open_text.empty()) {
			fail("VOLUMETRIC_WALL_OPEN is set, but this build has no VOLUMETRIC_WALL_OPEN_X (the open boundary is a compile-time option; use an *_open target)");
		}
#endif // VOLUMETRIC_WALL_OPEN_X
		box.attach(lbm);
		run_with_frames(lbm, steps, frames_text, out_folder, view, view_center_x);
		lbm.update_fields(); // rho and u of the final level from the populations; does nothing in an UPDATE_FIELDS build (the fields are then one step behind, see the file comment)
		lbm.rho.read_from_device();
		lbm.u.read_from_device();
		const FluidMeans means{ box.means(lbm) };
		char text[256]{};
		std::snprintf(text, sizeof(text), "volumetric wall: %llu steps, fluid volume %.6f, mean density %.9f, mean velocity (%.6e, %.6e, %.6e)",
			static_cast<unsigned long long>(steps), means.fluid_volume, means.density, means.velocity[0], means.velocity[1], means.velocity[2]);
		print_info(std::string(text));
		if(!out_folder.empty()) {
			write_fields(lbm, out_folder);
			write_state(lbm, out_folder, steps, force);
		}
#ifdef VOLUMETRIC_WALL_OPEN_X
		const uint open_failures{ lbm.volumetric_wall_open_failures() };
		if(open_failures!=0u) { // task 105: an E state failed a refusal 7.6 (recorded in meta.txt above when an output folder is given)
			fail("open x: "+std::to_string(open_failures)+" E state(s) failed a refusal 7.6 (backflow, wrong-sign inflow, |u_n| >= c, rho <= 0 or not finite); the run is invalid");
		}
#endif // VOLUMETRIC_WALL_OPEN_X
	} catch(const std::exception& error) {
		fail(error.what());
	}
}
