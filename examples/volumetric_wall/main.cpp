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
void run_with_frames(LBM& lbm, const ulong steps, const std::string& frames_text, const std::string& out_folder) {
#ifdef GRAPHICS
	std::vector<ulong> frames{};
	size_t begin{ 0u };
	while(begin<frames_text.size()) {
		const size_t comma{ frames_text.find(',', begin) };
		const std::string item{ frames_text.substr(begin, comma==std::string::npos ? std::string::npos : comma-begin) };
		if(!item.empty()) frames.push_back(static_cast<ulong>(std::stoull(item)));
		begin = comma==std::string::npos ? frames_text.size() : comma+1u;
	}
	if(!frames.empty() && !out_folder.empty()) {
		lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_FIELD;
		lbm.graphics.field_mode = 0;                                 // the velocity magnitude
		lbm.graphics.slice_mode = 2;                                 // one y slice
		lbm.graphics.slice_y = static_cast<int>(lbm.get_Ny()/2u);
		lbm.graphics.set_camera_centered(-90.0f, 0.0f, 60.0f, 1.0f);  // fixed: looking along -y at the y slice (x right, z up)
		ulong done{ 0ull };
		for(const ulong frame : frames) {
			if(frame<done || frame>steps) throw std::invalid_argument("VOLUMETRIC_WALL_FRAMES must be ascending and <= the steps");
			lbm.run(frame-done);
			done = frame;
			char name[32]{};
			std::snprintf(name, sizeof(name), "frame_%06llu", static_cast<unsigned long long>(frame));
			lbm.graphics.write_frame(out_folder+"/", std::string(name), ".png");
		}
		lbm.run(steps-done);
		return;
	}
#else
	(void)frames_text; (void)out_folder;
#endif // GRAPHICS
	lbm.run(steps);
}

}  // namespace

void main_setup() { // the native volumetric slip wall; required extensions in defines.hpp: VOLUMETRIC_WALL, VOLUME_FORCE (UPDATE_FIELDS optional, see the file comment)
	const std::string container_path{ environment("VOLUMETRIC_WALL_CONTAINER") };
	const std::string steps_text{ environment("VOLUMETRIC_WALL_STEPS") };
	const std::string out_folder{ environment("VOLUMETRIC_WALL_OUT") };
	const std::string force_text{ environment("VOLUMETRIC_WALL_FORCE") };
	const std::string frames_text{ environment("VOLUMETRIC_WALL_FRAMES") };
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
		LBM lbm(box.size(0u), box.size(1u), box.size(2u), box.viscosity(), force[0], force[1], force[2]);
		box.set_flags_and_fields(lbm);
		box.attach(lbm);
		run_with_frames(lbm, steps, frames_text, out_folder);
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
	} catch(const std::exception& error) {
		fail(error.what());
	}
}
