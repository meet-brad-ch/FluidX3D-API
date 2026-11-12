#pragma once

#include "defines.hpp"
#include "opencl.hpp"
#include "graphics.hpp"
#include "units.hpp"
#include "info.hpp"
#if defined(SCALAR_FV)&&(!defined(TEMPERATURE)||!defined(THERMAL_DIFFUSIVITY_FIX))
#error "SCALAR_FV carries T (and Q) by finite volumes with alpha as the diffusivity: it needs TEMPERATURE and THERMAL_DIFFUSIVITY_FIX."
#endif // SCALAR_FV

uint bytes_per_cell_host(); // returns the number of Bytes per cell allocated in host memory
uint bytes_per_cell_device(); // returns the number of Bytes per cell allocated in device memory
uint bandwidth_bytes_per_cell_device(); // returns the bandwidth in Bytes per cell per time step from/to device memory
uint3 resolution(const float3 box_aspect_ratio, const uint memory); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution

string default_filename(const string& path, const string& name, const string& extension, const ulong t); // generate a default filename with timestamp
string default_filename(const string& name, const string& extension, const ulong t); // generate a default filename with timestamp at exe_path/export/

/// @brief The halo layers on each side of a domain cut (this fork). The LBM's streaming reaches 1 cell. With SCALAR_FV
/// a cell's face values reach 2 cells along every link (kernel.cpp, fv_change: the value 1 cell behind and 2 ahead),
/// and both cells of a face must see the same 4 values, or its flux differs on its two sides and the scalars leak.
/// @return the halo width in cells: 2 with SCALAR_FV, else 1
inline uint halo_width() {
#ifdef SCALAR_FV
	return 2u;
#else // SCALAR_FV
	return 1u;
#endif // SCALAR_FV
}

#ifdef SCALAR_FV
/// @brief The bytes per boundary cell of the exchange of theta (and q) over the 2 halo layers (this fork).
/// @return halo_width() layers of one float per scalar
inline uint fv_transfer_bytes() {
#ifdef MOISTURE
	return halo_width()*2u*4u;
#else // MOISTURE
	return halo_width()*4u;
#endif // MOISTURE
}
#endif // SCALAR_FV

#pragma warning(disable:26812)
/// The fields the domains exchange. With SCALAR_FV: fv (theta, q) and fv_half (their half-step values) over 2 layers,
/// fvD (the diffusivity) over 1 layer, flags_fv (the flags) over 2 layers once at initialize() (this fork).
enum enum_transfer_field { fi, rho_u_flags, flags, F, phi_massex_flags, gi, T, hi, fv, fv_half, fvD, flags_fv, enum_transfer_field_length };


#ifdef VOLUMETRIC_WALL
// Task 19 (2026-11-08): opt-in native LBM-path volumetric free-slip wall (the CF10 bounded operator with the CF13
// facet state). Scope: one device, D3Q19 SRT, no SGS, FP32 shifted fi, FP64 boundary arithmetic. Task 43: the uniform
// VOLUME_FORCE (LBM constructor fx, fy, fz; set_f) is allowed and acts in the R cells as in the ordinary cells; every
// other forcing type stays rejected. Task 45: SUBGRID (the Smagorinsky closure of stream_collide) is allowed and is
// applied in the R cells with the same formula in FP64 (vw_collide).
// Task 98 (2026-11-10): VOLUMETRIC_WALL_RR27 (opt-in, off by default) is the qualified D3Q27 revision (task 90/94/96 reference,
// research/wall_model/integration/task95/TASK95_ACCEPTED_FORMULATION_SPEC_E2_2026-11-10.md): RR collision with the full force
// term Phi in EVERY cell in FP64, the wall-extrapolated even reference, the sampled friction, ONE mass correction per facet,
// the feq3-Taylor centred scatter and the unchanged pending transport; FP32 shifted population storage. Without the macro the
// D3Q19 path below is unchanged.
#ifdef VOLUMETRIC_WALL_RR27
#if !defined(D3Q27)||!defined(SRT)
#error VOLUMETRIC_WALL_RR27 requires D3Q27 (SRT is only the carrier macro: the RR collision replaces its collision)
#endif
#if !defined(VOLUMETRIC_WALL_FP64_COLLISION)
#error VOLUMETRIC_WALL_RR27 requires VOLUMETRIC_WALL_FP64_COLLISION (the ordinary cells collide in FP64)
#endif
#if defined(SUBGRID)||defined(VOLUMETRIC_WALL_SCATTER_INCREMENT)
#error VOLUMETRIC_WALL_RR27 rejects SUBGRID (not qualified with the RR collision) and VOLUMETRIC_WALL_SCATTER_INCREMENT (an experimental variant outside the frozen model)
#endif
#else // VOLUMETRIC_WALL_RR27
#if !defined(D3Q19)||!defined(SRT)
#error VOLUMETRIC_WALL requires D3Q19 and SRT (task 19 initial scope; D3Q27 only with VOLUMETRIC_WALL_RR27)
#endif
#endif // VOLUMETRIC_WALL_RR27
#if defined(FP16S)||defined(FP16C)
#error VOLUMETRIC_WALL requires FP32 population storage (FP16 storage in the wall region is not established)
#endif
#if defined(FORCE_FIELD)||defined(LEVEL_FORCING)||defined(TEMPERATURE)||defined(MOISTURE)||defined(SCALAR_FV)||defined(SURFACE)||defined(PARTICLES)||defined(MOVING_BOUNDARIES)||defined(EQUILIBRIUM_BOUNDARIES)
#error VOLUMETRIC_WALL rejects FORCE_FIELD, LEVEL_FORCING, TEMPERATURE/MOISTURE/SCALAR_FV, SURFACE, PARTICLES, MOVING/EQUILIBRIUM boundaries (only the uniform VOLUME_FORCE is supported as a driver, task 43)
#endif
#define TYPE_V TYPE_X // the cells of the wall region R (TYPE_X is free without MOISTURE)
#include "experimental/volumetric_wall_native.hpp"
#endif // VOLUMETRIC_WALL
// Task 57 (2026-11-09): VOLUMETRIC_WALL_FP64_COLLISION (opt-in, off by default) collides the ORDINARY cells of a
// VOLUMETRIC_WALL build with the FP64 formula of vw_collide (exact-in-double lattice weights, 53-bit density and
// velocity); the population storage stays FP32. Reason: the float32 weight constants do not satisfy the lattice sum
// identities (mass loss proportional to u^2) and rho * fl32(1/rho) is biased when rho is below 1 (excess momentum);
// research/wall_model/audits/20261109_collision_e2. Without the macro stream_collide is unchanged.
#if defined(VOLUMETRIC_WALL_FP64_COLLISION)&&!defined(VOLUMETRIC_WALL)
#error VOLUMETRIC_WALL_FP64_COLLISION requires VOLUMETRIC_WALL (D3Q19, SRT, FP32 storage, an FP64-capable device)
#endif
#if defined(VOLUMETRIC_WALL_RR27)&&!defined(VOLUMETRIC_WALL)
#error VOLUMETRIC_WALL_RR27 requires VOLUMETRIC_WALL
#endif
// V2 (2026-11-11): VOLUMETRIC_WALL_SGS (opt-in, off by default) gives the RR collision of vw_rr27 the per-cell rate of the
// static Smagorinsky closure (research/wall_model/v2/sgs_m1/e2/V2_SGS_01_M1_DERIVATION_E2_2026-11-10.md; qualified in the
// M1 reference: S/Q-L and D). The same rate acts in the R cells, the support cells and the ordinary cells. The closure
// constant k is set with LBM::set_volumetric_wall_sgs before the attachment; k = 0 (the default) is the rate omega exactly.
// SUBGRID stays refused with RR27. Without the macro the code is the V1 code.
#if defined(VOLUMETRIC_WALL_SGS)&&!defined(VOLUMETRIC_WALL_RR27)
#error VOLUMETRIC_WALL_SGS requires VOLUMETRIC_WALL_RR27
#endif
// V2 (2026-11-11): VOLUMETRIC_WALL_ROUGH (opt-in, off by default) gives the wall friction of vw_wall_rr27 the facetwise rough
// law K_a = K0_a + 3 C_a |u_s,a| (research/wall_model/v2/rough_wall/e2/V2_ROUGH_ENGINE_PORT_PLAN_E2_2026-11-11.md; qualified
// in the M1 reference: V2-RW-01). The law comes from the optional container section facet_friction (K0_a, C_a); a container
// without it gives (K, 0) on every facet, i.e. the scalar friction K exactly. A container WITH facet_friction is refused in a
// build without the macro. Without the macro the code is the V1/V2-SGS code.
#if defined(VOLUMETRIC_WALL_ROUGH)&&!defined(VOLUMETRIC_WALL_RR27)
#error VOLUMETRIC_WALL_ROUGH requires VOLUMETRIC_WALL_RR27
#endif
// Task 119 (2026-11-11, EXPERIMENTAL, isolated port; not part of the qualified V2): VOLUMETRIC_WALL_T119 (opt-in, off by default) adds the
// whole-prism redistribution of research/wall_model/MATHEMATICAL_MODEL.md section 15.11 to the RR27 wall: the centring mean of the scatter
// becomes the whole-prism mean (fluid and owned parts) and the pending amounts of the owned parts get their destination terms. It needs the
// optional container sections t119_gdelta and t119_wmoment (a container with them is refused in a build without the macro, and the macro
// build refuses a container without them). Closed x only: it is refused with VOLUMETRIC_WALL_OPEN_X. Without the macro the code is unchanged.
#if defined(VOLUMETRIC_WALL_T119)&&!defined(VOLUMETRIC_WALL_RR27)
#error VOLUMETRIC_WALL_T119 requires VOLUMETRIC_WALL_RR27
#endif
// Task 123: the former compile-time refusal of T119 with OPEN_X is replaced by a runtime refusal in set_volumetric_wall_affine (guarded compatibility: modified rows, G donors, gradient stencils and R destinations strictly interior).
// V2 (task 105): VOLUMETRIC_WALL_OPEN_X (opt-in, off by default) opens the x axis of the wall build: E cells at x = 0 (inlet)
// and x = Nx - 1 (outlet) hold feq3 of an FP64 E state that the isothermal characteristic (LODI) update advances; the
// facets of the two E columns are exterior (no balance: they absorb the gathers from the interior and return v feq3 of the
// adjacent E state). Contract: research/wall_model/v2/open_terrain/e2/V2_OPEN_DOMAIN_LEDGER_CONTRACT_E2_2026-11-10.md
// Sections 3-4, 6c, 7; CPU reference open_terrain_model.py. The E cells are the mesh cells at x = 0 and Nx - 1 (no TYPE_E flag,
// so EQUILIBRIUM_BOUNDARIES stays refused); the embedding makes every E cell an R cell, so vw_collide (and vw_support_rr27 for
// stencil reads) give them feq3 and stream_collide never sees them. LBM::set_volumetric_wall_open_x declares the boundary
// before the attachment.
// Without the macro the code is the closed V2 code.
#if defined(VOLUMETRIC_WALL_OPEN_X)&&!(defined(VOLUMETRIC_WALL_RR27)&&defined(VOLUMETRIC_WALL_FP64_COLLISION))
#error VOLUMETRIC_WALL_OPEN_X requires VOLUMETRIC_WALL_RR27 and VOLUMETRIC_WALL_FP64_COLLISION (the E branch of the ordinary cells is in the FP64 RR27 path)
#endif

class LBM_Domain {
private:
	uint Nx=1u, Ny=1u, Nz=1u; // (local) lattice dimensions
	uint Dx=1u, Dy=1u, Dz=1u; // lattice domains
	int Ox=0, Oy=0, Oz=0; // lattice domain offset
	ulong t = 0ull; // discrete time step in LBM units

	float nu = 1.0f/6.0f; // kinematic shear viscosity
	float fx=0.0f, fy=0.0f, fz=0.0f; // global force per volume
	float sigma=0.0f; // surface tension coefficient
	float alpha=1.0f, beta=1.0f, T_avg=1.0f; // alpha = thermal diffusion coefficient, beta = (volumetric) thermal expansion coefficient, T_avg = 1 = average temperature
	float heat_rate=0.0f; // TEMPERATURE: temperature added per time step to each fluid cell flagged TYPE_Y (a heat source, e.g. a heated floor)
	float inv_prandtl_t=0.0f; // TEMPERATURE with SUBGRID: 1/Pr_t, the eddy heat diffusivity is nu_t/Pr_t; 0 = no eddy heat diffusivity
	float moisture_rate=0.0f; // MOISTURE: Q added per time step to each fluid cell flagged TYPE_Y (the surface moisture flux, this fork)
	float theta_0=300.0f; // MOISTURE: the mixed layer's potential temperature (K) that T = 1 stands for, T = 1 + (theta_l - theta_0) / def_T_scale
	uint particles_N = 0u;
	float particles_rho = 1.0f;

	Device device; // OpenCL device associated with this LBM domain
#ifdef VOLUMETRIC_WALL
	// Task 19: the wall region R on the device. Rows, states and maps are restricted to R and re-indexed on the host
	// (experimental/volumetric_wall_native.cpp); the native fi stays the only persistent population storage.
	bool vw_enabled = false;
	uint vw_rcells=0u, vw_facets=0u, vw_rows=0u, vw_owned=0u, vw_pending_slot=0u;
	Memory<ulong> vw_cells, vw_addr0, vw_addr1; // lattice cell of every R cell; level-(t+1) address of every owned row at even/odd t
	Memory<int> vw_c, vw_fx2lex, vw_lex2fx, vw_mindex; // lexicographic c table (57), the direction permutations, mesh cell of every lattice cell (-1 solid)
	ulong vw_steps_completed = 0ull; // wall steps enqueued since attachment; the capture buffers are unavailable before the first
	Memory<uint> vw_fo, vw_rd, vw_odir, vw_so, vw_si, vw_go, vw_gi, vw_doff, vw_di, vw_ro, vw_ri, vw_wo, vw_wi; // facet offsets, row directions, owned directions, CSR offsets/indices of S, G, D_owned, R_owned, W
	Memory<uint> vw_clamp; // native clamp activation counter (allocated always; stream_collide increments it)
	Memory<double> vw_sw, vw_gw, vw_dw, vw_rw, vw_ww, vw_vi, vw_normal, vw_area; // CSR weights, row prism volumes, facet normals and areas
	Memory<double> vw_f, vw_post, vw_fields, vw_incoming, vw_Q, vw_impulse, vw_pending0, vw_pending1, vw_prestore, vw_f0; // step buffers, the two pending buffers, the supplied initial state of every mesh cell
	Memory<float> vw_gpre; // the actual stored g of the R cells read at the start of the step
	Memory<double> vw_phi; uint vw_use_phi=0u; // per-row friction vectors (3 per row) given to set_volumetric_wall (task 43; no environment variable); unused when 0
	Kernel vw_kernel_load, vw_kernel_collide, vw_kernel_wall0, vw_kernel_wall1, vw_kernel_stream01, vw_kernel_stream10, vw_kernel_seed; // wall0/stream01 read pending0 and write pending1; wall1/stream10 the reverse
	// Task 55: the affine transport candidate, used only when the container holds its six sections (vw_affine)
	bool vw_affine = false;
	uint vw_scount=0u, vw_gcount=0u; // support cells (ordinary cells in a gradient stencil), gradient cells
	Memory<ulong> vw_scells; // lattice cell of every support cell (ascending mesh order)
	Memory<uint> vw_goff, vw_gext, vw_gdon, vw_ocell; // gradient CSR offsets and extended stencil indices; gradient index of the donor of every G entry; R index of the receiver of every owned row
	Memory<int> vw_dgrad, vw_ograd; // gradient index of the donor of every owned D entry (-1: no arm term); of the receiver of every owned row (-1: no scatter)
	Memory<double> vw_gwt, vw_garm, vw_darm, vw_rarm, vw_rsw; // gradient weights; arms of G entries, of owned D entries, of owned R entries; h/V of owned R entries
	Memory<double> vw_spost, vw_sfields, vw_grad, vw_ebar, vw_returned, vw_dinc; // support post and fields; 69 gradient values per gradient cell; prism mean of e per row; capture: R Q + centred scatter and the direct arm term per R state
	Kernel vw_kernel_support, vw_kernel_gradient, vw_kernel_wall_affine0, vw_kernel_wall_affine1, vw_kernel_stream_affine01, vw_kernel_stream_affine10;
#ifdef VOLUMETRIC_WALL_RR27
	Memory<uint> vw_sgrad; // task 98: gradient index of the cell of every sampler (S) entry (the wall-extrapolated reference reads S[L rho], S[L u])
	Memory<double> vw_wstate, vw_delta; // task 98: rho_w, u_w (4 per facet) and the one-per-facet mass correction delta, written by vw_wall_affine (capture)
	Memory<uint> vw_wfail; // task 98: count of facets with a non-finite or non-positive rho_w (no clamp; any count invalidates the step)
#endif // VOLUMETRIC_WALL_RR27
#ifdef VOLUMETRIC_WALL_SGS
	double vw_sgs_k = 0.0; ///< V2: the SGS closure constant k of vw_rr27 (0: the rate omega); set before the attachment, given to the three kernels there
#endif // VOLUMETRIC_WALL_SGS
#ifdef VOLUMETRIC_WALL_ROUGH
	Memory<double> vw_facet_law; ///< V2: (K0_a, C_a) per facet for vw_wall_rr27, from the container section facet_friction or (K, 0) for a container without it
#endif // VOLUMETRIC_WALL_ROUGH
#ifdef VOLUMETRIC_WALL_T119
	Memory<double> vw_t119_delta, vw_t119_wcol, vw_t119_summ, vw_t119_wm, vw_t119_out; ///< task 119: delta_j per G entry (3); stored W column sum per row; sum_b m_b per row (3); owned first moment per W entry (3); B0, a (3), L/S per row (5, written by vw_wall_rr27)
	Memory<uint> vw_t119_neg, vw_t119_p1; ///< task 119: count of row updates with raw Q < 0, and of facet updates outside the mass/tangential feasibility screen 1e-13 (the two PR2 projection triggers; the engine has no projection)
#endif // VOLUMETRIC_WALL_T119
#ifdef VOLUMETRIC_WALL_SRD
	Memory<ulong> srd_lat; ///< task 121: lattice cell of every mesh cell
	Memory<uint> srd_estart, srd_ecell, srd_sstart, srd_scell, srd_acell, srd_astart, srd_aentry, srd_stats; ///< task 121: CSR offsets and cells of the group entries, the stencil, the active cells, and the six device counters
	Memory<int> srd_sidx; ///< task 121: seed index of every stencil cell or -1
	Memory<double> srd_ewv, srd_ew, srd_edelta, srd_vhat, srd_sw, srd_sd, srd_b, srd_swsum, srd_acoef, srd_qhat, srd_r; ///< task 121: tables (FP64 copies), neighbourhood means (S x q), reconstructed values (E x q)
	Kernel srd_kernel_means, srd_kernel_recon, srd_kernel_apply; ///< task 121: the three stages
	uint srd_seeds = 0u, srd_entries = 0u, srd_active = 0u; ///< task 121: table sizes
	bool srd_enabled = false; ///< task 121: the tables are attached
	Memory<fpxx> srd_pre_fi, vw_pre_wall_fi; bool srd_capture = false; ///< task 124 (diagnostic only): a copy of the stored level-(t+1) state taken immediately before the three SRD stages
#endif // VOLUMETRIC_WALL_SRD
#ifdef VOLUMETRIC_WALL_OPEN_X
	bool vw_open_declared = false; ///< task 105: set_volumetric_wall_open_x was called (required before the attachment)
	fluidx3d::experimental::VolumetricWallOpenX vw_open{}; ///< task 105: the declared boundary (immutable after the attachment)
	std::vector<std::uint8_t> vw_open_exterior_row; ///< task 105: exterior row flags of the attached container (host; the affine setup zeroes their scatter)
	uint vw_ecount = 0u; ///< task 105: number of E cells
	std::array<uint, 3> vw_open_e_paths{}; ///< task 105: E cells whose feq3 comes from vw_collide (R), vw_support_rr27 (support), stream_collide (ordinary)
	Memory<double> vw_estate, vw_estate_next; ///< task 105: the E state (rho, ux, uy, uz) per slot (2 Ny Nz slots), current and next level; allocated in allocate() (stream_collide reads vw_estate)
	Memory<ulong> vw_elattice, vw_einward; ///< task 105: lattice cell of every E cell; lattice cells b-1, b-2 of every E cell
	Memory<uint> vw_eslot; ///< task 105: slot of every E cell
	Memory<int> vw_eside, vw_eslot_r, vw_eslot_s, vw_efacet; ///< task 105: outward sign per E cell; slot of every R cell and support cell (-1 interior); R index of the E cell of every facet (-1 interior facet)
	Memory<uint> vw_open_fail; ///< task 105: E states that failed a 7.6 refusal (count; any count invalidates the run)
	Kernel vw_kernel_open_state; ///< task 105: the LODI update of the E state (level-n fields -> next state)
	Kernel vw_kernel_open_store; ///< task 105: the stored E populations, fl32_RTE(feq3(E) - w) in the load set of every E cell at a level
	std::unique_ptr<fluidx3d::experimental::VolumetricWallOpenLayout> vw_open_layout; ///< task 105: the classification of the attached container (set before set_volumetric_wall)
	void set_volumetric_wall_open_buffers(const fluidx3d::experimental::VolumetricWallNativeInput& input, const fluidx3d::experimental::VolumetricWallEmbedding& embedding); ///< task 105: E tables, initial E state, the LODI kernel (end of set_volumetric_wall)
#endif // VOLUMETRIC_WALL_OPEN_X
	void set_volumetric_wall_affine(const fluidx3d::experimental::VolumetricWallNativeInput& input, const fluidx3d::experimental::VolumetricWallEmbedding& embedding); // after the common buffers exist
#endif // VOLUMETRIC_WALL
	Kernel kernel_initialize; // initialization kernel
	Kernel kernel_stream_collide; // main LBM kernel
	Kernel kernel_update_fields; // reads DDFs and updates (rho, u, T) in device memory
	Memory<fpxx> fi; // LBM density distribution functions (DDFs); only exist in device memory
	ulong t_last_update_fields = max_ulong; // optimization to not call kernel_update_fields multiple times if (rho, u, T) are already up-to-date
#ifdef FORCE_FIELD
	Kernel kernel_update_force_field; // calculate forces from fluid on TYPE_S cells
	Kernel kernel_reset_force_field; // reset force field (also on TYPE_S cells)
	Kernel kernel_object_center_of_mass; // calculate center of mass of all cells flagged with flag_marker
	Kernel kernel_object_force; // add up force for all cells flagged with flag_marker
	Kernel kernel_object_torque; // add up torque around specified rotation_center for all cells flagged with flag_marker
	ulong t_last_force_field = max_ulong; // optimization to not call kernel_update_force_field multiple times if F is already up-to-date
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
	Kernel kernel_update_moving_boundaries; // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
#endif // MOVING_BOUNDARIES
#ifdef SURFACE
	Kernel kernel_surface_0; // additional kernel for computing mass conservation and mass flux computation
	Kernel kernel_surface_1; // additional kernel for flag handling
	Kernel kernel_surface_2; // additional kernel for flag handling
	Kernel kernel_surface_3; // additional kernel for flag handling and mass conservation
	Memory<float> mass; // fluid mass; phi=mass/rho
	Memory<float> massex; // excess mass; used for mass conservation
#endif // SURFACE
#ifdef TEMPERATURE
	Memory<fpxx> gi; // thermal DDFs
#endif // TEMPERATURE
#ifdef LEVEL_FORCING
	Memory<float> level; // per level z: fx, fy (a force per step), the damping rate of w, the reference temperature, the pressure (Pa) and the reference virtual theta (K) for MOISTURE, 2 spare (8 floats per level; this fork)
#endif // LEVEL_FORCING
#ifdef MOISTURE
	Memory<fpxx> hi; // moisture DDFs (the twin of gi, this fork)
#endif // MOISTURE
#ifdef SCALAR_FV
	Kernel kernel_fv_half; // the midpoint step's half stage after stream_collide, with the step's link mass fluxes (this fork)
	Kernel kernel_fv_full; // its full stage, with the sources: writes rho_FV, rho theta, theta and T (and Q) in place
	Kernel kernel_fv_init; // rho_FV and the shifted fields from the start fields, once at initialize()
	Memory<float> fvR; // rho_FV - 1, the scheme's own density (device only)
	Memory<float> fvT; // theta = T - 1 = (rho theta) / rho_FV: the per-step changes survive the float storage (device only)
	Memory<float> fvT_half; // theta at the half step (device only)
	Memory<float> fvRT; // rho theta, the conserved quantity (device only)
	Memory<float> fvD; // the diffusivity of every cell, alpha + nu_t/Pr_t, written by stream_collide (device only)
#ifdef MOISTURE
	Memory<float> fvQ; // q = Q - 1 (device only)
	Memory<float> fvQ_half; // q at the half step (device only)
	Memory<float> fvRQ; // rho q (device only)
#endif // MOISTURE
#endif // SCALAR_FV
#ifdef PARTICLES
	Kernel kernel_integrate_particles; // intgegrates particles forward in time and couples particles to fluid
#endif // PARTICLES

	void allocate(Device& device); // allocate all memory for data fields on host and device and set up kernels
	string device_defines(const Device_Info& device_info) const; // returns preprocessor constants for embedding in OpenCL C code

public:
	Memory<float> rho; // density of every cell
	Memory<float> u; // velocity of every cell
	Memory<uchar> flags; // flags of every cell
#ifdef FORCE_FIELD
	Memory<float> F; // individual force for every cell
	Memory<float> object_sum; // sum of individual cell data for an object
#endif // FORCE_FIELD
#ifdef SURFACE
	Memory<float> phi; // fill level of every cell
#endif // SURFACE
#ifdef TEMPERATURE
	Memory<float> T; // temperature of every cell
#endif // TEMPERATURE
#ifdef MOISTURE
	Memory<float> Q; // total water of every cell as 1 + qt / def_q_scale (this fork)
#endif // MOISTURE
#ifdef PARTICLES
	Memory<float> particles; // particle positions
#endif // PARTICLES

	Memory<char> transfer_buffer_p, transfer_buffer_m; // transfer buffers for multi-device domain communication, only allocate one set of transfer buffers in plus/minus directions, for all x/y/z transfers
	Kernel kernel_transfer[enum_transfer_field::enum_transfer_field_length][2]; // for each field one extract and one insert kernel
	void allocate_transfer(Device& device); // allocate all memory for multi-device transfer
	ulong get_area(const uint direction);
	void enqueue_transfer_extract_field(Kernel& kernel_transfer_extract_field, const uint direction, const uint bytes_per_cell);
	void enqueue_transfer_insert_field(Kernel& kernel_transfer_insert_field, const uint direction, const uint bytes_per_cell);

	LBM_Domain(const Device_Info& device_info, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const int Ox, const int Oy, const int Oz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho); // compiles OpenCL C code and allocates memory

#ifdef VOLUMETRIC_WALL
	/// @brief Upload the embedded wall (task 19). Requires an FP64-capable device; throws std::invalid_argument on a map that
	/// references a cell outside R or a facet with more than 19 rows.
	void set_volumetric_wall(const fluidx3d::experimental::VolumetricWallNativeInput& input, const fluidx3d::experimental::VolumetricWallEmbedding& embedding, const std::vector<double>& friction_rows); // friction_rows: rows*3 per-row friction vectors chosen by the caller (task 43), or empty = the original kernel friction expression (research only)
	bool volumetric_wall_enabled() const { return vw_enabled; }
	void enqueue_volumetric_wall_load(); // level-t states of the R cells into the FP64 scratch (before stream_collide)
	void enqueue_volumetric_wall_step(); // collide, wall, stream + pending transfer (after stream_collide), then the pending swap
	void enqueue_volumetric_wall_initial_state(); // the declared level-0 pre-collision state of every fluid cell into its t = 0 load addresses (after initialize): R cells from f0, ordinary cells from the native equilibrium
	fluidx3d::experimental::VolumetricWallNativeCapture read_volumetric_wall_buffers(); // blocking reads of every R buffer, both pendings, the clamp count
#ifdef VOLUMETRIC_WALL_RR27
#ifdef VOLUMETRIC_WALL_T119
	void volumetric_wall_t119_counts(uint& negative_q, uint& p1_failures); // task 119: the two PR2-trigger counts since the attachment (blocking)
#endif // VOLUMETRIC_WALL_T119
#ifdef VOLUMETRIC_WALL_SRD
	void set_volumetric_wall_srd(const fluidx3d::experimental::VolumetricWallNativeInput& input, const fluidx3d::experimental::VolumetricWallEmbedding& embedding); // task 121: tables and kernels (called by set_volumetric_wall)
	void enqueue_volumetric_wall_srd(const ulong t1); // task 121: the three stages on the complete level-t1 load sets (also used alone by the operator check)
	void volumetric_wall_srd_read(vector<double>& qhat, vector<double>& recon, vector<uint>& stats); // task 121: neighbourhood means, reconstructed values and the six counters (blocking)
	void set_volumetric_wall_srd_capture(const bool on) { srd_capture = on; } // task 124: diagnostic capture of the pre-SRD stored state (call before the attachment); changes no kernel, no buffer the run reads
	void volumetric_wall_srd_pre_fi(vector<fpxx>& host); // task 124: the captured pre-SRD stored state of the last update (blocking)
	void volumetric_wall_pre_wall_fi(vector<fpxx>& host); // task 125 (diagnostic only, open builds with the capture on): the stored state after the wall stream and BEFORE the E store of the last update (blocking)
#endif // VOLUMETRIC_WALL_SRD
	void volumetric_wall_failure_counts(uint& wall_state_failures, uint& clamps); // #2034: the invalid-wall-state and clamp counters since the attachment (blocking, two values)
#endif // VOLUMETRIC_WALL_RR27
#ifdef VOLUMETRIC_WALL_SGS
	/// @brief Store the SGS closure constant for the attachment (V2). The caller (LBM::set_volumetric_wall_sgs) checks it.
	/// @param k the closure constant k = 18 sqrt(2) (C Delta)^2, finite and >= 0
	void set_volumetric_wall_sgs_k(const double k) { vw_sgs_k = k; }
	double volumetric_wall_sgs_k() const { return vw_sgs_k; } ///< the stored closure constant
#endif // VOLUMETRIC_WALL_SGS
#ifdef VOLUMETRIC_WALL_OPEN_X
	/// @brief Store the open-x declaration for the attachment (task 105). The caller (LBM::set_volumetric_wall_open_x) checks it.
	void set_volumetric_wall_open_x(const fluidx3d::experimental::VolumetricWallOpenX& open) { vw_open = open; vw_open_declared = true; }
	bool volumetric_wall_open_declared() const { return vw_open_declared; } ///< true after the declaration
	fluidx3d::experimental::VolumetricWallOpenX volumetric_wall_open_declaration() const { return vw_open; } ///< the stored declaration
	std::array<uint, 3> volumetric_wall_open_e_paths() const { return vw_open_e_paths; } ///< E cells in R, support, ordinary
	/// @brief The layout of the attachment (task 105). @throws std::logic_error before the attachment
	const fluidx3d::experimental::VolumetricWallOpenLayout& volumetric_wall_open_layout() const { if(!vw_open_layout) throw std::logic_error("open x: no attachment"); return *vw_open_layout; }
	/// @brief Keep the classification of the container for the attachment (task 105; LBM::attach_volumetric_wall).
	void set_volumetric_wall_open_layout(const fluidx3d::experimental::VolumetricWallOpenLayout& layout) { vw_open_layout = std::make_unique<fluidx3d::experimental::VolumetricWallOpenLayout>(layout); }
	/// @brief The E state (rho, ux, uy, uz per slot, 2 Ny Nz slots; solid slots 0) of the current level (blocking).
	void vw_read_open_state(vector<double>& host);
	/// @brief The count of E states that failed a 7.6 refusal since the attachment (blocking).
	uint volumetric_wall_open_failures();
#endif // VOLUMETRIC_WALL_OPEN_X
	void vw_read_fi(vector<fpxx>& host); // raw stored populations (the parity t gives their meaning)
	void vw_write_fi(const vector<fpxx>& host);
	void vw_read_pending(vector<double>& host); // the current pending inventory
	void vw_write_pending(const vector<double>& host);
	void set_time_step(const ulong steps) { t = steps; } // restart: the parity of a restored fi
#endif // VOLUMETRIC_WALL
	void read_fi_raw(vector<fpxx>& host); // blocking copy of the raw stored populations (slot*N + cell; their meaning needs the parity t); host only, every build (task 19 P1 instrument)
	void enqueue_initialize(); // write all data fields to device and call kernel_initialize
	void enqueue_stream_collide(); // call kernel_stream_collide to perform one LBM time step
	void enqueue_update_fields(); // update fields (rho, u, T) manually
#ifdef SCALAR_FV
	void enqueue_fv_init() { kernel_fv_init.enqueue_run(); } // the shifted fields T - 1 (and Q - 1) from the fields on the device (SCALAR_FV)
	/// @brief The midpoint step's half stage (SCALAR_FV): after stream_collide and the domains' exchange of fi and fvD,
	/// so a cell next to a cut reads its neighbour domain's link fluxes and diffusivity.
	void enqueue_fv_half();
	/// @brief The midpoint step's full stage (SCALAR_FV): after the domains' exchange of the half-step fields.
	void enqueue_fv_full();
#endif // SCALAR_FV
#ifdef SURFACE
	void enqueue_surface_0();
	void enqueue_surface_1();
	void enqueue_surface_2();
	void enqueue_surface_3();
#endif // SURFACE
#ifdef FORCE_FIELD
	void enqueue_update_force_field(); // calculate forces from fluid on TYPE_S cells
	void enqueue_object_center_of_mass(const uchar flag_marker=TYPE_S); // calculate center of mass of all cells flagged with flag_marker
	void enqueue_object_force(const uchar flag_marker=TYPE_S); // add up force for all cells flagged with flag_marker
	void enqueue_object_torque(const float3& rotation_center, const uchar flag_marker=TYPE_S); // add up torque around specified rotation_center for all cells flagged with flag_marker
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
	void enqueue_update_moving_boundaries(); // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
#endif // MOVING_BOUNDARIES
#ifdef PARTICLES
	void enqueue_integrate_particles(const uint time_step_multiplicator=1u); // intgegrates particles forward in time and couples particles to fluid
#endif // PARTICLES

	void increment_time_step(const ulong steps=1ull); // increment time step
	void reset_time_step(); // reset time step
	void finish_queue();

	const Device& get_device() const { return device; }
	uint get_Nx() const { return Nx; } // get (local) lattice dimensions in x-direction
	uint get_Ny() const { return Ny; } // get (local) lattice dimensions in y-direction
	uint get_Nz() const { return Nz; } // get (local) lattice dimensions in z-direction
	ulong get_N() const { return (ulong)Nx*(ulong)Ny*(ulong)Nz; } // get (local) number of lattice points
	uint get_Dx() const { return Dx; } // get lattice domains in x-direction
	uint get_Dy() const { return Dy; } // get lattice domains in y-direction
	uint get_Dz() const { return Dz; } // get lattice domains in z-direction
	uint get_D() const { return Dx*Dy*Dz; } // get number of lattice domains
	float get_nu() const { return nu; } // get kinematic shear viscosity
	float get_tau() const { return 3.0f*get_nu()+0.5f; } // get LBM relaxation time
	float get_fx() const { return fx; } // get global froce per volume
	float get_fy() const { return fy; } // get global froce per volume
	float get_fz() const { return fz; } // get global froce per volume
	float get_sigma() const { return sigma; } // get surface tension coefficient
	float get_alpha() const { return alpha; } // get thermal diffusion coefficient
	float get_beta() const { return beta; } // get thermal expansion coefficient
	float get_heat_rate() const { return heat_rate; } // get temperature added per time step to TYPE_Y fluid cells
	float get_inv_prandtl_t() const { return inv_prandtl_t; } // get 1/Pr_t of the eddy heat diffusivity
	ulong get_t() const { return t; } // get discrete time step in LBM units
	uint get_velocity_set() const; // get LBM velocity set
	void set_heat_rate(const float heat_rate) { this->heat_rate = heat_rate; } // set temperature added per time step to TYPE_Y fluid cells
#ifdef SCALAR_FV
	/// Reads the finite-volume inventory from the device: the sums of rho_FV theta (the quantity the link-flux scheme conserves)
	/// and of rho_FV over the fluid cells of this domain, halo cells included (this fork; for the budget tests on one domain).
	void read_fv_inventory(double& sum_rho_theta, double& sum_rho_fv);
#endif // SCALAR_FV
#ifdef LEVEL_FORCING
	/// Writes the per-level terms (each vector holds Nz values): the force fx, fy per step, the damping rate r of w (a force -r rho w) and
	/// the temperature T_ref that T relaxes toward at the rate r (a source r (T_ref - T) per step, TEMPERATURE only).
	void set_level_forcing(const std::vector<float>& fx, const std::vector<float>& fy, const std::vector<float>& rate, const std::vector<float>& T_ref) {
		if(fx.size()!=(size_t)Nz||fy.size()!=(size_t)Nz||rate.size()!=(size_t)Nz||T_ref.size()!=(size_t)Nz) print_error("LBM::set_level_forcing(): every vector needs Nz = "+to_string(Nz)+" values.");
		for(uint z=0u; z<Nz; z++) { level[8ull*z] = fx[z]; level[8ull*z+1ull] = fy[z]; level[8ull*z+2ull] = rate[z]; level[8ull*z+3ull] = T_ref[z]; }
		level.write_to_device();
	}
	/// MOISTURE: the per-level pressure (Pa) and reference virtual potential temperature (K) the buoyancy uses (p = 0 on a level: dry buoyancy).
	void set_level_moist(const std::vector<float>& pressure, const std::vector<float>& theta_v_ref) {
		if(pressure.size()!=(size_t)Nz||theta_v_ref.size()!=(size_t)Nz) print_error("LBM::set_level_moist(): every vector needs Nz = "+to_string(Nz)+" values.");
		for(uint z=0u; z<Nz; z++) { level[8ull*z+4ull] = pressure[z]; level[8ull*z+5ull] = theta_v_ref[z]; }
		level.write_to_device();
	}
#endif // LEVEL_FORCING
	void set_moisture_rate(const float moisture_rate) { this->moisture_rate = moisture_rate; } // set Q added per time step to TYPE_Y fluid cells (MOISTURE)
	float get_moisture_rate() const { return moisture_rate; }
	void set_theta_0(const float theta_0) { this->theta_0 = theta_0; }
	void set_inv_prandtl_t(const float inv_prandtl_t) { this->inv_prandtl_t = inv_prandtl_t; } // set 1/Pr_t of the eddy heat diffusivity (needs SUBGRID)
	void set_fx(const float fx) { this->fx = fx; } // set global froce per volume
	void set_fy(const float fy) { this->fy = fy; } // set global froce per volume
	void set_fz(const float fz) { this->fz = fz; } // set global froce per volume
	void set_f(const float fx, const float fy, const float fz) { set_fx(fx); set_fy(fy); set_fz(fz); } // set global froce per volume

	void voxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S, const float3& rotation_center=float3(0.0f), const float3& linear_velocity=float3(0.0f), const float3& rotational_velocity=float3(0.0f)); // voxelize mesh
	void voxelize_sdf_on_device(const SDF* sdf, const float3& center, const float3x3& rotation, const float scale, const uchar flag); // voxelize SDF on GPU
	void enqueue_unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S); // remove voxelized triangle mesh from LBM grid

#ifdef GRAPHICS
	class Graphics {
	private:
		Kernel kernel_clear; // reset bitmap and zbuffer
		Memory<int> bitmap; // bitmap for rendering
		Memory<int> zbuffer; // z-buffer for rendering
		Memory<float> camera_parameters; // contains camera position, rotation, field of view etc.

		LBM_Domain* lbm = nullptr;
		Kernel kernel_graphics_flags; // render flag lattice with wireframe
		Kernel kernel_graphics_flags_mc; // render flag lattice with marching-cubes
		Kernel kernel_graphics_field; // render a colored velocity vector for each cell
		Kernel kernel_graphics_field_slice; // render one slice of velocity field according to slics settings
		Kernel kernel_graphics_streamline; // render streamlines
		Kernel kernel_graphics_q; // render vorticity (Q-criterion)

#ifdef SURFACE
		const string path_skybox = get_resource_path("skybox8k.png");
		Image* skybox_image = nullptr;
		Memory<int> skybox; // skybox for free surface raytracing
		Kernel kernel_graphics_rasterize_phi; // rasterize free surface
		Kernel kernel_graphics_raytrace_phi; // raytrace free surface
		Image* get_skybox_image() const { return skybox_image; }
#endif // SURFACE

#ifdef PARTICLES
		Kernel kernel_graphics_particles;
#endif // PARTICLES

		ulong t_last_rendered_frame = max_ulong; // optimization to not call draw_frame() multiple times if camera_parameters and LBM time step are unchanged
		bool update_camera(); // update camera_parameters and return if they are changed from their previous state

	public:
		Graphics() {} // default constructor
		Graphics(LBM_Domain* lbm) {
			this->lbm = lbm;
#ifdef SURFACE
			skybox_image = read_png(path_skybox);
#endif // SURFACE
		}
		Graphics& operator=(const Graphics& graphics) { // copy assignment
			lbm = graphics.lbm;
#ifdef SURFACE
			skybox_image = graphics.get_skybox_image();
#endif // SURFACE
			return *this;
		}
		void allocate(Device& device); // allocate memory for bitmap and zbuffer
		bool enqueue_draw_frame(const int visualization_modes, const int field_mode=0, const int slice_mode=0, const int slice_x=0, const int slice_y=0, const int slice_z=0, const bool visualization_change=true); // main rendering function, calls rendering kernels, returns true if new frame is rendered, false if old frame is returned when camera has not moved
		int* get_bitmap(); // returns pointer to bitmap
		int* get_zbuffer(); // returns pointer to zbuffer
		string device_defines(const Device_Info& device_info) const; // returns preprocessor constants for embedding in OpenCL C code
	}; // Graphics
	Graphics graphics;
#endif // GRAPHICS
}; // LBM_Domain



class LBM {
private:
	uint Nx=1u, Ny=1u, Nz=1u; // (global) lattice dimensions
	uint Dx=1u, Dy=1u, Dz=1u; // lattice domains
	bool initialized = false; // becomes true after LBM::initialize() has been called

	void sanity_checks_constructor(const vector<Device_Info>& device_infos, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho); // sanity checks on grid resolution and extension support
	void sanity_checks_initialization(); // sanity checks during initialization on used extensions based on used flags
	void initialize(); // write all data fields to device and call kernel_initialize
	void do_time_step(); // call kernel_stream_collide to perform one LBM time step

	void communicate_field(const enum_transfer_field field, const uint bytes_per_cell);

	void communicate_fi();
	void communicate_rho_u_flags();
	void communicate_flags();
#ifdef VOLUMETRIC_WALL
	void attach_volumetric_wall(const fluidx3d::experimental::VolumetricWallNativeInput& input, const std::vector<double>& friction_rows); // common part of set_volumetric_wall and set_volumetric_wall_research (task 43); throws
#endif // VOLUMETRIC_WALL
#ifdef FORCE_FIELD
	void communicate_F();
#endif // FORCE_FIELD
#ifdef SURFACE
	void communicate_phi_massex_flags();
#endif // SURFACE
#ifdef TEMPERATURE
	void communicate_gi();
	void communicate_T();
#ifdef MOISTURE
	void communicate_hi();
#endif // MOISTURE
#endif // TEMPERATURE
#ifdef SCALAR_FV
	void communicate_fv(); // theta and q over the 2 halo layers (after the full stage and after fv_init)
	void communicate_fv_half(); // their half-step values over the 2 halo layers (after the half stage)
	void communicate_fvD(); // the diffusivity over the halo layer next to the cut (after stream_collide)
	void communicate_flags_fv(); // the flags over the 2 halo layers (once at initialize(): fv_solid() reads 2 cells along a link)
#endif // SCALAR_FV
#ifdef PARTICLES
	void communicate_particles();
#endif // PARTICLES

public:
	template<typename T> class Memory_Container { // does not hold any data itsef, just links to LBM_Domain data
	private:
		ulong N = 0ull; // buffer length
		uint d = 1u; // buffer dimensions
		LBM* lbm = nullptr;
		Memory<T>** buffers = nullptr; // host buffers
		string name = "";

		uint Nx=1u, Ny=1u, Nz=1u, Dx=1u, Dy=1u, Dz=1u, D=1u; // auxiliary variables: (local) lattice dimensions, lattice domains, number of domains
		uint NxDx=1u, NyDy=1u, NzDz=1u, Hx=0u, Hy=0u, Hz=0u; // auxiliary variables: number of domains, shortcuts for N_/D_, halo offsets
		ulong NxNy=1ull, local_Nx=1ull, local_Ny=1ull, local_Nz=1ull, local_N=1ull; // auxiliary variables: shortcut for Nx*Ny, size of each domain, number of cells in each domain
		inline void initialize_auxiliary_variables() { // these variables are frequently used in reference() functions, so pre-compute them only once here
			Nx = lbm->get_Nx(); Ny = lbm->get_Ny(); Nz = lbm->get_Nz();
			Dx = lbm->get_Dx(); Dy = lbm->get_Dy(); Dz = lbm->get_Dz();
			D = Dx*Dy*Dz; // number of domains
			NxNy = (ulong)Nx*(ulong)Ny; // shortcut for Nx*Ny
			NxDx=Nx/Dx; NyDy=Ny/Dy; NzDz=Nz/Dz; // shortcuts for N_/D_
			Hx=(Dx>1u)*halo_width(); Hy=(Dy>1u)*halo_width(); Hz=(Dz>1u)*halo_width(); // halo offsets
			local_Nx=(ulong)(NxDx+2u*Hx); local_Ny=(ulong)(NyDy+2u*Hy); local_Nz=(ulong)(NzDz+2u*Hz); // size of each domain
			local_N = local_Nx*local_Ny*local_Nz; // number of cells in each domain
		}
		inline void initialize_auxiliary_pointers() {
			/********/ x = Pointer(this, 0x0u);
			if(d>0x1u) y = Pointer(this, 0x1u);
			if(d>0x2u) z = Pointer(this, 0x2u);
		}
		inline T& reference(const ulong i) { // stitch together domain buffers and make them appear as one single large buffer
			if(D==1u) { // take shortcut for single domain
				return buffers[0]->data()[i]; // array of structures
			} else { // decompose index for multiple domains
				const ulong global_i=i%N, t=global_i%NxNy;
				const uint x=(uint)(t%(ulong)Nx), y=(uint)(t/(ulong)Nx), z=(uint)(global_i/NxNy); // n = x+(y+z*Ny)*Nx
				const uint px=x%NxDx, py=y%NyDy, pz=z%NzDz, dx=x/NxDx, dy=y/NyDy, dz=z/NzDz, domain=dx+(dy+dz*Dy)*Dx; // 3D position within domain and which domain
				const ulong local_i = (ulong)(px+Hx)+((ulong)(py+Hy)+(ulong)(pz+Hz)*local_Ny)*local_Nx; // add halo offsets
				const ulong local_dimension = i/N;
				return buffers[domain]->data()[local_i+local_dimension*local_N]; // array of structures
			}
		}
		inline T& reference(const ulong i, const uint dimension) { // stitch together domain buffers and make them appear as one single large buffer
			if(D==1u) { // take shortcut for single domain
				return buffers[0]->data()[i+(ulong)dimension*N]; // array of structures
			} else { // decompose index for multiple domains
				const ulong global_i=i%N, t=global_i%NxNy;
				const uint x=(uint)(t%(ulong)Nx), y=(uint)(t/(ulong)Nx), z=(uint)(global_i/NxNy); // n = x+(y+z*Ny)*Nx
				const uint px=x%NxDx, py=y%NyDy, pz=z%NzDz, dx=x/NxDx, dy=y/NyDy, dz=z/NzDz, domain=dx+(dy+dz*Dy)*Dx; // 3D position within domain and which domain
				const ulong local_i = (ulong)(px+Hx)+((ulong)(py+Hy)+(ulong)(pz+Hz)*local_Ny)*local_Nx; // add halo offsets
				const ulong local_dimension = max(i/N, (ulong)dimension);
				return buffers[domain]->data()[local_i+local_dimension*local_N]; // array of structures
			}
		}
		inline string vtk_type() const {
			/**/ if constexpr(std::is_same<T, char >::value) return "char" ; else if constexpr(std::is_same<T, uchar >::value) return "unsigned_char" ;
			else if constexpr(std::is_same<T, short>::value) return "short"; else if constexpr(std::is_same<T, ushort>::value) return "unsigned_short";
			else if constexpr(std::is_same<T, int  >::value) return "int"  ; else if constexpr(std::is_same<T, uint  >::value) return "unsigned_int"  ;
			else if constexpr(std::is_same<T, slong>::value) return "long" ; else if constexpr(std::is_same<T, ulong >::value) return "unsigned_long" ;
			else if constexpr(std::is_same<T, float>::value) return "float"; else if constexpr(std::is_same<T, double>::value) return "double"        ;
			else print_error("Error in vtk_type(): Type not supported.");
			return "";
		}
		inline void write_vtk(const string& path, const bool convert_to_si_units=true) { // write binary .vtk file
			float spacing = 1.0f;
			T unit_conversion_factor = (T)1;
			if(convert_to_si_units) {
				spacing = units.si_x(1.0f);
				if(name=="rho") unit_conversion_factor = (T)units.si_rho(1.0f);
				if(name=="u"  ) unit_conversion_factor = (T)units.si_u  (1.0f);
				if(name=="F"  ) unit_conversion_factor = (T)units.si_F  (1.0f);
				if(name=="T"  ) unit_conversion_factor = (T)units.si_T  (1.0f);
			}
			const string filename = create_file_extension(path, ".vtk");
			const float3 origin = spacing*float3(0.5f-0.5f*(float)Nx, 0.5f-0.5f*(float)Ny, 0.5f-0.5f*(float)Nz);
			const string header =
				"# vtk DataFile Version 3.0\nFluidX3D "+filename.substr(filename.rfind('/')+1)+"\nBINARY\nDATASET STRUCTURED_POINTS\n"
				"DIMENSIONS "+to_string(Nx)+" "+to_string(Ny)+" "+to_string(Nz)+"\n"
				"ORIGIN "+to_string(origin.x)+" "+to_string(origin.y)+" "+to_string(origin.z)+"\n"
				"SPACING "+to_string(spacing)+" "+to_string(spacing)+" "+to_string(spacing)+"\n"
				"POINT_DATA "+to_string((ulong)Nx*(ulong)Ny*(ulong)Nz)+"\n"
				"SCALARS data "+vtk_type()+" "+to_string(dimensions())+"\nLOOKUP_TABLE default\n"
			;
			const uint chunk_size_MB = 4u*thread::hardware_concurrency(); // in MB; convert and write data in chunks, to reduce memory footprint and time for large memory allocation
			const ulong chunk_elements = (1048576ull*(ulong)chunk_size_MB)/((ulong)dimensions()*sizeof(T));
			const ulong chunks=length()/chunk_elements, chunk_remainder=length()%chunk_elements;
			T* data = new T[chunk_elements*(ulong)dimensions()];
			create_folder(filename);
			std::ofstream file(filename, std::ios::out|std::ios::binary);
			file.write(header.c_str(), header.length()); // write non-binary file header
			for(ulong c=0u; c<chunks+1ull; c++) { // iterate over all full chunks + last chunk_remainder chunk
				const ulong N = c<chunks ? chunk_elements : chunk_remainder;
				if(N==0ull) break; // chunk_remainder may be 0, then skip last iteration
				parallel_for(N, [&](ulong i) {
					for(uint d=0u; d<dimensions(); d++) { // LBM to SI units, LittleEndian to BigEndian, AoS to SoA
						data[i*(ulong)dimensions()+(ulong)d] = reverse_bytes((T)(unit_conversion_factor*reference(c*chunk_elements+i, d)));
					}
				});
				file.write((char*)data, N*(ulong)dimensions()*sizeof(T)); // write binary data
			}
			file.close();
			delete[] data;
			info.allow_printing.lock();
			print_info("File \""+filename+"\" saved.");
			info.allow_printing.unlock();
		}

	public:
		class Pointer {
		private:
			Memory_Container* memory = nullptr;
			uint dimension = 0u;
		public:
			inline Pointer() {}; // default constructor
			inline Pointer(Memory_Container* memory, const uint dimension) {
				this->memory = memory;
				this->dimension = dimension;
			}
			inline T& operator[](const ulong i) { return memory->reference(i, dimension); }
			inline const T& operator[](const ulong i) const { return memory->reference(i, dimension); }
		};
		Pointer x, y, z; // host buffer auxiliary pointers for multi-dimensional array access (array of structures)

		inline Memory_Container(LBM* lbm, Memory<T>** buffers, const string& name) {
			this->N = lbm->get_N();
			this->d = buffers[0]->dimensions();
			if(this->N*(ulong)this->d==0ull) print_error("Memory size must be larger than 0.");
			this->lbm = lbm;
			this->buffers = buffers;
			this->name = name;
			initialize_auxiliary_variables();
			initialize_auxiliary_pointers();
		}
		inline Memory_Container() {} // default constructor
		inline Memory_Container& operator=(Memory_Container&& memory) noexcept { // move assignment
			this->N = memory.N;
			this->d = memory.d;
			this->lbm = memory.lbm;
			this->buffers = memory.buffers;
			this->name = memory.name;
			initialize_auxiliary_variables();
			initialize_auxiliary_pointers();
			return *this;
		}
		inline void reset(const T value=(T)0) {
			for(uint domain=0u; domain<D; domain++) buffers[domain]->reset(value);
		}
		inline const ulong length() const { return N; }
		inline const uint dimensions() const { return d; }
		inline const ulong range() const { return N*(ulong)d; }
		inline const ulong capacity() const { return N*(ulong)d*sizeof(T); } // returns capacity of the buffer in Byte
		inline T& operator[](const ulong i) { return reference(i); }
		inline const T& operator[](const ulong i) const { return reference(i); }
		inline const T operator()(const ulong i) const { return reference(i); }
		inline const T operator()(const ulong i, const uint dimension) const { return reference(i, dimension); } // array of structures
		inline void read_from_device() {
#ifndef UPDATE_FIELDS
			if(lbm->initialized) for(uint domain=0u; domain<D; domain++) lbm->lbm_domain[domain]->enqueue_update_fields(); // only if simulation has already been initialized: make sure data in device memory is up-to-date
#endif // UPDATE_FIELDS
			for(uint domain=0u; domain<D; domain++) buffers[domain]->enqueue_read_from_device();
			for(uint domain=0u; domain<D; domain++) buffers[domain]->finish_queue();
		}
		inline void write_to_device() {
			for(uint domain=0u; domain<D; domain++) buffers[domain]->enqueue_write_to_device();
			for(uint domain=0u; domain<D; domain++) buffers[domain]->finish_queue();
		}
		inline void write_host_to_vtk(const string& path="", const bool convert_to_si_units=true) { // write binary .vtk file
			write_vtk(default_filename(path, name, ".vtk", lbm->get_t()), convert_to_si_units);
		}
		inline void write_device_to_vtk(const string& path="", const bool convert_to_si_units=true) { // write binary .vtk file
			read_from_device();
			write_host_to_vtk(path, convert_to_si_units);
		}
	};

	LBM_Domain** lbm_domain; // one LBM domain per GPU

	Memory_Container<float> rho; // density of every cell
	Memory_Container<float> u; // velocity of every cell
	Memory_Container<uchar> flags; // flags of every cell
#ifdef FORCE_FIELD
	Memory_Container<float> F; // individual force for every cell
#endif // FORCE_FIELD
#ifdef SURFACE
	Memory_Container<float> phi; // fill level of every cell
#endif // SURFACE
#ifdef TEMPERATURE
	Memory_Container<float> T; // temperature of every cell
#endif // TEMPERATURE
#ifdef MOISTURE
	Memory_Container<float> Q; // total water of every cell as 1 + qt / def_q_scale (this fork)
#endif // MOISTURE
#ifdef PARTICLES
	Memory<float>* particles; // particle positions
#endif // PARTICLES

	LBM(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=0.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const uint particles_N, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint3 N, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=0.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint3 N, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint3 N, const float nu, const uint particles_N, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	~LBM();

	void run(const ulong steps=max_ulong, const ulong total_steps=max_ulong); // initializes the LBM simulation (copies data to device and runs initialize kernel), then runs LBM
	void read_fi_raw(vector<fpxx>& host) { if(get_D()!=1u) print_error("read_fi_raw(): one domain only"); lbm_domain[0]->read_fi_raw(host); } // raw stored populations of the single domain (task 19 P1 instrument; host only)
	void update_fields(); // update fields (rho, u, T) manually
	void reset(); // reset simulation (takes effect in following run() call)
#ifdef FORCE_FIELD
	void update_force_field(); // calculate forces from fluid on TYPE_S cells
	float3 object_center_of_mass(const uchar flag_marker=TYPE_S); // calculate center of mass of all cells flagged with flag_marker
	float3 object_force(const uchar flag_marker=TYPE_S); // add up force for all cells flagged with flag_marker
	float3 object_torque(const float3& rotation_center, const uchar flag_marker=TYPE_S); // add up torque around specified rotation_center for all cells flagged with flag_marker
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
	void update_moving_boundaries(); // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
#endif // MOVING_BOUNDARIES
#if defined(PARTICLES)&&!defined(FORCE_FIELD)
	void integrate_particles(const ulong steps=max_ulong, const ulong total_steps=max_ulong, const uint time_step_multiplicator=1u); // intgegrate passive tracer particles forward in time in stationary flow field
#endif // PARTICLES&&!FORCE_FIELD

#ifdef VOLUMETRIC_WALL
	/// @brief The supported attachment (task 43). Attach the native volumetric wall from a VWNAT01 container before run():
	/// the host flags must already mark the solids (TYPE_S); the box must equal the lattice; R cells get TYPE_V. The
	/// container MUST hold the section friction_rows (geometry-bound friction rows in its own row order; the supported
	/// packer writes the accepted P2 construction). The rows are checked against the container's facets
	/// (coefficient identities, VolumetricWallNativeInput::validate_friction_rows; necessary conditions, not a proof of
	/// the construction) and no other source of coefficients exists on this path (no environment variable, no fallback).
	/// @param vwnat_path path of the container
	/// @note Fails closed with a named reason (print_error): unreadable container, a failed container identity, absent or
	/// incompatible friction rows, box or viscosity mismatch, more than one domain, a call after run(), a second
	/// attachment (replacing an attached wall is not implemented).
	void set_volumetric_wall(const string& vwnat_path);
	/// @brief The research attachment: reproduces the archived fixtures. It is NOT the supported path.
	/// @param vwnat_path path of the container (a friction_rows section in it is ignored here)
	/// @param rows_path "" = the original kernel friction expression (tasks 19-27, the "present" rows); otherwise a file of
	/// exactly rows*3 little-endian doubles in the exported row order (the former VW_PHI file), checked only for length and
	/// finite values, not against the geometry
	/// @note Fails closed (print_error) on the same errors as set_volumetric_wall and on a wrong or non-finite rows file.
	void set_volumetric_wall_research(const string& vwnat_path, const string& rows_path);
#ifdef VOLUMETRIC_WALL_SGS
	/// @brief V2: the constant of the static Smagorinsky closure in the RR collision of every cell (VOLUMETRIC_WALL_SGS).
	/// The per-cell rate is wl = 1/(tau0/2 + sqrt(tau0^2 + k sqrt(A:A)/rho)/2), tau0 = 1/omega, with the full force-aware
	/// second moment A of the collision and Delta = 1. Without a call, k = 0: the rate omega exactly (the V1 collision).
	/// @param k the closure constant k = 18 sqrt(2) (C Delta)^2 (not C). The qualified value is (double)0.76421222f.
	/// @note Fails closed (print_error): k not finite or negative, a call after run(), a call after the attachment
	/// (set_volumetric_wall gives k to the kernels).
	void set_volumetric_wall_sgs(const double k);
	double get_volumetric_wall_sgs() const { return lbm_domain[0]->volumetric_wall_sgs_k(); } ///< the stored SGS constant k (0 without a call)
#endif // VOLUMETRIC_WALL_SGS
	/// @brief Task 105: declare the open-x boundary (VOLUMETRIC_WALL_OPEN_X) before set_volumetric_wall(). The attachment
	/// classifies the container (E cells, exterior facets, x-wrap refusals), sets the initial E state and gives the
	/// declaration to the kernels; it is immutable afterwards. An open build refuses the attachment without it, run() with a
	/// nonzero force (contract A5). The open boundary is a compile-time option: a build without VOLUMETRIC_WALL_OPEN_X
	/// refuses this call (print_error), so the closed builds keep their code and behaviour.
	/// @param open the declaration (validated here; the initial E state is checked against the refusals 7.6 at the attachment)
	/// @note Fails closed (print_error): no VOLUMETRIC_WALL_OPEN_X, an invalid declaration, a second call, a call after the
	/// attachment or after run().
	void set_volumetric_wall_open_x(const fluidx3d::experimental::VolumetricWallOpenX& open);
#ifdef VOLUMETRIC_WALL_OPEN_X
	/// @brief The current E state, 4 doubles (rho, ux, uy, uz) per slot side*(Ny Nz) + y + Ny z (blocking read).
	void volumetric_wall_read_open_state(vector<double>& host) { lbm_domain[0]->vw_read_open_state(host); }
	/// @brief E states that failed a refusal 7.6 since the attachment (blocking; any count invalidates the run).
	uint volumetric_wall_open_failures() { return lbm_domain[0]->volumetric_wall_open_failures(); }
	/// @brief The declaration as stored and given to the kernels (the kernel arguments are copies of these values).
	fluidx3d::experimental::VolumetricWallOpenX volumetric_wall_open_declaration() const { return lbm_domain[0]->volumetric_wall_open_declaration(); }
	/// @brief How many E cells take their feq3 from vw_collide (R cells), vw_support_rr27 (support cells) and stream_collide (ordinary cells).
	std::array<uint, 3> volumetric_wall_open_e_paths() const { return lbm_domain[0]->volumetric_wall_open_e_paths(); }
	/// @brief The classification the attachment used (E cells in mesh order, their slots and sides, exterior facets and rows).
	const fluidx3d::experimental::VolumetricWallOpenLayout& volumetric_wall_open_layout() const { return lbm_domain[0]->volumetric_wall_open_layout(); }
#endif // VOLUMETRIC_WALL_OPEN_X
	bool volumetric_wall_enabled() const { return lbm_domain[0]->volumetric_wall_enabled(); }
	fluidx3d::experimental::VolumetricWallNativeCapture volumetric_wall_capture() { return lbm_domain[0]->read_volumetric_wall_buffers(); }
	void volumetric_wall_read_fi(vector<fpxx>& host) { lbm_domain[0]->vw_read_fi(host); }
	void volumetric_wall_write_fi(const vector<fpxx>& host) { lbm_domain[0]->vw_write_fi(host); }
#ifdef VOLUMETRIC_WALL_SRD
	void volumetric_wall_srd_apply(const ulong t1) { lbm_domain[0]->enqueue_volumetric_wall_srd(t1); lbm_domain[0]->finish_queue(); } // task 121: the redistribution alone on the stored level-t1 load sets (operator check)
	void volumetric_wall_srd_read(vector<double>& qhat, vector<double>& recon, vector<uint>& stats) { lbm_domain[0]->volumetric_wall_srd_read(qhat, recon, stats); }
	void set_volumetric_wall_srd_capture(const bool on) { lbm_domain[0]->set_volumetric_wall_srd_capture(on); }
	void volumetric_wall_srd_pre_fi(vector<fpxx>& host) { lbm_domain[0]->volumetric_wall_srd_pre_fi(host); }
	void volumetric_wall_pre_wall_fi(vector<fpxx>& host) { lbm_domain[0]->volumetric_wall_pre_wall_fi(host); }
#endif // VOLUMETRIC_WALL_SRD
	void volumetric_wall_read_pending(vector<double>& host) { lbm_domain[0]->vw_read_pending(host); }
	void volumetric_wall_write_pending(const vector<double>& host) { lbm_domain[0]->vw_write_pending(host); }
	void volumetric_wall_set_time_step(const ulong t) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_time_step(t); } // restart: the parity of the restored fi
#endif // VOLUMETRIC_WALL
	uint get_Nx() const { return Nx; } // get (global) lattice dimensions in x-direction
	uint get_Ny() const { return Ny; } // get (global) lattice dimensions in y-direction
	uint get_Nz() const { return Nz; } // get (global) lattice dimensions in z-direction
	ulong get_N() const { return (ulong)Nx*(ulong)Ny*(ulong)Nz; } // get (global) number of lattice points
	uint get_Dx() const { return Dx; } // get lattice domains in x-direction
	uint get_Dy() const { return Dy; } // get lattice domains in y-direction
	uint get_Dz() const { return Dz; } // get lattice domains in z-direction
	uint get_D() const { return Dx*Dy*Dz; } // get number of lattice domains
	float get_nu() const { return lbm_domain[0]->get_nu(); } // get kinematic shear viscosity
	float get_tau() const { return 3.0f*get_nu()+0.5f; } // get LBM relaxation time
	float get_Re_max() const { return 0.57735027f*sqrt((float)(sq(Nx)+sq(Ny)+sq(Nz)))/get_nu(); } // Re < Re_max = c*L_max/nu
	float get_fx() const { return lbm_domain[0]->get_fx(); } // get global froce per volume
	float get_fy() const { return lbm_domain[0]->get_fy(); } // get global froce per volume
	float get_fz() const { return lbm_domain[0]->get_fz(); } // get global froce per volume
	float get_sigma() const { return lbm_domain[0]->get_sigma(); } // get surface tension coefficient
	float get_alpha() const { return lbm_domain[0]->get_alpha(); } // get thermal diffusion coefficient
	float get_beta() const { return lbm_domain[0]->get_beta(); } // get thermal expansion coefficient
	float get_heat_rate() const { return lbm_domain[0]->get_heat_rate(); } // get temperature added per time step to TYPE_Y fluid cells
	float get_inv_prandtl_t() const { return lbm_domain[0]->get_inv_prandtl_t(); } // get 1/Pr_t of the eddy heat diffusivity
	void set_heat_rate(const float heat_rate) { // set temperature added per time step to each fluid cell flagged TYPE_Y (TEMPERATURE)
#ifndef TEMPERATURE
		if(heat_rate!=0.0f) print_error("LBM::set_heat_rate() needs TEMPERATURE enabled.");
#endif // TEMPERATURE
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_heat_rate(heat_rate);
	}
#ifdef SCALAR_FV
	/// The finite-volume inventory of a one-domain lattice: the sums of rho_FV theta (the conserved quantity) and of rho_FV over
	/// the fluid cells, read from the device (this fork; the budget tests).
	void fv_inventory(double& sum_rho_theta, double& sum_rho_fv) {
		if(get_D()!=1u) print_error("LBM::fv_inventory() reads one domain only.");
		lbm_domain[0]->read_fv_inventory(sum_rho_theta, sum_rho_fv);
	}
#endif // SCALAR_FV
	/// LEVEL_FORCING: the per-level terms for every domain (the domains must not be split along z: the level index is the local z).
	void set_level_forcing(const std::vector<float>& fx, const std::vector<float>& fy, const std::vector<float>& rate, const std::vector<float>& T_ref) {
#ifndef LEVEL_FORCING
		print_error("LBM::set_level_forcing() needs LEVEL_FORCING enabled.");
#else // LEVEL_FORCING
		if(get_Dz()>1u) print_error("LBM::set_level_forcing(): domains split along z are not supported (the level index is the local z).");
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_level_forcing(fx, fy, rate, T_ref);
#endif // LEVEL_FORCING
	}
	void set_moisture_rate(const float moisture_rate) { // set Q added per time step to each fluid cell flagged TYPE_Y (MOISTURE)
#ifndef MOISTURE
		if(moisture_rate!=0.0f) print_error("LBM::set_moisture_rate() needs MOISTURE enabled.");
#endif // MOISTURE
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_moisture_rate(moisture_rate);
	}
	/// MOISTURE: the potential temperature (K) that T = 1 stands for (the saturation adjustment needs the real temperature).
	void set_theta_0(const float theta_0) {
#ifndef MOISTURE
		print_error("LBM::set_theta_0() needs MOISTURE enabled.");
#else // MOISTURE
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_theta_0(theta_0);
#endif // MOISTURE
	}
	/// MOISTURE with LEVEL_FORCING: the per-level pressure and reference virtual potential temperature for every domain.
	void set_level_moist(const std::vector<float>& pressure, const std::vector<float>& theta_v_ref) {
#if !defined(MOISTURE)||!defined(LEVEL_FORCING)
		print_error("LBM::set_level_moist() needs MOISTURE and LEVEL_FORCING enabled.");
#else
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_level_moist(pressure, theta_v_ref);
#endif
	}
	void set_inv_prandtl_t(const float inv_prandtl_t) { // set 1/Pr_t: the eddy heat diffusivity nu_t/Pr_t of the SUBGRID model is added to alpha (TEMPERATURE with SUBGRID)
#if !defined(TEMPERATURE)||!defined(SUBGRID)
		if(inv_prandtl_t!=0.0f) print_error("LBM::set_inv_prandtl_t() needs both TEMPERATURE and SUBGRID enabled.");
#endif // TEMPERATURE and SUBGRID
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_inv_prandtl_t(inv_prandtl_t);
	}
	ulong get_t() const { return lbm_domain[0]->get_t(); } // get discrete time step in LBM units
	uint get_velocity_set() const { return lbm_domain[0]->get_velocity_set(); }
	void set_fx(const float fx) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_fx(fx); } // set global froce per volume
	void set_fy(const float fy) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_fy(fy); } // set global froce per volume
	void set_fz(const float fz) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_fz(fz); } // set global froce per volume
	void set_f(const float fx, const float fy, const float fz) { set_fx(fx); set_fy(fy); set_fz(fz); } // set global froce per volume

	void coordinates(const ulong n, uint& x, uint& y, uint& z) const { // disassemble 1D linear index to 3D coordinates (n -> x,y,z)
		const ulong t = n%((ulong)Nx*(ulong)Ny); // n = x+(y+z*Ny)*Nx
		x = (uint)(t%(ulong)Nx);
		y = (uint)(t/(ulong)Nx);
		z = (uint)(n/((ulong)Nx*(ulong)Ny));
	}
	void coordinates(const float3& p, uint& x, uint& y, uint& z) const { // turn 3D position into closest 3D grid coordinates
		const float3 mp = mirror_position(p);
		x = (uint)(mp.x+1.5f*(float)Nx)%Nx;
		y = (uint)(mp.y+1.5f*(float)Ny)%Ny;
		z = (uint)(mp.z+1.5f*(float)Nz)%Nz;
	}
	ulong index(const uint x, const uint y, const uint z) const { // turn 3D coordinates into 1D linear index
		return (ulong)x+((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx;
	}
	ulong index(const uint3 xyz) const { // turn 3D coordinates into 1D linear index
		return index(xyz.x, xyz.y, xyz.z);
	}
	ulong index(const float3& p) const { // turn 3D position into closest 1D linear index
		uint x=0u, y=0u, z=0u;
		coordinates(p, x, y, z);
		return index(x, y, z);
	}
	float3 position(const uint x, const uint y, const uint z) const { // returns position in box [-Nx/2, Nx/2] x [-Ny/2, Ny/2] x [-Nz/2, Nz/2]
		return float3((float)x-0.5f*(float)Nx+0.5f, (float)y-0.5f*(float)Ny+0.5f, (float)z-0.5f*(float)Nz+0.5f);
	}
	float3 position(const ulong n) const { // returns position in box [-Nx/2, Nx/2] x [-Ny/2, Ny/2] x [-Nz/2, Nz/2]
		uint x, y, z;
		coordinates(n, x, y, z);
		return position(x, y, z);
	}
	float3 mirror_position(const float3& p) const { // mirror position into periodic boundaries
		float3 r;
		r.x = sign(p.x)*(fmod(fabs(p.x)+0.5f*(float)Nx, (float)Nx)-0.5f*(float)Nx);
		r.y = sign(p.y)*(fmod(fabs(p.y)+0.5f*(float)Ny, (float)Ny)-0.5f*(float)Ny);
		r.z = sign(p.z)*(fmod(fabs(p.z)+0.5f*(float)Nz, (float)Nz)-0.5f*(float)Nz);
		return r;
	}
	float3 size() const { // returns size of box
		return float3((float)Nx, (float)Ny, (float)Nz);
	}
	float3 center() const { // returns center of box
		return float3(0.5f*(float)Nx-0.5f, 0.5f*(float)Ny-0.5f, 0.5f*(float)Nz-0.5f);
	}
	uint smallest_side_length() const {
		return min(min(Nx, Ny), Nz);
	}
	uint largest_side_length() const {
		return max(max(Nx, Ny), Nz);
	}
	float3 relative_position(const uint x, const uint y, const uint z) const { // returns relative position in box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
		return float3(((float)x+0.5f)/(float)Nx-0.5f, ((float)y+0.5f)/(float)Ny-0.5f, ((float)z+0.5f)/(float)Nz-0.5f);
	}
	float3 relative_position(const ulong n) const { // returns relative position in box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
		uint x, y, z;
		coordinates(n, x, y, z);
		return relative_position(x, y, z);
	}
	void write_status(const string& path=""); // write LBM status report to a .txt file

	void voxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S, const float3& rotation_center=float3(0.0f), const float3& linear_velocity=float3(0.0f), const float3& rotational_velocity=float3(0.0f)); // voxelize mesh
	void unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S); // remove voxelized triangle mesh from LBM grid
	void write_mesh_to_vtk(const Mesh* mesh, const string& path="", const bool convert_to_si_units=true) const; // write mesh to binary .vtk file
	void voxelize_stl(const string& path, const float3& center, const float3x3& rotation, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file
	void voxelize_stl(const string& path, const float3x3& rotation, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file (place in box center)
	void voxelize_stl(const string& path, const float3& center, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file (no rotation)
	void voxelize_stl(const string& path, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file (place in box center, no rotation)
	void voxelize_sdf(const string& path, const float3& center, const float3x3& rotation, const float scale=1.0f, const uchar flag=TYPE_S); // read and voxelize binary SDF file
	void voxelize_sdf(const string& path, const float3x3& rotation, const float scale=1.0f, const uchar flag=TYPE_S); // read and voxelize binary SDF file (place in box center)
	void voxelize_sdf(const string& path, const float3& center, const float scale=1.0f, const uchar flag=TYPE_S); // read and voxelize binary SDF file (no rotation)
	void voxelize_sdf(const string& path, const float scale=1.0f, const uchar flag=TYPE_S); // read and voxelize binary SDF file (place in box center, no rotation)

#ifdef GRAPHICS
	class Graphics {
	private:
		LBM* lbm = nullptr;
		std::atomic_int running_encoders = 0;
		uint last_exported_frame = 0u; // for next_frame(...) function
		int last_visualization_modes=0, last_field_mode=0, last_slice_mode=0, last_slice_x=0, last_slice_y=0, last_slice_z=0; // don't render a new frame if the scene hasn't changed since last frame
		void default_settings() {
			visualization_modes |= VIS_FLAG_LATTICE;
#ifdef PARTICLES
			visualization_modes |= VIS_PARTICLES;
#endif // PARTICLES
		}

	public:
		int visualization_modes=0, field_mode=0, slice_mode=0, slice_x=0, slice_y=0, slice_z=0; // field_mode = { 0 (u), 1 (rho), 2 (T) }, slice_mode = { 0 (no slice), 1 (x), 2 (y), 3 (z), 4 (xz), 5 (xyz), 6 (yz), 7 (xy) }, slice_{xyz} = position of slices

		Graphics() {} // default constructor
		Graphics(LBM* lbm) {
			this->lbm = lbm;
			camera.set_zoom(0.5f*(float)fmax(fmax(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz()));
			slice_x = (int)lbm->get_Nx()/2;
			slice_y = (int)lbm->get_Ny()/2;
			slice_z = (int)lbm->get_Nz()/2;
			default_settings();
		}
		~Graphics() { // destructor must wait for all encoder threads to finish
			int last_value = running_encoders.load();
			while(last_value>0) {
				const int current_value = running_encoders.load();
				if(last_value!=current_value) {
					print_info("Finishing encoder threads: "+to_string(current_value));
					last_value = current_value;
				}
				sleep(0.016);
			}
		}
		Graphics& operator=(const Graphics& graphics) { // copy assignment
			lbm = graphics.lbm;
			visualization_modes = graphics.visualization_modes;
			field_mode = graphics.field_mode;
			slice_mode = graphics.slice_mode;
			slice_x = graphics.slice_x;
			slice_y = graphics.slice_y;
			slice_z = graphics.slice_z;
			return *this;
		}

		int* draw_frame(); // main rendering function, calls rendering kernels

		void set_camera_centered(const float rx=0.0f, const float ry=0.0f, const float fov=100.0f, const float zoom=1.0f); // set camera centered
		void set_camera_free(const float3& p=float3(0.0f), const float rx=0.0f, const float ry=0.0f, const float fov=100.0f); // set camera free
		bool next_frame(const ulong total_time_steps, const float video_length_seconds); // returns true once simulation time has progressed enough to render the next video frame for a 60fps video of specified length
		void print_frame(); // preview preview of current frame in console
		void write_frame(const string& path="", const string& name="image", const string& extension=".png", bool print_preview=false); // save current frame
		void write_frame(const uint x1, const uint y1, const uint x2, const uint y2, const string& path="", const string& name="image", const string& extension=".png", bool print_preview=false); // save current frame cropped with two corner points (x1,y1) and (x2,y2)
		void write_frame_png(const string& path="", bool print_preview=false); // save current frame as .png file (smallest file size, but slow)
		void write_frame_qoi(const string& path="", bool print_preview=false); // save current frame as .qoi file (small file size, fast)
		void write_frame_bmp(const string& path="", bool print_preview=false); // save current frame as .bmp file (large file size, fast)
		void write_frame_png(const uint x1, const uint y1, const uint x2, const uint y2, const string& path="", bool print_preview=false); // save current frame as .png file (smallest file size, but slow)
		void write_frame_qoi(const uint x1, const uint y1, const uint x2, const uint y2, const string& path="", bool print_preview=false); // save current frame as .qoi file (small file size, fast)
		void write_frame_bmp(const uint x1, const uint y1, const uint x2, const uint y2, const string& path="", bool print_preview=false); // save current frame as .bmp file (large file size, fast)
	}; // Graphics
	Graphics graphics;
#endif // GRAPHICS
}; // LBM
#ifdef VOLUMETRIC_WALL_OPEN_X
std::string vw_open_refusal(const double rho_b, const double ux, const double uy, const double uz, const int s); // task 125: the host admission predicate of one E state (empty text = admissible); declared for the control fixture
#endif // VOLUMETRIC_WALL_OPEN_X
