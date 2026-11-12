#include "lbm.hpp"



Units units; // for unit conversion

#if defined(D2Q9)
const uint velocity_set = 9u;
const uint dimensions = 2u;
const uint transfers = 3u;
#elif defined(D3Q15)
const uint velocity_set = 15u;
const uint dimensions = 3u;
const uint transfers = 5u;
#elif defined(D3Q19)
const uint velocity_set = 19u;
const uint dimensions = 3u;
const uint transfers = 5u;
#elif defined(D3Q27)
const uint velocity_set = 27u;
const uint dimensions = 3u;
const uint transfers = 9u;
#endif // D3Q27

uint bytes_per_cell_host() { // returns the number of Bytes per cell allocated in host memory
	uint bytes_per_cell = 17u; // rho, u, flags
#ifdef FORCE_FIELD
	bytes_per_cell += 12u; // F
#endif // FORCE_FIELD
#ifdef SURFACE
	bytes_per_cell += 4u; // phi
#endif // SURFACE
#ifdef TEMPERATURE
	bytes_per_cell += 4u; // T
#ifdef MOISTURE
	bytes_per_cell += 4u; // Q
#endif // MOISTURE
#endif // TEMPERATURE
	return bytes_per_cell;
}
uint bytes_per_cell_device() { // returns the number of Bytes per cell allocated in device memory
	uint bytes_per_cell = velocity_set*sizeof(fpxx)+17u; // fi, rho, u, flags
#ifdef FORCE_FIELD
	bytes_per_cell += 12u; // F
#endif // FORCE_FIELD
#ifdef SURFACE
	bytes_per_cell += 12u; // phi, mass, flags
#endif // SURFACE
#ifdef TEMPERATURE
#ifdef SCALAR_FV
	bytes_per_cell += 4u; // T (the finite volumes carry T: gi has one cell only)
#else // SCALAR_FV
	bytes_per_cell += 7u*sizeof(fpxx)+4u; // gi, T
#endif // SCALAR_FV
#ifdef MOISTURE
#ifdef SCALAR_FV
	bytes_per_cell += 4u; // Q (hi has one cell only)
#else // SCALAR_FV
	bytes_per_cell += 7u*sizeof(fpxx)+4u; // hi, Q
#endif // SCALAR_FV
#endif // MOISTURE
#endif // TEMPERATURE
#ifdef SCALAR_FV
	bytes_per_cell += 20u; // fvR, fvT, fvT_half, fvRT, fvD
#ifdef MOISTURE
	bytes_per_cell += 12u; // fvQ, fvQ_half, fvRQ
#endif // MOISTURE
#endif // SCALAR_FV
	return bytes_per_cell;
}
uint bandwidth_bytes_per_cell_device() { // returns the bandwidth in Bytes per cell per time step from/to device memory
	uint bandwidth_bytes_per_cell = velocity_set*2u*sizeof(fpxx)+1u; // fi, flags
#ifdef UPDATE_FIELDS
	bandwidth_bytes_per_cell += 16u; // rho, u
#ifdef TEMPERATURE
	bandwidth_bytes_per_cell += 4u; // T
#endif // TEMPERATURE
#endif // UPDATE_FIELDS
#ifdef FORCE_FIELD
	bandwidth_bytes_per_cell += 12u; // F
#endif // FORCE_FIELD
#if defined(MOVING_BOUNDARIES)||defined(SURFACE)||defined(TEMPERATURE)
	bandwidth_bytes_per_cell += (velocity_set-1u)*1u; // neighbor flags have to be loaded
#endif // MOVING_BOUNDARIES, SURFACE or TEMPERATURE
#ifdef SURFACE
	bandwidth_bytes_per_cell += (1u+(2u*velocity_set-1u)*sizeof(fpxx)+8u+(velocity_set-1u)*4u) + 1u + 1u + (4u+velocity_set+4u+4u+4u); // surface_0 (flags, fi, mass, massex), surface_1 (flags), surface_2 (flags), surface_3 (rho, flags, mass, massex, phi)
#endif // SURFACE
#if defined(TEMPERATURE)&&!defined(SCALAR_FV) // with SCALAR_FV, stream_collide neither loads nor stores gi and hi
	bandwidth_bytes_per_cell += 7u*2u*sizeof(fpxx); // 2*gi
#ifdef MOISTURE
	bandwidth_bytes_per_cell += 7u*2u*sizeof(fpxx); // 2*hi
#endif // MOISTURE
#endif // TEMPERATURE && !SCALAR_FV
	return bandwidth_bytes_per_cell;
}
uint3 resolution(const float3 box_aspect_ratio, const uint memory) { // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
#ifndef D2Q9
	float memory_required = (box_aspect_ratio.x*box_aspect_ratio.y*box_aspect_ratio.z)*(float)bytes_per_cell_device()/1048576.0f; // in MB
	float scaling = cbrt((float)memory/memory_required);
	return uint3(to_uint(scaling*box_aspect_ratio.x), to_uint(scaling*box_aspect_ratio.y), to_uint(scaling*box_aspect_ratio.z));
#else // D2Q9
	float memory_required = (box_aspect_ratio.x*box_aspect_ratio.y)*(float)bytes_per_cell_device()/1048576.0f; // in MB
	float scaling = sqrt((float)memory/memory_required);
	return uint3(to_uint(scaling*box_aspect_ratio.x), to_uint(scaling*box_aspect_ratio.y), 1u);
#endif // D2Q9
}

string default_filename(const string& path, const string& name, const string& extension, const ulong t) { // generate a default filename with timestamp
	string time = "00000000"+to_string(t);
	time = substring(time, length(time)-9u, 9u);
	return (path=="" ? get_exe_path()+"export/" : path)+create_file_extension((name=="" ? "file" : name)+"-"+time, extension);
}
string default_filename(const string& name, const string& extension, const ulong t) { // generate a default filename with timestamp at exe_path/export/
	return default_filename("", name, extension, t);
}



LBM_Domain::LBM_Domain(const Device_Info& device_info, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const int Ox, const int Oy, const int Oz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // constructor with manual device selection and domain offset
	this->Nx = Nx; this->Ny = Ny; this->Nz = Nz;
	this->Dx = Dx; this->Dy = Dy; this->Dz = Dz;
	this->Ox = Ox; this->Oy = Oy; this->Oz = Oz;
	this->nu = nu;
	this->fx = fx; this->fy = fy; this->fz = fz;
	this->sigma = sigma;
	this->alpha = alpha; this->beta = beta;
	this->particles_N = particles_N;
	this->particles_rho = particles_rho;
	string opencl_c_code;
#ifdef GRAPHICS
	graphics = Graphics(this);
	opencl_c_code = device_defines(device_info)+graphics.device_defines(device_info)+get_opencl_c_code();
#else // GRAPHICS
	opencl_c_code = device_defines(device_info)+get_opencl_c_code();
#endif // GRAPHICS
	this->device = Device(device_info, opencl_c_code);
	print_info("Allocating memory. This may take a few seconds.");
	allocate(device); // lbm first
#ifdef GRAPHICS
	graphics.allocate(device); // graphics after lbm
#endif // GRAPHICS
}

void LBM_Domain::allocate(Device& device) {
	const ulong N = get_N();
	fi = Memory<fpxx>(device, N, velocity_set, false);
	rho = Memory<float>(device, N, 1u, true, true, 1.0f);
	u = Memory<float>(device, N, 3u);
	flags = Memory<uchar>(device, N);
	kernel_initialize = Kernel(device, N, "initialize", fi, rho, u, flags);
	kernel_stream_collide = Kernel(device, N, "stream_collide", fi, rho, u, flags, t, fx, fy, fz);
	kernel_update_fields = Kernel(device, N, "update_fields", fi, rho, u, flags, t, fx, fy, fz);
#ifdef VOLUMETRIC_WALL
	vw_clamp = Memory<uint>(device, 1ull, 1u, true, true, 0u); // native clamp activations (task 19), zero at start
	vw_clamp.write_to_device();
	kernel_stream_collide.add_parameters(vw_clamp);
#endif // VOLUMETRIC_WALL
#ifdef VOLUMETRIC_WALL_FP64_COLLISION
	kernel_stream_collide.add_parameters(1.0/(3.0*(double)nu+0.5)); // task 57: parameter 9, the FP64 relaxation rate of the ordinary cells; set_volumetric_wall replaces it by the container's omega
#endif // VOLUMETRIC_WALL_FP64_COLLISION
#ifdef VOLUMETRIC_WALL_SGS
	kernel_stream_collide.add_parameters(0.0); // V2: parameter 10, the SGS constant k (0 = the rate omega); set_volumetric_wall gives it the stored vw_sgs_k
#endif // VOLUMETRIC_WALL_SGS
#ifdef VOLUMETRIC_WALL_OPEN_X
	vw_estate = Memory<double>(device, 8ull*(ulong)Ny*(ulong)Nz, 1u, true, true, 0.0); // task 105: rho, ux, uy, uz per slot (2 Ny Nz slots); the attachment writes the initial state
	vw_estate_next = Memory<double>(device, 8ull*(ulong)Ny*(ulong)Nz, 1u, true, true, 0.0);
#endif // VOLUMETRIC_WALL_OPEN_X

#ifdef FORCE_FIELD
	F = Memory<float>(device, N, 3u);
	object_sum = Memory<float>(device, 1u, 4u); // x, y, z, cell count
	kernel_stream_collide.add_parameters(F);
	kernel_update_fields.add_parameters(F);
	kernel_update_force_field = Kernel(device, N, "update_force_field", fi, flags, t, F);
	kernel_reset_force_field = Kernel(device, N, "reset_force_field", F);
	kernel_object_center_of_mass = Kernel(device, N, "object_center_of_mass", flags, (uchar)0u, object_sum);
	kernel_object_force = Kernel(device, N, "object_force", F, flags, (uchar)0u, object_sum);
	kernel_object_torque = Kernel(device, N, "object_torque", F, flags, (uchar)0u, 0.0f, 0.0f, 0.0f, object_sum);
#endif // FORCE_FIELD

#ifdef MOVING_BOUNDARIES
	kernel_update_moving_boundaries = Kernel(device, N, "update_moving_boundaries", u, flags);
#endif // MOVING_BOUNDARIES

#ifdef SURFACE
	phi = Memory<float>(device, N);
	mass = Memory<float>(device, N, 1u, false);
	massex = Memory<float>(device, N, 1u, false);
	kernel_initialize.add_parameters(mass, massex, phi);
	kernel_stream_collide.add_parameters(mass);
	kernel_surface_0 = Kernel(device, N, "surface_0", fi, rho, u, flags, mass, massex, phi, t, fx, fy, fz);
	kernel_surface_1 = Kernel(device, N, "surface_1", flags);
	kernel_surface_2 = Kernel(device, N, "surface_2", fi, rho, u, flags, t);
	kernel_surface_3 = Kernel(device, N, "surface_3", rho, flags, mass, massex, phi);
#endif // SURFACE

#ifdef TEMPERATURE
#ifdef SCALAR_FV
	gi = Memory<fpxx>(device, 1ull, 7u, false); // a placeholder: the finite volumes carry T, no kernel reads or writes gi (the argument order stays)
#else // SCALAR_FV
	gi = Memory<fpxx>(device, N, 7u, false);
#endif // SCALAR_FV
	T = Memory<float>(device, N, 1u, true, true, 1.0f);
	kernel_initialize.add_parameters(gi, T);
	kernel_stream_collide.add_parameters(gi, T, heat_rate, inv_prandtl_t);
	kernel_update_fields.add_parameters(gi, T);
#endif // TEMPERATURE
#ifdef LEVEL_FORCING
	level = Memory<float>(device, 8ull*(ulong)Nz, 1u, true, true, 0.0f); // 8 floats per level; last in the argument lists: the indices of the earlier parameters stay
	for(ulong z=0ull; z<(ulong)Nz; z++) level[8ull*z+3ull] = 1.0f; // the reference temperature defaults to 1: the buoyancy as without LEVEL_FORCING
	level.write_to_device();
	kernel_stream_collide.add_parameters(level);
	kernel_update_fields.add_parameters(level);
#endif // LEVEL_FORCING
#ifdef MOISTURE
#ifdef SCALAR_FV
	hi = Memory<fpxx>(device, 1ull, 7u, false); // a placeholder: the finite volumes carry Q, no kernel reads or writes hi (the argument order stays)
#else // SCALAR_FV
	hi = Memory<fpxx>(device, N, 7u, false);
#endif // SCALAR_FV
	Q = Memory<float>(device, N, 1u, true, true, 1.0f);
	kernel_initialize.add_parameters(hi, Q);
	kernel_stream_collide.add_parameters(hi, Q, moisture_rate, theta_0);
	kernel_update_fields.add_parameters(hi, Q, theta_0);
#endif // MOISTURE
#ifdef SCALAR_FV
	fvR = Memory<float>(device, N, 1u, false, true, 0.0f);
	fvT = Memory<float>(device, N, 1u, false, true, 0.0f);
	fvT_half = Memory<float>(device, N, 1u, false, true, 0.0f);
	fvRT = Memory<float>(device, N, 1u, false, true, 0.0f);
	fvD = Memory<float>(device, N, 1u, false, true, 0.0f);
	kernel_stream_collide.add_parameters(fvD);
	// the midpoint step with the step's link mass fluxes (read from fi after stream_collide; t_next, index 3, is set every step);
	// the full stage writes rho_FV, rho theta, theta and T in place
	kernel_fv_half = Kernel(device, N, "fv_scalars", fi, flags, fvD, 0ull, heat_rate, 0u, fvR, fvT, fvT_half, fvRT, T); // the half stage: phi at the half step
	kernel_fv_full = Kernel(device, N, "fv_scalars", fi, flags, fvD, 0ull, heat_rate, 1u, fvR, fvT, fvT_half, fvRT, T); // the full stage, with the sources
	kernel_fv_init = Kernel(device, N, "fv_init", rho, T, fvR, fvT, fvRT);
#ifdef MOISTURE
	fvQ = Memory<float>(device, N, 1u, false, true, 0.0f);
	fvQ_half = Memory<float>(device, N, 1u, false, true, 0.0f);
	fvRQ = Memory<float>(device, N, 1u, false, true, 0.0f);
	kernel_fv_half.add_parameters(fvQ, fvQ_half, fvRQ, Q, moisture_rate);
	kernel_fv_full.add_parameters(fvQ, fvQ_half, fvRQ, Q, moisture_rate);
	kernel_fv_init.add_parameters(Q, fvQ, fvRQ);
#endif // MOISTURE
#ifdef LEVEL_FORCING
	kernel_fv_half.add_parameters(level);
	kernel_fv_full.add_parameters(level);
#endif // LEVEL_FORCING
#endif // SCALAR_FV

#ifdef PARTICLES
	particles = Memory<float>(device, (ulong)particles_N, 3u);
	kernel_integrate_particles = Kernel(device, (ulong)particles_N, "integrate_particles", particles, u, flags, 1.0f);
#ifdef FORCE_FIELD
	kernel_integrate_particles.add_parameters(F, fx, fy, fz);
#endif // FORCE_FIELD
#endif // PARTICLES

	if(get_D()>1u) allocate_transfer(device);
}

void LBM_Domain::enqueue_initialize() { // call kernel_initialize
	kernel_initialize.enqueue_run();
}
void LBM_Domain::enqueue_stream_collide() { // call kernel_stream_collide to perform one LBM time step
	kernel_stream_collide.set_parameters(4u, t, fx, fy, fz);
#ifdef TEMPERATURE
	const uint heat_parameters = 10u // after fi, rho, u, flags, t, fx, fy, fz, gi, T
#ifdef FORCE_FIELD
		+1u // F
#endif // FORCE_FIELD
#ifdef SURFACE
		+1u // mass
#endif // SURFACE
	;
	kernel_stream_collide.set_parameters(heat_parameters, heat_rate, inv_prandtl_t);
#ifdef MOISTURE
	const uint moisture_parameter = heat_parameters+2u // heat_rate, inv_prandtl_t
#ifdef LEVEL_FORCING
		+1u // level
#endif // LEVEL_FORCING
		+2u; // hi, Q
	kernel_stream_collide.set_parameters(moisture_parameter, moisture_rate, theta_0);
#endif // MOISTURE
#endif // TEMPERATURE
	kernel_stream_collide.enqueue_run();
}
#ifdef SCALAR_FV
void LBM_Domain::enqueue_fv_half() { // the half stage: after stream_collide stored fi and wrote fvD, and the domains exchanged both (LBM::do_time_step())
	kernel_fv_half.set_parameters(3u, t+1ull, heat_rate); // t_next: the parity the next step's load_f uses, then heat_rate
#ifdef MOISTURE
	kernel_fv_half.set_parameters(15u, moisture_rate); // after fvQ, fvQ_half, fvRQ, Q
#endif // MOISTURE
	kernel_fv_half.enqueue_run();
}
void LBM_Domain::enqueue_fv_full() { // the full stage: after the domains exchanged the half-step fields, before the next step reads T
	kernel_fv_full.set_parameters(3u, t+1ull, heat_rate);
#ifdef MOISTURE
	kernel_fv_full.set_parameters(15u, moisture_rate);
#endif // MOISTURE
	kernel_fv_full.enqueue_run();
}
#endif // SCALAR_FV
void LBM_Domain::enqueue_update_fields() { // update fields (rho, u, T) manually
#ifndef UPDATE_FIELDS
	if(t!=t_last_update_fields) { // only run kernel_update_fields if the time step has changed since last update
		kernel_update_fields.set_parameters(4u, t, fx, fy, fz).enqueue_run();
		t_last_update_fields = t;
	}
#endif // UPDATE_FIELDS
}
#ifdef SURFACE
void LBM_Domain::enqueue_surface_0() {
	kernel_surface_0.set_parameters(7u, t, fx, fy, fz).enqueue_run();
}
void LBM_Domain::enqueue_surface_1() {
	kernel_surface_1.enqueue_run();
}
void LBM_Domain::enqueue_surface_2() {
	kernel_surface_2.set_parameters(4u, t).enqueue_run();
}
void LBM_Domain::enqueue_surface_3() {
	kernel_surface_3.enqueue_run();
}
#endif // SURFACE
#ifdef FORCE_FIELD
void LBM_Domain::enqueue_update_force_field() { // calculate forces from fluid on TYPE_S cells
	if(t!=t_last_force_field) { // only run kernel_update_force_field if the time step has changed since last update
		kernel_update_force_field.set_parameters(2u, t).enqueue_run();
		t_last_force_field = t;
	}
}
void LBM_Domain::enqueue_object_center_of_mass(const uchar flag_marker) { // calculate center of mass of all cells flagged with flag_marker
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_center_of_mass.set_parameters(1u, flag_marker).enqueue_run();
	object_sum.enqueue_read_from_device();
}
void LBM_Domain::enqueue_object_force(const uchar flag_marker) { // add up force for all cells flagged with flag_marker
	enqueue_update_force_field(); // update force field if it is not yet up-to-date
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_force.set_parameters(2u, flag_marker).enqueue_run();
	object_sum.enqueue_read_from_device();
}
void LBM_Domain::enqueue_object_torque(const float3& rotation_center, const uchar flag_marker) { // add up torque around specified rotation_center for all cells flagged with flag_marker
	enqueue_update_force_field(); // update force field if it is not yet up-to-date
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_torque.set_parameters(2u, flag_marker, rotation_center.x, rotation_center.y, rotation_center.z).enqueue_run();
	object_sum.enqueue_read_from_device();
}
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
void LBM_Domain::enqueue_update_moving_boundaries() { // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	kernel_update_moving_boundaries.enqueue_run();
}
#endif // MOVING_BOUNDARIES
#ifdef PARTICLES
void LBM_Domain::enqueue_integrate_particles(const uint time_step_multiplicator) { // intgegrate particles forward in time and couple particles to fluid
#ifdef FORCE_FIELD
	if(particles_rho!=1.0f) kernel_reset_force_field.enqueue_run(); // only reset force field if particles have buoyancy and apply forces on fluid
	kernel_integrate_particles.set_parameters(5u, fx, fy, fz);
#endif // FORCE_FIELD
	kernel_integrate_particles.set_parameters(3u, (float)time_step_multiplicator).enqueue_run();
}
#endif // PARTICLES

void LBM_Domain::increment_time_step(const ulong steps) {
	t += steps; // increment time step
#ifdef UPDATE_FIELDS
	t_last_update_fields = t;
#endif // UPDATE_FIELDS
}
void LBM_Domain::reset_time_step() {
	t = 0ull; // increment time step
#ifdef UPDATE_FIELDS
	t_last_update_fields = t;
#endif // UPDATE_FIELDS
}
void LBM_Domain::finish_queue() {
	device.finish_queue();
}

uint LBM_Domain::get_velocity_set() const {
	return velocity_set;
}

void LBM_Domain::voxelize_mesh_on_device(const Mesh* mesh, const uchar flag, const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity) { // voxelize triangle mesh
	Memory<float3> p0(device, mesh->triangle_number, 1u, mesh->p0);
	Memory<float3> p1(device, mesh->triangle_number, 1u, mesh->p1);
	Memory<float3> p2(device, mesh->triangle_number, 1u, mesh->p2);
	Memory<float> bounding_box_and_velocity(device, 16u);
	const float x0=mesh->pmin.x-2.0f, y0=mesh->pmin.y-2.0f, z0=mesh->pmin.z-2.0f, x1=mesh->pmax.x+2.0f, y1=mesh->pmax.y+2.0f, z1=mesh->pmax.z+2.0f; // use bounding box of mesh to speed up voxelization; add tolerance of 2 cells for re-voxelization of moving objects
	bounding_box_and_velocity[ 0] = as_float(mesh->triangle_number);
	bounding_box_and_velocity[ 1] = x0;
	bounding_box_and_velocity[ 2] = y0;
	bounding_box_and_velocity[ 3] = z0;
	bounding_box_and_velocity[ 4] = x1;
	bounding_box_and_velocity[ 5] = y1;
	bounding_box_and_velocity[ 6] = z1;
	bounding_box_and_velocity[ 7] = rotation_center.x;
	bounding_box_and_velocity[ 8] = rotation_center.y;
	bounding_box_and_velocity[ 9] = rotation_center.z;
	bounding_box_and_velocity[10] = linear_velocity.x;
	bounding_box_and_velocity[11] = linear_velocity.y;
	bounding_box_and_velocity[12] = linear_velocity.z;
	bounding_box_and_velocity[13] = rotational_velocity.x;
	bounding_box_and_velocity[14] = rotational_velocity.y;
	bounding_box_and_velocity[15] = rotational_velocity.z;
	uint direction = 0u;
	if(length(rotational_velocity)==0.0f) { // choose direction of minimum bounding-box cross-section area
		float v[3] = { (y1-y0)*(z1-z0), (z1-z0)*(x1-x0), (x1-x0)*(y1-y0) };
		float vmin = v[0];
		for(uint i=1u; i<3u; i++) {
			if(v[i]<vmin) {
				vmin = v[i];
				direction = i;
			}
		}
	} else { // choose direction closest to rotation axis
		float v[3] = { fabsf(rotational_velocity.x), fabsf(rotational_velocity.y), fabsf(rotational_velocity.z) };
		float vmax = v[0];
		for(uint i=1u; i<3u; i++) {
			if(v[i]>vmax) {
				vmax = v[i];
				direction = i; // find direction of minimum bounding-box cross-section area
			}
		}
	}
	const ulong A[3] = { (ulong)Ny*(ulong)Nz, (ulong)Nz*(ulong)Nx, (ulong)Nx*(ulong)Ny };
	Kernel kernel_voxelize_mesh(device, A[direction], "voxelize_mesh", direction, fi, u, flags, t+1ull, flag, p0, p1, p2, bounding_box_and_velocity);
#ifdef SURFACE
	kernel_voxelize_mesh.add_parameters(mass, massex);
#endif // SURFACE
	p0.write_to_device();
	p1.write_to_device();
	p2.write_to_device();
	bounding_box_and_velocity.write_to_device();
	kernel_voxelize_mesh.run();
}
void LBM_Domain::voxelize_sdf_on_device(const SDF* sdf, const float3& center, const float3x3& rotation, const float scale, const uchar flag) { // voxelize SDF on GPU
	// Upload SDF data to GPU
	const ulong sdf_size = (ulong)sdf->Nx * (ulong)sdf->Ny * (ulong)sdf->Nz;
	Memory<float> sdf_data(device, sdf_size, 1u, sdf->data);

	// Compute inverse rotation matrix (to transform LBM coords back to SDF space)
	// For orthogonal matrices, inverse = transpose
	const float3x3 rotation_inv = transpose(rotation);

	// Pack parameters into array (similar to voxelize_mesh approach)
	Memory<float> sdf_params(device, 20u);
	const float3 sdf_world_size = sdf->get_world_size();
	const float sdf_max_world_dim = fmax(fmax(sdf_world_size.x, sdf_world_size.y), sdf_world_size.z);
	const float world_to_lbm = scale / sdf_max_world_dim;

	sdf_params[0] = as_float(sdf->Nx);      // pack uint as float
	sdf_params[1] = as_float(sdf->Ny);
	sdf_params[2] = as_float(sdf->Nz);
	sdf_params[3] = sdf->cell_size;
	sdf_params[4] = sdf->origin.x;          // SDF origin
	sdf_params[5] = sdf->origin.y;
	sdf_params[6] = sdf->origin.z;
	sdf_params[7] = center.x;               // LBM center
	sdf_params[8] = center.y;
	sdf_params[9] = center.z;
	sdf_params[10] = world_to_lbm;
	// Inverse rotation matrix (row-major: row0, row1, row2)
	sdf_params[11] = rotation_inv.xx; sdf_params[12] = rotation_inv.xy; sdf_params[13] = rotation_inv.xz;
	sdf_params[14] = rotation_inv.yx; sdf_params[15] = rotation_inv.yy; sdf_params[16] = rotation_inv.yz;
	sdf_params[17] = rotation_inv.zx; sdf_params[18] = rotation_inv.zy; sdf_params[19] = rotation_inv.zz;

	// Create kernel and run
	Kernel kernel_voxelize_sdf(device, get_N(), "voxelize_sdf", flags, flag, sdf_data, sdf_params);
	sdf_data.write_to_device();
	sdf_params.write_to_device();
	kernel_voxelize_sdf.run();
}
void LBM_Domain::enqueue_unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag) { // remove voxelized triangle mesh from LBM grid
	const float x0=mesh->pmin.x, y0=mesh->pmin.y, z0=mesh->pmin.z, x1=mesh->pmax.x, y1=mesh->pmax.y, z1=mesh->pmax.z; // remove all flags in bounding box of mesh
	Kernel kernel_unvoxelize_mesh(device, get_N(), "unvoxelize_mesh", flags, flag, x0, y0, z0, x1, y1, z1);
	kernel_unvoxelize_mesh.run();
}

string LBM_Domain::device_defines(const Device_Info& device_info) const { return
	"\n	#define def_Nx "+to_string(Nx)+"u"
	"\n	#define def_Ny "+to_string(Ny)+"u"
	"\n	#define def_Nz "+to_string(Nz)+"u"
	"\n	#define def_N "+to_string(get_N())+"ul"
	"\n	#define uxx "+(get_N()<=(ulong)max_uint ? "uint" : "ulong")+"" // switchable data type for index calculation (32-bit uint / 64-bit ulong)

	"\n	#define def_GNx "+to_string((Nx-2u*halo_width()*(uint)(Dx>1u))*Dx)+"u" // global LBM grid resolution of all domains together
	"\n	#define def_GNy "+to_string((Ny-2u*halo_width()*(uint)(Dy>1u))*Dy)+"u"
	"\n	#define def_GNz "+to_string((Nz-2u*halo_width()*(uint)(Dz>1u))*Dz)+"u"
	"\n	#define def_H "+to_string(halo_width())+"u" // the halo layers on each side of a domain cut (this fork: 2 with SCALAR_FV)

	"\n	#define def_Dx "+to_string(Dx)+"u"
	"\n	#define def_Dy "+to_string(Dy)+"u"
	"\n	#define def_Dz "+to_string(Dz)+"u"

	"\n	#define def_Ox "+to_string(Ox)+"" // offsets are signed integer!
	"\n	#define def_Oy "+to_string(Oy)+""
	"\n	#define def_Oz "+to_string(Oz)+""

	"\n	#define def_Ax "+to_string(Ny*Nz)+"u"
	"\n	#define def_Ay "+to_string(Nz*Nx)+"u"
	"\n	#define def_Az "+to_string(Nx*Ny)+"u"

	"\n	#define def_domain_offset_x "+to_string(0.5f*(float)((int)Nx+2*Ox+(int)Dx*(2*(int)halo_width()*(int)(Dx>1u)-(int)Nx)))+"f"
	"\n	#define def_domain_offset_y "+to_string(0.5f*(float)((int)Ny+2*Oy+(int)Dy*(2*(int)halo_width()*(int)(Dy>1u)-(int)Ny)))+"f"
	"\n	#define def_domain_offset_z "+to_string(0.5f*(float)((int)Nz+2*Oz+(int)Dz*(2*(int)halo_width()*(int)(Dz>1u)-(int)Nz)))+"f"

	"\n	#define D"+to_string(dimensions)+"Q"+to_string(velocity_set)+"" // D2Q9/D3Q15/D3Q19/D3Q27
	"\n	#define def_velocity_set "+to_string(velocity_set)+"u" // LBM velocity set (D2Q9/D3Q15/D3Q19/D3Q27)
	"\n	#define def_dimensions "+to_string(dimensions)+"u" // number spatial dimensions (2D or 3D)
	"\n	#define def_transfers "+to_string(transfers)+"u" // number of DDFs that are transferred between multiple domains

	"\n	#define def_c 0.57735027f" // lattice speed of sound c = 1/sqrt(3)*dt
	"\n	#define def_w " +to_string(1.0f/get_tau())+"f" // relaxation rate w = dt/tau = dt/(nu/c^2+dt/2) = 1/(3*nu+1/2)
#if defined(D2Q9)
	"\n	#define def_w0 (1.0f/2.25f)" // center (0)
	"\n	#define def_ws (1.0f/9.0f)" // straight (1-4)
	"\n	#define def_we (1.0f/36.0f)" // edge (5-8)
#elif defined(D3Q15)
	"\n	#define def_w0 (1.0f/4.5f)" // center (0)
	"\n	#define def_ws (1.0f/9.0f)" // straight (1-6)
	"\n	#define def_wc (1.0f/72.0f)" // corner (7-14)
#elif defined(D3Q19)
	"\n	#define def_w0 (1.0f/3.0f)" // center (0)
	"\n	#define def_ws (1.0f/18.0f)" // straight (1-6)
	"\n	#define def_we (1.0f/36.0f)" // edge (7-18)
#elif defined(D3Q27)
	"\n	#define def_w0 (1.0f/3.375f)" // center (0)
	"\n	#define def_ws (1.0f/13.5f)" // straight (1-6)
	"\n	#define def_we (1.0f/54.0f)" // edge (7-18)
	"\n	#define def_wc (1.0f/216.0f)" // corner (19-26)
#endif // D3Q27

#if defined(SRT)
	"\n	#define SRT"
#elif defined(TRT)
	"\n	#define TRT"
#endif // TRT

	"\n	#define TYPE_S 0x01" // 0b00000001 // (stationary or moving) solid boundary
	"\n	#define TYPE_E 0x02" // 0b00000010 // equilibrium boundary (inflow/outflow)
	"\n	#define TYPE_T 0x04" // 0b00000100 // temperature boundary
	"\n	#define TYPE_F 0x08" // 0b00001000 // fluid
	"\n	#define TYPE_I 0x10" // 0b00010000 // interface
	"\n	#define TYPE_G 0x20" // 0b00100000 // gas
	"\n	#define TYPE_X 0x40" // 0b01000000 // reserved type X
	"\n	#define TYPE_Y 0x80" // 0b10000000 // reserved type Y

	"\n	#define TYPE_MS 0x03" // 0b00000011 // cell next to moving solid boundary
	"\n	#define TYPE_BO 0x03" // 0b00000011 // any flag bit used for boundaries (temperature excluded)
	"\n	#define TYPE_IF 0x18" // 0b00011000 // change from interface to fluid
	"\n	#define TYPE_IG 0x30" // 0b00110000 // change from interface to gas
	"\n	#define TYPE_GI 0x38" // 0b00111000 // change from gas to interface
	"\n	#define TYPE_SU 0x38" // 0b00111000 // any flag bit used for SURFACE
	"\n	#define TYPE_XY 0xC0" // 0b11000000 // any flag bit used for X or Y markers

#if defined(FP16S)
	"\n	#define fpxx half" // switchable data type (scaled IEEE-754 16-bit floating-point format: 1-5-10, exp-30, +-1.99902344, +-1.86446416E-9, +-1.81898936E-12, 3.311 digits)
	"\n	#define fpxx_copy ushort" // switchable data type for direct copying (scaled IEEE-754 16-bit floating-point format: 1-5-10, exp-30, +-1.99902344, +-1.86446416E-9, +-1.81898936E-12, 3.311 digits)
	"\n	#define load(p,o) (vload_half(o,p)*3.0517578E-5f)" // special function for loading half
	"\n	#define store(p,o,x) vstore_half_rte((x)*32768.0f,o,p)" // special function for storing half
#elif defined(FP16C)
	"\n	#define fpxx ushort" // switchable data type (custom 16-bit floating-point format: 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits), 12.5% slower than IEEE-754 16-bit
	"\n	#define fpxx_copy ushort" // switchable data type for direct copying (custom 16-bit floating-point format: 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits), 12.5% slower than IEEE-754 16-bit
	"\n	#define load(p,o) half_to_float_custom((p)[o])" // special function for loading half
	"\n	#define store(p,o,x) (p)[o]=float_to_half_custom(x)" // special function for storing half
#else // FP32
	"\n	#define fpxx float" // switchable data type (regular 32-bit float)
	"\n	#define fpxx_copy float" // switchable data type for direct copying (regular 32-bit float)
	"\n	#define load(p,o) (p)[o]" // regular float read
	"\n	#define store(p,o,x) (p)[o]=(x)" // regular float write
#endif // FP32

#ifdef UPDATE_FIELDS
	"\n	#define UPDATE_FIELDS"
#endif // UPDATE_FIELDS

#ifdef VOLUME_FORCE
	"\n	#define VOLUME_FORCE"
#endif // VOLUME_FORCE

#ifdef MOVING_BOUNDARIES
	"\n	#define MOVING_BOUNDARIES"
#endif // MOVING_BOUNDARIES

#ifdef EQUILIBRIUM_BOUNDARIES
	"\n	#define EQUILIBRIUM_BOUNDARIES"
#endif // EQUILIBRIUM_BOUNDARIES

#ifdef FORCE_FIELD
	"\n	#define FORCE_FIELD"
#endif // FORCE_FIELD

#ifdef SURFACE
	"\n	#define SURFACE"
	"\n	#define def_6_sigma "+to_string(6.0f*sigma)+"f" // rho_laplace = 2*o*K, rho = 1-rho_laplace/c^2 = 1-(6*o)*K
#endif // SURFACE

#ifdef TEMPERATURE
	"\n	#define TEMPERATURE"
#ifdef THERMAL_DIFFUSIVITY_FIX
	"\n	#define def_w_T "+to_string(1.0f/(4.0f*alpha+0.5f))+"f" // wT = dt/tauT = 1/(4*alpha+1/2): the D3Q7 lattice diffuses with D = (tauT-1/2)/4, so alpha is the thermal diffusion coefficient (tests/physics/thermal_diffusivity.cpp)
#else // THERMAL_DIFFUSIVITY_FIX
	"\n	#define def_w_T "+to_string(1.0f/(2.0f*alpha+0.5f))+"f" // wT = dt/tauT = 1/(2*alpha+1/2), alpha = thermal diffusion coefficient; measured: the lattice diffuses with alpha/2 (tests/physics/thermal_diffusivity.cpp, eddy_heat_diffusivity.cpp)
#endif // THERMAL_DIFFUSIVITY_FIX
	"\n	#define def_beta "+to_string(beta)+"f" // thermal expansion coefficient
	"\n	#define def_T_avg "+to_string(T_avg)+"f" // average temperature
#endif // TEMPERATURE

#ifdef SUBGRID
	"\n	#define SUBGRID"
#endif // SUBGRID
#ifdef VOLUMETRIC_WALL
	"\n	#define VOLUMETRIC_WALL" // the native volumetric wall (task 19)
	"\n	#define TYPE_V TYPE_X" // the cells of the wall region R
#endif // VOLUMETRIC_WALL
#ifdef VOLUMETRIC_WALL_FP64_COLLISION
	"\n	#define VOLUMETRIC_WALL_FP64_COLLISION" // task 57: FP64 collision arithmetic in the ordinary cells, FP32 storage
#endif // VOLUMETRIC_WALL_FP64_COLLISION
#ifdef VOLUMETRIC_WALL_RR27
	"\n	#define VOLUMETRIC_WALL_RR27" // task 98: the qualified D3Q27 RR revision of the wall
#endif // VOLUMETRIC_WALL_RR27
#ifdef VOLUMETRIC_WALL_SGS
	"\n	#define VOLUMETRIC_WALL_SGS" // V2: the static Smagorinsky rate in vw_rr27 (all three call sites)
#endif // VOLUMETRIC_WALL_SGS
#ifdef VOLUMETRIC_WALL_ROUGH
	"\n	#define VOLUMETRIC_WALL_ROUGH" // V2: the facetwise rough friction in vw_wall_rr27
#endif // VOLUMETRIC_WALL_ROUGH
#ifdef VOLUMETRIC_WALL_T119
	"\n	#define VOLUMETRIC_WALL_T119" // task 119 (experimental): the whole-prism redistribution of MATHEMATICAL_MODEL section 15.11
#endif // VOLUMETRIC_WALL_T119
#ifdef VOLUMETRIC_WALL_SRD
	"\n	#define VOLUMETRIC_WALL_SRD" // task 121 (experimental): the task-120 state redistribution after the transport
#endif // VOLUMETRIC_WALL_SRD
#ifdef VOLUMETRIC_WALL_OPEN_X
	"\n	#define VOLUMETRIC_WALL_OPEN_X" // task 105: open x (E cells at x = 0 and Nx - 1, exterior facets, the LODI E state)
#endif // VOLUMETRIC_WALL_OPEN_X

#ifdef LEVEL_FORCING
	"\n	#define LEVEL_FORCING" // per-level forcing (this fork)
#endif // LEVEL_FORCING
#ifdef MOISTURE
	"\n	#define MOISTURE" // the moisture lattice (this fork)
	"\n	#define def_q_scale 0.01f" // Q = 1 + qt / def_q_scale
	"\n	#define def_T_scale 1.0f" // T = 1 + (theta_l - theta_0) / def_T_scale, in K
#endif // MOISTURE
#ifdef SCALAR_FV
	"\n	#define SCALAR_FV" // T (and Q) by finite volumes (this fork)
#endif // SCALAR_FV

#ifdef PARTICLES
	"\n	#define PARTICLES"
	"\n	#define def_particles_N "+to_string(particles_N)+"ul"
	"\n	#define def_particles_rho "+to_string(particles_rho)+"f"
#endif // PARTICLES
;}

#ifdef GRAPHICS
void LBM_Domain::Graphics::allocate(Device& device) {
	bitmap = Memory<int>(device, camera.width*camera.height);
	zbuffer = Memory<int>(device, camera.width*camera.height, 1u, lbm->get_D()>1u); // if there are multiple domains, allocate zbuffer also on host side
	camera_parameters = Memory<float>(device, 15u);
	kernel_clear = Kernel(device, bitmap.length(), "graphics_clear", bitmap, zbuffer);
	kernel_graphics_flags = Kernel(device, lbm->get_N(), "graphics_flags", camera_parameters, bitmap, zbuffer, lbm->flags);
	{
#ifndef FORCE_FIELD
		const uint cache_required = (cb(GRAPHICS_LSF+1u)* 1u+1023u)/1024u; // in KB
#else // FORCE_FIELD
		const uint cache_required = (cb(GRAPHICS_LSF+1u)*13u+1023u)/1024u; // in KB
#endif // FORCE_FIELD
		const bool enable_ls = GRAPHICS_LSF>0u&&device.info.max_workgroup_size>=cb(GRAPHICS_LSF)&&device.info.local_cache>=cache_required;
		if(GRAPHICS_LSF>0u&&!enable_ls) print_warning(device.info.name+" does not support local memory optimization with GRAPHICS_LSF = "+to_string(GRAPHICS_LSF)+" (max supported workgroup size: "+to_string(device.info.max_workgroup_size)+" (required: "+to_string(cb(GRAPHICS_LSF))+"), cache: "+to_string(device.info.local_cache)+"KB (required: "+to_string(cache_required)+"KB)). Disabling local memory optimization.");
		const ulong N = enable_ls ? (ulong)((lbm->get_Nx()+GRAPHICS_LSF-2u)/GRAPHICS_LSF)*(ulong)((lbm->get_Ny()+GRAPHICS_LSF-2u)/GRAPHICS_LSF)*(ulong)((lbm->get_Nz()+GRAPHICS_LSF-2u)/GRAPHICS_LSF)*(ulong)cb(GRAPHICS_LSF) : (ulong)(lbm->get_Nx()-1u)*(ulong)(lbm->get_Ny()-1u)*(ulong)(lbm->get_Nz()-1u);
		const uint workgroup_size = enable_ls ? cb(GRAPHICS_LSF) : WORKGROUP_SIZE;
		kernel_graphics_flags_mc = Kernel(device, N, workgroup_size, "graphics_flags_mc", camera_parameters, bitmap, zbuffer, lbm->flags);
	}
	kernel_graphics_field = Kernel(device, lbm->get_D()==1u ? camera.width*camera.height : lbm->get_N(), lbm->get_D()==1u ? "graphics_field_rt" : "graphics_field", camera_parameters, bitmap, zbuffer, 0, lbm->rho, lbm->u, lbm->flags); // raytraced field visualization only works for single-GPU
	kernel_graphics_field_slice = Kernel(device, lbm->get_N(), "graphics_field_slice", camera_parameters, bitmap, zbuffer, 0, 0, 0, 0, 0, lbm->rho, lbm->u, lbm->flags);
#ifndef D2Q9
	kernel_graphics_streamline = Kernel(device, max(lbm->get_Nx()/GRAPHICS_STREAMLINE_SPARSE, 1u)*max(lbm->get_Ny()/GRAPHICS_STREAMLINE_SPARSE, 1u)*max(lbm->get_Nz()/GRAPHICS_STREAMLINE_SPARSE, 1u), "graphics_streamline", camera_parameters, bitmap, zbuffer, 0, 0, 0, 0, 0, lbm->rho, lbm->u, lbm->flags); // 3D
#else // D2Q9
	kernel_graphics_streamline = Kernel(device, max(lbm->get_Nx()/GRAPHICS_STREAMLINE_SPARSE, 1u)*max(lbm->get_Ny()/GRAPHICS_STREAMLINE_SPARSE, 1u), "graphics_streamline", camera_parameters, bitmap, zbuffer, 0, 0, 0, 0, 0, lbm->rho, lbm->u, lbm->flags); // 2D
#endif // D2Q9
	{
		const uint cache_required = (cb(GRAPHICS_LSQ+3u)*12u+1023u)/1024u; // in KB
		const bool enable_ls = GRAPHICS_LSQ>0u&&device.info.max_workgroup_size>=cb(GRAPHICS_LSQ)&&device.info.local_cache>=cache_required;
		if(GRAPHICS_LSQ>0u&&!enable_ls) print_warning(device.info.name+" does not support local memory optimization with GRAPHICS_LSQ = "+to_string(GRAPHICS_LSQ)+" (max supported workgroup size: "+to_string(device.info.max_workgroup_size)+" (required: "+to_string(cb(GRAPHICS_LSQ))+"), cache: "+to_string(device.info.local_cache)+"KB (required: "+to_string(cache_required)+"KB)). Disabling local memory optimization.");
		const ulong N = enable_ls ? (ulong)((lbm->get_Nx()+GRAPHICS_LSQ-2u)/GRAPHICS_LSQ)*(ulong)((lbm->get_Ny()+GRAPHICS_LSQ-2u)/GRAPHICS_LSQ)*(ulong)((lbm->get_Nz()+GRAPHICS_LSQ-2u)/GRAPHICS_LSQ)*(ulong)cb(GRAPHICS_LSQ) : (ulong)(lbm->get_Nx()-1u)*(ulong)(lbm->get_Ny()-1u)*(ulong)(lbm->get_Nz()-1u);
		const uint workgroup_size = enable_ls ? cb(GRAPHICS_LSQ) : WORKGROUP_SIZE;
		kernel_graphics_q = Kernel(device, N, workgroup_size, "graphics_q", camera_parameters, bitmap, zbuffer, 0, lbm->rho, lbm->u);
	}

#ifdef FORCE_FIELD
	kernel_graphics_flags.add_parameters(lbm->F);
	kernel_graphics_flags_mc.add_parameters(lbm->F);
#endif // FORCE_FIELD

#ifdef SURFACE
	skybox = Memory<int>(device, skybox_image->width()*skybox_image->height(), 1u, skybox_image->data());
	{
		const uint cache_required = (cb(GRAPHICS_LSP+1u)*4u+1023u)/1024u; // in KB
		const bool enable_ls = GRAPHICS_LSP>0u&&device.info.max_workgroup_size>=cb(GRAPHICS_LSP)&&device.info.local_cache>=cache_required;
		if(GRAPHICS_LSP>0u&&!enable_ls) print_warning(device.info.name+" does not support local memory optimization with GRAPHICS_LSP = "+to_string(GRAPHICS_LSP)+" (max supported workgroup size: "+to_string(device.info.max_workgroup_size)+" (required: "+to_string(cb(GRAPHICS_LSP))+"), cache: "+to_string(device.info.local_cache)+"KB (required: "+to_string(cache_required)+"KB)). Disabling local memory optimization.");
		const ulong N = enable_ls ? (ulong)((lbm->get_Nx()+GRAPHICS_LSP-2u)/GRAPHICS_LSP)*(ulong)((lbm->get_Ny()+GRAPHICS_LSP-2u)/GRAPHICS_LSP)*(ulong)((lbm->get_Nz()+GRAPHICS_LSP-2u)/GRAPHICS_LSP)*(ulong)cb(GRAPHICS_LSP) : (ulong)(lbm->get_Nx()-1u)*(ulong)(lbm->get_Ny()-1u)*(ulong)(lbm->get_Nz()-1u);
		const uint workgroup_size = enable_ls ? cb(GRAPHICS_LSP) : WORKGROUP_SIZE;
		kernel_graphics_rasterize_phi = Kernel(device, N, workgroup_size, "graphics_rasterize_phi", camera_parameters, bitmap, zbuffer, lbm->phi);
	}
	kernel_graphics_raytrace_phi = Kernel(device, bitmap.length(), "graphics_raytrace_phi", camera_parameters, bitmap, skybox, lbm->phi, lbm->flags);
	kernel_graphics_q.add_parameters(lbm->flags);
#endif // SURFACE

#ifdef TEMPERATURE
	kernel_graphics_field.add_parameters(lbm->T);
	kernel_graphics_field_slice.add_parameters(lbm->T);
	kernel_graphics_streamline.add_parameters(lbm->T);
	kernel_graphics_q.add_parameters(lbm->T);
#endif // TEMPERATURE

#ifdef PARTICLES
	kernel_graphics_particles = Kernel(device, lbm->particles.length(), "graphics_particles", camera_parameters, bitmap, zbuffer, lbm->particles);
#endif // PARTICLES
}

bool LBM_Domain::Graphics::update_camera() {
	camera.update_matrix();
	bool change = false;
	for(uint i=0u; i<15u; i++) {
		const float data = camera.data(i);
		change |= (camera_parameters[i]!=data);
		camera_parameters[i] = data;
	}
	return change; // return false if camera parameters remain unchanged
}
bool LBM_Domain::Graphics::enqueue_draw_frame(const int visualization_modes, const int field_mode, const int slice_mode, const int slice_x, const int slice_y, const int slice_z, const bool visualization_change) {
	const bool camera_update = update_camera();
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
	if(!visualization_change&&!camera_update&&lbm->get_t()==t_last_rendered_frame) return false; // don't render a new frame if the scene hasn't changed since last frame
#endif // INTERACTIVE_GRAPHICS||INTERACTIVE_GRAPHICS_ASCII
	t_last_rendered_frame = lbm->get_t();
	if(camera_update) camera_parameters.enqueue_write_to_device(); // camera_parameters PCIe transfer and kernel_clear execution can happen simulataneously
	kernel_clear.enqueue_run();
	const int sx=slice_x-lbm->Ox, sy=slice_y-lbm->Oy, sz=slice_z-lbm->Oz; // subtract domain offsets
#ifdef SURFACE
	if((visualization_modes&VIS_PHI_RAYTRACE)&&lbm->get_D()==1u) kernel_graphics_raytrace_phi.enqueue_run(); // disable raytracing for multi-GPU (domain decomposition rendering doesn't work for raytracing)
	if(visualization_modes&VIS_PHI_RASTERIZE) kernel_graphics_rasterize_phi.enqueue_run();
#endif // SURFACE
	if(visualization_modes&VIS_FLAG_LATTICE) kernel_graphics_flags.enqueue_run();
	if(visualization_modes&VIS_FLAG_SURFACE) kernel_graphics_flags_mc.enqueue_run();
	if(visualization_modes&VIS_STREAMLINES) kernel_graphics_streamline.set_parameters(3u, field_mode, slice_mode, sx, sy, sz).enqueue_run();
	if(visualization_modes&VIS_Q_CRITERION) kernel_graphics_q.set_parameters(3u, field_mode).enqueue_run();
#ifdef PARTICLES
	if(visualization_modes&VIS_PARTICLES) kernel_graphics_particles.enqueue_run();
#endif // PARTICLES
	if(visualization_modes&VIS_FIELD) {
		switch(slice_mode) { // 0 (no slice), 1 (x), 2 (y), 3 (z), 4 (xz), 5 (xyz), 6 (yz), 7 (xy)
			case 0: // no slice
				kernel_graphics_field.set_parameters(3u, field_mode).enqueue_run();
				break;
			case 1: case 2: case 3: // x/y/z
				kernel_graphics_field_slice.set_ranges(lbm->get_area((uint)clamp(slice_mode-1, 0, 2))).set_parameters(3u, field_mode, slice_mode, sx, sy, sz).enqueue_run();
				break;
			case 4: // xz
				kernel_graphics_field_slice.set_ranges(lbm->get_area(0u)).set_parameters(3u, field_mode, 0u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(2u)).set_parameters(3u, field_mode, 2u+1u, sx, sy, sz).enqueue_run();
				break;
			case 5: // xyz
				kernel_graphics_field_slice.set_ranges(lbm->get_area(0u)).set_parameters(3u, field_mode, 0u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(1u)).set_parameters(3u, field_mode, 1u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(2u)).set_parameters(3u, field_mode, 2u+1u, sx, sy, sz).enqueue_run();
				break;
			case 6: // yz
				kernel_graphics_field_slice.set_ranges(lbm->get_area(1u)).set_parameters(3u, field_mode, 1u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(2u)).set_parameters(3u, field_mode, 2u+1u, sx, sy, sz).enqueue_run();
				break;
			case 7: // xy
				kernel_graphics_field_slice.set_ranges(lbm->get_area(0u)).set_parameters(3u, field_mode, 0u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(1u)).set_parameters(3u, field_mode, 1u+1u, sx, sy, sz).enqueue_run();
				break;
		}
	}
	bitmap.enqueue_read_from_device();
	if(lbm->get_D()>1u) zbuffer.enqueue_read_from_device();
	return true; // new frame has been rendered
}
int* LBM_Domain::Graphics::get_bitmap() { // returns pointer to zbuffer
	return bitmap.data();
}
int* LBM_Domain::Graphics::get_zbuffer() { // returns pointer to zbuffer
	return zbuffer.data();
}

string LBM_Domain::Graphics::device_defines(const Device_Info& device_info) const { return
	"\n	#define GRAPHICS"
	"\n	#define def_background_color " +to_string(GRAPHICS_BACKGROUND_COLOR)+""
	"\n	#define def_screen_width "     +to_string(camera.width)+"u"
	"\n	#define def_screen_height "    +to_string(camera.height)+"u"
	"\n	#define def_scale_u "          +to_string(1.0f/(0.57735027f*(GRAPHICS_U_MAX)))+"f"
	"\n	#define def_scale_rho "        +to_string(0.5f/(GRAPHICS_RHO_DELTA))+"f"
	"\n	#define def_scale_T "          +to_string(0.5f/(GRAPHICS_T_DELTA))+"f"
	"\n	#define def_scale_F "          +to_string(0.5f/(GRAPHICS_F_MAX))+"f"
	"\n	#define def_scale_Q_min "      +to_string(GRAPHICS_Q_CRITERION)+"f"
	"\n	#define def_streamline_sparse "+to_string(GRAPHICS_STREAMLINE_SPARSE)+"u"
	"\n	#define def_streamline_length "+to_string(GRAPHICS_STREAMLINE_LENGTH)+"u"
	"\n	#define def_n "                +to_string(1.333f)+"f" // refractive index of water for raytracing graphics
	"\n	#define def_attenuation "      +to_string(ln(clamp(GRAPHICS_RAYTRACING_TRANSMITTANCE, 1E-9f, 1.0f))/(float)max(max(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz()))+"f" // (negative) attenuation parameter for raytracing graphics
	"\n	#define def_absorption_color " +to_string(GRAPHICS_RAYTRACING_COLOR)+"" // absorption color of fluid for raytracing graphics

	"\n	#define COLOR_S (127<<16|127<<8|127)" // (stationary or moving) solid boundary
	"\n	#define COLOR_E (  0<<16|255<<8|  0)" // equilibrium boundary (inflow/outflow)
	"\n	#define COLOR_M (255<<16|  0<<8|255)" // cells next to moving solid boundary
	"\n	#define COLOR_T (255<<16|  0<<8|  0)" // temperature boundary
	"\n	#define COLOR_F (  0<<16|  0<<8|255)" // fluid
	"\n	#define COLOR_I (  0<<16|255<<8|255)" // interface
	"\n	#define COLOR_0 (127<<16|127<<8|127)" // regular cell or gas
	"\n	#define COLOR_X (255<<16|127<<8|  0)" // reserved type X
	"\n	#define COLOR_Y (255<<16|255<<8|  0)" // reserved type Y
	"\n	#define COLOR_P (255<<16|255<<8|191)" // particles

#ifdef GRAPHICS_TRANSPARENCY
	"\n	#define GRAPHICS_TRANSPARENCY "+to_string(GRAPHICS_TRANSPARENCY)+"f"
#endif // GRAPHICS_TRANSPARENCY

#ifndef SURFACE
	"\n	#define def_skybox_width 1u"
	"\n	#define def_skybox_height 1u"
#else // SURFACE
	"\n	#define def_skybox_width " +to_string(skybox_image->width() )+"u"
	"\n	#define def_skybox_height "+to_string(skybox_image->height())+"u"
#endif // SURFACE

#ifndef FORCE_FIELD
	"\n	#define LSF "+to_string((GRAPHICS_LSF>0u&&device_info.max_workgroup_size>=cb(GRAPHICS_LSF)&&device_info.local_cache>=(cb(GRAPHICS_LSF+1u)* 1u+1023u)/1024u) ? GRAPHICS_LSF : 0u)+"u" // local box size for graphics_flags_mc() kernel (default: 4)
#else // FORCE_FIELD
	"\n	#define LSF "+to_string((GRAPHICS_LSF>0u&&device_info.max_workgroup_size>=cb(GRAPHICS_LSF)&&device_info.local_cache>=(cb(GRAPHICS_LSF+1u)*13u+1023u)/1024u) ? GRAPHICS_LSF : 0u)+"u" // local box size for graphics_flags_mc() kernel (default: 4)
#endif // FORCE_FIELD
	"\n	#define LSQ "+to_string((GRAPHICS_LSQ>0u&&device_info.max_workgroup_size>=cb(GRAPHICS_LSQ)&&device_info.local_cache>=(cb(GRAPHICS_LSQ+3u)*12u+1023u)/1024u) ? GRAPHICS_LSQ : 0u)+"u" // local box size for graphics_q() kernel (default: 8)
	"\n	#define LSP "+to_string((GRAPHICS_LSP>0u&&device_info.max_workgroup_size>=cb(GRAPHICS_LSP)&&device_info.local_cache>=(cb(GRAPHICS_LSP+1u)* 4u+1023u)/1024u) ? GRAPHICS_LSP : 0u)+"u" // local box size for graphics_rasterize_phi() kernel (default: 4)
;}
#endif // GRAPHICS



vector<Device_Info> smart_device_selection(const uint D) {
	const vector<Device_Info>& devices = get_devices(); // a vector of all available OpenCL devices
	vector<Device_Info> device_infos(D);
	const int user_specified_devices = (int)main_arguments.size();
	if(user_specified_devices>0) { // user has selevted specific devices as command line arguments
		if(user_specified_devices==D) { // as much specified devices as domains
			for(uint d=0; d<D; d++) device_infos[d] = select_device_with_id(to_uint(main_arguments[d]), devices); // use list of devices IDs specified by user
		} else {
			print_warning("Incorrect number of devices specified. Using single fastest device for all domains.");
			for(uint d=0; d<D; d++) device_infos[d] = select_device_with_most_flops(devices);
		}
	} else { // device auto-selection
		vector<vector<Device_Info>> device_type_ids; // a vector of all different devices, containing vectors of their device IDs
		for(uint i=0u; i<(uint)devices.size(); i++) {
			const string name_i = devices[i].name;
			bool already_exists = false;
			for(uint j=0u; j<(uint)device_type_ids.size(); j++) {
				const string name_j = device_type_ids[j][0].name;
				if(name_i==name_j) {
					device_type_ids[j].push_back(devices[i]);
					already_exists = true;
				}
			}
			if(!already_exists) device_type_ids.push_back(vector<Device_Info>(1, devices[i]));
		}
		float best_value = -1.0f;
		int best_j = -1;
		for(uint j=0u; j<(uint)device_type_ids.size(); j++) {
			const float value = device_type_ids[j][0].tflops;
			if((uint)device_type_ids[j].size()>=D && value>best_value) {
				best_value = value;
				best_j = j;
			}
		}
		if(best_j>=0) { // select all devices of fastest device type with at least D devices of the same type
			for(uint d=0; d<D; d++) device_infos[d] = device_type_ids[best_j][d];
		} else {
			print_warning("Not enough devices of the same type available. Using single fastest device for all domains.");
			for(uint d=0; d<D; d++) device_infos[d] = select_device_with_most_flops(devices);
		}
		//for(uint j=0u; j<(uint)device_type_ids.size(); j++) print_info("Device Type "+to_string(j)+" ("+device_type_ids[j][0].name+"): "+to_string((uint)device_type_ids[j].size())+"x");
	}
	return device_infos;
}

LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) // single device
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho)
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, fx, fy, fz, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const uint particles_N, const float particles_rho)
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, Dx, Dy, Dz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) // single device
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, fx, fy, fz, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // multiple devices
	const uint NDx=(Nx/Dx)*Dx, NDy=(Ny/Dy)*Dy, NDz=(Nz/Dz)*Dz; // make resolution equally divisible by domains
	if(NDx!=Nx||NDy!=Ny||NDz!=Nz) print_warning("LBM grid ("+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+") is not equally divisible in domains ("+to_string(Dx)+"x"+to_string(Dy)+"x"+to_string(Dz)+"). Changing resolution to ("+to_string(NDx)+"x"+to_string(NDy)+"x"+to_string(NDz)+").");
	this->Nx = NDx; this->Ny = NDy; this->Nz = NDz;
	this->Dx = Dx; this->Dy = Dy; this->Dz = Dz;
	const uint D = Dx*Dy*Dz;
	const uint Hx=(Dx>1u)*halo_width(), Hy=(Dy>1u)*halo_width(), Hz=(Dz>1u)*halo_width(); // halo offsets
	const vector<Device_Info>& device_infos = smart_device_selection(D);
	sanity_checks_constructor(device_infos, this->Nx, this->Ny, this->Nz, Dx, Dy, Dz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	lbm_domain = new LBM_Domain*[D];
	for(uint d=0u; d<D; d++) { // parallel_for((ulong)D, D, [&](ulong d) {
		const uint x=((uint)d%(Dx*Dy))%Dx, y=((uint)d%(Dx*Dy))/Dx, z=(uint)d/(Dx*Dy); // d = x+(y+z*Dy)*Dx
		lbm_domain[d] = new LBM_Domain(device_infos[d], this->Nx/Dx+2u*Hx, this->Ny/Dy+2u*Hy, this->Nz/Dz+2u*Hz, Dx, Dy, Dz, (int)(x*this->Nx/Dx)-(int)Hx, (int)(y*this->Ny/Dy)-(int)Hy, (int)(z*this->Nz/Dz)-(int)Hz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	} // });
	{
		Memory<float>** buffers_rho = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_rho[d] = &(lbm_domain[d]->rho);
		rho = Memory_Container(this, buffers_rho, "rho");
	} {
		Memory<float>** buffers_u = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_u[d] = &(lbm_domain[d]->u);
		u = Memory_Container(this, buffers_u, "u");
	} {
		Memory<uchar>** buffers_flags = new Memory<uchar>*[D];
		for(uint d=0u; d<D; d++) buffers_flags[d] = &(lbm_domain[d]->flags);
		flags = Memory_Container(this, buffers_flags, "flags");
	} {
#ifdef FORCE_FIELD
		Memory<float>** buffers_F = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_F[d] = &(lbm_domain[d]->F);
		F = Memory_Container(this, buffers_F, "F");
#endif // FORCE_FIELD
	} {
#ifdef SURFACE
		Memory<float>** buffers_phi = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_phi[d] = &(lbm_domain[d]->phi);
		phi = Memory_Container(this, buffers_phi, "phi");
#endif // SURFACE
	} {
#ifdef TEMPERATURE
		Memory<float>** buffers_T = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_T[d] = &(lbm_domain[d]->T);
		T = Memory_Container(this, buffers_T, "T");
#endif // TEMPERATURE
#ifdef MOISTURE
		Memory<float>** buffers_Q = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_Q[d] = &(lbm_domain[d]->Q);
		Q = Memory_Container(this, buffers_Q, "Q");
#endif // MOISTURE
	} {
#ifdef PARTICLES
		particles = &(lbm_domain[0]->particles);
#endif // PARTICLES
	}
#ifdef GRAPHICS
	graphics = Graphics(this);
#endif // GRAPHICS
}
LBM::~LBM() {
#ifdef GRAPHICS
	camera.allow_rendering = false;
#endif // GRAPHICS
	info.print_finalize();
	for(uint d=0u; d<get_D(); d++) delete lbm_domain[d];
	delete[] lbm_domain;
}

void LBM::sanity_checks_constructor(const vector<Device_Info>& device_infos, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // sanity checks on grid resolution and extension support
#ifdef VOLUMETRIC_WALL
	if(Dx*Dy*Dz!=1u) print_error("VOLUMETRIC_WALL requires one domain (task 19 initial scope)");
#endif // VOLUMETRIC_WALL
	if((ulong)Nx*(ulong)Ny*(ulong)Nz==0ull) print_error("Grid point number is 0: "+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+" = 0.");
	if(Dx*Dy*Dz==0u) print_error("You specified 0 LBM grid domains ("+to_string(Dx)+"x"+to_string(Dy)+"x"+to_string(Dz)+"). There has to be at least 1 domain in every direction. Check your input in LBM constructor.");
	const uint local_Nx=Nx/Dx+2u*(Dx>1u), local_Ny=Ny/Dy+2u*(Dy>1u), local_Nz=Nz/Dz+2u*(Dz>1u);
	uint memory_available = max_uint; // in MB
	for(Device_Info device_info : device_infos) memory_available = min(memory_available, device_info.memory);
	uint memory_required = (uint)((ulong)Nx*(ulong)Ny*(ulong)Nz/((ulong)(Dx*Dy*Dz))*(ulong)bytes_per_cell_device()/1048576ull); // in MB
	if(memory_required>memory_available) {
		float factor = cbrt((float)memory_available/(float)memory_required);
		const uint maxNx=(uint)(factor*(float)Nx), maxNy=(uint)(factor*(float)Ny), maxNz=(uint)(factor*(float)Nz);
		string message = "Grid resolution ("+to_string(Nx)+", "+to_string(Ny)+", "+to_string(Nz)+") is too large: "+to_string(Dx*Dy*Dz)+"x "+to_string(memory_required)+" MB required, "+to_string(Dx*Dy*Dz)+"x "+to_string(memory_available)+" MB available. Largest possible resolution is ("+to_string(maxNx)+", "+to_string(maxNy)+", "+to_string(maxNz)+"). Restart the simulation with lower resolution or on different device(s) with more memory.";
#if !defined(FP16S)&&!defined(FP16C)
		uint memory_required_fp16 = (uint)((ulong)Nx*(ulong)Ny*(ulong)Nz/((ulong)(Dx*Dy*Dz))*(ulong)(bytes_per_cell_device()-velocity_set*2u)/1048576ull); // in MB
		float factor_fp16 = cbrt((float)memory_available/(float)memory_required_fp16);
		const uint maxNx_fp16=(uint)(factor_fp16*(float)Nx), maxNy_fp16=(uint)(factor_fp16*(float)Ny), maxNz_fp16=(uint)(factor_fp16*(float)Nz);
		message += " Consider using FP16S/FP16C memory compression to double maximum grid resolution to a maximum of ("+to_string(maxNx_fp16)+", "+to_string(maxNy_fp16)+", "+to_string(maxNz_fp16)+"); for this, uncomment \"#define FP16S\" or \"#define FP16C\" in defines.hpp.";
#endif // !FP16S&&!FP16C
		print_error(message);
	}
	if(nu==0.0f) print_error("Viscosity cannot be 0. Change it in setup.cpp."); // sanity checks for viscosity
	else if(nu<0.0f) print_error("Viscosity cannot be negative. Remove the \"-\" in setup.cpp.");
#ifdef D2Q9
	if(Nz!=1u) print_error("D2Q9 is the 2D velocity set. You have to set Nz=1u in the LBM constructor! Currently you have set Nz="+to_string(Nz)+"u.");
#endif // D2Q9
#if !defined(SRT)&&!defined(TRT)
	print_error("No LBM collision operator selected. Uncomment either \"#define SRT\" or \"#define TRT\" in defines.hpp");
#elif defined(SRT)&&defined(TRT)
	print_error("Too many LBM collision operators selected. Comment out either \"#define SRT\" or \"#define TRT\" in defines.hpp");
#endif // SRT && TRT
#ifndef VOLUME_FORCE
	if(fx!=0.0f||fy!=0.0f||fz!=0.0f) print_error("Volume force is set in LBM constructor in main_setup(), but VOLUME_FORCE is not enabled. Uncomment \"#define VOLUME_FORCE\" in defines.hpp.");
#else // VOLUME_FORCE
#ifndef FORCE_FIELD
	if(fx==0.0f&&fy==0.0f&&fz==0.0f) print_warning("The VOLUME_FORCE extension is enabled but the volume force in LBM constructor is set to zero. You may disable the extension by commenting out \"#define VOLUME_FORCE\" in defines.hpp.");
#endif // FORCE_FIELD
#endif // VOLUME_FORCE
#ifndef SURFACE
	if(sigma!=0.0f) print_error("Surface tension is set in LBM constructor in main_setup(), but SURFACE is not enabled. Uncomment \"#define SURFACE\" in defines.hpp.");
#endif // SURFACE
#ifndef TEMPERATURE
	if(alpha!=0.0f||beta!=0.0f) print_error("Thermal diffusion/expansion coefficients are set in LBM constructor in main_setup(), but TEMPERATURE is not enabled. Uncomment \"#define TEMPERATURE\" in defines.hpp.");
#else // TEMPERATURE
	if(alpha==0.0f&&beta==0.0f) print_warning("The TEMPERATURE extension is enabled but the thermal diffusion/expansion coefficients alpha/beta in the LBM constructor are both set to zero. You may disable the extension by commenting out \"#define TEMPERATURE\" in defines.hpp.");
#endif // TEMPERATURE
#ifdef PARTICLES
	if(particles_N==0u) print_error("The PARTICLES extension is enabled but the number of particles is set to 0. Comment out \"#define PARTICLES\" in defines.hpp.");
#if !defined(VOLUME_FORCE)||!defined(FORCE_FIELD)
	if(particles_rho!=1.0f) print_error("Particle density is set unequal to 1, but particle-fluid 2-way-coupling is not enabled. Uncomment both \"#define VOLUME_FORCE\" and \"#define FORCE_FIELD\" in defines.hpp.");
#endif // !VOLUME_FORCE||!FORCE_FIELD
#ifdef FORCE_FIELD
	if(particles_rho==1.0f) print_warning("Particle density is set to 1, so particles behave as passive tracers without acting a force on the fluid, but particle-fluid 2-way-coupling is enabled. You may comment out \"#define FORCE_FIELD\" in defines.hpp.");
#endif // FORCE_FIELD
#else // PARTICLES
	if(particles_N>0u) print_error("The PARTICLES extension is disabled but the number of particles is set to "+to_string(particles_N)+">0. Uncomment \"#define PARTICLES\" in defines.hpp.");
#endif // PARTICLES
}

void LBM::sanity_checks_initialization() { // sanity checks during initialization on used extensions based on used flags
	uchar flags_used = 0u;
	bool moving_boundaries_used=false, equilibrium_boundaries_used=false, surface_used=false, temperature_used=false; // identify used extensions based used flags
	const uint threads = thread::hardware_concurrency();
	vector<uchar> t_flags_used(threads, 0u);
	vector<char> t_moving_boundaries_used(threads, false); // don't use vector<bool> as it uses bit-packing which is broken for multithreading
	vector<char> t_equilibrium_boundaries_used(threads, false); // don't use vector<bool> as it uses bit-packing which is broken for multithreading
	parallel_for(get_N(), threads, [&](ulong n, uint t) {
		const uchar flagsn = flags[n];
		const uchar flagsn_bo = flagsn&(TYPE_S|TYPE_E);
		t_flags_used[t] = t_flags_used[t]|flagsn;
		if(flagsn_bo&TYPE_S) t_moving_boundaries_used[t] = t_moving_boundaries_used[t] || (((flagsn_bo==TYPE_S)&&(u.x[n]!=0.0f||u.y[n]!=0.0f||u.z[n]!=0.0f))||(flagsn_bo==(TYPE_S|TYPE_E)));
		t_equilibrium_boundaries_used[t] = t_equilibrium_boundaries_used[t] || flagsn_bo==TYPE_E;
	});
	for(uint t=0u; t<threads; t++) {
		flags_used = flags_used|t_flags_used[t];
		moving_boundaries_used = moving_boundaries_used || t_moving_boundaries_used[t];
		equilibrium_boundaries_used = equilibrium_boundaries_used || t_equilibrium_boundaries_used[t];
	}
	surface_used = (bool)(flags_used&(TYPE_F|TYPE_I|TYPE_G));
	temperature_used = (bool)(flags_used&TYPE_T);
#ifndef MOVING_BOUNDARIES
	if(moving_boundaries_used) print_warning("Some boundary cells have non-zero velocity, but MOVING_BOUNDARIES is not enabled. If you intend to use moving boundaries, uncomment \"#define MOVING_BOUNDARIES\" in defines.hpp.");
#else // MOVING_BOUNDARIES
	if(!moving_boundaries_used) print_warning("The MOVING_BOUNDARIES extension is enabled but no moving boundary cells (TYPE_S flag and velocity unequal to zero) are placed in the simulation box. You may disable the extension by commenting out \"#define MOVING_BOUNDARIES\" in defines.hpp.");
#endif // MOVING_BOUNDARIES
#ifndef EQUILIBRIUM_BOUNDARIES
	if(equilibrium_boundaries_used) print_error("Some cells are set as equilibrium boundaries with the TYPE_E flag, but EQUILIBRIUM_BOUNDARIES is not enabled. Uncomment \"#define EQUILIBRIUM_BOUNDARIES\" in defines.hpp.");
#else // EQUILIBRIUM_BOUNDARIES
	if(!equilibrium_boundaries_used) print_warning("The EQUILIBRIUM_BOUNDARIES extension is enabled but no equilibrium boundary cells (TYPE_E flag) are placed in the simulation box. You may disable the extension by commenting out \"#define EQUILIBRIUM_BOUNDARIES\" in defines.hpp.");
#endif // EQUILIBRIUM_BOUNDARIES
#ifndef SURFACE
	if(surface_used) print_error("Some cells are set as fluid/interface/gas with the TYPE_F/TYPE_I/TYPE_G flags, but SURFACE is not enabled. Uncomment \"#define SURFACE\" in defines.hpp.");
#else // SURFACE
	if(!surface_used) print_error("The SURFACE extension is enabled but no fluid/interface/gas cells (TYPE_F/TYPE_I/TYPE_G flags) are placed in the simulation box. Disable the extension by commenting out \"#define SURFACE\" in defines.hpp.");
#endif // SURFACE
#ifndef TEMPERATURE
	if(temperature_used) print_error("Some cells are set as temperature boundary with the TYPE_T flag, but TEMPERATURE is not enabled. Uncomment \"#define TEMPERATURE\" in defines.hpp.");
#endif // TEMPERATURE
}

void LBM::initialize() { // write all data fields to device and call kernel_initialize
#ifndef BENCHMARK
	sanity_checks_initialization();
#endif // BENCHMARK
#ifdef SCALAR_FV
	{ // the 2-layer exchange extracts each domain's first 2 interior layers: a domain needs at least 2 halo widths of interior cells along a split axis
		const uint need = 2u*halo_width();
		if((Dx>1u&&Nx/Dx<need)||(Dy>1u&&Ny/Dy<need)||(Dz>1u&&Nz/Dz<need)) print_error("SCALAR_FV in domains: each domain needs at least "+to_string(need)+" interior cells along a split axis (here "+to_string(Nx/Dx)+" x "+to_string(Ny/Dy)+" x "+to_string(Nz/Dz)+").");
	}
#endif // SCALAR_FV

	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->rho.enqueue_write_to_device();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->u.enqueue_write_to_device();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->flags.enqueue_write_to_device();
#ifdef FORCE_FIELD
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->F.enqueue_write_to_device();
	communicate_F();
#endif // FORCE_FIELD
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->phi.enqueue_write_to_device();
#endif // SURFACE
#ifdef TEMPERATURE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->T.enqueue_write_to_device();
#endif // TEMPERATURE
#ifdef MOISTURE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->Q.enqueue_write_to_device(); // the start moisture field: the initialize kernel sets hi from it, as gi from T
#endif // MOISTURE
#ifdef SCALAR_FV
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_fv_init(); // the shifted fields from the fields just written (SCALAR_FV)
#endif // SCALAR_FV
#ifdef PARTICLES
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->particles.enqueue_write_to_device();
	communicate_particles();
#endif // PARTICLES

	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step(); // the communicate calls at initialization need an odd time step
	communicate_rho_u_flags();
#ifdef SURFACE
	communicate_phi_massex_flags();
#endif // SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_initialize(); // odd time step is baked-in the kernel
#ifdef VOLUMETRIC_WALL
	if(lbm_domain[0]->volumetric_wall_enabled()) lbm_domain[0]->enqueue_volumetric_wall_initial_state(); // declared level-0 pre-collision state into the t = 0 load sets of all fluid cells (R: supplied f0; ordinary: native equilibrium); initialize()'s post-collision-of-step-(-1) seeding is replaced, never silently kept
#endif // VOLUMETRIC_WALL
	communicate_rho_u_flags();
#ifdef SURFACE
	communicate_phi_massex_flags();
#endif // SURFACE
	communicate_fi(); // time step must be odd here
#ifdef TEMPERATURE
	communicate_T(); // T halo data is required for field_slice rendering
#ifndef SCALAR_FV // with SCALAR_FV, gi and hi are one-cell placeholders: nothing to exchange
	communicate_gi(); // time step must be odd here
#ifdef MOISTURE
	communicate_hi();
#endif // MOISTURE
#endif // SCALAR_FV
#endif // TEMPERATURE
#ifdef SCALAR_FV
	communicate_flags_fv(); // fv_solid() reads the flags 2 cells along a link: both halo layers
	communicate_fv(); // fv_init wrote the halo cells from fields the host left empty there: their values from the cells they mirror
#endif // SCALAR_FV
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->reset_time_step(); // set time step to 0 again
	initialized = true;
}

void LBM::do_time_step() { // call kernel_stream_collide to perform one LBM time step
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_0();
#endif // SURFACE
#ifdef VOLUMETRIC_WALL
	if(lbm_domain[0]->volumetric_wall_enabled()) lbm_domain[0]->enqueue_volumetric_wall_load(); // level-t states of R (reads only R address sets), before the ordinary kernel
#endif // VOLUMETRIC_WALL
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_stream_collide(); // run LBM stream_collide kernel after domain communication
#ifdef VOLUMETRIC_WALL
	if(lbm_domain[0]->volumetric_wall_enabled()) lbm_domain[0]->enqueue_volumetric_wall_step(); // collide, wall (all Q), then stream + pending on the in-order queue
#endif // VOLUMETRIC_WALL
#if defined(SURFACE) || defined(GRAPHICS)
	communicate_rho_u_flags(); // rho/u/flags halo data is required for SURFACE extension, and u halo data is required for Q-criterion rendering
#endif // SURFACE || GRAPHICS
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_1();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_2();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_3();
	communicate_phi_massex_flags();
#endif // SURFACE
	communicate_fi();
#ifdef TEMPERATURE
#ifdef GRAPHICS
	communicate_T(); // T halo data is required for field_slice rendering
#endif // GRAPHICS
#ifndef SCALAR_FV
	communicate_gi();
#ifdef MOISTURE
	communicate_hi();
#endif // MOISTURE
#endif // SCALAR_FV
#endif // TEMPERATURE
#ifdef SCALAR_FV
	// the finite volumes after the flow: a cell next to a cut reads its neighbour domain's link fluxes (fi, just
	// exchanged), diffusivity and values up to 2 cells across the cut, so each stage follows an exchange
	communicate_fvD();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_fv_half();
	communicate_fv_half();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_fv_full();
	communicate_fv();
#endif // SCALAR_FV
#ifdef PARTICLES
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_integrate_particles(); // intgegrate particles forward in time and couple particles to fluid
	communicate_particles(); // communicate_F() is not required in do_time_step()
#endif // PARTICLES
	if(get_D()==1u) for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // this additional domain synchronization barrier is only required in single-GPU, as communication calls already provide all necessary synchronization barriers in multi-GPU
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step();
}

void LBM_Domain::read_fi_raw(vector<fpxx>& host) { // blocking raw copy of fi (every build; no kernel involved)
	host.resize(fi.range());
	device.finish_queue();
	device.get_cl_queue().enqueueReadBuffer(fi.get_cl_buffer(), CL_TRUE, 0ull, fi.capacity(), (void*)host.data());
}

void LBM::run(const ulong steps, const ulong total_steps) { // initializes the LBM simulation (copies data to device and runs initialize kernel), then runs LBM
	info.append(steps, total_steps, get_t()); // total_steps parameter is just for runtime estimation
#ifdef VOLUMETRIC_WALL_RR27
	if(!lbm_domain[0]->volumetric_wall_enabled()) print_error("VOLUMETRIC_WALL_RR27: run() without an attached wall is refused (the RR collision of this build differs from the baseline; attach with set_volumetric_wall() before run(), or use a build without the macro)"); // task 98 (#1991)
#endif // VOLUMETRIC_WALL_RR27
#ifdef VOLUMETRIC_WALL_OPEN_X
	if(lbm_domain[0]->get_fx()!=0.0f||lbm_domain[0]->get_fy()!=0.0f||lbm_domain[0]->get_fz()!=0.0f) print_error("VOLUMETRIC_WALL_OPEN_X: run() with a nonzero force is refused (the open boundaries drive the flow, contract A5)"); // task 105
#endif // VOLUMETRIC_WALL_OPEN_X
	if(!initialized) {
		initialize();
		info.print_initialize(this); // only print setup info if the setup is new (run() was not called before)
#ifdef GRAPHICS
		camera.allow_rendering = true;
#endif // GRAPHICS
	}
	Clock clock;
	for(ulong i=1ull; i<=steps; i++) {
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
		while(!key_P&&running) sleep(0.016);
		if(!running) break;
#endif // INTERACTIVE_GRAPHICS_ASCII || INTERACTIVE_GRAPHICS
		clock.start();
		do_time_step();
		info.update(clock.stop());
	}
	if(get_D()>1u) for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // wait for everything to finish (multi-GPU only)
#ifdef VOLUMETRIC_WALL_RR27
	{ // #2034: the run is invalid when any facet had a non-positive or non-finite wall density, or a velocity reached the clamp limit, since the attachment. Detected HERE, at the end of run() (one read of two counters), not at the failing step; the state is not clamped or repaired.
		uint wall_state_failures = 0u, clamps = 0u;
		lbm_domain[0]->volumetric_wall_failure_counts(wall_state_failures, clamps);
		if(wall_state_failures>0u||clamps>0u) print_error("VOLUMETRIC_WALL_RR27: invalid run: "+to_string(wall_state_failures)+" invalid wall states (rho_w <= 0 or not finite) and "+to_string(clamps)+" velocity clamps since the attachment (detected at the end of run(), t = "+to_string(get_t())+")");
	}
#endif // VOLUMETRIC_WALL_RR27
}

#ifdef SCALAR_FV
void LBM_Domain::read_fv_inventory(double& sum_rho_theta, double& sum_rho_fv) { // the conserved inventory from the device (this fork)
	const ulong N = get_N();
	Memory<float> rho_theta(device, N, 1u, true, true, 0.0f); // host copies of the device-only fields
	Memory<float> rho_minus_1(device, N, 1u, true, true, 0.0f);
	device.get_cl_queue().enqueueCopyBuffer(fvRT.get_cl_buffer(), rho_theta.get_cl_buffer(), 0ull, 0ull, N*sizeof(float));
	device.get_cl_queue().enqueueCopyBuffer(fvR.get_cl_buffer(), rho_minus_1.get_cl_buffer(), 0ull, 0ull, N*sizeof(float));
	rho_theta.read_from_device();
	rho_minus_1.read_from_device();
	flags.read_from_device();
	sum_rho_theta = 0.0;
	sum_rho_fv = 0.0;
	for(ulong n=0ull; n<N; n++) {
		if(flags[n]&TYPE_S) continue;
		sum_rho_theta += (double)rho_theta[n];
		sum_rho_fv += 1.0+(double)rho_minus_1[n];
	}
}
#endif // SCALAR_FV

void LBM::update_fields() { // update fields (rho, u, T) manually
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_fields();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}

void LBM::reset() { // reset simulation (takes effect in following run() call)
	initialized = false;
}

#ifdef FORCE_FIELD
void LBM::update_force_field() { // calculate forces from fluid on TYPE_S cells
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_force_field();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}
float3 LBM::object_center_of_mass(const uchar flag_marker) { // calculate center of mass of all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_center_of_mass(flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_com = float3(0.0f, 0.0f, 0.0f);
	ulong object_cells = 0ull;
	for(uint d=0u; d<get_D(); d++) {
		object_com += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
		object_cells += (ulong)as_uint(lbm_domain[d]->object_sum.w[0]);
	}
	return object_com/(float)object_cells;
}
float3 LBM::object_force(const uchar flag_marker) { // add up force for all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_force(flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_force = float3(0.0f, 0.0f, 0.0f);
	for(uint d=0u; d<get_D(); d++) object_force += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
	return object_force;
}
float3 LBM::object_torque(const float3& rotation_center, const uchar flag_marker) { // add up torque around specified rotation center for all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_torque(rotation_center, flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_torque = float3(0.0f, 0.0f, 0.0f);
	for(uint d=0u; d<get_D(); d++) object_torque += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
	return object_torque;
}
#endif // FORCE_FIELD

#ifdef MOVING_BOUNDARIES
void LBM::update_moving_boundaries() { // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_moving_boundaries();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
#ifdef GRAPHICS
	camera.key_update = true; // to prevent flickering of flags in interactive graphics when camera is not moved
#endif // GRAPHICS
}
#endif // MOVING_BOUNDARIES

#if defined(PARTICLES)&&!defined(FORCE_FIELD)
void LBM::integrate_particles(const ulong steps, const ulong total_steps, const uint time_step_multiplicator) { // intgegrate passive tracer particles forward in time in stationary flow field
	info.append(steps, total_steps, get_t());
	Clock clock;
	for(ulong i=1ull; i<=steps; i+=(ulong)time_step_multiplicator) {
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
		while(!key_P&&running) sleep(0.016);
		if(!running) break;
#endif // INTERACTIVE_GRAPHICS_ASCII || INTERACTIVE_GRAPHICS
		clock.start();
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_integrate_particles(time_step_multiplicator);
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step(time_step_multiplicator);
		info.update(clock.stop());
	}
}
#endif // PARTICLES&&!FORCE_FIELD

void LBM::write_status(const string& path) { // write LBM status report to a .txt file
	string status = "";
	status += "Grid Resolution = "+to_string(Nx)+" x "+to_string(Ny)+" x "+to_string(Nz)+" = "+to_string(get_N())+"\n";
	status += "Grid Domains = "+to_string(Dx)+" x "+to_string(Dy)+" x "+to_string(Dz)+" = "+to_string(get_D())+"\n";
	status += "LBM Type = D"+string(get_velocity_set()==9 ? "2" : "3")+"Q"+to_string(get_velocity_set())+" "+info.collision+"\n";
	status += "Memory Usage = CPU "+to_string(info.cpu_mem_required)+" MB, GPU "+to_string(get_D())+"x "+to_string(info.gpu_mem_required)+" MB\n";
	status += "Maximum Allocation Size = "+to_string((uint)(get_N()/(ulong)get_D()*(ulong)(get_velocity_set()*sizeof(fpxx))/1048576ull))+" MB\n";
	status += "Time Steps = "+to_string(get_t())+" / "+(info.steps==max_ulong ? "infinite" : to_string(info.steps))+"\n";
	status += "Runtime = "+print_time(info.runtime_total)+" (total) = "+print_time(info.runtime_lbm)+" (LBM) + "+print_time(info.runtime_total-info.runtime_lbm)+" (rendering and data evaluation)\n";
	status += "Average MLUPs/s = "+to_string(to_uint(1E-6*(double)get_N()*(double)get_t()/info.runtime_lbm))+"\n";
	status += "Kinematic Viscosity = "+to_string(get_nu())+"\n";
	status += "Relaxation Time = "+to_string(get_tau())+"\n";
	status += "Maximum Reynolds Number = "+to_string(get_Re_max())+"\n";
#ifdef VOLUME_FORCE
	status += "Volume Force = ("+to_string(get_fx())+", "+to_string(get_fy())+", "+to_string(get_fz())+")\n";
#endif // VOLUME_FORCE
#ifdef SURFACE
	status += "Surface Tension Coefficient = "+to_string(get_sigma())+"\n";
#endif // SURFACE
#ifdef TEMPERATURE
	status += "Thermal Diffusion Coefficient = "+to_string(get_alpha())+"\n";
	status += "Thermal Expansion Coefficient = "+to_string(get_beta())+"\n";
#endif // TEMPERATURE
	const string filename = default_filename(path, "status", ".txt", get_t());
	write_file(filename, status);
}

void LBM::voxelize_mesh_on_device(const Mesh* mesh, const uchar flag, const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity) { // voxelize triangle mesh
	if(get_D()==1u) {
		lbm_domain[0]->voxelize_mesh_on_device(mesh, flag, rotation_center, linear_velocity, rotational_velocity); // if this crashes on Windows, create a TdrDelay 32-bit DWORD with decimal value 300 in Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
	} else {
		parallel_for(get_D(), get_D(), [&](uint d) {
			lbm_domain[d]->voxelize_mesh_on_device(mesh, flag, rotation_center, linear_velocity, rotational_velocity);
		});
	}
#ifdef MOVING_BOUNDARIES
	if((flag&(TYPE_S|TYPE_E))==TYPE_S&&(length(linear_velocity)>0.0f||length(rotational_velocity)>0.0f)) update_moving_boundaries();
#endif // MOVING_BOUNDARIES
	if(!initialized) {
		flags.read_from_device();
		u.read_from_device();
	}
}
void LBM::unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag) { // remove voxelized triangle mesh from LBM grid by removing all flags in mesh bounding box (only required when bounding box size changes during re-voxelization)
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_unvoxelize_mesh_on_device(mesh, flag);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}
void LBM::write_mesh_to_vtk(const Mesh* mesh, const string& path, const bool convert_to_si_units) const { // write mesh to binary .vtk file
	const string filename = default_filename(path, "mesh", ".vtk", get_t());
	const string header_1 = "# vtk DataFile Version 3.0\nFluidX3D "+filename.substr(filename.rfind('/')+1)+"\nBINARY\nDATASET POLYDATA\nPOINTS "+to_string(3u*mesh->triangle_number)+" float\n";
	const string header_2 = "POLYGONS "+to_string(mesh->triangle_number)+" "+to_string(4u*mesh->triangle_number)+"\n";
	float* points = new float[9u*mesh->triangle_number];
	int* triangles = new int[4u*mesh->triangle_number];
	const float spacing = convert_to_si_units ? units.si_x(1.0f) : 1.0f;
	const float3 offset = center();
	parallel_for(mesh->triangle_number, [&](uint i) {
		points[9u*i   ] = reverse_bytes(spacing*(mesh->p0[i].x-offset.x));
		points[9u*i+1u] = reverse_bytes(spacing*(mesh->p0[i].y-offset.y));
		points[9u*i+2u] = reverse_bytes(spacing*(mesh->p0[i].z-offset.z));
		points[9u*i+3u] = reverse_bytes(spacing*(mesh->p1[i].x-offset.x));
		points[9u*i+4u] = reverse_bytes(spacing*(mesh->p1[i].y-offset.y));
		points[9u*i+5u] = reverse_bytes(spacing*(mesh->p1[i].z-offset.z));
		points[9u*i+6u] = reverse_bytes(spacing*(mesh->p2[i].x-offset.x));
		points[9u*i+7u] = reverse_bytes(spacing*(mesh->p2[i].y-offset.y));
		points[9u*i+8u] = reverse_bytes(spacing*(mesh->p2[i].z-offset.z));
		triangles[4u*i   ] = reverse_bytes(3); // 3 vertices per triangle
		triangles[4u*i+1u] = reverse_bytes(3*(int)i  ); // vertex 0
		triangles[4u*i+2u] = reverse_bytes(3*(int)i+1); // vertex 1
		triangles[4u*i+3u] = reverse_bytes(3*(int)i+2); // vertex 2
	});
	create_folder(filename);
	std::ofstream file(filename, std::ios::out|std::ios::binary);
	file.write(header_1.c_str(), header_1.length()); // write non-binary file header
	file.write((char*)points, 4u*9u*mesh->triangle_number); // write binary data
	file.write(header_2.c_str(), header_2.length()); // write non-binary file header
	file.write((char*)triangles, 4u*4u*mesh->triangle_number); // write binary data
	file.close();
	delete[] points;
	delete[] triangles;
	info.allow_printing.lock();
	print_info("File \""+filename+"\" saved.");
	info.allow_printing.unlock();
}
void LBM::voxelize_stl(const string& path, const float3& center, const float3x3& rotation, const float size, const uchar flag) { // voxelize triangle mesh
	const Mesh* mesh = read_stl(path, this->size(), center, rotation, size);
	flags.write_to_device();
	voxelize_mesh_on_device(mesh, flag);
	delete mesh;
	flags.read_from_device();
}
void LBM::voxelize_stl(const string& path, const float3x3& rotation, const float size, const uchar flag) { // read and voxelize binary .stl file (place in box center)
	voxelize_stl(path, center(), rotation, size, flag);
}
void LBM::voxelize_stl(const string& path, const float3& center, const float size, const uchar flag) { // read and voxelize binary .stl file (no rotation)
	voxelize_stl(path, center, float3x3(1.0f), size, flag);
}
void LBM::voxelize_stl(const string& path, const float size, const uchar flag) { // read and voxelize binary .stl file (place in box center, no rotation)
	voxelize_stl(path, center(), float3x3(1.0f), size, flag);
}

void LBM::voxelize_sdf(const string& path, const float3& center, const float3x3& rotation, const float scale, const uchar flag) { // voxelize SDF
	const SDF* sdf = read_sdf(path);
	if(sdf == nullptr) return; // Error already printed by read_sdf

	flags.write_to_device();

	print_info("Voxelizing SDF on GPU...");
	print_info("LBM grid: " + to_string(get_Nx()) + " x " + to_string(get_Ny()) + " x " + to_string(get_Nz()));

	// Voxelize on GPU using kernel
	for(uint d = 0u; d < get_D(); d++) {
		lbm_domain[d]->voxelize_sdf_on_device(sdf, center, rotation, scale, flag);
	}
	flags.read_from_device();

	delete sdf;

	// Count total solid voxels
	uint total_solid = 0;
	for(ulong n = 0; n < get_N(); n++) {
		if(flags[n] == flag) total_solid++;
	}

	flags.write_to_device(); // Push the modified flags back to GPU
	print_info("SDF voxelization complete. Total solid voxels: " + to_string(total_solid));
}
void LBM::voxelize_sdf(const string& path, const float3x3& rotation, const float scale, const uchar flag) { // read and voxelize binary SDF file (place in box center)
	voxelize_sdf(path, center(), rotation, scale, flag);
}
void LBM::voxelize_sdf(const string& path, const float3& center, const float scale, const uchar flag) { // read and voxelize binary SDF file (no rotation)
	voxelize_sdf(path, center, float3x3(1.0f), scale, flag);
}
void LBM::voxelize_sdf(const string& path, const float scale, const uchar flag) { // read and voxelize binary SDF file (place in box center, no rotation)
	voxelize_sdf(path, center(), float3x3(1.0f), scale, flag);
}

#ifdef GRAPHICS
int* LBM::Graphics::draw_frame() {
#ifndef UPDATE_FIELDS
	if(visualization_modes&(VIS_FIELD|VIS_STREAMLINES|VIS_Q_CRITERION)) {
		for(uint d=0u; d<lbm->get_D(); d++) lbm->lbm_domain[d]->enqueue_update_fields(); // only call update_fields() if the time step has changed since the last rendered frame
	}
#endif // UPDATE_FIELDS
	if(key_1) { visualization_modes = (visualization_modes&~0b11)|(((visualization_modes&0b11)+1)%4); key_1 = false; }
	if(key_2) { visualization_modes ^= VIS_FIELD        ; key_2 = false; }
	if(key_3) { visualization_modes ^= VIS_STREAMLINES  ; key_3 = false; }
	if(key_4) { visualization_modes ^= VIS_Q_CRITERION  ; key_4 = false; }
	if(key_5) { visualization_modes ^= VIS_PHI_RASTERIZE; key_5 = false; }
	if(key_6) { visualization_modes ^= VIS_PHI_RAYTRACE ; key_6 = false; }
	if(key_7) { visualization_modes ^= VIS_PARTICLES    ; key_7 = false; }
	if(key_T) {
		slice_mode = (slice_mode+1)%8; key_T = false;
	}
	if(key_Z) {
#ifndef TEMPERATURE
		field_mode = (field_mode+1)%2; key_Z = false; // field_mode = { 0 (u), 1 (rho) }
#else // TEMPERATURE
		field_mode = (field_mode+1)%3; key_Z = false; // field_mode = { 0 (u), 1 (rho), 2 (T) }
#endif // TEMPERATURE
	}
	if(slice_mode==1u) {
		if(key_Q) { slice_x = clamp(slice_x-1, 0, (int)lbm->get_Nx()-1); key_Q = false; }
		if(key_E) { slice_x = clamp(slice_x+1, 0, (int)lbm->get_Nx()-1); key_E = false; }
	}
	if(slice_mode==2u) {
		if(key_Q) { slice_y = clamp(slice_y-1, 0, (int)lbm->get_Ny()-1); key_Q = false; }
		if(key_E) { slice_y = clamp(slice_y+1, 0, (int)lbm->get_Ny()-1); key_E = false; }
	}
	if(slice_mode==3u) {
		if(key_Q) { slice_z = clamp(slice_z-1, 0, (int)lbm->get_Nz()-1); key_Q = false; }
		if(key_E) { slice_z = clamp(slice_z+1, 0, (int)lbm->get_Nz()-1); key_E = false; }
	}
	const bool visualization_change = camera.key_update||last_visualization_modes!=visualization_modes||last_field_mode!=field_mode||last_slice_mode!=slice_mode||last_slice_x!=slice_x||last_slice_y!=slice_y||last_slice_z!=slice_z;
	camera.key_update = false;
	last_visualization_modes = visualization_modes;
	last_field_mode = field_mode;
	last_slice_mode = slice_mode;
	last_slice_x = slice_x;
	last_slice_y = slice_y;
	last_slice_z = slice_z;
	bool new_frame = true;
	for(uint d=0u; d<lbm->get_D(); d++) new_frame = new_frame && lbm->lbm_domain[d]->graphics.enqueue_draw_frame(visualization_modes, field_mode, slice_mode, slice_x, slice_y, slice_z, visualization_change);
	for(uint d=0u; d<lbm->get_D(); d++) lbm->lbm_domain[d]->finish_queue();
	int* bitmap = lbm->lbm_domain[0]->graphics.get_bitmap();
	int* zbuffer = lbm->lbm_domain[0]->graphics.get_zbuffer();
	for(uint d=1u; d<lbm->get_D()&&new_frame; d++) {
		const int* const bitmap_d = lbm->lbm_domain[d]->graphics.get_bitmap(); // each domain renders its own frame
		const int* const zbuffer_d = lbm->lbm_domain[d]->graphics.get_zbuffer();
		for(uint i=0u; i<camera.width*camera.height; i++) {
#ifndef GRAPHICS_TRANSPARENCY
			const int zdi = zbuffer_d[i];
			if(zdi>zbuffer[i]) {
				bitmap[i] = bitmap_d[i]; // overlay frames using their z-buffers
				zbuffer[i] = zdi;
			}
#else // GRAPHICS_TRANSPARENCY
			bitmap[i] = color_add(bitmap[i], bitmap_d[i]);
#endif // GRAPHICS_TRANSPARENCY
		}
	}
	camera.allow_labeling = new_frame; // only print new label on frame if a new frame has been rendered
	return bitmap;
}

void LBM::Graphics::set_camera_centered(const float rx, const float ry, const float fov, const float zoom) {
	camera.free = false;
	camera.rx = 0.5*pi+((double)rx*pi/180.0);
	camera.ry = pi-((double)ry*pi/180.0);
	camera.fov = clamp((float)fov, 1E-6f, 179.0f);
	camera.set_zoom(0.5f*(float)fmax(fmax(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz())/zoom);
}
void LBM::Graphics::set_camera_free(const float3& p, const float rx, const float ry, const float fov) {
	camera.free = true;
	camera.rx = 0.5*pi+((double)rx*pi/180.0);
	camera.ry = pi-((double)ry*pi/180.0);
	camera.fov = clamp((float)fov, 1E-6f, 179.0f);
	camera.zoom = 1E16f;
	camera.pos = p;
}
bool LBM::Graphics::next_frame(const ulong total_time_steps, const float video_length_seconds) { // returns true once simulation time has progressed enough to render the next video frame for a 60fps video of specified length
	const uint new_frame = to_uint((float)lbm->get_t()/(float)total_time_steps*video_length_seconds*60.0f);
	if(new_frame!=last_exported_frame) {
		last_exported_frame = new_frame;
		return true;
	} else {
		return false;
	}
}
void LBM::Graphics::print_frame() { // preview current frame in console
#ifndef INTERACTIVE_GRAPHICS_ASCII
	camera.rendring_frame.lock(); // block rendering for other threads until finished
	camera.key_update = true; // force rendering new frame
	int* image_data = draw_frame(); // make sure the frame is fully rendered
	Image* image = new Image(camera.width, camera.height, image_data);
	info.allow_printing.lock();
	println();
	print_image(image);
	info.allow_printing.unlock();
	delete image;
	camera.rendring_frame.unlock();
#endif // INTERACTIVE_GRAPHICS_ASCII
}
void encode_image(Image* image, const string& filename, const string& extension, std::atomic_int* running_encoders) {
	if(extension==".png") write_png(filename, image);
	if(extension==".qoi") write_qoi(filename, image);
	if(extension==".bmp") write_bmp(filename, image);
	delete image; // delete image when done
	(*running_encoders)--;
}
void LBM::Graphics::write_frame(const string& path, const string& name, const string& extension, bool print_preview) { // save current frame as .png file (smallest file size, but slow)
	write_frame(0u, 0u, camera.width, camera.height, path, name, extension, print_preview);
}
void LBM::Graphics::write_frame(const uint x1, const uint y1, const uint x2, const uint y2, const string& path, const string& name, const string& extension, bool print_preview) { // save a cropped current frame with two corner points (x1,y1) and (x2,y2)
	camera.rendring_frame.lock(); // block rendering for other threads until finished
	camera.key_update = true; // force rendering new frame
	int* image_data = draw_frame(); // make sure the frame is fully rendered
	const string filename = default_filename(path, name, extension, lbm->get_t());
	const uint xa=max(min(x1, x2), 0u), xb=min(max(x1, x2), camera.width ); // sort coordinates if necessary
	const uint ya=max(min(y1, y2), 0u), yb=min(max(y1, y2), camera.height);
	Image* image = new Image(xb-xa, yb-ya); // create local copy of frame buffer
	for(uint y=0u; y<image->height(); y++) for(uint x=0u; x<image->width(); x++) image->set_color(x, y, image_data[camera.width*(ya+y)+(xa+x)]);
#ifndef INTERACTIVE_GRAPHICS_ASCII
	if(print_preview) {
		info.allow_printing.lock();
		println();
		print_image(image);
		print_info("Image \""+filename+"\" saved.");
		info.allow_printing.unlock();
	}
#endif // INTERACTIVE_GRAPHICS_ASCII
	running_encoders++;
	thread encoder(encode_image, image, filename, extension, &running_encoders); // the main bottleneck in rendering images to the hard disk is .png encoding, so encode image in new thread
	encoder.detach(); // detatch thread so it can run concurrently
	camera.rendring_frame.unlock();
}
void LBM::Graphics::write_frame_png(const string& path, bool print_preview) { // save current frame as .png file (smallest file size, but slow)
	write_frame(path, "image", ".png", print_preview);
}
void LBM::Graphics::write_frame_qoi(const string& path, bool print_preview) { // save current frame as .qoi file (small file size, fast)
	write_frame(path, "image", ".qoi", print_preview);
}
void LBM::Graphics::write_frame_bmp(const string& path, bool print_preview) { // save current frame as .bmp file (large file size, fast)
	write_frame(path, "image", ".bmp", print_preview);
}
void LBM::Graphics::write_frame_png(const uint x1, const uint y1, const uint x2, const uint y2, const string& path, bool print_preview) { // save current frame as .png file (smallest file size, but slow)
	write_frame(x1, y1, x2, y2, path, "image", ".png", print_preview);
}
void LBM::Graphics::write_frame_qoi(const uint x1, const uint y1, const uint x2, const uint y2, const string& path, bool print_preview) { // save current frame as .qoi file (small file size, fast)
	write_frame(x1, y1, x2, y2, path, "image", ".qoi", print_preview);
}
void LBM::Graphics::write_frame_bmp(const uint x1, const uint y1, const uint x2, const uint y2, const string& path, bool print_preview) { // save current frame as .bmp file (large file size, fast)
	write_frame(x1, y1, x2, y2, path, "image", ".bmp", print_preview);
}
#endif // GRAPHICS



void LBM_Domain::allocate_transfer(Device& device) { // allocate all memory for multi-device trqansfer
	ulong Amax = 0ull; // maximum domain side area of communicated directions
	if(Dx>1u) Amax = max(Amax, (ulong)Ny*(ulong)Nz); // Ax
	if(Dy>1u) Amax = max(Amax, (ulong)Nz*(ulong)Nx); // Ay
	if(Dz>1u) Amax = max(Amax, (ulong)Nx*(ulong)Ny); // Az

	uint bytes_per_area_cell = max(transfers*(uint)sizeof(fpxx), 17u); // fi, or rho, u and flags
#ifdef SCALAR_FV
	bytes_per_area_cell = max(bytes_per_area_cell, fv_transfer_bytes()); // theta and q over 2 halo layers
#endif // SCALAR_FV
	transfer_buffer_p = Memory<char>(device, Amax, bytes_per_area_cell, true, true, 0, false); // only allocate one set of transfer buffers in plus/minus directions, for all x/y/z transfers
	transfer_buffer_m = Memory<char>(device, Amax, bytes_per_area_cell, true, true, 0, false); // these transfer buffers must not be zero-copy!

	kernel_transfer[enum_transfer_field::fi              ][0] = Kernel(device, 0ull, "transfer_extract_fi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fi);
	kernel_transfer[enum_transfer_field::fi              ][1] = Kernel(device, 0ull, "transfer__insert_fi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fi);
	kernel_transfer[enum_transfer_field::rho_u_flags     ][0] = Kernel(device, 0ull, "transfer_extract_rho_u_flags"     , 0u, t, transfer_buffer_p, transfer_buffer_m, rho, u, flags);
	kernel_transfer[enum_transfer_field::rho_u_flags     ][1] = Kernel(device, 0ull, "transfer__insert_rho_u_flags"     , 0u, t, transfer_buffer_p, transfer_buffer_m, rho, u, flags);
	kernel_transfer[enum_transfer_field::flags           ][0] = Kernel(device, 0ull, "transfer_extract_flags"           , 0u, t, transfer_buffer_p, transfer_buffer_m, flags);
	kernel_transfer[enum_transfer_field::flags           ][1] = Kernel(device, 0ull, "transfer__insert_flags"           , 0u, t, transfer_buffer_p, transfer_buffer_m, flags);
#ifdef FORCE_FIELD
	kernel_transfer[enum_transfer_field::F               ][0] = Kernel(device, 0ull, "transfer_extract_F"               , 0u, t, transfer_buffer_p, transfer_buffer_m, F);
	kernel_transfer[enum_transfer_field::F               ][1] = Kernel(device, 0ull, "transfer__insert_F"               , 0u, t, transfer_buffer_p, transfer_buffer_m, F);
#endif // FORCE_FIELD
#ifdef SURFACE
	kernel_transfer[enum_transfer_field::phi_massex_flags][0] = Kernel(device, 0ull, "transfer_extract_phi_massex_flags", 0u, t, transfer_buffer_p, transfer_buffer_m, phi, massex, flags);
	kernel_transfer[enum_transfer_field::phi_massex_flags][1] = Kernel(device, 0ull, "transfer__insert_phi_massex_flags", 0u, t, transfer_buffer_p, transfer_buffer_m, phi, massex, flags);
#endif // SURFACE
#ifdef TEMPERATURE
	kernel_transfer[enum_transfer_field::gi              ][0] = Kernel(device, 0ull, "transfer_extract_gi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, gi);
	kernel_transfer[enum_transfer_field::gi              ][1] = Kernel(device, 0ull, "transfer__insert_gi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, gi);
#ifdef MOISTURE
	kernel_transfer[enum_transfer_field::hi              ][0] = Kernel(device, 0ull, "transfer_extract_hi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, hi);
	kernel_transfer[enum_transfer_field::hi              ][1] = Kernel(device, 0ull, "transfer__insert_hi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, hi);
#endif // MOISTURE
	kernel_transfer[enum_transfer_field::T               ][0] = Kernel(device, 0ull, "transfer_extract_T"               , 0u, t, transfer_buffer_p, transfer_buffer_m, T);
	kernel_transfer[enum_transfer_field::T               ][1] = Kernel(device, 0ull, "transfer__insert_T"               , 0u, t, transfer_buffer_p, transfer_buffer_m, T);
#endif // TEMPERATURE
#ifdef SCALAR_FV
	kernel_transfer[enum_transfer_field::fv              ][0] = Kernel(device, 0ull, "transfer_extract_fv"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fvT);
	kernel_transfer[enum_transfer_field::fv              ][1] = Kernel(device, 0ull, "transfer__insert_fv"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fvT);
	kernel_transfer[enum_transfer_field::fv_half         ][0] = Kernel(device, 0ull, "transfer_extract_fv"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fvT_half);
	kernel_transfer[enum_transfer_field::fv_half         ][1] = Kernel(device, 0ull, "transfer__insert_fv"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fvT_half);
#ifdef MOISTURE
	kernel_transfer[enum_transfer_field::fv              ][0].add_parameters(fvQ);
	kernel_transfer[enum_transfer_field::fv              ][1].add_parameters(fvQ);
	kernel_transfer[enum_transfer_field::fv_half         ][0].add_parameters(fvQ_half);
	kernel_transfer[enum_transfer_field::fv_half         ][1].add_parameters(fvQ_half);
#endif // MOISTURE
	kernel_transfer[enum_transfer_field::fvD             ][0] = Kernel(device, 0ull, "transfer_extract_fvD"             , 0u, t, transfer_buffer_p, transfer_buffer_m, fvD);
	kernel_transfer[enum_transfer_field::fvD             ][1] = Kernel(device, 0ull, "transfer__insert_fvD"             , 0u, t, transfer_buffer_p, transfer_buffer_m, fvD);
	kernel_transfer[enum_transfer_field::flags_fv        ][0] = Kernel(device, 0ull, "transfer_extract_flags_fv"        , 0u, t, transfer_buffer_p, transfer_buffer_m, flags);
	kernel_transfer[enum_transfer_field::flags_fv        ][1] = Kernel(device, 0ull, "transfer__insert_flags_fv"        , 0u, t, transfer_buffer_p, transfer_buffer_m, flags);
#endif // SCALAR_FV
}

ulong LBM_Domain::get_area(const uint direction) {
	const ulong A[3] = { (ulong)Ny*(ulong)Nz, (ulong)Nz*(ulong)Nx, (ulong)Nx*(ulong)Ny };
	return A[direction];
}
void LBM_Domain::enqueue_transfer_extract_field(Kernel& kernel_transfer_extract_field, const uint direction, const uint bytes_per_cell) {
	kernel_transfer_extract_field.set_ranges(get_area(direction)); // direction: x=0, y=1, z=2
	kernel_transfer_extract_field.set_parameters(0u, direction, get_t()).enqueue_run(); // selective in-VRAM copy
	transfer_buffer_p.enqueue_read_from_device(0ull, kernel_transfer_extract_field.range()*(ulong)bytes_per_cell); // PCIe copy (+)
	transfer_buffer_m.enqueue_read_from_device(0ull, kernel_transfer_extract_field.range()*(ulong)bytes_per_cell); // PCIe copy (-)
}
void LBM_Domain::enqueue_transfer_insert_field(Kernel& kernel_transfer_insert_field, const uint direction, const uint bytes_per_cell) {
	kernel_transfer_insert_field.set_ranges(get_area(direction)); // direction: x=0, y=1, z=2
	transfer_buffer_p.enqueue_write_to_device(0ull, kernel_transfer_insert_field.range()*(ulong)bytes_per_cell); // PCIe copy (+)
	transfer_buffer_m.enqueue_write_to_device(0ull, kernel_transfer_insert_field.range()*(ulong)bytes_per_cell); // PCIe copy (-)
	kernel_transfer_insert_field.set_parameters(0u, direction, get_t()).enqueue_run(); // selective in-VRAM copy
}
void LBM::communicate_field(const enum_transfer_field field, const uint bytes_per_cell) {
	if(Dx>1u) { // communicate in x-direction
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_transfer_extract_field(lbm_domain[d]->kernel_transfer[field][0], 0u, bytes_per_cell); // selective in-VRAM copy (x) + PCIe copy
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
		for(uint d=0u; d<get_D(); d++) {
			const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy), dxp=((x+1u)%Dx)+(y+z*Dy)*Dx; // d = x+(y+z*Dy)*Dx
			lbm_domain[d]->transfer_buffer_p.exchange_host_buffer(lbm_domain[dxp]->transfer_buffer_m.exchange_host_buffer(lbm_domain[d]->transfer_buffer_p.data())); // CPU pointer swaps
		}
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]-> enqueue_transfer_insert_field(lbm_domain[d]->kernel_transfer[field][1], 0u, bytes_per_cell); // PCIe copy + selective in-VRAM copy (x)
	}
	if(Dy>1u) { // communicate in y-direction
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_transfer_extract_field(lbm_domain[d]->kernel_transfer[field][0], 1u, bytes_per_cell); // selective in-VRAM copy (y) + PCIe copy
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
		for(uint d=0u; d<get_D(); d++) {
			const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy), dyp=x+(((y+1u)%Dy)+z*Dy)*Dx; // d = x+(y+z*Dy)*Dx
			lbm_domain[d]->transfer_buffer_p.exchange_host_buffer(lbm_domain[dyp]->transfer_buffer_m.exchange_host_buffer(lbm_domain[d]->transfer_buffer_p.data())); // CPU pointer swaps
		}
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]-> enqueue_transfer_insert_field(lbm_domain[d]->kernel_transfer[field][1], 1u, bytes_per_cell); // PCIe copy + selective in-VRAM copy (y)
	}
	if(Dz>1u) { // communicate in z-direction
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_transfer_extract_field(lbm_domain[d]->kernel_transfer[field][0], 2u, bytes_per_cell); // selective in-VRAM copy (z) + PCIe copy
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
		for(uint d=0u; d<get_D(); d++) {
			const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy), dzp=x+(y+((z+1u)%Dz)*Dy)*Dx; // d = x+(y+z*Dy)*Dx
			lbm_domain[d]->transfer_buffer_p.exchange_host_buffer(lbm_domain[dzp]->transfer_buffer_m.exchange_host_buffer(lbm_domain[d]->transfer_buffer_p.data())); // CPU pointer swaps
		}
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]-> enqueue_transfer_insert_field(lbm_domain[d]->kernel_transfer[field][1], 2u, bytes_per_cell); // PCIe copy + selective in-VRAM copy (z)
	}
}

void LBM::communicate_fi() {
	communicate_field(enum_transfer_field::fi, transfers*sizeof(fpxx));
}
void LBM::communicate_rho_u_flags() {
	communicate_field(enum_transfer_field::rho_u_flags, 17u);
}
void LBM::communicate_flags() {
	communicate_field(enum_transfer_field::flags, 1u);
}
#ifdef FORCE_FIELD
void LBM::communicate_F() {
	communicate_field(enum_transfer_field::F, 12u);
}
#endif // FORCE_FIELD
#ifdef SURFACE
void LBM::communicate_phi_massex_flags() {
	communicate_field(enum_transfer_field::phi_massex_flags, 9u);
}
#endif // SURFACE
#ifdef TEMPERATURE
void LBM::communicate_gi() {
	communicate_field(enum_transfer_field::gi, sizeof(fpxx));
}
#ifdef MOISTURE
void LBM::communicate_hi() {
	communicate_field(enum_transfer_field::hi, sizeof(fpxx));
}
#endif // MOISTURE
void LBM::communicate_T() {
	communicate_field(enum_transfer_field::T, 4u);
}
#endif // TEMPERATURE
#ifdef SCALAR_FV
void LBM::communicate_fv() {
	communicate_field(enum_transfer_field::fv, fv_transfer_bytes());
}
void LBM::communicate_fv_half() {
	communicate_field(enum_transfer_field::fv_half, fv_transfer_bytes());
}
void LBM::communicate_fvD() {
	communicate_field(enum_transfer_field::fvD, 4u);
}
void LBM::communicate_flags_fv() {
	communicate_field(enum_transfer_field::flags_fv, halo_width());
}
#endif // SCALAR_FV
#ifdef PARTICLES
void LBM::communicate_particles() {
	if(get_D()>1u) {
		if(initialized) {
			for(uint d=0u; d<get_D(); d++) lbm_domain[d]->particles.enqueue_read_from_device();
			for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
			for(ulong n=0ull; n<lbm_domain[0]->particles.length(); n++) { // parallel_for(lbm_domain[0]->particles.length(), [&](ulong n) {
				for(uint d=1u; d<get_D(); d++) { // gather modified particle positions
					const float lbm_domain_d___particles_x_n_ = lbm_domain[d]->particles.x[n];
					if(as_uint(lbm_domain_d___particles_x_n_)!=0xFFFFFFFFu) { // particle was in domain d and has been modified
						lbm_domain[0]->particles.x[n] = lbm_domain_d___particles_x_n_;
						lbm_domain[0]->particles.y[n] = lbm_domain[d]->particles.y[n];
						lbm_domain[0]->particles.z[n] = lbm_domain[d]->particles.z[n];
						break; // particle can only be in one domain at a time, no need to check other domains once it has been found
					}
				}
			} // });
		}
		for(uint d=0u; d<get_D(); d++) { // broadcast unified particle positions, using pointer of lbm_domain[0] instead of memory copy
			float* lbm_domain_d_particles_data = lbm_domain[d]->particles.exchange_host_buffer(lbm_domain[0]->particles.data());
			lbm_domain[d]->particles.enqueue_write_to_device();
			lbm_domain[d]->particles.exchange_host_buffer(lbm_domain_d_particles_data);
		}
	}
}
#endif // PARTICLES

#ifdef VOLUMETRIC_WALL
// ---------------------------------------------------------------- Task 19: native volumetric wall, host side
#include <stdexcept>
#include <algorithm>
#include <map>
#include <fstream>
#include <cstdlib>
#include <cmath>
namespace {
template<class T> void vw_upload(Device& device, Memory<T>& memory, const std::vector<T>& values) { // host vector -> new device buffer (at least one element)
	const ulong count = (ulong)std::max((size_t)1u, values.size());
	memory = Memory<T>(device, count);
	for(size_t i=0u; i<values.size(); i++) memory[(ulong)i] = values[i];
	memory.write_to_device();
}
template<class T> std::vector<T> vw_download(Memory<T>& memory) { // blocking device -> host copy, values unchecked (evidence)
	memory.read_from_device();
	std::vector<T> values((size_t)memory.range());
	for(ulong i=0ull; i<memory.range(); i++) values[(size_t)i] = memory[i];
	return values;
}
}
void LBM_Domain::set_volumetric_wall(const fluidx3d::experimental::VolumetricWallNativeInput& input, const fluidx3d::experimental::VolumetricWallEmbedding& embedding, const std::vector<double>& friction_rows) {
	using namespace fluidx3d::experimental;
	if(device.info.is_fp64_capable==0u) throw std::runtime_error("VOLUMETRIC_WALL needs an FP64-capable device (cl_khr_fp64)");
#ifndef VOLUMETRIC_WALL_ROUGH
	if(!input.facet_friction.empty()) throw std::invalid_argument("the container has a facet_friction section (the facetwise rough law); this build has no VOLUMETRIC_WALL_ROUGH and would ignore it"); // V2: never run a rough container with the scalar K
#endif // VOLUMETRIC_WALL_ROUGH
#ifndef VOLUMETRIC_WALL_T119
	if(input.t119()) throw std::invalid_argument("the container has task-119 sections (t119_gdelta, t119_wmoment); this build has no VOLUMETRIC_WALL_T119 and would ignore them");
#else // VOLUMETRIC_WALL_T119
	if(!input.t119()) throw std::invalid_argument("this build has VOLUMETRIC_WALL_T119 but the container has no t119_gdelta/t119_wmoment sections");
#endif // VOLUMETRIC_WALL_T119
#ifndef VOLUMETRIC_WALL_SRD
	if(input.has_srd()) throw std::invalid_argument("the container has task-121 sections (t121_*); this build has no VOLUMETRIC_WALL_SRD and would ignore them");
#else // VOLUMETRIC_WALL_SRD
	if(!input.has_srd()) throw std::invalid_argument("this build has VOLUMETRIC_WALL_SRD but the container has no t121_* sections");
#endif // VOLUMETRIC_WALL_SRD
	if(!std::isfinite(input.friction)||input.friction<0.0) throw std::invalid_argument("VOLUMETRIC_WALL friction K must be finite and >= 0 (task 21: nonzero K is the registered finite-friction scope; the kernel term -K vi q_i odd_i is executed as written)");
	if((float)input.viscosity!=nu) throw std::invalid_argument("container viscosity "+to_string(input.viscosity)+" differs from the LBM nu "+to_string(nu)+" under the FP32 conversion (#273)"); // declared FP32 equivalence, no tolerance
	// Task 45 (#1015): the former check float32(container omega) == 1.0f/get_tau() is removed. It compared two
	// roundings of the same declared viscosity: the container's omega = 1/(3 nu + 1/2) in FP64 cast to float, and the
	// engine's float expression. They need not be the same float (they differ by one unit at nu = 1e-5), so the check
	// refused valid input. What binds the two rates is kept: float32(container viscosity) == the LBM nu exactly (above),
	// and the container's own omega = 1/(3 nu + 1/2) within 1e-12 (VolumetricWallNativeInput::validate). The ordinary
	// cells then relax with 1.0f/get_tau() and the wall cells with the container's FP64 omega, as before; no rate is changed.
	if(!(input.omega>0.0&&input.omega<2.0)) throw std::invalid_argument("container omega "+to_string(input.omega)+" is not inside (0, 2)");
	const uint q = input.velocity_set; // task 98: the container's velocity set must be the build's
	if(q!=velocity_set) throw std::invalid_argument("the container is D3Q"+to_string(q)+", the build is D3Q"+to_string(velocity_set));
#ifdef VOLUMETRIC_WALL_RR27
	if(!input.affine()) throw std::invalid_argument("VOLUMETRIC_WALL_RR27 needs the affine container sections (arms, gradient map) and the friction rows (task 98)");
#endif // VOLUMETRIC_WALL_RR27
	const std::vector<std::int64_t>& rcells = embedding.r_cells();
	const std::vector<std::int64_t>& rindex = embedding.r_index_of_cell();
	const NativeDirections& dir = embedding.directions();
	vw_rcells = (uint)rcells.size(); vw_facets = input.facets; vw_rows = input.rows; vw_owned = (uint)embedding.owned().size();
	const auto remap_cell = [&](const uint cell) { const std::int64_t r = rindex[cell]; if(r<0) throw std::invalid_argument("a map references a cell outside R"); return (uint)r; };
	std::vector<ulong> cells(vw_rcells);
	for(uint r=0u; r<vw_rcells; r++) cells[r] = (ulong)embedding.lattice_of_cell()[(size_t)rcells[r]];
	std::vector<int> mindex_of_lattice((size_t)get_N(), -1); // explicit-state contract (#273): every fluid cell has a container state
	for(uint cell=0u; cell<input.cells; cell++) mindex_of_lattice[(size_t)embedding.lattice_of_cell()[cell]] = (int)cell;
	for(ulong n=0ull; n<get_N(); n++) if(!(flags[n]&TYPE_S) && mindex_of_lattice[(size_t)n]<0) throw std::invalid_argument("fluid cell "+to_string(n)+" has no state in the container: the attachment seeds every fluid cell from f0 and refuses to substitute an equilibrium");
	std::vector<int> c(3u*q), fx2lex(q), lex2fx(q);
	for(uint d=0u; d<q; d++) { for(uint a=0u; a<3u; a++) c[3u*d+a] = dir.lex_c(d, a); lex2fx[d] = (int)dir.lex_to_fx(d); fx2lex[d] = (int)dir.fx_to_lex(d); }
	std::vector<uint> si(input.S.indices.size()), gi(input.G.indices.size());
	for(size_t z=0u; z<si.size(); z++) si[z] = remap_cell(input.S.indices[z]);
	for(size_t z=0u; z<gi.size(); z++) gi[z] = remap_cell(input.G.indices[z]/q)*q+input.G.indices[z]%q;
	std::vector<uint> doff(vw_owned+1u, 0u), ro(vw_owned+1u, 0u), di, ri, odir(vw_owned);
	std::vector<double> dw, rw;
	std::vector<ulong> addr0(vw_owned), addr1(vw_owned);
	for(uint o=0u; o<vw_owned; o++) {
		const NativeOwnedRow& row = embedding.owned()[o];
		const uint state = row.receiver_cell*q+row.direction;
		odir[o] = row.direction; addr0[o] = (ulong)row.address_parity0; addr1[o] = (ulong)row.address_parity1;
		for(uint z=input.D.offsets[state]; z<input.D.offsets[state+1u]; z++) { di.push_back(remap_cell(input.D.indices[z]/q)*q+input.D.indices[z]%q); dw.push_back(input.D.weights[z]); }
		doff[o+1u] = (uint)di.size();
		for(uint z=input.R.offsets[state]; z<input.R.offsets[state+1u]; z++) { ri.push_back(input.R.indices[z]); rw.push_back(input.R.weights[z]); }
		ro[o+1u] = (uint)ri.size();
	}
	std::vector<double> f0(input.f0.begin(), input.f0.end()); // the supplied level-0 state of every mesh cell (lexicographic, cell-major)
	for(uint cell=0u; cell<input.cells; cell++) { // the field cache of every mesh cell from its f0 moments (level 0, pre-collision)
		double rho_c=0.0, jx=0.0, jy=0.0, jz=0.0;
		for(uint d=0u; d<q; d++) { const double v = input.f0[cell*q+d]; rho_c += v; jx += v*dir.lex_c(d, 0); jy += v*dir.lex_c(d, 1); jz += v*dir.lex_c(d, 2); }
		const ulong n = (ulong)embedding.lattice_of_cell()[cell];
		rho[n] = (float)rho_c; u.x[n] = (float)(jx/rho_c); u.y[n] = (float)(jy/rho_c); u.z[n] = (float)(jz/rho_c);
	}
	vw_upload(device, vw_cells, cells); vw_upload(device, vw_addr0, addr0); vw_upload(device, vw_addr1, addr1);
	vw_upload(device, vw_c, c); vw_upload(device, vw_fx2lex, fx2lex); vw_upload(device, vw_lex2fx, lex2fx); vw_upload(device, vw_mindex, mindex_of_lattice);
	vw_upload(device, vw_fo, input.facet_offsets); vw_upload(device, vw_rd, input.row_direction); vw_upload(device, vw_odir, odir);
	vw_upload(device, vw_so, input.S.offsets); vw_upload(device, vw_si, si); vw_upload(device, vw_sw, input.S.weights);
	vw_upload(device, vw_go, input.G.offsets); vw_upload(device, vw_gi, gi); vw_upload(device, vw_gw, input.G.weights);
	vw_upload(device, vw_doff, doff); vw_upload(device, vw_di, di); vw_upload(device, vw_dw, dw);
	vw_upload(device, vw_ro, ro); vw_upload(device, vw_ri, ri); vw_upload(device, vw_rw, rw);
	vw_upload(device, vw_wo, input.W.offsets); vw_upload(device, vw_wi, input.W.indices); vw_upload(device, vw_ww, input.W.weights);
	vw_upload(device, vw_vi, input.prism_volume); vw_upload(device, vw_normal, input.normal); vw_upload(device, vw_area, input.area);
	vw_upload(device, vw_f0, f0);
	{ // task 43: the per-row friction vectors are an explicit argument (rows*3 doubles, exported row order); empty = the original kernel friction expression (research path only). No environment variable is read.
		const size_t phi_size = (size_t)input.rows*3u;
		if(!friction_rows.empty() && friction_rows.size()!=phi_size) throw std::invalid_argument("friction rows: the size is not rows*3");
		for(const double v : friction_rows) if(!std::isfinite(v)) throw std::invalid_argument("friction rows: non-finite value");
		vw_use_phi = friction_rows.empty() ? 0u : 1u;
		const std::vector<double> zero_rows(friction_rows.empty() ? phi_size : (size_t)0u, 0.0); // the buffer of the unused branch
		vw_upload(device, vw_phi, friction_rows.empty() ? zero_rows : friction_rows);
	}
	vw_upload(device, vw_pending0, input.s0);
	vw_upload(device, vw_pending1, std::vector<double>(input.rows, 0.0));
	vw_f = Memory<double>(device, (ulong)q*vw_rcells); vw_post = Memory<double>(device, (ulong)q*vw_rcells); vw_fields = Memory<double>(device, 4ull*vw_rcells);
	vw_gpre = Memory<float>(device, (ulong)q*vw_rcells);
	vw_incoming = Memory<double>(device, (ulong)vw_rows); vw_Q = Memory<double>(device, (ulong)vw_rows); vw_impulse = Memory<double>(device, 3ull*vw_facets);
	vw_prestore = Memory<double>(device, (ulong)std::max(1u, vw_owned));
	vw_pending_slot = 0u;
	vw_kernel_load = Kernel(device, (ulong)vw_rcells, "vw_load", vw_rcells, vw_cells, vw_fx2lex, vw_c, fi, t, vw_f, vw_gpre);
	vw_kernel_collide = Kernel(device, (ulong)vw_rcells, "vw_collide", vw_rcells, input.omega, vw_c, vw_f, vw_post, vw_fields, vw_cells, rho, u);
#ifdef VOLUME_FORCE
	vw_kernel_collide.add_parameters(fx, fy, fz); // task 43: parameters 9, 10, 11; rewritten with the current force at every dispatch (enqueue_volumetric_wall_step)
#endif // VOLUME_FORCE
#ifdef VOLUMETRIC_WALL_SGS
	vw_kernel_collide.add_parameters(vw_sgs_k); // V2: the SGS constant k (the last parameter)
#endif // VOLUMETRIC_WALL_SGS
#ifdef VOLUMETRIC_WALL_OPEN_X
	{ // task 105: the E slot of every R cell (-1 interior); vw_collide writes feq3 of the current E state for the E cells
		if(!vw_open_layout) throw std::logic_error("open x: the layout is not set before the attachment");
		std::vector<int> eslot_r(vw_rcells, -1);
		for(uint r=0u; r<vw_rcells; r++) eslot_r[r] = (int)vw_open_layout->slot_of_cell()[(size_t)rcells[r]];
		vw_upload(device, vw_eslot_r, eslot_r);
		vw_kernel_collide.add_parameters(vw_eslot_r, vw_estate);
	}
#endif // VOLUMETRIC_WALL_OPEN_X
	vw_kernel_wall0 = Kernel(device, (ulong)vw_facets, "vw_wall", vw_facets, input.friction, vw_c, vw_fo, vw_rd, vw_vi, vw_normal, vw_area, vw_so, vw_si, vw_sw, vw_go, vw_gi, vw_gw, vw_post, vw_fields, vw_pending0, vw_incoming, vw_Q, vw_impulse, vw_use_phi, vw_phi);
	vw_kernel_wall1 = Kernel(device, (ulong)vw_facets, "vw_wall", vw_facets, input.friction, vw_c, vw_fo, vw_rd, vw_vi, vw_normal, vw_area, vw_so, vw_si, vw_sw, vw_go, vw_gi, vw_gw, vw_post, vw_fields, vw_pending1, vw_incoming, vw_Q, vw_impulse, vw_use_phi, vw_phi);
	const ulong stream_range = (ulong)vw_owned+(ulong)vw_rows;
	vw_kernel_stream01 = Kernel(device, stream_range, "vw_stream_pending", vw_owned, vw_rows, vw_c, vw_odir, vw_addr0, vw_addr1, t, vw_doff, vw_di, vw_dw, vw_ro, vw_ri, vw_rw, vw_post, vw_Q, vw_wo, vw_wi, vw_ww, vw_vi, vw_pending1, fi, vw_prestore);
	vw_kernel_stream10 = Kernel(device, stream_range, "vw_stream_pending", vw_owned, vw_rows, vw_c, vw_odir, vw_addr0, vw_addr1, t, vw_doff, vw_di, vw_dw, vw_ro, vw_ri, vw_rw, vw_post, vw_Q, vw_wo, vw_wi, vw_ww, vw_vi, vw_pending0, fi, vw_prestore);
	vw_kernel_seed = Kernel(device, get_N(), "vw_seed_initial", flags, vw_mindex, vw_lex2fx, vw_c, vw_f0, fi);
#ifdef VOLUMETRIC_WALL_FP64_COLLISION
	kernel_stream_collide.set_parameters(9u, input.omega); // task 57: the ordinary cells relax with the same double omega as vw_collide
	print_info("volumetric wall: FP64 collision of the ordinary cells (task 57), omega from the container");
#endif // VOLUMETRIC_WALL_FP64_COLLISION
#ifdef VOLUMETRIC_WALL_SGS
	kernel_stream_collide.set_parameters(10u, vw_sgs_k); // V2: the ordinary cells use the same SGS constant as vw_collide and vw_support_rr27
	{
		char text[160] = {};
		std::snprintf(text, sizeof(text), "volumetric wall SGS: k = %.17g (ell2 = k/(18 sqrt 2) = %.17g)%s", vw_sgs_k, vw_sgs_k/(18.0*std::sqrt(2.0)), vw_sgs_k==0.0 ? ", the rate omega" : "");
		print_info(string(text));
	}
#endif // VOLUMETRIC_WALL_SGS
	vw_affine = false;
	if(input.affine()) set_volumetric_wall_affine(input, embedding); // task 55: only with the six container sections; the accepted path above is unchanged
#ifdef VOLUMETRIC_WALL_SRD
	set_volumetric_wall_srd(input, embedding); // task 121: the redistribution tables and kernels (requires the sections; checked above)
#endif // VOLUMETRIC_WALL_SRD
#ifdef VOLUMETRIC_WALL_OPEN_X
	set_volumetric_wall_open_buffers(input, embedding); // task 105: after the affine setup (it adds the open parameters of the wall and support kernels)
#endif // VOLUMETRIC_WALL_OPEN_X
	vw_steps_completed = 0ull;
	vw_enabled = true;
}
void LBM_Domain::set_volumetric_wall_affine(const fluidx3d::experimental::VolumetricWallNativeInput& input, const fluidx3d::experimental::VolumetricWallEmbedding& embedding) { // task 55: buffers and kernels of the affine transport candidate (TASK55_ONESTEP_REFERENCE section 1, projected trial)
	using namespace fluidx3d::experimental;
	const std::vector<std::int64_t>& rindex = embedding.r_index_of_cell();
	const NativeDirections& dir = embedding.directions();
	const uint q = input.velocity_set; // task 98
	const double arm_zero = 1e-9; // the threshold of the CPU reference (t53_screen.py): an arm of a donor without a gradient must be below it
	// gradient cells: all must be R cells; gradient index of every R cell
	std::vector<int> grad_of_r(vw_rcells, -1);
	vw_gcount = (uint)input.gradient_cells.size();
	for(uint g=0u; g<vw_gcount; g++) {
		const std::int64_t r = rindex[input.gradient_cells[g]];
		if(r<0) throw std::invalid_argument("affine: a gradient cell is not an R cell");
		grad_of_r[(size_t)r] = (int)g;
	}
	// support cells: stencil cells outside R, ascending mesh order; extended index = R index, or rcells + support index
	std::vector<std::uint32_t> support;
	for(const std::uint32_t cell : input.gradient_indices) if(rindex[cell]<0) support.push_back(cell);
	std::sort(support.begin(), support.end());
	support.erase(std::unique(support.begin(), support.end()), support.end());
	vw_scount = (uint)support.size();
	std::vector<ulong> scells(vw_scount);
	std::map<std::uint32_t, uint> support_index;
	for(uint s=0u; s<vw_scount; s++) { scells[s] = (ulong)embedding.lattice_of_cell()[support[s]]; support_index[support[s]] = s; }
	std::vector<uint> gext(input.gradient_indices.size());
	for(size_t z=0u; z<gext.size(); z++) { const std::uint32_t cell = input.gradient_indices[z]; gext[z] = rindex[cell]>=0 ? (uint)rindex[cell] : vw_rcells+support_index.at(cell); }
	// G entries: arm and the gradient index of the donor (every G donor must have a gradient)
	std::vector<uint> gdon(input.G.indices.size());
	for(size_t z=0u; z<gdon.size(); z++) {
		const int g = grad_of_r[(size_t)rindex[input.G.indices[z]/q]];
		if(g<0) throw std::invalid_argument("affine: a gather donor has no gradient");
		gdon[z] = (uint)g;
	}
	// native rows: the ordinary kernel streams them without an arm term, so their direct arms must be zero
	for(const std::int64_t state : embedding.native_rows()) {
		for(uint z=input.D.offsets[(size_t)state]; z<input.D.offsets[(size_t)state+1u]; z++) {
			for(uint q=0u; q<3u; q++) if(std::fabs(input.direct_arm[3u*z+q])>arm_zero) throw std::invalid_argument("affine: a state streamed by the ordinary kernel has a direct arm");
		}
	}
	// owned rows, in the order of set_volumetric_wall: D entries (arm, donor gradient), R entries (arm and h/V of the matching G entry), receiver
	std::vector<double> darm, rarm, rsw;
	std::vector<int> dgrad, ograd(vw_owned, -1);
	std::vector<uint> ocell(vw_owned);
	for(uint o=0u; o<vw_owned; o++) {
		const NativeOwnedRow& row = embedding.owned()[o];
		const uint state = row.receiver_cell*q+row.direction;
		const std::int64_t x = rindex[row.receiver_cell]; // an owned row may write into an ordinary cell (rows_r_to_ordinary): its direct arm term applies, it has no capture entry
		if(x<0 && input.R.offsets[state]<input.R.offsets[state+1u]) throw std::invalid_argument("affine: a scatter receiver is outside R");
		ocell[o] = x<0 ? 0xFFFFFFFFu : (uint)x;
		for(uint z=input.D.offsets[state]; z<input.D.offsets[state+1u]; z++) {
			const int g = grad_of_r[(size_t)rindex[input.D.indices[z]/q]];
			double size = 0.0;
			for(uint q=0u; q<3u; q++) { darm.push_back(input.direct_arm[3u*z+q]); size = std::max(size, std::fabs(input.direct_arm[3u*z+q])); }
			if(g<0 && size>arm_zero) throw std::invalid_argument("affine: a direct donor with an arm has no gradient");
			dgrad.push_back(g);
		}
		if(input.R.offsets[state]<input.R.offsets[state+1u]) {
			ograd[o] = grad_of_r[(size_t)ocell[o]];
			if(ograd[o]<0) throw std::invalid_argument("affine: a scatter receiver has no gradient");
		}
		const uint opposite = dir.lex_opposite(row.direction);
		for(uint z=input.R.offsets[state]; z<input.R.offsets[state+1u]; z++) {
			const uint r = input.R.indices[z];
			if(input.row_direction[r]!=row.direction) throw std::invalid_argument("affine: a scatter entry has another direction than its row");
			bool found = false;
			for(uint y=input.G.offsets[r]; y<input.G.offsets[r+1u]; y++) {
				if(input.G.indices[y]==row.receiver_cell*q+opposite) {
					for(uint q=0u; q<3u; q++) rarm.push_back(input.prism_arm[3u*y+q]);
#ifdef VOLUMETRIC_WALL_OPEN_X
					rsw.push_back(vw_open_layout->exterior_row()[r]!=0u ? 0.0 : input.G.weights[y]/input.volume[row.receiver_cell]); // task 105 (O3a): no scatter on exterior rows
#else // VOLUMETRIC_WALL_OPEN_X
					rsw.push_back(input.G.weights[y]/input.volume[row.receiver_cell]); // h / V_x, as the reference
#endif // VOLUMETRIC_WALL_OPEN_X
					found = true;
					break;
				}
			}
			if(!found) throw std::invalid_argument("affine: a scatter entry has no gather entry of the same prism part");
		}
	}
	vw_upload(device, vw_scells, scells); vw_upload(device, vw_goff, input.gradient_offsets); vw_upload(device, vw_gext, gext); vw_upload(device, vw_gwt, input.gradient_weights);
	vw_upload(device, vw_garm, input.prism_arm); vw_upload(device, vw_gdon, gdon);
	vw_upload(device, vw_darm, darm); vw_upload(device, vw_dgrad, dgrad); vw_upload(device, vw_rarm, rarm); vw_upload(device, vw_rsw, rsw);
	vw_upload(device, vw_ocell, ocell); vw_upload(device, vw_ograd, ograd);
	vw_spost = Memory<double>(device, (ulong)q*std::max(1u, vw_scount)); vw_sfields = Memory<double>(device, 4ull*std::max(1u, vw_scount));
	vw_grad = Memory<double>(device, 3ull*((ulong)q+4ull)*std::max(1u, vw_gcount)); vw_ebar = Memory<double>(device, (ulong)vw_rows);
	vw_upload(device, vw_returned, std::vector<double>((ulong)q*vw_rcells, 0.0)); // states that no wall kernel writes stay 0 (agreed capture convention)
	vw_upload(device, vw_dinc, std::vector<double>((ulong)q*vw_rcells, 0.0));
	vw_kernel_gradient = Kernel(device, (ulong)std::max(1u, vw_gcount), "vw_gradient", vw_gcount, vw_rcells, vw_goff, vw_gext, vw_gwt, vw_post, vw_fields, vw_spost, vw_sfields, vw_grad);
#ifdef VOLUMETRIC_WALL_RR27
	{ // task 98: the gradient index of the cell of every sampler entry (every sample cell is a gradient cell), the capture buffers, the RR27 kernels
		std::vector<uint> sgrad(input.S.indices.size());
		for(size_t z=0u; z<sgrad.size(); z++) {
			const std::int64_t r = rindex[input.S.indices[z]];
			const int g = r<0 ? -1 : grad_of_r[(size_t)r];
			if(g<0) throw std::invalid_argument("RR27: a sample cell has no gradient (the wall-extrapolated reference reads S[L rho] and S[L u])");
			sgrad[z] = (uint)g;
		}
		vw_upload(device, vw_sgrad, sgrad);
		vw_upload(device, vw_wstate, std::vector<double>(4ull*vw_facets, 0.0));
		vw_upload(device, vw_delta, std::vector<double>((size_t)vw_facets, 0.0));
		vw_wfail = Memory<uint>(device, 1ull, 1u, true, true, 0u);
		vw_wfail.write_to_device();
	}
	vw_kernel_support = Kernel(device, (ulong)std::max(1u, vw_scount), "vw_support_rr27", vw_scount, vw_scells, vw_fx2lex, vw_c, fi, t, input.omega, fx, fy, fz, vw_spost, vw_sfields);
#ifdef VOLUMETRIC_WALL_SGS
	vw_kernel_support.add_parameters(vw_sgs_k); // V2: parameter 12, the SGS constant k
#endif // VOLUMETRIC_WALL_SGS
#ifdef VOLUMETRIC_WALL_OPEN_X
	{ // task 105: the E slot of every support cell (-1 interior); vw_support_rr27 writes feq3 of the current E state for the E cells
		std::vector<int> eslot_s(std::max(1u, vw_scount), -1);
		for(uint s=0u; s<vw_scount; s++) eslot_s[s] = (int)vw_open_layout->slot_of_cell()[(size_t)support[s]];
		vw_upload(device, vw_eslot_s, eslot_s);
		vw_kernel_support.add_parameters(vw_eslot_s, vw_estate);
	}
#endif // VOLUMETRIC_WALL_OPEN_X
	vw_kernel_wall_affine0 = Kernel(device, (ulong)vw_facets, "vw_wall_rr27", vw_facets, input.friction, vw_c, vw_fo, vw_rd, vw_vi, vw_normal, vw_so, vw_si, vw_sw, vw_sgrad, vw_go, vw_gi, vw_gw, vw_post, vw_fields, vw_pending0, vw_incoming, vw_Q, vw_impulse, vw_phi, vw_garm, vw_gdon, vw_grad, vw_ebar, vw_wstate, vw_delta, vw_wfail);
	vw_kernel_wall_affine1 = Kernel(device, (ulong)vw_facets, "vw_wall_rr27", vw_facets, input.friction, vw_c, vw_fo, vw_rd, vw_vi, vw_normal, vw_so, vw_si, vw_sw, vw_sgrad, vw_go, vw_gi, vw_gw, vw_post, vw_fields, vw_pending1, vw_incoming, vw_Q, vw_impulse, vw_phi, vw_garm, vw_gdon, vw_grad, vw_ebar, vw_wstate, vw_delta, vw_wfail);
#ifdef VOLUMETRIC_WALL_ROUGH
	{ // V2: the facet law (K0_a, C_a): the container section facet_friction (checked by validate()), or (K, 0) on every facet
		std::vector<double> law((size_t)vw_facets*2u, 0.0);
		if(input.facet_friction.empty()) {
			for(uint a=0u; a<vw_facets; a++) law[2u*a] = input.friction; // C = 0: the kernel uses K0 = K directly (the scalar friction, bitwise)
		} else {
			if(input.facet_friction.size()!=(size_t)vw_facets*2u) throw std::invalid_argument("facet_friction size is not facets*2");
			law = input.facet_friction;
		}
		vw_upload(device, vw_facet_law, law);
		vw_kernel_wall_affine0.add_parameters(vw_facet_law); // the last parameter of vw_wall_rr27
		vw_kernel_wall_affine1.add_parameters(vw_facet_law);
		uint rough{ 0u };
		for(uint a=0u; a<vw_facets; a++) rough += law[2u*a+1u]!=0.0 ? 1u : 0u;
		print_info("volumetric wall ROUGH: facet law from "+string(input.facet_friction.empty() ? "the scalar K (no facet_friction section)" : "facet_friction")+", "+to_string(rough)+" of "+to_string(vw_facets)+" facets with C > 0");
	}
#endif // VOLUMETRIC_WALL_ROUGH
#ifdef VOLUMETRIC_WALL_OPEN_X
	{ // task 105 (O3a): the R index of the E cell of every exterior facet (-1: interior facet); vw_wall_rr27 returns v feq3 of its fields
		std::vector<int> efacet(vw_facets, -1);
		for(uint a=0u; a<vw_facets; a++) {
			const std::int64_t e = vw_open_layout->e_cell_of_facet()[a];
			if(e<0) continue;
			const std::int64_t r = rindex[(size_t)e];
			if(r<0) throw std::invalid_argument("open x: the E cell of an exterior facet is not an R cell (it must be a gather donor of that facet)");
			efacet[a] = (int)r;
		}
		vw_upload(device, vw_efacet, efacet);
		vw_kernel_wall_affine0.add_parameters(vw_efacet); // the last parameter of vw_wall_rr27
		vw_kernel_wall_affine1.add_parameters(vw_efacet);
	}
#endif // VOLUMETRIC_WALL_OPEN_X
#ifdef VOLUMETRIC_WALL_T119
	{ // task 119 (section 15.11): the static companion buffers and the extra parameters (the last of vw_wall_rr27, after the law)
		const size_t rows_n = (size_t)vw_rows;
#ifdef VOLUMETRIC_WALL_OPEN_X
		{ // task 123: guarded compatibility of the T119 correction with the open-x branch (replaces the former compile-time refusal). Refuse unless every row the correction modifies (a W entry as
			// receiver or producer) is an interior row whose G donors, the gradient stencils of those donors and the fluid R/scatter destination cells of the W producers all lie at lattice
			// x in [3, nx-4] and are neither an E cell nor an inward cell, and unless no exterior row has a W entry or a nonzero t119_gdelta. Exact checks, no repair.
			const VolumetricWallOpenLayout& layout = volumetric_wall_open_layout();
			const int nx = layout.nx();
			const uint qv = input.velocity_set;
			auto bad_cell = [&](const std::uint32_t cell) { const int x = input.cell_xyz[3u*cell]; return x<3 || x>nx-4 || layout.slot_of_cell()[cell]>=0; };
			std::vector<std::uint8_t> modified(rows_n, 0u), producer(rows_n, 0u);
			for(uint r=0u; r<input.W.rows; r++) {
				if(input.W.offsets[r]==input.W.offsets[r+1u]) continue;
				modified[r] = 1u;
				for(uint z=input.W.offsets[r]; z<input.W.offsets[r+1u]; z++) { modified[input.W.indices[z]] = 1u; producer[input.W.indices[z]] = 1u; }
			}
			std::map<std::uint32_t, size_t> grad_index;
			for(size_t g=0u; g<input.gradient_cells.size(); g++) grad_index[input.gradient_cells[g]] = g;
			std::set<std::uint32_t> donors;
			for(size_t r=0u; r<rows_n; r++) {
				const bool exterior = layout.exterior_row()[r]!=0u;
				if(exterior) {
					if(modified[r]) throw std::invalid_argument("open x + T119: exterior row "+to_string(r)+" has a W entry");
					for(uint z=input.G.offsets[r]; z<input.G.offsets[r+1u]; z++) for(uint k=0u; k<3u; k++) if(input.t119_gdelta[3u*z+k]!=0.0) throw std::invalid_argument("open x + T119: exterior row "+to_string(r)+" has a nonzero t119_gdelta");
					continue;
				}
				if(!modified[r]) continue;
				for(uint z=input.G.offsets[r]; z<input.G.offsets[r+1u]; z++) donors.insert(input.G.indices[z]/qv);
			}
			for(const std::uint32_t d : donors) {
				if(bad_cell(d)) throw std::invalid_argument("open x + T119: a G donor cell of a modified row is in or next to an E column (cell "+to_string(d)+")");
				const auto it = grad_index.find(d);
				if(it==grad_index.end()) throw std::invalid_argument("open x + T119: a G donor of a modified row has no gradient");
				for(uint ax=0u; ax<3u; ax++) for(uint z=input.gradient_offsets[3u*it->second+ax]; z<input.gradient_offsets[3u*it->second+ax+1u]; z++) if(bad_cell(input.gradient_indices[z])) throw std::invalid_argument("open x + T119: a gradient stencil cell of a G donor is in or next to an E column");
			}
			for(uint st=0u; st<input.R.rows; st++) for(uint z=input.R.offsets[st]; z<input.R.offsets[st+1u]; z++) if(producer[input.R.indices[z]] && bad_cell(st/qv)) throw std::invalid_argument("open x + T119: a fluid R/scatter destination cell of a W producer is in or next to an E column (cell "+to_string(st/qv)+")");
			print_info("open x + T119: guarded compatibility checks passed ("+to_string(donors.size())+" donor cells)");
		}
#endif // VOLUMETRIC_WALL_OPEN_X
		std::vector<double> wcol(rows_n, 0.0), summ(3u*rows_n, 0.0);
		for(uint r=0u; r<input.W.rows; r++) {
			for(uint z=input.W.offsets[r]; z<input.W.offsets[r+1u]; z++) {
				const uint p = input.W.indices[z];
				wcol[p] += input.W.weights[z];
				for(uint k=0u; k<3u; k++) summ[3u*p+k] += input.t119_wmoment[3u*z+k];
			}
		}
		vw_upload(device, vw_t119_delta, input.t119_gdelta); vw_upload(device, vw_t119_wcol, wcol); vw_upload(device, vw_t119_summ, summ);
		vw_upload(device, vw_t119_wm, input.t119_wmoment); vw_upload(device, vw_t119_out, std::vector<double>(5u*rows_n, 0.0));
		vw_t119_neg = Memory<uint>(device, 1ull, 1u, true, true, 0u);
		vw_t119_neg.write_to_device();
		vw_t119_p1 = Memory<uint>(device, 1ull, 1u, true, true, 0u);
		vw_t119_p1.write_to_device();
		vw_kernel_wall_affine0.add_parameters(vw_t119_delta, vw_t119_wcol, vw_t119_summ, vw_t119_out, vw_t119_neg, vw_t119_p1, vw_area);
		vw_kernel_wall_affine1.add_parameters(vw_t119_delta, vw_t119_wcol, vw_t119_summ, vw_t119_out, vw_t119_neg, vw_t119_p1, vw_area);
		uint with_w = 0u;
		for(size_t r=0u; r<rows_n; r++) with_w += wcol[r]>0.0 ? 1u : 0u;
		print_info("volumetric wall T119 (whole-prism correction, experimental): "+to_string(with_w)+" of "+to_string(vw_rows)+" rows have a stored W column");
	}
#endif // VOLUMETRIC_WALL_T119
#else // VOLUMETRIC_WALL_RR27
	vw_kernel_support = Kernel(device, (ulong)std::max(1u, vw_scount), "vw_support", vw_scount, vw_scells, vw_fx2lex, vw_c, fi, t, fx, fy, fz, vw_spost, vw_sfields);
	vw_kernel_wall_affine0 = Kernel(device, (ulong)vw_facets, "vw_wall_affine", vw_facets, input.friction, vw_c, vw_fo, vw_rd, vw_vi, vw_normal, vw_area, vw_so, vw_si, vw_sw, vw_go, vw_gi, vw_gw, vw_post, vw_fields, vw_pending0, vw_incoming, vw_Q, vw_impulse, vw_phi, vw_garm, vw_gdon, vw_grad, vw_ebar);
	vw_kernel_wall_affine1 = Kernel(device, (ulong)vw_facets, "vw_wall_affine", vw_facets, input.friction, vw_c, vw_fo, vw_rd, vw_vi, vw_normal, vw_area, vw_so, vw_si, vw_sw, vw_go, vw_gi, vw_gw, vw_post, vw_fields, vw_pending1, vw_incoming, vw_Q, vw_impulse, vw_phi, vw_garm, vw_gdon, vw_grad, vw_ebar);
#endif // VOLUMETRIC_WALL_RR27
	const ulong stream_range = (ulong)vw_owned+(ulong)vw_rows;
	vw_kernel_stream_affine01 = Kernel(device, stream_range, "vw_stream_pending_affine", vw_owned, vw_rows, vw_c, vw_odir, vw_addr0, vw_addr1, t, vw_doff, vw_di, vw_dw, vw_ro, vw_ri, vw_rw, vw_post, vw_Q, vw_wo, vw_wi, vw_ww, vw_vi, vw_pending1, fi, vw_prestore,
		vw_darm, vw_dgrad, vw_rarm, vw_rsw, vw_ocell, vw_ograd, vw_fields, vw_grad, vw_ebar, vw_returned, vw_dinc);
	vw_kernel_stream_affine10 = Kernel(device, stream_range, "vw_stream_pending_affine", vw_owned, vw_rows, vw_c, vw_odir, vw_addr0, vw_addr1, t, vw_doff, vw_di, vw_dw, vw_ro, vw_ri, vw_rw, vw_post, vw_Q, vw_wo, vw_wi, vw_ww, vw_vi, vw_pending0, fi, vw_prestore,
		vw_darm, vw_dgrad, vw_rarm, vw_rsw, vw_ocell, vw_ograd, vw_fields, vw_grad, vw_ebar, vw_returned, vw_dinc);
#ifdef VOLUMETRIC_WALL_T119
	vw_kernel_stream_affine01.add_parameters(vw_t119_wm, vw_t119_out); // the last parameters of vw_stream_pending_affine
	vw_kernel_stream_affine10.add_parameters(vw_t119_wm, vw_t119_out);
#endif // VOLUMETRIC_WALL_T119
	vw_affine = true;
	print_info("volumetric wall AFFINE candidate (task 55): "+to_string(vw_gcount)+" gradient cells, "+to_string(vw_scount)+" support cells");
}
#ifdef VOLUMETRIC_WALL_OPEN_X
/// @brief Task 105: the refusals 7.6 of one E state (the reference's CharacteristicBoundary.admissible). Task 108: the sets are
/// closed at zero speed (outlet u_n >= 0, inlet u_n <= 0), so an exact-rest E state is admitted; wrong-sign flow is refused.
/// @param rho_b density; ux, uy, uz velocity; s the outward sign (-1 inlet, +1 outlet)
/// @return an empty text when admissible, else the reason
std::string vw_open_refusal(const double rho_b, const double ux, const double uy, const double uz, const int s) {
	const double c = 1.0/std::sqrt(3.0);
	if(!(std::isfinite(rho_b)&&std::isfinite(ux)&&std::isfinite(uy)&&std::isfinite(uz)&&rho_b>0.0)) return "an E state is not finite with rho_b > 0";
	const double un = (double)s*ux;
	const double eps_sign = 1e-13; // task 125: the registered numerical sign resolution (the same value as the device test in vw_open_state); the E velocity is never clipped or snapped
	if(s>0 && !(un>=-eps_sign)) return "outlet backflow: u_n < -eps_sign at an outlet E node"; // task 108: zero speed admitted; task 125: rounding-level negative values within eps_sign admitted
	if(s<0 && !(un<=eps_sign)) return "inlet outflow: u_n > eps_sign at an inlet E node";
	if(!(std::fabs(un)<c)) return "|u_n| >= c at an E node";
	return "";
}
void LBM_Domain::set_volumetric_wall_open_buffers(const fluidx3d::experimental::VolumetricWallNativeInput& input, const fluidx3d::experimental::VolumetricWallEmbedding& embedding) {
	using namespace fluidx3d::experimental;
	const VolumetricWallOpenLayout& layout = *vw_open_layout;
	if(input.box[0]!=(int)Nx||input.box[1]!=(int)Ny||input.box[2]!=(int)Nz) throw std::invalid_argument("open x: the container box differs from the domain");
	vw_ecount = (uint)layout.e_cells().size();
	std::vector<ulong> elattice(vw_ecount), einward(2u*vw_ecount);
	std::vector<uint> eslot(vw_ecount);
	std::vector<int> eside(vw_ecount);
	for(uint k=0u; k<vw_ecount; k++) {
		elattice[k] = (ulong)embedding.lattice_of_cell()[layout.e_cells()[k]];
		einward[2u*k] = (ulong)embedding.lattice_of_cell()[layout.e_inward()[2u*k]];
		einward[2u*k+1u] = (ulong)embedding.lattice_of_cell()[layout.e_inward()[2u*k+1u]];
		eslot[k] = layout.e_slot()[k];
		eside[k] = layout.e_side()[k];
		const std::string refusal = vw_open_refusal(vw_open.initial_rho, vw_open.initial_velocity[0], vw_open.initial_velocity[1], vw_open.initial_velocity[2], eside[k]);
		if(!refusal.empty()) throw std::invalid_argument("open x: the initial E state is refused (7.6): "+refusal);
		for(uint v=0u; v<2u; v++) { // the E state of both levels starts as the declared state
			Memory<double>& state = v==0u ? vw_estate : vw_estate_next;
			state[4ull*eslot[k]] = vw_open.initial_rho;
			state[4ull*eslot[k]+1ull] = vw_open.initial_velocity[0];
			state[4ull*eslot[k]+2ull] = vw_open.initial_velocity[1];
			state[4ull*eslot[k]+3ull] = vw_open.initial_velocity[2];
		}
	}
	vw_estate.write_to_device();
	vw_estate_next.write_to_device();
	vw_upload(device, vw_elattice, elattice); vw_upload(device, vw_einward, einward); vw_upload(device, vw_eslot, eslot); vw_upload(device, vw_eside, eside);
	vw_open_fail = Memory<uint>(device, 1ull, 1u, true, true, 0u);
	vw_open_fail.write_to_device();
	vw_open_exterior_row = layout.exterior_row();
	vw_kernel_open_state = Kernel(device, (ulong)std::max(1u, vw_ecount), "vw_open_state", vw_ecount, vw_elattice, vw_einward, vw_eside, vw_eslot, vw_fx2lex, vw_c, fi, t,
		vw_estate, vw_estate_next, vw_open_fail, vw_open.outlet_relaxation(), vw_open.outlet_rho_inf, vw_open.hard_inlet ? 1u : 0u, vw_open.inlet_relaxation,
		vw_open.inlet_velocity[0], vw_open.inlet_velocity[1], vw_open.inlet_velocity[2]);
	vw_kernel_open_store = Kernel(device, (ulong)std::max(1u, vw_ecount), "vw_open_store", vw_ecount, vw_elattice, vw_eslot, vw_lex2fx, vw_c, vw_estate, fi, t);
	{ // task 105: which kernel gives each E cell its feq3: vw_collide (R cells), vw_support_rr27 (support cells), stream_collide (ordinary cells)
		std::vector<std::uint8_t> in_support((size_t)input.cells, 0u);
		for(const std::uint32_t cell : input.gradient_indices) if(embedding.r_index_of_cell()[cell]<0) in_support[cell] = 1u;
		uint e_r = 0u, e_support = 0u, e_ordinary = 0u;
		for(const std::uint32_t cell : layout.e_cells()) {
			if(embedding.r_index_of_cell()[cell]>=0) e_r++; else if(in_support[cell]!=0u) e_support++; else e_ordinary++;
		}
		vw_open_e_paths = { e_r, e_support, e_ordinary };
		if(e_support!=0u||e_ordinary!=0u) throw std::logic_error("open x: an E cell is not an R cell (the embedding forces every E cell into R)");
	}
	print_info("volumetric wall OPEN X: "+layout.text()+"; E cells in R "+to_string(vw_open_e_paths[0])+", support "+to_string(vw_open_e_paths[1])+", ordinary "+to_string(vw_open_e_paths[2])+"; K_p = "+to_string(vw_open.outlet_relaxation(), 17u)+(vw_open.hard_inlet ? ", hard inlet" : ", relaxed inlet"));
}
void LBM_Domain::vw_read_open_state(vector<double>& host) { device.finish_queue(); host = vw_download(vw_estate); }
uint LBM_Domain::volumetric_wall_open_failures() { device.finish_queue(); return vw_download(vw_open_fail)[0]; }
#endif // VOLUMETRIC_WALL_OPEN_X
void LBM_Domain::enqueue_volumetric_wall_load() {
	vw_kernel_load.set_parameters(5u, t).enqueue_run();
#ifdef VOLUMETRIC_WALL_RR27
	if(vw_affine&&vw_scount>0u) { // task 98: the support cells collide in FP64 from their level-t state BEFORE stream_collide (it reads the same load set and then overwrites it)
		vw_kernel_support.set_parameters(5u, t);
		vw_kernel_support.set_parameters(7u, fx, fy, fz);
		vw_kernel_support.enqueue_run();
	}
#endif // VOLUMETRIC_WALL_RR27
#ifdef VOLUMETRIC_WALL_OPEN_X
	vw_kernel_open_state.set_parameters(8u, t).enqueue_run(); // task 105: E^(n+1) from E^n and the level-n fields of b-1, b-2 (before stream_collide overwrites their load sets)
#endif // VOLUMETRIC_WALL_OPEN_X
}
void LBM_Domain::enqueue_volumetric_wall_step() {
#ifdef VOLUMETRIC_WALL_SRD
	if(srd_capture) device.get_cl_queue().enqueueCopyBuffer(fi.get_cl_buffer(), vw_pre_wall_fi.get_cl_buffer(), 0ull, 0ull, fi.capacity()); // task 125 (diagnostic only): the snapshot AFTER the ordinary stream_collide has written its post store sets and BEFORE any wall receiver store (vw_stream*) and before the E store, so every ordinary donor's stored post is intact
#endif // VOLUMETRIC_WALL_SRD
#ifdef VOLUME_FORCE
	vw_kernel_collide.set_parameters(9u, fx, fy, fz); // task 43 (#979): the same current force as enqueue_stream_collide gives the ordinary kernel, so set_f/set_fx/set_fy/set_fz cannot diverge
#endif // VOLUME_FORCE
	vw_kernel_collide.enqueue_run();
	if(vw_affine) { // task 55: support read (after stream_collide, before any wall store), gradients, affine wall, affine stream
#ifndef VOLUMETRIC_WALL_RR27 // task 98: with RR27 the support cells were collided in enqueue_volumetric_wall_load
		vw_kernel_support.set_parameters(5u, t);
#ifdef VOLUME_FORCE
		vw_kernel_support.set_parameters(6u, fx, fy, fz);
#endif // VOLUME_FORCE
		vw_kernel_support.enqueue_run();
#endif // VOLUMETRIC_WALL_RR27
		vw_kernel_gradient.enqueue_run();
		if(vw_pending_slot==0u) { vw_kernel_wall_affine0.enqueue_run(); vw_kernel_stream_affine01.set_parameters(6u, t).enqueue_run(); }
		else { vw_kernel_wall_affine1.enqueue_run(); vw_kernel_stream_affine10.set_parameters(6u, t).enqueue_run(); }
	} else {
		if(vw_pending_slot==0u) { vw_kernel_wall0.enqueue_run(); vw_kernel_stream01.set_parameters(6u, t).enqueue_run(); } // wall reads pending0, stream writes pending1
		else { vw_kernel_wall1.enqueue_run(); vw_kernel_stream10.set_parameters(6u, t).enqueue_run(); }
	}
	vw_pending_slot = 1u-vw_pending_slot; // host-side swap after the launches are enqueued; every facet consumed the old pending in vw_wall before vw_stream_pending started (in-order queue)
#ifdef VOLUMETRIC_WALL_OPEN_X
	// task 105: E^(n+1) becomes the current state (every kernel of this step read E^n), then the E load sets of level t+1 get
	// fl32_RTE(feq3(E^(n+1)) - w), after vw_stream wrote the owned states of level t+1 (in-order queue)
	device.get_cl_queue().enqueueCopyBuffer(vw_estate_next.get_cl_buffer(), vw_estate.get_cl_buffer(), 0ull, 0ull, vw_estate.capacity());
	vw_kernel_open_store.set_parameters(7u, t+1ull).enqueue_run();
#endif // VOLUMETRIC_WALL_OPEN_X
#ifdef VOLUMETRIC_WALL_SRD
	enqueue_volumetric_wall_srd(t+1ull); // task 121: the redistribution ONCE on the complete level-(t+1) load sets (transport and pending arrivals are finished on the in-order queue), before the next collision
#endif // VOLUMETRIC_WALL_SRD
	vw_steps_completed++;
}
void LBM_Domain::enqueue_volumetric_wall_initial_state() { // after initialize(): every fluid cell's level-0 load set rewritten with the declared pre-collision state
	vw_kernel_seed.enqueue_run();
#ifdef VOLUMETRIC_WALL_OPEN_X
	vw_kernel_open_store.set_parameters(7u, 0ull).enqueue_run(); // task 105: the E cells' level-0 load set (parity 0, as vw_seed_initial; the host t is odd inside initialize()) holds feq3 of the declared initial E state
#endif // VOLUMETRIC_WALL_OPEN_X
}
#ifdef VOLUMETRIC_WALL_T119
void LBM_Domain::volumetric_wall_t119_counts(uint& negative_q, uint& p1_failures) { // task 119: read after the queue is finished
	device.finish_queue();
	negative_q = vw_download(vw_t119_neg)[0]; p1_failures = vw_download(vw_t119_p1)[0];
}
#endif // VOLUMETRIC_WALL_T119
#ifdef VOLUMETRIC_WALL_RR27
void LBM_Domain::volumetric_wall_failure_counts(uint& wall_state_failures, uint& clamps) { // #2034: the two counters, read after the queue is finished
	device.finish_queue();
	wall_state_failures = vw_affine ? vw_download(vw_wfail)[0] : 0u;
	clamps = vw_download(vw_clamp)[0];
}
#endif // VOLUMETRIC_WALL_RR27
#ifdef VOLUMETRIC_WALL_SRD
void LBM_Domain::set_volumetric_wall_srd(const fluidx3d::experimental::VolumetricWallNativeInput& input, const fluidx3d::experimental::VolumetricWallEmbedding& embedding) { // task 121: the tables of the task-120 redistribution, uploaded unchanged (FP64, integers), and the three stages
	using namespace fluidx3d::experimental;
	const VolumetricWallNativeInput::SrdTables& t = input.srd;
	t.validate(input.cells); // sizes, ranges, CSR structure, positivity, SPD B (throws, named)
#ifdef VOLUMETRIC_WALL_OPEN_X
	{ // admitted open-x demo: no group member, stencil cell, seed or active cell may be an E cell or an inward cell (b-1, b-2) of an E column, and nothing may wrap in x (fail fast)
		const VolumetricWallOpenLayout& layout = volumetric_wall_open_layout();
		const int nx = layout.nx();
		auto check = [&](const std::uint32_t cell, const char* what) {
			const int x = input.cell_xyz[3u*cell];
			if(x<3 || x>nx-4 || layout.slot_of_cell()[cell]>=0) throw std::invalid_argument(std::string("srd on the open-x branch: ")+what+" cell "+std::to_string(cell)+" is in or next to an E column (x "+std::to_string(x)+" of "+std::to_string(nx)+"); the admitted demo requires x in [3, nx-4]");
		};
		for(const std::uint32_t c : t.seeds) check(c, "seed");
		for(const std::uint32_t c : t.entry_cell) check(c, "group member");
		for(const std::uint32_t c : t.st_cell) check(c, "stencil");
		for(const std::uint32_t c : t.active_cells) check(c, "active");
		// no x wrap: a wrapped pair has a short stored offset, so compare the stored offset with the lattice x difference of the cells themselves (chart consistency): a wrap makes the cell difference about nx while the offset stays short
		for(size_t sd=0u; sd<t.seeds.size(); sd++) {
			const int xs = input.cell_xyz[3u*t.seeds[sd]];
			for(std::uint32_t z=t.st_starts[sd]; z<t.st_starts[sd+1u]; z++) {
				const double dx_cells = (double)(input.cell_xyz[3u*t.st_cell[z]]-xs);
				if(std::fabs(dx_cells-t.st_d[3u*z])>4.0) throw std::invalid_argument("srd on the open-x branch: a stencil entry is not in the seed chart without an x wrap (seed "+std::to_string(sd)+")");
			}
			for(std::uint32_t z=t.entry_starts[sd]; z<t.entry_starts[sd+1u]; z++) {
				const double dx_cells = (double)(input.cell_xyz[3u*t.entry_cell[z]]-xs);
				if(std::fabs(dx_cells-t.entry_delta[3u*z])>4.0) throw std::invalid_argument("srd on the open-x branch: a group member is not in the seed chart without an x wrap (seed "+std::to_string(sd)+")");
			}
		}
	}
#endif // VOLUMETRIC_WALL_OPEN_X
	const std::vector<std::uint64_t>& lattice = embedding.lattice_of_cell();
	if(lattice.size()!=(size_t)input.cells) throw std::logic_error("srd: the embedding does not map every mesh cell");
	std::vector<ulong> lat(lattice.size());
	for(size_t m=0u; m<lat.size(); m++) lat[m] = (ulong)lattice[m];
	vw_upload(device, srd_lat, lat);
	vw_upload(device, srd_estart, t.entry_starts); vw_upload(device, srd_ecell, t.entry_cell);
	vw_upload(device, srd_sstart, t.st_starts); vw_upload(device, srd_scell, t.st_cell);
	vw_upload(device, srd_acell, t.active_cells); vw_upload(device, srd_astart, t.active_starts); vw_upload(device, srd_aentry, t.active_entry);
	vw_upload(device, srd_sidx, t.st_seed_index);
	vw_upload(device, srd_ewv, t.entry_wv); vw_upload(device, srd_ew, t.entry_w); vw_upload(device, srd_edelta, t.entry_delta); vw_upload(device, srd_vhat, t.vhat);
	vw_upload(device, srd_sw, t.st_w); vw_upload(device, srd_sd, t.st_d); vw_upload(device, srd_b, t.B); vw_upload(device, srd_swsum, t.st_wsum); vw_upload(device, srd_acoef, t.active_coef);
	vw_upload(device, srd_stats, std::vector<uint>(6u, 0u));
	srd_seeds = (uint)t.seeds.size(); srd_entries = (uint)t.entry_cell.size(); srd_active = (uint)t.active_cells.size();
	srd_qhat = Memory<double>(device, (ulong)velocity_set*srd_seeds);
	srd_r = Memory<double>(device, (ulong)velocity_set*srd_entries);
	const ulong t_placeholder = 0ull;
	srd_kernel_means = Kernel(device, (ulong)srd_seeds, "srd_means", srd_seeds, srd_lat, srd_estart, srd_ecell, srd_ewv, srd_vhat, vw_fx2lex, vw_c, fi, t_placeholder, srd_qhat, srd_stats);
	srd_kernel_recon = Kernel(device, (ulong)srd_seeds, "srd_recon", srd_seeds, srd_lat, srd_estart, srd_edelta, srd_sstart, srd_scell, srd_sw, srd_sd, srd_sidx, srd_b, srd_swsum, srd_qhat, vw_fx2lex, vw_c, fi, t_placeholder, srd_r, srd_stats);
	srd_kernel_apply = Kernel(device, (ulong)srd_active, "srd_apply", srd_active, srd_lat, srd_acell, srd_acoef, srd_astart, srd_aentry, srd_ew, srd_r, vw_fx2lex, vw_c, fi, t_placeholder, srd_stats);
	if(srd_capture) { srd_pre_fi = Memory<fpxx>(device, fi.range()); vw_pre_wall_fi = Memory<fpxx>(device, fi.range()); } // tasks 124, 125: diagnostic copy targets
	srd_enabled = true;
	print_info("volumetric wall SRD (task 121): "+to_string(srd_seeds)+" seeds, "+to_string(srd_entries)+" group entries, "+to_string(t.st_cell.size())+" stencil entries, "+to_string(srd_active)+" active cells");
}
void LBM_Domain::enqueue_volumetric_wall_srd(const ulong t1) { // the three stages in order on the in-order queue; the second reads only decoded stored values and the means, the third writes the stored sets of the active cells
	if(!srd_enabled) throw std::logic_error("srd: the tables are not attached");
	if(srd_capture) device.get_cl_queue().enqueueCopyBuffer(fi.get_cl_buffer(), srd_pre_fi.get_cl_buffer(), 0ull, 0ull, fi.capacity()); // task 124: the stored state the three stages consume (in-order queue)
	srd_kernel_means.set_parameters(9u, t1).enqueue_run();
	srd_kernel_recon.set_parameters(15u, t1).enqueue_run();
	srd_kernel_apply.set_parameters(11u, t1).enqueue_run();
}
void LBM_Domain::volumetric_wall_pre_wall_fi(vector<fpxx>& host) {
	if(!srd_capture) throw std::logic_error("srd: the diagnostic capture is not enabled");
	host.resize(fi.range()); device.finish_queue(); device.get_cl_queue().enqueueReadBuffer(vw_pre_wall_fi.get_cl_buffer(), CL_TRUE, 0ull, fi.capacity(), (void*)host.data());
}
void LBM_Domain::volumetric_wall_srd_pre_fi(vector<fpxx>& host) {
	if(!srd_capture) throw std::logic_error("srd: the diagnostic capture is not enabled");
	host.resize(fi.range()); device.finish_queue(); device.get_cl_queue().enqueueReadBuffer(srd_pre_fi.get_cl_buffer(), CL_TRUE, 0ull, fi.capacity(), (void*)host.data());
}
void LBM_Domain::volumetric_wall_srd_read(vector<double>& qhat, vector<double>& recon, vector<uint>& stats) {
	device.finish_queue();
	qhat = vw_download(srd_qhat); recon = vw_download(srd_r); stats = vw_download(srd_stats);
}
#endif // VOLUMETRIC_WALL_SRD
fluidx3d::experimental::VolumetricWallNativeCapture LBM_Domain::read_volumetric_wall_buffers() {
	fluidx3d::experimental::VolumetricWallNativeCapture capture;
	device.finish_queue();
	capture.available = vw_steps_completed>0ull; // before the first wall step the buffers hold no level (#273)
	capture.t = t;
	if(!capture.available) { capture.clamp_count = vw_download(vw_clamp)[0]; return capture; }
	capture.gpre = vw_download(vw_gpre); capture.f = vw_download(vw_f); capture.post = vw_download(vw_post); capture.fields = vw_download(vw_fields);
	capture.incoming = vw_download(vw_incoming); capture.Q = vw_download(vw_Q); capture.impulse = vw_download(vw_impulse); capture.prestore = vw_download(vw_prestore);
	capture.pending_new = vw_download(vw_pending_slot==0u ? vw_pending0 : vw_pending1);
	capture.pending_old = vw_download(vw_pending_slot==0u ? vw_pending1 : vw_pending0);
	capture.clamp_count = vw_download(vw_clamp)[0];
#ifdef VOLUMETRIC_WALL_SRD
	capture.srd_stats = vw_download(srd_stats); // task 121: cumulative counters of the redistribution
#endif // VOLUMETRIC_WALL_SRD
	if(vw_affine) { capture.support_post = vw_download(vw_spost); capture.returned = vw_download(vw_returned); capture.direct_increment = vw_download(vw_dinc); } // task 55
#ifdef VOLUMETRIC_WALL_RR27
	capture.wall_state = vw_download(vw_wstate); capture.delta = vw_download(vw_delta); capture.wall_state_failures = vw_download(vw_wfail)[0]; // task 98
#ifdef VOLUMETRIC_WALL_ROUGH
	if(vw_affine) capture.facet_law = vw_download(vw_facet_law); // V2: the law as the kernel reads it (2 per facet)
#endif // VOLUMETRIC_WALL_ROUGH
#ifdef VOLUMETRIC_WALL_T119
	capture.t119_out = vw_download(vw_t119_out); capture.t119_negative_q = vw_download(vw_t119_neg)[0]; capture.t119_p1 = vw_download(vw_t119_p1)[0]; // task 119: B0, a (3), L/S per row of the last update; the PR2-trigger count
#endif // VOLUMETRIC_WALL_T119
#endif // VOLUMETRIC_WALL_RR27
	return capture;
}
void LBM_Domain::vw_read_fi(vector<fpxx>& host) { host.resize(fi.range()); device.finish_queue(); device.get_cl_queue().enqueueReadBuffer(fi.get_cl_buffer(), CL_TRUE, 0ull, fi.capacity(), (void*)host.data()); }
void LBM_Domain::vw_write_fi(const vector<fpxx>& host) { if((ulong)host.size()!=fi.range()) print_error("vw_write_fi(): wrong size"); device.finish_queue(); device.get_cl_queue().enqueueWriteBuffer(fi.get_cl_buffer(), CL_TRUE, 0ull, fi.capacity(), (const void*)host.data()); }
void LBM_Domain::vw_read_pending(vector<double>& host) { device.finish_queue(); host = vw_download(vw_pending_slot==0u ? vw_pending0 : vw_pending1); }
void LBM_Domain::vw_write_pending(const vector<double>& host) {
	if((ulong)host.size()!=(ulong)vw_rows) print_error("vw_write_pending(): wrong size");
	Memory<double>& current = vw_pending_slot==0u ? vw_pending0 : vw_pending1;
	for(size_t i=0u; i<host.size(); i++) current[(ulong)i] = host[i];
	current.write_to_device();
}
void LBM::attach_volumetric_wall(const fluidx3d::experimental::VolumetricWallNativeInput& input, const std::vector<double>& friction_rows) { // the common part of both attachments; throws, the callers report
	using namespace fluidx3d::experimental;
	if(initialized) throw std::logic_error("call it before run()");
	if(get_D()!=1u) throw std::logic_error("VOLUMETRIC_WALL requires one domain");
	if(lbm_domain[0]->volumetric_wall_enabled()) throw std::logic_error("a wall is already attached (a second attachment is refused: the first one's TYPE_V flags and buffers would remain)"); // #979
	if(input.box[0]!=(int)Nx||input.box[1]!=(int)Ny||input.box[2]!=(int)Nz) throw std::invalid_argument("the container box differs from the lattice");
	const ulong N = get_N();
	std::vector<std::uint8_t> solid((size_t)N);
	for(ulong n=0ull; n<N; n++) solid[(size_t)n] = (flags[n]&TYPE_S)!=0u ? 1u : 0u; // TYPE_S bit (TYPE_BO is device-side only)
#ifdef VOLUMETRIC_WALL_OPEN_X
	// task 105: the open-x classification of the (periodic) container, the E-cell gradient rows set to zero (O3), then the
	// usual embedding and attachment of that input; the domain keeps the layout for the E tables and the exterior rows
	if(!lbm_domain[0]->volumetric_wall_open_declared()) throw std::invalid_argument("VOLUMETRIC_WALL_OPEN_X: call set_volumetric_wall_open_x() before set_volumetric_wall() (the build has no closed mode)");
	const VolumetricWallOpenLayout layout(input);
	const VolumetricWallNativeInput open_input = layout.without_e_gradients(input);
	const VolumetricWallEmbedding embedding(open_input, solid, layout.e_cells()); // the E cells are R cells by declaration (vw_collide gives them feq3); open_input and solid outlive the embedding
	for(const std::int64_t cell : embedding.r_cells()) flags[embedding.lattice_of_cell()[(size_t)cell]] |= TYPE_V;
	lbm_domain[0]->set_volumetric_wall_open_layout(layout);
	lbm_domain[0]->set_volumetric_wall(open_input, embedding, friction_rows);
#else // VOLUMETRIC_WALL_OPEN_X
	const VolumetricWallEmbedding embedding(input, solid); // input and solid outlive the embedding within this function
	for(const std::int64_t cell : embedding.r_cells()) flags[embedding.lattice_of_cell()[(size_t)cell]] |= TYPE_V;
	lbm_domain[0]->set_volumetric_wall(input, embedding, friction_rows);
#endif // VOLUMETRIC_WALL_OPEN_X
}
void LBM::set_volumetric_wall_open_x(const fluidx3d::experimental::VolumetricWallOpenX& open) { // task 105: the open-x declaration, stored for the attachment; fail closed
#ifdef VOLUMETRIC_WALL_OPEN_X
	try {
		open.validate();
	} catch(const std::exception& error) {
		print_error(string("set_volumetric_wall_open_x(): ")+error.what());
	}
	if(initialized) print_error("set_volumetric_wall_open_x(): call it before run()");
	if(lbm_domain[0]->volumetric_wall_enabled()) print_error("set_volumetric_wall_open_x(): call it before set_volumetric_wall() (the attachment classifies the container with it)");
	if(lbm_domain[0]->volumetric_wall_open_declared()) print_error("set_volumetric_wall_open_x(): the declaration is immutable (a second call is refused)");
	lbm_domain[0]->set_volumetric_wall_open_x(open);
#else // VOLUMETRIC_WALL_OPEN_X
	(void)open;
	print_error("set_volumetric_wall_open_x(): this build has no VOLUMETRIC_WALL_OPEN_X (the open boundary is a compile-time option; use an open target)");
#endif // VOLUMETRIC_WALL_OPEN_X
}
#ifdef VOLUMETRIC_WALL_SGS
void LBM::set_volumetric_wall_sgs(const double k) { // V2: the SGS closure constant, stored for the attachment; fail closed
	if(!std::isfinite(k)||k<0.0) print_error("set_volumetric_wall_sgs(): k must be finite and >= 0");
	if(initialized) print_error("set_volumetric_wall_sgs(): call it before run()");
	if(lbm_domain[0]->volumetric_wall_enabled()) print_error("set_volumetric_wall_sgs(): call it before set_volumetric_wall() (the attachment gives k to the kernels)");
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_volumetric_wall_sgs_k(k); // one domain at the attachment (refused otherwise)
}
#endif // VOLUMETRIC_WALL_SGS
void LBM::set_volumetric_wall(const string& vwnat_path) { // the supported path (task 43): geometry-bound friction rows from the container, coefficient identities checked, fail closed
	using namespace fluidx3d::experimental;
	try {
		const VolumetricWallNativeInput input = VolumetricWallNativeInput::read(vwnat_path);
		input.validate();
#ifdef VOLUMETRIC_WALL_RR27
		if(!input.affine()) throw std::invalid_argument("VOLUMETRIC_WALL_RR27 needs the affine container sections (arms, gradient map) and the friction rows (task 98)"); // task 98 (#1999): before the D3Q19 friction-row validation, so this guard decides
#endif // VOLUMETRIC_WALL_RR27
		if(input.affine()) input.validate_affine(); // task 55: the candidate sections; rows: tensor identity and tangency only (base rows have a shell sum)
		else input.validate_friction_rows(); // absent section, wrong geometry or row order, rows with shell mass: refused
		attach_volumetric_wall(input, input.friction_rows);
		const VolumetricWallNativeInput::FrictionRowResiduals& r = input.friction_row_residuals;
		char text[256] = {};
		std::snprintf(text, sizeof(text), "volumetric wall: friction rows from the container, coefficient identities checked (residuals / A: shell %.3e, impulse tensor %.3e, normal %.3e)", r.shell_sum, r.impulse_tensor, r.normal_component);
		print_info(string(text));
	} catch(const std::exception& error) {
		print_error(string("set_volumetric_wall(): ")+error.what());
	}
}
void LBM::set_volumetric_wall_research(const string& vwnat_path, const string& rows_path) { // archived fixtures: the original kernel expression, or an unvalidated rows file (the former VW_PHI)
	using namespace fluidx3d::experimental;
	try {
		const VolumetricWallNativeInput input = VolumetricWallNativeInput::read(vwnat_path);
		input.validate();
		if(input.affine()) throw std::invalid_argument("the affine candidate container (task 55) is attached only through the supported path set_volumetric_wall");
		std::vector<double> rows_from_file;
		if(!rows_path.empty()) {
			rows_from_file.assign((size_t)input.rows*3u, 0.0);
			std::ifstream in(rows_path, std::ios::binary);
			if(!in) throw std::invalid_argument("cannot open the rows file"); // no path in the message: print_error fails on a long word without spaces
			in.read((char*)rows_from_file.data(), (std::streamsize)(rows_from_file.size()*sizeof(double)));
			if(!in || in.peek()!=std::ifstream::traits_type::eof()) throw std::invalid_argument("the rows file is not rows*3 doubles");
		}
		attach_volumetric_wall(input, rows_from_file);
		if(rows_path.empty()) print_info("volumetric wall RESEARCH path: the original kernel friction expression");
		else print_info("volumetric wall RESEARCH path: friction rows from the file "+rows_path+" (not checked against the geometry)");
	} catch(const std::exception& error) {
		print_error(string("set_volumetric_wall_research(): ")+error.what());
	}
}
#endif // VOLUMETRIC_WALL
