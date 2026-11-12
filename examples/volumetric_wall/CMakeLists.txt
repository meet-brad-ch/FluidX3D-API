# The ordinary setup: rho and u of the final level through LBM::update_fields() (no UPDATE_FIELDS).
add_fluidx3d_example(NAME volumetric_wall
    EXTENSIONS VOLUMETRIC_WALL VOLUME_FORCE
)
# The same setup in the research fixture's configuration: UPDATE_FIELDS writes the fields inside each step, so after
# run(N) they are level N-1 (task 44 compares both; the populations are the same).
add_fluidx3d_example(NAME volumetric_wall_update_fields
    EXTENSIONS VOLUMETRIC_WALL VOLUME_FORCE UPDATE_FIELDS
)
# The wall's host arithmetic is FP64 and must round exactly (no fast floating-point model), as the research fixture.
set_target_properties(volumetric_wall volumetric_wall_setup volumetric_wall_update_fields volumetric_wall_update_fields_setup PROPERTIES FLUIDX3D_PRECISE_FLOAT ON)

# The qualified D3Q27 revision (VOLUMETRIC_WALL_RR27, research/slip_wall/README.md) and its matched baseline.
# volumetric_wall_rr27: the wall, no graphics, no diagnostics (the cost target).
add_fluidx3d_example(NAME volumetric_wall_rr27
    EXTENSIONS D3Q27 VOLUMETRIC_WALL VOLUMETRIC_WALL_FP64_COLLISION VOLUMETRIC_WALL_RR27 VOLUME_FORCE
)
# volumetric_wall_noslip: the same container cells as plain voxels with FluidX3D's bounce-back and its FP32 SRT
# collision (the established baseline; geometry AND collision differ from the wall build). No graphics (the cost target).
add_fluidx3d_example(NAME volumetric_wall_noslip
    EXTENSIONS D3Q27 VOLUME_FORCE
)
# The visual pair: the same two builds with GRAPHICS (frames at VOLUMETRIC_WALL_FRAMES) and the same fixed colour scale.
# GRAPHICS_U_MAX 0.008f (scientist #2063): a presentation scale with headroom over the flat-channel bulk speed estimate
# (about 0.0055 at force 1e-5, nu 0.1); not an acceptance threshold and not a prediction of the terrain flow.
add_fluidx3d_example(NAME volumetric_wall_rr27_visual
    EXTENSIONS D3Q27 VOLUMETRIC_WALL VOLUMETRIC_WALL_FP64_COLLISION VOLUMETRIC_WALL_RR27 VOLUME_FORCE GRAPHICS UPDATE_FIELDS
    SETTINGS "GRAPHICS_U_MAX 0.008f"
)
add_fluidx3d_example(NAME volumetric_wall_noslip_visual
    EXTENSIONS D3Q27 VOLUME_FORCE GRAPHICS UPDATE_FIELDS
    SETTINGS "GRAPHICS_U_MAX 0.008f"
)
# V2: the same wall with the static Smagorinsky rate in every cell (VOLUMETRIC_WALL_SGS; VOLUMETRIC_WALL_SGS_K sets k). No
# graphics. Without VOLUMETRIC_WALL_SGS_K it runs with k = 0 (the V1 rate).
add_fluidx3d_example(NAME volumetric_wall_rr27_sgs
    EXTENSIONS D3Q27 VOLUMETRIC_WALL VOLUMETRIC_WALL_FP64_COLLISION VOLUMETRIC_WALL_RR27 VOLUMETRIC_WALL_SGS VOLUME_FORCE
)
set_target_properties(volumetric_wall_rr27_sgs volumetric_wall_rr27_sgs_setup PROPERTIES FLUIDX3D_PRECISE_FLOAT ON)
# V2: the SGS wall with the facetwise rough friction (VOLUMETRIC_WALL_ROUGH). The law comes from the container section
# facet_friction (scripts/slip_wall/terrain_container.py --rough-z0); a container without it runs with the scalar K.
add_fluidx3d_example(NAME volumetric_wall_rr27_sgs_rough
    EXTENSIONS D3Q27 VOLUMETRIC_WALL VOLUMETRIC_WALL_FP64_COLLISION VOLUMETRIC_WALL_RR27 VOLUMETRIC_WALL_SGS VOLUMETRIC_WALL_ROUGH VOLUME_FORCE
)
set_target_properties(volumetric_wall_rr27_sgs_rough volumetric_wall_rr27_sgs_rough_setup PROPERTIES FLUIDX3D_PRECISE_FLOAT ON)
# V2 task 105: the same rough SGS wall with the open x axis (VOLUMETRIC_WALL_OPEN_X: E cells at x = 0 and Nx - 1, exterior
# facets, the LODI E state). A compile-time option: this target requires VOLUMETRIC_WALL_OPEN=x and the open declaration, and
# refuses a nonzero force; every other target is the closed build and refuses VOLUMETRIC_WALL_OPEN.
add_fluidx3d_example(NAME volumetric_wall_rr27_sgs_rough_open
    EXTENSIONS D3Q27 VOLUMETRIC_WALL VOLUMETRIC_WALL_FP64_COLLISION VOLUMETRIC_WALL_RR27 VOLUMETRIC_WALL_SGS VOLUMETRIC_WALL_ROUGH VOLUMETRIC_WALL_OPEN_X VOLUME_FORCE
)
set_target_properties(volumetric_wall_rr27_sgs_rough_open volumetric_wall_rr27_sgs_rough_open_setup PROPERTIES FLUIDX3D_PRECISE_FLOAT ON)
# V2 task 107: the actual-hill example. Two modes with the same terrain cells, inflow speed, viscosity, camera and colour
# scale; VOLUMETRIC_WALL_VIEW=hill draws the terrain, the streamlines and the mode label. *_visual writes frames (GRAPHICS);
# *_live opens the interactive window (INTERACTIVE_GRAPHICS; P starts the run).
# GRAPHICS_U_MAX 0.14f (units of c): a presentation scale of |u| from 0 to 0.081, headroom over the inflow speed 0.05 of the
# example; not an acceptance threshold and not a prediction of the hill flow.
# The wall: the open rough SGS build above with graphics and the qualified macro set of the V2 hill run (task 128): the whole-prism
# correction (VOLUMETRIC_WALL_T119) and the conservative state redistribution (VOLUMETRIC_WALL_SRD). These two macros need a container with the
# t119_gdelta/t119_wmoment and t121_* sections (research/slip_wall/README.md); any other container is refused at the attachment.
add_fluidx3d_example(NAME volumetric_wall_rr27_sgs_rough_open_visual
    EXTENSIONS D3Q27 VOLUMETRIC_WALL VOLUMETRIC_WALL_FP64_COLLISION VOLUMETRIC_WALL_RR27 VOLUMETRIC_WALL_SGS VOLUMETRIC_WALL_ROUGH VOLUMETRIC_WALL_OPEN_X VOLUMETRIC_WALL_T119 VOLUMETRIC_WALL_SRD VOLUME_FORCE GRAPHICS UPDATE_FIELDS
    SETTINGS "GRAPHICS_U_MAX 0.14f" "GRAPHICS_STREAMLINE_SPARSE 10u"
)
add_fluidx3d_example(NAME volumetric_wall_rr27_sgs_rough_open_live
    EXTENSIONS D3Q27 VOLUMETRIC_WALL VOLUMETRIC_WALL_FP64_COLLISION VOLUMETRIC_WALL_RR27 VOLUMETRIC_WALL_SGS VOLUMETRIC_WALL_ROUGH VOLUMETRIC_WALL_OPEN_X VOLUMETRIC_WALL_T119 VOLUMETRIC_WALL_SRD VOLUME_FORCE INTERACTIVE_GRAPHICS
    SETTINGS "GRAPHICS_U_MAX 0.14f" "GRAPHICS_STREAMLINE_SPARSE 10u"
)
# The no-slip voxel baseline: the same container cells as voxels with FluidX3D's bounce-back, its FP32 SRT collision with
# SUBGRID, and EQUILIBRIUM_BOUNDARIES cells at x = 0 and x = Nx - 1 (rho 1, the inlet velocity). Geometry, wall, collision,
# subgrid closure and outlet rule all differ from the wall build.
add_fluidx3d_example(NAME volumetric_wall_noslip_open_visual
    EXTENSIONS D3Q27 SUBGRID EQUILIBRIUM_BOUNDARIES VOLUME_FORCE GRAPHICS UPDATE_FIELDS
    SETTINGS "GRAPHICS_U_MAX 0.14f" "GRAPHICS_STREAMLINE_SPARSE 10u"
)
add_fluidx3d_example(NAME volumetric_wall_noslip_open_live
    EXTENSIONS D3Q27 SUBGRID EQUILIBRIUM_BOUNDARIES VOLUME_FORCE INTERACTIVE_GRAPHICS
    SETTINGS "GRAPHICS_U_MAX 0.14f" "GRAPHICS_STREAMLINE_SPARSE 10u"
)
set_target_properties(volumetric_wall_rr27_sgs_rough_open_visual volumetric_wall_rr27_sgs_rough_open_visual_setup
    volumetric_wall_rr27_sgs_rough_open_live volumetric_wall_rr27_sgs_rough_open_live_setup
    volumetric_wall_noslip_open_visual volumetric_wall_noslip_open_visual_setup
    volumetric_wall_noslip_open_live volumetric_wall_noslip_open_live_setup
    PROPERTIES FLUIDX3D_PRECISE_FLOAT ON)
set_target_properties(volumetric_wall_rr27 volumetric_wall_rr27_setup volumetric_wall_noslip volumetric_wall_noslip_setup
    volumetric_wall_rr27_visual volumetric_wall_rr27_visual_setup volumetric_wall_noslip_visual volumetric_wall_noslip_visual_setup
    PROPERTIES FLUIDX3D_PRECISE_FLOAT ON)
