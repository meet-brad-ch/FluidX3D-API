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
set_target_properties(volumetric_wall_rr27 volumetric_wall_rr27_setup volumetric_wall_noslip volumetric_wall_noslip_setup
    volumetric_wall_rr27_visual volumetric_wall_rr27_visual_setup volumetric_wall_noslip_visual volumetric_wall_noslip_visual_setup
    PROPERTIES FLUIDX3D_PRECISE_FLOAT ON)
