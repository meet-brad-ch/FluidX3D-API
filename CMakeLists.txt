cmake_minimum_required(VERSION 3.27) # 3.27: the Au dependency needs it

project(FluidX3D VERSION 3.5 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(configs)               # options, build type, fluidx3d::build_options, fluidx3d::warnings
include(FluidX3DDependencies)  # pinned third-party code: LodePNG, OpenCL headers, SDFGen
include(FluidX3DPlatform)      # fluidx3d::platform: OpenCL ICD loader, system libraries, X11, threads

add_subdirectory(src/sdf_cache)  # fluidx3d::sdf_cache: SDFs of STL models, generated with SDFGen and cached
add_subdirectory(lib/setup)    # fluidx3d::setup: the Setup API layers that do not need the LBM

include(FluidX3DCore)          # fluidx3d::core and the core source lists
include(FluidX3DExample)       # add_fluidx3d_example()
include(FluidX3DModule)        # add_fluidx3d_module(), with FLUIDX3D_PYTHON_MODULES=ON

if(FLUIDX3D_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
add_subdirectory(examples)

# Detect platform
if(WIN32)
    set(PLATFORM_NAME "Windows")
    set(BUNDLED_LIBS "OpenCL")
elseif(APPLE)
    set(PLATFORM_NAME "macOS")
    set(BUNDLED_LIBS "OpenCL, X11, Xrandr")
elseif(UNIX)
    set(PLATFORM_NAME "Linux")
    set(BUNDLED_LIBS "OpenCL, X11, Xrandr")
else()
    set(PLATFORM_NAME "Unknown")
    set(BUNDLED_LIBS "OpenCL")
endif()

# =============================================================================
# STL file download information
# =============================================================================
find_program(PYTHON_EXECUTABLE NAMES python3 python)
if(PYTHON_EXECUTABLE)
    if(WIN32)
        message(STATUS "To download STL files:")
        message(STATUS "  cd resources && python download_all_thingiverse_stl.py")
    else()
        message(STATUS "To download STL files:")
        message(STATUS "  cd resources && python3 download_all_thingiverse_stl.py")
    endif()
else()
    message(STATUS "Python not found - cannot download STL files automatically")
endif()

# Print configuration summary
message(STATUS "===========================================")
message(STATUS "FluidX3D CMake Configuration")
message(STATUS "===========================================")
message(STATUS "Version:         ${PROJECT_VERSION}")
message(STATUS "Platform:        ${PLATFORM_NAME}")
message(STATUS "Compiler:        ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Architecture:    Unity build per example, Setup API library")
message(STATUS "Using bundled:   ${BUNDLED_LIBS}")
message(STATUS "Tests:           ${FLUIDX3D_BUILD_TESTS}")
message(STATUS "Warnings=errors: ${FLUIDX3D_WERROR}")
message(STATUS "===========================================")

# The host library of the native volumetric slip wall (the VWNAT01 container reader and checks; no OpenCL), its container
# checker and its unit tests. The wall itself is the setup extension VOLUMETRIC_WALL (examples/volumetric_wall,
# research/slip_wall/README.md); it is always compiled into the core, this option only adds the tools and tests.
option(FLUIDX3D_EXPERIMENTAL_VOLUMETRIC_WALL "Build the GPU-free container library, the container checker and the unit tests of the native volumetric slip wall" OFF)
if(FLUIDX3D_EXPERIMENTAL_VOLUMETRIC_WALL)
    add_library(fluidx3d_volumetric_wall_native STATIC src/experimental/volumetric_wall_native.cpp)
    target_include_directories(fluidx3d_volumetric_wall_native PUBLIC src)
    target_link_libraries(fluidx3d_volumetric_wall_native PUBLIC fluidx3d::build_options)
    set_target_properties(fluidx3d_volumetric_wall_native PROPERTIES FLUIDX3D_PRECISE_FLOAT ON)
    add_executable(volumetric_wall_native_check tests/physics/volumetric_wall_native_check.cpp)
    target_link_libraries(volumetric_wall_native_check PRIVATE fluidx3d_volumetric_wall_native)
    set_target_properties(volumetric_wall_native_check PROPERTIES FLUIDX3D_PRECISE_FLOAT ON RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
    # The compiled terrain container builder (scripts/slip_wall/terrain_container.py, builder3d.py). Its results must be
    # bit-identical to the Python builder, so no fast math, and the small dense algebra that Python takes from numpy (the
    # sampler and gradient least squares) uses the same OpenBLAS: scipy-openblas64 0.3.29 equals numpy 2.2.6's bundled
    # library bit for bit on gemm, gesv and gemv (20000 of 20000 stencils, 2026-11-12); plain loops do not (5 of 20000).
    if(NOT WIN32)
        message(FATAL_ERROR "FLUIDX3D_EXPERIMENTAL_VOLUMETRIC_WALL: the terrain container builder pins the win_amd64 OpenBLAS wheel; it is Windows only")
    endif()
    FetchContent_Declare(scipy_openblas64
        URL https://files.pythonhosted.org/packages/2c/e0/c867ebc370df7ccbce3beb2da46427609000ce5d6989b654283069d55627/scipy_openblas64-0.3.29.265.1-py3-none-win_amd64.whl
        URL_HASH SHA256=bdaddfa0cc7fab816f380503085117064641ccc722464c4b900309c77cbe2e96
        DOWNLOAD_NAME scipy_openblas64-0.3.29.265.1.zip  # a wheel is a zip; the name lets CMake extract it
    )
    FetchContent_MakeAvailable(scipy_openblas64)
    add_library(fluidx3d_openblas SHARED IMPORTED GLOBAL) # GLOBAL: tests/ links it too
    set_target_properties(fluidx3d_openblas PROPERTIES
        IMPORTED_LOCATION "${scipy_openblas64_SOURCE_DIR}/scipy_openblas64/lib/libscipy_openblas64_.dll"
        IMPORTED_IMPLIB "${scipy_openblas64_SOURCE_DIR}/scipy_openblas64/lib/libscipy_openblas64_.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${scipy_openblas64_SOURCE_DIR}/scipy_openblas64/include")
    add_library(fluidx3d_wall_builder STATIC
        src/experimental/wall_builder/wide_int.cpp
        src/experimental/wall_builder/correct_rounding.cpp
        src/experimental/wall_builder/dyadic.cpp
        src/experimental/wall_builder/poly.cpp
        src/experimental/wall_builder/terrain.cpp
        src/experimental/wall_builder/passes.cpp
        src/experimental/wall_builder/pass_runner.cpp
        src/experimental/wall_builder/sampler.cpp
        src/experimental/wall_builder/dense_algebra.cpp
        src/experimental/wall_builder/stencil.cpp
        src/experimental/wall_builder/python_text.cpp
        src/experimental/wall_builder/sha256.cpp
        src/experimental/wall_builder/terrain_input.cpp
        src/experimental/wall_builder/maps.cpp
        src/experimental/wall_builder/wall_tables.cpp
        src/experimental/wall_builder/container_writer.cpp)
    target_include_directories(fluidx3d_wall_builder PUBLIC src)
    target_link_libraries(fluidx3d_wall_builder PUBLIC fluidx3d::build_options fluidx3d_openblas bcrypt)
    set_target_properties(fluidx3d_wall_builder PROPERTIES FLUIDX3D_PRECISE_FLOAT ON)
    add_executable(volumetric_wall_builder src/experimental/wall_builder/main.cpp)
    target_link_libraries(volumetric_wall_builder PRIVATE fluidx3d_wall_builder)
    set_target_properties(volumetric_wall_builder PROPERTIES FLUIDX3D_PRECISE_FLOAT ON RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
    add_custom_command(TARGET volumetric_wall_builder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:fluidx3d_openblas> $<TARGET_FILE_DIR:volumetric_wall_builder>)
endif()
