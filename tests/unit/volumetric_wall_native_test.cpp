// Task 19 (#265): fail-closed tests of the VWNAT01 container and the embedding on the REGISTERED N16 container
// (research/wall_model/integration/native_inputs/N16_decay_finite.vwnat, path from FLUIDX3D_VWNAT_FIXTURE): the
// writer/reader round trip, trailing bytes and impossible lengths, mutated coefficients/directions/geometry/scalars
// rejected by validate(), and the embedding counts of the registered oracle. No GPU, no physics.
#include "experimental/volumetric_wall_native.hpp"
#include "../physics/volumetric_wall_native_protocol.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace fluidx3d::experimental;

namespace {

#ifndef FLUIDX3D_VWNAT_FIXTURE
#error FLUIDX3D_VWNAT_FIXTURE (path of the registered N16 container) must be defined by CMake
#endif
#ifndef FLUIDX3D_VWNAT27_FIXTURE
#error FLUIDX3D_VWNAT27_FIXTURE (path of the D3Q27 M3 container) must be defined by CMake
#endif

VolumetricWallNativeInput fixture() {
    return VolumetricWallNativeInput::read(FLUIDX3D_VWNAT_FIXTURE);   // throws (test fails) when the registered input is missing
}

std::string temp_path(const char* name) {
    const char* temp{ std::getenv("TEMP") };
    return std::string(temp ? temp : ".")+"/"+name;
}

}  // namespace

TEST(VolumetricWallNative, DirectionPermutationMatchesContractTable) {
    const NativeDirections dir{};
    const int expected[19]{ 9, 16, 2, 12, 6, 10, 8, 18, 0, 17, 1, 13, 5, 14, 4, 15, 3, 11, 7 };
    for(std::uint32_t k=0u; k<19u; k++) {
        EXPECT_EQ(dir.fx_to_lex(k), static_cast<std::uint32_t>(expected[k]));
        EXPECT_EQ(dir.lex_to_fx(dir.fx_to_lex(k)), k);
    }
}

TEST(VolumetricWallNative, AddressAlgebraOnSmallBox) {
    const NativeDirections dir{};
    const NativeAddresses addr({ 4, 3, 2 }, dir);   // the object owns its tables (#265 item 3)
    for(std::uint64_t n=0u; n<addr.cells(); n++) {
        for(std::uint32_t parity=0u; parity<2u; parity++) {
            for(const std::uint64_t a : addr.own_set(n, parity)) {
                EXPECT_EQ(addr.native_writer(a, parity), n);
            }
            for(std::uint32_t k=1u; k<19u; k++) {
                const std::uint32_t back{ k%2u==1u ? k+1u : k-1u };
                const auto donor_set{ addr.own_set(addr.neighbour(n, back), parity) };
                const std::uint64_t a{ addr.next_state_address(n, k, parity) };
                EXPECT_NE(std::find(donor_set.begin(), donor_set.end(), a), donor_set.end());
            }
        }
    }
}

// Task 98: the D3Q27 tables. The FluidX3D table is the c_data D3Q27 branch of src/kernel.cpp, copied here row by row.
TEST(VolumetricWallNative, D3Q27DirectionTablesMatchKernel) {
    const NativeDirections dir(27u);
    ASSERT_EQ(dir.q(), 27u);
    const int kernel_x[27]{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
    const int kernel_y[27]{ 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
    const int kernel_z[27]{ 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };
    const std::vector<int> flat{ dir.fx_table_flat() };
    ASSERT_EQ(flat.size(), 81u);
    double weight_sum{ 0.0 };
    for(std::uint32_t k=0u; k<27u; k++) {
        EXPECT_EQ(flat[k], kernel_x[k]); EXPECT_EQ(flat[27u+k], kernel_y[k]); EXPECT_EQ(flat[54u+k], kernel_z[k]);
        const std::uint32_t d{ dir.fx_to_lex(k) };
        EXPECT_EQ(dir.lex_to_fx(d), k);
        for(std::uint32_t a=0u; a<3u; a++) EXPECT_EQ(dir.lex_c(d, a), dir.fx_c(k, a));   // the same vector in both tables
        weight_sum += dir.weight_lex(d);
    }
    EXPECT_NEAR(weight_sum, 1.0, 1e-15);
    for(std::uint32_t i=1u; i<27u; i+=2u) {   // the FluidX3D pairs (i, i+1) are opposite
        for(std::uint32_t a=0u; a<3u; a++) EXPECT_EQ(dir.fx_c(i, a), -dir.fx_c(i+1u, a));
    }
    // lexicographic: x slowest, z fastest; rest 13; opposite 26 - d; the double weights of vw_weight
    for(std::uint32_t d=0u; d<27u; d++) {
        EXPECT_EQ(dir.lex_c(d, 0u), static_cast<int>(d/9u)-1); EXPECT_EQ(dir.lex_c(d, 1u), static_cast<int>((d/3u)%3u)-1); EXPECT_EQ(dir.lex_c(d, 2u), static_cast<int>(d%3u)-1);
        EXPECT_EQ(dir.lex_opposite(d), 26u-d);
        const int shell{ dir.lex_c(d, 0u)*dir.lex_c(d, 0u)+dir.lex_c(d, 1u)*dir.lex_c(d, 1u)+dir.lex_c(d, 2u)*dir.lex_c(d, 2u) };
        const double expected{ shell==0 ? 8.0/27.0 : (shell==1 ? 2.0/27.0 : (shell==2 ? 1.0/54.0 : 1.0/216.0)) };
        EXPECT_EQ(dir.weight_lex(d), expected);
    }
    // the D3Q19 tables are unchanged by the parameter (default constructor = D3Q19, weights as before)
    const NativeDirections d19{};
    EXPECT_EQ(d19.q(), 19u);
    for(std::uint32_t d=0u; d<19u; d++) EXPECT_EQ(d19.weight_lex(d), NativeDirections::rest_weight_lex(d));
}

TEST(VolumetricWallNative, D3Q27AddressAlgebraOnSmallBox) {
    const NativeDirections dir(27u);
    const NativeAddresses addr({ 4, 3, 2 }, dir);
    for(std::uint64_t n=0u; n<addr.cells(); n++) {
        for(std::uint32_t parity=0u; parity<2u; parity++) {
            const auto own{ addr.own_set(n, parity) };
            ASSERT_EQ(own.size(), 27u);
            for(const std::uint64_t a : own) {
                EXPECT_EQ(addr.native_writer(a, parity), n);
            }
            for(std::uint32_t k=1u; k<27u; k++) {
                const std::uint32_t back{ k%2u==1u ? k+1u : k-1u };
                const auto donor_set{ addr.own_set(addr.neighbour(n, back), parity) };
                const std::uint64_t a{ addr.next_state_address(n, k, parity) };
                EXPECT_NE(std::find(donor_set.begin(), donor_set.end(), a), donor_set.end());
            }
        }
    }
}

// Task 98: the D3Q27 M3 container (tests/data/slip_wall/M3_d3q27.vwnat, FLUIDX3D_VWNAT27_FIXTURE; never skipped):
// q from the D shape, validate(), validate_affine() (base rows: tensor identity and tangency), and the ownership verification
// of the embedding (every non-mesh cell solid).
TEST(VolumetricWallNative, D3Q27TerrainContainerOwnership) {
    const char* path{ FLUIDX3D_VWNAT27_FIXTURE };   // shipped fixture (tests/data/slip_wall); a missing file fails the read
    const VolumetricWallNativeInput in{ VolumetricWallNativeInput::read(path) };
    EXPECT_EQ(in.velocity_set, 27u);
    EXPECT_EQ(in.rest_direction(), 13u);
    ASSERT_NO_THROW(in.validate());
    ASSERT_NO_THROW(in.validate_affine());
    EXPECT_EQ(in.cells, 272u);
    EXPECT_EQ(in.facets, 248u);
    EXPECT_EQ(in.rows, 2856u);
    EXPECT_EQ(in.W.indices.size(), 152u);
    const NativeAddresses addr(in.box, NativeDirections(27u));
    std::vector<std::uint8_t> solid(static_cast<size_t>(addr.cells()), 1u);
    for(std::uint32_t c=0u; c<in.cells; c++) solid[static_cast<size_t>(addr.cell(in.cell_xyz[3u*c], in.cell_xyz[3u*c+1u], in.cell_xyz[3u*c+2u]))] = 0u;
    const VolumetricWallEmbedding embedding(in, solid);   // throws when the ownership verification fails
    EXPECT_TRUE(embedding.report().pass()) << embedding.report().text();
    std::printf("D3Q27 M3 embedding: R cells %zu, owned rows %zu, %s\n", embedding.r_cells().size(), embedding.owned().size(), embedding.report().text().c_str());
}

// Task 98 (#1991): a fully occluded row (no fluid gather part) is refused by validate_affine (unsupported geometry), not
// left to a 0/0 in the kernels. The M3 D3Q27 container with the G volumes of row 0 set to zero.
TEST(VolumetricWallNative, D3Q27FullyOccludedRowRefused) {
    const char* path{ FLUIDX3D_VWNAT27_FIXTURE };   // shipped fixture (tests/data/slip_wall); a missing file fails the read
    VolumetricWallNativeInput in{ VolumetricWallNativeInput::read(path) };
    ASSERT_NO_THROW(in.validate_affine());
    ASSERT_LT(in.G.offsets[0], in.G.offsets[1]);
    for(std::uint32_t z=in.G.offsets[0]; z<in.G.offsets[1]; z++) in.G.weights[z] = 0.0;
    try {
        in.validate_affine();
        FAIL() << "a fully occluded row was accepted";
    } catch(const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("no fluid gather part"), std::string::npos) << error.what();
    }
}

TEST(VolumetricWallNative, RegisteredInputValidates) {
    const VolumetricWallNativeInput in{ fixture() };
    ASSERT_NO_THROW(in.validate());
    EXPECT_EQ(in.cells, 1204u);
    EXPECT_EQ(in.facets, 368u);
    EXPECT_EQ(in.rows, 2944u);
    EXPECT_LE(in.residuals.donor_partition, 1e-10);
    EXPECT_LE(in.residuals.receiver_coverage, 1e-10);
    EXPECT_LE(in.residuals.prism_coverage, 1e-10);
    EXPECT_LE(in.residuals.return_delivery, 1e-10);
    EXPECT_LE(in.residuals.sampler_constant, 1e-10);
    EXPECT_LE(in.residuals.sampler_affine, 1e-10);
}

TEST(VolumetricWallNative, RoundTripIsBitwise) {
    const VolumetricWallNativeInput in{ fixture() };
    const std::string path{ temp_path("vwnat_roundtrip.vwnat") };
    in.write(path);
    const VolumetricWallNativeInput back{ VolumetricWallNativeInput::read(path) };
    EXPECT_EQ(back.box, in.box);
    EXPECT_EQ(back.cell_xyz, in.cell_xyz);
    EXPECT_EQ(back.volume, in.volume);
    EXPECT_EQ(back.centroid, in.centroid);
    EXPECT_EQ(back.sample_point, in.sample_point);
    EXPECT_EQ(back.normal, in.normal);
    EXPECT_EQ(back.facet_offsets, in.facet_offsets);
    EXPECT_EQ(back.row_direction, in.row_direction);
    EXPECT_EQ(back.prism_volume, in.prism_volume);
    EXPECT_EQ(back.row_origin, in.row_origin);
    EXPECT_EQ(back.D.weights, in.D.weights);
    EXPECT_EQ(back.D.indices, in.D.indices);
    EXPECT_EQ(back.G.weights, in.G.weights);
    EXPECT_EQ(back.R.weights, in.R.weights);
    EXPECT_EQ(back.S.weights, in.S.weights);
    EXPECT_EQ(back.W.weights, in.W.weights);
    EXPECT_EQ(back.f0, in.f0);
    EXPECT_EQ(back.s0, in.s0);
    EXPECT_EQ(back.provenance, in.provenance);
    ASSERT_NO_THROW(back.validate());
    std::remove(path.c_str());
}

TEST(VolumetricWallNative, NonzeroFrictionRoundTripsAndValidates) {   // task 21: K is a registered scalar of the finite-friction scope
    VolumetricWallNativeInput in{ fixture() };
    in.friction = 0.01875;   // K16 = 0.3 / 16
    ASSERT_NO_THROW(in.validate());
    const std::string path{ temp_path("vwnat_friction.vwnat") };
    in.write(path);
    const VolumetricWallNativeInput back{ VolumetricWallNativeInput::read(path) };
    EXPECT_EQ(back.friction, 0.01875);
    ASSERT_NO_THROW(back.validate());
    in.friction = -1.0;
    EXPECT_THROW(in.validate(), std::invalid_argument);
    std::remove(path.c_str());
}

TEST(VolumetricWallNative, SeriesSchemaV5Names) {   // task 21: the 73 v4 columns first, then the 15 new ones
    const auto& names{ native_protocol::SeriesRow::names() };
    ASSERT_EQ(names.size(), 88u);
    EXPECT_EQ(names[0], "step");
    EXPECT_EQ(names[72], "storage_allowance_momz");
    EXPECT_EQ(names[73], "requested_tangential_impulse_x");
    EXPECT_EQ(names[79], "delivered_tangential_impulse_x");
    EXPECT_EQ(names[87], "traction_identity_bound");
    EXPECT_EQ(std::count(names.begin(), names.end(), "Jz_cum"), 1);
}

// ---- task 22: all-axis periodic sampler validation (NATIVE_PERIODICITY_CONTRACT U1-U4) ----------------------------
namespace {

/// @brief The accepted (1a3aed3) sampler affine residual: minimal image on axis 2 only, std::round; reference for U1/U2.
double accepted_sampler_affine(const VolumetricWallNativeInput& in) {
    double worst{ 0.0 };
    for(std::uint32_t a=0u; a<in.facets; a++) {
        std::array<double, 3> moment{};
        for(std::uint32_t z=in.S.offsets[a]; z<in.S.offsets[a+1u]; z++) {
            for(std::uint32_t axis=0u; axis<3u; axis++) {
                double delta{ in.centroid[3u*in.S.indices[z]+axis]-in.sample_point[3u*a+axis] };
                if(axis==2u) {
                    const double period{ static_cast<double>(in.box[2]) };
                    delta -= period*std::round(delta/period);
                }
                moment[axis] += in.S.weights[z]*delta;
            }
        }
        for(const double m : moment) { worst = std::max(worst, std::fabs(m)); }
    }
    return worst;
}

}  // namespace

TEST(VolumetricWallNative, U4MinimalImageHalfUpRuleAndHalfPeriodRejection) {
    EXPECT_EQ(minimal_image(0.25, 4.0, 0u), 0.25);
    EXPECT_EQ(minimal_image(3.75, 4.0, 0u), -0.25);
    EXPECT_EQ(minimal_image(-3.75, 4.0, 1u), 0.25);
    EXPECT_EQ(minimal_image(1.0, 4.0, 2u), 1.0);      // the accepted duct z stencils (|delta_z| = 1, box 4) stay valid
    EXPECT_EQ(minimal_image(-1.0, 4.0, 2u), -1.0);
    EXPECT_THROW(minimal_image(2.0, 4.0, 0u), std::invalid_argument);    // exact +period/2: ambiguous
    EXPECT_THROW(minimal_image(-2.0, 4.0, 0u), std::invalid_argument);
    EXPECT_THROW(minimal_image(1.0, 0.0, 0u), std::invalid_argument);
    // #399: representable values immediately inside and outside both half periods are wrapped exactly, never onto the tie
    const double in_p{ std::nextafter(2.0, 0.0) }, out_p{ std::nextafter(2.0, 4.0) }, in_m{ std::nextafter(-2.0, 0.0) }, out_m{ std::nextafter(-2.0, -4.0) };
    EXPECT_EQ(minimal_image(in_p, 4.0, 0u), in_p);
    EXPECT_EQ(minimal_image(in_m, 4.0, 0u), in_m);
    EXPECT_EQ(minimal_image(out_p, 4.0, 0u), out_p-4.0);
    EXPECT_EQ(minimal_image(out_m, 4.0, 0u), out_m+4.0);
    EXPECT_LT(std::fabs(minimal_image(out_p, 4.0, 0u)), 2.0);
    // #399: non-finite inputs rejected
    const double nan{ std::nan("") }, inf{ HUGE_VAL };
    EXPECT_THROW(minimal_image(nan, 4.0, 0u), std::invalid_argument);
    EXPECT_THROW(minimal_image(inf, 4.0, 0u), std::invalid_argument);
    EXPECT_THROW(minimal_image(-inf, 4.0, 0u), std::invalid_argument);
    EXPECT_THROW(minimal_image(1.0, nan, 0u), std::invalid_argument);
    EXPECT_THROW(minimal_image(1.0, inf, 0u), std::invalid_argument);
}

TEST(VolumetricWallNative, U5SubtractionOverflowRejected) {   // #399: finite coordinates whose difference overflows
    VolumetricWallNativeInput in{ fixture() };
    in.centroid[3u*in.S.indices[in.S.offsets[0]]] = 1.7e308;
    in.sample_point[0] = -1.7e308;
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

// U1 (the bitwise sampler residual of the 17 accepted research containers, VWNAT_U1_CONTAINERS) is a research-run test:
// it stays on the research branch (volumetric-wall-model) and is not part of the product suite (scientist #2025).

TEST(VolumetricWallNative, U2SeamCrossingStencilValidates) {
    VolumetricWallNativeInput in{ fixture() };
    ASSERT_NO_THROW(in.validate());
    const double before{ in.residuals.sampler_affine };
    const double box_x{ static_cast<double>(in.box[0]) };
    for(std::uint32_t z=in.S.offsets[0]; z<in.S.offsets[1]; z++) in.centroid[3u*in.S.indices[z]] -= box_x;   // facet 0's stencil one x period away
    ASSERT_NO_THROW(in.validate());
    EXPECT_LE(in.residuals.sampler_affine, 1e-10);
    EXPECT_NEAR(in.residuals.sampler_affine, before, 1e-12);
    EXPECT_GT(accepted_sampler_affine(in), 1.0);        // the z-only validator could not validate this stencil
}

TEST(VolumetricWallNative, U3HalfPeriodStencilRejected) {
    VolumetricWallNativeInput in{ fixture() };
    const double box_x{ static_cast<double>(in.box[0]) };
    in.sample_point[0] = std::floor(in.sample_point[0]);                   // exact values: the displacement is exactly box/2
    in.centroid[3u*in.S.indices[in.S.offsets[0]]] = in.sample_point[0]+0.5*box_x;
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, TrailingBytesRejected) {
    const VolumetricWallNativeInput in{ fixture() };
    const std::string path{ temp_path("vwnat_trailing.vwnat") };
    in.write(path);
    { std::ofstream app(path, std::ios::binary|std::ios::app); const char junk[3]{ 1, 2, 3 }; app.write(junk, 3); }
    EXPECT_THROW(VolumetricWallNativeInput::read(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(VolumetricWallNative, ImpossibleLengthRejectedBeforeAllocation) {
    const VolumetricWallNativeInput in{ fixture() };
    const std::string path{ temp_path("vwnat_length.vwnat") };
    in.write(path);
    { // the element count of the first section (8 magic + 4 count + 32 name + 4 dtype = offset 48) made impossible
        std::fstream f(path, std::ios::binary|std::ios::in|std::ios::out);
        f.seekp(48);
        const std::uint64_t huge{ 0xFFFFFFFFFFFFull };
        f.write(reinterpret_cast<const char*>(&huge), 8);
    }
    EXPECT_THROW(VolumetricWallNativeInput::read(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(VolumetricWallNative, MutatedDirectCoefficientRejected) {
    VolumetricWallNativeInput in{ fixture() };
    in.D.weights[1000] *= 1.0+1e-6;   // breaks donor partition / receiver coverage beyond 1e-10
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, MutatedGatherCoefficientRejected) {
    VolumetricWallNativeInput in{ fixture() };
    in.G.weights[100] *= 1.0+1e-6;    // breaks donor partition and prism coverage
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, MutatedDirectDirectionRejected) {
    VolumetricWallNativeInput in{ fixture() };
    const std::uint32_t state{ in.D.indices[1000] };
    in.D.indices[1000] = (state/19u)*19u+((state%19u+1u)%19u);   // the entry changes the direction
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, MutatedGatherDirectionRejected) {
    VolumetricWallNativeInput in{ fixture() };
    const std::uint32_t state{ in.G.indices[100] };
    in.G.indices[100] = (state/19u)*19u+((state%19u+1u)%19u);   // gathers a direction other than OPP(outgoing)
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, MutatedRowDirectionRejected) {
    VolumetricWallNativeInput in{ fixture() };
    const NativeDirections dir{};
    in.row_direction[0] = dir.lex_opposite(in.row_direction[0]);   // no longer outgoing (c.n < 0)
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, MutatedPrismVolumeRejected) {
    VolumetricWallNativeInput in{ fixture() };
    in.prism_volume[5] *= 1.0+1e-9;   // vi != A (c.n)
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, MutatedNormalRejected) {
    VolumetricWallNativeInput in{ fixture() };
    in.normal[0] *= 1.0+1e-6;         // not a unit normal
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, MutatedSamplerRejected) {
    VolumetricWallNativeInput in{ fixture() };
    in.S.weights[0] += 1e-6;          // constant reproduction broken
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, MalformedInteriorFacetOffsetRejected) {
    VolumetricWallNativeInput in{ fixture() };
    in.facet_offsets[1] = in.rows+5u;   // interior offset above rows while the last offset still equals rows
    EXPECT_THROW(in.validate(), std::invalid_argument);
    VolumetricWallNativeInput decreasing{ fixture() };
    decreasing.facet_offsets[2] = decreasing.facet_offsets[1]-1u;   // decreasing offsets
    EXPECT_THROW(decreasing.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, FacetMissingOneShellRejected) {
    VolumetricWallNativeInput in{ fixture() };
    const NativeDirections dir{};
    // facet 0: move every shell-2 row onto a shell-1 outgoing direction's... a facet keeps its row count, so give its
    // shell-2 rows the shell-1 direction of its first row (duplicates are rejected too, but the shell check comes
    // first in no particular order: the test only requires rejection)
    const std::uint32_t begin{ in.facet_offsets[0] }, end{ in.facet_offsets[1] };
    std::uint32_t shell1_direction{ 19u };
    for(std::uint32_t row=begin; row<end; row++) {
        const std::uint32_t d{ in.row_direction[row] };
        if(dir.lex_c(d, 0)*dir.lex_c(d, 0)+dir.lex_c(d, 1)*dir.lex_c(d, 1)+dir.lex_c(d, 2)*dir.lex_c(d, 2)==1) { shell1_direction = d; }
    }
    ASSERT_LT(shell1_direction, 19u);
    for(std::uint32_t row=begin; row<end; row++) {
        const std::uint32_t d{ in.row_direction[row] };
        if(dir.lex_c(d, 0)*dir.lex_c(d, 0)+dir.lex_c(d, 1)*dir.lex_c(d, 1)+dir.lex_c(d, 2)*dir.lex_c(d, 2)==2) { in.row_direction[row] = shell1_direction; }
    }
    EXPECT_THROW(in.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, NegativePendingAndInconsistentOmegaRejected) {
    VolumetricWallNativeInput negative{ fixture() };
    negative.s0[0] = -1e-3;
    EXPECT_THROW(negative.validate(), std::invalid_argument);
    VolumetricWallNativeInput omega{ fixture() };
    omega.omega = 1.3;
    EXPECT_THROW(omega.validate(), std::invalid_argument);
    VolumetricWallNativeInput volume{ fixture() };
    volume.volume[0] = 1.0+1e-12;
    EXPECT_THROW(volume.validate(), std::invalid_argument);
}

TEST(VolumetricWallNative, EmbeddingReproducesTheRegisteredCounts) {
    const VolumetricWallNativeInput in{ fixture() };
    const NativeAddresses addr(in.box, NativeDirections{});
    std::vector<std::uint8_t> solid(static_cast<size_t>(addr.cells()), 1u);
    for(std::uint32_t c=0u; c<in.cells; c++) {
        solid[static_cast<size_t>(addr.cell(in.cell_xyz[3u*c], in.cell_xyz[3u*c+1u], in.cell_xyz[3u*c+2u]))] = 0u;
    }
    const VolumetricWallEmbedding embedding(in, solid);
    EXPECT_TRUE(embedding.report().pass());
    EXPECT_EQ(embedding.r_cells().size(), 644u);          // structural_check.py and the independent oracle, N16
    EXPECT_EQ(embedding.r_rows().size(), 12236u);
    EXPECT_EQ(embedding.native_rows().size(), 10640u);
    EXPECT_EQ(embedding.report().inactive_old_owner_permitted, 3072u);
    EXPECT_EQ(embedding.report().near_unit_rows, 1572u);
    EXPECT_EQ(embedding.closure_added(), 0u);
}

TEST(VolumetricWallNative, EmbeddingRejectsAMeshCellFlaggedSolid) {
    const VolumetricWallNativeInput in{ fixture() };
    const NativeAddresses addr(in.box, NativeDirections{});
    std::vector<std::uint8_t> solid(static_cast<size_t>(addr.cells()), 1u);
    for(std::uint32_t c=1u; c<in.cells; c++) {   // cell 0 stays solid in the flags
        solid[static_cast<size_t>(addr.cell(in.cell_xyz[3u*c], in.cell_xyz[3u*c+1u], in.cell_xyz[3u*c+2u]))] = 0u;
    }
    EXPECT_THROW(VolumetricWallEmbedding(in, solid), std::invalid_argument);
}

// ---------------------------------------------------------------- Task 43: the friction_rows section (supported path)
#ifndef FLUIDX3D_VWNAT_P2_FIXTURE
#error FLUIDX3D_VWNAT_P2_FIXTURE (path of the N16 container with P2 friction rows) must be defined by CMake
#endif

namespace {

VolumetricWallNativeInput p2_fixture() {
    return VolumetricWallNativeInput::read(FLUIDX3D_VWNAT_P2_FIXTURE);   // integration/task43/inputs, written by vwnat_rows.py
}

/// @brief The kernel's original friction expression as rows: phi_r = 6 w_r vi_r^2 / A (c_r - n (c_r . n)).
std::vector<double> original_rows(const VolumetricWallNativeInput& in) {
    const NativeDirections dir{};
    std::vector<double> phi(static_cast<size_t>(in.rows)*3u, 0.0);
    for(std::uint32_t a=0u; a<in.facets; a++) {
        for(std::uint32_t row=in.facet_offsets[a]; row<in.facet_offsets[a+1u]; row++) {
            const std::uint32_t d{ in.row_direction[row] };
            const double cn{ dir.lex_c(d, 0u)*in.normal[3u*a]+dir.lex_c(d, 1u)*in.normal[3u*a+1u]+dir.lex_c(d, 2u)*in.normal[3u*a+2u] };
            const double scale{ 6.0*NativeDirections::rest_weight_lex(d)*in.prism_volume[row]*in.prism_volume[row]/in.area[a] };
            for(std::uint32_t j=0u; j<3u; j++) {
                phi[3u*row+j] = scale*(dir.lex_c(d, j)-in.normal[3u*a+j]*cn);
            }
        }
    }
    return phi;
}

}  // namespace

TEST(VolumetricWallNativeRows, ContainerWithoutRowsIsValidButRefusedByTheSupportedPath) {
    const VolumetricWallNativeInput in{ fixture() };
    EXPECT_TRUE(in.friction_rows.empty());
    ASSERT_NO_THROW(in.validate());                                   // archived containers stay valid VWNAT01 files
    EXPECT_THROW(in.validate_friction_rows(), std::invalid_argument); // no fallback on the supported path
}

TEST(VolumetricWallNativeRows, P2ContainerValidatesAndKeepsTheParentSections) {
    const VolumetricWallNativeInput in{ p2_fixture() };
    const VolumetricWallNativeInput parent{ fixture() };
    ASSERT_EQ(in.friction_rows.size(), static_cast<size_t>(in.rows)*3u);
    ASSERT_NO_THROW(in.validate());
    ASSERT_NO_THROW(in.validate_friction_rows());
    EXPECT_LE(in.friction_row_residuals.shell_sum, 1e-14);
    EXPECT_LE(in.friction_row_residuals.impulse_tensor, 1e-14);
    EXPECT_LE(in.friction_row_residuals.normal_component, 1e-14);
    EXPECT_EQ(in.normal, parent.normal);
    EXPECT_EQ(in.area, parent.area);
    EXPECT_EQ(in.row_direction, parent.row_direction);
    EXPECT_EQ(in.prism_volume, parent.prism_volume);
    EXPECT_EQ(in.G.weights, parent.G.weights);
    EXPECT_EQ(in.W.weights, parent.W.weights);
    EXPECT_EQ(in.f0, parent.f0);
    EXPECT_EQ(in.s0, parent.s0);
    EXPECT_EQ(in.provenance, parent.provenance);
}

TEST(VolumetricWallNativeRows, RoundTripIsBitwise) {
    const VolumetricWallNativeInput in{ p2_fixture() };
    const std::string path{ temp_path("vwnat_rows_roundtrip.vwnat") };
    in.write(path);
    const VolumetricWallNativeInput back{ VolumetricWallNativeInput::read(path) };
    EXPECT_EQ(back.friction_rows, in.friction_rows);
    EXPECT_EQ(back.f0, in.f0);
    ASSERT_NO_THROW(back.validate());
    ASSERT_NO_THROW(back.validate_friction_rows());
    std::remove(path.c_str());
}

TEST(VolumetricWallNativeRows, WrongLengthRejected) {
    VolumetricWallNativeInput in{ p2_fixture() };
    in.friction_rows.pop_back();
    EXPECT_THROW(in.validate(), std::invalid_argument);
    EXPECT_THROW(in.validate_friction_rows(), std::invalid_argument);
}

TEST(VolumetricWallNativeRows, NonFiniteRejected) {
    VolumetricWallNativeInput in{ p2_fixture() };
    in.friction_rows[7] = std::nan("");
    EXPECT_THROW(in.validate(), std::invalid_argument);
    EXPECT_THROW(in.validate_friction_rows(), std::invalid_argument);
}

TEST(VolumetricWallNativeRows, OriginalKernelRowsRejected) {   // nonzero shell sums: the artefact P2 removes
    VolumetricWallNativeInput in{ p2_fixture() };
    in.friction_rows = original_rows(in);
    ASSERT_NO_THROW(in.validate());                                   // finite and of the right length
    EXPECT_THROW(in.validate_friction_rows(), std::invalid_argument);
}

TEST(VolumetricWallNativeRows, SwappedRowOrderRejected) {
    VolumetricWallNativeInput in{ p2_fixture() };
    const std::uint32_t first{ in.facet_offsets[0] };
    for(std::uint32_t j=0u; j<3u; j++) {
        std::swap(in.friction_rows[3u*first+j], in.friction_rows[3u*(first+1u)+j]);
    }
    EXPECT_THROW(in.validate_friction_rows(), std::invalid_argument);
}

TEST(VolumetricWallNativeRows, RowsOfAnotherFacetRejected) {   // the rows of a facet with another normal, same row count
    VolumetricWallNativeInput in{ p2_fixture() };
    const std::uint32_t count0{ in.facet_offsets[1]-in.facet_offsets[0] };
    std::uint32_t other{ 0u };
    for(std::uint32_t a=1u; a<in.facets && other==0u; a++) {
        const bool same_count{ in.facet_offsets[a+1u]-in.facet_offsets[a]==count0 };
        const bool other_normal{ in.normal[3u*a]!=in.normal[0] || in.normal[3u*a+1u]!=in.normal[1] || in.normal[3u*a+2u]!=in.normal[2] };
        if(same_count && other_normal) {
            other = a;
        }
    }
    ASSERT_NE(other, 0u);
    for(std::uint32_t k=0u; k<3u*count0; k++) {
        in.friction_rows[3u*in.facet_offsets[0]+k] = in.friction_rows[3u*in.facet_offsets[other]+k];
    }
    EXPECT_THROW(in.validate_friction_rows(), std::invalid_argument);
}

TEST(VolumetricWallNativeRows, SmallCoefficientErrorRejected) {
    VolumetricWallNativeInput in{ p2_fixture() };
    in.friction_rows[0] += 1e-8*in.area[0];
    EXPECT_THROW(in.validate_friction_rows(), std::invalid_argument);
}

TEST(VolumetricWallNativeRows, EmptySectionRejectedAtRead) {
    VolumetricWallNativeInput in{ p2_fixture() };
    const std::string path{ temp_path("vwnat_rows_empty.vwnat") };
    in.write(path);
    std::vector<char> bytes{};
    { std::ifstream file(path, std::ios::binary); bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()); }
    const std::string name{ "friction_rows" };
    const auto found{ std::search(bytes.begin(), bytes.end(), name.begin(), name.end()) };
    ASSERT_NE(found, bytes.end());
    const size_t header{ static_cast<size_t>(found-bytes.begin()) };       // 32 name bytes, 4 dtype bytes, 8 count bytes, payload
    const size_t payload{ static_cast<size_t>(in.rows)*3u*8u };
    const std::uint64_t zero{ 0u };
    std::memcpy(bytes.data()+header+36u, &zero, 8u);
    bytes.erase(bytes.begin()+static_cast<std::ptrdiff_t>(header+44u), bytes.begin()+static_cast<std::ptrdiff_t>(header+44u+payload));
    { std::ofstream file(path, std::ios::binary); file.write(bytes.data(), static_cast<std::streamsize>(bytes.size())); }
    EXPECT_THROW(VolumetricWallNativeInput::read(path), std::runtime_error);
    std::remove(path.c_str());
}
