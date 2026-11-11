// Task 19: GPU-free structural check of a VWNAT01 input with the engine's own embedding class. Writes the sorted
// int64 lists (for the element-wise comparison with the registered oracle arrays) and prints the report.
//   volumetric_wall_native_check INPUT.vwnat LIST_PREFIX
// Exit 0 when the embedding passes, 1 when the input or ownership fails (the message names the check), 2 on usage.
#include "experimental/volumetric_wall_native.hpp"
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(const int argc, char** argv) {
    if(argc!=3) {
        std::cerr << "usage: volumetric_wall_native_check INPUT.vwnat LIST_PREFIX" << std::endl;
        return 2;
    }
    try {
        using namespace fluidx3d::experimental;
        const VolumetricWallNativeInput input{ VolumetricWallNativeInput::read(argv[1]) };
        input.validate();
        // Task 22 G11: the identity residuals that validate() measured, at full precision, for the record.
        const VolumetricWallNativeInput::Residuals& r{ input.residuals };
        std::cout.precision(17);
        std::cout << "residuals donor_partition=" << r.donor_partition << " receiver_coverage=" << r.receiver_coverage
                  << " prism_coverage=" << r.prism_coverage << " return_delivery=" << r.return_delivery << " orientation=" << r.orientation
                  << " sampler_constant=" << r.sampler_constant << " sampler_affine=" << r.sampler_affine << " omega=" << r.omega << std::endl;
        // Task 43: the coefficient identities of the optional friction_rows section (the supported path requires them).
        if(input.friction_rows.empty()) {
            std::cout << "friction_rows absent (research path only)" << std::endl;
        } else if(input.affine()) {   // task 98: the affine container (base rows: tensor identity and tangency; every row with a fluid gather part)
            input.validate_affine();
            const VolumetricWallNativeInput::FrictionRowResiduals& f{ input.friction_row_residuals };
            std::cout << "affine sections checked; friction_rows impulse_tensor=" << f.impulse_tensor << " normal_component=" << f.normal_component << std::endl;
        } else {
            input.validate_friction_rows();
            const VolumetricWallNativeInput::FrictionRowResiduals& f{ input.friction_row_residuals };
            std::cout << "friction_rows shell_sum=" << f.shell_sum << " impulse_tensor=" << f.impulse_tensor << " normal_component=" << f.normal_component << std::endl;
        }
        const std::uint64_t lattice_cells{ static_cast<std::uint64_t>(input.box[0])*static_cast<std::uint64_t>(input.box[1])*static_cast<std::uint64_t>(input.box[2]) };
        std::vector<std::uint8_t> solid(static_cast<size_t>(lattice_cells), 1u);   // the duct fixture: every non-mesh cell is solid
        const NativeAddresses lattice(input.box, NativeDirections(input.velocity_set));   // task 98: q of the container
        for(std::uint32_t c=0u; c<input.cells; c++) {
            solid[static_cast<size_t>(lattice.cell(input.cell_xyz[3u*c], input.cell_xyz[3u*c+1u], input.cell_xyz[3u*c+2u]))] = 0u;
        }
        const VolumetricWallEmbedding embedding(input, solid);
        embedding.write_lists(argv[2]);
        std::cout << "velocity_set=D3Q" << input.velocity_set << std::endl;
        std::cout << "cells=" << input.cells << " cut=" << embedding.cut_cells() << " T=" << embedding.t_cells() << " S=" << embedding.s_cells()
                  << " R=" << embedding.r_cells().size() << " closure_added=" << embedding.closure_added()
                  << " owned=" << embedding.owned().size() << std::endl;
        std::cout << embedding.report().text() << std::endl;
        std::cout << "PASS" << std::endl;
        return 0;
    } catch(const std::exception& error) {
        std::cout << "FAIL " << error.what() << std::endl;
        return 1;
    }
}
