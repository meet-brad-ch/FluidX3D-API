#pragma once
// Task 19 (2026-11-08): host side of the opt-in native LBM-path volumetric free-slip wall (compile-time VOLUMETRIC_WALL).
// This header has no OpenCL or LBM dependency, so the embedding can be checked GPU-free and compared element-wise
// with the registered structural oracle (research/wall_model/integration/structural_check.py, audits/.../native_ownership_*).
//
// Contents:
//   * VolumetricWallNativeInput  - the versioned input container VWNAT01 (mesh maps of the CF10 bounded operator with
//                                  the CF13 half-lattice sampler, the lattice embedding, scalars, initial f0 and s0);
//   * VolumetricWallEmbedding    - regions T, S, R from the actual CSR supports (exact predicates), per-row ownership,
//                                  both-parity native addresses, fail-closed checks, device tables.
// Conventions: mesh directions are lexicographic D3Q19 or D3Q27 (x, y, z in {-1,0,1}^3, |c|^2 <= 2 or <= 3; task 98: the
// velocity set q is a property of the container, read from the D shape); FluidX3D directions are the engine's table (rest,
// +-x, +-y, +-z, the edge pairs, then for D3Q27 the corner pairs; opposite pairs at (i, i+1) for odd i). Esoteric-Pull
// addresses follow load_f/store_f of src/kernel.cpp: an address is (lattice cell, slot), linear index slot*N + cell.
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fluidx3d::experimental {

/// @brief Compressed sparse rows with double weights (the same layout as SparseMap64 of the FP64 path).
struct NativeSparseMap {
    std::uint32_t rows{};                       ///< Number of rows.
    std::uint32_t columns{};                    ///< Number of columns.
    std::vector<std::uint32_t> offsets{};       ///< rows+1 offsets.
    std::vector<std::uint32_t> indices{};       ///< Column per entry.
    std::vector<double> weights{};              ///< Weight per entry.
};

/// @brief The VWNAT01 input: everything the native wall needs, read from one file written by the registered exporter.
struct VolumetricWallNativeInput {
    std::uint32_t velocity_set{ 19u };          ///< Task 98: q of the container (19 or 27), set by read() from the D shape (D.rows = cells*q).
    std::array<std::int32_t, 3> box{};          ///< Lattice box Nx, Ny, Nz the mesh is embedded in (one domain).
    std::uint32_t cells{};                      ///< Mesh (fluid) cells.
    std::uint32_t facets{};                     ///< Wall facets.
    std::uint32_t rows{};                       ///< Wall rows (facet, outgoing direction).
    std::vector<std::int32_t> cell_xyz{};       ///< cells*3 lattice coordinates (0-based inside the box).
    std::vector<double> volume{};               ///< Cell volumes, 0 < V <= 1.
    std::vector<double> centroid{};             ///< cells*3 fluid centroids in lattice units (for the sampler check).
    std::vector<double> sample_point{};         ///< facets*3 facet-state sample targets (CF13: centroid + 0.5 n).
    std::vector<double> mode_phi{};             ///< cells: the registered (1,1) duct mode cell averages (CF10 shapes "phi") for the amplitude series.
    std::vector<double> normal{};               ///< facets*3 unit normals into the fluid.
    std::vector<double> area{};                 ///< Facet areas.
    std::vector<std::uint32_t> facet_offsets{}; ///< facets+1 row offsets.
    std::vector<std::uint32_t> row_direction{}; ///< Outgoing lexicographic direction per row.
    std::vector<double> prism_volume{};         ///< vi per row.
    std::vector<std::uint32_t> row_origin{};    ///< Registered (research) row index of every exported facet-major row.
    NativeSparseMap D{};                        ///< Direct transport, states x states (state = cell*19 + direction).
    NativeSparseMap G{};                        ///< Gather, rows x states.
    NativeSparseMap R{};                        ///< Scatter, states x rows.
    NativeSparseMap S{};                        ///< Facet-state sampler (CF13 S'), facets x cells.
    NativeSparseMap W{};                        ///< Facet-to-facet pending transfer, rows x rows.
    double viscosity{};                         ///< Recorded; the kernels use omega.
    double omega{};                             ///< BGK relaxation rate in (0, 2).
    double friction{};                          ///< K >= 0 (the supported scope requires 0).
    std::vector<double> f0{};                   ///< Initial populations, states, densities, lexicographic order.
    std::vector<double> s0{};                   ///< Initial pending amounts per row.
    std::vector<double> friction_rows{};        ///< Optional section "friction_rows" (task 43): rows*3 friction vectors phi_r in the exported row order, lattice units of area; empty when the container has none. The supported packer writes the accepted P2 construction.
    // Task 55 (2026-11-09): the six OPTIONAL sections of the affine transport candidate (research; all six or none).
    // Their presence selects the candidate path of LBM::set_volumetric_wall. A container without them is read, checked
    // and executed exactly as before.
    std::vector<double> direct_arm{};           ///< Section "direct_arm": 3 per D entry (stored CSR order): centroid of the direct part of the donor minus the fluid centroid of the donor cell, lattice units.
    std::vector<double> prism_arm{};            ///< Section "prism_arm": 3 per G entry (stored CSR order): centroid of the prism part in the cell minus the fluid centroid of the cell, lattice units.
    std::vector<std::uint32_t> gradient_cells{};   ///< Section "gradient_cells": the cells that have a gradient (mesh cell indices).
    std::vector<std::uint32_t> gradient_offsets{}; ///< Section "gradient_offsets": 3 rows (x, y, z) per gradient cell, 3*cells+1 offsets.
    std::vector<std::uint32_t> gradient_indices{}; ///< Section "gradient_indices": stencil cell per entry (mesh cell index); the own cell is the first entry of every row.
    std::vector<double> gradient_weights{};     ///< Section "gradient_weights": least-squares gradient weight per entry, 1 / lattice unit; every row sums to zero.
    std::string provenance{};                   ///< JSON text written by the exporter (hashes, case, commit).

    /// @brief Read a VWNAT01 file; every section is required and size-checked.
    /// @throws std::runtime_error naming the first failed check.
    static VolumetricWallNativeInput read(const std::string& path);
    /// @brief Write a VWNAT01 file (used by tests to round-trip; the registered exporter is Python).
    void write(const std::string& path) const;
    /// @brief Structural validation of sizes, finiteness, 0 < V <= 1, unit normals, map shapes, scalars, AND the
    /// registered CF10 static identities of the maps (#265): nonnegative D/G/R/W, nonempty facet row sets with unique
    /// outgoing directions and vi = A (c.n) > 0 on both shells, direction ownership of D/G/R/W (W: producer outgoing
    /// direction opposite to the receiving row's), donor partition v^T D + 1^T G = v^T, receiver coverage D 1 + R vi = 1,
    /// prism coverage G 1 + W 1 = vi, return delivery v^T R + 1^T W diag(1/vi) = 1, sampler constant and affine
    /// reproduction at the registered tolerances (identities 1e-10 = CF10 GATES static_routing/static_quadrature,
    /// sampler 1e-10 = GATES sampler_reproduction), positive cell densities, nonnegative pending, omega = 1/(3 nu + 1/2).
    /// @throws std::invalid_argument naming the first failed check.
    void validate() const;
    /// @brief The identity residuals validate() measured (filled by validate(); zeros before).
    struct Residuals { double donor_partition{}, receiver_coverage{}, prism_coverage{}, return_delivery{}, orientation{}, sampler_constant{}, sampler_affine{}, omega{}; };
    mutable Residuals residuals{};
    /// @brief Task 43: the fail-closed coefficient-identity check of the supported friction path. The section
    /// friction_rows must be present and, for every facet (normal n, area A, rows r with direction c_r), satisfy the
    /// identities that bind the rows to THIS geometry and row order: the sum of phi_r over each speed shell is zero (no shell mass), sum_r c_r (x) phi_r =
    /// (A/3)(I - n n^T) (the tangential friction impulse -K rho cs^2 A u_t and no normal impulse), and phi_r . n = 0.
    /// Every residual is divided by A and compared with 1e-10 (the registered tolerance of the container identities).
    /// Call validate() first (sizes and finiteness). The kernel's original friction expression does not satisfy the
    /// shell identity, so rows of that kind, rows of another geometry and rows in another order are all rejected.
    /// @note The identities are NECESSARY conditions, not a proof of one construction: other row families satisfy them
    /// too (the research family P1 does, with its defect at grazing directions). Which construction a container holds is
    /// a matter of its provenance: the supported packer (research/wall_model/integration/task43/vwnat_rows.py) writes
    /// the accepted P2 construction and records the row hash.
    /// @throws std::invalid_argument naming the first failed check (absent section, facet index, identity).
    void validate_friction_rows() const;
    /// @brief The largest residuals validate_friction_rows() measured, each divided by the facet area (zeros before).
    struct FrictionRowResiduals { double shell_sum{}, impulse_tensor{}, normal_component{}; };
    mutable FrictionRowResiduals friction_row_residuals{};
    /// @brief Task 55: true when the container holds the six sections of the affine transport candidate.
    bool affine() const { return !gradient_cells.empty(); }
    /// @brief Task 55: the fail-closed check of the candidate sections. Sizes against D and G, finite values, arms not
    /// longer than the cell diagonal, three rows of equal stencil per gradient cell with the own cell first, stencil
    /// cells inside the mesh, zero row sums (1e-10). The friction rows are checked for the tensor identity and the
    /// tangency only: the base rows of the candidate have a shell sum by construction, so the zero-shell-sum identity of
    /// validate_friction_rows() does not apply to them. Call validate() first.
    /// @throws std::invalid_argument naming the first failed check.
    void validate_affine() const;
private:
    void validate_physical() const;
    void check_friction_rows(bool zero_shell_sum) const; ///< The identities of validate_friction_rows(); the shell identity only when asked (task 55).
public:
    std::uint32_t states() const { return cells*velocity_set; }
    /// @brief Task 98: the lexicographic rest index of the container's velocity set (9 for D3Q19, 13 for D3Q27).
    std::uint32_t rest_direction() const { return velocity_set==27u ? 13u : 9u; }
};

/// @brief Minimal periodic image of a displacement on a lattice axis (task 22, NATIVE_PERIODICITY_CONTRACT, #399): the
/// exact IEEE remainder std::remainder(delta, period), required strictly inside (-period/2, +period/2).
/// @param delta displacement along the axis, lattice units, finite
/// @param period box size along the axis, lattice units, finite and > 0
/// @param axis axis index 0..2, for the message only
/// @return the wrapped displacement
/// @throws std::invalid_argument for a non-finite delta, period or result, a non-positive period, or a wrapped |delta|
/// equal to period/2 (an actual half-period ambiguity: rejected, never resolved)
double minimal_image(double delta, double period, std::uint32_t axis);

/// @brief The two direction tables (lexicographic and FluidX3D) of D3Q19 or D3Q27 and the permutation between them, computed
/// by matching vectors (task 98: q as a parameter; the default is D3Q19, so the existing D3Q19 callers are unchanged).
class NativeDirections {
public:
    /// @param q 19 or 27
    /// @throws std::invalid_argument for another q or an inconsistent table
    explicit NativeDirections(std::uint32_t q = 19u);
    /// @brief Number of directions (19 or 27).
    std::uint32_t q() const { return q_; }
    /// @brief Lexicographic direction d -> FluidX3D direction k.
    std::uint32_t lex_to_fx(std::uint32_t d) const { return lex_to_fx_[d]; }
    /// @brief FluidX3D direction k -> lexicographic direction d.
    std::uint32_t fx_to_lex(std::uint32_t k) const { return fx_to_lex_[k]; }
    /// @brief Lexicographic c vector component (axis 0..2) of direction d.
    int lex_c(std::uint32_t d, std::uint32_t axis) const { return lex_[d][axis]; }
    /// @brief FluidX3D c vector component of direction k (the table c_data of src/kernel.cpp, D3Q19 or D3Q27 branch).
    int fx_c(std::uint32_t k, std::uint32_t axis) const { return fx_[k][axis]; }
    /// @brief Opposite lexicographic direction.
    std::uint32_t lex_opposite(std::uint32_t d) const { return lex_opp_[d]; }
    /// @brief The D3Q19 rest weight 1/3, 1/18, 1/36 of lexicographic direction d (correctly rounded doubles; D3Q19 only).
    static double rest_weight_lex(std::uint32_t d);
    /// @brief Task 98: the rest weight of lexicographic direction d of THIS set, the double of the kernels' vw_weight:
    /// D3Q19 1.0/3.0, 1.0/18.0, 1.0/36.0; D3Q27 8.0/27.0, 2.0/27.0, 1.0/54.0, 1.0/216.0 (correctly rounded quotients).
    double weight_lex(std::uint32_t d) const;
    /// @brief The FluidX3D table as a flat list for a textual comparison with kernel.cpp (x row, y row, z row; 3q values).
    std::vector<int> fx_table_flat() const;
private:
    std::uint32_t q_{ 19u };
    std::array<std::array<int, 3>, 27> lex_{};
    std::array<std::array<int, 3>, 27> fx_{};
    std::array<std::uint32_t, 27> lex_to_fx_{};
    std::array<std::uint32_t, 27> fx_to_lex_{};
    std::array<std::uint32_t, 27> lex_opp_{};
};

/// @brief Esoteric-Pull address algebra on one periodic lattice box (src/kernel.cpp load_f/store_f).
class NativeAddresses {
public:
    /// @param box lattice box; the direction tables are copied (no reference is kept, #265 item 3)
    NativeAddresses(const std::array<std::int32_t, 3>& box, const NativeDirections& directions);
    /// @brief Linear cell index of lattice coordinates (periodic wrap).
    std::uint64_t cell(std::int64_t x, std::int64_t y, std::int64_t z) const;
    /// @brief Coordinates of a cell index.
    std::array<std::int64_t, 3> xyz(std::uint64_t n) const;
    /// @brief Neighbour n + c_k (FluidX3D direction k), periodic.
    std::uint64_t neighbour(std::uint64_t n, std::uint32_t k) const;
    /// @brief Linear fi address of (cell, slot): slot*N + cell, as index_f.
    std::uint64_t address(std::uint64_t n, std::uint32_t slot) const { return static_cast<std::uint64_t>(slot)*N_+n; }
    /// @brief The q addresses load_f(n, t) reads and store_f(n, t) writes at parity t % 2 (FluidX3D direction order).
    std::vector<std::uint64_t> own_set(std::uint64_t n, std::uint32_t parity) const;
    /// @brief The address cell n loads as FluidX3D direction k at t+1, when step t has the given parity.
    std::uint64_t next_state_address(std::uint64_t n, std::uint32_t k, std::uint32_t parity) const;
    /// @brief The cell whose store_f(., t) set contains the address, at parity t % 2.
    std::uint64_t native_writer(std::uint64_t address, std::uint32_t parity) const;
    std::uint64_t cells() const { return N_; }
private:
    std::array<std::int64_t, 3> box_{};
    std::uint64_t N_{};
    NativeDirections dir_{};
};

/// @brief One owned row's device record.
struct NativeOwnedRow {
    std::uint32_t receiver_cell{};              ///< Mesh cell of the receiver state.
    std::uint32_t direction{};                  ///< Lexicographic direction of the state.
    std::uint64_t address_parity0{};            ///< fi address of the level-(t+1) state when step t is even.
    std::uint64_t address_parity1{};            ///< ... when step t is odd.
};

/// @brief Counts of the ownership verification (the same names as structural_check.py).
struct NativeOwnershipReport {
    std::uint64_t rows_r_owned{}, rows_native{}, rows_r_to_ordinary{}, inactive_old_owner_permitted{};
    std::uint64_t conflict_active_ordinary_writer{}, donor_outside_R{}, native_row_writer_mismatch{};
    std::uint64_t duplicate_r_address{}, ordinary_non_pure_rows{}, missing_direct_row{};
    std::uint64_t near_unit_rows{};             ///< Single weight within 1e-12 of 1 but not exactly 1 (report only).
    double near_unit_row_max_deviation{};
    std::uint64_t near_unit_volumes{};
    std::uint64_t algebra_writer_violations{}, algebra_stream_violations{};
    bool pass() const;
    std::string text() const;
};

/// @brief Blocking copy of every R buffer after a step (task 19 instrument; raw values, non-finite kept as evidence).
struct VolumetricWallNativeCapture {
    bool available{ false };                    ///< False before the first wall step: the buffers hold no level, never read them as diagnostics (#273).
    std::uint64_t t{};                          ///< Host time step after the step (fi holds level t at parity t % 2).
    std::vector<float> gpre{};                  ///< Actual stored g of the R cells read at the start of the last step (q per R cell, lexicographic).
    std::vector<double> f{};                    ///< fl64(w + gpre).
    std::vector<double> post{};                 ///< After the FP64 collision.
    std::vector<double> fields{};               ///< rho, ux, uy, uz per R cell after the collision.
    std::vector<double> incoming{};             ///< Gathered + pending per row.
    std::vector<double> Q{};                    ///< Returned amounts per row.
    std::vector<double> impulse{};              ///< 3 per facet.
    std::vector<double> pending_old{};          ///< The pending consumed by the last step.
    std::vector<double> pending_new{};          ///< The pending produced by the last step (the current one).
    std::vector<double> prestore{};             ///< Owned-row values before the fl32 store, in owned-row order.
    std::vector<double> support_post{};         ///< Task 55, affine path only: post-collision populations of the support cells as the wall pass used them (q per support cell, ascending mesh order, lexicographic). Empty otherwise.
    std::vector<double> returned{};             ///< Task 55, affine path only: R Q + centred scatter per R state (q per R cell, lexicographic); 0 where no wall kernel writes.
    std::vector<double> direct_increment{};     ///< Task 55, affine path only: the direct arm term sum D dD.grad p per R state; 0 where no wall kernel writes.
    std::vector<double> wall_state{};           ///< Task 98, RR27 path only: rho_w, u_w (x, y, z) per facet, the even reference of the return. Empty otherwise.
    std::vector<double> delta{};                ///< Task 98, RR27 path only: the one-per-facet mass correction delta per facet. Empty otherwise.
    std::uint32_t wall_state_failures{};        ///< Task 98, RR27 path only: facets with a non-finite or non-positive rho_w since attachment (no clamp; any count invalidates the run).
    std::uint32_t clamp_count{};                ///< Native clamp activations since construction.
};

/// @brief Regions and ownership of the native embedding, derived from the actual maps with exact predicates.
class VolumetricWallEmbedding {
public:
    /// @param input validated VWNAT01 input
    /// @param solid one byte per lattice cell of the box: nonzero = TYPE_S (inactive), as the host flags will be (copied)
    /// @throws std::invalid_argument on a structural failure (named), after the full report is formed
    /// @note `input` is referenced, not copied: it must outlive the embedding (the LBM attaches both within one call).
    VolumetricWallEmbedding(const VolumetricWallNativeInput& input, const std::vector<std::uint8_t>& solid);

    const NativeOwnershipReport& report() const { return report_; }
    const NativeDirections& directions() const { return directions_; }
    const NativeAddresses& addresses() const { return addresses_; }
    /// @brief Sorted mesh indices of the R cells (R_own = T + S, closed).
    const std::vector<std::int64_t>& r_cells() const { return r_cells_; }
    /// @brief Sorted mesh state indices (cell*q + d) of the rows the R kernels write.
    const std::vector<std::int64_t>& r_rows() const { return r_rows_; }
    /// @brief Sorted mesh state indices of the rows left to the native store_f.
    const std::vector<std::int64_t>& native_rows() const { return native_rows_; }
    /// @brief Lattice cell index of every mesh cell.
    const std::vector<std::uint64_t>& lattice_of_cell() const { return lattice_of_cell_; }
    /// @brief R-local index of a mesh cell, or -1 when the cell is ordinary.
    const std::vector<std::int64_t>& r_index_of_cell() const { return r_index_of_cell_; }
    /// @brief Device records of the owned rows, in r_rows() order.
    const std::vector<NativeOwnedRow>& owned() const { return owned_; }
    /// @brief Write the three sorted lists as little-endian int64 files <prefix>_R_cells.i64, ... for the oracle comparison.
    void write_lists(const std::string& prefix) const;
    std::uint64_t cut_cells() const { return cut_cells_; }
    std::uint64_t t_cells() const { return t_cells_; }
    std::uint64_t s_cells() const { return s_cells_; }
    std::uint64_t closure_added() const { return closure_added_; }

private:
    struct RowEntries {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> donors{};   ///< (cell, direction)
        std::vector<double> weights{};
        std::uint32_t scatter_entries{};
    };
    RowEntries row_entries(std::uint32_t cell, std::uint32_t d) const;
    bool pure(std::uint32_t cell, std::uint32_t d);
    void derive_regions();
    void verify_ownership();
    void verify_algebra();

    const VolumetricWallNativeInput& in_;
    std::vector<std::uint8_t> solid_{};
    NativeDirections directions_{};
    NativeAddresses addresses_;
    std::vector<std::uint64_t> lattice_of_cell_{};
    std::map<std::uint64_t, std::uint32_t> cell_of_lattice_{};
    std::set<std::uint32_t> r_set_{};
    std::vector<std::int64_t> r_cells_{}, r_rows_{}, native_rows_{}, r_index_of_cell_{};
    std::vector<NativeOwnedRow> owned_{};
    std::map<std::uint64_t, bool> pure_cache_{};
    NativeOwnershipReport report_{};
    std::uint64_t cut_cells_{}, t_cells_{}, s_cells_{}, closure_added_{};
};

}  // namespace fluidx3d::experimental
