#pragma once
/// @file maps.hpp
/// @brief The surface maps of the container: terrain_container.py assemble (G, R, W, S and the rows), MapIdentities
/// (the row identities, permute_rows) and NativeIdentityGate (its row part), with scipy.sparse's rules for building and
/// multiplying CSR matrices. The direct map D and the state identities grow with the volume; the slab stream makes them.

#include "pass_runner.hpp"
#include "sampler.hpp"
#include "terrain.hpp"
#include "terrain_input.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief One (row, column, value) entry of a sparse matrix in construction order.
struct Triplet {
    std::int64_t row{};    ///< Row.
    std::int64_t column{}; ///< Column.
    double value{};        ///< Value.
};

/// @brief A CSR matrix with sorted column indices (scipy's canonical format).
struct SparseMatrix {
    std::int64_t rows{};                ///< Rows.
    std::int64_t columns{};             ///< Columns.
    std::vector<std::int64_t> offsets{}; ///< rows + 1 entry offsets.
    std::vector<std::int64_t> indices{}; ///< Column per entry.
    std::vector<double> values{};        ///< Value per entry.
};

/// @brief A CSR matrix with one row per lattice state (cell * q + d) of which only the rows with entries are stored:
/// the same matrix as scipy's, without the offsets of the empty rows (they grow with the volume).
struct StateRows {
    std::int64_t states{};               ///< Rows of the full matrix.
    std::int64_t columns{};              ///< Columns.
    std::vector<std::int64_t> state{};   ///< The rows with entries, increasing.
    std::vector<std::int64_t> offsets{}; ///< state.size() + 1 entry offsets.
    std::vector<std::int64_t> indices{}; ///< Column per entry, sorted per row.
    std::vector<double> values{};        ///< Value per entry.
};

/// @brief A vector over the lattice states of which only some entries are stored; the others are 0.0.
struct StateValues {
    std::vector<std::int64_t> state{}; ///< Stored states, increasing.
    std::vector<double> value{};       ///< Their values.
};

/// @brief scipy.sparse's operations, with its exact summation orders.
class SparseOps {
public:
    /// @brief coo_matrix((v, (r, c)), shape).tocsr(): entries kept (explicit zeros too), indices sorted per row, duplicates
    /// summed in sorted order.
    /// @throws std::runtime_error for three or more entries at one position (their sum order is not reproducible)
    [[nodiscard]] static SparseMatrix from_triplets(std::int64_t rows, std::int64_t columns, const std::vector<Triplet>& entries);

    /// @brief from_triplets for a matrix with one row per state, stored as StateRows.
    /// @throws std::runtime_error for three or more entries at one position
    [[nodiscard]] static StateRows state_rows(std::int64_t states, std::int64_t columns, const std::vector<Triplet>& entries);

    /// @brief m @ x (csr_matvec: per row, a sum from 0.0 in index order).
    [[nodiscard]] static std::vector<double> times(const SparseMatrix& m, const std::vector<double>& x);

    /// @brief m @ x for the stored rows of m (the other rows are 0.0).
    [[nodiscard]] static StateValues times(const StateRows& m, const std::vector<double>& x);

    /// @brief m @ ones (csr_matvec with x = 1.0: per row, a sum of value * 1.0 from 0.0 in index order).
    [[nodiscard]] static std::vector<double> row_sums(const SparseMatrix& m);

    /// @brief x @ m (csc_matvec of m's transpose: y[column] += value * x[row], rows in order).
    [[nodiscard]] static std::vector<double> left_times(const std::vector<double>& x, const SparseMatrix& m);

    /// @brief ones @ m for a matrix whose columns are states: y[column] += value * 1.0, rows in order; only the columns
    /// with entries are stored.
    [[nodiscard]] static StateValues column_sums(const SparseMatrix& m);

    /// @brief P_rows @ m @ P_columns^T for permutation matrices: new row i is old row row_order[i], new column i is old
    /// column column_order[i] (empty: unchanged). As in csr_matmat, exact zeros are dropped; indices are sorted.
    [[nodiscard]] static SparseMatrix permuted(const SparseMatrix& m, const std::vector<std::int64_t>& row_order,
                                               const std::vector<std::int64_t>& column_order);

    /// @brief m @ P_columns^T: new column i is old column column_order[i]; exact zeros dropped, indices sorted, rows that
    /// become empty are not stored.
    [[nodiscard]] static StateRows permuted_columns(const StateRows& m, const std::vector<std::int64_t>& column_order);
};

/// @brief One step of float(np.max(np.abs(x))) over the residuals of an identity.
class IdentityMax {
public:
    /// @brief max(current, |value|), NaN when either is NaN (numpy's max propagates NaN). The first current is -1.0.
    [[nodiscard]] static double largest(double current, double value);
};

/// @brief One wall row (facet index, direction).
struct WallRow {
    int facet{};     ///< Index in the sorted facets.
    int direction{}; ///< Direction index.
};

/// @brief The surface maps of assemble(), before the row permutation.
struct AssembledMaps {
    SparseMatrix g{};             ///< Gather, rows x states.
    StateRows r{};                ///< Scatter, states x rows.
    SparseMatrix s{};             ///< Sampler, facets x cells.
    SparseMatrix w{};             ///< Facet to facet, rows x rows.
    std::vector<double> vi{};     ///< Prism volume per row.
    std::vector<WallRow> rows{};  ///< The rows, facet-major.
};

/// @brief The surface maps after permute_rows and the CSR copies of ContainerBuild.build.
struct PermutedMaps {
    SparseMatrix g{};
    StateRows r{};
    SparseMatrix s{};
    SparseMatrix w{};
    std::vector<std::int64_t> row_facet{};     ///< rf[order].
    std::vector<std::int64_t> row_direction{}; ///< rd[order].
    std::vector<double> vi{};                  ///< vi[order].
    std::vector<double> s0{};                  ///< (W @ w[rd])[order].
    std::vector<std::int64_t> order{};         ///< Stable argsort of rf.
};

/// @brief The four map identities (max |residual|), MapIdentities.identities.
struct IdentityValues {
    double donor_partition{};
    double receiver_coverage{};
    double prism_coverage{};
    double return_delivery{};
};

/// @brief The failing items of one identity: up to 50, largest residual first (NaN counts as largest).
struct GateFailure {
    std::string name{};                  ///< The identity.
    bool per_state{};                    ///< true: items are states (cell * q + d); false: rows.
    std::vector<std::int64_t> items{};   ///< State or row indices.
    std::vector<double> residuals{};     ///< Their residuals.
};

/// @brief The products over states that come from the surface maps (MapIdentities.identities, NativeIdentityGate).
struct StateProducts {
    StateValues one_g{}; ///< 1 @ G.
    StateValues r_vi{};  ///< R @ vi.
};

/// @brief The products over rows (MapIdentities.identities and NativeIdentityGate.check).
struct RowProducts {
    std::vector<double> g_one{}; ///< G @ 1.
    std::vector<double> w_one{}; ///< W @ 1.
    std::vector<double> vs_r{};  ///< Vs @ R.
    std::vector<double> one_w{}; ///< 1 @ W.
};

/// @brief NativeIdentityGate.check.
struct GateReport {
    static constexpr double TOLERANCE{ 1e-10 }; ///< identity_tolerance of the native checker.
    bool finite{};                    ///< V, vi and every map value finite.
    bool passed{};                    ///< finite and no identity above the tolerance.
    IdentityValues worst{};           ///< Largest relative residual per identity (NaN when one is not finite).
    std::vector<GateFailure> failing{}; ///< One entry per failing identity.
};

/// @brief The row part of NativeIdentityGate.check.
struct RowGate {
    bool finite{};             ///< vi and the values of G, R and W finite.
    double prism_worst{};      ///< Largest relative prism residual (NaN when one is not finite or there is no row).
    double return_worst{};     ///< Largest return residual (NaN likewise).
    GateFailure prism{};       ///< Failing rows of prism_coverage.
    GateFailure ret{};         ///< Failing rows of return_delivery.
};

/// @brief The surface maps of one terrain.
class MapAssembly {
public:
    MapAssembly(const TerrainGeometry& geometry, const VelocityTable& velocities, const PassRunner& passes, const SamplerSearch& sampler);

    /// @brief assemble(vs, b, sam) without D. @throws std::runtime_error with the builder's message for a broken map
    [[nodiscard]] AssembledMaps assemble() const;

    /// @brief The products over states of the identities.
    [[nodiscard]] static StateProducts state_products(const AssembledMaps& m);

    /// @brief The four products over rows, each on its own thread.
    [[nodiscard]] RowProducts row_products(const AssembledMaps& m) const;

    /// @brief prism_coverage and return_delivery of MapIdentities.identities(m); the state identities are left at 0.
    [[nodiscard]] static IdentityValues row_identities(const AssembledMaps& m, const RowProducts& p);

    /// @brief The row part of NativeIdentityGate.check(m, b).
    [[nodiscard]] static RowGate row_gate(const AssembledMaps& m, const RowProducts& p);

    /// @brief permute_rows plus the sorted CSR copies of ContainerBuild.build (D is not here).
    [[nodiscard]] PermutedMaps permute(const AssembledMaps& m) const;

    /// @brief The index of a cell in the sorted cell list, -1 when it is not a fluid cell (i, j inside the box).
    [[nodiscard]] std::int64_t cell_index(int i, int j, int k) const;

    /// @brief The index of a facet key in the sorted facet list. @throws std::runtime_error when unknown
    [[nodiscard]] int facet_index(const FacetKey& key) const;

    /// @brief Producer row (facet, direction); exists is false for the rest direction and a row without content.
    [[nodiscard]] ProducerRowView producer_unit(int facet, int direction) const;

    /// @brief The failing items of one identity: up to 50, largest residual first (NaN as largest), ties by index.
    [[nodiscard]] static GateFailure failure(const std::string& name, bool per_state, const std::vector<double>& value);

private:
    const TerrainGeometry& geometry_;
    const VelocityTable& velocities_;
    const PassRunner& passes_;
    const SamplerSearch& sampler_;
    std::vector<int> moving_slot_{};      ///< Per direction: its position among the moving directions, -1 for rest.
    std::vector<std::uint64_t> facet_codes_{}; ///< Sorted packed keys.
    std::vector<int> facet_by_code_{};    ///< Facet index per sorted packed key.

    [[nodiscard]] static std::uint64_t pack(const FacetKey& key);
    void gather_entries(const std::vector<std::int64_t>& row_of, std::vector<Triplet>& g, std::vector<Triplet>& r) const;
    void pending_entries(const std::vector<std::int64_t>& row_of, std::vector<Triplet>& out) const;
    [[nodiscard]] std::vector<double> volume_times_r(const StateRows& r) const;
};

} // namespace fluidx3d::experimental::wall_builder
