#pragma once
/// @file maps.hpp
/// @brief The transport maps of the container: terrain_container.py assemble, MapIdentities (identities, permute_rows)
/// and NativeIdentityGate, with scipy.sparse's rules for building and multiplying CSR matrices.

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

/// @brief scipy.sparse's operations, with its exact summation orders.
class SparseOps {
public:
    /// @brief coo_matrix((v, (r, c)), shape).tocsr(): entries kept (explicit zeros too), indices sorted per row, duplicates
    /// summed in sorted order.
    /// @throws std::runtime_error for three or more entries at one position (their sum order is not reproducible)
    [[nodiscard]] static SparseMatrix from_triplets(std::int64_t rows, std::int64_t columns, const std::vector<Triplet>& entries);

    /// @brief m @ x (csr_matvec: per row, a sum from 0.0 in index order).
    [[nodiscard]] static std::vector<double> times(const SparseMatrix& m, const std::vector<double>& x);

    /// @brief x @ m (csc_matvec of m's transpose: y[column] += value * x[row], rows in order).
    [[nodiscard]] static std::vector<double> left_times(const std::vector<double>& x, const SparseMatrix& m);

    /// @brief P_rows @ m @ P_columns^T for permutation matrices: new row i is old row row_order[i], new column i is old
    /// column column_order[i] (empty: unchanged). As in csr_matmat, exact zeros are dropped; indices are sorted.
    [[nodiscard]] static SparseMatrix permuted(const SparseMatrix& m, const std::vector<std::int64_t>& row_order,
                                               const std::vector<std::int64_t>& column_order);
};

/// @brief One wall row (facet index, direction).
struct WallRow {
    int facet{};     ///< Index in the sorted facets.
    int direction{}; ///< Direction index.
};

/// @brief The maps of assemble(), before the row permutation.
struct AssembledMaps {
    std::vector<double> volume{}; ///< V per cell.
    SparseMatrix d{};             ///< Direct transport, states x states.
    SparseMatrix g{};             ///< Gather, rows x states.
    SparseMatrix r{};             ///< Scatter, states x rows.
    SparseMatrix s{};             ///< Sampler, facets x cells.
    SparseMatrix w{};             ///< Facet to facet, rows x rows.
    std::vector<double> vi{};     ///< Prism volume per row.
    std::vector<WallRow> rows{};  ///< The rows, facet-major.
};

/// @brief The maps after permute_rows and the CSR copies of ContainerBuild.build.
struct PermutedMaps {
    SparseMatrix d{};
    SparseMatrix g{};
    SparseMatrix r{};
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

/// @brief The sparse products both identity checks use (MapIdentities.identities and NativeIdentityGate.check).
struct MapProducts {
    std::vector<double> vs{};    ///< V repeated per state.
    std::vector<double> vs_d{};  ///< Vs @ D.
    std::vector<double> one_g{}; ///< 1 @ G.
    std::vector<double> d_one{}; ///< D @ 1.
    std::vector<double> r_vi{};  ///< R @ vi.
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

/// @brief The maps of one terrain.
class MapAssembly {
public:
    MapAssembly(const TerrainGeometry& geometry, const VelocityTable& velocities, const PassRunner& passes, const SamplerSearch& sampler);

    /// @brief assemble(vs, b, sam). @throws std::runtime_error with the builder's message for a broken map
    [[nodiscard]] AssembledMaps assemble() const;

    /// @brief The eight sparse products of the identities, each on its own thread.
    [[nodiscard]] MapProducts products(const AssembledMaps& m) const;

    /// @brief MapIdentities.identities(m).
    [[nodiscard]] IdentityValues identities(const AssembledMaps& m, const MapProducts& p) const;

    /// @brief NativeIdentityGate.check(m, b): tolerance 1e-10 on the native (relative) definitions.
    [[nodiscard]] GateReport gate(const AssembledMaps& m, const MapProducts& p) const;

    /// @brief permute_rows plus the sorted CSR copies of ContainerBuild.build.
    [[nodiscard]] PermutedMaps permute(const AssembledMaps& m) const;

    /// @brief The index of a cell in the sorted cell list, -1 when it is not a fluid cell.
    [[nodiscard]] std::int64_t cell_index(int i, int j, int k) const;

    /// @brief The index of a facet key in the sorted facet list. @throws std::runtime_error when unknown
    [[nodiscard]] int facet_index(const FacetKey& key) const;

    /// @brief The pass results.
    [[nodiscard]] const PassRunner& passes() const { return passes_; }

    /// @brief The unit of producer row (facet, direction), or nullptr for the rest direction.
    [[nodiscard]] const UnitResult* producer_unit(int facet, int direction) const;

private:
    const TerrainGeometry& geometry_;
    const VelocityTable& velocities_;
    const PassRunner& passes_;
    const SamplerSearch& sampler_;
    int kmin_{};
    int layers_{};
    std::vector<std::int64_t> cell_slots_{};
    std::vector<int> moving_slot_{};      ///< Per direction: its position among the moving directions, -1 for rest.
    std::vector<std::uint64_t> facet_codes_{}; ///< Sorted packed keys.
    std::vector<int> facet_by_code_{};    ///< Facet index per sorted packed key.

    [[nodiscard]] static std::uint64_t pack(const FacetKey& key);
    void direct_entries(const std::vector<double>& volume, std::vector<Triplet>& out) const;
    void gather_entries(const std::vector<double>& volume, const std::vector<std::int64_t>& row_of, std::vector<Triplet>& g,
                        std::vector<Triplet>& r) const;
    void pending_entries(const std::vector<std::int64_t>& row_of, std::vector<Triplet>& out) const;
    [[nodiscard]] static GateFailure failure(const std::string& name, bool per_state, const std::vector<double>& value);
};

} // namespace fluidx3d::experimental::wall_builder
