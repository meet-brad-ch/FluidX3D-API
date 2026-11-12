#pragma once
/// @file slab_stream.hpp
/// @brief The parts of the container that grow with the volume, made one slab of x-columns at a time: the direct map D
/// and its arms (written to temporary files), the state identities of MapIdentities and NativeIdentityGate, the pure
/// test of wall_list (rule #253), the direct arms of the sample-support cells and the donor diagnostics. The cell order
/// is sorted by (i, j, k), so the cells of a slab are one index range of every per-cell section.

#include "maps.hpp"
#include "pass_runner.hpp"
#include "terrain.hpp"
#include "terrain_input.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief One section of the container written to a temporary file, value by value, with a buffer.
class SpoolFile {
public:
    /// @brief Creates the file. @throws std::runtime_error when it cannot be created
    SpoolFile(const std::filesystem::path& path, std::string section);

    SpoolFile(const SpoolFile&) = delete;
    SpoolFile& operator=(const SpoolFile&) = delete;

    /// @brief Appends a uint32 value. @throws std::runtime_error when it does not fit uint32 ("INVALID: ... does not fit")
    void u32(std::int64_t value);

    /// @brief Appends a float64 value.
    void f64(double value);

    /// @brief Appends values that are already in their file form (checked by the caller).
    /// @param data the bytes
    /// @param bytes their size
    /// @param values the number of values they hold
    /// @throws std::runtime_error when a write fails
    void raw(const void* data, std::size_t bytes, std::uint64_t values);

    /// @brief Writes the buffer and closes the file. @throws std::runtime_error when the write fails
    void close();

    /// @brief The number of values written.
    [[nodiscard]] std::uint64_t count() const { return count_; }

    /// @brief The file.
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    static constexpr std::size_t BUFFER{ std::size_t{ 4 } << 20 };

    std::filesystem::path path_{};
    std::string section_{};
    std::ofstream file_{};
    std::vector<char> buffer_{};
    std::uint64_t count_{};

    void append(const void* data, std::size_t size);
    void flush();
};

/// @brief The 50 largest residuals of one state identity over all slabs, as MapAssembly::failure orders them.
class FailureTop {
public:
    /// @brief Keeps the item when its residual fails (not finite, or above GateReport::TOLERANCE).
    void add(std::int64_t item, double residual);

    /// @brief Adds the kept items of another part (the result does not depend on the order of the parts).
    void merge(const FailureTop& other);

    /// @brief The failing items, largest residual first (NaN as largest), ties by index, at most 50.
    [[nodiscard]] GateFailure result(const std::string& name) const;

private:
    /// @brief One failing item.
    struct Item {
        std::int64_t index{};
        double residual{};
    };
    std::vector<Item> items_{}; ///< Kept items; pruned to the best 50 when it grows.

    void prune();
};

/// @brief The largest value of a residual array and whether all of it is finite (NativeIdentityGate's worst()).
class WorstValue {
public:
    /// @brief Adds one value.
    void add(double value);

    /// @brief Adds the values of another part (the result does not depend on the order of the parts).
    void merge(const WorstValue& other);

    /// @brief The largest value; NaN when a value is not finite or there is none.
    [[nodiscard]] double result() const;

private:
    bool any_{};
    bool finite_{ true };
    double largest_{};
};

/// @brief The inputs of the slab stream that come from the surface maps.
struct SlabInputs {
    const TerrainGeometry* geometry{};            ///< The cells.
    const VelocityTable* velocities{};            ///< The velocity set.
    const StateProducts* products{};              ///< 1 @ G and R @ vi.
    const StateRows* permuted_r{};                ///< The permuted R (its empty rows are the states without return).
    const std::vector<char>* wall{};              ///< Per cell: 1 for a gather donor cell (arms().wall_cells).
    const std::vector<std::int64_t>* support{};   ///< The S stencil cells, sorted unique.
};

/// @brief What the stream found, for the wall list, the gate and the provenance.
struct SlabResults {
    std::vector<char> added{};                ///< Per cell: 1 when rule #253 adds it to the wall list.
    std::vector<double> support_arm{};        ///< Per support cell: the largest |direct arm| of the D entries it donates.
    double donor_partition{ -1.0 };           ///< MapIdentities.identities, the state part.
    double receiver_coverage{ -1.0 };
    WorstValue donor_worst{};                 ///< NativeIdentityGate, the state part.
    WorstValue receiver_worst{};
    FailureTop donor_failures{};
    FailureTop receiver_failures{};
    bool finite{ true };                      ///< V and the values of D finite.
    std::uint64_t d_entries{};                ///< nnz of D.
};

/// @brief The slab loop.
class SlabStream {
public:
    /// @brief Opens the temporary files of D_offsets, D_indices, D_weights and direct_arm in the folder.
    /// @param threads the threads of the donor passes, 1 or more
    /// @throws std::runtime_error when a file cannot be created
    SlabStream(const SlabInputs& inputs, const std::filesystem::path& folder, int threads);

    /// @brief Runs every slab: the donor passes of the slab and its two neighbour columns, then its states.
    /// @param slab_cells the cells of one slab (at least one x-column per slab); its donor results are held at once
    /// @param slab_columns a fixed slab width in x-columns, or 0 to size the slabs by slab_cells
    /// @param diag the facet diagnostics; the cell units are added in the serial order
    /// @throws std::runtime_error with the builder's message for a direct volume into a non-fluid cell, a value that
    /// does not fit its section, or a write that fails
    void run(std::size_t slab_cells, int slab_columns, DiagTotals& diag);

    /// @brief The results (complete after run()).
    [[nodiscard]] const SlabResults& results() const { return results_; }

    /// @brief The temporary file of a streamed section: "D_offsets", "D_indices", "D_weights" or "direct_arm".
    /// @throws std::invalid_argument for another name
    [[nodiscard]] const SpoolFile& spool(const std::string& name) const;

private:
    /// @brief A support cell's direct arm, found in one x-column (merged into SlabResults::support_arm).
    struct SupportArm {
        std::size_t position{}; ///< Index in the support cell list.
        double largest{};       ///< The largest |arm component| of one D entry.
    };

    /// @brief What the states of one x-column give; one thread fills it, the merge adds it in column order.
    struct ColumnStates {
        std::vector<std::uint32_t> ends{};       ///< Per state: the column's D entries up to and with it (D_offsets less the base).
        std::vector<std::uint32_t> d_indices{};  ///< Per entry: the donor state (checked to fit uint32).
        std::vector<double> d_weights{};         ///< Per entry: the D value.
        std::vector<double> arms{};              ///< Per entry: the direct arm x, y, z.
        std::vector<std::int64_t> added{};       ///< Cells that rule #253 adds (also in other x-columns).
        std::vector<SupportArm> support{};       ///< Arms of support cells, in entry order.
        double donor_partition{ -1.0 };          ///< MapIdentities.identities, this column's part.
        double receiver_coverage{ -1.0 };
        WorstValue donor_worst{};                ///< NativeIdentityGate, this column's part.
        WorstValue receiver_worst{};
        FailureTop donor_failures{};
        FailureTop receiver_failures{};
        bool finite{ true };                     ///< V and the D values of this column finite.
        std::size_t one_g_at{};                  ///< Cursor in products->one_g.
        std::size_t r_vi_at{};                   ///< Cursor in products->r_vi.
        std::size_t return_at{};                 ///< Cursor in permuted_r->state.
        std::string error{};                     ///< The first error of the column (the builder's message), empty when none.
    };

    /// @brief One lattice state: its cell, the cell index, the direction, the state index and the cell volume.
    struct StatePlace {
        std::array<int, 3> cell{};   ///< (i, j, k), i and j inside the box.
        std::int64_t n{};            ///< Cell index.
        std::int64_t d{};            ///< Direction index.
        std::int64_t s{};            ///< State index n * q + d.
        double volume{};             ///< V of the cell.
    };

    /// @brief The D entry of a receiver state (at most one).
    struct ReceiverEntry {
        bool has_entry{};            ///< The donor at -c keeps a nonzero volume.
        double ratio{};              ///< D value: the donor's kept volume / V of the receiver.
        std::int64_t donor{ -1 };    ///< The donor cell, -1 when the cell at -c is not fluid.
    };

    SlabInputs in_{};
    int threads_{ 1 };
    DonorColumns donors_;
    SpoolFile d_offsets_;
    SpoolFile d_indices_;
    SpoolFile d_weights_;
    SpoolFile direct_arm_;
    SlabResults results_{};
    std::vector<char> support_flag_{}; ///< Per cell: 1 for a support cell.

    /// @brief The x-column range [first, last) of the slab that starts at first.
    [[nodiscard]] int slab_end(int first, std::size_t slab_cells, int slab_columns) const;
    /// @brief The states of the cells of x-columns [first, last): threads x-columns at a time on all threads, each
    /// merged in column order (the order of the cells).
    /// @throws std::runtime_error with the first error in cell order
    void states(int first, int last);
    /// @brief The states of x-column i into its buffers; an error is kept in out.error.
    void column_states(int i, ColumnStates& out) const;
    /// @brief Adds one x-column's buffers to the spool files and the results.
    /// @throws std::runtime_error when a value does not fit its section or a write fails
    void merge(const ColumnStates& part);
    /// @brief The states of one cell (i, j, k) with index n.
    void cell_states(int i, int j, int k, std::int64_t n, ColumnStates& out) const;
    /// @brief The state as a donor: the D column check and the donor_partition residual.
    /// @throws std::runtime_error for a kept volume streamed into a non-fluid cell
    void donor_role(const StatePlace& at, ColumnStates& out) const;
    /// @brief The state as a receiver: its D row and the receiver_coverage residual.
    [[nodiscard]] ReceiverEntry receiver_role(const StatePlace& at, ColumnStates& out) const;
    /// @brief Keeps the D entry and its direct arm; keeps the arm of a support cell.
    void write_entry(const StatePlace& at, const ReceiverEntry& entry, const DonorState& from, ColumnStates& out) const;
    /// @brief wall_list's rule #253 for the state.
    void rule_253(const StatePlace& at, const ReceiverEntry& entry, ColumnStates& out) const;
    /// @brief The value of a sparse state vector at state s; the cursor moves forward only.
    [[nodiscard]] static double value_at(const StateValues& v, std::int64_t s, std::size_t& at);
    /// @brief True when the permuted R has entries in row s; the cursor moves forward only.
    [[nodiscard]] bool has_return(std::int64_t s, std::size_t& at) const;
    /// @brief The donor state of cell n (in x-column i) along direction d.
    [[nodiscard]] const DonorState& donor_state(int i, std::int64_t n, std::size_t d) const;
    /// @brief Adds the donor units of the cells of x-columns [first, last) to the diagnostics, in cell order.
    void replay(int first, int last, DiagTotals& diag) const;
};

} // namespace fluidx3d::experimental::wall_builder
