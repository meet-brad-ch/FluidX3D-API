#pragma once
/// @file container_writer.hpp
/// @brief The VWNAT01 container of terrain_container.py (ContainerBuild.sections and Writer) and its provenance text,
/// written straight to the file: the sections that grow with the volume are made cell by cell or copied from the slab
/// stream's temporary files, so the container is never held in memory.

#include "maps.hpp"
#include "pass_runner.hpp"
#include "python_text.hpp"
#include "sampler.hpp"
#include "sha256.hpp"
#include "slab_stream.hpp"
#include "terrain_input.hpp"
#include "wall_tables.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief The physical options of one container (the CLI of terrain_container.py).
struct ContainerOptions {
    double nu{};                                  ///< Kinematic viscosity.
    double friction{};                            ///< K.
    std::optional<double> rough_z0{};             ///< --rough-z0.
    double rough_k0{};                            ///< --rough-K0.
    std::optional<std::array<double, 3>> initial_velocity{}; ///< --initial-velocity.
    bool allow_steep{};                           ///< --allow-steep.
};

/// @brief RoughTerrain: C = (kappa / ln(d / z0))^2 for the terrain facets.
class RoughLaw {
public:
    static constexpr double KAPPA{ 0.4 };
    static constexpr double SAMPLE_DISTANCE{ 0.5 };

    /// @throws std::invalid_argument for a z0 or K0 outside the law's domain
    RoughLaw(double z0, double k0);

    double z0{};
    double k0{};
    double c{};
};

/// @brief Everything the container is made from.
struct ContainerParts {
    const TerrainInput* input{};
    const VelocityTable* velocities{};
    const TerrainGeometry* geometry{};
    const SamplerSearch* sampler{};
    const PermutedMaps* permuted{};
    IdentityValues identities{};
    DiagTotals diag{};                 ///< The diagnostics of all facets and cells.
    const SlabStream* stream{};        ///< D and direct_arm in temporary files.
    const ArmTables* arms{};
    const WallList* wall{};
    const GradientMap* gradient{};
};

/// @brief The output file and its SHA-256, block by block: while one thread writes a full block and another hashes it,
/// the caller fills the next block.
class ContainerSink {
public:
    /// @brief Creates the file. @throws std::runtime_error when it cannot be created
    explicit ContainerSink(const std::filesystem::path& path);
    /// @brief Waits for the block in flight (its errors are dropped here; finish() reports them).
    ~ContainerSink();
    ContainerSink(const ContainerSink&) = delete;
    ContainerSink& operator=(const ContainerSink&) = delete;

    /// @brief Appends bytes. @throws std::runtime_error when a write or the hash of an earlier block failed
    void put(const void* data, std::size_t size);

    /// @brief Appends the whole content of a file. @throws std::runtime_error when it cannot be read
    void copy_file(const std::filesystem::path& path, std::uint64_t bytes);

    /// @brief Writes the rest, closes the file and returns the digest of all bytes.
    /// @throws std::runtime_error when a write or the hash failed
    [[nodiscard]] std::string finish();

    /// @brief The bytes appended so far.
    [[nodiscard]] std::uint64_t bytes() const { return bytes_; }

private:
    static constexpr std::size_t BLOCK{ std::size_t{ 16 } << 20 };

    std::filesystem::path path_{};
    std::ofstream file_{};
    Sha256Stream hash_{};
    std::vector<char> filling_{};   ///< The block the caller fills.
    std::vector<char> pending_{};   ///< The block in flight.
    std::thread writer_{};
    std::thread hasher_{};
    std::exception_ptr write_error_{};
    std::exception_ptr hash_error_{};
    std::uint64_t bytes_{};

    /// @brief Waits for the block in flight and rethrows its error.
    void wait();
    /// @brief Hands the filled block to the two threads.
    void hand_over();
};

/// @brief Writes the container.
class ContainerWriter {
public:
    ContainerWriter(const ContainerParts& parts, const ContainerOptions& options);

    /// @brief Writes the file: the header, then every section in the order of ContainerBuild.sections.
    /// @return the SHA-256 of the file
    /// @throws std::runtime_error for a value that does not fit its section type, bad base rows, or a failed write
    [[nodiscard]] std::string write(ContainerSink& sink) const;

    /// @brief The provenance text (json.dumps(prov, indent=1)); the exporter fields name this tool.
    [[nodiscard]] std::string provenance() const;

    /// @brief The f0 values of one cell (the same for every cell: rest w_i, or UniformStart's feq).
    /// @throws std::runtime_error when numpy's product gives different rows (then one row does not stand for all)
    [[nodiscard]] std::vector<double> initial_row() const;

    /// @brief Parses "ux,uy,uz" as UniformStart.parse. @throws std::invalid_argument for a bad text or |u| >= 1/sqrt(3)
    [[nodiscard]] static std::array<double, 3> parse_velocity(const std::string& text);

    /// @brief The number of sections of a container.
    [[nodiscard]] std::uint32_t section_count() const;

    /// @brief The OpenBLAS the dense algebra uses (recorded in the provenance).
    static constexpr const char* OPENBLAS{ "scipy-openblas64 0.3.29.265.1" };

private:
    ContainerParts parts_{};
    ContainerOptions options_{};
    int z_lo_{};

    /// @brief The provenance fields of the case: format, sizes, box, nu, K, f0 and s0.
    void case_json(JsonWriter& json) const;
    /// @brief The identities, the gradient report and the rule #253 count.
    void reports_json(JsonWriter& json) const;
    /// @brief The builder diagnostics and the sampler branches.
    /// @throws std::runtime_error when CNG fails
    void diagnostics_json(JsonWriter& json) const;
    /// @brief The exporter fields of this tool, the support closure and the rough law. @throws std::runtime_error when CNG fails
    void tool_json(JsonWriter& json) const;
    [[nodiscard]] std::vector<double> base_rows(const std::vector<double>& area, const std::vector<double>& normal) const;
    // each writes its sections in the order of ContainerBuild.sections and returns how many
    [[nodiscard]] std::uint32_t cell_sections(ContainerSink& sink) const;
    [[nodiscard]] std::uint32_t surface_sections(ContainerSink& sink, const std::vector<double>& normal, const std::vector<double>& area) const;
    [[nodiscard]] std::uint32_t map_sections(ContainerSink& sink) const;
    [[nodiscard]] std::uint32_t state_sections(ContainerSink& sink, const std::vector<double>& normal, const std::vector<double>& area) const;
    [[nodiscard]] std::uint32_t tail_sections(ContainerSink& sink) const;
};

} // namespace fluidx3d::experimental::wall_builder
