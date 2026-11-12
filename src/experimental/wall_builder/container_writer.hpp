#pragma once
/// @file container_writer.hpp
/// @brief The VWNAT01 container of terrain_container.py (ContainerBuild.sections and Writer), its provenance text and the
/// build report.

#include "maps.hpp"
#include "pass_runner.hpp"
#include "sampler.hpp"
#include "terrain_input.hpp"
#include "wall_tables.hpp"

#include <cstdint>
#include <optional>
#include <string>
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
    const PassRunner* passes{};
    const SamplerSearch* sampler{};
    const AssembledMaps* maps{};
    const PermutedMaps* permuted{};
    IdentityValues identities{};
    const ArmTables* arms{};
    const WallList* wall{};
    const GradientMap* gradient{};
};

/// @brief Writes the container.
class ContainerWriter {
public:
    ContainerWriter(const ContainerParts& parts, const ContainerOptions& options);

    /// @brief The container as consecutive chunks: the file header, then one chunk per section in the order of
    /// ContainerBuild.sections. The file is their concatenation.
    /// @throws std::runtime_error for a value that does not fit its section type or bad base rows
    [[nodiscard]] std::vector<std::vector<char>> chunks() const;

    /// @brief The provenance text (json.dumps(prov, indent=1)); the exporter fields name this tool.
    [[nodiscard]] std::string provenance() const;

    /// @brief The f0 section values.
    [[nodiscard]] std::vector<double> initial_state() const;

    /// @brief Parses "ux,uy,uz" as UniformStart.parse. @throws std::invalid_argument for a bad text or |u| >= 1/sqrt(3)
    [[nodiscard]] static std::array<double, 3> parse_velocity(const std::string& text);

    /// @brief The OpenBLAS the dense algebra uses (recorded in the provenance).
    static constexpr const char* OPENBLAS{ "scipy-openblas64 0.3.29.265.1" };

private:
    ContainerParts parts_{};
    ContainerOptions options_{};
    int z_lo_{};

    [[nodiscard]] std::vector<double> base_rows(const std::vector<double>& area, const std::vector<double>& normal) const;
};

} // namespace fluidx3d::experimental::wall_builder
