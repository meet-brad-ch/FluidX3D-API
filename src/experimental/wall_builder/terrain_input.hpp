#pragma once
/// @file terrain_input.hpp
/// @brief The terrain input of terrain_container.py: the heights CSV, the column diagonals, the domain checks (R10, R11,
/// R1), the manifest record text and the velocity set.

#include "terrain.hpp"

#include <string>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief A lattice velocity set in lexicographic order (VelocitySet.named).
struct VelocityTable {
    std::string name{};                  ///< "D3Q27" or "D3Q19".
    std::vector<Direction> directions{}; ///< Lexicographic: x slowest, z fastest.
    std::vector<double> weights{};       ///< float(Fraction) of each weight.
    std::vector<int> opposites{};        ///< Index of -c.

    /// @brief The named set. @throws std::invalid_argument for another name
    [[nodiscard]] static VelocityTable named(const std::string& name);
};

/// @brief TerrainManifest of terrain_container.py.
class TerrainInput {
public:
    /// @brief Reads a CSV of Ny lines of Nx node heights (line j, value i).
    /// @throws std::runtime_error naming the line for a bad file
    [[nodiscard]] static TerrainInput from_csv(const std::string& path, int z_top, const std::string& name);

    /// @brief R10 (clearance >= 2), R11 (slopes below 1, unless allow_steep) and R1 (column connectivity).
    /// @throws std::runtime_error with the rule's message
    void domain_checks(bool allow_steep) const;

    /// @brief json.dumps(record, sort_keys=True) of manifest_record (the text of manifest_sha256).
    [[nodiscard]] std::string record_json() const;

    /// @brief The record of the geometry.
    [[nodiscard]] TerrainRecord record(const VelocityTable& velocities, int neighbours) const;

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] int lx() const { return lx_; }
    [[nodiscard]] int ly() const { return ly_; }
    [[nodiscard]] int z_top() const { return z_top_; }

private:
    std::string name_{};
    int lx_{};
    int ly_{};
    int z_top_{};
    std::vector<double> heights_{}; ///< Index i * ly + j.
    std::vector<bool> main_{};      ///< Diagonal per column, index i * ly + j.

    [[nodiscard]] double height(int i, int j) const;
    void lowest_node_diagonals();
};

} // namespace fluidx3d::experimental::wall_builder
