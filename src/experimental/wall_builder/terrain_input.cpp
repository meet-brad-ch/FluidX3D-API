/// @file terrain_input.cpp
/// @brief The terrain input of terrain_container.py: heights CSV, diagonals, domain checks, record text, velocity set.

#include "terrain_input.hpp"

#include "dyadic.hpp"
#include "python_text.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fluidx3d::experimental::wall_builder {

namespace {

/// @brief One height-field node of a column triangle: integer x, y and the exact height.
struct Node {
    int x{};    ///< Unwrapped x.
    int y{};    ///< Unwrapped y.
    Dyadic z{}; ///< Exact height.
};

/// @brief One key of the record's "heights" or "diagonal" dict ("i,j"), for sort_keys.
struct RecordKey {
    std::string text{}; ///< "i,j".
    int i{};            ///< Column x.
    int j{};            ///< Column y.
};

/// @brief The exact slopes of a column triangle (terrain_container.plane_of without gamma).
class TriangleSlopes {
public:
    /// @throws std::domain_error for a degenerate triangle
    TriangleSlopes(const Node& p0, const Node& p1, const Node& p2)
    {
        const int det{ (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y) };
        if (det != 1 && det != -1) {
            throw std::domain_error("plane_of: determinant " + std::to_string(det));
        }
        const Dyadic dz1{ p1.z - p0.z };
        const Dyadic dz2{ p2.z - p0.z };
        const Dyadic sign{ Dyadic::from_integer(det) };
        a = (dz1 * Dyadic::from_integer(p2.y - p0.y) - dz2 * Dyadic::from_integer(p1.y - p0.y)) * sign;
        b = (Dyadic::from_integer(p1.x - p0.x) * dz2 - Dyadic::from_integer(p2.x - p0.x) * dz1) * sign;
    }

    /// @brief True when |a| >= 1 or |b| >= 1 (R11).
    [[nodiscard]] bool steep() const
    {
        const Dyadic one{ Dyadic::from_integer(1) };
        const Dyadic minus_one{ Dyadic::from_integer(-1) };
        return a.compare(one) >= 0 || a.compare(minus_one) <= 0 || b.compare(one) >= 0 || b.compare(minus_one) <= 0;
    }

    Dyadic a{}; ///< dz/dx.
    Dyadic b{}; ///< dz/dy.
};

std::string trimmed(const std::string& text)
{
    const std::size_t first{ text.find_first_not_of(" \t\r\f\v") };
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last{ text.find_last_not_of(" \t\r\f\v") };
    return text.substr(first, last - first + 1);
}

} // namespace

// ------------------------------------------------------------------------------------------------ VelocityTable

VelocityTable VelocityTable::named(const std::string& name)
{
    int limit{ 0 };
    std::array<double, 4> shell{};
    if (name == "D3Q27") {
        limit = 3;
        shell = { 8.0 / 27.0, 2.0 / 27.0, 1.0 / 54.0, 1.0 / 216.0 };
    } else if (name == "D3Q19") {
        limit = 2;
        shell = { 1.0 / 3.0, 1.0 / 18.0, 1.0 / 36.0, 0.0 };
    } else {
        throw std::invalid_argument("unknown velocity set " + name + "; expected D3Q19 or D3Q27");
    }
    VelocityTable table{};
    table.name = name;
    for (int x{ -1 }; x <= 1; ++x) {
        for (int y{ -1 }; y <= 1; ++y) {
            for (int z{ -1 }; z <= 1; ++z) {
                const int length{ x * x + y * y + z * z };
                if (length <= limit) {
                    table.directions.push_back(Direction{ x, y, z });
                    table.weights.push_back(shell[static_cast<std::size_t>(length)]);
                }
            }
        }
    }
    for (const Direction& c : table.directions) {
        const auto found{ std::find_if(table.directions.begin(), table.directions.end(),
                                       [&c](const Direction& o) { return o.x == -c.x && o.y == -c.y && o.z == -c.z; }) };
        table.opposites.push_back(static_cast<int>(found - table.directions.begin()));
    }
    return table;
}

// ------------------------------------------------------------------------------------------------ TerrainInput

TerrainInput TerrainInput::from_csv(const std::string& path, const int z_top, const std::string& name)
{
    std::ifstream stream{ path };
    if (!stream) {
        throw std::runtime_error("cannot open the heights " + path);
    }
    std::vector<std::vector<double>> grid{};
    std::string line{};
    for (int number{ 1 }; std::getline(stream, line); ++number) {
        const std::string row{ trimmed(line) };
        if (row.empty()) {
            continue;
        }
        std::vector<double> values{};
        std::istringstream fields{ row };
        std::string field{};
        while (std::getline(fields, field, ',')) {
            const std::string text{ trimmed(field) };
            char* end{ nullptr };
            const double value{ std::strtod(text.c_str(), &end) };
            if (text.empty() || *end != '\0' || !std::isfinite(value)) {
                throw std::runtime_error("HEIGHTS.csv line " + std::to_string(number) + ": '" + text + "' is not a finite height");
            }
            values.push_back(value);
        }
        grid.push_back(std::move(values));
    }
    const std::size_t ly{ grid.size() };
    const std::size_t lx{ ly > 0 ? grid[0].size() : 0u };
    const bool rectangular{ std::all_of(grid.begin(), grid.end(), [lx](const std::vector<double>& r) { return r.size() == lx; }) };
    if (!rectangular || lx < 2 || ly < 2) {
        throw std::runtime_error("HEIGHTS.csv: every line needs the same number (>= 2) of values, and >= 2 lines");
    }
    TerrainInput input{};
    input.name_ = name;
    input.lx_ = static_cast<int>(lx);
    input.ly_ = static_cast<int>(ly);
    input.z_top_ = z_top;
    input.heights_.assign(lx * ly, 0.0);
    for (std::size_t i{ 0 }; i < lx; ++i) {
        for (std::size_t j{ 0 }; j < ly; ++j) {
            input.heights_[i * ly + j] = grid[j][i];
        }
    }
    input.lowest_node_diagonals();
    return input;
}

double TerrainInput::height(const int i, const int j) const
{
    const int wi{ TerrainGeometry::wrap(i, lx_) };
    const int wj{ TerrainGeometry::wrap(j, ly_) };
    return heights_[static_cast<std::size_t>(wi * ly_ + wj)];
}

void TerrainInput::lowest_node_diagonals()
{
    main_.assign(heights_.size(), false);
    for (int i{ 0 }; i < lx_; ++i) {
        for (int j{ 0 }; j < ly_; ++j) {
            // the lowest node; ties by the order (0,0), (1,0), (0,1), (1,1): only a strictly lower node replaces it
            const std::array<double, 4> nodes{ height(i, j), height(i + 1, j), height(i, j + 1), height(i + 1, j + 1) };
            std::size_t low{ 0 };
            for (std::size_t n{ 1 }; n < 4; ++n) {
                if (nodes[n] < nodes[low]) {
                    low = n;
                }
            }
            main_[static_cast<std::size_t>(i * ly_ + j)] = low == 0 || low == 3;
        }
    }
}

void TerrainInput::domain_checks(const bool allow_steep) const
{
    const double highest{ *std::max_element(heights_.begin(), heights_.end()) };
    if (highest > static_cast<double>(z_top_ - 2)) {
        throw std::runtime_error("R10: clearance " + PythonFloat::repr(static_cast<double>(z_top_) - highest) + " < 2");
    }
    for (int i{ 0 }; i < lx_ && !allow_steep; ++i) {
        for (int j{ 0 }; j < ly_; ++j) {
            const Node p00{ i, j, Dyadic::from_double(height(i, j)) };
            const Node p10{ i + 1, j, Dyadic::from_double(height(i + 1, j)) };
            const Node p11{ i + 1, j + 1, Dyadic::from_double(height(i + 1, j + 1)) };
            const Node p01{ i, j + 1, Dyadic::from_double(height(i, j + 1)) };
            const bool main{ main_[static_cast<std::size_t>(i * ly_ + j)] };
            const TriangleSlopes t0{ main ? TriangleSlopes{ p00, p10, p11 } : TriangleSlopes{ p00, p10, p01 } };
            const TriangleSlopes t1{ main ? TriangleSlopes{ p00, p11, p01 } : TriangleSlopes{ p10, p11, p01 } };
            if (t0.steep() || t1.steep()) {
                throw std::runtime_error("R11: slope of at least 1 at column (" + std::to_string(i) + ", " + std::to_string(j) + ")");
            }
        }
    }
    for (int i{ 0 }; i < lx_; ++i) {
        for (int j{ 0 }; j < ly_; ++j) {
            const bool main{ main_[static_cast<std::size_t>(i * ly_ + j)] };
            const double n00{ height(i, j) };
            const double n10{ height(i + 1, j) };
            const double n01{ height(i, j + 1) };
            const double n11{ height(i + 1, j + 1) };
            const double lo{ main ? std::max(n10, n01) : std::max(n00, n11) };
            const double hi{ main ? std::min(n00, n11) : std::min(n10, n01) };
            const double level{ std::floor(lo) + 1.0 };
            if (level <= hi) {
                throw std::runtime_error("R1: column (" + std::to_string(i) + ", " + std::to_string(j) + ") disconnected at level "
                                         + std::to_string(static_cast<long long>(level)));
            }
        }
    }
}

std::string TerrainInput::record_json() const
{
    std::vector<RecordKey> keys{};
    for (int i{ 0 }; i < lx_; ++i) {
        for (int j{ 0 }; j < ly_; ++j) {
            keys.push_back(RecordKey{ std::to_string(i) + "," + std::to_string(j), i, j });
        }
    }
    std::sort(keys.begin(), keys.end(), [](const RecordKey& a, const RecordKey& b) { return a.text < b.text; });
    JsonWriter json{ false };
    json.begin_object();
    json.key("Lx");
    json.value(lx_);
    json.key("Ly");
    json.value(ly_);
    json.key("Z_top");
    json.value(z_top_);
    json.key("check_clearance");
    json.value(true);
    json.key("check_slope");
    json.value(true);
    json.key("diagonal");
    json.begin_object();
    for (const RecordKey& k : keys) {
        json.key(k.text);
        json.value(main_[static_cast<std::size_t>(k.i * ly_ + k.j)] ? "main" : "anti");
    }
    json.end_object();
    json.key("heights");
    json.begin_object();
    for (const RecordKey& k : keys) {
        json.key(k.text);
        json.value(PythonFloat::hex(heights_[static_cast<std::size_t>(k.i * ly_ + k.j)]));
    }
    json.end_object();
    json.key("name");
    json.value(name_);
    json.end_object();
    return json.text();
}

TerrainRecord TerrainInput::record(const VelocityTable& velocities, const int neighbours) const
{
    TerrainRecord record{};
    record.lx = lx_;
    record.ly = ly_;
    record.z_top = z_top_;
    record.heights = heights_;
    record.main_diagonal = main_;
    record.directions = velocities.directions;
    record.neighbours = neighbours;
    return record;
}

} // namespace fluidx3d::experimental::wall_builder
