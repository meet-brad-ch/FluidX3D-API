/// @file wall_builder_test.cpp
/// @brief Unit tests of the wall builder's exact arithmetic against Python reference vectors
/// (scripts/slip_wall/make_wall_builder_vectors.py writes tests/data/slip_wall/wall_builder_vectors.txt).

#include "experimental/wall_builder/correct_rounding.hpp"
#include "experimental/wall_builder/dyadic.hpp"
#include "experimental/wall_builder/terrain.hpp"
#include "experimental/wall_builder/wide_int.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using fluidx3d::experimental::wall_builder::CorrectRounding;
using fluidx3d::experimental::wall_builder::Direction;
using fluidx3d::experimental::wall_builder::Dyadic;
using fluidx3d::experimental::wall_builder::TerrainGeometry;
using fluidx3d::experimental::wall_builder::TerrainRecord;
using fluidx3d::experimental::wall_builder::WideInt;

/// @brief One line of the reference file.
struct Vector {
    std::string kind{};                ///< "S", "M" or "R".
    std::vector<std::string> fields{}; ///< The fields after the kind.
};

/// @brief Reads the reference file.
class VectorFile {
public:
    /// @brief Reads every line. @throws std::runtime_error when the file cannot be opened
    VectorFile()
    {
        std::ifstream stream{ FLUIDX3D_WALL_BUILDER_VECTORS };
        if (!stream) {
            throw std::runtime_error(std::string("cannot open ") + FLUIDX3D_WALL_BUILDER_VECTORS);
        }
        std::string line{};
        while (std::getline(stream, line)) {
            std::istringstream words{ line };
            Vector vector{};
            words >> vector.kind;
            std::string field{};
            while (words >> field) {
                vector.fields.push_back(field);
            }
            vectors.push_back(vector);
        }
    }

    std::vector<Vector> vectors{}; ///< The lines in file order.
};

/// @brief Parses a signed decimal integer with the public operations of WideInt.
/// @param text an optional '-' and decimal digits
/// @return the value
WideInt parse_decimal(const std::string& text)
{
    const bool negative{ !text.empty() && text[0] == '-' };
    const std::size_t first{ negative ? 1u : 0u };
    const WideInt ten{ 10 };
    WideInt value{};
    for (std::size_t i{ first }; i < text.size(); ++i) {
        const WideInt digit{ static_cast<std::int64_t>(text[i] - '0') };
        value = value * ten + digit;
    }
    return negative ? -value : value;
}

/// @brief The bits of a binary64, for an exact comparison that also tells -0.0 from 0.0.
std::uint64_t bits_of(const double value)
{
    return std::bit_cast<std::uint64_t>(value);
}

TEST(WallBuilderWideInt, DecimalRoundTripAndSmallValues)
{
    EXPECT_EQ(WideInt{ 0 }.to_string(), "0");
    EXPECT_EQ(WideInt{ -1 }.to_string(), "-1");
    EXPECT_EQ(WideInt{ INT64_MIN }.to_string(), "-9223372036854775808");
    const std::string big{ "-123456789012345678901234567890123456789012345678901234567890" };
    EXPECT_EQ(parse_decimal(big).to_string(), big);
    EXPECT_EQ(WideInt::from_shifted(3, 100).to_string(), "3802951800684688204490109616128");
}

TEST(WallBuilderWideInt, OverflowThrows)
{
    const WideInt top{ WideInt::from_shifted(1, WideInt::BITS - 1) };
    EXPECT_THROW(static_cast<void>(top + top), std::overflow_error);
    EXPECT_THROW(static_cast<void>(top * WideInt{ 2 }), std::overflow_error);
    EXPECT_THROW(static_cast<void>(top.shifted_left(1)), std::overflow_error);
}

TEST(WallBuilderWideInt, SumsAndProductsMatchPython)
{
    const VectorFile file{};
    int checked{ 0 };
    for (const Vector& vector : file.vectors) {
        if (vector.kind != "S" && vector.kind != "M") {
            continue;
        }
        const WideInt a{ parse_decimal(vector.fields[0]) };
        const WideInt b{ parse_decimal(vector.fields[1]) };
        const WideInt result{ vector.kind == "S" ? a + b : a * b };
        EXPECT_EQ(result.to_string(), vector.fields[2]) << vector.kind << " " << vector.fields[0] << " " << vector.fields[1];
        ++checked;
    }
    EXPECT_GT(checked, 0);
}

TEST(WallBuilderCorrectRounding, QuotientsMatchPythonBitForBit)
{
    const VectorFile file{};
    int checked{ 0 };
    for (const Vector& vector : file.vectors) {
        if (vector.kind != "R") {
            continue;
        }
        const WideInt numerator{ parse_decimal(vector.fields[0]) };
        const auto divisor{ static_cast<std::uint32_t>(std::stoul(vector.fields[1])) };
        const int exponent{ std::stoi(vector.fields[2]) };
        const double expected{ std::strtod(vector.fields[3].c_str(), nullptr) };
        const double actual{ CorrectRounding::round_quotient(numerator, divisor, exponent) };
        EXPECT_EQ(bits_of(actual), bits_of(expected)) << vector.fields[0] << " / " << vector.fields[1] << " * 2^" << vector.fields[2];
        ++checked;
    }
    EXPECT_GT(checked, 1000);
}

TEST(WallBuilderDyadic, SumsProductsAndFloorMatchPython)
{
    const VectorFile file{};
    int checked{ 0 };
    for (const Vector& vector : file.vectors) {
        if (vector.kind != "D") {
            continue;
        }
        const double x{ std::strtod(vector.fields[0].c_str(), nullptr) };
        const double y{ std::strtod(vector.fields[1].c_str(), nullptr) };
        const Dyadic exact_x{ Dyadic::from_double(x) };
        const Dyadic exact_y{ Dyadic::from_double(y) };
        const Dyadic sum{ exact_x + exact_y };
        const Dyadic product{ exact_x * exact_y };
        const std::int64_t expected_floor{ std::stoll(vector.fields[2]) };
        const double expected_sum{ std::strtod(vector.fields[3].c_str(), nullptr) };
        const double expected_product{ std::strtod(vector.fields[4].c_str(), nullptr) };
        EXPECT_EQ(sum.floor(), expected_floor) << vector.fields[0] << " + " << vector.fields[1];
        EXPECT_EQ(bits_of(sum.to_double()), bits_of(expected_sum)) << vector.fields[0] << " + " << vector.fields[1];
        EXPECT_EQ(bits_of(product.to_double()), bits_of(expected_product)) << vector.fields[0] << " * " << vector.fields[1];
        EXPECT_EQ(bits_of(exact_x.to_double()), bits_of(x)) << vector.fields[0];
        ++checked;
    }
    EXPECT_GT(checked, 1000);
}

TEST(WallBuilderLibm, Atan2MatchesPythonBitForBit)
{
    const VectorFile file{};
    int checked{ 0 };
    for (const Vector& vector : file.vectors) {
        if (vector.kind != "A") {
            continue;
        }
        const double y{ std::strtod(vector.fields[0].c_str(), nullptr) };
        const double x{ std::strtod(vector.fields[1].c_str(), nullptr) };
        const double expected{ std::strtod(vector.fields[2].c_str(), nullptr) };
        const double actual{ std::atan2(y, x) };
        EXPECT_EQ(bits_of(actual), bits_of(expected)) << "atan2(" << vector.fields[0] << ", " << vector.fields[1] << ")";
        ++checked;
    }
    EXPECT_GT(checked, 10000);
}

// Only the edge vectors of cell_pieces (components -1, 0, 1): Python's math.hypot differs from std::hypot elsewhere.
TEST(WallBuilderLibm, HypotMatchesPythonOnCellEdges)
{
    const VectorFile file{};
    int checked{ 0 };
    for (const Vector& vector : file.vectors) {
        if (vector.kind != "H") {
            continue;
        }
        const double x{ std::strtod(vector.fields[0].c_str(), nullptr) };
        const double y{ std::strtod(vector.fields[1].c_str(), nullptr) };
        const double expected{ std::strtod(vector.fields[2].c_str(), nullptr) };
        const double actual{ std::hypot(x, y) };
        EXPECT_EQ(bits_of(actual), bits_of(expected)) << "hypot(" << vector.fields[0] << ", " << vector.fields[1] << ")";
        ++checked;
    }
    EXPECT_EQ(checked, 9);
}

// The column index of TerrainGeometry replaces the stored cell list: cell n is the n-th (i, j, k) in sorted order, the
// first fluid layer of a column is the lowest k with k + 1 > its lowest node height, and the index wraps i and j.
TEST(WallBuilderTerrain, ColumnIndexMatchesTheSortedCellList)
{
    TerrainRecord record{};
    record.lx = 3;
    record.ly = 2;
    record.z_top = 5;
    record.heights = { 0.5, 1.25, 2.0, 0.0, 3.5, 1.0 };
    record.main_diagonal = { true, false, true, false, true, false };
    record.directions = { Direction{ 0, 0, 0 } };
    const TerrainGeometry geometry{ record, 2 };
    std::size_t expected{ 0 };
    for (int i{ 0 }; i < record.lx; ++i) {
        for (int j{ 0 }; j < record.ly; ++j) {
            const double h00{ record.heights[static_cast<std::size_t>(i * record.ly + j)] };
            const double h01{ record.heights[static_cast<std::size_t>(i * record.ly + (j + 1) % record.ly)] };
            const double h10{ record.heights[static_cast<std::size_t>(((i + 1) % record.lx) * record.ly + j)] };
            const double h11{ record.heights[static_cast<std::size_t>(((i + 1) % record.lx) * record.ly + (j + 1) % record.ly)] };
            const double lowest{ std::min({ h00, h01, h10, h11 }) };
            int k_first{ 0 };
            while (static_cast<double>(k_first + 1) <= lowest) {
                ++k_first;
            }
            EXPECT_EQ(geometry.first_layer(i, j), k_first) << i << " " << j;
            EXPECT_EQ(geometry.cell_index(i, j, k_first - 1), -1);
            EXPECT_EQ(geometry.cell_index(i, j, record.z_top), -1);
            for (int k{ k_first }; k < record.z_top; ++k) {
                const std::array<int, 3> ijk{ geometry.cell_ijk(expected) };
                EXPECT_EQ(ijk, (std::array<int, 3>{ i, j, k }));
                EXPECT_EQ(geometry.cell_index(i, j, k), static_cast<std::int64_t>(expected));
                EXPECT_EQ(geometry.cell_index(i + record.lx, j - record.ly, k), static_cast<std::int64_t>(expected));
                ++expected;
            }
        }
    }
    EXPECT_EQ(geometry.cell_count(), expected);
    EXPECT_EQ(geometry.column_first(record.lx), expected);
}

TEST(WallBuilderCorrectRounding, SubnormalThrows)
{
    EXPECT_THROW(static_cast<void>(CorrectRounding::round_quotient(WideInt{ 1 }, 1u, -1074)), std::domain_error);
}

} // namespace
