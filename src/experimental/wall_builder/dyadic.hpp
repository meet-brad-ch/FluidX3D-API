#pragma once
/// @file dyadic.hpp
/// @brief An exact dyadic rational: an integer times a power of two.

#include "wide_int.hpp"

#include <cstdint>

namespace fluidx3d::experimental::wall_builder {

/// @brief The exact value numerator * 2^exponent.
///
/// Every node height is a binary64 value, so it is dyadic. A column triangle has its vertices at integer x, y offsets 0
/// and 1, so the determinant of its plane is +1 or -1, and the plane coefficients a, b and g are integer combinations
/// of heights: dyadic as well. This class holds them exactly, where builder3d.py uses fractions.Fraction.
class Dyadic {
public:
    /// @brief Makes zero.
    Dyadic() = default;

    /// @brief Makes the exact value of a finite binary64.
    /// @param value a finite value
    /// @return the same value, with an odd numerator (or zero)
    /// @throws std::domain_error when value is not finite
    [[nodiscard]] static Dyadic from_double(double value);

    /// @brief Makes the exact value of an integer.
    /// @param value any signed 64-bit integer
    [[nodiscard]] static Dyadic from_integer(std::int64_t value);

    /// @brief The exact sum. @throws std::overflow_error when the aligned numerators need more than WideInt::BITS (1024) bits
    [[nodiscard]] Dyadic operator+(const Dyadic& other) const;

    /// @brief The exact difference. @throws std::overflow_error when the aligned numerators need more than WideInt::BITS (1024) bits
    [[nodiscard]] Dyadic operator-(const Dyadic& other) const;

    /// @brief The exact product. @throws std::overflow_error when the numerator needs more than WideInt::BITS (1024) bits
    [[nodiscard]] Dyadic operator*(const Dyadic& other) const;

    /// @brief The value with the opposite sign.
    [[nodiscard]] Dyadic operator-() const;

    /// @brief The sign. @return -1, 0 or +1
    [[nodiscard]] int sign() const;

    /// @brief Exact three-way comparison. @return -1, 0 or +1 for this <, =, > other
    [[nodiscard]] int compare(const Dyadic& other) const;

    /// @brief The largest integer not above the value (Python's math.floor of a Fraction).
    /// @throws std::overflow_error when the result does not fit in a signed 64-bit integer
    [[nodiscard]] std::int64_t floor() const;

    /// @brief The nearest binary64, ties to even (Python's float(Fraction)).
    /// @throws std::domain_error when the value is subnormal or outside binary64
    [[nodiscard]] double to_double() const;

    /// @brief The integer part of the representation.
    [[nodiscard]] const WideInt& numerator() const { return numerator_; }

    /// @brief The power of two of the representation.
    [[nodiscard]] int exponent() const { return exponent_; }

private:
    WideInt numerator_{}; ///< The integer factor.
    int exponent_{};      ///< The power of two; 0 for zero.

    /// @brief Makes numerator * 2^exponent; zero gets the exponent 0.
    Dyadic(const WideInt& numerator, int exponent);

    /// @brief The numerator of this value written with a smaller or equal exponent.
    /// @param exponent the target exponent, at most exponent_
    /// @throws std::overflow_error when the shifted numerator needs more than WideInt::BITS (1024) bits
    [[nodiscard]] WideInt numerator_at(int exponent) const;
};

} // namespace fluidx3d::experimental::wall_builder
