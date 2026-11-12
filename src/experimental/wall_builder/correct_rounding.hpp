#pragma once
/// @file correct_rounding.hpp
/// @brief Rounds an exact rational value to binary64 exactly as Python's int / int does.

#include "wide_int.hpp"

#include <cstdint>

namespace fluidx3d::experimental::wall_builder {

/// @brief Rounds numerator / divisor * 2^exponent to the nearest binary64, ties to even.
///
/// This is the rounding of Python's true division of two integers, which builder3d.py uses for every volume
/// (sum of determinants / 6 D^3, with D a power of two) and float(Fraction) uses for every exact plane value.
/// A result in the subnormal range or above the binary64 range throws: the builder never makes such values, and a
/// silent difference from Python is worse than a stop.
class CorrectRounding {
public:
    /// @brief The correctly rounded value of numerator / divisor * 2^exponent.
    /// @param numerator any value
    /// @param divisor a small positive divisor (1 for a dyadic value, 3 for a volume)
    /// @param exponent the power of two
    /// @return the nearest binary64, ties to even; 0.0 for a zero numerator
    /// @throws std::invalid_argument when divisor is 0
    /// @throws std::domain_error when the result is subnormal or does not fit in binary64
    [[nodiscard]] static double round_quotient(const WideInt& numerator, std::uint32_t divisor, int exponent);

private:
    static constexpr int MANTISSA_BITS{ 53 };  ///< Significant bits of binary64.
    static constexpr int QUOTIENT_BITS{ 55 };  ///< Quotient bits kept: the mantissa, the round bit and one more.
    static constexpr int MIN_NORMAL_EXPONENT{ -1022 }; ///< Exponent of the smallest normal binary64.
    static constexpr int MAX_EXPONENT{ 1023 }; ///< Exponent of the largest binary64.

    /// @brief Rounds quotient * 2^exponent, with the information that a nonzero remainder lies below the quotient.
    /// @param quotient a positive integer of QUOTIENT_BITS bits or more
    /// @param remainder_nonzero true when the exact value is above quotient * 2^exponent
    /// @param exponent the power of two of the quotient's least significant bit
    /// @return the nearest binary64, ties to even
    /// @throws std::domain_error when the result is subnormal or does not fit in binary64
    [[nodiscard]] static double round_positive(const WideInt& quotient, bool remainder_nonzero, int exponent);
};

} // namespace fluidx3d::experimental::wall_builder
