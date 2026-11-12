/// @file correct_rounding.cpp
/// @brief Rounds an exact rational value to binary64 exactly as Python's int / int does.

#include "correct_rounding.hpp"

#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fluidx3d::experimental::wall_builder {

double CorrectRounding::round_quotient(const WideInt& numerator, const std::uint32_t divisor, const int exponent)
{
    if (divisor == 0u) {
        throw std::invalid_argument("CorrectRounding::round_quotient: divisor 0");
    }
    if (numerator.is_zero()) {
        return 0.0;
    }
    // Shift the numerator so that the quotient has at least QUOTIENT_BITS bits. Then the remainder lies below the
    // round bit and only decides the sticky bit.
    const WideInt magnitude{ numerator.magnitude() };
    const int divisor_bits{ static_cast<int>(std::bit_width(divisor)) };
    const int wanted_bits{ QUOTIENT_BITS + divisor_bits };
    const int length{ magnitude.bit_length() };
    const int shift{ length < wanted_bits ? wanted_bits - length : 0 };
    const WideInt scaled{ magnitude.shifted_left(shift) };
    WideInt quotient{};
    const std::uint32_t remainder{ scaled.divide_magnitude(divisor, quotient) };
    const bool remainder_nonzero{ remainder != 0u };
    const double value{ round_positive(quotient, remainder_nonzero, exponent - shift) };
    return numerator.sign() < 0 ? -value : value;
}

double CorrectRounding::round_positive(const WideInt& quotient, const bool remainder_nonzero, const int exponent)
{
    const int length{ quotient.bit_length() };
    const int dropped{ length - MANTISSA_BITS };
    std::uint64_t mantissa{ quotient.bits_from(dropped, MANTISSA_BITS) };
    const bool round_bit{ quotient.bit(dropped - 1) };
    const bool sticky{ remainder_nonzero || quotient.any_bit_below(dropped - 1) };
    const bool odd{ (mantissa & 1u) != 0u };
    if (round_bit && (sticky || odd)) {
        mantissa += 1u; // 2^53 at most, still exact in binary64
    }
    const int result_exponent{ exponent + dropped };
    const int mantissa_width{ static_cast<int>(std::bit_width(mantissa)) };
    const int leading_exponent{ result_exponent + mantissa_width - 1 };
    if (leading_exponent < MIN_NORMAL_EXPONENT || leading_exponent > MAX_EXPONENT) {
        throw std::domain_error("CorrectRounding: the result 2^" + std::to_string(leading_exponent)
                                + " is subnormal or outside binary64; the builder does not support it");
    }
    const double mantissa_value{ static_cast<double>(mantissa) };
    return std::ldexp(mantissa_value, result_exponent);
}

} // namespace fluidx3d::experimental::wall_builder
