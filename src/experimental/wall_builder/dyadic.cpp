/// @file dyadic.cpp
/// @brief An exact dyadic rational: an integer times a power of two.

#include "dyadic.hpp"

#include "correct_rounding.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fluidx3d::experimental::wall_builder {

namespace {
constexpr int BINARY64_MANTISSA_BITS{ 53 };
}

Dyadic::Dyadic(const WideInt& numerator, const int exponent)
    : numerator_{ numerator }
    , exponent_{ numerator.is_zero() ? 0 : exponent }
{
}

Dyadic Dyadic::from_double(const double value)
{
    if (!std::isfinite(value)) {
        throw std::domain_error("Dyadic::from_double: the value " + std::to_string(value) + " is not finite");
    }
    if (value == 0.0) {
        return Dyadic{};
    }
    // value = fraction * 2^power with 0.5 <= |fraction| < 1; fraction * 2^53 is an exact integer
    int power{ 0 };
    const double fraction{ std::frexp(value, &power) };
    const double scaled_fraction{ std::ldexp(fraction, BINARY64_MANTISSA_BITS) };
    const auto integer{ static_cast<std::int64_t>(scaled_fraction) };
    const std::uint64_t magnitude{ static_cast<std::uint64_t>(integer < 0 ? -integer : integer) };
    // an odd numerator keeps later alignments short
    const int trailing{ std::countr_zero(magnitude) };
    const std::int64_t odd{ integer / (std::int64_t{ 1 } << trailing) };
    const int exponent{ power - BINARY64_MANTISSA_BITS + trailing };
    return Dyadic{ WideInt{ odd }, exponent };
}

Dyadic Dyadic::from_integer(const std::int64_t value)
{
    return Dyadic{ WideInt{ value }, 0 };
}

WideInt Dyadic::numerator_at(const int exponent) const
{
    if (exponent > exponent_ && !numerator_.is_zero()) {
        throw std::invalid_argument("Dyadic::numerator_at: exponent " + std::to_string(exponent) + " above " + std::to_string(exponent_));
    }
    if (numerator_.is_zero()) {
        return WideInt{};
    }
    return numerator_.shifted_left(exponent_ - exponent);
}

Dyadic Dyadic::operator+(const Dyadic& other) const
{
    if (numerator_.is_zero()) {
        return other;
    }
    if (other.numerator_.is_zero()) {
        return *this;
    }
    const int common{ std::min(exponent_, other.exponent_) };
    const WideInt own_part{ numerator_at(common) };
    const WideInt other_part{ other.numerator_at(common) };
    return Dyadic{ own_part + other_part, common };
}

Dyadic Dyadic::operator-() const
{
    return Dyadic{ -numerator_, exponent_ };
}

Dyadic Dyadic::operator-(const Dyadic& other) const
{
    const Dyadic negated_other{ -other };
    return *this + negated_other;
}

Dyadic Dyadic::operator*(const Dyadic& other) const
{
    const WideInt product{ numerator_ * other.numerator_ };
    return Dyadic{ product, exponent_ + other.exponent_ };
}

int Dyadic::sign() const
{
    return numerator_.sign();
}

int Dyadic::compare(const Dyadic& other) const
{
    const Dyadic difference{ *this - other };
    return difference.sign();
}

std::int64_t Dyadic::floor() const
{
    if (numerator_.is_zero()) {
        return 0;
    }
    // |value| = magnitude / 2^shift; floor is the truncated quotient, one less for a negative value with a fraction
    const WideInt raw_magnitude{ numerator_.magnitude() };
    const WideInt magnitude{ exponent_ > 0 ? raw_magnitude.shifted_left(exponent_) : raw_magnitude };
    const int shift{ exponent_ < 0 ? -exponent_ : 0 };
    const int integer_bits{ magnitude.bit_length() - shift };
    if (integer_bits > 62) {
        throw std::overflow_error("Dyadic::floor: the integer part of " + numerator_.to_string() + " * 2^" + std::to_string(exponent_)
                                  + " does not fit in 64 bits");
    }
    const std::uint64_t truncated_bits{ integer_bits > 0 ? magnitude.bits_from(shift, integer_bits) : 0u };
    const auto truncated{ static_cast<std::int64_t>(truncated_bits) };
    const int fraction_bits{ std::min(shift, WideInt::BITS) };
    const bool has_fraction{ magnitude.any_bit_below(fraction_bits) };
    if (numerator_.sign() > 0) {
        return truncated;
    }
    return has_fraction ? -truncated - 1 : -truncated;
}

double Dyadic::to_double() const
{
    return CorrectRounding::round_quotient(numerator_, 1u, exponent_);
}

} // namespace fluidx3d::experimental::wall_builder
