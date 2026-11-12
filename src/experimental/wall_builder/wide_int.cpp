/// @file wide_int.cpp
/// @brief The fixed-width signed integer of the wall builder: 64-bit limbs with the MSVC x64 intrinsics.

#include "wide_int.hpp"

#include <intrin.h>

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace fluidx3d::experimental::wall_builder {

namespace {
constexpr int DECIMAL_DIGITS_MAX{ 320 }; // 2^1024 has 309 decimal digits
}

WideInt::WideInt(const WideInt& other)
    : negative_{ other.negative_ }
    , used_{ other.used_ }
{
    std::copy_n(other.limbs_.begin(), used_, limbs_.begin());
}

WideInt& WideInt::operator=(const WideInt& other)
{
    negative_ = other.negative_;
    used_ = other.used_;
    std::copy_n(other.limbs_.begin(), used_, limbs_.begin());
    return *this;
}

WideInt::WideInt(const std::int64_t value)
{
    const std::uint64_t as_unsigned{ static_cast<std::uint64_t>(value) };
    // 0 - x in unsigned arithmetic is the magnitude of a negative value, also for INT64_MIN
    limbs_[0] = value < 0 ? 0u - as_unsigned : as_unsigned;
    used_ = limbs_[0] != 0u ? 1 : 0;
    negative_ = value < 0;
}

WideInt WideInt::from_limbs(const std::uint64_t* const limbs, const int count)
{
    if (count < 0 || count > LIMBS) {
        throw std::invalid_argument("WideInt::from_limbs: count " + std::to_string(count));
    }
    WideInt result{};
    std::copy_n(limbs, count, result.limbs_.begin());
    result.trim(count);
    return result;
}

WideInt WideInt::from_shifted(const std::int64_t value, const int shift)
{
    const WideInt base{ value };
    return base.shifted_left(shift);
}

void WideInt::trim(const int bound)
{
    int count{ bound };
    while (count > 0 && limbs_[static_cast<std::size_t>(count - 1)] == 0u) {
        --count;
    }
    used_ = count;
    if (used_ == 0) {
        negative_ = false;
    }
}

int WideInt::sign() const
{
    if (used_ == 0) {
        return 0;
    }
    return negative_ ? -1 : 1;
}

WideInt WideInt::magnitude() const
{
    WideInt result{ *this };
    result.negative_ = false;
    return result;
}

WideInt WideInt::operator-() const
{
    WideInt result{ *this };
    result.negative_ = used_ != 0 && !negative_;
    return result;
}

int WideInt::compare_magnitude(const WideInt& a, const WideInt& b)
{
    if (a.used_ != b.used_) {
        return a.used_ < b.used_ ? -1 : 1;
    }
    for (int i{ a.used_ - 1 }; i >= 0; --i) {
        const std::uint64_t x{ a.limbs_[static_cast<std::size_t>(i)] };
        const std::uint64_t y{ b.limbs_[static_cast<std::size_t>(i)] };
        if (x != y) {
            return x < y ? -1 : 1;
        }
    }
    return 0;
}

WideInt WideInt::add_magnitude(const WideInt& a, const WideInt& b)
{
    const int count{ std::max(a.used_, b.used_) };
    WideInt result{};
    unsigned char carry{ 0 };
    for (int i{ 0 }; i < count; ++i) {
        const auto n{ static_cast<std::size_t>(i) };
        carry = _addcarry_u64(carry, a.limb(i), b.limb(i), &result.limbs_[n]);
    }
    if (carry == 0) {
        result.trim(count);
        return result;
    }
    if (count == LIMBS) {
        throw std::overflow_error("WideInt: the sum needs more than 1024 bits");
    }
    result.limbs_[static_cast<std::size_t>(count)] = 1u;
    result.used_ = count + 1;
    return result;
}

WideInt WideInt::subtract_magnitude(const WideInt& a, const WideInt& b)
{
    const int count{ a.used_ };
    WideInt result{};
    unsigned char borrow{ 0 };
    for (int i{ 0 }; i < count; ++i) {
        const auto n{ static_cast<std::size_t>(i) };
        borrow = _subborrow_u64(borrow, a.limbs_[n], b.limb(i), &result.limbs_[n]);
    }
    result.trim(count);
    return result;
}

WideInt WideInt::signed_sum(const WideInt& a, const WideInt& b)
{
    if (b.used_ == 0) {
        return a;
    }
    if (a.used_ == 0) {
        return b;
    }
    if (a.negative_ == b.negative_) {
        WideInt result{ add_magnitude(a, b) };
        result.negative_ = a.negative_;
        return result;
    }
    const int order{ compare_magnitude(a, b) };
    if (order == 0) {
        return WideInt{};
    }
    const bool a_larger{ order > 0 };
    WideInt result{ a_larger ? subtract_magnitude(a, b) : subtract_magnitude(b, a) };
    result.negative_ = result.used_ != 0 && (a_larger ? a.negative_ : b.negative_);
    return result;
}

WideInt WideInt::operator+(const WideInt& other) const
{
    return signed_sum(*this, other);
}

WideInt WideInt::operator-(const WideInt& other) const
{
    const WideInt negated_other{ -other };
    return signed_sum(*this, negated_other);
}

WideInt WideInt::operator*(const WideInt& other) const
{
    const int a_count{ used_ };
    const int b_count{ other.used_ };
    if (a_count == 0 || b_count == 0) {
        return WideInt{};
    }
    const int top{ a_count + b_count };
    if (top - 1 > LIMBS) {
        throw std::overflow_error("WideInt: the product needs more than 1024 bits");
    }
    // only the a_count + b_count limbs of the product are touched
    std::array<std::uint64_t, 2 * LIMBS> product; // NOLINT(cppcoreguidelines-pro-type-member-init): the first top limbs are set below
    std::fill_n(product.begin(), top, std::uint64_t{ 0 });
    for (int i{ 0 }; i < a_count; ++i) {
        const std::uint64_t a_limb{ limbs_[static_cast<std::size_t>(i)] };
        std::uint64_t carry{ 0u };
        for (int j{ 0 }; j < b_count; ++j) {
            const auto at{ static_cast<std::size_t>(i + j) };
            std::uint64_t high{ 0u };
            const std::uint64_t low{ _umul128(a_limb, other.limbs_[static_cast<std::size_t>(j)], &high) };
            // product[at] + low + carry, with the carries into high (never overflows: high <= 2^64 - 2)
            unsigned char c1{ _addcarry_u64(0, product[at], low, &product[at]) };
            unsigned char c2{ _addcarry_u64(0, product[at], carry, &product[at]) };
            carry = high + c1 + c2;
        }
        product[static_cast<std::size_t>(i + b_count)] = carry;
    }
    if (top > LIMBS && product[static_cast<std::size_t>(top - 1)] != 0u) {
        throw std::overflow_error("WideInt: the product needs more than 1024 bits");
    }
    const int kept{ std::min(top, LIMBS) };
    WideInt result{};
    std::copy_n(product.begin(), kept, result.limbs_.begin());
    result.trim(kept);
    result.negative_ = result.used_ != 0 && negative_ != other.negative_;
    return result;
}

int WideInt::compare(const WideInt& other) const
{
    const int own_sign{ sign() };
    const int other_sign{ other.sign() };
    if (own_sign != other_sign) {
        return own_sign < other_sign ? -1 : 1;
    }
    if (own_sign == 0) {
        return 0;
    }
    const int magnitude_order{ compare_magnitude(*this, other) };
    return own_sign > 0 ? magnitude_order : -magnitude_order;
}

bool WideInt::operator==(const WideInt& other) const
{
    return compare(other) == 0;
}

WideInt WideInt::shifted_left(const int bits) const
{
    if (bits < 0) {
        throw std::invalid_argument("WideInt::shifted_left: negative shift " + std::to_string(bits));
    }
    const int length{ bit_length() };
    if (length == 0) {
        return WideInt{};
    }
    if (length + bits > BITS) {
        throw std::overflow_error("WideInt::shifted_left: " + std::to_string(length) + " bits shifted by " + std::to_string(bits)
                                  + " need more than 1024 bits");
    }
    const int word_shift{ bits / LIMB_BITS };
    const int bit_shift{ bits % LIMB_BITS };
    const int bound{ std::min(LIMBS, used_ + word_shift + 1) };
    WideInt result{};
    std::fill_n(result.limbs_.begin(), bound, std::uint64_t{ 0 });
    for (int i{ 0 }; i < used_; ++i) {
        const std::uint64_t value{ limbs_[static_cast<std::size_t>(i)] };
        const auto low_index{ static_cast<std::size_t>(i + word_shift) };
        result.limbs_[low_index] |= value << bit_shift;
        if (bit_shift > 0) {
            const std::uint64_t high_part{ value >> (LIMB_BITS - bit_shift) };
            if (high_part != 0u) {
                result.limbs_[low_index + 1] |= high_part;
            }
        }
    }
    result.trim(bound);
    result.negative_ = negative_;
    return result;
}

int WideInt::bit_length() const
{
    if (used_ == 0) {
        return 0;
    }
    const std::uint64_t top_limb{ limbs_[static_cast<std::size_t>(used_ - 1)] };
    return (used_ - 1) * LIMB_BITS + static_cast<int>(std::bit_width(top_limb));
}

bool WideInt::bit(const int index) const
{
    if (index < 0 || index >= BITS) {
        throw std::out_of_range("WideInt::bit: index " + std::to_string(index) + " is outside 0 to 1023");
    }
    const int word{ index / LIMB_BITS };
    const int position{ index % LIMB_BITS };
    return ((limb(word) >> position) & 1u) != 0u;
}

bool WideInt::any_bit_below(const int index) const
{
    if (index > BITS) {
        throw std::out_of_range("WideInt::any_bit_below: index " + std::to_string(index) + " is above 1024");
    }
    if (index <= 0) {
        return false;
    }
    const int full_words{ index / LIMB_BITS };
    for (int i{ 0 }; i < full_words; ++i) {
        if (limb(i) != 0u) {
            return true;
        }
    }
    const int rest{ index % LIMB_BITS };
    if (rest == 0) {
        return false;
    }
    const std::uint64_t rest_mask{ (std::uint64_t{ 1 } << rest) - 1u };
    return (limb(full_words) & rest_mask) != 0u;
}

std::uint64_t WideInt::bits_from(const int low, const int count) const
{
    if (low < 0 || count < 1 || count > 64 || low + count > BITS) {
        throw std::out_of_range("WideInt::bits_from: bits " + std::to_string(low) + " + " + std::to_string(count) + " are outside 0 to 1023");
    }
    const int word{ low / LIMB_BITS };
    const int offset{ low % LIMB_BITS };
    std::uint64_t value{ limb(word) >> offset };
    if (offset > 0) {
        value |= limb(word + 1) << (LIMB_BITS - offset);
    }
    const std::uint64_t mask{ count == 64 ? ~std::uint64_t{ 0 } : (std::uint64_t{ 1 } << count) - 1u };
    return value & mask;
}

std::uint32_t WideInt::divide_magnitude(const std::uint32_t divisor, WideInt& quotient) const
{
    if (divisor == 0u) {
        throw std::invalid_argument("WideInt::divide_magnitude: divisor 0");
    }
    quotient = WideInt{};
    std::uint64_t remainder{ 0u };
    for (int i{ used_ - 1 }; i >= 0; --i) {
        // (remainder * 2^64 + limb) / divisor: remainder < divisor, so the quotient fits in 64 bits
        quotient.limbs_[static_cast<std::size_t>(i)] = _udiv128(remainder, limbs_[static_cast<std::size_t>(i)], divisor, &remainder);
    }
    quotient.trim(used_);
    return static_cast<std::uint32_t>(remainder);
}

std::string WideInt::to_string() const
{
    if (used_ == 0) {
        return "0";
    }
    std::string digits{};
    WideInt rest{ magnitude() };
    for (int count{ 0 }; count < DECIMAL_DIGITS_MAX && !rest.is_zero(); ++count) {
        WideInt next{};
        const std::uint32_t digit{ rest.divide_magnitude(10u, next) };
        digits.push_back(static_cast<char>('0' + digit));
        rest = next;
    }
    if (negative_) {
        digits.push_back('-');
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

} // namespace fluidx3d::experimental::wall_builder
