#pragma once
/// @file wide_int.hpp
/// @brief A fixed-width signed integer for the exact determinants and exact plane values of the wall builder.

#include <array>
#include <cstdint>
#include <string>

namespace fluidx3d::experimental::wall_builder {

/// @brief A signed integer of 1024 bits in sign-magnitude form: 16 limbs of 64 bits, least significant limb first.
///
/// Only the limbs below the used count are written and read; construction and copies touch only those, so the cost
/// follows the size of the value (the volumes of the wall builder use a few limbs; measured worst case 492 bits on L24).
/// Zero-filling all 16 limbs of every temporary was most of the exact-volume cost (measured 2026-11-12). The limb products use
/// the MSVC x64 intrinsics (the builder is Windows only). Every result that needs more than 1024 bits throws
/// std::overflow_error: the value is never wrapped.
class WideInt {
public:
    static constexpr int LIMBS{ 16 };                  ///< Number of 64-bit limbs.
    static constexpr int LIMB_BITS{ 64 };              ///< Bits per limb.
    static constexpr int BITS{ LIMBS * LIMB_BITS };    ///< Total magnitude bits.

    /// @brief Makes zero. User-provided on purpose: with "= default", WideInt{} would zero all 16 limbs first.
    WideInt() {} // NOLINT(modernize-use-equals-default): see the brief

    /// @brief Copies the used limbs only.
    WideInt(const WideInt& other);

    /// @brief Copies the used limbs only.
    WideInt& operator=(const WideInt& other);

    /// @brief Makes the value of a signed 64-bit integer.
    /// @param value any value, including INT64_MIN
    explicit WideInt(std::int64_t value);

    /// @brief Makes value * 2^shift.
    /// @param value any signed 64-bit integer
    /// @param shift the power of two, 0 or more
    /// @throws std::invalid_argument when shift is negative
    /// @throws std::overflow_error when the result needs more than BITS bits
    [[nodiscard]] static WideInt from_shifted(std::int64_t value, int shift);

    /// @brief Makes a positive value from limbs, least significant first.
    /// @param limbs the limbs
    /// @param count how many, 0 to LIMBS
    /// @throws std::invalid_argument when count is outside 0 to LIMBS
    [[nodiscard]] static WideInt from_limbs(const std::uint64_t* limbs, int count);

    /// @brief Tells if the value is zero.
    [[nodiscard]] bool is_zero() const { return used_ == 0; }

    /// @brief The sign of the value. @return -1, 0 or +1
    [[nodiscard]] int sign() const;

    /// @brief The absolute value.
    [[nodiscard]] WideInt magnitude() const;

    /// @brief The value with the opposite sign.
    [[nodiscard]] WideInt operator-() const;

    /// @brief The sum. @throws std::overflow_error when the sum needs more than BITS bits
    [[nodiscard]] WideInt operator+(const WideInt& other) const;

    /// @brief The difference. @throws std::overflow_error when the difference needs more than BITS bits
    [[nodiscard]] WideInt operator-(const WideInt& other) const;

    /// @brief The product. @throws std::overflow_error when the product needs more than BITS bits
    [[nodiscard]] WideInt operator*(const WideInt& other) const;

    /// @brief Signed three-way comparison. @return -1 when this < other, 0 when equal, +1 when this > other
    [[nodiscard]] int compare(const WideInt& other) const;

    /// @brief Tells if both values are equal.
    [[nodiscard]] bool operator==(const WideInt& other) const;

    /// @brief The value times 2^bits.
    /// @throws std::invalid_argument when bits is negative
    /// @throws std::overflow_error when a set bit leaves the BITS bits
    [[nodiscard]] WideInt shifted_left(int bits) const;

    /// @brief The number of significant bits of the magnitude. @return 0 for zero, else the highest set bit index plus 1
    [[nodiscard]] int bit_length() const;

    /// @brief Tells if one bit of the magnitude is set. @param index 0 = least significant, 0 to BITS - 1
    [[nodiscard]] bool bit(int index) const;

    /// @brief Tells if any bit of the magnitude below an index is set. @param index bits 0 to index - 1; 0 or less: none
    [[nodiscard]] bool any_bit_below(int index) const;

    /// @brief Bits low to low + count - 1 of the magnitude, bit low at position 0. @param count 1 to 64
    [[nodiscard]] std::uint64_t bits_from(int low, int count) const;

    /// @brief Divides the magnitude by a small positive divisor.
    /// @param divisor the divisor, 1 or more
    /// @param quotient receives floor(|this| / divisor), positive
    /// @return the remainder |this| mod divisor
    /// @throws std::invalid_argument when divisor is 0
    [[nodiscard]] std::uint32_t divide_magnitude(std::uint32_t divisor, WideInt& quotient) const;

    /// @brief The decimal text of the value, for messages and tests.
    [[nodiscard]] std::string to_string() const;

private:
    bool negative_{};                           ///< True for a negative nonzero value; zero is never negative.
    int used_{};                                ///< Limbs up to the highest nonzero one; the limbs above are zero.
    std::array<std::uint64_t, LIMBS> limbs_; ///< NOLINT(cppcoreguidelines-pro-type-member-init): only limbs below used_ hold values, see the class comment

    /// @brief Limb n of the magnitude, zero at and above the used count.
    [[nodiscard]] std::uint64_t limb(int n) const { return n < used_ ? limbs_[static_cast<std::size_t>(n)] : 0u; }

    /// @brief Sets used_ from the limbs below a bound (the limbs at and above it are zero).
    void trim(int bound);

    /// @brief Compares the magnitudes. @return -1, 0 or +1 for |a| <, =, > |b|
    [[nodiscard]] static int compare_magnitude(const WideInt& a, const WideInt& b);

    /// @brief |a| + |b|, positive. @throws std::overflow_error on a carry out of the top limb
    [[nodiscard]] static WideInt add_magnitude(const WideInt& a, const WideInt& b);

    /// @brief |a| - |b|, positive; the caller makes sure |a| >= |b|.
    [[nodiscard]] static WideInt subtract_magnitude(const WideInt& a, const WideInt& b);

    /// @brief The sum of a and b for any signs. @throws std::overflow_error when the sum needs more than BITS bits
    [[nodiscard]] static WideInt signed_sum(const WideInt& a, const WideInt& b);
};

} // namespace fluidx3d::experimental::wall_builder
