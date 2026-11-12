/// @file sha256.cpp
/// @brief SHA-256 by the Windows CNG provider (bcrypt).

#include "sha256.hpp"

#define WIN32_LEAN_AND_MEAN // NOLINT(cppcoreguidelines-macro-usage): the documented switch that trims windows.h
#include <windows.h>

#include <bcrypt.h>

#include <array>
#include <stdexcept>

namespace fluidx3d::experimental::wall_builder {

namespace {

/// @brief A CNG algorithm handle, closed on destruction.
class Algorithm {
public:
    Algorithm()
    {
        const NTSTATUS status{ BCryptOpenAlgorithmProvider(&handle, BCRYPT_SHA256_ALGORITHM, nullptr, 0) };
        if (status < 0) {
            throw std::runtime_error("BCryptOpenAlgorithmProvider(SHA256) failed");
        }
    }
    ~Algorithm() { BCryptCloseAlgorithmProvider(handle, 0); }
    Algorithm(const Algorithm&) = delete;
    Algorithm& operator=(const Algorithm&) = delete;

    BCRYPT_ALG_HANDLE handle{ nullptr }; ///< The provider.
};

/// @brief Lower-case hexadecimal of a digest.
std::string hex_text(const std::array<unsigned char, 32>& hash)
{
    const char* const hex_digits{ "0123456789abcdef" };
    std::string out{};
    for (const unsigned char byte : hash) {
        out.push_back(hex_digits[byte >> 4]);
        out.push_back(hex_digits[byte & 0xFu]);
    }
    return out;
}

} // namespace

std::string Sha256::of(const std::vector<std::vector<char>>& chunks)
{
    const Algorithm algorithm{};
    BCRYPT_HASH_HANDLE handle{ nullptr };
    if (BCryptCreateHash(algorithm.handle, &handle, nullptr, 0, nullptr, 0, 0) < 0) {
        throw std::runtime_error("BCryptCreateHash failed");
    }
    bool ok{ true };
    for (const std::vector<char>& chunk : chunks) {
        // BCryptHashData takes a non-const input pointer but does not change the input
        auto* const data{ reinterpret_cast<unsigned char*>(const_cast<char*>(chunk.data())) };
        ok = ok && BCryptHashData(handle, data, static_cast<unsigned long>(chunk.size()), 0) >= 0;
    }
    std::array<unsigned char, 32> hash{};
    ok = ok && BCryptFinishHash(handle, hash.data(), 32, 0) >= 0;
    BCryptDestroyHash(handle);
    if (!ok) {
        throw std::runtime_error("BCryptHashData failed");
    }
    return hex_text(hash);
}

std::string Sha256::digest(const unsigned char* const data, const unsigned long size)
{
    const Algorithm algorithm{};
    std::array<unsigned char, 32> hash{};
    // BCryptHash takes a non-const input pointer but does not change the input
    const NTSTATUS status{ BCryptHash(algorithm.handle, nullptr, 0, const_cast<unsigned char*>(data), size, hash.data(), 32) };
    if (status < 0) {
        throw std::runtime_error("BCryptHash failed");
    }
    return hex_text(hash);
}

std::string Sha256::of(const std::string& bytes)
{
    return digest(reinterpret_cast<const unsigned char*>(bytes.data()), static_cast<unsigned long>(bytes.size()));
}

std::string Sha256::of(const std::vector<char>& bytes)
{
    return digest(reinterpret_cast<const unsigned char*>(bytes.data()), static_cast<unsigned long>(bytes.size()));
}

} // namespace fluidx3d::experimental::wall_builder
