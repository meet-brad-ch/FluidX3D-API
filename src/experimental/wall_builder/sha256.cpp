/// @file sha256.cpp
/// @brief SHA-256 by the Windows CNG provider (bcrypt).

#include "sha256.hpp"

#define WIN32_LEAN_AND_MEAN // NOLINT(cppcoreguidelines-macro-usage): the documented switch that trims windows.h
#include <windows.h>

#include <bcrypt.h>

#include <algorithm>
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

// ------------------------------------------------------------------------------------------------ Sha256Stream

Sha256Stream::Sha256Stream()
{
    BCRYPT_ALG_HANDLE algorithm{ nullptr };
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) {
        throw std::runtime_error("BCryptOpenAlgorithmProvider(SHA256) failed");
    }
    algorithm_ = algorithm;
    BCRYPT_HASH_HANDLE hash{ nullptr };
    if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        throw std::runtime_error("BCryptCreateHash failed");
    }
    hash_ = hash;
}

Sha256Stream::~Sha256Stream()
{
    if (hash_ != nullptr) {
        BCryptDestroyHash(static_cast<BCRYPT_HASH_HANDLE>(hash_));
    }
    BCryptCloseAlgorithmProvider(static_cast<BCRYPT_ALG_HANDLE>(algorithm_), 0);
}

void Sha256Stream::update(const char* const data, const std::size_t size)
{
    if (hash_ == nullptr) {
        throw std::logic_error("Sha256Stream: update after finish");
    }
    // BCryptHashData takes a non-const input pointer and an unsigned long size; it does not change the input
    std::size_t done{ 0 };
    while (done < size) {
        const std::size_t part{ std::min<std::size_t>(size - done, std::size_t{ 1 } << 30) };
        auto* const bytes{ reinterpret_cast<unsigned char*>(const_cast<char*>(data + done)) };
        if (BCryptHashData(static_cast<BCRYPT_HASH_HANDLE>(hash_), bytes, static_cast<unsigned long>(part), 0) < 0) {
            throw std::runtime_error("BCryptHashData failed");
        }
        done += part;
    }
}

std::string Sha256Stream::finish()
{
    if (hash_ == nullptr) {
        throw std::logic_error("Sha256Stream: finish twice");
    }
    std::array<unsigned char, 32> digest{};
    const NTSTATUS status{ BCryptFinishHash(static_cast<BCRYPT_HASH_HANDLE>(hash_), digest.data(), 32, 0) };
    BCryptDestroyHash(static_cast<BCRYPT_HASH_HANDLE>(hash_));
    hash_ = nullptr;
    if (status < 0) {
        throw std::runtime_error("BCryptFinishHash failed");
    }
    return hex_text(digest);
}

} // namespace fluidx3d::experimental::wall_builder
