#pragma once
/// @file sha256.hpp
/// @brief SHA-256 by the Windows CNG provider (bcrypt), as lower-case hexadecimal text.

#include <cstddef>
#include <string>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief SHA-256 digests.
class Sha256 {
public:
    /// @brief The digest of a byte string. @throws std::runtime_error when CNG fails
    [[nodiscard]] static std::string of(const std::string& bytes);

    /// @brief The digest of a byte buffer. @throws std::runtime_error when CNG fails
    [[nodiscard]] static std::string of(const std::vector<char>& bytes);

private:
    [[nodiscard]] static std::string digest(const unsigned char* data, unsigned long size);
};

/// @brief An incremental SHA-256: the digest of all bytes given to update(), in order.
class Sha256Stream {
public:
    /// @brief Opens the CNG provider and the hash. @throws std::runtime_error when CNG fails
    Sha256Stream();
    ~Sha256Stream();
    Sha256Stream(const Sha256Stream&) = delete;
    Sha256Stream& operator=(const Sha256Stream&) = delete;

    /// @brief Adds bytes. @throws std::runtime_error when CNG fails
    void update(const char* data, std::size_t size);

    /// @brief The digest of everything added; the stream cannot be used after it. @throws std::runtime_error when CNG fails
    [[nodiscard]] std::string finish();

private:
    void* algorithm_{}; ///< BCRYPT_ALG_HANDLE.
    void* hash_{};      ///< BCRYPT_HASH_HANDLE.
};

} // namespace fluidx3d::experimental::wall_builder
