#pragma once
/// @file sha256.hpp
/// @brief SHA-256 by the Windows CNG provider (bcrypt), as lower-case hexadecimal text.

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

    /// @brief The digest of the concatenation of chunks, without joining them. @throws std::runtime_error when CNG fails
    [[nodiscard]] static std::string of(const std::vector<std::vector<char>>& chunks);

private:
    [[nodiscard]] static std::string digest(const unsigned char* data, unsigned long size);
};

} // namespace fluidx3d::experimental::wall_builder
