#pragma once
/// @file python_text.hpp
/// @brief Text exactly as Python writes it: repr() and float.hex() of a binary64, and json.dumps (indent=1, or compact).

#include <cstdint>
#include <string>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief Python's float formatting.
class PythonFloat {
public:
    /// @brief repr(x): the shortest round-trip digits; exponent form when the decimal exponent is below -4 or above 15.
    [[nodiscard]] static std::string repr(double x);

    /// @brief float.hex(x): "0x1.<13 hex digits>p<exp>", "0x0.0p+0" for zero, subnormals as "0x0.<digits>p-1022".
    [[nodiscard]] static std::string hex(double x);
};

/// @brief A streaming writer of json.dumps text: indent=1 (the provenance) or compact with ", " and ": " (the record).
/// The caller writes the keys in the order Python would (insertion order, or sorted for sort_keys).
class JsonWriter {
public:
    /// @brief A writer with one space per level (indent=1), or the compact form when indented is false.
    explicit JsonWriter(bool indented);

    void begin_object();
    void end_object();
    void begin_list();
    void end_list();
    /// @brief The key of the next value in an object.
    void key(const std::string& name);
    void value(const std::string& text);
    void value(const char* text);
    void value(double number);
    void value(std::int64_t number);
    void value(int number);
    void value(bool flag);

    /// @brief The text written so far.
    [[nodiscard]] const std::string& text() const { return out_; }

private:
    bool indented_{};
    std::string out_{};
    std::vector<int> counts_{}; ///< Items written per open container.
    bool after_key_{};

    void before_item();
    void close(char bracket);
    static std::string quoted(const std::string& text);
};

} // namespace fluidx3d::experimental::wall_builder
