/// @file python_text.cpp
/// @brief Text exactly as Python writes it: repr() and float.hex() of a binary64, and json.dumps.

#include "python_text.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <system_error>

namespace fluidx3d::experimental::wall_builder {

std::string PythonFloat::repr(const double x)
{
    if (std::isnan(x)) {
        return "nan";
    }
    if (std::isinf(x)) {
        return x > 0 ? "inf" : "-inf";
    }
    if (x == 0.0) {
        return std::signbit(x) ? "-0.0" : "0.0";
    }
    // the shortest round-trip digits in scientific form, e.g. "-1.2345e-07"
    std::array<char, 64> buffer{};
    const std::to_chars_result written{ std::to_chars(buffer.data(), buffer.data() + buffer.size(), x, std::chars_format::scientific) };
    if (written.ec != std::errc{}) {
        throw std::runtime_error("PythonFloat::repr: to_chars failed");
    }
    const std::string scientific(buffer.data(), written.ptr);
    const bool negative{ scientific[0] == '-' };
    const std::size_t e_at{ scientific.find('e') };
    std::string digits{};
    for (std::size_t n{ negative ? 1u : 0u }; n < e_at; ++n) {
        if (scientific[n] != '.') {
            digits.push_back(scientific[n]);
        }
    }
    const int exponent{ std::stoi(scientific.substr(e_at + 1)) };
    const int point{ exponent + 1 }; // x = 0.d1d2... * 10^point
    std::string out{ negative ? "-" : "" };
    if (point > -4 && point <= 16) {
        // fixed notation, with ".0" for an integer
        if (point <= 0) {
            out += "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
            return out;
        }
        const auto whole{ static_cast<std::size_t>(point) };
        if (digits.size() <= whole) {
            out += digits + std::string(whole - digits.size(), '0') + ".0";
            return out;
        }
        out += digits.substr(0, whole) + "." + digits.substr(whole);
        return out;
    }
    out += digits.substr(0, 1);
    if (digits.size() > 1) {
        out += "." + digits.substr(1);
    }
    const int shown{ point - 1 };
    const int magnitude{ shown < 0 ? -shown : shown };
    out += shown < 0 ? "e-" : "e+";
    out += (magnitude < 10 ? "0" : "") + std::to_string(magnitude);
    return out;
}

std::string PythonFloat::hex(const double x)
{
    if (std::isnan(x)) {
        return "nan";
    }
    if (std::isinf(x)) {
        return x > 0 ? "inf" : "-inf";
    }
    const std::uint64_t bits{ std::bit_cast<std::uint64_t>(x) };
    const bool negative{ (bits >> 63) != 0u };
    const auto biased{ static_cast<int>((bits >> 52) & 0x7FFu) };
    const std::uint64_t fraction{ bits & ((std::uint64_t{ 1 } << 52) - 1u) };
    std::string out{ negative ? "-" : "" };
    if (biased == 0 && fraction == 0u) {
        return out + "0x0.0p+0";
    }
    const char* const hex_digits{ "0123456789abcdef" };
    std::string mantissa{};
    for (int shift{ 48 }; shift >= 0; shift -= 4) {
        mantissa.push_back(hex_digits[(fraction >> shift) & 0xFu]);
    }
    const int exponent{ biased == 0 ? -1022 : biased - 1023 };
    out += biased == 0 ? "0x0." : "0x1.";
    out += mantissa;
    out += exponent < 0 ? "p-" : "p+";
    out += std::to_string(exponent < 0 ? -exponent : exponent);
    return out;
}

JsonWriter::JsonWriter(const bool indented)
    : indented_{ indented }
{
}

void JsonWriter::before_item()
{
    if (after_key_) {
        after_key_ = false;
        return;
    }
    if (counts_.empty()) {
        return;
    }
    if (counts_.back() > 0) {
        out_ += ",";
    }
    if (indented_) {
        out_ += "\n" + std::string(counts_.size(), ' ');
    } else if (counts_.back() > 0) {
        out_ += " ";
    }
    counts_.back() += 1;
}

void JsonWriter::close(const char bracket)
{
    const int items{ counts_.back() };
    counts_.pop_back();
    if (indented_ && items > 0) {
        out_ += "\n" + std::string(counts_.size(), ' ');
    }
    out_.push_back(bracket);
}

std::string JsonWriter::quoted(const std::string& text)
{
    std::string out{ "\"" };
    for (const char ch : text) {
        const auto code{ static_cast<unsigned char>(ch) };
        if (ch == '"' || ch == '\\') {
            out.push_back('\\');
            out.push_back(ch);
        } else if (ch == '\n') {
            out += "\\n";
        } else if (code < 0x20u || code > 0x7Eu) {
            const char* const hex_digits{ "0123456789abcdef" };
            out += "\\u00";
            out.push_back(hex_digits[code >> 4]);
            out.push_back(hex_digits[code & 0xFu]);
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('"');
    return out;
}

void JsonWriter::begin_object()
{
    before_item();
    out_.push_back('{');
    counts_.push_back(0);
}

void JsonWriter::end_object()
{
    close('}');
}

void JsonWriter::begin_list()
{
    before_item();
    out_.push_back('[');
    counts_.push_back(0);
}

void JsonWriter::end_list()
{
    close(']');
}

void JsonWriter::key(const std::string& name)
{
    before_item();
    out_ += quoted(name) + ": ";
    after_key_ = true;
}

void JsonWriter::value(const std::string& text)
{
    before_item();
    out_ += quoted(text);
}

void JsonWriter::value(const char* const text)
{
    value(std::string{ text });
}

void JsonWriter::value(const double number)
{
    before_item();
    // json.dumps writes NaN and Infinity for the special values (allow_nan=True)
    if (std::isnan(number)) {
        out_ += "NaN";
        return;
    }
    if (std::isinf(number)) {
        out_ += number > 0 ? "Infinity" : "-Infinity";
        return;
    }
    out_ += PythonFloat::repr(number);
}

void JsonWriter::value(const std::int64_t number)
{
    before_item();
    out_ += std::to_string(number);
}

void JsonWriter::value(const int number)
{
    value(static_cast<std::int64_t>(number));
}

void JsonWriter::value(const bool flag)
{
    before_item();
    out_ += flag ? "true" : "false";
}

} // namespace fluidx3d::experimental::wall_builder
