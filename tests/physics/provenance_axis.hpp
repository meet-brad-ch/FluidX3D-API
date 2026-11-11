// Task 23 (#442): a bounded, correctly tokenized reader of the TOP-LEVEL members of the container provenance (a JSON
// object), and the reference-axis / u0 resolution built on it. Header-only so that the GPU-free control executable
// (provenance_axis_test.cpp) tests exactly the code the protocol fixture uses. No kernel, host-library or physics code.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace provenance_axis {

/// @brief One top-level member value: its JSON kind and, for strings, the decoded text, for numbers the raw token.
struct Value {
    enum class Kind { string, number, literal, object, array };
    Kind kind{ Kind::literal };
    std::string text{};   ///< decoded string, raw number token, or the literal (true/false/null); empty for object/array
};

/// @brief Reads the top-level members of one JSON object. JSON whitespace (space, tab, CR, LF) is allowed everywhere
/// between tokens; strings are decoded with all their escapes, \uXXXX included (surrogate pairs validated and combined,
/// lone surrogates rejected, output UTF-8; #445), so an escaped key or value is compared by its decoded text; nested
/// objects and arrays are skipped as whole values (their
/// keys are never seen as top-level keys). Every key and its count are kept, so duplicates are detectable.
class TopLevelReader {
public:
    /// @brief Parses `text` as one JSON object followed only by whitespace.
    /// @param text The provenance text.
    /// @throws std::runtime_error On any syntax error.
    explicit TopLevelReader(const std::string& text) : s_(text) {
        skip_ws();
        expect('{');
        skip_ws();
        if(peek()=='}') {
            pos_++;
        } else {
            for(std::size_t member=0u; member<s_.size(); member++) {   // bounded: at most one member per character
                skip_ws();
                const std::string key{ read_string() };
                skip_ws();
                expect(':');
                skip_ws();
                const Value value{ read_value(0u) };
                members_.push_back({ key, value });
                skip_ws();
                const char next{ take() };
                if(next=='}') break;
                if(next!=',') fail("expected ',' or '}' after a member");
            }
        }
        skip_ws();
        if(pos_!=s_.size()) fail("trailing characters after the top-level object");
    }

    /// @brief Number of top-level members with this key.
    std::size_t count(const std::string& key) const {
        std::size_t n{ 0u };
        for(const auto& m : members_) {
            if(m.first==key) n++;
        }
        return n;
    }

    /// @brief The value of the single top-level member `key`.
    /// @throws std::runtime_error When the key is absent or present more than once.
    const Value& at(const std::string& key) const {
        if(count(key)!=1u) throw std::runtime_error("provenance key '"+key+"' is not present exactly once");
        for(const auto& m : members_) {
            if(m.first==key) return m.second;
        }
        throw std::runtime_error("unreachable");
    }

private:
    static constexpr std::size_t max_depth{ 64u };
    const std::string& s_;
    std::size_t pos_{ 0u };
    std::vector<std::pair<std::string, Value>> members_{};

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("provenance JSON: "+what+" at offset "+std::to_string(pos_));
    }
    char peek() const {
        if(pos_>=s_.size()) fail("unexpected end");
        return s_[pos_];
    }
    char take() {
        const char c{ peek() };
        pos_++;
        return c;
    }
    void expect(const char c) {
        if(take()!=c) fail(std::string("expected '")+c+"'");
    }
    void skip_ws() {
        while(pos_<s_.size() && (s_[pos_]==' ' || s_[pos_]=='\t' || s_[pos_]=='\n' || s_[pos_]=='\r')) pos_++;
    }
    std::string read_string() {
        expect('"');
        std::string out{};
        while(true) {
            const char c{ take() };
            if(c=='"') return out;
            if(static_cast<unsigned char>(c)<0x20u) fail("control character in a string");
            if(c!='\\') {
                out.push_back(c);
                continue;
            }
            const char e{ take() };
            if(e=='"' || e=='\\' || e=='/') {
                out.push_back(e);
            } else if(e=='b') {
                out.push_back('\b');
            } else if(e=='f') {
                out.push_back('\f');
            } else if(e=='n') {
                out.push_back('\n');
            } else if(e=='r') {
                out.push_back('\r');
            } else if(e=='t') {
                out.push_back('\t');
            } else if(e=='u') {
                // #445: decode, never substitute. A high surrogate must be followed by an escaped low surrogate; a lone
                // surrogate of either kind is rejected. The code point is appended as UTF-8.
                std::uint32_t code{ read_hex4() };
                if(code>=0xD800u && code<=0xDBFFu) {
                    if(take()!='\\' || take()!='u') fail("high surrogate not followed by a \\u escape");
                    const std::uint32_t low{ read_hex4() };
                    if(low<0xDC00u || low>0xDFFFu) fail("high surrogate not followed by a low surrogate");
                    code = 0x10000u+((code-0xD800u)<<10u)+(low-0xDC00u);
                } else if(code>=0xDC00u && code<=0xDFFFu) {
                    fail("lone low surrogate");
                }
                append_utf8(out, code);
            } else {
                fail("bad escape");
            }
        }
    }
    std::uint32_t read_hex4() {
        std::uint32_t value{ 0u };
        for(int k=0; k<4; k++) {
            const char h{ take() };
            std::uint32_t digit{ 0u };
            if(h>='0' && h<='9') {
                digit = static_cast<std::uint32_t>(h-'0');
            } else if(h>='a' && h<='f') {
                digit = static_cast<std::uint32_t>(h-'a'+10);
            } else if(h>='A' && h<='F') {
                digit = static_cast<std::uint32_t>(h-'A'+10);
            } else {
                fail("bad \\u escape digit");
            }
            value = value*16u+digit;
        }
        return value;
    }
    static void append_utf8(std::string& out, const std::uint32_t code) {
        if(code<0x80u) {
            out.push_back(static_cast<char>(code));
        } else if(code<0x800u) {
            out.push_back(static_cast<char>(0xC0u|(code>>6u)));
            out.push_back(static_cast<char>(0x80u|(code&0x3Fu)));
        } else if(code<0x10000u) {
            out.push_back(static_cast<char>(0xE0u|(code>>12u)));
            out.push_back(static_cast<char>(0x80u|((code>>6u)&0x3Fu)));
            out.push_back(static_cast<char>(0x80u|(code&0x3Fu)));
        } else {
            out.push_back(static_cast<char>(0xF0u|(code>>18u)));
            out.push_back(static_cast<char>(0x80u|((code>>12u)&0x3Fu)));
            out.push_back(static_cast<char>(0x80u|((code>>6u)&0x3Fu)));
            out.push_back(static_cast<char>(0x80u|(code&0x3Fu)));
        }
    }
    std::string read_number() {
        const std::size_t start{ pos_ };
        if(peek()=='-') pos_++;
        if(peek()=='0') {
            pos_++;
        } else if(peek()>='1' && peek()<='9') {
            while(pos_<s_.size() && s_[pos_]>='0' && s_[pos_]<='9') pos_++;
        } else {
            fail("bad number");
        }
        if(pos_<s_.size() && s_[pos_]=='.') {
            pos_++;
            if(!(pos_<s_.size() && s_[pos_]>='0' && s_[pos_]<='9')) fail("bad fraction");
            while(pos_<s_.size() && s_[pos_]>='0' && s_[pos_]<='9') pos_++;
        }
        if(pos_<s_.size() && (s_[pos_]=='e' || s_[pos_]=='E')) {
            pos_++;
            if(pos_<s_.size() && (s_[pos_]=='+' || s_[pos_]=='-')) pos_++;
            if(!(pos_<s_.size() && s_[pos_]>='0' && s_[pos_]<='9')) fail("bad exponent");
            while(pos_<s_.size() && s_[pos_]>='0' && s_[pos_]<='9') pos_++;
        }
        return s_.substr(start, pos_-start);
    }
    Value read_value(const std::size_t depth) {
        if(depth>max_depth) fail("nesting too deep");
        const char c{ peek() };
        Value v{};
        if(c=='"') {
            v.kind = Value::Kind::string;
            v.text = read_string();
        } else if(c=='{' || c=='[') {
            v.kind = c=='{' ? Value::Kind::object : Value::Kind::array;
            skip_container(depth);
        } else if(c=='-' || (c>='0' && c<='9')) {
            v.kind = Value::Kind::number;
            v.text = read_number();
        } else {
            for(const char* literal : { "true", "false", "null" }) {
                const std::string lit{ literal };
                if(s_.compare(pos_, lit.size(), lit)==0) {
                    pos_ += lit.size();
                    v.kind = Value::Kind::literal;
                    v.text = lit;
                    return v;
                }
            }
            fail("bad value");
        }
        return v;
    }
    void skip_container(const std::size_t depth) {
        const char open{ take() };
        const char close{ open=='{' ? '}' : ']' };
        skip_ws();
        if(peek()==close) {
            pos_++;
            return;
        }
        for(std::size_t item=0u; item<s_.size(); item++) {   // bounded
            skip_ws();
            if(open=='{') {
                read_string();
                skip_ws();
                expect(':');
                skip_ws();
            }
            read_value(depth+1u);
            skip_ws();
            const char next{ take() };
            if(next==close) return;
            if(next!=',') fail("expected ',' or a closing bracket");
        }
        fail("container too long");
    }
};

/// @brief The reference axis from the provenance and the VW_AXIS binding (task 23 contract).
/// Absent top-level key = z (legacy). Accepted: the strings "y" and "z". Rejected: a duplicated top-level key, a
/// non-string value, any other string, and a non-empty `vw_axis` that differs from the resolved letter.
/// @param provenance The container provenance text (a JSON object).
/// @param vw_axis The VW_AXIS environment value ("" when unset).
/// @return 1 for y, 2 for z (lexicographic component index).
/// @throws std::runtime_error On any rejected input or a JSON syntax error.
inline std::uint32_t resolve_axis(const std::string& provenance, const std::string& vw_axis) {
    const TopLevelReader reader(provenance);
    std::string letter{ "z" };
    const std::size_t n{ reader.count("reference_axis") };
    if(n>1u) throw std::runtime_error("provenance reference_axis appears more than once at the top level");
    if(n==1u) {
        const Value& v{ reader.at("reference_axis") };
        if(v.kind!=Value::Kind::string) throw std::runtime_error("provenance reference_axis is not a string");
        letter = v.text;
    }
    std::uint32_t axis{ 0u };
    if(letter=="y") {
        axis = 1u;
    } else if(letter=="z") {
        axis = 2u;
    } else {
        throw std::runtime_error("unsupported reference_axis '"+letter+"' (task 23 accepts y and z)");
    }
    if(!vw_axis.empty() && vw_axis!=letter) throw std::runtime_error("container reference_axis '"+letter+"' differs from VW_AXIS '"+vw_axis+"'");
    return axis;
}

/// @brief The signed u0 from the top-level number u0_lattice (absent = 0, as before).
/// @param provenance The container provenance text.
/// @return u0 (finite).
/// @throws std::runtime_error On a duplicated key, a non-number value or a non-finite number.
inline double resolve_u0(const std::string& provenance) {
    const TopLevelReader reader(provenance);
    const std::size_t n{ reader.count("u0_lattice") };
    if(n>1u) throw std::runtime_error("provenance u0_lattice appears more than once at the top level");
    if(n==0u) return 0.0;
    const Value& v{ reader.at("u0_lattice") };
    if(v.kind!=Value::Kind::number) throw std::runtime_error("provenance u0_lattice is not a number");
    double u0{ 0.0 };
    try {
        u0 = std::stod(v.text);
    } catch(const std::out_of_range&) {
        throw std::runtime_error("provenance u0_lattice is not finite (out of the double range)");
    }
    if(!std::isfinite(u0)) throw std::runtime_error("provenance u0_lattice is not finite");
    return u0;
}

}  // namespace provenance_axis
