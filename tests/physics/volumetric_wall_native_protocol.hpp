#pragma once
// Task 19: the protocol run mode of the native volumetric wall fixture (VW_MODE=protocol). Every step: host diagnostics
// of all stages (raw fi decoded with the address algebra, the R capture), scalar inventories, positivity and
// finiteness by bit pattern, the native clamp count, the storage increment and its half-ULP bound, and a rolling
// SHA-256 of the raw fi; raw checkpoints (fi, pending, t) at the registered steps and at the first failure. No
// physical verdict here: the series and checkpoints are analysed by the research tree.
#include "experimental/volumetric_wall_native.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <string>
#include <vector>

namespace native_protocol {

/// @brief Compact SHA-256 (FIPS 180-4) for the rolling raw-fi hash; verified against the Python hashlib result by the
/// research analysis (the checkpoint files are hashed on both sides).
class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        state_ = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
        buffer_.clear();
        total_ = 0u;
    }
    void update(const void* data, const std::size_t size) {
        const std::uint8_t* bytes{ static_cast<const std::uint8_t*>(data) };
        total_ += size;
        buffer_.insert(buffer_.end(), bytes, bytes+size);
        std::size_t offset{ 0u };
        while(buffer_.size()-offset>=64u) {
            block(buffer_.data()+offset);
            offset += 64u;
        }
        buffer_.erase(buffer_.begin(), buffer_.begin()+static_cast<std::ptrdiff_t>(offset));
    }
    std::string hex() {   // finalizes a copy; the running state stays usable
        Sha256 copy{ *this };
        copy.finish();
        static const char* digits{ "0123456789abcdef" };
        std::string out{};
        for(const std::uint32_t word : copy.state_) {
            for(int shift=28; shift>=0; shift-=4) {
                out += digits[(word>>shift)&0xFu];
            }
        }
        return out;
    }
private:
    static std::uint32_t rotr(const std::uint32_t x, const std::uint32_t n) { return (x>>n)|(x<<(32u-n)); }
    void block(const std::uint8_t* p) {
        static const std::uint32_t k[64]{
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u };
        std::uint32_t w[64]{};
        for(int i=0; i<16; i++) {
            w[i] = (static_cast<std::uint32_t>(p[4*i])<<24)|(static_cast<std::uint32_t>(p[4*i+1])<<16)|(static_cast<std::uint32_t>(p[4*i+2])<<8)|static_cast<std::uint32_t>(p[4*i+3]);
        }
        for(int i=16; i<64; i++) {
            const std::uint32_t s0{ rotr(w[i-15], 7u)^rotr(w[i-15], 18u)^(w[i-15]>>3u) };
            const std::uint32_t s1{ rotr(w[i-2], 17u)^rotr(w[i-2], 19u)^(w[i-2]>>10u) };
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        std::uint32_t a{ state_[0] }, b{ state_[1] }, c{ state_[2] }, d{ state_[3] }, e{ state_[4] }, f{ state_[5] }, g{ state_[6] }, h{ state_[7] };
        for(int i=0; i<64; i++) {
            const std::uint32_t S1{ rotr(e, 6u)^rotr(e, 11u)^rotr(e, 25u) };
            const std::uint32_t ch{ (e&f)^((~e)&g) };
            const std::uint32_t t1{ h+S1+ch+k[i]+w[i] };
            const std::uint32_t S0{ rotr(a, 2u)^rotr(a, 13u)^rotr(a, 22u) };
            const std::uint32_t maj{ (a&b)^(a&c)^(b&c) };
            const std::uint32_t t2{ S0+maj };
            h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d; state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }
    void finish() {
        const std::uint64_t bits{ total_*8u };
        std::vector<std::uint8_t> pad{ buffer_ };
        pad.push_back(0x80u);
        while(pad.size()%64u!=56u) { pad.push_back(0u); }
        for(int shift=56; shift>=0; shift-=8) { pad.push_back(static_cast<std::uint8_t>((bits>>shift)&0xFFu)); }
        for(std::size_t offset=0u; offset<pad.size(); offset+=64u) { block(pad.data()+offset); }
    }
    std::array<std::uint32_t, 8> state_{};
    std::vector<std::uint8_t> buffer_{};
    std::uint64_t total_{};
};

/// @brief The per-step host series, schema v5 (v4 = the first 73 columns, unchanged in order and meaning; v5 appends 15 columns, task 21). Row s (s >= 1) describes the step from level s-1 to level s: inventories,
/// minima and extrema AFTER the step (level s), the impulse of the step, the native collision defect v^T(p - f) with the
/// ordinary post from the stored-next slots and the R post from the capture, the static coefficient terms e_cell.p and
/// e_out.Q (mass; momentum forms with c weights), the shell residual 1^T(Q - incoming), the measured R storage increment
/// and its conservative half-ULP bound (g = f - w, |c| weights for momentum), the identity residuals (mass: delta(fluid +
/// pending) - (defect + coeff_cell + coeff_out + shell + storage); momentum: delta(P_fluid + P_pending) - J - (defect_mom +
/// coeff_cell_mom + coeff_out_mom + storage_mom), pending momentum along -c_r), the amplitude of the registered (1,1)
/// mode A = sum V u_z phi / sum V phi^2, max_transverse = max |u_xy|, min_post_all over the ordinary actual post and the
/// R post, min_pending_ratio = min pending/capacity over rows with capacity > 1e-14 vi (CF10 definition, capacity = W
/// row sum), zero_capacity_rows the count excluded. Row 0 holds the level-0 inventories/extrema/minima and 0 (or NaN
/// written as 0) for the step-specific terms; its min_post_all, min_pending_ratio and impulse are 0 by convention.
struct SeriesRow {
    static const std::vector<std::string>& names() {
        static const std::vector<std::string> list{
            "step", "t_after", "mass_fluid", "mass_pending", "momx_fluid", "momy_fluid", "momz_fluid", "momx_pending", "momy_pending", "momz_pending",
            "Jx", "Jy", "Jz", "Jx_cum", "Jy_cum", "Jz_cum", "min_f", "min_post_R", "min_post_all", "min_Q_over_vi", "min_pending", "min_pending_ratio", "zero_capacity_rows",
            "max_density_departure", "max_speed", "max_uniform_error", "max_transverse", "amplitude", "clamp_count", "nonfinite_fi", "nonfinite_capture",
            "storage_increment_mass", "storage_increment_momx", "storage_increment_momy", "storage_increment_momz",
            "storage_bound_mass", "storage_bound_momx", "storage_bound_momy", "storage_bound_momz",
            "collision_defect_mass", "collision_defect_momx", "collision_defect_momy", "collision_defect_momz",
            "coeff_cell_mass", "coeff_out_mass", "coeff_cell_momx", "coeff_cell_momy", "coeff_cell_momz", "coeff_out_momx", "coeff_out_momy", "coeff_out_momz",
            "shell_residual", "mass_identity_residual", "mom_identity_residual_x", "mom_identity_residual_y", "mom_identity_residual_z", "incoming_sum",
            "max_abs_pending_zero_capacity", "low_capacity_rows", "ledger_abs_sum_mass", "ledger_ops_mass", "ledger_bound_mass", "ledger_abs_sum_momx", "ledger_abs_sum_momy", "ledger_abs_sum_momz",
            "ledger_ops_mom", "ledger_bound_momx", "ledger_bound_momy", "ledger_bound_momz", "storage_allowance_mass", "storage_allowance_momx", "storage_allowance_momy", "storage_allowance_momz",
            // schema v5 (task 21, FRICTION_SCREEN_SPECIFICATION section 5): the friction law at the sampled state (requested) and the per-facet
            // impulse minus its normal part (delivered), per step and cumulative, and the traction identity comparison proxy (abs sum, ops, bound)
            "requested_tangential_impulse_x", "requested_tangential_impulse_y", "requested_tangential_impulse_z",
            "requested_tangential_impulse_x_cum", "requested_tangential_impulse_y_cum", "requested_tangential_impulse_z_cum",
            "delivered_tangential_impulse_x", "delivered_tangential_impulse_y", "delivered_tangential_impulse_z",
            "delivered_tangential_impulse_x_cum", "delivered_tangential_impulse_y_cum", "delivered_tangential_impulse_z_cum",
            "traction_identity_abs_sum", "traction_identity_ops", "traction_identity_bound" };
        return list;
    }
    static std::string header() { std::string h{}; for(std::size_t i=0u; i<names().size(); i++) { h += (i ? "," : "")+names()[i]; } return h; }
    std::vector<double> v{ std::vector<double>(names().size(), 0.0) };
    double& at(const char* name) {
        const auto& n{ names() };
        for(std::size_t i=0u; i<n.size(); i++) { if(n[i]==name) return v[i]; }
        throw std::runtime_error(std::string("unknown series column ")+name);
    }
};

/// @brief Compensated (Neumaier) summation for the host inventories and identity terms: the rounding error of the
/// accumulated sum is bounded by 2 u64 sum|x| + O(n u64^2) sum|x| independent of the summation length, so the identity
/// residuals of the series measure the ledger, not the host reduction order.
struct Neumaier {
    double sum{}, compensation{}, abs_sum{};
    std::uint64_t count{};
    void add(const double x) {
        const double t{ sum+x };
        if(std::fabs(sum)>=std::fabs(x)) compensation += (sum-t)+x; else compensation += (x-t)+sum;
        sum = t;
        abs_sum += std::fabs(x);
        count++;
    }
    double value() const { return sum+compensation; }
};

/// @brief PREREGISTERED CONSERVATIVE INSTRUMENT COMPARISON PROXY for a host identity residual (#286 item 5, #287): NOT a
/// theorem of the complete ledger error. gamma_n(k) = k u /(1 - k u) with u = 2^-53 for the k additions (the plain
/// recursive-summation bound; the host sums are compensated), plus 4 u per operand for the decode (w + g), product
/// (V f, V c f), difference (p - f) and coefficient-product roundings. Not accounted for explicitly: the construction of
/// the static coefficient vectors, the kernel impulse reduction and the worst-case nesting of the compensation; an
/// exceedance is investigated (FAILED_INVESTIGATE), never excused. The absolute sums (actual start AND actual next
/// level) and k are columns, so the proxy is recomputable from the series.
inline double ledger_bound(const std::uint64_t operations, const double abs_sum) {
    const double u{ std::ldexp(1.0, -53) };
    const double ku{ static_cast<double>(operations)*u };
    const double gamma_n{ ku<1.0 ? ku/(1.0-ku) : 1.0 };
    return (gamma_n+4.0*u)*abs_sum;
}

/// @brief Half ULP of a float32 value with the conservative adjacent-spacing convention: the spacing of the binade
/// of |g| rounded up to the larger neighbouring binade at a binade boundary, and the subnormal spacing below the
/// smallest normal (#278). Used for the storage bound on g = f - w (the shifted quantity).
inline double half_ulp32_conservative(const double g) {
    const double a{ std::fabs(g) };
    const float min_normal{ 1.17549435e-38f };
    if(a<static_cast<double>(min_normal)) {
        return 0.5*static_cast<double>(1.40129846e-45f);   // subnormal spacing
    }
    int exponent{};
    std::frexp(a, &exponent);                 // a = m * 2^exponent, m in [0.5, 1)
    double spacing{ std::ldexp(1.0, exponent-24) };  // ulp of the binade of a (for a float32: 2^(e-24) with a in [2^(e-1), 2^e))
    const double rounded_to_float{ static_cast<double>(static_cast<float>(a)) };
    if(rounded_to_float>=std::ldexp(1.0, exponent)) {   // a rounds up to the next binade boundary: take the larger spacing
        spacing *= 2.0;
    }
    return 0.5*spacing;
}

}  // namespace native_protocol
