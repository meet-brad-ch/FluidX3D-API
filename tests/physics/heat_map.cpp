// Heat map: a periodic box of fluid at rest whose layer z = 0 is flagged TYPE_Y, each cell heated at its own rate from
// LBM::heat (HEAT_MAP) in place of the one heat_rate. The source layer's cells x < Nx/2 get the low rate, the others four
// times it, and the row y = 0 gets zero (flagged TYPE_Y, but no heat). Without walls every bit of added heat stays in the
// fluid cells, so the mean temperature rises by sum(heat) / (cells) per time step.
// Passes when the mean temperature rise from step 1000 to 2000 is within 0.1 % of that, and the first layer above the hot
// half has warmed more than the one above the cold half (the map, not one rate, heats the cells).

// On the core directly, not the atmosphere API, as tests/physics/heat_source.cpp: a periodic box without walls, where a sum
// over the fluid cells holds all the added heat.

#include "setup/setup.hpp"
#include "physics_check.hpp"

#include <cmath>
#include <string>

namespace {

/// @brief One periodic box at rest whose layer z = 0 is heated by a map: a cold half, a hot half and an unheated row.
/// Checks the added heat against the map's sum and that the hot half's air warmed more than the cold half's.
class HeatMapBox {
public:
	/// @brief Runs the box and reports its two checks.
	static void run() {
		LBM lbm(cells_x, cells_y, cells_z, viscosity, 0.0f, 0.0f, 0.0f, 0.0f, alpha, 0.0f); // no gravity: the fluid stays at rest
		const double heat_per_step{ fill_map(lbm) };
		lbm.run(steps);
		const double rise1{ mean_rise(lbm) };
		lbm.run(steps);
		const double rise2{ mean_rise(lbm) };
		const double cold{ half_layer_rise(lbm, false) };
		const double hot{ half_layer_rise(lbm, true) };
		const double cells{ static_cast<double>(lbm.get_N()) };
		const double expected{ heat_per_step * static_cast<double>(steps) / cells }; // mean rise over one run of steps
		const unsigned long long printed_steps{ static_cast<unsigned long long>(steps) };
		const unsigned long long printed_both{ 2ull * printed_steps };
		std::printf("\nheat map: sum of the map %.6e per step; mean temperature rise %.6e after %llu steps, %.6e after %llu (expected per %llu steps %.6e); layer z = 1 above the cold half %.6e, above the hot half %.6e\n",
		            heat_per_step, rise1, printed_steps, rise2, printed_both, printed_steps, expected, cold, hot);
		const double heat_error{ std::fabs((rise2 - rise1) / expected - 1.0) };
		PhysicsCheck::report("heat map: the added heat is the map's sum", heat_error, heat_limit);
		const double map_error{ hot > cold ? 0.0 : 1.0 };
		PhysicsCheck::report("heat map: the hot half's air warmed more than the cold half's", map_error, map_limit);
	}

private:
	/// @brief Flags the source layer z = 0 TYPE_Y and fills the heat map: the low rate at x < Nx/2, four times it at
	/// x >= Nx/2, zero on the row y = 0.
	/// @param lbm the box's lattice
	/// @return the sum of the map, the temperature added per time step to the whole box
	static double fill_map(LBM& lbm) {
		double sum{ 0.0 };
		const ulong cells = lbm.get_N();
		for(ulong n{ 0ull }; n < cells; n++) {
			uint x{ 0u };
			uint y{ 0u };
			uint z{ 0u };
			lbm.coordinates(n, x, y, z);
			if(z != 0u) { // not the source layer
				continue;
			}
			lbm.flags[n] = TYPE_Y;
			const bool unheated_row{ y == 0u };
			const bool hot_half{ x >= cells_x / 2u };
			const float rate{ unheated_row ? 0.0f : (hot_half ? hot_factor * low_rate : low_rate) };
			lbm.heat[n] = rate;
			sum += static_cast<double>(rate);
		}
		return sum;
	}

	/// @brief Gives the mean temperature rise of all cells.
	/// @param lbm the box's lattice
	/// @return the mean of T - 1 in lattice temperature
	static double mean_rise(LBM& lbm) {
		lbm.update_fields();
		lbm.T.read_from_device();
		double sum{ 0.0 };
		const ulong cells = lbm.get_N();
		for(ulong n{ 0ull }; n < cells; n++) {
			sum += static_cast<double>(lbm.T[n]) - 1.0;
		}
		const double mean{ sum / static_cast<double>(cells) };
		return mean;
	}

	/// @brief Gives the mean temperature rise of the layer z = 1 above one half of the source layer, from the last read.
	/// @param lbm the box's lattice
	/// @param hot_half true: the cells x >= Nx/2; false: x < Nx/2
	/// @return the mean of T - 1 in lattice temperature
	static double half_layer_rise(LBM& lbm, bool hot_half) {
		double sum{ 0.0 };
		ulong counted{ 0ull };
		const ulong cells = lbm.get_N();
		for(ulong n{ 0ull }; n < cells; n++) {
			uint x{ 0u };
			uint y{ 0u };
			uint z{ 0u };
			lbm.coordinates(n, x, y, z);
			const bool in_half{ (x >= cells_x / 2u) == hot_half };
			if(z == 1u && in_half) {
				sum += static_cast<double>(lbm.T[n]) - 1.0;
				counted++;
			}
		}
		const double mean{ sum / static_cast<double>(counted) };
		return mean;
	}

	static constexpr uint cells_x = 16u;      ///< the box's cells along x
	static constexpr uint cells_y = 16u;      ///< the box's cells along y
	static constexpr uint cells_z = 16u;      ///< the box's cells along z
	static constexpr float viscosity = 0.1f;  ///< the kinematic viscosity in lattice units
	static constexpr float alpha = 0.05f;     ///< the thermal diffusion coefficient in lattice units
	/// the temperature added per time step to a cold-half source cell; large against the rounding of T = 1+sum(g) in
	/// float, as in tests/physics/heat_source.cpp
	static constexpr float low_rate = 1.0E-3f;
	static constexpr float hot_factor = 4.0f;      ///< the hot half's rate over the cold half's
	static constexpr ulong steps = 1000ull;        ///< the steps of each of the two runs
	static constexpr double heat_limit = 0.001;    ///< the largest relative error of the added heat that passes (0.1 %)
	static constexpr double map_limit = 0.5;       ///< hot half not warmer fails (error 1), warmer passes (error 0)
};

} // namespace

void main_setup() { // extensions: TEMPERATURE HEAT_MAP
	HeatMapBox::run();
}
