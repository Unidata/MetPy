#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace metpy_constants {
// Gas constants (J / kg / K)
constexpr double R = 8.314462618; // Universal gas constant (J / mol / K)

// Dry air
constexpr double Md = 28.96546e-3;  // Molar mass of dry air (kg / mol)
constexpr double Rd = R / Md;     // Dry air (J / kg / K)
constexpr double dry_air_spec_heat_ratio = 1.4;
constexpr double Cp_d = Rd * dry_air_spec_heat_ratio / (dry_air_spec_heat_ratio - 1.0); // (J / kg / K)
constexpr double Cv_d = Cp_d / dry_air_spec_heat_ratio; // (J / kg / K)

// Water
constexpr double Mw = 18.015268e-3; // Molar mass of water (kg / mol)
constexpr double Rv = R / Mw;      // Water vapor (J / kg / K)
constexpr double wv_spec_heat_ratio = 1.33;
constexpr double Cp_v = Rv * wv_spec_heat_ratio / (wv_spec_heat_ratio - 1.0); // (J / kg / K)
constexpr double Cv_v = Cp_v / wv_spec_heat_ratio; // (J / kg / K)
constexpr double Cp_l = 4.2194e3; // Specific heat capacity of liquid water (J / kg / K)
constexpr double Lv = 2.50084e6; // Latent heat of vaporization of water (J / kg)
constexpr double T0 = 273.16; // Triple point of water (K)

// General Meteorological constants
constexpr double epsilon = Mw / Md; // ≈ 0.622


// Standard gravity
constexpr double g = 9.80665;       // m / s^2

// Reference pressure
constexpr double P0 = 100000.0;     // Pa (hPa = 1000)

}

#endif // CONSTANTS_HPP
