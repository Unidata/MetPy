#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace metpy_constants {

    // Gas constants (J / kg / K)
//    extern double R;

    // Dry air
//    extern double Md;
//    extern double Rd;
//    extern double dry_air_spec_heat_ratio;
//    extern double Cp_d = Rd * dry_air_spec_heat_ratio / (dry_air_spec_heat_ratio - 1.0); // (J / kg / K)
//    extern double Cv_d = Cp_d / dry_air_spec_heat_ratio; // (J / kg / K)

    // Water
    extern double Mw;
    extern double Rd;
    extern double Rv;
//    extern double wv_spec_heat_ratio = 1.33;
    extern double Cp_d;
    extern double Cp_v;
//    extern double Cv_v = Cp_v / wv_spec_heat_ratio; // (J / kg / K)
    extern double Cp_l;
    extern double Lv;
    extern double sat_pressure_0c;
    extern double T0;
    extern double Ls;
    extern double Cp_i;
    extern double zero_degc;
    extern double epsilon;
    extern double kappa;

    // General Meteorological constants
//    extern double epsilon = Mw / Md; // ≈ 0.622


    // Standard gravity
//    extern double g = 9.80665;       // m / s^2

    // Reference pressure
//    extern double P0 = 100000.0;     // Pa (hPa = 1000)


    void load_constants_from_python();  // call once in your PYBIND11_MODULE
}

#endif

