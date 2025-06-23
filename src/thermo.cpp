#include <cmath>
#include <string>
#include <vector>
#include <utility> // For std::pair
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <iostream>   // for std::cerr
#include <limits>     // for std::numeric_limits
#include "math.hpp"
#include "constants.hpp"
#include "thermo.hpp"

namespace py = pybind11;
namespace mc = metpy_constants;

double MoistAirGasConstant(double specific_humidity) {
    return (1.0 - specific_humidity) * mc::Rd + specific_humidity * mc::Rv;
}

double MoistAirSpecificHeatPressure(double specific_humidity) {
    return (1.0 - specific_humidity) * mc::Cp_d + specific_humidity * mc::Cp_v;
}

double WaterLatentHeatVaporization(double temperature) {
    return mc::Lv - (mc::Cp_l - mc::Cp_v) * (temperature - mc::T0);
}

double WaterLatentHeatSublimation(double temperature) {
    return mc::Ls - (mc::Cp_i - mc::Cp_v) * (temperature - mc::T0);
}

double RelativeHumidityFromDewPoint(double temperature, double dewpoint, std::string phase) {
    double e_s = SaturationVaporPressure(temperature, phase);
    double e = SaturationVaporPressure(dewpoint, phase);
    return e / e_s;
}

double DryLapse(double pressure, double ref_temperature, double ref_pressure) {
    // calculate temperature at pressure given reference temperature and pressure
    // assuming dry adiabatic process
    return ref_temperature * pow(pressure / ref_pressure, mc::kappa);
}

std::vector<double> DryLapseProfile(const std::vector<double>& pressure_profile,
                                    double ref_temperature,
                                    double ref_pressure) {
    // Vectorized version of DryLapse for a pressure profile. C++ internally use.

    // calculate temperature profile of an air parcel lifting dry adiabatically
    // through the given pressure profile.
    std::vector<double> temperature_profile;
    temperature_profile.reserve(pressure_profile.size());

    for (double p : pressure_profile) {
        temperature_profile.push_back(DryLapse(p, ref_temperature, ref_pressure));
    }
    return temperature_profile;
}

double CaldlnTdlnP(double temperature, double pressure) {
    // Calculate dlnT/dlnP for a moist (saturated) adiabatic process.
    double rs = SaturationMixingRatio(pressure, temperature, "liquid");
    
    //double dlnT_dlnP_linfel = (mc::Rd + rs * mc::Rv) / (mc::Cp_d + rs * mc::Cp_v + 
    //        (mc::Lv * mc::Lv * rs * mc::epsilon) / (mc::Rd * temperature * temperature));
    double dlnT_dlnP_Bakhshaii2013 = (mc::Rd + mc::Lv * rs / temperature) / (mc::Cp_d + 
            (mc::Lv * mc::Lv * rs * mc::epsilon) / (mc::Rd * temperature * temperature));
    
    return dlnT_dlnP_Bakhshaii2013;
}

double MoistLapse(double pressure, double ref_temperature, double ref_pressure, int nstep) {
    // calculate temperature at pressure given reference temperature and pressure
    // assuming moist adiabatic expansion (vapor condenses and removed from the air
    // parcel)
    
    double dlnP = log(pressure / ref_pressure) / (double)nstep;
    double T1 = ref_temperature;
    double P1 = ref_pressure;
    double k[4];

    for (int i = 0; i < nstep; ++i) {
        k[0] = CaldlnTdlnP(T1, P1); 
        k[1] = CaldlnTdlnP(T1 * exp(k[0] * dlnP/2.), P1 * exp(dlnP/2.));
        k[2] = CaldlnTdlnP(T1 * exp(k[1] * dlnP/2.), P1 * exp(dlnP/2.));
        k[3] = CaldlnTdlnP(T1 * exp(k[2] * dlnP), P1 * exp(dlnP));

        T1 = T1 * exp((k[0] + 2.0 * k[1] + 2.0 * k[2] + k[3]) * dlnP / 6.0);
        P1 = P1 * exp(dlnP);
    }

    return T1; // check final T1 P1
}

std::vector<double> MoistLapseProfile(const std::vector<double>& pressure_profile,
                                    double ref_temperature,
                                    double ref_pressure) {
    // Vectorized version of MoistLapse for a pressure profile. C++ internally use.

    // calculate temperature profile of an air parcel lifting saturated adiabatically
    // through the given pressure profile.
    std::vector<double> temperature_profile;
    temperature_profile.reserve(pressure_profile.size());

    //double T1 = ref_temperature;
    //double P1 = ref_pressure;
    double T;
    for (size_t i = 0; i < pressure_profile.size(); ++i) {
        T = MoistLapse(pressure_profile[i], ref_temperature, ref_pressure, 50);
        temperature_profile.push_back(T);
    }
    return temperature_profile;
}

std::pair<double, double> LCL(double pressure, double temperature, double dewpoint) {
    if (temperature <= dewpoint) {
        std::cerr << "Warning in function '" << __func__
            << "': Temperature must be greater than dew point for LCL calculation.\n";
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }

    double q = SpecificHumidityFromDewPoint(pressure, dewpoint, "liquid");
    double moist_heat_ratio = MoistAirSpecificHeatPressure(q) / MoistAirGasConstant(q);
    double spec_heat_diff = mc::Cp_l - mc::Cp_v;
    
    double a = moist_heat_ratio + spec_heat_diff / mc::Rv; 
    double b = -(mc::Lv + spec_heat_diff * mc::T0) / (mc::Rv * temperature);
    double c = b / a;

    double rh = RelativeHumidityFromDewPoint(temperature, dewpoint, "liquid");
    double w_minus1 = lambert_wm1(pow(rh, 1.0 / a) * c * exp(c));
    double t_lcl = c / w_minus1 * temperature;
    double p_lcl = pressure * pow(t_lcl / temperature, moist_heat_ratio);

    return {p_lcl, t_lcl};
}

bool _CheckPressure(const std::vector<double>& pressure) {
    for (size_t i = 0; i + 1 < pressure.size(); ++i) {
        if (pressure[i] < pressure[i + 1]) {
            return false;  // pressure increases (invalid)
        }
    }
    return true;  // strictly non-increasing
}

void _ParcelProfileHelper(const std::vector<double>& pressure, double temperature, double dewpoint) {
    // Check that pressure does not increase.
    if (!_CheckPressure(pressure)) {
        throw std::runtime_error(
            "Pressure increases between at least two points in your sounding. "
            "Using a smoothing filter (e.g., scipy.signal.medfilt) may fix this.");
    }
    
    // Find the LCL
    auto [press_lcl, temp_lcl] = LCL(pressure[0], temperature, dewpoint);
    
    // Establish profile below LCL
    std::vector<double> press_lower;
    for (double p : pressure) {
        if (p >= press_lcl) {
            press_lower.push_back(p);
        }
    }
    press_lower.push_back(press_lcl);
    std::vector<double> temp_lower = DryLapseProfile(press_lower, temperature, press_lower[0]);

    // Early return if profile ends before reaching above LCL
    if (pressure.back() >= press_lcl) {
        press_lower.pop_back();
        temp_lower.pop_back();
//        return {press_lower, {}, press_lcl, temp_lower, {}, temp_lcl};
    }

    // Establish profile above LCL
    std::vector<double> press_upper;
    press_upper.push_back(press_lcl);
    for (double p : pressure) {
        if (p < press_lcl) {
            press_upper.push_back(p);
        }
    }


}

double _SaturationVaporPressureLiquid(double temperature) {
    double latent_heat = WaterLatentHeatVaporization(temperature);
    double heat_power = (mc::Cp_l - mc::Cp_v) / mc::Rv;
    double exp_term = (mc::Lv / mc::T0 - latent_heat / temperature) / mc::Rv;

    return mc::sat_pressure_0c * exp(exp_term) * pow(mc::T0 / temperature, heat_power);
}

double _SaturationVaporPressureSolid(double temperature) {
    double latent_heat = WaterLatentHeatSublimation(temperature);
    double heat_power = (mc::Cp_i - mc::Cp_v) / mc::Rv;
    double exp_term = (mc::Ls / mc::T0 - latent_heat / temperature) / mc::Rv;

    return mc::sat_pressure_0c * exp(exp_term) * pow(mc::T0 / temperature, heat_power);
}

double SaturationVaporPressure(double temperature, std::string phase) {
    if (phase == "liquid") {
        return _SaturationVaporPressureLiquid(temperature);
    } else if (phase == "solid") {
        return _SaturationVaporPressureSolid(temperature);
    } else if (phase == "auto") {
        if (temperature > mc::T0) {
            return _SaturationVaporPressureLiquid(temperature);
        } else {
            return _SaturationVaporPressureSolid(temperature);
        }
    } else {
        throw std::invalid_argument("'" + phase + "' is not a valid option for phase. "
                                    "Valid options are 'liquid', 'solid', or 'auto'.");
    }
}

double DewPoint(double vapor_pressure) {
    double val = log(vapor_pressure / mc::sat_pressure_0c);
    return mc::zero_degc + 243.5 * val / (17.67 - val);
}

double MixingRatio(double partial_press, double total_press, double epsilon) {
    return epsilon * partial_press / (total_press - partial_press);
}

double SaturationMixingRatio(double total_press, double temperature, std::string phase) {
    double e_s = SaturationVaporPressure(temperature, phase);
    if (e_s >= total_press) {
        std::cerr << "Warning in function '" << __func__
            << "': Total pressure must be greater than the saturation vapor pressure "
            << "for liquid water to be in equilibrium.\n";
        return std::numeric_limits<double>::quiet_NaN();
    }
    return MixingRatio(e_s, total_press);
}

double SpecificHumidityFromMixingRatio(double mixing_ratio) {
    return mixing_ratio / (mixing_ratio + 1.0);
}

double SpecificHumidityFromDewPoint(double pressure, double dewpoint, std::string phase) {
    double mixing_ratio = SaturationMixingRatio(pressure, dewpoint, phase);
    return SpecificHumidityFromMixingRatio(mixing_ratio);
}

double VirtualTemperature(double temperature, double mixing_ratio, double epsilon) {
    return temperature * (mixing_ratio + epsilon) / (epsilon * (1. + mixing_ratio));
}
