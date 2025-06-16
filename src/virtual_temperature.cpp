#include <cmath>
#include <string>
#include <utility> // For std::pair
#include <pybind11/pybind11.h>
#include "math.hpp"
#include "constants.hpp"
#include "virtual_temperature.hpp"
#include <stdexcept>
#include <iostream>   // for std::cerr
#include <limits>     // for std::numeric_limits

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

    return {p_lcl, t_lcl}; // returning t_lcl and p_lcl together is needed
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
