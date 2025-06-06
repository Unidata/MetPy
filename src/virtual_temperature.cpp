#include <cmath>
#include <string>
#include <pybind11/pybind11.h>
#include "constants.hpp"
#include "virtual_temperature.hpp"
#include <stdexcept>

namespace py = pybind11;
namespace mc = metpy_constants;

double WaterLatentHeatVaporization(double temperature) {
    return mc::Lv - (mc::Cp_l - mc::Cp_v) * (temperature - mc::T0);
}

double WaterLatentHeatSublimation(double temperature) {
    return mc::Ls - (mc::Cp_i - mc::Cp_v) * (temperature - mc::T0);
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
    return 243.5 * val / (17.67 - val);  // use SI units instead
}

double VirtualTemperature(double temperature, double mixing_ratio, double epsilon) {
    return temperature * (mixing_ratio + epsilon) / (epsilon * (1. + mixing_ratio));
}
