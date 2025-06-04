#include <cmath>
#include <pybind11/pybind11.h>
#include "constants.hpp"
#include "virtual_temperature.hpp"
//#include <stdexcept>

namespace py = pybind11;

double water_latent_heat_vaporization(double temperature) {
    // Calculate the latent heat of vaporization of water in J/kg at a given temperature.
//    using namespace metpy_constants;
//    return Lv - (Cp_l - Cp_v) * (temperature - T0);
}

double _saturation_vapor_pressure(double temperature) {
    // Calculate saturation (equilibrium) water vapor (partial) pressure over liquid water.
    // Constants for the Magnus-Tetens approximation
    //const double a = 17.67;
    //const double b = 243.5;

    // Calculate saturation vapor pressure using the Magnus-Tetens formula
    //return 6.112 * exp((a * temperature) / (b + temperature));
}

double DewPoint(double vapor_pressure) {
    // fetch constants from python module
    //py::object default_mod = py::module_::import("metpy.constants.default");
    // unit issue ignored for now
    //double sat_pressure_0c = default_mod.attr("sat_pressure_0c").attr("magnitude").cast<double>();

    double val = log(vapor_pressure / sat_pressure_0c);
    return 243.5 * val / (17.67 - val);
}

double VirtualTemperature(double temperature, double mixing_ratio, double epsilon) {
    return temperature * (mixing_ratio + epsilon) / (epsilon * (1. + mixing_ratio));
}
