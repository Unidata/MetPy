#include "constants.hpp"
#include "virtual_temperature.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(_calc_mod, m) {
    m.doc() = "accelerator module docstring";

    metpy_constants::load_constants_from_python();

    m.def("add", &add, "Add two numbers");

    m.def("water_latent_heat_vaporization", py::vectorize(WaterLatentHeatVaporization),
            "Calculate water latent heat vaporization from temperature.",
            py::arg("temperature"));

    m.def("water_latent_heat_sublimation", py::vectorize(WaterLatentHeatSublimation),
            "Calculate water latent heat sublimation from temperature.",
            py::arg("temperature"));

    m.def("saturation_vapor_pressure", py::vectorize(SaturationVaporPressure),
            "Calculate saturation vapor pressure from temperature.",
            py::arg("temperature"), py::arg("phase") = "liquid");
    
    m.def("_saturation_vapor_pressure_liquid", py::vectorize(_SaturationVaporPressureLiquid),
            "Calculate saturation vapor pressure from temperature.",
            py::arg("temperature"));
    
    m.def("_saturation_vapor_pressure_solid", py::vectorize(_SaturationVaporPressureSolid),
            "Calculate saturation vapor pressure from temperature.",
            py::arg("temperature"));

    m.def("dewpoint", py::vectorize(DewPoint),
            "Calculate dew point from water vapor partial pressure.",
            py::arg("vapor_pressure"));

    m.def("mixing_ratio", py::vectorize(MixingRatio),
            "Calculate the mixing ratio of a gas.",
            py::arg("partial_press"), py::arg("total_press"), py::arg("epsilon"));

    m.def("saturation_mixing_ratio", py::vectorize(SaturationMixingRatio),
            "Calculate the saturation mixing ratio of water vapor given total atmospheric pressure and temperature.",
            py::arg("total_press"), py::arg("temperature"), py::arg("phase"));

    m.def("specific_humidity_from_mixing_ratio", py::vectorize(SpecificHumidityFromMixingRatio),
            "Calculate the specific humidity from the mixing ratio.",
            py::arg("mixing_ratio"));

    m.def("specific_humidity_from_dewpoint", py::vectorize(SpecificHumidityFromDewPoint),
            "Calculate the specific humidity from the dewpoint temperature and pressure.",
            py::arg("pressure"), py::arg("dewpoint"), py::arg("phase"));

    
    m.def("virtual_temperature", py::vectorize(VirtualTemperature),
            "Calculate virtual temperature from temperature and mixing ratio.",
            py::arg("temperature"), py::arg("mixing_ratio"), py::arg("epsilon"));
}
