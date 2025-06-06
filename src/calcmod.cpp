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

    m.def("_saturation_vapor_pressure_liquid", py::vectorize(_SaturationVaporPressureLiquid),
            "Calculate saturation vapor pressure from temperature.",
            py::arg("temperature"));
    
    m.def("saturation_vapor_pressure", py::vectorize(SaturationVaporPressure),
            "Calculate saturation vapor pressure from temperature.",
            py::arg("temperature"), py::arg("phase") = "liquid");
    
    // Unified binding with default epsilon
    m.def("dewpoint", py::vectorize(DewPoint),
            "Calculate dew point from water vapor partial pressure.",
            py::arg("vapor_pressure"));

    m.def("virtual_temperature", py::vectorize(VirtualTemperature),
            "Calculate virtual temperature from temperature and mixing ratio.",
            py::arg("temperature"), py::arg("mixing_ratio"), py::arg("epsilon"));
}
