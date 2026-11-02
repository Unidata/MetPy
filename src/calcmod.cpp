#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "constants.hpp"
#include "thermo.hpp"
#include <utility> // For std::pair
#include <tuple>   // For std::make_tuple
#include <iostream>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(_calc_mod, m) {
    m.doc() = "accelerator module docstring";

    metpy_constants::load_constants_from_python();

    m.def("add", &add, "Add two numbers");
    
    m.def("moist_air_gas_constant", py::vectorize(MoistAirGasConstant),
            "Calculate R_m, the gas constant for moist air.",
            py::arg("specific_humidity"));

    m.def("moist_air_specific_heat_pressure", py::vectorize(MoistAirSpecificHeatPressure),
            "Calculate C_pm, the specific heat of moist air at constant pressure.",
            py::arg("specific_humidity"));

    m.def("water_latent_heat_vaporization", py::vectorize(WaterLatentHeatVaporization),
            "Calculate water latent heat vaporization from temperature.",
            py::arg("temperature"));

    m.def("water_latent_heat_sublimation", py::vectorize(WaterLatentHeatSublimation),
            "Calculate water latent heat sublimation from temperature.",
            py::arg("temperature"));

    m.def("relative_humidity_from_dewpoint", py::vectorize(RelativeHumidityFromDewPoint),
            "Calculate relative humidity from temperature and dewpoint.",
            py::arg("temperature"), py::arg("dewpoint"), py::arg("phase"));

    m.def("dry_lapse", py::vectorize(DryLapse),
            "Calculate the temperature along a pressure profile assuming dry adiabatic process.",
            py::arg("pressure"), py::arg("ref_temperature"), py::arg("ref_pressure"));

    m.def("moist_lapse", &MoistLapseVectorized,
          "Calculate the temperature along a pressure profile assuming saturated adiabatic process.",
          py::arg("pressure"), py::arg("ref_temperature"), py::arg("ref_pressure"), py::arg("rk_nstep"));


    m.def("lcl", &LCLVectorized,
            "Calculate the lifting condensation level (LCL) from pressure, temperature and dewpoint.",
            py::arg("pressure"), py::arg("temperature"), py::arg("dewpoint"));

    m.def("parcel_profile",
            [](py::array_t<double> pressure, double temperature, double dewpoint) {
                // pressure.data() gives the beginning pointer of the data buffer
                std::vector<double> pressure_vec(pressure.data(), pressure.data() + pressure.size());
                std::vector<double> temp_prof = ParcelProfile(pressure_vec, temperature, dewpoint);
                return py::array_t<double>(temp_prof.size(), temp_prof.data());
            },
            "Compute the parcel temperature profile as it rises from a given pressure and temperature.",
            py::arg("pressure"), py::arg("temperature"), py::arg("dewpoint"));


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
            "Calculate dewpoint from water vapor partial pressure.",
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

    m.def("virtual_temperature_from_dewpoint", py::vectorize(VirtualTemperatureFromDewpoint),
            "Calculate virtual temperature from dewpoint.",
            py::arg("pressure"), py::arg("temperature"), py::arg("dewpoint"), py::arg("epsilon"), py::arg("phase"));

}
