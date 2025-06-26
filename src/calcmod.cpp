#include "constants.hpp"
#include "thermo.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
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
            "Calculate the temperature at pressure levels assuming dry adiabatic process.",
            py::arg("pressure"), py::arg("ref_temperature"), py::arg("ref_pressure"));

    m.def("moist_lapse", [](py::array_t<double> pressure,
                                py::array_t<double> ref_temperature,
                                double ref_pressure,
                                int rk_nstep) {
            // This function calculates the moist adiabatic profile for multiple starting
            // temperatures (2D surface) and a single communal starting pressure, along a 
            // 1D pressure profile.

            // --- Step 1: Prepare the C++ vector for pressure levels ---
            if (pressure.ndim() != 1) {
                throw std::runtime_error("Input 'pressure' array must be 1D.");
            }
            std::vector<double> pressure_vec(pressure.data(), pressure.data() + pressure.size());

            // --- Step 2: Ensure the reference temperature array is contiguous ---
            auto ref_temp_contig = py::array::ensure(ref_temperature, py::array::c_style);
            
            // --- Step 3: Define the shape of the output array: (N+1) dimension---
            // Create a vector to hold the shape dimensions.
            std::vector<ssize_t> out_shape;
            for(int i = 0; i < ref_temp_contig.ndim(); ++i) {
                out_shape.push_back(ref_temp_contig.shape(i));
            }
            ssize_t profile_len = pressure_vec.size();
            out_shape.push_back(profile_len);
            
            // Create the final output array with the correct N+1 dimensional shape.
            auto out_array = py::array_t<double>(out_shape);

            // --- Step 4: Get direct pointers to data buffers for fast access ---
            const double* ref_temp_ptr = static_cast<const double*>(ref_temp_contig.request().ptr);
            double* out_array_ptr = out_array.mutable_data();
            ssize_t num_profiles = ref_temp_contig.size(); // Total number of profiles to calculate

            // --- Step 5: Loop through each reference temperature ---
            for (ssize_t i = 0; i < num_profiles; ++i) {
                for (ssize_t j = 0; j < profile_len; ++j) {
                    out_array_ptr[i * profile_len + j] = MoistLapse(pressure_vec[j], ref_temp_ptr[i], ref_pressure, rk_nstep);
                }
            }

            return out_array;
          }, "Calculate the temperature along a pressure profile assuming saturated adiabatic process.",
          py::arg("pressure"), py::arg("ref_temperature"), py::arg("ref_pressure"), py::arg("rk_nstep"));


    m.def("lcl", [](py::array_t<double> pressure,
                    py::array_t<double> temperature,
                    py::array_t<double> dewpoint) {
            
            // This helper ensures the arrays are in C-style contiguous memory.
            // If an input array is already contiguous, it's a zero-cost operation.
            // If it's a slice or has a different memory layout, it creates a copy.
            // This makes the subsequent looping simple and safe.
            auto p_contig = py::array::ensure(pressure, py::array::c_style);
            auto t_contig = py::array::ensure(temperature, py::array::c_style);
            auto d_contig = py::array::ensure(dewpoint, py::array::c_style);

            // --- Step 1: Check that all input arrays have the same shape ---
            if (p_contig.ndim() != t_contig.ndim() || p_contig.ndim() != d_contig.ndim()) {
                throw std::runtime_error("Input arrays must have the same number of dimensions.");
            }
            for (int i = 0; i < p_contig.ndim(); ++i) {
                if (p_contig.shape(i) != t_contig.shape(i) || p_contig.shape(i) != d_contig.shape(i)) {
                    throw std::runtime_error("Input arrays must have the same shape.");
                }
            }

            // --- Step 2: Create output arrays with the exact same N-D shape as the inputs ---
            auto p_lcl = py::array_t<double>(p_contig.request().shape);
            auto t_lcl = py::array_t<double>(p_contig.request().shape);

            // --- Step 3: Get the total number of elements to loop over ---
            ssize_t size = p_contig.size();

            // --- Step 4: Get direct pointers to the (now contiguous) data buffers ---
            const double* p_ptr = static_cast<const double*>(p_contig.request().ptr);
            const double* t_ptr = static_cast<const double*>(t_contig.request().ptr);
            const double* d_ptr = static_cast<const double*>(d_contig.request().ptr);
            double* p_lcl_ptr = p_lcl.mutable_data();
            double* t_lcl_ptr = t_lcl.mutable_data();
            
            // --- Step 5: Loop through the data as if it were a single flat 1D array ---
            for (ssize_t i = 0; i < size; i++) {
                // Call the scalar c++ function for each element
                std::pair<double, double> result = LCL(p_ptr[i], t_ptr[i], d_ptr[i]);
                
                p_lcl_ptr[i] = result.first;
                t_lcl_ptr[i] = result.second;
            }

            // --- Step 6: Return a tuple of the two new, N-dimensional arrays ---
            return std::make_tuple(p_lcl, t_lcl);

        }, "Calculate the lifting condensation level (LCL) from pressure, temperature and dewpoint.",
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
}
