#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <string>
#include <vector>
#include <tuple>   // For std::make_tuple
#include <utility> // For std::pair
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


py::array_t<double> DryLapseVectorized(py::array_t<double> pressure,
                                         py::array_t<double> ref_temperature,
                                         double ref_pressure) {
    // This function calculates the dry adiabatic profile for multiple starting
    // temperatures (2D surface) and a single communal starting pressure, along a 
    // 1D pressure profile.
    // --- Step 1: Prepare the C++ vector for pressure levels ---
    if (pressure.ndim() > 1) {
        throw std::runtime_error("Input 'pressure' must be 1D array or a single value.");
    }
    std::vector<double> pressure_vec(pressure.data(), pressure.data() + pressure.size());

    // --- Step 2: Ensure the reference temperature array is contiguous ---
    auto ref_temp_contig = py::array::ensure(ref_temperature, py::array::c_style);
    
    // --- Step 3: Define the shape of the output array: (N+1) dimension---
    std::vector<size_t> out_shape;
    for(int i = 0; i < ref_temp_contig.ndim(); ++i) {
        out_shape.push_back(ref_temp_contig.shape(i));
    }
    size_t profile_len = pressure_vec.size();
    out_shape.push_back(profile_len);
    
    auto out_array = py::array_t<double>(out_shape);

    // --- Step 4: Get direct pointers to data buffers for fast access ---
    const double* ref_temp_ptr = static_cast<const double*>(ref_temp_contig.request().ptr);
    double* out_array_ptr = out_array.mutable_data();
    size_t num_profiles = ref_temp_contig.size();

    // --- Step 5: Loop through each reference temperature ---
    for (size_t i = 0; i < num_profiles; ++i) {
        for (size_t j = 0; j < profile_len; ++j) {
            out_array_ptr[i * profile_len + j] = DryLapse(pressure_vec[j], ref_temp_ptr[i], ref_pressure);
        }
    }

    return out_array;
}

py::array_t<double> DryLapseVectorized_3D(py::array_t<double> pressure,
                                         py::array_t<double> ref_temperature,
                                         py::array_t<double> ref_pressure) {
    // --- Step 1: Ensure all input arrays are using a contiguous memory layout ---
    auto p_contig = py::array::ensure(pressure, py::array::c_style);
    auto ref_temp_contig = py::array::ensure(ref_temperature, py::array::c_style);
    auto ref_press_contig = py::array::ensure(ref_pressure, py::array::c_style);

    // --- Step 2: Perform comprehensive shape validation ---
    if (ref_temp_contig.ndim() != ref_press_contig.ndim()) {
        throw std::runtime_error("Input 'ref_temperature' and 'ref_pressure' must have the same number of dimensions.");
    }
    if (p_contig.ndim() != ref_temp_contig.ndim() + 1) {
        throw std::runtime_error("Input 'pressure' must have one more dimension than 'ref_temperature'.");
    }
    for (int i = 0; i < ref_temp_contig.ndim(); ++i) {
        if (ref_temp_contig.shape(i) != ref_press_contig.shape(i) ||
            p_contig.shape(i+1) != ref_temp_contig.shape(i)) {
            throw std::runtime_error("The horizontal dimensions of all input arrays must match.");
        }
    }

    // --- Step 3: Define the shape of the output array ---
    // The output shape will be identical to the input pressure array's shape.
    auto out_array = py::array_t<double>(p_contig.request().shape);

    // --- Step 4: Get direct pointers to data buffers for fast access ---
    const double* pressure_ptr = static_cast<const double*>(p_contig.request().ptr);
    const double* ref_temp_ptr = static_cast<const double*>(ref_temp_contig.request().ptr);
    const double* ref_press_ptr = static_cast<const double*>(ref_press_contig.request().ptr);
    double* out_array_ptr = out_array.mutable_data();

    // --- Step 5: Define loop boundaries ---
    size_t num_profiles = ref_temp_contig.size(); // Total number of horizontal points
    size_t profile_len = p_contig.shape(0); // Length of the vertical dimension
    
    // --- Step 6: Loop through each horizontal point and its vertical profile ---
    for (int j = 0; j < profile_len; ++j) {
        for (int i = 0; i < num_profiles; ++i) {
            // Calculate the index for the current point in the flattened arrays.
            out_array_ptr[i+j*num_profiles] = DryLapse(pressure_ptr[i+j*num_profiles], 
                                                    ref_temp_ptr[i], 
                                                    ref_press_ptr[i]);
        }
    }

    return out_array;
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

double MoistLapse(double pressure, double ref_temperature, double ref_pressure, int rk_nstep) {
    // calculate temperature at pressure given reference temperature and pressure
    // assuming moist adiabatic expansion (vapor condenses and removed from the air
    // parcel)
    
    double dlnP = log(pressure / ref_pressure) / (double)rk_nstep;
    double T1 = ref_temperature;
    double P1 = ref_pressure;
    double k[4];

    for (int i = 0; i < rk_nstep; ++i) {
        k[0] = CaldlnTdlnP(T1, P1); 
        k[1] = CaldlnTdlnP(T1 * exp(k[0] * dlnP/2.), P1 * exp(dlnP/2.));
        k[2] = CaldlnTdlnP(T1 * exp(k[1] * dlnP/2.), P1 * exp(dlnP/2.));
        k[3] = CaldlnTdlnP(T1 * exp(k[2] * dlnP), P1 * exp(dlnP));

        T1 = T1 * exp((k[0] + 2.0 * k[1] + 2.0 * k[2] + k[3]) * dlnP / 6.0);
        P1 = P1 * exp(dlnP);
    }

    return T1; // check final T1 P1
}

std::vector<double> MoistLapseProfile(const std::vector<double>& press_profile,
                                    double ref_temperature,
                                    double ref_pressure,
                                    int rk_nstep) {
    // MoistLapse for one full pressure profile given one ref_temperature. C++ internally use.

    // calculate temperature profile of an air parcel lifting saturated adiabatically
    // through the given pressure profile.
    std::vector<double> temp_profile;
    temp_profile.reserve(press_profile.size());

    for (double p : press_profile) {
        temp_profile.push_back(MoistLapse(p, ref_temperature, ref_pressure, rk_nstep));
    }
    return temp_profile;
}

py::array_t<double> MoistLapseVectorized(py::array_t<double> pressure,
                                         py::array_t<double> ref_temperature,
                                         double ref_pressure,
                                         int rk_nstep) {
    // This function calculates the moist adiabatic profile for multiple starting
    // temperatures (2D surface) and a single communal starting pressure, along a 
    // 1D pressure profile.
    // --- Step 1: Prepare the C++ vector for pressure levels ---
    if (pressure.ndim() > 1) {
        throw std::runtime_error("Input 'pressure' must be 1D array or a single value.");
    }
    std::vector<double> pressure_vec(pressure.data(), pressure.data() + pressure.size());

    // --- Step 2: Ensure the reference temperature array is contiguous ---
    auto ref_temp_contig = py::array::ensure(ref_temperature, py::array::c_style);
    
    // --- Step 3: Define the shape of the output array: (N+1) dimension---
    std::vector<size_t> out_shape;
    for(int i = 0; i < ref_temp_contig.ndim(); ++i) {
        out_shape.push_back(ref_temp_contig.shape(i));
    }
    size_t profile_len = pressure_vec.size();
    out_shape.push_back(profile_len);
    
    auto out_array = py::array_t<double>(out_shape);

    // --- Step 4: Get direct pointers to data buffers for fast access ---
    const double* ref_temp_ptr = static_cast<const double*>(ref_temp_contig.request().ptr);
    double* out_array_ptr = out_array.mutable_data();
    size_t num_profiles = ref_temp_contig.size();

    // --- Step 5: Loop through each reference temperature ---
    for (size_t i = 0; i < num_profiles; ++i) {
        for (size_t j = 0; j < profile_len; ++j) {
            out_array_ptr[i * profile_len + j] = MoistLapse(pressure_vec[j], ref_temp_ptr[i], ref_pressure, rk_nstep);
        }
    }

    return out_array;
}


std::pair<double, double> LCL(double pressure, double temperature, double dewpoint) {
    if (temperature < dewpoint) {
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


std::tuple<py::array_t<double>, py::array_t<double>> LCLVectorized(py::array_t<double> pressure,
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
    size_t size = p_contig.size();

    // --- Step 4: Get direct pointers to the (now contiguous) data buffers ---
    const double* p_ptr = static_cast<const double*>(p_contig.request().ptr);
    const double* t_ptr = static_cast<const double*>(t_contig.request().ptr);
    const double* d_ptr = static_cast<const double*>(d_contig.request().ptr);
    double* p_lcl_ptr = p_lcl.mutable_data();
    double* t_lcl_ptr = t_lcl.mutable_data();
    
    // --- Step 5: Loop through the data as if it were a single flat 1D array ---
    for (size_t i = 0; i < size; i++) {
        // Call the scalar c++ function for each element
        std::pair<double, double> result = LCL(p_ptr[i], t_ptr[i], d_ptr[i]);
        p_lcl_ptr[i] = result.first;
        t_lcl_ptr[i] = result.second;
    }

    // --- Step 6: Return a tuple of the two new, N-dimensional arrays ---
    return std::make_tuple(p_lcl, t_lcl);
}

bool _CheckPressure(const std::vector<double>& pressure) {
    for (size_t i = 0; i + 1 < pressure.size(); ++i) {
        if (pressure[i] < pressure[i + 1]) {
            return false;  // pressure increases (invalid)
        }
    }
    return true;  // strictly non-increasing
}


std::vector<double> ParcelProfile(const std::vector<double>& pressure,
                                  double temperature,
                                  double dewpoint) {
    // Returns a vector of temperatures corresponding to the input pressure levels.

    ParProStruct profile = _ParcelProfileHelper(pressure, temperature, dewpoint);

    // Combine lower and upper temperature profiles
    std::vector<double> combined_temp;
    combined_temp.reserve(pressure.size());

    combined_temp.insert(combined_temp.end(), profile.temp_lower.begin(), profile.temp_lower.end());
    combined_temp.insert(combined_temp.end(), profile.temp_upper.begin(), profile.temp_upper.end());

    return combined_temp;
}

ParProStruct _ParcelProfileHelper(const std::vector<double>& pressure, double temperature, double dewpoint) {
    // Check that pressure does not increase.
    if (!_CheckPressure(pressure)) {
        throw std::runtime_error(
            "Pressure increases between at least two points in your sounding. "
            "Using a smoothing filter (e.g., scipy.signal.medfilt) may fix this.");
    }
    
    // Find the LCL
    std::pair<double, double> lcl_result = LCL(pressure[0], temperature, dewpoint);
    double press_lcl = lcl_result.first;
    double temp_lcl = lcl_result.second;
    
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
        return {press_lower, {}, press_lcl, temp_lower, {}, temp_lcl};
    }

    // Establish profile above LCL
    std::vector<double> press_upper;
    press_upper.push_back(press_lcl);
    for (double p : pressure) {
        if (p < press_lcl) {
            press_upper.push_back(p);
        }
    }
    std::vector<double> temp_upper = MoistLapseProfile(press_upper, temp_lower.back(), press_lcl, 30);

    press_lower.pop_back();
    temp_lower.pop_back();
    press_upper.erase(press_upper.begin());
    temp_upper.erase(temp_upper.begin());

    return {press_lower, press_upper, press_lcl, temp_lower, temp_upper, temp_lcl};
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

double VirtualTemperatureFromDewpoint(double pressure, double temperature,
                                      double dewpoint, double epsilon,
                                      std::string phase) {
    double mixing_ratio = SaturationMixingRatio(pressure, dewpoint, phase);
    return VirtualTemperature(temperature, mixing_ratio, epsilon);
}
