#ifndef THERMO_HPP // if not defined
#define THERMO_HPP // define the header file

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <utility> // For std::pair
#include <tuple>   // For std::tuple
#include "constants.hpp"

namespace py = pybind11;
namespace mc = metpy_constants;

double MoistAirGasConstant(double specific_humidity);
double MoistAirSpecificHeatPressure(double specific_humidity);

double WaterLatentHeatVaporization(double temperature);
double WaterLatentHeatSublimation(double temperature);

double RelativeHumidityFromDewPoint(double temperature, double dewpoint, std::string phase="liquid");

double DryLapse(double pressure, double ref_temperature, double ref_pressure);
std::vector<double> DryLapseProfile(const std::vector<double>& pressure_profile,
                                    double ref_temperature,
                                    double ref_pressure);
py::array_t<double> DryLapseVectorized(py::array_t<double> pressure,
                                         py::array_t<double> ref_temperature,
                                         double ref_pressure);
py::array_t<double> DryLapseVectorized_3D(py::array_t<double> pressure,
                                         py::array_t<double> ref_temperature,
                                         py::array_t<double> ref_pressure);

double CaldlnTdlnP(double temperature, double pressure);
double MoistLapse(double pressure, double ref_temperature, double ref_pressure, int rk_nstep);
std::vector<double> MoistLapseProfile(const std::vector<double>& press_profile,
                                    double ref_temperature,
                                    double ref_pressure,
                                    int rk_nstep);
py::array_t<double> MoistLapseVectorized(py::array_t<double> pressure,
                                         py::array_t<double> ref_temperature,
                                         double ref_pressure,
                                         int rk_nstep);

std::pair<double, double> LCL(double pressure, double temperature, double dewpoint);
std::tuple<py::array_t<double>, py::array_t<double>> LCLVectorized(py::array_t<double> pressure,
                                                                   py::array_t<double> temperature,
                                                                   py::array_t<double> dewpoint);

bool _CheckPressure(const std::vector<double>& pressure);

// Return struct for _ParcelProfileHelper
struct ParProStruct {
    std::vector<double> press_lower, press_upper;
    double press_lcl;
    std::vector<double> temp_lower, temp_upper;
    double temp_lcl;
};

std::vector<double> ParcelProfile(const std::vector<double>& pressure,
                                  double temperature,
                                  double dewpoint);

ParProStruct _ParcelProfileHelper(const std::vector<double>& pressure, double temperature, double dewpoint);

double _SaturationVaporPressureLiquid(double temperature);
double _SaturationVaporPressureSolid(double temperature);
double SaturationVaporPressure(double temperature, std::string phase);

double DewPoint(double vapor_pressure);
double MixingRatio(double partial_press, double total_press, double epsilon=mc::epsilon);
double SaturationMixingRatio(double total_press, double temperature, std::string phase);
double SpecificHumidityFromMixingRatio(double mixing_ratio);
double SpecificHumidityFromDewPoint(double pressure, double dew_point, std::string phase);

double VirtualTemperature(double temperature, double mixing_ratio, double epsilon);
double VirtualTemperatureFromDewpoint(double pressure, double temperature,
                                      double dewpoint, double epsilon,
                                      std::string phase);
#endif // THERMO_HPP
