#ifndef VIRTUAL_TEMPERATURE_HPP // if not defined
#define VIRTUAL_TEMPERATURE_HPP // define the header file

#include <string>
#include <vector>
#include <utility> // For std::pair
#include "constants.hpp"

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

double CaldlnTdlnP(double temperature, double pressure);
double MoistLapse(double pressure, double ref_temperature, double ref_pressure, int nstep);
std::vector<double> MoistLapseProfile(const std::vector<double>& pressure_profile,
                                    double ref_temperature,
                                    double ref_pressure);


std::pair<double, double> LCL(double pressure, double temperature, double dewpoint);

bool _CheckPressure(const std::vector<double>& pressure);

double _SaturationVaporPressureLiquid(double temperature);
double _SaturationVaporPressureSolid(double temperature);
double SaturationVaporPressure(double temperature, std::string phase);

double DewPoint(double vapor_pressure);
double MixingRatio(double partial_press, double total_press, double epsilon=mc::epsilon);
double SaturationMixingRatio(double total_press, double temperature, std::string phase);
double SpecificHumidityFromMixingRatio(double mixing_ratio);
double SpecificHumidityFromDewPoint(double pressure, double dew_point, std::string phase);

double VirtualTemperature(double temperature, double mixing_ratio, double epsilon);

#endif // VIRTUAL_TEMPERATURE_HPP
