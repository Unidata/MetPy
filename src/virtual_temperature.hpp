#ifndef VIRTUAL_TEMPERATURE_HPP // if not defined
#define VIRTUAL_TEMPERATURE_HPP // define the header file

#include <string>
#include <utility> // For std::pair
#include "constants.hpp"

namespace mc = metpy_constants;

double MoistAirGasConstant(double specific_humidity);
double MoistAirSpecificHeatPressure(double specific_humidity);

double WaterLatentHeatVaporization(double temperature);
double WaterLatentHeatSublimation(double temperature);

double RelativeHumidityFromDewPoint(double temperature, double dewpoint, std::string phase="liquid");

std::pair<double, double> LCL(double pressure, double temperature, double dewpoint);

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
