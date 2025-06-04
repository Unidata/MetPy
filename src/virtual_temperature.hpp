#ifndef VIRTUAL_TEMPERATURE_HPP // if not defined
#define VIRTUAL_TEMPERATURE_HPP // define the header file

//#include <vector>
double WaterLatentHeatVaporization(double temperature);

double _SaturationVaporPressureLiquid(double temperature);
double DewPoint(double vapor_pressure);

double VirtualTemperature(double temperature, double mixing_ratio, double epsilon);

#endif // VIRTUAL_TEMPERATURE_HPP
