#ifndef VIRTUAL_TEMPERATURE_HPP // if not defined
#define VIRTUAL_TEMPERATURE_HPP // define the header file

#include <vector>

// Compute virtual temperature: vector + vector
std::vector<double> VirtualTemperature(
    const std::vector<double>& temperature,
    const std::vector<double>& mixing_ratio,
    double epsilon = 0.622);

// Overload: scalar + vector
std::vector<double> VirtualTemperature(
    double temperature,
    const std::vector<double>& mixing_ratio,
    double epsilon = 0.622);

// Overload: vector + scalar
std::vector<double> VirtualTemperature(
    const std::vector<double>& temperature,
    double mixing_ratio,
    double epsilon = 0.622);

#endif // VIRTUAL_TEMPERATURE_HPP
