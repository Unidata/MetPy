#ifndef VIRTUAL_TEMPERATURE_HPP // if not defined
#define VIRTUAL_TEMPERATURE_HPP // define the header file

#include <vector>

// Compute virtual temperature: vector + vector
std::vector<double> VirtualTemperature(
    const std::vector<double>& temperature,
    const std::vector<double>& mixing_ratio,
    double epsilon);

#endif // VIRTUAL_TEMPERATURE_HPP
