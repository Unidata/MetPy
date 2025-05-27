#include "virtual_temperature.hpp"
//#include <stdexcept>

std::vector<double> VirtualTemperature(
    const std::vector<double>& temperature,
    const std::vector<double>& mixing_ratio,
    double epsilon
) {

    std::vector<double> result(temperature.size());
    double T, w;
    for (size_t i = 0; i < temperature.size(); ++i) {
        T = temperature[i];
        w = mixing_ratio[i];
        result[i] = T * (w + epsilon) / (epsilon * (1 + w));
    }

    return result;
}
