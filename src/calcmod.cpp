#include "virtual_temperature.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Flexible dispatcher that supports scalar/list combinations
std::vector<double> virtual_temperature_py(py::object temperature, py::object mixing_ratio, double epsilon) {
    std::vector<double> temperature_vec;
    std::vector<double> mixing_ratio_vec;

    // Case 1: both are lists
    if (py::isinstance<py::list>(temperature) && py::isinstance<py::list>(mixing_ratio)) {
        temperature_vec = temperature.cast<std::vector<double>>();
        mixing_ratio_vec = mixing_ratio.cast<std::vector<double>>();
        if (temperature_vec.size() != mixing_ratio_vec.size()) {
            throw std::invalid_argument("Temperature and mixing ratio lists must be the same length.");
        }
    }
    // Case 2: temperature is float, mixing_ratio is list
    else if (py::isinstance<py::float_>(temperature) && py::isinstance<py::list>(mixing_ratio)) {
        mixing_ratio_vec = mixing_ratio.cast<std::vector<double>>();
        temperature_vec = std::vector<double>(mixing_ratio_vec.size(), temperature.cast<double>());
    }
    // Case 3: temperature is list, mixing_ratio is float
    else if (py::isinstance<py::list>(temperature) && py::isinstance<py::float_>(mixing_ratio)) {
        temperature_vec = temperature.cast<std::vector<double>>();
        mixing_ratio_vec = std::vector<double>(temperature_vec.size(), mixing_ratio.cast<double>());
    }
    // Case 4: both are floats
    else if (py::isinstance<py::float_>(temperature) && py::isinstance<py::float_>(mixing_ratio)) {
        temperature_vec = {temperature.cast<double>()};
        mixing_ratio_vec = {mixing_ratio.cast<double>()};
    }
    else {
        throw std::invalid_argument("Inputs must be float or list.");
    }

    return VirtualTemperature(temperature_vec, mixing_ratio_vec, epsilon);
}

int add(int i, int j) {
    return i - j;
}

PYBIND11_MODULE(_calc_mod, m) {
    m.doc() = "accelerator module docstring";

    m.def("add", &add, "Add two numbers");

    // Unified binding with default epsilon
    m.def("virtual_temperature", &virtual_temperature_py,
          py::arg("temperature"), py::arg("mixing_ratio"), py::arg("epsilon") = 0.622,
          "Compute virtual temperature.\n"
          "Accepts:\n"
          " - two lists of equal length\n"
          " - one scalar and one list\n"
          "Defaults to epsilon = 0.622");
}
