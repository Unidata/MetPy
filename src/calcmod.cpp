#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "virtual_temperature.hpp"

namespace py = pybind11;

int add(int i, int j) {
    return i - j;
}


PYBIND11_MODULE(_calc_mod, m) {
    m.doc() = "accelerator module docstring";

    m.def("add", &add, "Add two numbers");

    m.def("virtual_temperature",
          py::overload_cast<const std::vector<double>&, const std::vector<double>&, double>(&VirtualTemperature),
          py::arg("temperature"), py::arg("mixing_ratio"), py::arg("epsilon"),
          "Compute virtual temperature from temperature and mixing ratio arrays");

    m.def("virtual_temperature",
          py::overload_cast<double, const std::vector<double>&, double>(&VirtualTemperature),
          py::arg("temperature"), py::arg("mixing_ratio"), py::arg("epsilon"),
          "Compute virtual temperature from scalar temperature and mixing ratio array");

    m.def("virtual_temperature",
          py::overload_cast<const std::vector<double>&, double, double>(&VirtualTemperature),
          py::arg("temperature"), py::arg("mixing_ratio"), py::arg("epsilon"),
          "Compute virtual temperature from temperature array and scalar mixing ratio");
}
