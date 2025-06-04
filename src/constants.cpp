// constants.cpp
#include "constants.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace metpy_constants;

// Definitions (required in C++11)
double sat_pressure_0c;

void load_constants_from_python() {
    py::object mod = py::module_::import("metpy.constants.default");

    sat_pressure_0c = mod.attr("sat_pressure_0c").attr("magnitude").cast<double>();
}

