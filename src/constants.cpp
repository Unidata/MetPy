// constants.cpp
#include "constants.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace metpy_constants {
    double Mw;
    double Rv;
    double Cp_v;
    double Cp_l;
    double Lv;
    double sat_pressure_0c;
    double T0;

    void load_constants_from_python() {
        py::object mod = py::module_::import("metpy.constants.default");

        Mw = mod.attr("Mw").attr("to")("kg / mol").attr("magnitude").cast<double>();
        Rv = mod.attr("Rv").attr("magnitude").cast<double>();
        Cp_v = mod.attr("Cp_v").attr("magnitude").cast<double>();
        Cp_l = mod.attr("Cp_l").attr("magnitude").cast<double>();
        Lv = mod.attr("Lv").attr("magnitude").cast<double>(); 
        sat_pressure_0c = mod.attr("sat_pressure_0c").attr("magnitude").cast<double>();
        T0 = mod.attr("T0").attr("magnitude").cast<double>();
    }
}
