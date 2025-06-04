#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace metpy_constants {
    extern double sat_pressure_0c;

    void load_constants_from_python();  // call once in your PYBIND11_MODULE
}

#endif

