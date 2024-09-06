#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(_calc_mod, m) {
    m.doc() = "accelerator module docstring";
    m.def("add", &add, "Add two numbers");
}