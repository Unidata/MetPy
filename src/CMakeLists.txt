cmake_minimum_required(VERSION 3.15...3.26)
project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

# Enforce C++14 or higher
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_calc_mod MODULE
  calcmod.cpp
  constants.cpp
  math.cpp
  thermo.cpp
)

target_compile_definitions(_calc_mod
                           PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _calc_mod DESTINATION metpy)
