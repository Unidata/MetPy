# Copyright (c) 2008,2015,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test data definitions for the `basic` module."""

import numpy as np
from metpy.units import units

_s2 = np.sqrt(2.)

# function_test_data is a nested dictionary of function names with test arguments
# and expected values. It is formatted as follows:
#
# function_test_data = {
#   'function_name': {
#       'argument1': np.arange(5) * units('dimensionless')
#       'argument2': np.arange(5) * units('dimensionless')
#       ...
#       'argumentN': np.arange(5) * units('dimensionless')
#       'return1': np.arange(5, 10) * units('dimensionless')
#       'return2': np.arange(5, 10) * units('dimensionless')
#       ...
#       'returnN': np.arange(5, 10) * units('dimensionless')
#       }
# }
#
# Where argument_i is the name of an argument that matches a parameter for 'function_name',
# and any entries after arguments are assumed to be return values in the correct order.

function_test_data = {
    'wind_speed': {
        'u': np.array([4., 2., 0., 0.]) * units('m/s'),
        'v': np.array([0., 2., 4., 0.]) * units('m/s'),
        'speed': np.array([4., 2 * _s2, 4., 0.]) * units('m/s'),
    },
    'wind_direction': {
        'u': np.array([4., 2., 0., 0.]) * units('m/s'),
        'v': np.array([0., 2., 4., 0.]) * units('m/s'),
        'direc': np.array([270., 225., 180., 0.]) * units.deg,
    },
    'wind_components': {
        'speed': np.array([4, 4, 4, 4, 25, 25, 25, 25, 10.]) * units.mph,
        'wind_direction': np.array([0, 45, 90, 135, 180, 225, 270, 315, 360]) * units.deg,
        'u': np.array([0, -4 / _s2, -4, -4 / _s2, 0, 25 / _s2, 25, 25 / _s2, 0]) * units.mph,
        'v': np.array([-4, -4 / _s2, 0, 4 / _s2, 25, 25 / _s2, 0, -25 / _s2, -10]) * units.mph,
    },
    'windchill': {
        'temperature': np.array([40, -10, -45, 20]) * units.degF,
        'speed': np.array([5, 55, 25, 15]) * units.mph,
        'windchill': np.array([36, -46, -84, 6]) * units.degF,
        'decimal': 0,
    },
    'heat_index': {
        'temperature': np.array([80, 88, 92, 110, 86]) * units.degF,
        'relative_humidity': np.array([40, 100, 70, 40, 88]) * units.percent,
        'heat_index': np.array([80, 121, 112, 136, 104]) * units.degF,
        'decimal': 0,
    },
    'height_to_geopotential': {
        'height': np.array([0, 1000, 2000, 3000]) * units.m,
        'geopotential': np.array([0., 9805, 19607, 29406]) * units('m**2 / second**2'),
        'decimal': 0,
    },
    'geopotential_to_height': {
        'geopotential': np.array([
                0., 9805.11102602, 19607.14506998, 29406.10358006
            ]) * units('m**2 / second**2'),
        'height': np.array([0, 1000, 2000, 3000]) * units.m,
        'decimal': 0,
    },
    'coriolis_parameter': {
        'latitude': np.array([-90., -30., 0., 30., 90.]) * units.degrees,
        'coriolis': np.array([
                -1.4584232E-4, -.72921159E-4, 0, .72921159E-4, 1.4584232E-4
            ]) * units('s^-1'),
    },
    'pressure_to_height_std': {
        'pressure': np.array([975.2, 987.5, 956., 943.]) * units.mbar,
        'height': np.array([321.5, 216.5, 487.6, 601.7]) * units.meter,
        'decimal': 1,
    },
    'height_to_pressure_std': {
        'height': np.array([321.5, 216.5, 487.6, 601.7]) * units.meter,
        'pressure': np.array([975.2, 987.5, 956., 943.]) * units.mbar,
        'decimal': 1,
    },
    # 'sigma_to_pressure': {
    #     'sigma': np.arange(0., 1.1, 0.1),
    #     'pressure_sfc': np.array([1000.]) * units.hPa,
    #     'pressure_top': np.array([0.]) * units.hPa,
    #     'pressure': np.arange(0., 1100., 100.) * units.hPa,
    #     'decimal': 5,
    # }
    'apparent_temperature': {
        'temperature': np.array([[90, 90, 70],
                                 [20, 20, 60]]) * units.degF,
        'relative_humidity': np.array([[60, 20, 60],
                                       [10, 10, 10]]) * units.percent,
        'speed': np.array([[5, 3, 3],
                           [10, 1, 10]]) * units.mph,
        'apparent_temperature': units.Quantity(np.ma.array(
            [[99.6777178, 86.3357671, 70],
             [8.8140662, 20, 60]],
            mask=[[False, False, True], [False, True, True]]), units.degF),
        'decimal': 6,
    },
    'altimeter_to_station_pressure': {
        'altimeter_value': np.array([1054.4, 1054.2, 1054.1, 1054.9, 1054.5, 1013.]) * units.hPa,
        'height': np.array([1236., 1236., 1236., 1513., 1513., 500.]) * units.meter,
        'station_pressure': np.array([910.0, 909.9, 909.8, 880.5, 880.1, 954.6]) * units.hPa,
        'decimal': 0,
    },
    'altimeter_to_sea_level_pressure': {
        'altimeter_value': np.array([1054.4, 1054.2, 1054.1, 1054.9, 1054.5, 1013.]) * units.hPa,
        'height': np.array([1236., 1236., 1236., 1513., 1513., 500.]) * units.meter,
        'temperature': np.zeros(6) * units.degC,
        'sea_level_pressure': np.array([1062.2, 1062.0, 1061.9, 1064.0, 1063.5, 1016.2]) * units.hPa,
        'decimal': 0,
    }
}
