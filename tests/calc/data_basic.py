# Copyright (c) 2008,2015,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test data definitions for the `basic` module."""

import numpy as np
from metpy.units import units

_s2 = np.sqrt(2.)

# Each list element in function_test_data is a tuple of function name (string)
# followed by a dictionary with the function name (string), test input values
# (numpy arrays with units), and expected truth values (numpy arrays with units) e.g.,
#
# [
#     ('function_name', {
#         'name': 'function_name',
#         'values': [np.array([0.])],
#         'truth': [np.array([0.])],
#         }
#     )
# ]
#
# Additionally, the optional integer parameter 'decimal' can be specified, which tells
# the testing functions to compare the output and truth at a certain precision.
# This would look like
#
# [
#     ('function_name', {
#         'name': 'function_name',
#         'values': [np.array([0.])],
#         'truth': [np.array([0.])],
#         'decimal': 1,
#         }
#     )
# ]
#
# The default for decimal is 4.
#
#
# For an example of how this list of tuples is being parsed, see
# https://docs.pytest.org/en/stable/example/parametrize.html#a-quick-port-of-testscenarios


function_test_data = [
    ('wind_speed', {
        'name': 'wind_speed',
        'values': [
            np.array([4., 2., 0., 0.]) * units('m/s'),
            np.array([0., 2., 4., 0.]) * units('m/s'),
            ],
        'truth': [np.array([4., 2 * _s2, 4., 0.]) * units('m/s')],
        },
    ),
    ('wind_direction', {
        'name': 'wind_direction',
        'values': [
            np.array([4., 2., 0., 0.]) * units('m/s'),
            np.array([0., 2., 4., 0.]) * units('m/s'),
            ],
        'truth': [np.array([270., 225., 180., 0.]) * units.deg],
        },
    ),
    ('wind_components', {
        'name': 'wind_components',
        'values': [
            np.array([4, 4, 4, 4, 25, 25, 25, 25, 10.]) * units.mph,
            np.array([0, 45, 90, 135, 180, 225, 270, 315, 360]) * units.deg,
            ],
        'truth': [
            np.array([0, -4 / _s2, -4, -4 / _s2, 0, 25 / _s2, 25, 25 / _s2, 0]) * units.mph,
            np.array([-4, -4 / _s2, 0, 4 / _s2, 25, 25 / _s2, 0, -25 / _s2, -10]) * units.mph,
            ],
        },
    ),
    ('windchill', {
        'name': 'windchill',
        'values': [
            np.array([40, -10, -45, 20]) * units.degF,
            np.array([5, 55, 25, 15]) * units.mph,
            ],
        'truth': [np.array([36, -46, -84, 6]) * units.degF],
        'decimal': 0,
        },
    ),
    ('heat_index', {
        'name': 'heat_index',
        'values': [
            np.array([80, 88, 92, 110, 86]) * units.degF,
            np.array([40, 100, 70, 40, 88]) * units.percent,
            ],
        'truth': [np.array([80, 121, 112, 136, 104]) * units.degF],
        'decimal': 0,
        },
    ),
    ('height_to_geopotential', {
        'name': 'height_to_geopotential',
        'values': [np.array([0, 1000, 2000, 3000]) * units.m],
        'truth': [np.array([0., 9805, 19607, 29406]) * units('m**2 / second**2')],
        'decimal': 0,
        },
    ),
    ('geopotential_to_height', {
        'name': 'geopotential_to_height',
        'values': [np.array([0., 9805.11102602, 19607.14506998, 29406.10358006]) * units('m**2 / second**2')],
        'truth': [np.array([0, 1000, 2000, 3000]) * units.m],
        'decimal': 0,
        },
    ),
    ('coriolis_parameter', {
        'name': 'coriolis_parameter',
        'values': [np.array([-90., -30., 0., 30., 90.]) * units.degrees],
        'truth': [np.array([-1.4584232E-4, -.72921159E-4, 0, .72921159E-4, 1.4584232E-4]) * units('s^-1')],
        'decimal': 7,
        },
    ),
    ('pressure_to_height_std', {
        'name': 'pressure_to_height_std',
        'values': [np.array([975.2, 987.5, 956., 943.]) * units.mbar],
        'truth': [np.array([321.5, 216.5, 487.6, 601.7]) * units.meter],
        'decimal': 1,
        },
    ),
    ('height_to_pressure_std', {
        'name': 'height_to_pressure_std',
        'values': [np.array([321.5, 216.5, 487.6, 601.7]) * units.meter],
        'truth': [np.array([975.2, 987.5, 956., 943.]) * units.mbar],
        'decimal': 1,
        },
    ),
    # ('sigma_to_pressure', {
    #     'name': 'sigma_to_pressure',
    #     'values': [
    #         np.arange(0., 1.1, 0.1),
    #         np.array([1000.]) * units.hPa,
    #         np.array([0.]) * units.hPa
    #         ],
    #     'truth': [np.arange(0., 1100., 100.) * units.hPa],
    #     'decimal': 5,
    #     },
    # ),
    ('apparent_temperature', {
        'name': 'apparent_temperature',
        'values': [
            np.array([[90, 90, 70],
                      [20, 20, 60]]) * units.degF,
            np.array([[60, 20, 60],
                      [10, 10, 10]]) * units.percent,
            np.array([[5, 3, 3],
                      [10, 1, 10]]) * units.mph,
            ],
        'truth': [
            units.Quantity(np.ma.array([[99.6777178, 86.3357671, 70], [8.8140662, 20, 60]],
                                           mask=[[False, False, True], [False, True, True]]),
                               units.degF)
            ],
        'decimal': 6,
        },
    ),
    ('altimeter_to_station_pressure', {
        'name': 'altimeter_to_station_pressure',
        'values': [
            np.array([1054.4, 1054.2, 1054.1, 1054.9, 1054.5, 1013.]) * units.hPa,
            np.array([1236., 1236., 1236., 1513., 1513., 500.]) * units.meter,
            ],
        'truth': [np.array([910.0, 909.9, 909.8, 880.5, 880.1, 954.6]) * units.hPa],
        'decimal': 0,
        },
    ),
    ('altimeter_to_sea_level_pressure', {
        'name': 'altimeter_to_sea_level_pressure',
        'values': [
            np.array([1054.4, 1054.2, 1054.1, 1054.9, 1054.5, 1013.]) * units.hPa,
            np.array([1236., 1236., 1236., 1513., 1513., 500.]) * units.meter,
            np.zeros(6) * units.degC,
            ],
        'truth': [np.array([1062.2, 1062.0, 1061.9, 1064.0, 1063.5, 1016.2]) * units.hPa],
        'decimal': 0,
        },
    ),
]
