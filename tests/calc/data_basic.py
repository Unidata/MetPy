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
# See https://docs.pytest.org/en/stable/example/parametrize.html#a-quick-port-of-testscenarios


function_test_data = [
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
    ('wind_speed', {
        'name': 'wind_speed',
        'values': [
            np.array([4., 2., 0., 0.]) * units('m/s'),
            np.array([0., 2., 4., 0.]) * units('m/s'),
            ],
        'truth': [
            np.array([4., 2 * _s2, 4., 0.]) * units('m/s'),
            ],
        },
    ),
    ('pressure_to_heights', {
        'name': 'pressure_to_height_std',
        'values': [np.array([975.2, 987.5, 956., 943.]) * units.mbar],
        'truth': [np.array([321.5, 216.5, 487.6, 601.7]) * units.meter],
        },
    ),
]
