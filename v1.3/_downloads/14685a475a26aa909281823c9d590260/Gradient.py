# Copyright (c) 2015-2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
========
Gradient
========

Use `metpy.calc.gradient`.

This example demonstrates the various ways that MetPy's gradient function
can be utilized.
"""

import numpy as np

import metpy.calc as mpcalc
from metpy.units import units

###########################################
# Create some test data to use for our example
data = np.array([[23, 24, 23],
                 [25, 26, 25],
                 [27, 28, 27],
                 [24, 25, 24]]) * units.degC

# Create an array of x position data (the coordinates of our temperature data)
x = np.array([[1, 2, 3],
              [1, 2, 3],
              [1, 2, 3],
              [1, 2, 3]]) * units.kilometer

y = np.array([[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3],
              [4, 4, 4]]) * units.kilometer

###########################################
# Calculate the gradient using the coordinates of the data
grad = mpcalc.gradient(data, coordinates=(y, x))
print('Gradient in y direction: ', grad[0])
print('Gradient in x direction: ', grad[1])

###########################################
# It's also possible that we do not have the position of data points, but know
# that they are evenly spaced. We can then specify a scalar delta value for each
# axes.
x_delta = 2 * units.km
y_delta = 1 * units.km
grad = mpcalc.gradient(data, deltas=(y_delta, x_delta))
print('Gradient in y direction: ', grad[0])
print('Gradient in x direction: ', grad[1])

###########################################
# Finally, the deltas can be arrays for unevenly spaced data.
x_deltas = np.array([[2, 3],
                     [1, 3],
                     [2, 3],
                     [1, 2]]) * units.kilometer
y_deltas = np.array([[2, 3, 1],
                     [1, 3, 2],
                     [2, 3, 1]]) * units.kilometer
grad = mpcalc.gradient(data, deltas=(y_deltas, x_deltas))
print('Gradient in y direction: ', grad[0])
print('Gradient in x direction: ', grad[1])
