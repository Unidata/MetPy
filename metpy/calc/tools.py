# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ..package_tools import Exporter

exporter = Exporter(globals())


@exporter.export
def resample_nn_1d(a, centers):
    """Helper function that returns one-dimensional nearest-neighbor
    indexes based on user-specified centers.

    Parameters
    ----------
    a : array-like
        1-dimensional array of numeric values from which to
        extract indexes of nearest-neighbors
    centers : array-like
        1-dimensional array of numeric values representing a subset of values to approximate

    Returns
    -------
        An array of indexes representing values closest to given array values

    """
    ix = []
    for center in centers:
        index = (np.abs(a - center)).argmin()
        if index not in ix:
            ix.append(index)
    return ix


@exporter.export
def nearest_intersection_idx(a, b):
    """
    Determines the index of the point just before two lines with
    common x values.

    Parameters
    ----------
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2

    Returns
    -------
        An array of indexes representing the index of the values
        just before the intersection(s) of the two lines.
    """
    # Difference in the two y-value sets
    difference = a - b

    # Determine the point just before the intersection of the lines
    # Will return multiple points for multiple intersections
    sign_change_idx = np.where(np.diff(np.sign(difference)))[0]

    return sign_change_idx


@exporter.export
def find_intersections(x, a, b):

    """
    Calculates the best estimate of the intersection(s) of two y-value
    data sets that share a common x-value set.

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric x-values
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2

    Returns
    -------
        An array of (x,y) intersections of the lines.
    """

    # Find the index of the points just before the intersection(s)
    nearest_idx = nearest_intersection_idx(a, b)

    # Make an empty array to hold what we'll need for calculating
    # approximate intersections. Each row is an intersection

    # x1, x2, ya1, ya2, yb1, yb2
    line_data = np.zeros((np.size(nearest_idx), 6))

    # m1, m2, b1, b2
    line_fits = np.zeros((np.size(nearest_idx), 4))

    # xi, yi
    intersections = np.zeros((np.size(nearest_idx), 2))

    # Place x values for each intersection
    line_data[:, 0] = x[nearest_idx]
    line_data[:, 1] = x[nearest_idx + 1]

    # Place y values for the first line
    line_data[:, 2] = a[nearest_idx]
    line_data[:, 3] = a[nearest_idx + 1]

    # Place y values for the second line
    line_data[:, 4] = b[nearest_idx]
    line_data[:, 5] = b[nearest_idx + 1]

    # Calculate the slope of each line (delta y / delta x)
    line_fits[:, 0] = ((line_data[:, 3] - line_data[:, 2]) /
                       (line_data[:, 1] - line_data[:, 0]))

    line_fits[:, 1] = ((line_data[:, 5] - line_data[:, 4]) /
                       (line_data[:, 1] - line_data[:, 0]))

    # Calculate the intercept of each line
    # b = y - m * x
    line_fits[:, 2] = line_data[:, 2] - line_fits[:, 0] * line_data[:, 0]

    line_fits[:, 3] = line_data[:, 4] - line_fits[:, 1] * line_data[:, 0]

    # Calculate the x-intersection of the lines
    # (b2 - b1) / (m1 - m2)
    intersections[:, 0] = ((line_fits[:, 3] - line_fits[:, 2]) /
                           (line_fits[:, 0] - line_fits[:, 1]))

    # Calculate the y-intersection of the lines
    # y = m * x + b
    intersections[:, 1] = line_fits[:, 0] * intersections[:, 0] + line_fits[:, 2]

    return intersections
