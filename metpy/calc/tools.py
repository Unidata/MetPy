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
    sign_change_idx, = np.nonzero(np.diff(np.sign(difference)))

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
        A tuple (x, y) of array-like with the x and y coordinates of the
        intersections of the lines.
    """

    # Find the index of the points just before the intersection(s)
    nearest_idx = nearest_intersection_idx(a, b)
    next_idx = nearest_idx + 1

    # x-values around each intersection
    x0 = x[nearest_idx]
    x1 = x[next_idx]

    # y-values around each intersection for the first line
    a0 = a[nearest_idx]
    a1 = a[next_idx]

    # y-values around each intersection for the second line
    b0 = b[nearest_idx]
    b1 = b[next_idx]

    # Calculate the x-intersection. This comes from finding the equations of the two lines,
    # one through (x0, a0) and (x1, a1) and the other through (x0, b0) and (x1, b1),
    # finding their intersection, and reducing with a bunch of algebra.
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1
    intersect_x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)

    # Calculate the y-intersection of the lines. Just plug the x above into the equation
    # for the line through the a points. One could solve for y like x above, but this
    # causes weirder unit behavior and seems a little less good numerically.
    intersect_y = ((intersect_x - x0) / (x1 - x0)) * (a1 - a0) + a0

    return intersect_x, intersect_y
