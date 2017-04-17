# Copyright (c) 2008-2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains a collection of generally useful calculation tools."""

import functools

import numpy as np
import numpy.ma as ma
from scipy.spatial import cKDTree

from ..package_tools import Exporter

exporter = Exporter(globals())


@exporter.export
def resample_nn_1d(a, centers):
    """Return one-dimensional nearest-neighbor indexes based on user-specified centers.

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
    """Determine the index of the point just before two lines with common x values.

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
def find_intersections(x, a, b, direction='all'):
    """Calculate the best estimate of intersection.

    Calculates the best estimates of the intersection of two y-value
    data sets that share a common x-value set.

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric x-values
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2
    direction : string
        specifies direction of crossing. 'all', 'increasing' (a becoming greater than b),
        or 'decreasing' (b becoming greater than a).

    Returns
    -------
        A tuple (x, y) of array-like with the x and y coordinates of the
        intersections of the lines.
    """
    # Find the index of the points just before the intersection(s)
    nearest_idx = nearest_intersection_idx(a, b)
    next_idx = nearest_idx + 1

    # Determine the sign of the change
    sign_change = np.sign(a[next_idx] - b[next_idx])

    # x-values around each intersection
    _, x0 = _next_non_masked_element(x, nearest_idx)
    _, x1 = _next_non_masked_element(x, next_idx)

    # y-values around each intersection for the first line
    _, a0 = _next_non_masked_element(a, nearest_idx)
    _, a1 = _next_non_masked_element(a, next_idx)

    # y-values around each intersection for the second line
    _, b0 = _next_non_masked_element(b, nearest_idx)
    _, b1 = _next_non_masked_element(b, next_idx)

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

    # Make a mask based on the direction of sign change desired
    if direction == 'increasing':
        mask = sign_change > 0
    elif direction == 'decreasing':
        mask = sign_change < 0
    elif direction == 'all':
        return intersect_x, intersect_y
    else:
        raise ValueError('Unknown option for direction: {0}'.format(str(direction)))
    return intersect_x[mask], intersect_y[mask]


@exporter.export
def interpolate_nans(x, y, kind='linear'):
    """Interpolate NaN values in y.

    Interpolate NaN values in the y dimension. Works with unsorted x values.

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric x-values
    y : array-like
        1-dimensional array of numeric y-values
    kind : string
        specifies the kind of interpolation x coordinate - 'linear' or 'log'

    Returns
    -------
        An array of the y coordinate data with NaN values interpolated.
    """
    x_sort_args = np.argsort(x)
    x = x[x_sort_args]
    y = y[x_sort_args]
    nans = np.isnan(y)
    if kind is 'linear':
        y[nans] = np.interp(x[nans], x[~nans], y[~nans])
    elif kind is 'log':
        y[nans] = np.interp(np.log(x[nans]), np.log(x[~nans]), y[~nans])
    else:
        raise ValueError('Unknown option for kind: {0}'.format(str(kind)))
    return y[x_sort_args]


def _next_non_masked_element(a, idx):
    """Return the next non masked element of a masked array.

    If an array is masked, return the next non-masked element (if the given index is masked).
    If no other unmasked points are after the given masked point, returns none.

    Parameters
    ----------
    a : array-like
        1-dimensional array of numeric values
    idx : integer
        index of requested element

    Returns
    -------
        Index of next non-masked element and next non-masked element
    """
    try:
        next_idx = idx + a[idx:].mask.argmin()
        if ma.is_masked(a[next_idx]):
            return None, None
        else:
            return next_idx, a[next_idx]
    except (AttributeError, TypeError, IndexError):
        return idx, a[idx]


def delete_masked_points(*arrs):
    """Delete masked points from arrays.

    Takes arrays and removes masked points to help with calculations and plotting.

    Parameters
    ----------
    arrs : one or more array-like
        source arrays

    Returns
    -------
    arrs : one or more array-like
        arrays with masked elements removed

    """
    if any(hasattr(a, 'mask') for a in arrs):
        keep = ~functools.reduce(np.logical_or, (np.ma.getmaskarray(a) for a in arrs))
        return tuple(ma.asarray(a[keep]) for a in arrs)
    else:
        return arrs


@exporter.export
def reduce_point_density(points, radius, priority=None):
    r"""Return a mask to reduce the density of points in irregularly-spaced data.

    This function is used to down-sample a collection of scattered points (e.g. surface
    data), returning a mask that can be used to select the points from one or more arrays
    (e.g. arrays of temperature and dew point). The points selected can be controlled by
    providing an array of ``priority`` values (e.g. rainfall totals to ensure that
    stations with higher precipitation remain in the mask).

    Parameters
    ----------
    points : (N, K) array-like
        N locations of the points in K dimensional space
    radius : float
        minimum radius allowed between points
    priority : (N, K) array-like, optional
        If given, this should have the same shape as ``points``; these values will
        be used to control selection priority for points.

    Returns
    -------
        (N,) array-like of boolean values indicating whether points should be kept. This
        can be used directly to index numpy arrays to return only the desired points.

    Examples
    --------
    >>> metpy.calc.reduce_point_density(np.array([1, 2, 3]), 1.)
    array([ True, False,  True], dtype=bool)
    >>> metpy.calc.reduce_point_density(np.array([1, 2, 3]), 1.,
    ... priority=np.array([0.1, 0.9, 0.3]))
    array([False,  True, False], dtype=bool)
    """
    # Handle 1D input
    if points.ndim < 2:
        points = points.reshape(-1, 1)

    # Make a kd-tree to speed searching of data.
    tree = cKDTree(points)

    # Need to use sorted indices rather than sorting the position
    # so that the keep mask matches *original* order.
    if priority is not None:
        # Need to sort the locations in decreasing priority.
        sorted_indices = np.argsort(priority)[::-1]
    else:
        # Take advantage of iterator nature of range here to avoid making big lists
        sorted_indices = range(len(points))

    # Keep all points initially
    keep = np.ones(len(points), dtype=np.bool)

    # Loop over all the potential points
    for ind in sorted_indices:
        # Only proceed if we haven't already excluded this point
        if keep[ind]:
            # Find the neighbors and eliminate them
            neighbors = tree.query_ball_point(points[ind], radius)
            keep[neighbors] = False

            # We just removed ourselves, so undo that
            keep[ind] = True

    return keep
