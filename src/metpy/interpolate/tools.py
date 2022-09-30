# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Assorted tools in support of interpolation functionality."""

import numpy as np
from scipy.spatial.distance import cdist

from ..package_tools import Exporter

exporter = Exporter(globals())


def calc_kappa(spacing, kappa_star=5.052):
    r"""Calculate the kappa parameter for barnes interpolation.

    Parameters
    ----------
    spacing: float
        Average spacing between observations
    kappa_star: float
        Non-dimensional response parameter. Default 5.052.

    Returns
    -------
        kappa: float

    """
    return kappa_star * (2.0 * spacing / np.pi)**2


def average_spacing(points):
    """Calculate the average spacing to the nearest other point.

    Parameters
    ----------
    points : (M, N) array-like
         M points in N dimensional space

    Returns
    -------
        The average distance to the nearest neighbor across all points

    """
    dist_matrix = cdist(points, points)
    diag = np.arange(len(dist_matrix))
    dist_matrix[diag, diag] = np.nan
    return np.nanmin(dist_matrix, axis=0).mean()


@exporter.export
def remove_observations_below_value(x, y, z, val=0):
    r"""Remove all x, y, and z where z is less than val.

    Will not destroy original values.

    Parameters
    ----------
    x: array-like
        x coordinate.
    y: array-like
        y coordinate.
    z: array-like
        Observation value.
    val: float
        Value at which to threshold z.

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        observation values less than val.

    """
    x_ = x[z >= val]
    y_ = y[z >= val]
    z_ = z[z >= val]

    return x_, y_, z_


@exporter.export
def remove_nan_observations(x, y, z):
    r"""Remove all x, y, and z where z is nan.

    Will not destroy original values.

    Parameters
    ----------
    x: array-like
        x coordinate
    y: array-like
        y coordinate
    z: array-like
        observation value

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        nan valued observations.

    """
    x_ = x[~np.isnan(z)]
    y_ = y[~np.isnan(z)]
    z_ = z[~np.isnan(z)]

    return x_, y_, z_


@exporter.export
def remove_repeat_coordinates(x, y, z):
    r"""Remove all x, y, and z where (x,y) is repeated and keep the first occurrence only.

    Will not destroy original values.

    Parameters
    ----------
    x: array-like
        x coordinate
    y: array-like
        y coordinate
    z: array-like
        observation value

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        repeated coordinates.

    """
    coords = []
    variable = []

    for (x_, y_, t_) in zip(x, y, z):
        if (x_, y_) not in coords:
            coords.append((x_, y_))
            variable.append(t_)

    coords = np.array(coords)

    x_ = coords[:, 0]
    y_ = coords[:, 1]

    z_ = np.array(variable)

    return x_, y_, z_


def barnes_weights(sq_dist, kappa, gamma):
    r"""Calculate the Barnes weights from squared distance values.

    Parameters
    ----------
    sq_dist: (N, ) numpy.ndarray
        Squared distances from interpolation point
        associated with each observation in meters.
    kappa: float
        Response parameter for barnes interpolation. Default None.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default None.

    Returns
    -------
    weights: (N, ) numpy.ndarray
        Calculated weights for the given observations determined by their distance
        to the interpolation point.

    """
    return np.exp(-1.0 * sq_dist / (kappa * gamma))


def cressman_weights(sq_dist, r):
    r"""Calculate the Cressman weights from squared distance values.

    Parameters
    ----------
    sq_dist: (N, ) numpy.ndarray
        Squared distances from interpolation point associated with each observation in meters.
    r: float
        Maximum distance an observation can be from an interpolation point to be considered
        in the interpolation calculation.

    Returns
    -------
    weights: (N, ) numpy.ndarray
        Calculated weights for the given observations determined by their distance
        to the interpolation point.

    """
    return (r**2 - sq_dist) / (r**2 + sq_dist)
