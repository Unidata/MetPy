# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division

import numpy as np

from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import cdist

from metpy.gridding import interpolation
from metpy.gridding import points

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


def remove_observations_below_value(x, y, z, val=0):
    r"""Given (x,y) coordinates and an associated observation (z),
    remove all x, y, and z where z is less than val. Will not destroy
    original values.

    Parameters
    ----------
    x: float
        x coordinate.
    y: float
        y coordinate.
    z: float
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


def remove_nan_observations(x, y, z):
    r"""Given (x,y) coordinates and an associated observation (z),
    remove all x, y, and z where z is nan. Will not destroy
    original values.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
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


def remove_repeat_coordinates(x, y, z):
    r"""Given x,y coordinates and an associated observation (z),
    remove all x, y, and z where (x,y) is repeated and keep the
    first occurrence only. Will not destroy original values.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
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


@exporter.export
def interpolate(x, y, z, interp_type='linear', hres=50000,
                buffer=1, minimum_neighbors=3, gamma=0.25,
                kappa_star=5.052, search_radius=None, rbf_func='linear', rbf_smooth=0):
    r"""Interpolate given (x,y), observation (z) pairs to a grid based on given parameters.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
        observation value
    interp_type: str
        What type of interpolation to use. Available options include:
        1) "linear", "nearest", "cubic", or "rbf" from Scipy.interpolate.
        2) "natural_neighbor", "barnes", or "cressman" from Metpy.mapping .
        Default "linear".
    hres: float
        The horizontal resolution of the generated grid. Default 50000 meters.
    buffer: float
        How many meters to add to the bounds of the grid. Default 1000 meters.
    minimum_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation for a
        point. Default is 3.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 0.25.
    kappa_star: float
        Response parameter for barnes interpolation, specified nondimensionally
        in terms of the Nyquist. Default 5.052
    search_radius: float
        A search radius to use for the barnes and cressman interpolation schemes.
        If search_radius is not specified, it will default to the average spacing of
        observations.
    rbf_func: str
        Specifies which function to use for Rbf interpolation.
        Options include: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
        'quintic', and 'thin_plate'. Defualt 'linear'. See scipy.interpolate.Rbf for more
        information.
    rbf_smooth: float
        Smoothing value applied to rbf interpolation.  Higher values result in more smoothing.

    Returns
    -------
    grid_x: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the x dimension
    grid_y: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the y dimension ndarray
    img: (M, N) ndarray
        2-dimensional array representing the interpolated values for each grid.
    """

    grid_x, grid_y = points.generate_grid(hres, points.get_boundary_coords(x, y),
                                          buffer)

    if interp_type in ['linear', 'nearest', 'cubic']:
        points_zip = np.array(list(zip(x, y)))
        img = griddata(points_zip, z, (grid_x, grid_y), method=interp_type)

    elif interp_type == 'natural_neighbor':
        img = interpolation.natural_neighbor(x, y, z, grid_x, grid_y)

    elif interp_type in ['cressman', 'barnes']:

        ave_spacing = np.mean((cdist(list(zip(x, y)), list(zip(x, y)))))

        if search_radius is None:
            search_radius = ave_spacing

        if interp_type == 'cressman':

            img = interpolation.inverse_distance(x, y, z, grid_x, grid_y, search_radius,
                                                 min_neighbors=minimum_neighbors,
                                                 kind=interp_type)

        elif interp_type == 'barnes':

            kappa = calc_kappa(ave_spacing, kappa_star)
            img = interpolation.inverse_distance(x, y, z, grid_x, grid_y, search_radius,
                                                 gamma, kappa, min_neighbors=minimum_neighbors,
                                                 kind=interp_type)

    elif interp_type == 'rbf':

        # 3-dimensional support not yet included.
        # Assign a zero to each z dimension for observations.
        h = np.zeros((len(x)))

        rbfi = Rbf(x, y, h, z, function=rbf_func, smooth=rbf_smooth)

        # 3-dimensional support not yet included.
        # Assign a zero to each z dimension grid cell position.
        hi = np.zeros(grid_x.shape)
        img = rbfi(grid_x, grid_y, hi)

    else:
        raise ValueError('Interpolation option not available. '
                         'Try: linear, nearest, cubic, natural_neighbor, '
                         'barnes, cressman, rbf')

    return grid_x, grid_y, img
