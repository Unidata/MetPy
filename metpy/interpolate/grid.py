# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools and calculations for interpolating specifically to a grid."""

from __future__ import division

import numpy as np

from .points import (interpolate_to_points, inverse_distance_to_points,
                     natural_neighbor_to_points)
from ..deprecation import deprecated
from ..package_tools import Exporter

exporter = Exporter(globals())


def generate_grid(horiz_dim, bbox):
    r"""Generate a meshgrid based on bounding box and x & y resolution.

    Parameters
    ----------
    horiz_dim: integer
        Horizontal resolution
    bbox: dictionary
        Dictionary containing coordinates for corners of study area.

    Returns
    -------
    grid_x: (X, Y) ndarray
        X dimension meshgrid defined by given bounding box
    grid_y: (X, Y) ndarray
        Y dimension meshgrid defined by given bounding box

    """
    x_steps, y_steps = get_xy_steps(bbox, horiz_dim)

    grid_x = np.linspace(bbox['west'], bbox['east'], x_steps)
    grid_y = np.linspace(bbox['south'], bbox['north'], y_steps)

    gx, gy = np.meshgrid(grid_x, grid_y)

    return gx, gy


def generate_grid_coords(gx, gy):
    r"""Calculate x,y coordinates of each grid cell.

    Parameters
    ----------
    gx: numeric
        x coordinates in meshgrid
    gy: numeric
        y coordinates in meshgrid

    Returns
    -------
    (X, Y) ndarray
        List of coordinates in meshgrid

    """
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def get_xy_range(bbox):
    r"""Return x and y ranges in meters based on bounding box.

    bbox: dictionary
        dictionary containing coordinates for corners of study area

    Returns
    -------
    x_range: float
        Range in meters in x dimension.
    y_range: float
        Range in meters in y dimension.

    """
    x_range = bbox['east'] - bbox['west']
    y_range = bbox['north'] - bbox['south']

    return x_range, y_range


def get_xy_steps(bbox, h_dim):
    r"""Return meshgrid spacing based on bounding box.

    bbox: dictionary
        Dictionary containing coordinates for corners of study area.
    h_dim: integer
        Horizontal resolution in meters.

    Returns
    -------
    x_steps, (X, ) ndarray
        Number of grids in x dimension.
    y_steps: (Y, ) ndarray
        Number of grids in y dimension.

    """
    x_range, y_range = get_xy_range(bbox)

    x_steps = np.ceil(x_range / h_dim)
    y_steps = np.ceil(y_range / h_dim)

    return int(x_steps), int(y_steps)


def get_boundary_coords(x, y, spatial_pad=0):
    r"""Return bounding box based on given x and y coordinates assuming northern hemisphere.

    x: numeric
        x coordinates.
    y: numeric
        y coordinates.
    spatial_pad: numeric
        Number of meters to add to the x and y dimensions to reduce
        edge effects.

    Returns
    -------
    bbox: dictionary
        dictionary containing coordinates for corners of study area

    """
    west = np.min(x) - spatial_pad
    east = np.max(x) + spatial_pad
    north = np.max(y) + spatial_pad
    south = np.min(y) - spatial_pad

    return {'west': west, 'south': south, 'east': east, 'north': north}


@exporter.export
def natural_neighbor_to_grid(xp, yp, variable, grid_x, grid_y):
    r"""Generate a natural neighbor interpolation of the given points to a regular grid.

    This assigns values to the given grid using the Liang and Hale [Liang2010]_.
    approach.

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations
    yp: (N, ) ndarray
        y-coordinates of observations
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i])
    grid_x: (M, 2) ndarray
        Meshgrid associated with x dimension
    grid_y: (M, 2) ndarray
        Meshgrid associated with y dimension

    Returns
    -------
    img: (M, N) ndarray
        Interpolated values on a 2-dimensional grid

    See Also
    --------
    natural_neighbor_to_points

    """
    # Handle grid-to-points conversion, and use function from `interpolation`
    points_obs = list(zip(xp, yp))
    points_grid = generate_grid_coords(grid_x, grid_y)
    img = natural_neighbor_to_points(points_obs, variable, points_grid)
    return img.reshape(grid_x.shape)


@exporter.export
@deprecated('0.9', addendum=' This function has been renamed natural_neighbor_to_grid.',
            pending=False)
def natural_neighbor(xp, yp, variable, grid_x, grid_y):
    """Wrap natural_neighbor_to_grid for deprecated natural_neighbor function."""
    return natural_neighbor_to_grid(xp, yp, variable, grid_x, grid_y)


natural_neighbor.__doc__ = (natural_neighbor_to_grid.__doc__
                            + '\n    .. deprecated:: 0.9.0\n        Function has been renamed '
                              'to `natural_neighbor_to_grid` and will be removed from MetPy in'
                              ' 0.12.0.')


@exporter.export
def inverse_distance_to_grid(xp, yp, variable, grid_x, grid_y, r, gamma=None, kappa=None,
                             min_neighbors=3, kind='cressman'):
    r"""Generate an inverse distance interpolation of the given points to a regular grid.

    Values are assigned to the given grid using inverse distance weighting based on either
    [Cressman1959]_ or [Barnes1964]_. The Barnes implementation used here based on [Koch1983]_.

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations.
    yp: (N, ) ndarray
        y-coordinates of observations.
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i]).
    grid_x: (M, 2) ndarray
        Meshgrid associated with x dimension.
    grid_y: (M, 2) ndarray
        Meshgrid associated with y dimension.
    r: float
        Radius from grid center, within which observations
        are considered and weighted.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default None.
    kappa: float
        Response parameter for barnes interpolation. Default None.
    min_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation
        for a point. Default is 3.
    kind: str
        Specify what inverse distance weighting interpolation to use.
        Options: 'cressman' or 'barnes'. Default 'cressman'

    Returns
    -------
    img: (M, N) ndarray
        Interpolated values on a 2-dimensional grid

    See Also
    --------
    inverse_distance_to_points

    """
    # Handle grid-to-points conversion, and use function from `interpolation`
    points_obs = list(zip(xp, yp))
    points_grid = generate_grid_coords(grid_x, grid_y)
    img = inverse_distance_to_points(points_obs, variable, points_grid, r, gamma=gamma,
                                     kappa=kappa, min_neighbors=min_neighbors, kind=kind)
    return img.reshape(grid_x.shape)


@exporter.export
@deprecated('0.9', addendum=' This function has been renamed inverse_distance_to_grid.',
            pending=False)
def inverse_distance(xp, yp, variable, grid_x, grid_y, r, gamma=None, kappa=None,
                     min_neighbors=3, kind='cressman'):
    """Wrap inverse_distance_to_grid for deprecated inverse_distance function."""
    return inverse_distance_to_grid(xp, yp, variable, grid_x, grid_y, r, gamma=gamma,
                                    kappa=kappa, min_neighbors=min_neighbors, kind=kind)


inverse_distance.__doc__ = (inverse_distance_to_grid.__doc__
                            + '\n    .. deprecated:: 0.9.0\n        Function has been renamed '
                              'to `inverse_distance_to_grid` and will be removed from MetPy in'
                              ' 0.12.0.')


@exporter.export
def interpolate_to_grid(x, y, z, interp_type='linear', hres=50000,
                        minimum_neighbors=3, gamma=0.25, kappa_star=5.052,
                        search_radius=None, rbf_func='linear', rbf_smooth=0,
                        boundary_coords=None):
    r"""Interpolate given (x,y), observation (z) pairs to a grid based on given parameters.

    Parameters
    ----------
    x: array_like
        x coordinate
    y: array_like
        y coordinate
    z: array_like
        observation value
    interp_type: str
        What type of interpolation to use. Available options include:
        1) "linear", "nearest", "cubic", or "rbf" from `scipy.interpolate`.
        2) "natural_neighbor", "barnes", or "cressman" from `metpy.interpolate`.
        Default "linear".
    hres: float
        The horizontal resolution of the generated grid, given in the same units as the
        x and y parameters. Default 50000.
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
        'quintic', and 'thin_plate'. Defualt 'linear'. See `scipy.interpolate.Rbf` for more
        information.
    rbf_smooth: float
        Smoothing value applied to rbf interpolation.  Higher values result in more smoothing.
    boundary_coords: dictionary
        Optional dictionary containing coordinates of the study area boundary. Dictionary
        should be in format: {'west': west, 'south': south, 'east': east, 'north': north}

    Returns
    -------
    grid_x: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the x dimension
    grid_y: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the y dimension ndarray
    img: (M, N) ndarray
        2-dimensional array representing the interpolated values for each grid.

    Notes
    -----
    This function acts as a wrapper for `interpolate_points` to allow it to generate a regular
    grid.

    See Also
    --------
    interpolate_to_points

    """
    # Generate the grid
    if boundary_coords is None:
        boundary_coords = get_boundary_coords(x, y)
    grid_x, grid_y = generate_grid(hres, boundary_coords)

    # Handle grid-to-points conversion, and use function from `interpolation`
    points_obs = np.array(list(zip(x, y)))
    points_grid = generate_grid_coords(grid_x, grid_y)
    img = interpolate_to_points(points_obs, z, points_grid, interp_type=interp_type,
                                minimum_neighbors=minimum_neighbors, gamma=gamma,
                                kappa_star=kappa_star, search_radius=search_radius,
                                rbf_func=rbf_func, rbf_smooth=rbf_smooth)

    return grid_x, grid_y, img.reshape(grid_x.shape)


@exporter.export
@deprecated('0.9', addendum=' This function has been renamed interpolate_to_grid.',
            pending=False)
def interpolate(x, y, z, interp_type='linear', hres=50000,
                minimum_neighbors=3, gamma=0.25, kappa_star=5.052,
                search_radius=None, rbf_func='linear', rbf_smooth=0,
                boundary_coords=None):
    """Wrap interpolate_to_grid for deprecated interpolate function."""
    return interpolate_to_grid(x, y, z, interp_type=interp_type, hres=hres,
                               minimum_neighbors=minimum_neighbors, gamma=gamma,
                               kappa_star=kappa_star, search_radius=search_radius,
                               rbf_func=rbf_func, rbf_smooth=rbf_smooth,
                               boundary_coords=boundary_coords)


interpolate.__doc__ = (interpolate_to_grid.__doc__
                       + '\n    .. deprecated:: 0.9.0\n        Function has been renamed to '
                         '`interpolate_to_grid` and will be removed from MetPy in 0.12.0.')
