# Copyright (c) 2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools and calculations for interpolating specifically to a grid."""

import numpy as np

from .points import (interpolate_to_points, inverse_distance_to_points,
                     natural_neighbor_to_points)
from ..package_tools import Exporter
from ..pandas import preprocess_pandas

exporter = Exporter(globals())


def generate_grid(horiz_dim, bbox):
    r"""Generate a meshgrid based on bounding box and x & y resolution.

    Parameters
    ----------
    horiz_dim: int or float
        Horizontal resolution
    bbox: dict
        Dictionary with keys 'east', 'west', 'north', 'south' with the box extents
        in those directions.

    Returns
    -------
    grid_x: (X, Y) numpy.ndarray
        X dimension meshgrid defined by given bounding box
    grid_y: (X, Y) numpy.ndarray
        Y dimension meshgrid defined by given bounding box

    """
    x_steps, y_steps = get_xy_steps(bbox, horiz_dim)

    grid_x = np.linspace(bbox['west'], bbox['east'], x_steps)
    grid_y = np.linspace(bbox['south'], bbox['north'], y_steps)

    return np.meshgrid(grid_x, grid_y)


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
    (X, Y) numpy.ndarray
        List of coordinates in meshgrid

    """
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def get_xy_range(bbox):
    r"""Return x and y ranges in meters based on bounding box.

    bbox: dict
        Dictionary with keys 'east', 'west', 'north', 'south' with the box extents
        in those directions.

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

    bbox: dict
        Dictionary with keys 'east', 'west', 'north', 'south' with the box extents
        in those directions.
    h_dim: int or float
        Horizontal resolution in meters.

    Returns
    -------
    x_steps, (X, ) numpy.ndarray
        Number of grids in x dimension.
    y_steps: (Y, ) numpy.ndarray
        Number of grids in y dimension.

    """
    x_range, y_range = get_xy_range(bbox)

    x_steps = np.ceil(x_range / h_dim) + 1
    y_steps = np.ceil(y_range / h_dim) + 1

    return int(x_steps), int(y_steps)


def get_boundary_coords(x, y, spatial_pad=0):
    r"""Return bounding box based on given x and y coordinates assuming northern hemisphere.

    x: numeric
        x coordinates.
    y: numeric
        y coordinates.
    spatial_pad: int or float
        Number of meters to add to the x and y dimensions to reduce edge effects.

    Returns
    -------
    bbox: dict
        Dictionary with keys 'east', 'west', 'north', 'south' with the box extents
        in those directions.

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
    xp: (P, ) numpy.ndarray
        x-coordinates of observations
    yp: (P, ) numpy.ndarray
        y-coordinates of observations
    variable: (P, ) numpy.ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i])
    grid_x: (M, N) numpy.ndarray
        Meshgrid associated with x dimension
    grid_y: (M, N) numpy.ndarray
        Meshgrid associated with y dimension

    Returns
    -------
    img: (M, N) numpy.ndarray
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
def inverse_distance_to_grid(xp, yp, variable, grid_x, grid_y, r, gamma=None, kappa=None,
                             min_neighbors=3, kind='cressman'):
    r"""Generate an inverse distance interpolation of the given points to a regular grid.

    Values are assigned to the given grid using inverse distance weighting based on either
    [Cressman1959]_ or [Barnes1964]_. The Barnes implementation used here based on [Koch1983]_.

    Parameters
    ----------
    xp: (N, ) numpy.ndarray
        x-coordinates of observations.
    yp: (N, ) numpy.ndarray
        y-coordinates of observations.
    variable: (N, ) numpy.ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i]).
    grid_x: (M, 2) numpy.ndarray
        Meshgrid associated with x dimension.
    grid_y: (M, 2) numpy.ndarray
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
    img: (M, N) numpy.ndarray
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
@preprocess_pandas
def interpolate_to_grid(x, y, z, interp_type='linear', hres=50000,
                        minimum_neighbors=3, gamma=0.25, kappa_star=5.052,
                        search_radius=None, rbf_func='linear', rbf_smooth=0,
                        boundary_coords=None):
    r"""Interpolate given (x,y), observation (z) pairs to a grid based on given parameters.

    Parameters
    ----------
    x: array-like
        x coordinate, can have units of linear distance or degrees
    y: array-like
        y coordinate, can have units of linear distance or degrees
    z: array-like
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
        Minimum number of neighbors needed to perform Barnes or Cressman interpolation for a
        point. Default is 3.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 0.25.
    kappa_star: float
        Response parameter for barnes interpolation, specified nondimensionally
        in terms of the Nyquist. Default 5.052
    search_radius: float
        A search radius to use for the Barnes and Cressman interpolation schemes.
        If search_radius is not specified, it will default to 5 times the average spacing of
        observations.
    rbf_func: str
        Specifies which function to use for Rbf interpolation.
        Options include: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
        'quintic', and 'thin_plate'. Default 'linear'. See `scipy.interpolate.Rbf` for more
        information.
    rbf_smooth: float
        Smoothing value applied to rbf interpolation.  Higher values result in more smoothing.
    boundary_coords: dict
        Optional dictionary containing coordinates of the study area boundary. Dictionary
        should be in format: {'west': west, 'south': south, 'east': east, 'north': north}

    Returns
    -------
    grid_x: (N, 2) numpy.ndarray
        Meshgrid for the resulting interpolation in the x dimension
    grid_y: (N, 2) numpy.ndarray
        Meshgrid for the resulting interpolation in the y dimension numpy.ndarray
    img: (M, N) numpy.ndarray
        2-dimensional array representing the interpolated values for each grid.

    See Also
    --------
    interpolate_to_points

    Notes
    -----
    This function acts as a wrapper for `interpolate_points` to allow it to generate a regular
    grid.

    This function interpolates points to a Cartesian plane, even if lat/lon coordinates
    are provided.

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
@preprocess_pandas
def interpolate_to_isosurface(level_var, interp_var, level, bottom_up_search=True):
    r"""Linear interpolation of a variable to a given vertical level from given values.

    This function assumes that highest vertical level (lowest pressure) is zeroth index.
    A classic use of this function would be to compute the potential temperature on the
    dynamic tropopause (2 PVU surface).

    Parameters
    ----------
    level_var: array-like (P, M, N)
        Level values in 3D grid on common vertical coordinate (e.g., PV values on
        isobaric levels). Assumes height dimension is highest to lowest in atmosphere.
    interp_var: array-like (P, M, N)
        Variable on 3D grid with same vertical coordinate as level_var to interpolate to
        given level (e.g., potential temperature on isobaric levels)
    level: int or float
        Desired interpolated level (e.g., 2 PVU surface)
    bottom_up_search : bool, optional
        Controls whether to search for levels bottom-up (starting at lower indices),
        or top-down (starting at higher indices). Defaults to True, which is bottom-up search.

    Returns
    -------
    interp_level: (M, N) numpy.ndarray
        The interpolated variable (e.g., potential temperature) on the desired level (e.g.,
        2 PVU surface)

    Notes
    -----
    This function implements a linear interpolation to estimate values on a given surface.
    The prototypical example is interpolation of potential temperature to the dynamic
    tropopause (e.g., 2 PVU surface)

    """
    from ..calc import find_bounding_indices

    # Find index values above and below desired interpolated surface value
    above, below, good = find_bounding_indices(level_var, [level], axis=0,
                                               from_below=bottom_up_search)

    # Linear interpolation of variable to interpolated surface value
    interp_level = (((level - level_var[above]) / (level_var[below] - level_var[above]))
                    * (interp_var[below] - interp_var[above])) + interp_var[above]

    # Handle missing values and instances where no values for surface exist above and below
    interp_level[~good] = np.nan
    minvar = (np.min(level_var, axis=0) >= level)
    maxvar = (np.max(level_var, axis=0) <= level)
    interp_level[0][minvar] = interp_var[-1][minvar]
    interp_level[0][maxvar] = interp_var[0][maxvar]
    return interp_level.squeeze()
