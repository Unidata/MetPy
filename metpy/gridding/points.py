# Copyright (c) 2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for working with points."""

from __future__ import division

import logging

import numpy as np
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


def get_points_within_r(center_points, target_points, r):
    r"""Get all target_points within a specified radius of a center point.

    All data must be in same coordinate system, or you will get undetermined results.

    Parameters
    ----------
    center_points: (X, Y) ndarray
        location from which to grab surrounding points within r
    target_points: (X, Y) ndarray
        points from which to return if they are within r of center_points
    r: integer
        search radius around center_points to grab target_points

    Returns
    -------
    matches: (X, Y) ndarray
        A list of points within r distance of, and in the same
        order as, center_points

    """
    tree = cKDTree(target_points)
    indices = tree.query_ball_point(center_points, r)
    return tree.data[indices].T


def get_point_count_within_r(center_points, target_points, r):
    r"""Get count of target points within a specified radius from center points.

    All data must be in same coordinate system, or you will get undetermined results.

    Parameters
    ----------
    center_points: (X, Y) ndarray
        locations from which to grab surrounding points within r
    target_points: (X, Y) ndarray
        points from which to return if they are within r of center_points
    r: integer
        search radius around center_points to grab target_points

    Returns
    -------
    matches: (N, ) ndarray
        A list of point counts within r distance of, and in the same
        order as, center_points

    """
    tree = cKDTree(target_points)
    indices = tree.query_ball_point(center_points, r)
    return np.array([len(x) for x in indices])


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
    return np.vstack([gx.ravel(), gy.ravel()]).T


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
