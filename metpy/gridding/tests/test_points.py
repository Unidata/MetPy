# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `points` module."""

from __future__ import division

import logging

import numpy as np
from numpy.testing import assert_array_almost_equal

from metpy.gridding.points import (generate_grid, generate_grid_coords, get_boundary_coords,
                                   get_point_count_within_r, get_points_within_r,
                                   get_xy_range, get_xy_steps)

logging.getLogger('metpy.gridding.points').setLevel(logging.ERROR)


def test_get_points_within_r():
    r"""Test get points within a radius function."""
    x = list(range(10))
    y = list(range(10))

    center = [1, 5]

    radius = 5

    matches = get_points_within_r(center, list(zip(x, y)), radius).T

    truth = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]

    assert_array_almost_equal(truth, matches)


def test_get_point_count_within_r():
    r"""Test get point count within a radius function."""
    x = list(range(10))
    y = list(range(10))

    center1 = [1, 5]
    center2 = [12, 10]

    radius = 5

    count = get_point_count_within_r([center1, center2], list(zip(x, y)), radius)

    truth = np.array([5, 2])

    assert_array_almost_equal(truth, count)


def test_get_boundary_coords():
    r"""Test get spatial corners of data positions function."""
    x = list(range(10))
    y = list(range(10))

    bbox = get_boundary_coords(x, y)

    truth = {'east': 9, 'north': 9, 'south': 0, 'west': 0}
    assert bbox == truth

    bbox = get_boundary_coords(x, y, 10)

    truth = {'east': 19, 'north': 19, 'south': -10, 'west': -10}
    assert bbox == truth


def test_get_xy_steps():
    r"""Test get count of grids function."""
    x = list(range(10))
    y = list(range(10))

    bbox = get_boundary_coords(x, y)

    x_steps, y_steps = get_xy_steps(bbox, 3)

    truth_x = 3
    truth_y = 3

    assert x_steps == truth_x
    assert y_steps == truth_y


def test_get_xy_range():
    r"""Test get range of data positions function."""
    x = list(range(10))
    y = list(range(10))

    bbox = get_boundary_coords(x, y)

    x_range, y_range = get_xy_range(bbox)

    truth_x = 9
    truth_y = 9

    assert truth_x == x_range
    assert truth_y == y_range


def test_generate_grid():
    r"""Test generate grid function."""
    x = list(range(10))
    y = list(range(10))

    bbox = get_boundary_coords(x, y)

    gx, gy = generate_grid(3, bbox)

    truth_x = np.array([[0.0, 4.5, 9.0],
                        [0.0, 4.5, 9.0],
                        [0.0, 4.5, 9.0]])

    truth_y = np.array([[0.0, 0.0, 0.0],
                        [4.5, 4.5, 4.5],
                        [9.0, 9.0, 9.0]])

    assert_array_almost_equal(gx, truth_x)
    assert_array_almost_equal(gy, truth_y)


def test_generate_grid_coords():
    r"""Test generate grid coordinates function."""
    x = list(range(10))
    y = list(range(10))

    bbox = get_boundary_coords(x, y)

    gx, gy = generate_grid(3, bbox)

    truth = [[0.0, 0.0],
             [4.5, 0.0],
             [9.0, 0.0],
             [0.0, 4.5],
             [4.5, 4.5],
             [9.0, 4.5],
             [0.0, 9.0],
             [4.5, 9.0],
             [9.0, 9.0]]

    pts = generate_grid_coords(gx, gy)

    assert_array_almost_equal(truth, pts)
