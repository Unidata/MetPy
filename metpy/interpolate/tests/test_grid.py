# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `grid` module."""

from __future__ import division

import logging

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from metpy.cbook import get_test_data
from metpy.deprecation import MetpyDeprecationWarning
from metpy.interpolate.grid import (generate_grid, generate_grid_coords, get_boundary_coords,
                                    get_xy_range, get_xy_steps, interpolate,
                                    interpolate_to_grid, inverse_distance,
                                    inverse_distance_to_grid, natural_neighbor,
                                    natural_neighbor_to_grid)

logging.getLogger('metpy.interpolate.grid').setLevel(logging.ERROR)


@pytest.fixture()
def test_coords():
    r"""Return data locations used for tests in this file."""
    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 58], dtype=float)
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16], dtype=float)

    return x, y


@pytest.fixture()
def test_data():
    r"""Return data used for tests in this file."""
    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 58], dtype=float)
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16], dtype=float)
    z = np.array([0.064, 4.489, 6.241, 0.1, 2.704, 2.809, 9.604, 1.156,
                  0.225, 3.364], dtype=float)

    return x, y, z


@pytest.fixture()
def test_grid():
    r"""Return grid locations used for tests in this file."""
    with get_test_data('interpolation_test_grid.npz') as fobj:
        data = np.load(fobj)
        return data['xg'], data['yg']


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


def test_natural_neighbor_to_grid(test_data, test_grid):
    r"""Test natural neighbor interpolation to grid function."""
    xp, yp, z = test_data
    xg, yg = test_grid

    img = natural_neighbor_to_grid(xp, yp, z, xg, yg)

    with get_test_data('nn_bbox0to100.npz') as fobj:
        truth = np.load(fobj)['img']

    assert_array_almost_equal(truth, img)


def test_natural_neighbor(test_data, test_grid):
    r"""Test deprecated natural neighbor interpolation function."""
    xp, yp, z = test_data
    xg, yg = test_grid

    with pytest.warns(MetpyDeprecationWarning):
        img = natural_neighbor(xp, yp, z, xg, yg)

    with get_test_data('nn_bbox0to100.npz') as fobj:
        truth = np.load(fobj)['img']

    assert_array_almost_equal(truth, img)


interp_methods = ['cressman', 'barnes']


@pytest.mark.parametrize('method', interp_methods)
def test_inverse_distance_to_grid(method, test_data, test_grid):
    r"""Test inverse distance interpolation to grid function."""
    xp, yp, z = test_data
    xg, yg = test_grid

    extra_kw = {}
    if method == 'cressman':
        extra_kw['r'] = 20
        extra_kw['min_neighbors'] = 1
        test_file = 'cressman_r20_mn1.npz'
    elif method == 'barnes':
        extra_kw['r'] = 40
        extra_kw['kappa'] = 100
        test_file = 'barnes_r40_k100.npz'

    img = inverse_distance_to_grid(xp, yp, z, xg, yg, kind=method, **extra_kw)

    with get_test_data(test_file) as fobj:
        truth = np.load(fobj)['img']

    assert_array_almost_equal(truth, img)


@pytest.mark.parametrize('method', interp_methods)
def test_inverse_distance(method, test_data, test_grid):
    r"""Test inverse distance interpolation function."""
    xp, yp, z = test_data
    xg, yg = test_grid

    extra_kw = {}
    if method == 'cressman':
        extra_kw['r'] = 20
        extra_kw['min_neighbors'] = 1
        test_file = 'cressman_r20_mn1.npz'
    elif method == 'barnes':
        extra_kw['r'] = 40
        extra_kw['kappa'] = 100
        test_file = 'barnes_r40_k100.npz'

    with pytest.warns(MetpyDeprecationWarning):
        img = inverse_distance(xp, yp, z, xg, yg, kind=method, **extra_kw)

    with get_test_data(test_file) as fobj:
        truth = np.load(fobj)['img']

    assert_array_almost_equal(truth, img)


interp_methods = ['natural_neighbor', 'cressman', 'barnes',
                  'linear', 'nearest', 'cubic', 'rbf']

boundary_types = [{'west': 80.0, 'south': 140.0, 'east': 980.0, 'north': 980.0},
                  None]


@pytest.mark.parametrize('method', interp_methods)
@pytest.mark.parametrize('boundary_coords', boundary_types)
def test_interpolate_to_grid(method, test_coords, boundary_coords):
    r"""Test main grid interpolation function."""
    xp, yp = test_coords

    xp *= 10
    yp *= 10

    z = np.array([0.064, 4.489, 6.241, 0.1, 2.704, 2.809, 9.604, 1.156,
                  0.225, 3.364])

    extra_kw = {}
    if method == 'cressman':
        extra_kw['search_radius'] = 200
        extra_kw['minimum_neighbors'] = 1
    elif method == 'barnes':
        extra_kw['search_radius'] = 400
        extra_kw['minimum_neighbors'] = 1
        extra_kw['gamma'] = 1

    if boundary_coords is not None:
        extra_kw['boundary_coords'] = boundary_coords

    _, _, img = interpolate_to_grid(xp, yp, z, hres=10, interp_type=method, **extra_kw)

    with get_test_data('{0}_test.npz'.format(method)) as fobj:
        truth = np.load(fobj)['img']

    assert_array_almost_equal(truth, img)


@pytest.mark.parametrize('method', interp_methods)
@pytest.mark.parametrize('boundary_coords', boundary_types)
def test_interpolate(method, test_coords, boundary_coords):
    r"""Test deprecated main interpolate function."""
    xp, yp = test_coords

    xp *= 10
    yp *= 10

    z = np.array([0.064, 4.489, 6.241, 0.1, 2.704, 2.809, 9.604, 1.156,
                  0.225, 3.364])

    extra_kw = {}
    if method == 'cressman':
        extra_kw['search_radius'] = 200
        extra_kw['minimum_neighbors'] = 1
    elif method == 'barnes':
        extra_kw['search_radius'] = 400
        extra_kw['minimum_neighbors'] = 1
        extra_kw['gamma'] = 1

    if boundary_coords is not None:
        extra_kw['boundary_coords'] = boundary_coords

    with pytest.warns(MetpyDeprecationWarning):
        _, _, img = interpolate(xp, yp, z, hres=10, interp_type=method, **extra_kw)

    with get_test_data('{0}_test.npz'.format(method)) as fobj:
        truth = np.load(fobj)['img']

    assert_array_almost_equal(truth, img)
