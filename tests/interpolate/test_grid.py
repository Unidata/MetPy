# Copyright (c) 2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `grid` module."""


import logging

import numpy as np
import pytest

from metpy.cbook import get_test_data
from metpy.interpolate.grid import (generate_grid, generate_grid_coords, get_boundary_coords,
                                    get_xy_range, get_xy_steps, interpolate_to_grid,
                                    interpolate_to_isosurface, inverse_distance_to_grid,
                                    natural_neighbor_to_grid)
from metpy.testing import assert_array_almost_equal
from metpy.units import units

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

    truth_x = 4
    truth_y = 4

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

    truth_x = np.array([[0.0, 3.0, 6.0, 9.0],
                        [0.0, 3.0, 6.0, 9.0],
                        [0.0, 3.0, 6.0, 9.0],
                        [0.0, 3.0, 6.0, 9.0]])

    truth_y = np.array([[0.0, 0.0, 0.0, 0.0],
                        [3.0, 3.0, 3.0, 3.0],
                        [6.0, 6.0, 6.0, 6.0],
                        [9.0, 9.0, 9.0, 9.0]])

    assert_array_almost_equal(gx, truth_x)
    assert_array_almost_equal(gy, truth_y)


def test_generate_grid_coords():
    r"""Test generate grid coordinates function."""
    x = list(range(10))
    y = list(range(10))

    bbox = get_boundary_coords(x, y)

    gx, gy = generate_grid(3, bbox)

    truth = [[0.0, 0.0],
             [3.0, 0.0],
             [6.0, 0.0],
             [9.0, 0.0],
             [0.0, 3.0],
             [3.0, 3.0],
             [6.0, 3.0],
             [9.0, 3.0],
             [0.0, 6.0],
             [3.0, 6.0],
             [6.0, 6.0],
             [9.0, 6.0],
             [0.0, 9.0],
             [3.0, 9.0],
             [6.0, 9.0],
             [9.0, 9.0]]

    pts = generate_grid_coords(gx, gy)

    assert_array_almost_equal(truth, pts)
    assert pts.flags['C_CONTIGUOUS']  # need output to be C-contiguous


def test_natural_neighbor_to_grid(test_data, test_grid):
    r"""Test natural neighbor interpolation to grid function."""
    xp, yp, z = test_data
    xg, yg = test_grid

    img = natural_neighbor_to_grid(xp, yp, z, xg, yg)

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
    test_file = ''
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


interp_methods = ['natural_neighbor', 'cressman', 'barnes', 'linear', 'nearest', 'rbf',
                  'cubic']
boundary_types = [{'west': 80.0, 'south': 140.0, 'east': 980.0, 'north': 980.0},
                  None]


def test_interpolate_to_isosurface():
    r"""Test interpolation to level function."""
    pv = np.array([[[4.29013406, 4.61736108, 4.97453387, 5.36730237, 5.75500645],
                    [3.48415057, 3.72492697, 4.0065845, 4.35128065, 4.72701041],
                    [2.87775662, 3.01866087, 3.21074864, 3.47971854, 3.79924194],
                    [2.70274738, 2.71627883, 2.7869988, 2.94197238, 3.15685712],
                    [2.81293318, 2.70649941, 2.65188277, 2.68109532, 2.77737801]],
                   [[2.43090597, 2.79248225, 3.16783697, 3.54497301, 3.89481001],
                    [1.61968826, 1.88924405, 2.19296648, 2.54191855, 2.91119712],
                    [1.09089606, 1.25384007, 1.46192044, 1.73476959, 2.05268876],
                    [0.97204726, 1.02016741, 1.11466014, 1.27721014, 1.4912234],
                    [1.07501523, 1.02474621, 1.01290749, 1.0638517, 1.16674712]],
                   [[0.61025484, 0.7315194, 0.85573147, 0.97430123, 1.08453329],
                    [0.31705299, 0.3987999, 0.49178996, 0.59602155, 0.71077394],
                    [0.1819831, 0.22650344, 0.28305811, 0.35654934, 0.44709885],
                    [0.15472957, 0.17382593, 0.20182338, 0.2445138, 0.30252574],
                    [0.15522068, 0.16333457, 0.17633552, 0.19834644, 0.23015555]]])

    theta = np.array([[[344.45776, 344.5063, 344.574, 344.6499, 344.735],
                       [343.98444, 344.02536, 344.08682, 344.16284, 344.2629],
                       [343.58792, 343.60876, 343.65628, 343.72818, 343.82834],
                       [343.21542, 343.2204, 343.25833, 343.32935, 343.43414],
                       [342.85272, 342.84982, 342.88556, 342.95645, 343.0634]],
                      [[326.70923, 326.67603, 326.63416, 326.57153, 326.49155],
                       [326.77695, 326.73468, 326.6931, 326.6408, 326.58405],
                       [326.95062, 326.88986, 326.83627, 326.78134, 326.7308],
                       [327.1913, 327.10928, 327.03894, 326.97546, 326.92587],
                       [327.47235, 327.3778, 327.29468, 327.2188, 327.15973]],
                      [[318.47897, 318.30374, 318.1081, 317.8837, 317.63837],
                       [319.155, 318.983, 318.79745, 318.58905, 318.36212],
                       [319.8042, 319.64206, 319.4669, 319.2713, 319.0611],
                       [320.4621, 320.3055, 320.13373, 319.9425, 319.7401],
                       [321.1375, 320.98648, 320.81473, 320.62186, 320.4186]]])

    dt_theta = interpolate_to_isosurface(pv, theta, 2)

    truth = np.array([[324.761318, 323.4567137, 322.3276748, 321.3501466, 320.5223535],
                      [330.286922, 327.7779134, 325.797487, 324.3984446, 323.1793418],
                      [335.4152061, 333.9585512, 332.0114516, 329.3572419, 326.4791125],
                      [336.7088576, 336.4165698, 335.6255217, 334.0758288, 331.9684081],
                      [335.6583567, 336.3500714, 336.6844744, 336.3286052, 335.3874244]])

    assert_array_almost_equal(truth, dt_theta)


@pytest.mark.parametrize('assume_units', [None, 'mbar'])
@pytest.mark.parametrize('method', interp_methods)
@pytest.mark.parametrize('boundary_coords', boundary_types)
def test_interpolate_to_grid(method, assume_units, test_coords, boundary_coords):
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

    with get_test_data(f'{method}_test.npz') as fobj:
        truth = np.load(fobj)['img']

    if assume_units:
        z = units.Quantity(z, assume_units)
        truth = units.Quantity(truth, assume_units)

    # Value is tuned to keep the old results working after fixing an off-by-one error
    # in the grid generation (desired value was 10) See #2319.
    hres = 10.121
    xg, yg, img = interpolate_to_grid(xp, yp, z, hres=hres, interp_type=method, **extra_kw)

    assert np.all(np.diff(xg, axis=-1) <= hres)
    assert np.all(np.diff(yg, axis=0) <= hres)
    assert_array_almost_equal(truth, img)


def test_interpolate_to_isosurface_from_below():
    r"""Test interpolation to level function."""
    pv = np.array([[[1.75, 1.875, 2., 2.125, 2.25],
                    [1.9, 2.025, 2.15, 2.275, 2.4],
                    [2.05, 2.175, 2.3, 2.425, 2.55],
                    [2.2, 2.325, 2.45, 2.575, 2.7],
                    [2.35, 2.475, 2.6, 2.725, 2.85]],
                   [[1.5, 1.625, 1.75, 1.875, 2.],
                    [1.65, 1.775, 1.9, 2.025, 2.15],
                    [1.8, 1.925, 2.05, 2.175, 2.3],
                    [1.95, 2.075, 2.2, 2.325, 2.45],
                    [2.1, 2.225, 2.35, 2.475, 2.6]],
                   [[1.25, 1.375, 1.5, 1.625, 1.75],
                    [1.4, 1.525, 1.65, 1.775, 1.9],
                    [1.55, 1.675, 1.8, 1.925, 2.05],
                    [1.7, 1.825, 1.95, 2.075, 2.2],
                    [1.85, 1.975, 2.1, 2.225, 2.35]]])

    theta = np.array([[[330., 350., 370., 390., 410.],
                       [340., 360., 380., 400., 420.],
                       [350., 370., 390., 410., 430.],
                       [360., 380., 400., 420., 440.],
                       [370., 390., 410., 430., 450.]],
                      [[320., 340., 360., 380., 400.],
                       [330., 350., 370., 390., 410.],
                       [340., 360., 380., 400., 420.],
                       [350., 370., 390., 410., 430.],
                       [360., 380., 400., 420., 440.]],
                      [[310., 330., 350., 370., 390.],
                       [320., 340., 360., 380., 400.],
                       [330., 350., 370., 390., 410.],
                       [340., 360., 380., 400., 420.],
                       [350., 370., 390., 410., 430.]]])

    dt_theta = interpolate_to_isosurface(pv, theta, 2, bottom_up_search=False)

    truth = np.array([[330., 350., 370., 385., 400.],
                      [340., 359., 374., 389., 404.],
                      [348., 363., 378., 393., 410.],
                      [352., 367., 382., 400., 420.],
                      [356., 371., 390., 410., 430.]])

    assert_array_almost_equal(truth, dt_theta)
