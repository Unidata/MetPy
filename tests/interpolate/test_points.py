# Copyright (c) 2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `points` module."""


import logging

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
from scipy.spatial import cKDTree, Delaunay

from metpy.cbook import get_test_data
from metpy.interpolate import (interpolate_to_points, inverse_distance_to_points,
                               natural_neighbor_to_points)
from metpy.interpolate.geometry import dist_2, find_natural_neighbors
from metpy.interpolate.points import barnes_point, cressman_point, natural_neighbor_point

logging.getLogger('metpy.interpolate.points').setLevel(logging.ERROR)


@pytest.fixture()
def test_data():
    r"""Return data used for tests in this file."""
    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 58], dtype=float)
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16], dtype=float)
    z = np.array([0.064, 4.489, 6.241, 0.1, 2.704, 2.809, 9.604, 1.156,
                  0.225, 3.364], dtype=float)

    return x, y, z


@pytest.fixture()
def test_points():
    r"""Return point locations used for tests in this file."""
    with get_test_data('interpolation_test_grid.npz') as fobj:
        data = np.load(fobj)
        return np.stack([data['xg'].reshape(-1), data['yg'].reshape(-1)], axis=1)


def test_nn_point(test_data):
    r"""Test find natural neighbors for a point interpolation function."""
    xp, yp, z = test_data

    tri = Delaunay(list(zip(xp, yp)))

    sim_gridx = [30]
    sim_gridy = [30]

    members, tri_info = find_natural_neighbors(tri,
                                               list(zip(sim_gridx, sim_gridy)))

    val = natural_neighbor_point(xp, yp, z, [sim_gridx[0], sim_gridy[0]],
                                 tri, members[0], tri_info)

    truth = 1.009

    assert_almost_equal(truth, val, 3)


def test_cressman_point(test_data):
    r"""Test Cressman interpolation for a point function."""
    xp, yp, z = test_data

    r = 40

    obs_tree = cKDTree(list(zip(xp, yp)))

    indices = obs_tree.query_ball_point([30, 30], r=r)

    dists = dist_2(30, 30, xp[indices], yp[indices])
    values = z[indices]

    truth = 1.05499444404

    value = cressman_point(dists, values, r)

    assert_almost_equal(truth, value)


def test_barnes_point(test_data):
    r"""Test Barnes interpolation for a point function."""
    xp, yp, z = test_data

    r = 40

    obs_tree = cKDTree(list(zip(xp, yp)))

    indices = obs_tree.query_ball_point([60, 60], r=r)

    dists = dist_2(60, 60, xp[indices], yp[indices])
    values = z[indices]

    assert_almost_equal(barnes_point(dists, values, 5762.7), 4.0871824)


def test_natural_neighbor_to_points(test_data, test_points):
    r"""Test natural neighbor interpolation to grid function."""
    xp, yp, z = test_data
    obs_points = np.vstack([xp, yp]).transpose()

    img = natural_neighbor_to_points(obs_points, z, test_points)

    with get_test_data('nn_bbox0to100.npz') as fobj:
        truth = np.load(fobj)['img'].reshape(-1)

    assert_array_almost_equal(truth, img)


interp_methods = ['cressman', 'barnes', 'shouldraise']


@pytest.mark.parametrize('method', interp_methods)
def test_inverse_distance_to_points(method, test_data, test_points):
    r"""Test inverse distance interpolation to grid function."""
    xp, yp, z = test_data
    obs_points = np.vstack([xp, yp]).transpose()

    extra_kw = {}
    if method == 'shouldraise':
        extra_kw['r'] = 40
        with pytest.raises(ValueError):
            inverse_distance_to_points(
                obs_points, z, test_points, kind=method, **extra_kw)
    else:
        test_file = ''
        if method == 'cressman':
            extra_kw['r'] = 20
            extra_kw['min_neighbors'] = 1
            test_file = 'cressman_r20_mn1.npz'
        elif method == 'barnes':
            extra_kw['r'] = 40
            extra_kw['kappa'] = 100
            test_file = 'barnes_r40_k100.npz'

        img = inverse_distance_to_points(obs_points, z, test_points, kind=method, **extra_kw)

        with get_test_data(test_file) as fobj:
            truth = np.load(fobj)['img'].reshape(-1)

        assert_array_almost_equal(truth, img)


@pytest.mark.parametrize('method', ['natural_neighbor', 'cressman', 'barnes', 'linear',
                                    'nearest', 'rbf', 'shouldraise', 'cubic'])
def test_interpolate_to_points(method, test_data):
    r"""Test main grid interpolation function."""
    xp, yp, z = test_data
    obs_points = np.vstack([xp, yp]).transpose() * 10

    with get_test_data('interpolation_test_points.npz') as fobj:
        test_points = np.load(fobj)['points']

    extra_kw = {}
    if method == 'cressman':
        extra_kw['search_radius'] = 200
        extra_kw['minimum_neighbors'] = 1
    elif method == 'barnes':
        extra_kw['search_radius'] = 400
        extra_kw['minimum_neighbors'] = 1
        extra_kw['gamma'] = 1
    elif method == 'shouldraise':
        with pytest.raises(ValueError):
            interpolate_to_points(
                obs_points, z, test_points, interp_type=method, **extra_kw)
        return

    img = interpolate_to_points(obs_points, z, test_points, interp_type=method, **extra_kw)

    with get_test_data(f'{method}_test.npz') as fobj:
        truth = np.load(fobj)['img'].reshape(-1)

    assert_array_almost_equal(truth, img)
