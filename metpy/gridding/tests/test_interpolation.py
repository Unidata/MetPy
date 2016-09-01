# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division

import pytest

from metpy.gridding.interpolation import (barnes_weights, nn_point, cressman_weights,
                                          cressman_point, barnes_point, natural_neighbor,
                                          inverse_distance)

from metpy.gridding.triangles import find_natural_neighbors, dist_2
from metpy.gridding.gridding_functions import calc_kappa
from metpy.cbook import get_test_data
from scipy.spatial import cKDTree, Delaunay

import numpy as np

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal

from scipy.spatial.distance import cdist


@pytest.fixture()
def test_data():
    r"""Test data used for tests in this file"""

    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 58], dtype=float)
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16], dtype=float)
    z = np.array([0.064, 4.489, 6.241, 0.1, 2.704, 2.809, 9.604, 1.156,
                  0.225, 3.364], dtype=float)

    return x, y, z


@pytest.fixture()
def test_grid():
    r"""Test grid locations used for tests in this file"""

    xg = np.load(get_test_data('interpolation_test_grid.npz'))['xg']
    yg = np.load(get_test_data('interpolation_test_grid.npz'))['yg']

    return xg, yg


def test_natural_neighbor(test_data, test_grid):
    r"""Tests natural neighbor interpolation function"""

    xp, yp, z = test_data
    xg, yg = test_grid

    img = natural_neighbor(xp, yp, z, xg, yg)

    truth = np.load(get_test_data('nn_bbox0to100.npz'))['img']

    assert_array_almost_equal(truth, img)


interp_methods = ['cressman', 'barnes']


@pytest.mark.parametrize('method', interp_methods)
def test_inverse_distance(method, test_data, test_grid):
    r"""Tests inverse distance interpolation function"""

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

    img = inverse_distance(xp, yp, z, xg, yg, kind=method, **extra_kw)

    truth = np.load(get_test_data(test_file))['img']

    assert_array_almost_equal(truth, img)


def test_nn_point(test_data):
    r"""Tests find natural neighbors for a point interpolation function"""

    xp, yp, z = test_data

    tri = Delaunay(list(zip(xp, yp)))

    sim_gridx = [30]
    sim_gridy = [30]

    members, tri_info = find_natural_neighbors(tri,
                                               list(zip(sim_gridx, sim_gridy)))

    val = nn_point(xp, yp, z, [sim_gridx[0], sim_gridy[0]],
                   tri, members[0], tri_info)

    truth = 1.009

    assert_almost_equal(truth, val, 3)


def test_barnes_weights():
    r"""Tests barnes weights function"""

    kappa = 1000000

    gamma = 0.5

    dist = np.array([1000, 2000, 3000, 4000])**2

    weights = barnes_weights(dist, kappa, gamma) * 10000000

    truth = [1353352.832366126918939,
             3354.626279025118388,
             .152299797447126,
             .000000126641655]

    assert_array_almost_equal(truth, weights)


def test_cressman_weights():
    r"""Tests cressman weights function"""

    r = 5000

    dist = np.array([1000, 2000, 3000, 4000])**2

    weights = cressman_weights(dist, r)

    truth = [0.923076923076923,
             0.724137931034482,
             0.470588235294117,
             0.219512195121951]

    assert_array_almost_equal(truth, weights)


def test_cressman_point(test_data):
    r"""Tests cressman interpolation for a point function"""

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
    r"""Tests barnes interpolation for a point function"""

    xp, yp, z = test_data

    r = 40

    obs_tree = cKDTree(list(zip(xp, yp)))

    indices = obs_tree.query_ball_point([60, 60], r=r)

    dists = dist_2(60, 60, xp[indices], yp[indices])
    values = z[indices]

    truth = 4.08718241061

    ave_spacing = np.mean((cdist(list(zip(xp, yp)), list(zip(xp, yp)))))

    kappa = calc_kappa(ave_spacing)

    value = barnes_point(dists, values, kappa)

    assert_almost_equal(truth, value)
