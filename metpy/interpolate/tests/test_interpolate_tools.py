# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `tools` module."""

from __future__ import division

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
from scipy.spatial.distance import cdist

from metpy.interpolate import (remove_nan_observations, remove_observations_below_value,
                               remove_repeat_coordinates)
from metpy.interpolate.tools import barnes_weights, calc_kappa, cressman_weights


@pytest.fixture()
def test_coords():
    r"""Return data locations used for tests in this file."""
    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 58], dtype=float)
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16], dtype=float)

    return x, y


def test_calc_kappa(test_coords):
    r"""Test calculate kappa parameter function."""
    x, y = test_coords

    spacing = np.mean((cdist(list(zip(x, y)),
                             list(zip(x, y)))))

    value = calc_kappa(spacing)

    truth = 5762.6872048

    assert_almost_equal(truth, value, decimal=6)


def test_remove_observations_below_value(test_coords):
    r"""Test threshold observations function."""
    x, y = test_coords[0], test_coords[1]

    z = np.array(list(range(-10, 10, 2)))

    x_, y_, z_ = remove_observations_below_value(x, y, z, val=0)

    truthx = np.array([53, 98, 34, 15, 58])
    truthy = np.array([66, 14, 24, 60, 16])
    truthz = np.array([0, 2, 4, 6, 8])

    assert_array_almost_equal(truthx, x_)
    assert_array_almost_equal(truthy, y_)
    assert_array_almost_equal(truthz, z_)


def test_remove_nan_observations(test_coords):
    r"""Test remove observations equal to nan function."""
    x, y = test_coords[0], test_coords[1]

    z = np.array([np.nan, np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1])

    x_, y_, z_ = remove_nan_observations(x, y, z)

    truthx = np.array([10, 52, 53, 98, 34, 15, 58])
    truthy = np.array([94, 98, 66, 14, 24, 60, 16])
    truthz = np.array([1, 1, 1, 1, 1, 1, 1])

    assert_array_almost_equal(truthx, x_)
    assert_array_almost_equal(truthy, y_)
    assert_array_almost_equal(truthz, z_)


def test_remove_repeat_coordinates(test_coords):
    r"""Test remove repeat coordinates function."""
    x, y = test_coords

    x[0] = 8.523
    x[-1] = 8.523
    y[0] = 24.123
    y[-1] = 24.123

    z = np.array(list(range(-10, 10, 2)))

    x_, y_, z_ = remove_repeat_coordinates(x, y, z)

    truthx = np.array([8.523, 67, 79, 10, 52, 53, 98, 34, 15])
    truthy = np.array([24.123, 87, 48, 94, 98, 66, 14, 24, 60])
    truthz = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6])

    assert_array_almost_equal(truthx, x_)
    assert_array_almost_equal(truthy, y_)
    assert_array_almost_equal(truthz, z_)


def test_barnes_weights():
    r"""Test Barnes weights function."""
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
    r"""Test Cressman weights function."""
    r = 5000

    dist = np.array([1000, 2000, 3000, 4000])**2

    weights = cressman_weights(dist, r)

    truth = [0.923076923076923,
             0.724137931034482,
             0.470588235294117,
             0.219512195121951]

    assert_array_almost_equal(truth, weights)
