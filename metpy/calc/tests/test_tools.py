# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.testing import assert_array_equal
from metpy.testing import assert_array_almost_equal

from metpy.calc.tools import *  # noqa: F403


def test_resample_nn():
    'Test 1d nearest neighbor functionality.'
    a = np.arange(5.)
    b = np.array([2, 3.8])
    truth = np.array([2, 4])

    assert_array_equal(truth, resample_nn_1d(a, b))


def test_nearest_intersection_idx():
    'Test nearest index to intersection functionality.'
    x = np.linspace(5, 30, 17)
    y1 = 3 * x**2
    y2 = 100 * x - 650
    truth = np.array([2, 12])

    assert_array_equal(truth, nearest_intersection_idx(x, y1, y2))


def test_find_intersections():
    'Test finding the intersection of two curves functionality.'
    x = np.linspace(5, 30, 17)
    y1 = 3 * x**2
    y2 = 100 * x - 650
    # Truth is what we will get with this sampling,
    # not the mathematical intersection
    truth = np.array([[8.88, 238.84],
                      [24.44, 1794.53]])

    assert_array_almost_equal(truth, find_intersections(x, y1, y2), 2)
