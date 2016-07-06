# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from metpy.mapping.map_functions import *

import numpy as np

from numpy.testing import assert_array_almost_equal, assert_almost_equal


def test_calc_kappa():

    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 58])
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16])

    spacing = np.mean((cdist(list(zip(x, y)), list(zip(x, y)))))

    value = calc_kappa(spacing)

    truth = 5762.6872048

    assert_almost_equal(truth, value)


def test_remove_observations_below_value():

    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 58])
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16])

    z = np.array(list(range(-10, 10, 2)))

    x_, y_, z_ = remove_observations_below_value(x, y, z, val=0)

    truthx = np.array([53, 98, 34, 15, 58])
    truthy = np.array([66, 14, 24, 60, 16])
    truthz = np.array([0, 2, 4, 6, 8])

    assert_array_almost_equal(truthx, x_)
    assert_array_almost_equal(truthy, y_)
    assert_array_almost_equal(truthz, z_)


def test_remove_nan_observations():

    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 58])
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16])

    z = np.array([np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1, 1])

    x_, y_, z_ = remove_nan_observations(x, y, z)

    truthx = np.array([79, 10, 52, 53, 98, 34, 15, 58])
    truthy = np.array([48, 94, 98, 66, 14, 24, 60, 16])
    truthz = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    assert_array_almost_equal(truthx, x_)
    assert_array_almost_equal(truthy, y_)
    assert_array_almost_equal(truthz, z_)


def test_remove_repeat_coordinates():

    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 8])
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 24])

    z = np.array(list(range(-10, 10, 2)))

    x_, y_, z_ = remove_repeat_coordinates(x, y, z)

    truthx = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15])
    truthy = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60])
    truthz = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6])

    assert_array_almost_equal(truthx, x_)
    assert_array_almost_equal(truthy, y_)
    assert_array_almost_equal(truthz, z_)

