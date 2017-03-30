# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for `calc.tools` module."""

import numpy as np
import numpy.ma as ma
import pytest

from metpy.calc import (find_intersections, interpolate_nans, nearest_intersection_idx,
                        reduce_point_density, resample_nn_1d)
from metpy.calc.tools import delete_masked_points, _next_non_masked_element
from metpy.testing import assert_array_almost_equal, assert_array_equal


def test_resample_nn():
    """Test 1d nearest neighbor functionality."""
    a = np.arange(5.)
    b = np.array([2, 3.8])
    truth = np.array([2, 4])

    assert_array_equal(truth, resample_nn_1d(a, b))


def test_nearest_intersection_idx():
    """Test nearest index to intersection functionality."""
    x = np.linspace(5, 30, 17)
    y1 = 3 * x**2
    y2 = 100 * x - 650
    truth = np.array([2, 12])

    assert_array_equal(truth, nearest_intersection_idx(y1, y2))


@pytest.mark.parametrize('direction, expected', [
    ('all', np.array([[8.88, 24.44], [238.84, 1794.53]])),
    ('increasing', np.array([[24.44], [1794.53]])),
    ('decreasing', np.array([[8.88], [238.84]]))
])
def test_find_intersections(direction, expected):
    """Test finding the intersection of two curves functionality."""
    x = np.linspace(5, 30, 17)
    y1 = 3 * x**2
    y2 = 100 * x - 650
    # Note: Truth is what we will get with this sampling, not the mathematical intersection
    assert_array_almost_equal(expected, find_intersections(x, y1, y2, direction=direction), 2)


def test_find_intersections_no_intersections():
    """Test finding the intersection of two curves with no intersections."""
    x = np.linspace(5, 30, 17)
    y1 = 3 * x + 0
    y2 = 5 * x + 5
    # Note: Truth is what we will get with this sampling, not the mathematical intersection
    truth = np.array([[],
                      []])
    assert_array_equal(truth, find_intersections(x, y1, y2))


def test_find_intersections_invalid_direction():
    """Test exception if an invalid direction is given."""
    x = np.linspace(5, 30, 17)
    y1 = 3 * x ** 2
    y2 = 100 * x - 650
    with pytest.raises(ValueError):
        find_intersections(x, y1, y2, direction='increaing')


def test_interpolate_nan_linear():
    """Test linear interpolation of arrays with NaNs in the y-coordinate."""
    x = np.linspace(0, 20, 15)
    y = 5 * x + 3
    nan_indexes = [1, 5, 11, 12]
    y_with_nan = y.copy()
    y_with_nan[nan_indexes] = np.nan
    assert_array_almost_equal(y, interpolate_nans(x, y_with_nan), 2)


def test_interpolate_nan_log():
    """Test log interpolation of arrays with NaNs in the y-coordinate."""
    x = np.logspace(1, 5, 15)
    y = 5 * np.log(x) + 3
    nan_indexes = [1, 5, 11, 12]
    y_with_nan = y.copy()
    y_with_nan[nan_indexes] = np.nan
    assert_array_almost_equal(y, interpolate_nans(x, y_with_nan, kind='log'), 2)


def test_interpolate_nan_invalid():
    """Test log interpolation with invalid parameter."""
    x = np.logspace(1, 5, 15)
    y = 5 * np.log(x) + 3
    with pytest.raises(ValueError):
        interpolate_nans(x, y, kind='loog')


@pytest.mark.parametrize('mask, expected_idx, expected_element', [
    ([False, False, False, False, False], 1, 1),
    ([False, True, True, False, False], 3, 3),
    ([False, True, True, True, True], None, None)
])
def test_non_masked_elements(mask, expected_idx, expected_element):
    """Test with a valid element."""
    a = ma.masked_array(np.arange(5), mask=mask)
    idx, element = _next_non_masked_element(a, 1)
    assert idx == expected_idx
    assert element == expected_element


@pytest.fixture
def thin_point_data():
    r"""Provide scattered points for testing."""
    xy = np.array([[0.8793620, 0.9005706], [0.5382446, 0.8766988], [0.6361267, 0.1198620],
                   [0.4127191, 0.0270573], [0.1486231, 0.3121822], [0.2607670, 0.4886657],
                   [0.7132257, 0.2827587], [0.4371954, 0.5660840], [0.1318544, 0.6468250],
                   [0.6230519, 0.0682618], [0.5069460, 0.2326285], [0.1324301, 0.5609478],
                   [0.7975495, 0.2109974], [0.7513574, 0.9870045], [0.9305814, 0.0685815],
                   [0.5271641, 0.7276889], [0.8116574, 0.4795037], [0.7017868, 0.5875983],
                   [0.5591604, 0.5579290], [0.1284860, 0.0968003], [0.2857064, 0.3862123]])
    return xy


@pytest.mark.parametrize('radius, truth',
                         [(2.0, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool)),
                          (1.0, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.bool)),
                          (0.3, np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
                                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.bool)),
                          (0.1, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                                          0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool))
                          ])
def test_reduce_point_density(thin_point_data, radius, truth):
    r"""Test that reduce_point_density works."""
    assert_array_equal(reduce_point_density(thin_point_data, radius=radius), truth)


@pytest.mark.parametrize('radius, truth',
                         [(2.0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.bool)),
                          (0.7, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=np.bool)),
                          (0.3, np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0,
                                          0, 0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=np.bool)),
                          (0.1, np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                          0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool))
                          ])
def test_reduce_point_density_priority(thin_point_data, radius, truth):
    r"""Test that reduce_point_density works properly with priority."""
    key = np.array([8, 6, 2, 8, 6, 4, 4, 8, 8, 6, 3, 4, 3, 0, 7, 4, 3, 2, 3, 3, 9])
    assert_array_equal(reduce_point_density(thin_point_data, radius, key), truth)


def test_reduce_point_density_1d():
    r"""Test that reduce_point_density works with 1D points."""
    x = np.array([1, 3, 4, 8, 9, 10])
    assert_array_equal(reduce_point_density(x, 2.5),
                       np.array([1, 0, 1, 1, 0, 0], dtype=np.bool))


a = ma.masked_array(np.arange(5), mask=[False, True, False, False, False])
b = ma.masked_array(np.arange(5), mask=[False, False, False, True, False])

@pytest.mark.parametrize('arrs, expected', [
    ((a, b), np.array([0, 2, 4])),
    ((a), np.array([0, 1, 2, 4])),
    ((a.compressed()), np.array([0, 1, 2, 4]))
])
def test_delete_masked_points(arrs, expected):
    """Test deleting masked points."""
    arrs = delete_masked_points(*arrs)
    assert arrs == expected
