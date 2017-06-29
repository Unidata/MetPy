# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for `calc.tools` module."""

import numpy as np
import numpy.ma as ma
import pytest

from metpy.calc import (find_intersections, get_layer, interpolate_nans, log_interp,
                        nearest_intersection_idx, pressure_to_height_std,
                        reduce_point_density, resample_nn_1d)
from metpy.calc.tools import (_get_bound_pressure_height, _next_non_masked_element,
                              delete_masked_points)
from metpy.testing import assert_array_almost_equal, assert_array_equal
from metpy.units import units


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


def test_delete_masked_points():
    """Test deleting masked points."""
    a = ma.masked_array(np.arange(5), mask=[False, True, False, False, False])
    b = ma.masked_array(np.arange(5), mask=[False, False, False, True, False])
    expected = np.array([0, 2, 4])
    a, b = delete_masked_points(a, b)
    assert_array_equal(a, expected)
    assert_array_equal(b, expected)


def test_log_interp():
    """Test interpolating with log x-scale."""
    x_log = np.array([1e3, 1e4, 1e5, 1e6])
    y_log = np.log(x_log) * 2 + 3
    x_interp = np.array([5e3, 5e4, 5e5])
    y_interp_truth = np.array([20.0343863828, 24.6395565688, 29.2447267548])
    y_interp = log_interp(x_interp, x_log, y_log)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


def test_log_interp_units():
    """Test interpolating with log x-scale with units."""
    x_log = np.array([1e3, 1e4, 1e5, 1e6]) * units.hPa
    y_log = (np.log(x_log.m) * 2 + 3) * units.degC
    x_interp = np.array([5e3, 5e4, 5e5]) * units.hPa
    y_interp_truth = np.array([20.0343863828, 24.6395565688, 29.2447267548]) * units.degC
    y_interp = log_interp(x_interp, x_log, y_log)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


@pytest.fixture
def get_bounds_data():
    """Provide pressure and height data for testing layer bounds calculation."""
    pressures = np.linspace(1000, 100, 10) * units.hPa
    heights = pressure_to_height_std(pressures)
    return pressures, heights


@pytest.mark.parametrize('pressure, bound, hgts, interp, expected', [
    (get_bounds_data()[0], 900 * units.hPa, None, True,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 900 * units.hPa, None, False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 870 * units.hPa, None, True,
     (870 * units.hPa, 1.2665298 * units.kilometer)),
    (get_bounds_data()[0], 870 * units.hPa, None, False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.9880028 * units.kilometer, None, True,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.9880028 * units.kilometer, None, False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 1.2665298 * units.kilometer, None, True,
     (870 * units.hPa, 1.2665298 * units.kilometer)),
    (get_bounds_data()[0], 1.2665298 * units.kilometer, None, False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 900 * units.hPa, get_bounds_data()[1], True,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 900 * units.hPa, get_bounds_data()[1], False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 870 * units.hPa, get_bounds_data()[1], True,
     (870 * units.hPa, 1.2643214 * units.kilometer)),
    (get_bounds_data()[0], 870 * units.hPa, get_bounds_data()[1], False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.9880028 * units.kilometer, get_bounds_data()[1], True,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.9880028 * units.kilometer, get_bounds_data()[1], False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 1.2665298 * units.kilometer, get_bounds_data()[1], True,
     (870 * units.hPa, 1.2665298 * units.kilometer)),
    (get_bounds_data()[0], 1.2665298 * units.kilometer, get_bounds_data()[1], False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.98800289 * units.kilometer, get_bounds_data()[1], True,
     (900 * units.hPa, 0.9880028 * units.kilometer))
])
def test_get_bound_pressure_height(pressure, bound, hgts, interp, expected):
    """Test getting bounds in layers with various parameter combinations."""
    bounds = _get_bound_pressure_height(pressure, bound, heights=hgts, interpolate=interp)
    assert_array_almost_equal(bounds[0], expected[0], 5)
    assert_array_almost_equal(bounds[1], expected[1], 5)


def test_get_bound_invalid_bound_units():
    """Test that value error is raised with invalid bound units."""
    p = np.arange(900, 300, -100) * units.hPa
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 100 * units.degC)


def test_get_bound_pressure_out_of_range():
    """Test when bound is out of data range in pressure."""
    p = np.arange(900, 300, -100) * units.hPa
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 100 * units.hPa)
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 1000 * units.hPa)


def test_get_bound_height_out_of_range():
    """Test when bound is out of data range in height."""
    p = np.arange(900, 300, -100) * units.hPa
    h = np.arange(1, 7) * units.kilometer
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 8 * units.kilometer, heights=h)
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 100 * units.meter, heights=h)


def test_get_layer_ragged_data():
    """Tests that error is raised for unequal length pressure and data arrays."""
    p = np.arange(10) * units.hPa
    y = np.arange(9) * units.degC
    with pytest.raises(ValueError):
        get_layer(p, y)


def test_get_layer_invalid_depth_units():
    """Tests that error is raised when depth has invalid units."""
    p = np.arange(10) * units.hPa
    y = np.arange(9) * units.degC
    with pytest.raises(ValueError):
        get_layer(p, y, depth=400 * units.degC)


@pytest.fixture
def layer_test_data():
    """Provide test data for testing of layer bounds."""
    pressure = np.arange(1000, 10, -100) * units.hPa
    temperature = np.linspace(25, -50, len(pressure)) * units.degC
    return pressure, temperature


@pytest.mark.parametrize('pressure, variable, heights, bottom, depth, interp, expected', [
    (layer_test_data()[0], layer_test_data()[1], None, None, 150 * units.hPa, True,
     (np.array([1000, 900, 850]) * units.hPa,
      np.array([25.0, 16.666666, 12.62262]) * units.degC)),
    (layer_test_data()[0], layer_test_data()[1], None, None, 150 * units.hPa, False,
     (np.array([1000, 900]) * units.hPa, np.array([25.0, 16.666666]) * units.degC)),
    (layer_test_data()[0], layer_test_data()[1], None, 2 * units.km, 3 * units.km, True,
     (np.array([794.85264282, 700., 600., 540.01696548]) * units.hPa,
      np.array([7.93049516, 0., -8.33333333, -13.14758845]) * units.degC))
])
def test_get_layer(pressure, variable, heights, bottom, depth, interp, expected):
    """Tests get_layer functionality."""
    p_layer, y_layer = get_layer(pressure, variable, heights=heights, bottom=bottom,
                                 depth=depth, interpolate=interp)
    assert_array_almost_equal(p_layer, expected[0], 5)
    assert_array_almost_equal(y_layer, expected[1], 5)
