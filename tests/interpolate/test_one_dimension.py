# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `one_dimension` module."""

import numpy as np
import pytest

from metpy.interpolate import interpolate_1d, interpolate_nans_1d, log_interpolate_1d
from metpy.testing import assert_array_almost_equal
from metpy.units import units


def test_interpolate_nans_1d_linear():
    """Test linear interpolation of arrays with NaNs in the y-coordinate."""
    x = np.linspace(0, 20, 15)
    y = 5 * x + 3
    nan_indexes = [1, 5, 11, 12]
    y_with_nan = y.copy()
    y_with_nan[nan_indexes] = np.nan
    assert_array_almost_equal(y, interpolate_nans_1d(x, y_with_nan), 2)


def test_interpolate_nans_1d_log():
    """Test log interpolation of arrays with NaNs in the y-coordinate."""
    x = np.logspace(1, 5, 15)
    y = 5 * np.log(x) + 3
    nan_indexes = [1, 5, 11, 12]
    y_with_nan = y.copy()
    y_with_nan[nan_indexes] = np.nan
    assert_array_almost_equal(y, interpolate_nans_1d(x, y_with_nan, kind='log'), 2)


def test_interpolate_nans_1d_invalid():
    """Test log interpolation with invalid parameter."""
    x = np.logspace(1, 5, 15)
    y = 5 * np.log(x) + 3
    with pytest.raises(ValueError):
        interpolate_nans_1d(x, y, kind='loog')


def test_log_interpolate_1d():
    """Test interpolating with log x-scale."""
    x_log = np.array([1e3, 1e4, 1e5, 1e6])
    y_log = np.log(x_log) * 2 + 3
    x_interp = np.array([5e3, 5e4, 5e5])
    y_interp_truth = np.array([20.0343863828, 24.6395565688, 29.2447267548])
    y_interp = log_interpolate_1d(x_interp, x_log, y_log)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


def test_log_interpolate_1d_units():
    """Test interpolating with log x-scale with units."""
    x_log = np.array([1e3, 1e4, 1e5, 1e6]) * units.hPa
    y_log = (np.log(x_log.m) * 2 + 3) * units.degC
    x_interp = np.array([5e5, 5e6, 5e7]) * units.Pa
    y_interp_truth = np.array([20.0343863828, 24.6395565688, 29.2447267548]) * units.degC
    y_interp = log_interpolate_1d(x_interp, x_log, y_log)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


def test_log_interpolate_2d():
    """Test interpolating with log x-scale in 2 dimensions."""
    x_log = np.array([[1e3, 1e4, 1e5, 1e6], [1e3, 1e4, 1e5, 1e6]])
    y_log = np.log(x_log) * 2 + 3
    x_interp = np.array([5e3, 5e4, 5e5])
    y_interp_truth = np.array([20.0343863828, 24.6395565688, 29.2447267548])
    y_interp = log_interpolate_1d(x_interp, x_log, y_log, axis=1)
    assert_array_almost_equal(y_interp[1], y_interp_truth, 7)


def test_log_interpolate_3d():
    """Test interpolating with log x-scale 3 dimensions along second axis."""
    x_log = np.ones((3, 4, 3)) * np.array([1e3, 1e4, 1e5, 1e6]).reshape(-1, 1)
    y_log = np.log(x_log) * 2 + 3
    x_interp = np.array([5e3, 5e4, 5e5])
    y_interp_truth = np.array([20.0343863828, 24.6395565688, 29.2447267548])
    y_interp = log_interpolate_1d(x_interp, x_log, y_log, axis=1)
    assert_array_almost_equal(y_interp[0, :, 0], y_interp_truth, 7)


def test_log_interpolate_4d():
    """Test interpolating with log x-scale 4 dimensions."""
    x_log = np.ones((2, 2, 3, 4)) * np.array([1e3, 1e4, 1e5, 1e6])
    y_log = np.log(x_log) * 2 + 3
    x_interp = np.array([5e3, 5e4, 5e5])
    y_interp_truth = np.array([20.0343863828, 24.6395565688, 29.2447267548])
    y_interp = log_interpolate_1d(x_interp, x_log, y_log, axis=3)
    assert_array_almost_equal(y_interp[0, 0, 0, :], y_interp_truth, 7)


def test_log_interpolate_2args():
    """Test interpolating with log x-scale with 2 arguments."""
    x_log = np.array([1e3, 1e4, 1e5, 1e6])
    y_log = np.log(x_log) * 2 + 3
    y_log2 = np.log(x_log) * 2 + 3
    x_interp = np.array([5e3, 5e4, 5e5])
    y_interp_truth = np.array([20.0343863828, 24.6395565688, 29.2447267548])
    y_interp = log_interpolate_1d(x_interp, x_log, y_log, y_log2)
    assert_array_almost_equal(y_interp[1], y_interp_truth, 7)
    assert_array_almost_equal(y_interp[0], y_interp_truth, 7)


def test_log_interpolate_set_nan_above():
    """Test interpolating with log x-scale setting out of bounds above data to nan."""
    x_log = np.array([1e3, 1e4, 1e5, 1e6])
    y_log = np.log(x_log) * 2 + 3
    x_interp = np.array([1e7])
    y_interp_truth = np.nan
    with pytest.warns(Warning):
        y_interp = log_interpolate_1d(x_interp, x_log, y_log)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


def test_log_interpolate_no_extrap():
    """Test interpolating with log x-scale setting out of bounds value error."""
    x_log = np.array([1e3, 1e4, 1e5, 1e6])
    y_log = np.log(x_log) * 2 + 3
    x_interp = np.array([1e7])
    with pytest.raises(ValueError):
        log_interpolate_1d(x_interp, x_log, y_log, fill_value=None)


def test_log_interpolate_set_nan_below():
    """Test interpolating with log x-scale setting out of bounds below data to nan."""
    x_log = np.array([1e3, 1e4, 1e5, 1e6])
    y_log = np.log(x_log) * 2 + 3
    x_interp = 1e2
    y_interp_truth = np.nan
    with pytest.warns(Warning):
        y_interp = log_interpolate_1d(x_interp, x_log, y_log)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


def test_interpolate_2args():
    """Test interpolation with 2 arguments."""
    x = np.array([1., 2., 3., 4.])
    y = x
    y2 = x
    x_interp = np.array([2.5000000, 3.5000000])
    y_interp_truth = np.array([2.5000000, 3.5000000])
    y_interp = interpolate_1d(x_interp, x, y, y2)
    assert_array_almost_equal(y_interp[0], y_interp_truth, 7)
    assert_array_almost_equal(y_interp[1], y_interp_truth, 7)


def test_interpolate_decrease():
    """Test interpolation with decreasing interpolation points."""
    x = np.array([1., 2., 3., 4.])
    y = x
    x_interp = np.array([3.5000000, 2.5000000])
    y_interp_truth = np.array([3.5000000, 2.5000000])
    y_interp = interpolate_1d(x_interp, x, y)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


def test_interpolate_decrease_xp():
    """Test interpolation with decreasing order."""
    x = np.array([4., 3., 2., 1.])
    y = x
    x_interp = np.array([3.5000000, 2.5000000])
    y_interp_truth = np.array([3.5000000, 2.5000000])
    y_interp = interpolate_1d(x_interp, x, y)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


def test_interpolate_end_point():
    """Test interpolation with point at data endpoints."""
    x = np.array([1., 2., 3., 4.])
    y = x
    x_interp = np.array([1.0, 4.0])
    y_interp_truth = np.array([1.0, 4.0])
    y_interp = interpolate_1d(x_interp, x, y)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


def test_interpolate_masked_units():
    """Test interpolating with masked arrays with units."""
    x = units.Quantity(np.ma.array([1., 2., 3., 4.]), units.m)
    y = units.Quantity(np.ma.array([50., 60., 70., 80.]), units.degC)
    x_interp = np.array([250., 350.]) * units.cm
    y_interp_truth = np.array([65., 75.]) * units.degC
    y_interp = interpolate_1d(x_interp, x, y)
    assert_array_almost_equal(y_interp, y_interp_truth, 7)


def test_interpolate_broadcast():
    """Test interpolate_1d with input levels needing broadcasting."""
    p = units.Quantity([850, 700, 500], 'hPa')
    t = units.Quantity(np.arange(60).reshape(3, 4, 5), 'degC')

    t_level = interpolate_1d(units.Quantity(700, 'hPa'), p[:, None, None], t)
    assert_array_almost_equal(t_level,
                              units.Quantity(np.arange(20., 40.).reshape(1, 4, 5), 'degC'), 7)
