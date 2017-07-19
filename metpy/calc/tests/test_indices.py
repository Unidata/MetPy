# Copyright (c) 2008-2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `indices` module."""

from datetime import datetime

from metpy.calc import mean_pressure_weighted, precipitable_water
from metpy.io import get_upper_air_data
from metpy.io.upperair import UseSampleData
from metpy.testing import assert_almost_equal, assert_array_equal
from metpy.units import units


def test_precipitable_water():
    """Test precipitable water with observed sounding."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    pw = precipitable_water(data.variables['dewpoint'][:], data.variables['pressure'][:])
    truth = (0.8899441949243486 * units('inches')).to('millimeters')
    assert_array_equal(pw, truth)


def test_mean_pressure_weighted():
    """Test pressure-weighted mean wind function with vertical interpolation."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    u, v = mean_pressure_weighted(data.variables['pressure'][:],
                                  data.variables['u_wind'][:],
                                  data.variables['v_wind'][:],
                                  heights=data.variables['height'][:],
                                  depth=6000 * units('meter'))
    assert_almost_equal(u, 6.0208700094534775 * units('m/s'), 7)
    assert_almost_equal(v, 7.966031839967931 * units('m/s'), 7)


def test_mean_pressure_weighted_elevated():
    """Test pressure-weighted mean wind function with a base above the surface."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    u, v = mean_pressure_weighted(data.variables['pressure'][:],
                                  data.variables['u_wind'][:],
                                  data.variables['v_wind'][:],
                                  heights=data.variables['height'][:],
                                  depth=3000 * units('meter'),
                                  bottom=data.variables['height'][0] + 3000 * units('meter'))
    assert_almost_equal(u, 8.270829843626476 * units('m/s'), 7)
    assert_almost_equal(v, 1.7392601775853547 * units('m/s'), 7)
