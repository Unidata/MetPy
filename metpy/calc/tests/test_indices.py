# Copyright (c) 2008-2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `indices` module."""

from datetime import datetime

from metpy.calc import (bulk_shear, bunkers_storm_motion, mean_pressure_weighted,
                        precipitable_water)
from metpy.io import get_upper_air_data
from metpy.io.upperair import UseSampleData
from metpy.testing import assert_almost_equal, assert_array_equal
from metpy.units import concatenate, units


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


def test_bunkers_motion():
    """Test Bunkers storm motion with observed sounding."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    motion = concatenate(bunkers_storm_motion(data.variables['pressure'][:],
                         data.variables['u_wind'][:], data.variables['v_wind'][:],
                         data.variables['height'][:]))
    truth = [1.4537892577864744, 2.0169333025630616, 10.587950761120482, 13.915130377372801,
             6.0208700094534775, 7.9660318399679308] * units('m/s')
    assert_almost_equal(motion.flatten(), truth, 8)


def test_bulk_shear():
    """Test bulk shear with observed sounding."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    u, v = bulk_shear(data.variables['pressure'][:], data.variables['u_wind'][:],
                      data.variables['v_wind'][:], heights=data.variables['height'][:],
                      depth=6000 * units('meter'))
    truth = [29.899581266946115, -14.389225800205509] * units('knots')
    assert_almost_equal(u.to('knots'), truth[0], 8)
    assert_almost_equal(v.to('knots'), truth[1], 8)


def test_bulk_shear_elevated():
    """Test bulk shear with observed sounding and a base above the surface."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    u, v = bulk_shear(data.variables['pressure'][:], data.variables['u_wind'][:],
                      data.variables['v_wind'][:], heights=data.variables['height'][:],
                      bottom=data.variables['height'][0] + 3000 * units('meter'),
                      depth=3000 * units('meter'))
    truth = [0.9655943923302139, -3.8405428777944466] * units('m/s')
    assert_almost_equal(u, truth[0], 8)
    assert_almost_equal(v, truth[1], 8)
