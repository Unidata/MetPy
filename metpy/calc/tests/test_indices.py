# Copyright (c) 2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `indices` module."""

from datetime import datetime
import warnings

import numpy as np
from metpy.calc import (bulk_shear, bunkers_storm_motion, mean_pressure_weighted,
                        precipitable_water, significant_tornado, supercell_composite)
from metpy.deprecation import MetpyDeprecationWarning
from metpy.io import get_upper_air_data
from metpy.io.upperair import UseSampleData
from metpy.testing import assert_almost_equal, assert_array_equal
from metpy.units import concatenate, units

warnings.simplefilter('ignore', MetpyDeprecationWarning)


def test_precipitable_water():
    """Test precipitable water with observed sounding."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    pw = precipitable_water(data.variables['dewpoint'][:], data.variables['pressure'][:],
                            top=400 * units.hPa)
    truth = (0.8899441949243486 * units('inches')).to('millimeters')
    assert_array_equal(pw, truth)


def test_precipitable_water_no_bounds():
    """Test precipitable water with observed sounding and no bounds given."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    dewpoint = data.variables['dewpoint'][:]
    pressure = data.variables['pressure'][:]
    inds = pressure >= 400 * units.hPa
    pw = precipitable_water(dewpoint[inds], pressure[inds])
    truth = (0.8899441949243486 * units('inches')).to('millimeters')
    assert_array_equal(pw, truth)


def test_precipitable_water_bound_error():
    """Test with no top bound given and data that produced floating point issue #596."""
    pressure = np.array([993., 978., 960.5, 927.6, 925., 895.8, 892., 876., 45.9, 39.9, 36.,
                         36., 34.3]) * units.hPa
    dewpoint = np.array([25.5, 24.1, 23.1, 21.2, 21.1, 19.4, 19.2, 19.2, -87.1, -86.5, -86.5,
                         -86.5, -88.1]) * units.degC
    pw = precipitable_water(dewpoint, pressure)
    truth = 89.86955998646951 * units('millimeters')
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


def test_bulk_shear_no_depth():
    """Test bulk shear with observed sounding and no depth given. Issue #568."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    u, v = bulk_shear(data.variables['pressure'][:], data.variables['u_wind'][:],
                      data.variables['v_wind'][:], heights=data.variables['height'][:])
    truth = [20.225018939, 22.602359692] * units('knots')
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


def test_supercell_composite():
    """Test supercell composite function."""
    mucape = [2000., 1000., 500., 2000.] * units('J/kg')
    esrh = [400., 150., 45., 45.] * units('m^2/s^2')
    ebwd = [30., 15., 5., 5.] * units('m/s')
    truth = [16., 2.25, 0., 0.]
    supercell_comp = supercell_composite(mucape, esrh, ebwd)
    assert_array_equal(supercell_comp, truth)


def test_sigtor():
    """Test significant tornado parameter function."""
    sbcape = [2000., 2000., 2000., 2000., 3000, 4000] * units('J/kg')
    sblcl = [3000., 1500., 500., 1500., 1500, 800] * units('meter')
    srh1 = [200., 200., 200., 200., 300, 400] * units('m^2/s^2')
    shr6 = [20., 5., 20., 35., 20., 35] * units('m/s')
    truth = [0., 0, 1.777778, 1.333333, 2., 10.666667]
    sigtor = significant_tornado(sbcape, sblcl, srh1, shr6)
    assert_almost_equal(sigtor, truth, 6)
