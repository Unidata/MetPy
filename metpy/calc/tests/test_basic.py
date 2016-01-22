# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.testing import assert_array_equal

from metpy.units import units
from metpy.testing import assert_almost_equal, assert_array_almost_equal
from metpy.calc.basic import *  # noqa


def test_wind_comps_basic():
    'Test the basic wind component calculation.'
    speed = np.array([4, 4, 4, 4, 25, 25, 25, 25, 10.]) * units.mph
    dirs = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360]) * units.deg
    s2 = np.sqrt(2.)

    u, v = get_wind_components(speed, dirs)

    true_u = np.array([0, -4 / s2, -4, -4 / s2, 0, 25 / s2, 25, 25 / s2, 0]) * units.mph
    true_v = np.array([-4, -4 / s2, 0, 4 / s2, 25, 25 / s2, 0, -25 / s2, -10]) * units.mph

    assert_array_almost_equal(true_u, u, 4)
    assert_array_almost_equal(true_v, v, 4)


def test_wind_comps_scalar():
    'Test scalar wind components'
    u, v = get_wind_components(8 * units('m/s'), 150 * units.deg)
    assert_almost_equal(u, -4 * units('m/s'), 3)
    assert_almost_equal(v, 6.9282 * units('m/s'), 3)


def test_speed():
    'Basic test of wind speed calculation'
    u = np.array([4., 2., 0., 0.]) * units('m/s')
    v = np.array([0., 2., 4., 0.]) * units('m/s')

    speed = get_wind_speed(u, v)

    s2 = np.sqrt(2.)
    true_speed = np.array([4., 2 * s2, 4., 0.]) * units('m/s')

    assert_array_almost_equal(true_speed, speed, 4)


def test_dir():
    'Basic test of wind direction calculation'
    u = np.array([4., 2., 0., 0.]) * units('m/s')
    v = np.array([0., 2., 4., 0.]) * units('m/s')

    direc = get_wind_dir(u, v)

    true_dir = np.array([90., 45., 0., 90.]) * units.deg

    assert_array_almost_equal(true_dir, direc, 4)


def test_scalar_speed():
    'Test wind speed with scalars'
    s = get_wind_speed(-3. * units('m/s'), -4. * units('m/s'))
    assert_almost_equal(s, 5. * units('m/s'), 3)


def test_scalar_dir():
    'Test wind direction with scalars'
    d = get_wind_dir(-3. * units('m/s'), -4. * units('m/s'))
    assert_almost_equal(d, 216.870 * units.deg, 3)


def test_windchill_scalar():
    'Test wind chill with scalars'
    wc = windchill(-5 * units.degC, 35 * units('m/s'))
    assert_almost_equal(wc, -18.9357 * units.degC, 0)


def test_windchill_basic():
    'Test the basic wind chill calculation.'
    temp = np.array([40, -10, -45, 20]) * units.degF
    speed = np.array([5, 55, 25, 15]) * units.mph

    wc = windchill(temp, speed)
    values = np.array([36, -46, -84, 6]) * units.degF
    assert_array_almost_equal(wc, values, 0)


def test_windchill_invalid():
    'Test for values that should be masked.'
    temp = np.array([10, 51, 49, 60, 80, 81]) * units.degF
    speed = np.array([4, 4, 3, 1, 10, 39]) * units.mph

    wc = windchill(temp, speed)
    mask = np.array([False, True, True, True, True, True])
    assert_array_equal(wc.mask, mask)


def test_windchill_undefined_flag():
    'Tests whether masking values can be disabled.'
    temp = units.Quantity(np.ma.array([49, 50, 49, 60, 80, 81]), units.degF)
    speed = units.Quantity(([4, 4, 3, 1, 10, 39]), units.mph)

    wc = windchill(temp, speed, mask_undefined=False)
    mask = np.array([False] * 6)
    assert_array_equal(wc.mask, mask)


def test_windchill_face_level():
    'Tests using the face_level flag'
    temp = np.array([20, 0, -20, -40]) * units.degF
    speed = np.array([15, 30, 45, 60]) * units.mph

    wc = windchill(temp, speed, face_level_winds=True)
    values = np.array([3, -30, -64, -98]) * units.degF
    assert_array_almost_equal(wc, values, 0)


def test_heat_index_basic():
    'Test the basic heat index calculation.'
    temp = np.array([80, 88, 92, 110]) * units.degF
    rh = np.array([40, 100, 70, 40]) * units.percent

    hi = heat_index(temp, rh)
    values = np.array([80, 121, 112, 136]) * units.degF
    assert_array_almost_equal(hi, values, 0)


def test_heat_index_scalar():
    'Test heat index using scalars'
    hi = heat_index(96 * units.degF, 65 * units.percent)
    assert_almost_equal(hi, 121 * units.degF, 0)


def test_heat_index_invalid():
    'Test for values that should be masked.'
    temp = np.array([80, 88, 92, 79, 30, 81]) * units.degF
    rh = np.array([40, 39, 2, 70, 50, 39]) * units.percent

    hi = heat_index(temp, rh)
    mask = np.array([False, True, True, True, True, True])
    assert_array_equal(hi.mask, mask)


def test_heat_index_undefined_flag():
    'Tests whether masking values can be disabled.'
    temp = units.Quantity(np.ma.array([80, 88, 92, 79, 30, 81]), units.degF)
    rh = np.ma.array([40, 39, 2, 70, 50, 39]) * units.percent

    hi = heat_index(temp, rh, mask_undefined=False)
    mask = np.array([False] * 6)
    assert_array_equal(hi.mask, mask)


def test_heat_index_units():
    'Test units coming out of heat index'
    temp = units.Quantity([35., 20.], units.degC)
    rh = 70 * units.percent
    hi = heat_index(temp, rh)
    assert_almost_equal(hi.to('degC'), units.Quantity([50.3405, np.nan], units.degC), 4)


def test_heat_index_ratio():
    'Test giving humidity as number [0, 1]'
    temp = units.Quantity([35., 20.], units.degC)
    rh = 0.7
    hi = heat_index(temp, rh)
    assert_almost_equal(hi.to('degC'), units.Quantity([50.3405, np.nan], units.degC), 4)

# class TestIrrad(object):
#    def test_basic(self):
#        'Test the basic solar irradiance calculation.'
#        from datetime import date

#        d = date(2008, 9, 28)
#        lat = 35.25
#        hours = np.linspace(6,18,10)

#        s = solar_irradiance(lat, d, hours)
#        values = np.array([0., 344.1, 682.6, 933.9, 1067.6, 1067.6, 933.9,
#            682.6, 344.1, 0.])
#        assert_array_almost_equal(s, values, 1)

#    def test_scalar(self):
#        from datetime import date
#        d = date(2008, 9, 28)
#        lat = 35.25
#        hour = 9.5
#        s = solar_irradiance(lat, d, hour)
#        assert_almost_equal(s, 852.1, 1)

#    def test_invalid(self):
#        'Test for values that should be masked.'
#        from datetime import date
#        d = date(2008, 9, 28)
#        lat = 35.25
#        hours = np.linspace(0,22,12)
#        s = solar_irradiance(lat, d, hours)

#        mask = np.array([ True,  True,  True,  True, False, False, False,
#            False, False, True,  True,  True])
#        assert_array_equal(s.mask, mask)
