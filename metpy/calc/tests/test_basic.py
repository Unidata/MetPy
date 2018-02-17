# Copyright (c) 2008,2015,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `basic` module."""

import numpy as np
import pytest

from metpy.calc import (add_height_to_pressure, add_pressure_to_height, coriolis_parameter,
                        geopotential_to_height, get_wind_components, get_wind_dir,
                        get_wind_speed, heat_index, height_to_geopotential,
                        height_to_pressure_std, pressure_to_height_std,
                        sigma_to_pressure, windchill)
from metpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from metpy.units import units


def test_wind_comps_basic():
    """Test the basic wind component calculation."""
    speed = np.array([4, 4, 4, 4, 25, 25, 25, 25, 10.]) * units.mph
    dirs = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360]) * units.deg
    s2 = np.sqrt(2.)

    u, v = get_wind_components(speed, dirs)

    true_u = np.array([0, -4 / s2, -4, -4 / s2, 0, 25 / s2, 25, 25 / s2, 0]) * units.mph
    true_v = np.array([-4, -4 / s2, 0, 4 / s2, 25, 25 / s2, 0, -25 / s2, -10]) * units.mph

    assert_array_almost_equal(true_u, u, 4)
    assert_array_almost_equal(true_v, v, 4)


def test_wind_comps_scalar():
    """Test wind components calculation with scalars."""
    u, v = get_wind_components(8 * units('m/s'), 150 * units.deg)
    assert_almost_equal(u, -4 * units('m/s'), 3)
    assert_almost_equal(v, 6.9282 * units('m/s'), 3)


def test_speed():
    """Test calculating wind speed."""
    u = np.array([4., 2., 0., 0.]) * units('m/s')
    v = np.array([0., 2., 4., 0.]) * units('m/s')

    speed = get_wind_speed(u, v)

    s2 = np.sqrt(2.)
    true_speed = np.array([4., 2 * s2, 4., 0.]) * units('m/s')

    assert_array_almost_equal(true_speed, speed, 4)


def test_dir():
    """Test calculating wind direction."""
    u = np.array([4., 2., 0., 0.]) * units('m/s')
    v = np.array([0., 2., 4., 0.]) * units('m/s')

    direc = get_wind_dir(u, v)

    true_dir = np.array([270., 225., 180., 270.]) * units.deg

    assert_array_almost_equal(true_dir, direc, 4)


def test_speed_dir_roundtrip():
    """Test round-tripping between speed/direction and components."""
    # Test each quadrant of the whole circle
    wspd = np.array([15., 5., 2., 10.]) * units.meters / units.seconds
    wdir = np.array([160., 30., 225., 350.]) * units.degrees

    u, v = get_wind_components(wspd, wdir)

    wdir_out = get_wind_dir(u, v)
    wspd_out = get_wind_speed(u, v)

    assert_array_almost_equal(wspd, wspd_out, 4)
    assert_array_almost_equal(wdir, wdir_out, 4)


def test_scalar_speed():
    """Test wind speed with scalars."""
    s = get_wind_speed(-3. * units('m/s'), -4. * units('m/s'))
    assert_almost_equal(s, 5. * units('m/s'), 3)


def test_scalar_dir():
    """Test wind direction with scalars."""
    d = get_wind_dir(3. * units('m/s'), 4. * units('m/s'))
    assert_almost_equal(d, 216.870 * units.deg, 3)


def test_windchill_scalar():
    """Test wind chill with scalars."""
    wc = windchill(-5 * units.degC, 35 * units('m/s'))
    assert_almost_equal(wc, -18.9357 * units.degC, 0)


def test_windchill_basic():
    """Test the basic wind chill calculation."""
    temp = np.array([40, -10, -45, 20]) * units.degF
    speed = np.array([5, 55, 25, 15]) * units.mph

    wc = windchill(temp, speed)
    values = np.array([36, -46, -84, 6]) * units.degF
    assert_array_almost_equal(wc, values, 0)


def test_windchill_kelvin():
    """Test wind chill when given Kelvin temperatures."""
    wc = windchill(268.15 * units.kelvin, 35 * units('m/s'))
    assert_almost_equal(wc, -18.9357 * units.degC, 0)


def test_windchill_invalid():
    """Test windchill for values that should be masked."""
    temp = np.array([10, 51, 49, 60, 80, 81]) * units.degF
    speed = np.array([4, 4, 3, 1, 10, 39]) * units.mph

    wc = windchill(temp, speed)
    # We don't care about the masked values
    truth = np.array([2.6230789, np.nan, np.nan, np.nan, np.nan, np.nan]) * units.degF
    mask = np.array([False, True, True, True, True, True])
    assert_array_almost_equal(truth, wc)
    assert_array_equal(wc.mask, mask)


def test_windchill_undefined_flag():
    """Test whether masking values for windchill can be disabled."""
    temp = units.Quantity(np.ma.array([49, 50, 49, 60, 80, 81]), units.degF)
    speed = units.Quantity(([4, 4, 3, 1, 10, 39]), units.mph)

    wc = windchill(temp, speed, mask_undefined=False)
    mask = np.array([False] * 6)
    assert_array_equal(wc.mask, mask)


def test_windchill_face_level():
    """Test windchill using the face_level flag."""
    temp = np.array([20, 0, -20, -40]) * units.degF
    speed = np.array([15, 30, 45, 60]) * units.mph

    wc = windchill(temp, speed, face_level_winds=True)
    values = np.array([3, -30, -64, -98]) * units.degF
    assert_array_almost_equal(wc, values, 0)


def test_heat_index_basic():
    """Test the basic heat index calculation."""
    temp = np.array([80, 88, 92, 110]) * units.degF
    rh = np.array([40, 100, 70, 40]) * units.percent

    hi = heat_index(temp, rh)
    values = np.array([80, 121, 112, 136]) * units.degF
    assert_array_almost_equal(hi, values, 0)


def test_heat_index_scalar():
    """Test heat index using scalars."""
    hi = heat_index(96 * units.degF, 65 * units.percent)
    assert_almost_equal(hi, 121 * units.degF, 0)


def test_heat_index_invalid():
    """Test heat index for values that should be masked."""
    temp = np.array([80, 88, 92, 79, 30, 81]) * units.degF
    rh = np.array([40, 39, 2, 70, 50, 39]) * units.percent

    hi = heat_index(temp, rh)
    mask = np.array([False, True, True, True, True, True])
    assert_array_equal(hi.mask, mask)


def test_heat_index_undefined_flag():
    """Test whether masking values can be disabled for heat index."""
    temp = units.Quantity(np.ma.array([80, 88, 92, 79, 30, 81]), units.degF)
    rh = np.ma.array([40, 39, 2, 70, 50, 39]) * units.percent

    hi = heat_index(temp, rh, mask_undefined=False)
    mask = np.array([False] * 6)
    assert_array_equal(hi.mask, mask)


def test_heat_index_units():
    """Test units coming out of heat index."""
    temp = units.Quantity([35., 20.], units.degC)
    rh = 70 * units.percent
    hi = heat_index(temp, rh)
    assert_almost_equal(hi.to('degC'), units.Quantity([50.3405, np.nan], units.degC), 4)


def test_heat_index_ratio():
    """Test giving humidity as number [0, 1] to heat index."""
    temp = units.Quantity([35., 20.], units.degC)
    rh = 0.7
    hi = heat_index(temp, rh)
    assert_almost_equal(hi.to('degC'), units.Quantity([50.3405, np.nan], units.degC), 4)


def test_height_to_geopotential():
    """Test conversion from height to geopotential."""
    height = units.Quantity([0, 1000, 2000, 3000], units.m)
    geopot = height_to_geopotential(height)
    assert_array_almost_equal(geopot, units.Quantity([0., 9817, 19632,
                              29443], units('m**2 / second**2')), 0)


def test_geopotential_to_height():
    """Test conversion from geopotential to height."""
    geopotential = units.Quantity([0, 9817.70342881, 19632.32592389,
                                  29443.86893527], units('m**2 / second**2'))
    height = geopotential_to_height(geopotential)
    assert_array_almost_equal(height, units.Quantity([0, 1000, 2000, 3000], units.m), 0)

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


def test_pressure_to_heights_basic():
    """Test basic pressure to height calculation for standard atmosphere."""
    pressures = np.array([975.2, 987.5, 956., 943.]) * units.mbar
    heights = pressure_to_height_std(pressures)
    values = np.array([321.5, 216.5, 487.6, 601.7]) * units.meter
    assert_almost_equal(heights, values, 1)


def test_heights_to_pressure_basic():
    """Test basic height to pressure calculation for standard atmosphere."""
    heights = np.array([321.5, 216.5, 487.6, 601.7]) * units.meter
    pressures = height_to_pressure_std(heights)
    values = np.array([975.2, 987.5, 956., 943.]) * units.mbar
    assert_almost_equal(pressures, values, 1)


def test_pressure_to_heights_units():
    """Test that passing non-mbar units works."""
    assert_almost_equal(pressure_to_height_std(29 * units.inHg), 262.859 * units.meter, 3)


def test_coriolis_force():
    """Test basic coriolis force calculation."""
    lat = np.array([-90., -30., 0., 30., 90.]) * units.degrees
    cor = coriolis_parameter(lat)
    values = np.array([-1.4584232E-4, -.72921159E-4, 0, .72921159E-4,
                       1.4584232E-4]) * units('s^-1')
    assert_almost_equal(cor, values, 7)


def test_add_height_to_pressure():
    """Test the pressure at height above pressure calculation."""
    pressure = add_height_to_pressure(1000 * units.hPa, 877.17421094 * units.meter)
    assert_almost_equal(pressure, 900 * units.hPa, 5)


def test_add_pressure_to_height():
    """Test the height at pressure above height calculation."""
    height = add_pressure_to_height(110.8286757 * units.m, 100 * units.hPa)
    assert_almost_equal(height, 988.0028867 * units.meter, 5)


def test_sigma_to_pressure():
    """Test sigma_to_pressure."""
    surface_pressure = 1000. * units.hPa
    model_top_pressure = 0. * units.hPa
    sigma = np.arange(0., 1.1, 0.1)
    expected = np.arange(0., 1100., 100.) * units.hPa
    pressure = sigma_to_pressure(sigma, surface_pressure, model_top_pressure)
    assert_array_almost_equal(pressure, expected, 5)


def test_warning_dir():
    """Test that warning is raised wind direction > 2Pi."""
    with pytest.warns(UserWarning):
        get_wind_components(3. * units('m/s'), 270)


def test_coriolis_warning():
    """Test that warning is raise when latitude larger than pi radians."""
    with pytest.warns(UserWarning):
        coriolis_parameter(50)
    with pytest.warns(UserWarning):
        coriolis_parameter(-50)
