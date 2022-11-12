import numpy as np
import pandas as pd
import pytest
import xarray as xr

from metpy.calc import (add_height_to_pressure, add_pressure_to_height,
                        altimeter_to_sea_level_pressure, altimeter_to_station_pressure,
                        apparent_temperature, coriolis_parameter, geopotential_to_height,
                        heat_index, height_to_geopotential, height_to_pressure_std,
                        pressure_to_height_std, sigma_to_pressure, smooth_circular,
                        smooth_gaussian, smooth_n_point, smooth_rectangular, smooth_window,
                        wind_components, wind_direction, wind_speed, windchill, zoom_xarray)
from metpy.cbook import get_test_data
from metpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from metpy.units import units, masked_array

def test_wind_speed():
    """Test wind speed with scalars."""
    u = 0. * units('m/s')
    v = 4. * units('m/s')
    s = wind_speed(u, v)
    assert_almost_equal(s, 4. * units('m/s'), 3)

def test_direction_no_kwarg():
    """Test wind direction with no kwarg."""
    d = wind_direction(3. * units('m/s'), 4. * units('m/s'))
    assert_almost_equal(d, 216.870 * units.deg, 3)

def test_direction_kwarg_from():
    """Test wind direction with kwarg set to 'from'."""
    d = wind_direction(3. * units('m/s'), 4. * units('m/s'), convention="from")
    assert_almost_equal(d, 216.870 * units.deg, 3)

def test_direction_kwarg_to():
    """Test wind direction with kwarg set to 'to'."""
    d = wind_direction(3. * units('m/s'), 4. * units('m/s'), convention="to")
    assert_almost_equal(d, 36.869 * units.deg, 3)

def test_wind_components():
    """Test wind components calculation with scalars."""
    u, v = wind_components(20 * units('m/s'), 50 * units.deg)
    assert_almost_equal(u, -15.320 * units('m/s'), 3)
    assert_almost_equal(v, -12.855 * units('m/s'), 3)

def test_windchill():
    """Test wind chill with scalars."""
    wc = windchill(10 * units.degF, 40 * units('m/s'))
    assert_almost_equal(wc, -22.7053 * units.degF, 0)

def test_windchill_face_level():
    """Test wind chill with scalars."""
    wc = windchill(10 * units.degF, 40 * units('m/s'), face_level_winds="true")
    assert_almost_equal(wc, -27.0383 * units.degF, 0)

def test_heat_index_scalar():
    hi = heat_index(30 * units.degC, 50 * units.percent)
    assert_almost_equal(hi, 87 * units.degF, 0)

def test_apparent_temperature_face_level_false():
    """Test the apparent temperature calculation with a scalar."""
    at = apparent_temperature(30 * units.degC, 20 * units.percent, 20 * units.mph, face_level_winds=False)

    assert_almost_equal(at, 28.238435 * units.degC, 6)

def test_apparent_temperature_face_level_true():
    """Test the apparent temperature calculation with a scalar."""
    at = apparent_temperature(30 * units.degC, 20 * units.percent, 20 * units.mph, face_level_winds=True)

    assert_almost_equal(at, 28.238435 * units.degC, 6)
