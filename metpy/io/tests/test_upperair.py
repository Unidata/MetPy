# Copyright (c) 2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `upper_air` module."""

from datetime import datetime
import warnings

from metpy.deprecation import MetpyDeprecationWarning
from metpy.io import get_upper_air_data
from metpy.io.upperair import UseSampleData
from metpy.testing import assert_almost_equal
from metpy.units import units

warnings.simplefilter('ignore', MetpyDeprecationWarning)


def test_wyoming():
    """Test that we are properly parsing data from the wyoming archive."""
    with UseSampleData():
        data = get_upper_air_data(datetime(1999, 5, 4, 0), 'OUN', source='wyoming')

    assert_almost_equal(data.variables['pressure'][5], 867.9 * units('hPa'), 2)
    assert_almost_equal(data.variables['height'][5], 1219. * units('meter'), 2)
    assert_almost_equal(data.variables['temperature'][5], 17.4 * units.degC, 2)
    assert_almost_equal(data.variables['dewpoint'][5], 14.3 * units.degC, 2)
    assert_almost_equal(data.variables['u_wind'][5], 6.60 * units.knot, 2)
    assert_almost_equal(data.variables['v_wind'][5], 37.42 * units.knot, 2)
    assert_almost_equal(data.variables['speed'][5], 38.0 * units.knot, 1)
    assert_almost_equal(data.variables['direction'][5], 190.0 * units.deg, 1)


def test_iastate():
    """Test that we properly parse data from Iowa State archive."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 7, 30, 12), 'KDEN', source='iastate')

    assert_almost_equal(data.variables['pressure'][0], 838.0 * units('hPa'), 2)
    assert_almost_equal(data.variables['height'][0], 1611. * units('meter'), 2)
    assert_almost_equal(data.variables['temperature'][0], 17.0 * units.degC, 2)
    assert_almost_equal(data.variables['dewpoint'][0], 15.2 * units.degC, 2)
    assert_almost_equal(data.variables['u_wind'][0], 1.72 * units.knot, 2)
    assert_almost_equal(data.variables['v_wind'][0], 2.46 * units.knot, 2)
    assert_almost_equal(data.variables['speed'][0], 3.0 * units.knot, 1)
    assert_almost_equal(data.variables['direction'][0], 215.0 * units.deg, 1)


def test_high_alt_wyoming():
    """Test Wyoming data that starts at pressure less than 925 hPa."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2010, 12, 9, 12), 'BOI', source='wyoming')

    assert_almost_equal(data.variables['pressure'][2], 890.0 * units('hPa'), 2)
    assert_almost_equal(data.variables['height'][2], 1133. * units('meter'), 2)
    assert_almost_equal(data.variables['temperature'][2], 5.4 * units.degC, 2)
    assert_almost_equal(data.variables['dewpoint'][2], 3.9 * units.degC, 2)
    assert_almost_equal(data.variables['u_wind'][2], -0.42 * units.knot, 2)
    assert_almost_equal(data.variables['v_wind'][2], 5.99 * units.knot, 2)
