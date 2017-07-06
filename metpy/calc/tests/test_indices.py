# Copyright (c) 2008-2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `indices` module."""

from datetime import datetime

from metpy.calc import precipitable_water
from metpy.io import get_upper_air_data
from metpy.io.upperair import UseSampleData
from metpy.testing import assert_array_equal
from metpy.units import units


def test_precipitable_water():
    """Test precipitable water with observed sounding."""
    with UseSampleData():
        data = get_upper_air_data(datetime(2016, 5, 22, 0), 'DDC', source='wyoming')
    pw = precipitable_water(data.variables['dewpoint'][:], data.variables['pressure'][:])
    truth = (0.8899441949243486 * units('inches')).to('millimeters')
    assert_array_equal(pw, truth)
