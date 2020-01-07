# Copyright (c) 2008,2015,2016,2017,2018,2019,2020 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test station data information."""
from numpy.testing import assert_almost_equal
import pandas as pd

from metpy.io import add_station_lat_lon


def test_add_lat_lon_station_data():
    """Test for when the METAR does not correspond to a station in the dictionary."""
    df = pd.DataFrame({'station': ['KOUN', 'KVPZ', 'KDEN']})

    df = add_station_lat_lon(df, 'station')
    assert_almost_equal(df.loc[df.station == 'KOUN'].latitude.values, 35.25)
    assert_almost_equal(df.loc[df.station == 'KOUN'].longitude.values, -97.47)
    assert_almost_equal(df.loc[df.station == 'KVPZ'].latitude.values, 41.45)
    assert_almost_equal(df.loc[df.station == 'KVPZ'].longitude.values, -87)
    assert_almost_equal(df.loc[df.station == 'KDEN'].latitude.values, 39.85)
    assert_almost_equal(df.loc[df.station == 'KDEN'].longitude.values, -104.65)
