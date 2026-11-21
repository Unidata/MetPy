# Copyright (c) 2020,2026 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test station data information."""
from pathlib import Path

import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import pytest

from metpy.io import add_station_lat_lon, station_info


def test_add_lat_lon_station_data():
    """Test for when the METAR does not correspond to a station in the dictionary."""
    df = pd.DataFrame({'station': ['KOUN', 'KVPZ', 'KDEN', 'PAAA']})

    df = add_station_lat_lon(df, 'station')
    assert_almost_equal(df.loc[df.station == 'KOUN'].latitude.values[0], 35.25)
    assert_almost_equal(df.loc[df.station == 'KOUN'].longitude.values[0], -97.47)
    assert_almost_equal(df.loc[df.station == 'KVPZ'].latitude.values[0], 41.45)
    assert_almost_equal(df.loc[df.station == 'KVPZ'].longitude.values[0], -87)
    assert_almost_equal(df.loc[df.station == 'KDEN'].latitude.values[0], 39.85)
    assert_almost_equal(df.loc[df.station == 'KDEN'].longitude.values[0], -104.65)
    assert_almost_equal(df.loc[df.station == 'PAAA'].latitude.values[0], np.nan)
    assert_almost_equal(df.loc[df.station == 'PAAA'].longitude.values[0], np.nan)
    assert df['longitude'].dtype == np.float64


def test_add_lat_lon_station_data_optional():
    """Test for when only one argument is passed."""
    df = pd.DataFrame({'station': ['KOUN', 'KVPZ', 'KDEN', 'PAAA']})

    df = add_station_lat_lon(df)
    assert_almost_equal(df.loc[df.station == 'KOUN'].latitude.values[0], 35.25)


def test_add_lat_lon_station_data_not_found():
    """Test case where key cannot be inferred."""
    df = pd.DataFrame({'location': ['KOUN']})

    with pytest.raises(KeyError):
        add_station_lat_lon(df)


def test_add_lat_lon_station_data_existing_col():
    """Test will fail if one of these columns exists already."""
    df = pd.DataFrame({'station': ['KOUN', 'KVPZ'],
                       'latitude': [44, 42]})

    with pytest.raises(ValueError, match='cannot insert'):
        add_station_lat_lon(df)


def test_station_lookup_get_station():
    """Test that you can get a station by ID from the lookup."""
    assert station_info['KOUN'].id == 'KOUN'


def test_station_lookup_airport_codes():
    """Airport codes are indexed as station records in the lookup."""
    info = station_info['00A']
    assert info.id == '00A'
    assert info.synop_id == 99999
    assert info.name == 'Total Rf Heliport'
    assert info.state == 'PA'
    assert info.country == 'US'
    assert_almost_equal(info.latitude, 40.07080078)
    assert_almost_equal(info.longitude, -74.93360138)
    assert_almost_equal(info.altitude, 3.3528, decimal=4)

    missing_elevation = station_info['01MD']
    assert missing_elevation.id == '01MD'
    assert np.isnan(missing_elevation.altitude)
    assert missing_elevation.state == 'MD'
    assert missing_elevation.country == 'US'


def test_station_lookup_all_definitions():
    """Every source definition of a station ID is available, in priority order."""
    definitions = station_info.lookup_all('FYOW')
    assert [Path(station.source).name for station in definitions] == [
        'sfstns.tbl', 'master.txt', 'stations.txt', 'airport-codes.csv'
    ]
    assert definitions[0] == station_info['FYOW']
    assert definitions[0].synop_id == 999999
    assert definitions[1].synop_id == '68004'
    assert station_info.lookup_all('NOT_A_STATION') == []


def test_station_find_synop_id():
    """SYNOP lookup searches every source, including lower-priority definitions."""
    for synop_id in (68004, '68004', 68004.0):
        matches = station_info.find(synop_id=synop_id)
        assert len(matches) == 1
        info = matches[0]
        assert info.id == 'FYOW'
        assert info.synop_id == '68004'
        assert info.source.endswith('master.txt')
        assert_almost_equal(info.latitude, -20.45)
        assert_almost_equal(info.longitude, 16 + 38 / 60)

    # Indexing still returns the highest-priority record, which lacks this SYNOP id.
    assert station_info['FYOW'].synop_id == 999999


def test_station_find_preserves_text_identifiers():
    """Zero-padded SYNOP text is not compared as an integer."""
    matches = station_info.find(synop_id='06450')
    assert matches
    assert all(station.synop_id == '06450' for station in matches)
    assert matches[0].id == 'EBAW'
    assert matches[0].source.endswith('master.txt')


def test_station_find_requires_criterion():
    """Searching with no metadata is rejected."""
    with pytest.raises(ValueError, match='search criterion'):
        station_info.find()


def test_station_find_returns_matches_in_source_order():
    """Repeated SYNOP ids follow the same source priority as the lookup."""
    matches = station_info.find(synop_id='68018')
    assert [Path(station.source).name for station in matches] == [
        'master.txt', 'stations.txt'
    ]
    assert all(station.id == 'FYRU' for station in matches)


def test_add_lat_lon_synop_id():
    """WMO/SYNOP identifiers can be located when lookup is explicit."""
    df = pd.DataFrame({'station': [68004, 67221, 67221, 68007]})

    df = add_station_lat_lon(df, lookup='synop_id')
    assert_almost_equal(df.loc[0, 'latitude'], -20.45)
    assert_almost_equal(df.loc[0, 'longitude'], 16 + 38 / 60)
    assert_almost_equal(df.loc[1, 'latitude'], -(13 + 43 / 60))
    assert_almost_equal(df.loc[1, 'longitude'], 37 + 32 / 60)
    assert_almost_equal(df.loc[2, 'latitude'], -(13 + 43 / 60))
    assert_almost_equal(df.loc[3, 'latitude'], -(17 + 28 / 60))
    assert_almost_equal(df.loc[3, 'longitude'], 16 + 19 / 60)


def test_add_lat_lon_synop_id_string_float_and_placeholder():
    """String and float SYNOP ids match, and placeholder ids stay unresolved."""
    df = pd.DataFrame({'station': ['68004', 68004.0, 999999, '99999', '']})

    df = add_station_lat_lon(df, 'station', lookup='synop_id')
    assert_almost_equal(df.loc[0, 'latitude'], -20.45)
    assert_almost_equal(df.loc[1, 'latitude'], -20.45)
    assert np.isnan(df.loc[2, 'latitude'])
    assert np.isnan(df.loc[3, 'latitude'])
    assert np.isnan(df.loc[4, 'latitude'])


def test_add_lat_lon_default_lookup_does_not_search_synop_id():
    """Station-id lookup does not treat a SYNOP number as a station ID."""
    df = pd.DataFrame({'station': [68004]})

    df = add_station_lat_lon(df)
    assert np.isnan(df.loc[0, 'latitude'])
    assert np.isnan(df.loc[0, 'longitude'])


def test_add_lat_lon_synop_lookup_does_not_use_station_id():
    """SYNOP lookup does not fall back to station identifiers."""
    df = pd.DataFrame({'station': ['KOUN']})

    df = add_station_lat_lon(df, lookup='synop_id')
    assert np.isnan(df.loc[0, 'latitude'])


def test_add_lat_lon_synop_uses_highest_priority_source():
    """Shared SYNOP ids take coordinates from the earliest source."""
    matches = station_info.find(synop_id='68018')
    df = add_station_lat_lon(pd.DataFrame({'station': ['68018']}), lookup='synop_id')
    assert_almost_equal(df.loc[0, 'latitude'], matches[0].latitude)
    assert_almost_equal(df.loc[0, 'longitude'], matches[0].longitude)


def test_add_lat_lon_invalid_lookup():
    """Unknown lookup modes are rejected before the DataFrame is modified."""
    df = pd.DataFrame({'station': ['KOUN']})

    with pytest.raises(ValueError, match='lookup'):
        add_station_lat_lon(df, lookup='name')
    assert list(df.columns) == ['station']


def test_station_lookup_len():
    """Test that you can get the length of the station data."""
    assert len(station_info) == 50340


def test_station_lookup_iter():
    """Test iterating over the station data."""
    for stid in station_info:
        assert stid in station_info
