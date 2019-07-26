# Copyright (c) 2008,2015,2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test various metars."""
from datetime import datetime

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from metpy.cbook import get_test_data
from metpy.io import metar, surface_station_data


def test_station_id_not_in_dictionary():
    """Test for when the METAR does not correspond to a station in the dictionary."""
    df = metar.parse_metar_to_pandas('METAR KLBG 261155Z AUTO 00000KT 10SM CLR 05/00 A3001 \
    RMK AO2=')
    assert df.station_id.values == 'KLBG'
    assert_almost_equal(df.latitude.values, np.nan)
    assert_almost_equal(df.longitude.values, np.nan)
    assert_almost_equal(df.elevation.values, np.nan)
    assert_almost_equal(df.wind_direction.values, 0)
    assert_almost_equal(df.wind_speed.values, 0)


def test_broken_clouds():
    """Test for skycover when there are broken clouds."""
    df = metar.parse_metar_to_pandas('METAR KLOT 261155Z AUTO 00000KT 10SM BKN100 05/00 A3001 \
    RMK AO2=')
    assert df.skyc1.values == 'BKN'
    assert df.cloudcover.values == 6


def test_few_clouds_():
    """Test for skycover when there are few clouds."""
    df = metar.parse_metar_to_pandas('METAR KMKE 261155Z AUTO /////KT 10SM FEW100 05/00 A3001 \
    RMK AO2=')
    assert df.skyc1.values == 'FEW'
    assert df.cloudcover.values == 2
    assert_almost_equal(df.wind_direction.values, np.nan)
    assert_almost_equal(df.wind_speed.values, np.nan)


def test_all_weather_given():
    """Test when all possible weather slots are given."""
    df = metar.parse_metar_to_pandas('METAR RJOI 261155Z 00000KT 4000 -SHRA BR VCSH BKN009 \
    BKN015 OVC030 OVC040 22/21 A2987 RMK SHRAB35E44 SLP114 VCSH S-NW P0000 60021 70021 \
    T02220206 10256 20211 55000=')
    assert df.station_id.values == 'RJOI'
    assert_almost_equal(df.latitude.values, 34.14, decimal=2)
    assert_almost_equal(df.longitude.values, 132.24, decimal=2)
    assert df.current_wx1.values == '-SHRA'
    assert df.current_wx2.values == 'BR'
    assert df.current_wx3.values == 'VCSH'
    assert df.skyc1.values == 'BKN'
    assert df.skylev1.values == 900
    assert df.skyc3.values == 'OVC'
    assert df.skylev3.values == 3000


def test_missing_temp_dewp():
    """Test when missing both temperature and dewpoint."""
    df = metar.parse_metar_to_pandas('KIOW 011152Z AUTO A3006 RMK AO2 SLPNO 70020 51013 \
    PWINO=')
    assert_almost_equal(df.temperature.values, np.nan)
    assert_almost_equal(df.dewpoint.values, np.nan)
    assert_almost_equal(df.cloudcover.values, 10)


def test_missing_values():
    """Test for missing values from nearly every field."""
    df = metar.parse_metar_to_pandas('METAR KBOU 011152Z AUTO 02006KT //// // \
    ////// 42/02 Q1004=')
    assert_almost_equal(df.current_wx1.values, np.nan)
    assert_almost_equal(df.current_wx2.values, np.nan)
    assert_almost_equal(df.current_wx3.values, np.nan)
    assert_almost_equal(df.current_wx1_symbol.values, 0)
    assert_almost_equal(df.current_wx2_symbol.values, 0)
    assert_almost_equal(df.current_wx3_symbol.values, 0)
    assert_almost_equal(df.skyc1.values, np.nan)
    assert_almost_equal(df.skyc2.values, np.nan)
    assert_almost_equal(df.skyc3.values, np.nan)


def test_vertical_vis():
    """Test for when vertical visibility is given."""
    df = metar.parse_metar_to_pandas('KSLK 011151Z AUTO 21005KT 1/4SM FG VV002 14/13 \
        A1013 RMK AO2 SLP151 70043 T01390133 10139 20094 53002=')
    assert df.skyc1.values == 'VV'


def test_date_time_given():
    """Test for when date_time is given."""
    df = metar.parse_metar_to_pandas('K6B0 261200Z AUTO 00000KT 10SM CLR 20/M17 A3002 \
                                     RMK AO2 T01990165=', year=2019, month=6)
    assert_equal(df['date_time'][0], datetime(2019, 6, 26, 12))


def test_named_tuple_test1():
    """Test the named tuple parsing function."""
    df = metar.parse_metar_to_named_tuple('KDEN 012153Z 09010KT 10SM FEW060 BKN110 \
                                           BKN220 27/13 A3010 RMK AO2 LTG DSNT SW AND \
                                           W SLP114 OCNL LTGICCG DSNT SW CB DSNT \
                                           SW MOV E T02670128',
                                          surface_station_data.station_dict())
    assert_equal(df.wind_direction, 90)
    assert_equal(df.wind_speed, 10)
    assert_equal(df.temperature, 27)
    assert_equal(df.dewpoint, 13)


def test_file_test(input_file=None):
    """Test the parser on an entire file."""
    if input_file is None:
        input_file = get_test_data('metar_20190701_1200.txt', as_file_obj=False)
    df = metar.text_file_parse(input_file)
    test = df[df.station_id == 'KVPZ']
    assert test.temperature.values == 23
