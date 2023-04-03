# Copyright (c) 2008,2015,2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test various METARs."""
from datetime import datetime

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from metpy.cbook import get_test_data
from metpy.io import parse_metar_file, parse_metar_to_dataframe
from metpy.io._metar_parser.metar_parser import parse
from metpy.io.metar import Metar, parse_metar
from metpy.units import is_quantity, units


@pytest.mark.parametrize(['metar', 'truth'], [
    # Missing station
    ('METAR KLBG 261155Z AUTO 00000KT 10SM CLR 05/00 A3001 RMK AO2=',
     Metar('KLBG', np.nan, np.nan, np.nan, datetime(2017, 5, 26, 11, 55), 0, 0, np.nan,
           16093.44, np.nan, np.nan, np.nan, 'CLR', np.nan, np.nan, np.nan, np.nan, np.nan,
           np.nan, np.nan, 0, 5, 0, 30.01, 0, 0, 0, 'AO2')),
    # Broken clouds
    ('METAR KLOT 261155Z AUTO 00000KT 10SM BKN100 05/00 A3001 RMK AO2=',
     Metar('KLOT', 41.6, -88.1, 205, datetime(2017, 5, 26, 11, 55), 0, 0, np.nan, 16093.44,
           np.nan, np.nan, np.nan, 'BKN', 10000, np.nan, np.nan, np.nan, np.nan, np.nan,
           np.nan, 6, 5, 0, 30.01, 0, 0, 0, 'AO2')),
    # Few clouds, bad time and winds
    ('METAR KMKE 266155Z AUTO /////KT 10SM FEW100 05/00 A3001 RMK AO2=',
     Metar('KMKE', 42.95, -87.9, 206, np.nan, np.nan, np.nan, np.nan, 16093.44, np.nan, np.nan,
           np.nan, 'FEW', 10000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2, 5, 0,
           30.01, 0, 0, 0, 'AO2')),
    # Many weather and cloud slots taken
    ('METAR RJOI 261155Z 00000KT 4000 -SHRA BR VCSH BKN009 BKN015 OVC030 OVC040 22/21 A2987 '
     'RMK SHRAB35E44 SLP114 VCSH S-NW P0000 60021 70021 T02220206 10256 20211 55000=',
     Metar('RJOI', 34.13, 132.22, 2, datetime(2017, 5, 26, 11, 55), 0, 0, np.nan, 4000,
           '-SHRA', 'BR', 'VCSH', 'BKN', 900, 'BKN', 1500, 'OVC', 3000, 'OVC', 4000, 8, 22, 21,
           29.87, 80, 10, 16,
           'SHRAB35E44 SLP114 VCSH S-NW P0000 60021 70021 T02220206 10256 20211 55000')),
    # Smoke for current weather
    ('KFLG 252353Z AUTO 27005KT 10SM FU BKN036 BKN085 22/03 A3018 RMK AO2 SLP130 T02220033 '
     '10250 20217 55007=',
     Metar('KFLG', 35.13, -111.67, 2134, datetime(2017, 5, 25, 23, 53), 270, 5, np.nan,
           16093.44, 'FU', np.nan, np.nan, 'BKN', 3600, 'BKN', 8500, np.nan, np.nan, np.nan,
           np.nan, 6, 22, 3, 30.18, 4, 0, 0, 'AO2 SLP130 T02220033 10250 20217 55007')),
    # CAVOK for visibility group
    ('METAR OBBI 011200Z 33012KT CAVOK 40/18 Q0997 NOSIG=',
     Metar('OBBI', 26.27, 50.63, 2, datetime(2017, 5, 1, 12, 00), 330, 12, np.nan, 10000,
           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
           np.nan, 0, 40, 18, units.Quantity(997, 'hPa').m_as('inHg'), 0, 0, 0, 'NOSIG')),
    # Visibility using a mixed fraction
    ('K2I0 011155Z AUTO 05004KT 1 3/4SM BR SCT001 22/22 A3009 RMK AO2 70001 T02210221 10223 '
     '20208=',
     Metar('K2I0', 37.35, -87.4, 134, datetime(2017, 5, 1, 11, 55), 50, 4, np.nan, 2816.352,
           'BR', np.nan, np.nan, 'SCT', 100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4,
           22, 22, 30.09, 10, 0, 0, 'AO2 70001 T02210221 10223 20208')),
    # Missing temperature
    ('KIOW 011152Z AUTO A3006 RMK AO2 SLPNO 70020 51013 PWINO=',
     Metar('KIOW', 41.63, -91.55, 198, datetime(2017, 5, 1, 11, 52), np.nan, np.nan, np.nan,
           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
           np.nan, np.nan, 10, np.nan, np.nan, 30.06, 0, 0, 0, 'AO2 SLPNO 70020 51013 PWINO')),
    # Missing data
    ('METAR KBOU 011152Z AUTO 02006KT //// // ////// 42/02 Q1004=',
     Metar('KBOU', 40., -105.33, 1625, datetime(2017, 5, 1, 11, 52), 20, 6, np.nan, np.nan,
           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
           np.nan, 10, 42, 2, units.Quantity(1004, 'hPa').m_as('inHg'), 0, 0, 0, '')),
    # Vertical visibility
    ('KSLK 011151Z AUTO 21005KT 1/4SM FG VV002 14/13 A1013 RMK AO2 SLP151 70043 T01390133 '
     '10139 20094 53002=',
     Metar('KSLK', 44.4, -74.2, 498, datetime(2017, 5, 1, 11, 51), 210, 5, np.nan, 402.336,
           'FG', np.nan, np.nan, 'VV', 200, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8,
           14, 13, units.Quantity(1013, 'hPa').m_as('inHg'), 45, 0, 0,
           'AO2 SLP151 70043 T01390133 10139 20094 53002')),
    # Missing vertical visibility height
    ('SLCP 011200Z 18008KT 0100 FG VV/// 19/19 Q1019=',
     Metar('SLCP', -16.14, -62.02, 497, datetime(2017, 5, 1, 12, 00), 180, 8, np.nan, 100,
           'FG', np.nan, np.nan, 'VV', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
           8, 19, 19, units.Quantity(1019, 'hPa').m_as('inHg'), 45, 0, 0, '')),
    # BCFG current weather; also visibility is encoding 80SM which we're not adjusting
    ('METAR KMWN 011249Z 36037G45KT 80SM BCFG BKN/// FEW000 07/05 RMK BCFG FEW000 TPS LWR '
     'BKN037 BCFG INTMT=',
     Metar('KMWN', 44.27, -71.3, 1910, datetime(2017, 5, 1, 12, 49), 360, 37, 45,
           units.Quantity(80, 'mi').m_as('m'), 'BCFG', np.nan, np.nan, 'BKN', np.nan,
           'FEW', 0, np.nan, np.nan, np.nan, np.nan, 6, 7, 5, np.nan, 41, 0, 0,
           'BCFG FEW000 TPS LWR BKN037 BCFG INTMT')),
    # -DZ current weather
    ('KULM 011215Z AUTO 22003KT 10SM -DZ CLR 19/19 A3000 RMK AO2=',
     Metar('KULM', 44.32, -94.5, 308, datetime(2017, 5, 1, 12, 15), 220, 3, np.nan, 16093.44,
           '-DZ', np.nan, np.nan, 'CLR', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
           np.nan, 0, 19, 19, 30., 51, 0, 0, 'AO2')),
    # CB trailing on cloud group
    ('METAR AGGH 011200Z 25003KT 9999 FEW015 FEW017CB BKN030 25/24 Q1011=',
     Metar('AGGH', -9.42, 160.05, 9, datetime(2017, 5, 1, 12, 00), 250, 3., np.nan, 9999,
           np.nan, np.nan, np.nan, 'FEW', 1500, 'FEW', 1700, 'BKN', 3000, np.nan, np.nan, 6,
           25, 24, units.Quantity(1011, 'hPa').m_as('inHg'), 0, 0, 0, '')),
    # 5 levels of clouds
    ('METAR KSEQ 011158Z AUTO 08003KT 9SM FEW009 BKN020 BKN120 BKN150 OVC180 22/22 A3007 RMK '
     'AO2 RAB12E46RAB56E57 CIG 020V150 BKN020 V FEW SLP179 P0000 60000 70001 52008=',
     Metar('KSEQ', 29.566666666666666, -97.91666666666667, 160, datetime(2017, 5, 1, 11, 58),
           80, 3., np.nan, units.Quantity(9, 'miles').m_as('m'), np.nan, np.nan, np.nan, 'FEW',
           900., 'BKN', 2000., 'BKN', 12000., 'BKN', 15000., 8, 22., 22., 30.07, 0, 0, 0,
           'AO2 RAB12E46RAB56E57 CIG 020V150 BKN020 V FEW SLP179 P0000 60000 70001 52008')),
    # -FZUP
    ('SPECI CBBC 060030Z AUTO 17009G15KT 9SM -FZUP FEW011 SCT019 BKN026 OVC042 02/01 A3004 '
     'RMK ICG INTMT SLP177=',
     Metar('CBBC', 52.18, -128.15, 43, datetime(2017, 5, 6, 0, 30), 170, 9., 15.,
           units.Quantity(9, 'miles').m_as('m'), '-FZUP', np.nan, np.nan, 'FEW', 1100.,
           'SCT', 1900., 'BKN', 2600., 'OVC', 4200., 8, 2, 1, 30.04, 147, 0, 0,
           'ICG INTMT SLP177')),
    # Weird VV group and +SG
    ('BGGH 060750Z AUTO 36004KT 0100NDV +SG VV001/// 05/05 Q1000',
     Metar('BGGH', 64.2, -51.68, 70, datetime(2017, 5, 6, 7, 50), 360, 4, np.nan, 100, '+SG',
           np.nan, np.nan, 'VV', 100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8, 5, 5,
           units.Quantity(1000, 'hPa').m_as('inHg'), 77, 0, 0, '')),
    # COR at beginning, also wind MPS (m/s)
    ('COR ZLLL 101100Z 13010MPS 5000 -SHRA BLDU FEW033CB BKN046 21/11 Q1014 BECMG TL1240 '
     '04004MPS NSW',
     Metar('ZLLL', 36.52, 103.62, 1947, datetime(2017, 5, 10, 11, 0), 130,
           units.Quantity(10, 'm/s').m_as('knots'), np.nan, 5000, '-SHRA', 'BLDU', np.nan,
           'FEW', 3300, 'BKN', 4600, np.nan, np.nan, np.nan, np.nan, 6, 21, 11,
           units.Quantity(1014, 'hPa').m_as('inHg'), 80, 1007, 0,
           'BECMG TL1240 04004MPS NSW')),
    # M1/4SM vis, -VCTSSN weather
    ('K4BM 020127Z AUTO 04013G24KT 010V080 M1/4SM -VCTSSN OVC002 07/06 A3060 '
     'RMK AO2 LTG DSNT SE THRU SW',
     Metar('K4BM', 39.04, -105.52, 3438, datetime(2017, 5, 2, 1, 27), 40, 13, 24,
           units.Quantity(0.25, 'mi').m_as('m'), '-VCTSSN', np.nan, np.nan, 'OVC', 200,
           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8, 7, 6, 30.60, 2095, 0, 0,
           'AO2 LTG DSNT SE THRU SW')),
    # Variable visibility group
    ('ENBS 121620Z 36008KT 9999 3000N VCFG -DZ SCT006 BKN009 12/11 Q1014',
     Metar('ENBS', 70.62, 29.72, 10, datetime(2017, 5, 12, 16, 20), 360, 8, np.nan, 9999,
           'VCFG', '-DZ', np.nan, 'SCT', 600, 'BKN', 900, np.nan, np.nan, np.nan, np.nan, 6,
           12, 11, units.Quantity(1014, 'hPa').m_as('inHg'), 40, 51, 0, '')),
    # More complicated runway visibility
    ('CYYC 030047Z 26008G19KT 170V320 1SM R35L/5500VP6000FT/D R29/P6000FT/D R35R/P6000FT/D '
     '+TSRAGS BR OVC009CB 18/16 A2993 RMK CB8 FRQ LTGIC OVRHD PRESRR SLP127 DENSITY ALT '
     '4800FT',
     Metar('CYYC', 51.12, -114.02, 1084, datetime(2017, 5, 3, 0, 47), 260, 8, 19,
           units.Quantity(1, 'mi').m_as('m'), '+TSRAGS', 'BR', np.nan, 'OVC', 900, np.nan,
           np.nan, np.nan, np.nan, np.nan, np.nan, 8, 18, 16, 29.93, 99, 10, 0,
           'CB8 FRQ LTGIC OVRHD PRESRR SLP127 DENSITY ALT 4800FT')),
    # Oddly-placed COR
    ('KDMA 110240Z COR AUTO 08039G47KT 1/4SM -TSRA DS FEW008 BKN095 27/19 A2998 RMK AO2 '
     'RAB0159E20DZB20E27DZB27E27RAB35 TSB00E15TSB32 PRESFR SLP106 $ COR 0246',
     Metar('KDMA', 32.17, -110.87, 824, datetime(2017, 5, 11, 2, 40), 80, 39, 47, 402.336,
           '-TSRA', 'DS', np.nan, 'FEW', 800, 'BKN', 9500, np.nan, np.nan, np.nan, np.nan,
           6, 27, 19, 29.98, 1095, 31, 0,
           'AO2 RAB0159E20DZB20E27DZB27E27RAB35 TSB00E15TSB32 PRESFR SLP106 $ COR 0246')),
    # Ice crystals (IC) and no dewpoint -- South Pole!
    ('NZSP 052350Z 03009KT 3200 IC BLSN SCT019 M58/ A2874 RMK SKWY WNDS ESTMD CLN AIR 03005KT '
     'ALL WNDS GRID',
     Metar('NZSP', -89.98, 179.98, 2830, datetime(2017, 5, 5, 23, 50), 30, 9, np.nan, 3200,
           'IC', 'BLSN', np.nan, 'SCT', 1900, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
           4, -58, np.nan, 28.74, 78, 38, 0, 'SKWY WNDS ESTMD CLN AIR 03005KT ALL WNDS GRID')),
    # NSW
    ('VEIM 121200Z 09008KT 8000 NSW FEW018 SCT100 28/23 Q1008 BECMG 7000 NSW',
     Metar('VEIM', 24.77, 93.9, 781, datetime(2017, 5, 12, 12, 0), 90, 8, np.nan, 8000, np.nan,
           np.nan, np.nan, 'FEW', 1800, 'SCT', 10000, np.nan, np.nan, np.nan, np.nan, 4,
           28, 23, units.Quantity(1008, 'hPa').m_as('inHg'), 0, 0, 0, 'BECMG 7000 NSW')),
    # Variable vis with no direction
    ('TFFF 111830Z AUTO 11019G30KT 1000 0600 R10/1100D RA BCFG FEW014/// BKN021/// BKN027/// '
     '///CB 27/24 Q1015',
     Metar('TFFF', 14.6, -61.0, 5, datetime(2017, 5, 11, 18, 30), 110, 19, 30, 1000, 'RA',
           'BCFG', np.nan, 'FEW', 1400, 'BKN', 2100, 'BKN', 2700, np.nan, np.nan, 6, 27, 24,
           units.Quantity(1015, 'hPa').m_as('inHg'), 63, 41, 0, '')),
    # Interchanged wind and vis groups
    ('KBMI 121456Z COR 10SM 10005KT SCT055 OVC065 23/22 A3000 RMK AO2 '
     'LTG DSNT E AND SE TSB1358E13 SLP150 6//// T02280222 53011=',
     Metar('KBMI', 40.47, -88.92, 267, datetime(2017, 5, 12, 14, 56), np.nan, np.nan, np.nan,
           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
           np.nan, np.nan, 10, np.nan, np.nan, np.nan, 0, 0, 0,
           '5KT SCT055 OVC065 23/22 A3000 RMK AO2 LTG DSNT E AND SE TSB1358E13 SLP150 6//// '
           'T02280222 53011')),
    # Space between + and wx code
    ('SKCG 031730Z 13004KT 0500 + TSRA BKN010 25/25 Q1012 RMK A2990',
     Metar('SKCG', 10.43, -75.52, 1, datetime(2017, 5, 3, 17, 30), 130, 4, np.nan, 500,
           '+TSRA', np.nan, np.nan, 'BKN', 1000, np.nan, np.nan, np.nan, np.nan, np.nan,
           np.nan, 6, 25, 25, units.Quantity(1012, 'hPa').m_as('inHg'), 1097, 0, 0, 'A2990')),
    # Truncated VV group
    ('METAR ORER 172000Z 30006KT 0400 FG VV// 12/12 Q1013 NOSIG=',
     Metar('ORER', 36.22, 43.97, 409, datetime(2017, 5, 17, 20, 0), 300, 6.0, np.nan,
           400, 'FG', np.nan, np.nan, 'VV', np.nan, np.nan, np.nan, np.nan, np.nan,
           np.nan, np.nan, 8, 12, 12, units.Quantity(1013, 'hPa').m_as('inHg'), 45, 0, 0,
           'NOSIG')),
    # Invalid Visibility Unidata/Metpy#2652
    ('KGYR 072147Z 12006KT 1/0SM FEW100 SCT250 41/14 A2992',
     Metar('KGYR', 33.42, -112.37, 295, datetime(2017, 5, 7, 21, 47), 120, 6.0, np.nan,
           np.nan, np.nan, np.nan, np.nan, 'FEW', 10000, 'SCT', 25000, np.nan, np.nan,
           np.nan, np.nan, 4, 41, 14, 29.92, 0, 0, 0,
           '')),
    # Manual visibility can be [1,3,5]/16SM Unidata/Metpy#2807
    ('KDEN 241600Z 02010KT 1/16SM R35L/1000V1200FT FZFG VV001 M01/M02 A2954 RMK AO2 SFC VIS '
     'M1/4 T10111022',
     Metar('KDEN', 39.85, -104.65, 1640, datetime(2017, 5, 24, 16, 00), 20, 10.0, np.nan,
           units.Quantity(1 / 16, 'mi').m_as('m'), 'FZFG', np.nan, np.nan, 'VV', 100,
           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8, -1, -2, 29.54, 49, 0, 0,
           'AO2 SFC VIS M1/4 T10111022'))],
    ids=['missing station', 'BKN', 'FEW', 'current weather', 'smoke', 'CAVOK', 'vis fraction',
         'missing temps', 'missing data', 'vertical vis', 'missing vertical vis', 'BCFG',
         '-DZ', 'sky cover CB', '5 sky levels', '-FZUP', 'VV group', 'COR placement',
         'M1/4SM vis', 'variable vis', 'runway vis', 'odd COR', 'IC', 'NSW',
         'variable vis no dir', 'swapped wind and vis', 'space in wx code', 'truncated VV',
         'vis div zero', 'vis 1/16'])
def test_metar_parser(metar, truth):
    """Test parsing individual METARs."""
    assert parse_metar(metar, 2017, 5) == truth


def test_date_time_given():
    """Test for when date_time is given."""
    df = parse_metar_to_dataframe('K6B0 261200Z AUTO 00000KT 10SM CLR 20/M17 A3002 RMK AO2 '
                                  'T01990165=', year=2019, month=6)
    assert df.date_time[0] == datetime(2019, 6, 26, 12)
    assert df.eastward_wind[0] == 0
    assert df.northward_wind[0] == 0
    assert_almost_equal(df.air_pressure_at_sea_level[0], 1016.56)
    assert_almost_equal(df.visibility.values, 16093.44)


def test_parse_metar_df_positional_datetime_failure():
    """Test that positional year, month arguments fail for parse_metar_to_dataframe."""
    # pylint: disable=too-many-function-args
    with pytest.raises(TypeError, match='takes 1 positional argument but 3 were given'):
        parse_metar_to_dataframe('K6B0 261200Z AUTO 00000KT 10SM CLR 20/M17'
                                 'A3002 RMK AO2 T01990165=', 2019, 6)


def test_parse_metar_to_dataframe():
    """Test parsing a single METAR to a DataFrame."""
    df = parse_metar_to_dataframe('KDEN 012153Z 09010KT 10SM FEW060 BKN110 BKN220 27/13 '
                                  'A3010 RMK AO2 LTG DSNT SW AND W SLP114 OCNL LTGICCG '
                                  'DSNT SW CB DSNT SW MOV E T02670128')
    assert df.wind_direction.values == 90
    assert df.wind_speed.values == 10
    assert_almost_equal(df.eastward_wind.values, -10)
    assert_almost_equal(df.northward_wind.values, 0)
    assert_almost_equal(df.visibility.values, 16093.44)
    assert df.air_temperature.values == 27
    assert df.dew_point_temperature.values == 13


def test_parse_file():
    """Test the parser on an entire file."""
    input_file = get_test_data('metar_20190701_1200.txt', as_file_obj=False)
    df = parse_metar_file(input_file)

    # Check counts (non-NaN) of various fields
    counts = df.count()
    assert counts.station_id == 8980
    assert counts.latitude == 8968
    assert counts.longitude == 8968
    assert counts.elevation == 8968
    assert counts.date_time == 8980
    assert counts.wind_direction == 8577
    assert counts.wind_speed == 8844
    assert counts.wind_gust == 347
    assert counts.visibility == 8486
    assert counts.current_wx1 == 1090
    assert counts.current_wx2 == 82
    assert counts.current_wx3 == 1
    assert counts.low_cloud_type == 7361
    assert counts.low_cloud_level == 3867
    assert counts.medium_cloud_type == 1646
    assert counts.medium_cloud_level == 1641
    assert counts.high_cloud_type == 632
    assert counts.high_cloud_level == 626
    assert counts.highest_cloud_type == 37
    assert counts.highest_cloud_level == 37
    assert counts.cloud_coverage == 8980
    assert counts.air_temperature == 8779
    assert counts.dew_point_temperature == 8740
    assert counts.altimeter == 8458
    assert counts.remarks == 8980
    assert (df.current_wx1_symbol != 0).sum() == counts.current_wx1
    assert (df.current_wx2_symbol != 0).sum() == counts.current_wx2
    assert (df.current_wx3_symbol != 0).sum() == counts.current_wx3
    assert counts.air_pressure_at_sea_level == 8378
    assert counts.eastward_wind == 8577
    assert counts.northward_wind == 8577

    # KVPZ 011156Z AUTO 27005KT 10SM CLR 23/19 A3004 RMK AO2 SLP166
    test = df[df.station_id == 'KVPZ']
    assert test.air_temperature.values == 23
    assert test.dew_point_temperature.values == 19
    assert test.altimeter.values == 30.04
    assert_almost_equal(test.eastward_wind.values, 5)
    assert_almost_equal(test.northward_wind.values, 0)
    assert test.air_pressure_at_sea_level.values == 1016.76

    # Check that this ob properly gets all lines
    paku = df[df.station_id == 'PAKU']
    assert_almost_equal(paku.air_temperature.values, [9, 12])
    assert_almost_equal(paku.dew_point_temperature.values, [9, 10])
    assert_almost_equal(paku.altimeter.values, [30.02, 30.04])


def test_parse_file_positional_datetime_failure():
    """Test that positional year, month arguments fail for parse_metar_file."""
    # pylint: disable=too-many-function-args
    input_file = get_test_data('metar_20190701_1200.txt', as_file_obj=False)
    with pytest.raises(TypeError, match='takes 1 positional argument but 3 were given'):
        parse_metar_file(input_file, 2016, 12)


def test_parse_file_bad_encoding():
    """Test the parser on an entire file that has at least one bad utf-8 encoding."""
    input_file = get_test_data('2020010600_sao.wmo', as_file_obj=False)
    df = parse_metar_file(input_file)

    # Check counts (non-NaN) of various fields
    counts = df.count()
    assert counts.station_id == 8802
    assert counts.latitude == 8789
    assert counts.longitude == 8789
    assert counts.elevation == 8789
    assert counts.date_time == 8802
    assert counts.wind_direction == 8377
    assert counts.wind_speed == 8673
    assert counts.wind_gust == 1053
    assert counts.visibility == 8312
    assert counts.current_wx1 == 1412
    assert counts.current_wx2 == 213
    assert counts.current_wx3 == 3
    assert counts.low_cloud_type == 7672
    assert counts.low_cloud_level == 3816
    assert counts.medium_cloud_type == 1632
    assert counts.medium_cloud_level == 1623
    assert counts.high_cloud_type == 546
    assert counts.high_cloud_level == 545
    assert counts.highest_cloud_type == 40
    assert counts.highest_cloud_level == 40
    assert counts.cloud_coverage == 8802
    assert counts.air_temperature == 8597
    assert counts.dew_point_temperature == 8536
    assert counts.altimeter == 8252
    assert counts.remarks == 8802
    assert (df.current_wx1_symbol != 0).sum() == counts.current_wx1
    assert (df.current_wx2_symbol != 0).sum() == counts.current_wx2
    assert (df.current_wx3_symbol != 0).sum() == counts.current_wx3
    assert counts.air_pressure_at_sea_level == 8207
    assert counts.eastward_wind == 8377
    assert counts.northward_wind == 8377

    # KDEN 052353Z 16014KT 10SM FEW120 FEW220 02/M07 A3008 RMK AO2 SLP190 T00171072...
    test = df[df.station_id == 'KDEN']
    assert_almost_equal(test.visibility.values, 16093.44)
    assert test.air_temperature.values == 2
    assert test.air_pressure_at_sea_level.values == 1024.71


def test_parse_file_object():
    """Test the parser reading from a file-like object."""
    input_file = get_test_data('metar_20190701_1200.txt', mode='rt')
    # KOKC 011152Z 18006KT 7SM FEW080 FEW250 21/21 A3003 RMK AO2 SLP155 T02060206...
    df = parse_metar_file(input_file)
    test = df[df.station_id == 'KOKC']
    assert_almost_equal(test.visibility.values, 11265.408)
    assert test.air_temperature.values == 21
    assert test.dew_point_temperature.values == 21
    assert test.altimeter.values == 30.03
    assert_almost_equal(test.eastward_wind.values, 0)
    assert_almost_equal(test.northward_wind.values, 6)


def test_parse_no_pint_objects_in_df():
    """Test that there are no Pint quantities in dataframes created by parser."""
    input_file = get_test_data('metar_20190701_1200.txt', mode='rt')
    metar_str = ('KSLK 011151Z AUTO 21005KT 1/4SM FG VV002 14/13 A1013 RMK AO2 SLP151 70043 '
                 'T01390133 10139 20094 53002=')

    for df in (parse_metar_file(input_file), parse_metar_to_dataframe(metar_str)):
        for column in df:
            assert not is_quantity(df[column][0])


def test_repr():
    """Test that the TreeNode string representation works."""
    str1 = 'KSMQ 201953Z AUTO VRB05KT 10SM CLR 03/M10 A3026'
    tree = parse(str1)
    rep = repr(tree)
    assert rep == ("TreeNode1(text='KSMQ 201953Z AUTO VRB05KT 10SM CLR 03/M10 A3026', "
                   "offset=0, metar=TreeNode(text='', offset=0), "
                   "siteid=TreeNode(text='KSMQ', offset=0), "
                   "datetime=TreeNode2(text=' 201953Z', offset=4, "
                   "sep=TreeNode(text=' ', offset=4)), "
                   "auto=TreeNode(text=' AUTO', offset=12), "
                   "wind=TreeNode4(text=' VRB05KT', offset=17, "
                   "wind_dir=TreeNode(text='VRB', offset=18), "
                   "wind_spd=TreeNode(text='05', offset=21), "
                   "gust=TreeNode(text='', offset=23)), "
                   "vis=TreeNode6(text=' 10SM', offset=25, "
                   "sep=TreeNode(text=' ', offset=25)), run=TreeNode(text='', offset=30), "
                   "curwx=TreeNode(text='', offset=30), "
                   "skyc=TreeNode(text=' CLR', offset=30), "
                   "temp_dewp=TreeNode13(text=' 03/M10', offset=34, "
                   "sep=TreeNode(text=' ', offset=34), temp=TreeNode(text='03', offset=35), "
                   "dewp=TreeNode(text='M10', offset=38)), "
                   "altim=TreeNode(text=' A3026', offset=41), "
                   "remarks=TreeNode(text='', offset=47), end=TreeNode(text='', offset=47))")
