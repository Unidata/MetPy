from metpy.io import metar, surface_station_data
from metpy.units import units
from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_equal
import numpy as np
from datetime import datetime


def station_id_not_in_dictionary():
    df = metar.parse_metar_to_pandas("METAR KHCR 261155Z AUTO 00000KT 10SM CLR 05/00 A3001 \
    RMK AO2=")
    assert df.station_id.values == "KHCR"
    assert_almost_equal(df.latitude.values, np.nan)
    assert_almost_equal(df.longitude.values, np.nan)
    assert_almost_equal(df.elevation.values, np.nan)
    assert_almost_equal(df.wind_direction.values, 0)
    assert_almost_equal(df.wind_speed.values, 0)


def all_weather_given():
    df = metar.parse_metar_to_pandas("METAR RJOI 261155Z 00000KT 4000 -SHRA BR VCSH BKN009 \
    BKN015 OVC030 22/21 A2987 RMK SHRAB35E44 SLP114 VCSH S-NW P0000 60021 70021 \
    T02220206 10256 20211 55000=")
    assert df.station_id.values == 'RJOI'
    assert_almost_equal(df.latitude.values, 34.14, decimal = 2)
    assert_almost_equal(df.longitude.values, 132.24, decimal = 2)
    assert df.current_wx1.values == '-SHRA'
    assert df.current_wx2.values == 'BR'
    assert df.current_wx3.values == 'VCSH'
    assert df.skyc1.values == 'BKN'
    assert df.skylev1.values == 900
    assert df.skyc3.values == 'OVC'
    assert df.skylev3.values == 3000


def date_time_given():
    df = metar.parse_metar_to_pandas("K6B0 261200Z AUTO 00000KT 10SM CLR 20/17 A3002 \
    RMK AO2 T01990165=", year= 2019, month = 6)
    assert_equal(df['date_time'][0], datetime(2019, 6, 26, 12))


def named_tuple_test1():
    df = metar.parse_metar_to_named_tuple("KDEN 012153Z 09010KT 10SM FEW060 BKN110 \
    BKN220 27/13 A3010 RMK AO2 LTG DSNT SW AND W SLP114 OCNL LTGICCG DSNT SW CB DSNT \
    SW MOV E T02670128", surface_station_data.station_dict())
    assert_equal(df.wind_direction, 90)
    assert_equal(df.wind_speed, 10)
    assert_equal(df.temperature, 27)
    assert_equal(df.dewpoint, 13)

def file_test():
    df = metar.text_file_parse('metar_20190701_1200.txt')
    test = df[df.station_id == 'KVPZ']
    assert test.temperature.values == 23


if __name__ == '__main__':
    station_id_not_in_dictionary()
    all_weather_given()
    date_time_given()
    named_tuple_test1()
    print("Everything Passed")
