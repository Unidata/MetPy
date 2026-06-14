#!/usr/bin/python
# -*-coding:utf-8 -*-
"""Testing program for the MetPy boundary layer module"""

import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.calc import boundarylayer
from metpy.cbook import get_test_data
from metpy.units import units

# SAMPLE DATA
# ===========
col_names = ["pressure", "height", "temperature", "dewpoint", "direction", "speed"]

df = pd.read_fwf(
    get_test_data("may4_sounding.txt", as_file_obj=False),
    skiprows=5,
    usecols=[0, 1, 2, 3, 6, 7],
    names=col_names,
)

# Drop any rows with all NaN values for T, Td, winds
df = df.dropna(
    subset=("temperature", "dewpoint", "direction", "speed"), how="all"
).reset_index(drop=True)

height = df["height"].values * units.metres
pressure = df["pressure"].values * units.hPa
temperature = df["temperature"].values * units.degC
dewpoint = df["dewpoint"].values * units.degC
wind_speed = df["speed"].values * units.knots
wind_dir = df["direction"].values * units.degrees

u, v = mpcalc.wind_components(wind_speed, wind_dir)
relative_humidity = mpcalc.relative_humidity_from_dewpoint(temperature, dewpoint)
potential_temperature = mpcalc.potential_temperature(pressure, temperature)
specific_humidity = mpcalc.specific_humidity_from_dewpoint(pressure, dewpoint)


# BOUNDARY LAYER HEIGHT ESTIMATIONS
# =================================

def test_blh_from_richardson_bulk():
    blh = boundarylayer.blh_from_richardson_bulk(height, potential_temperature, u, v)
    blh_true = 1397 * units.meter
    assert blh == blh_true


def test_blh_from_parcel():
    blh = boundarylayer.blh_from_parcel(height, potential_temperature)
    blh_true = 610 * units.meter
    assert blh == blh_true


def test_blh_from_concentration_gradient():
    blh = boundarylayer.blh_from_concentration_gradient(height, specific_humidity)
    blh_true = 1766 * units.meter
    assert blh == blh_true


def test_blh_from_temperature_inversion():
    blh = boundarylayer.blh_from_temperature_inversion(height, potential_temperature)
    blh_true = 610 * units.meter
    assert blh == blh_true
