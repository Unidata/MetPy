# Copyright (c) 2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
=====================
Hydrostatic Thickness
=====================

Use functions from `metpy.calc` as well as pint's unit support to perform calculations.

The code below uses example data from our test suite to calculate the hydrostatic thickness
between the surface and 500-hPa level for the provided sounding data.
"""
import pandas as pd

from metpy.calc import (mixing_ratio_from_relative_humidity, relative_humidity_from_dewpoint,
                        thickness_hydrostatic)
from metpy.cbook import get_test_data
from metpy.units import units

###########################################
# Upper air data can be obtained using the siphon package, but for this example we will use
# some of MetPy's sample data.

# Set column names
col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']

# Read in test data using col_names
df = pd.read_fwf(get_test_data('jan20_sounding.txt', as_file_obj=False),
                 skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)

###########################################
# Drop any rows with all NaN values for T, Td, winds
df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'
                       ), how='all').reset_index(drop=True)

###########################################
# Isolate pressure, temperature, and dewpoint and add units
p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC

###########################################
# Calculate the relative humidity to compute the mixing ratio
rh = relative_humidity_from_dewpoint(T, Td)
mixrat = mixing_ratio_from_relative_humidity(p, T, rh)

###########################################
# Calculate the thickness from the pressure, temperature, and mixing ratio
# for the layer from the surface pressure to 500-hPa
print(thickness_hydrostatic(p, T, mixing_ratio=mixrat, depth=p[0] - 500 * units.hPa))
