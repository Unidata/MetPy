# Copyright (c) 2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
==========
Bulk Shear
==========

Use functions from `metpy.calc` as well as pint's unit support to perform calculations.

The code below uses example data from our test suite to calculate the bulk shear over the
lowest three kilometers for the provided sounding data.
"""
import pandas as pd

from metpy.calc import bulk_shear, wind_components
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
df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'),
               how='all').reset_index(drop=True)

###########################################
# Isolate pressure, wind direction, wid speed, and height and add units
p = df['pressure'].values * units.hPa
wdir = df['direction'].values * units.degree
sped = df['speed'].values * units.knot
height = df['height'].values * units.meter

###########################################
# Calculate the u and v-components of the wind
u, v = wind_components(sped, wdir)

###########################################
# Compute the bulk shear for the lowest three km
print(bulk_shear(p, u, v, height, depth=3 * units.km, bottom=height[0]))
