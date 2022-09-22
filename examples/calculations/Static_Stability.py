# Copyright (c) 2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
================
Static Stability
================

Use `metpy.calc.static_stability` as well as pint's unit support to perform calculations.

The code below uses example data from our test suite to calculate the static stability for each
layer of the provided sounding data.
"""
import pandas as pd

from metpy.calc import static_stability
from metpy.cbook import get_test_data
from metpy.constants import Rd
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
# Isolate pressure and temperature and add units
p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC

###########################################
# Calculate the static stability and convert to K/Pa units
print((static_stability(p, T) * (p / Rd)).to('K/Pa'))
