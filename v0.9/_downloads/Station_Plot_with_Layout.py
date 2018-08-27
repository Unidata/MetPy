# Copyright (c) 2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
Station Plot with Layout
========================

Make a station plot, complete with sky cover and weather symbols, using a
station plot layout built into MetPy.

The station plot itself is straightforward, but there is a bit of code to perform the
data-wrangling (hopefully that situation will improve in the future). Certainly, if you have
existing point data in a format you can work with trivially, the station plot will be simple.

The `StationPlotLayout` class is used to standardize the plotting various parameters
(i.e. temperature), keeping track of the location, formatting, and even the units for use in
the station plot. This makes it easy (if using standardized names) to re-use a given layout
of a station plot.
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd

from metpy.calc import wind_components
from metpy.cbook import get_test_data
from metpy.plots import (add_metpy_logo, simple_layout, StationPlot,
                         StationPlotLayout, wx_code_map)
from metpy.units import units

###########################################
# The setup
# ---------
#
# First read in the data. We use `numpy.loadtxt` to read in the data and use a structured
# `numpy.dtype` to allow different types for the various columns. This allows us to handle
# the columns with string data.
with get_test_data('station_data.txt') as f:
    data_arr = pd.read_csv(f, header=0, usecols=(1, 2, 3, 4, 5, 6, 7, 17, 18, 19),
                           names=['stid', 'lat', 'lon', 'slp', 'air_temperature',
                                  'cloud_fraction', 'dew_point_temperature', 'weather',
                                  'wind_dir', 'wind_speed'],
                           na_values=-99999)

    data_arr.set_index('stid', inplace=True)

###########################################
# This sample data has *way* too many stations to plot all of them. Instead, we just select
# a few from around the U.S. and pull those out of the data file.

# Pull out these specific stations
selected = ['OKC', 'ICT', 'GLD', 'MEM', 'BOS', 'MIA', 'MOB', 'ABQ', 'PHX', 'TTF',
            'ORD', 'BIL', 'BIS', 'CPR', 'LAX', 'ATL', 'MSP', 'SLC', 'DFW', 'NYC', 'PHL',
            'PIT', 'IND', 'OLY', 'SYR', 'LEX', 'CHS', 'TLH', 'HOU', 'GJT', 'LBB', 'LSV',
            'GRB', 'CLT', 'LNK', 'DSM', 'BOI', 'FSD', 'RAP', 'RIC', 'JAN', 'HSV', 'CRW',
            'SAT', 'BUY', '0CO', 'ZPC', 'VIH']

# Loop over all the whitelisted sites, grab the first data, and concatenate them
data_arr = data_arr.loc[selected]

# Drop rows with missing winds
data_arr = data_arr.dropna(how='any', subset=['wind_dir', 'wind_speed'])

# First, look at the names of variables that the layout is expecting:
simple_layout.names()

###########################################
# Next grab the simple variables out of the data we have (attaching correct units), and
# put them into a dictionary that we will hand the plotting function later:

# This is our container for the data
data = {}

# Copy out to stage everything together. In an ideal world, this would happen on
# the data reading side of things, but we're not there yet.
data['longitude'] = data_arr['lon'].values
data['latitude'] = data_arr['lat'].values
data['air_temperature'] = data_arr['air_temperature'].values * units.degC
data['dew_point_temperature'] = data_arr['dew_point_temperature'].values * units.degC
data['air_pressure_at_sea_level'] = data_arr['slp'].values * units('mbar')

###########################################
# Notice that the names (the keys) in the dictionary are the same as those that the
# layout is expecting.
#
# Now perform a few conversions:
#
# - Get wind components from speed and direction
# - Convert cloud fraction values to integer codes [0 - 8]
# - Map METAR weather codes to WMO codes for weather symbols

# Get the wind components, converting from m/s to knots as will be appropriate
# for the station plot
u, v = wind_components(data_arr['wind_speed'].values * units('m/s'),
                       data_arr['wind_dir'].values * units.degree)
data['eastward_wind'], data['northward_wind'] = u, v

# Convert the fraction value into a code of 0-8, which can be used to pull out
# the appropriate symbol
data['cloud_coverage'] = (8 * data_arr['cloud_fraction']).fillna(10).values.astype(int)

# Map weather strings to WMO codes, which we can use to convert to symbols
# Only use the first symbol if there are multiple
wx_text = data_arr['weather'].fillna('')
data['present_weather'] = [wx_code_map[s.split()[0] if ' ' in s else s] for s in wx_text]

###########################################
# All the data wrangling is finished, just need to set up plotting and go:
# Set up the map projection and set up a cartopy feature for state borders
proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=35,
                             standard_parallels=[35])

###########################################
# The payoff
# ----------

# Change the DPI of the resulting figure. Higher DPI drastically improves the
# look of the text rendering
plt.rcParams['savefig.dpi'] = 255

# Create the figure and an axes set to the projection
fig = plt.figure(figsize=(20, 10))
add_metpy_logo(fig, 1080, 290, size='large')
ax = fig.add_subplot(1, 1, 1, projection=proj)

# Add some various map elements to the plot to make it recognizable
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.BORDERS, linewidth=2)

# Set plot bounds
ax.set_extent((-118, -73, 23, 50))

#
# Here's the actual station plot
#

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 12 pt.
stationplot = StationPlot(ax, data['longitude'], data['latitude'],
                          transform=ccrs.PlateCarree(), fontsize=12)

# The layout knows where everything should go, and things are standardized using
# the names of variables. So the layout pulls arrays out of `data` and plots them
# using `stationplot`.
simple_layout.plot(stationplot, data)

plt.show()

###########################################
# or instead, a custom layout can be used:

# Just winds, temps, and dewpoint, with colors. Dewpoint and temp will be plotted
# out to Farenheit tenths. Extra data will be ignored
custom_layout = StationPlotLayout()
custom_layout.add_barb('eastward_wind', 'northward_wind', units='knots')
custom_layout.add_value('NW', 'air_temperature', fmt='.1f', units='degF', color='darkred')
custom_layout.add_value('SW', 'dew_point_temperature', fmt='.1f', units='degF',
                        color='darkgreen')

# Also, we'll add a field that we don't have in our dataset. This will be ignored
custom_layout.add_value('E', 'precipitation', fmt='0.2f', units='inch', color='blue')

# Create the figure and an axes set to the projection
fig = plt.figure(figsize=(20, 10))
add_metpy_logo(fig, 1080, 290, size='large')
ax = fig.add_subplot(1, 1, 1, projection=proj)

# Add some various map elements to the plot to make it recognizable
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.BORDERS, linewidth=2)

# Set plot bounds
ax.set_extent((-118, -73, 23, 50))

#
# Here's the actual station plot
#

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 12 pt.
stationplot = StationPlot(ax, data['longitude'], data['latitude'],
                          transform=ccrs.PlateCarree(), fontsize=12)

# The layout knows where everything should go, and things are standardized using
# the names of variables. So the layout pulls arrays out of `data` and plots them
# using `stationplot`.
custom_layout.plot(stationplot, data)

plt.show()
