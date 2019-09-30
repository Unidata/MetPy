# Copyright (c) 2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
Station Plot
============

Make a station plot, complete with sky cover and weather symbols.

The station plot itself is pretty straightforward, but there is a bit of code to perform the
data-wrangling (hopefully that situation will improve in the future). Certainly, if you have
existing point data in a format you can work with trivially, the station plot will be simple.
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metpy.calc import reduce_point_density
from metpy.calc import wind_components
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, current_weather, sky_cover, StationPlot, wx_code_map
from metpy.units import units

###########################################
# The setup
# ---------
#
# First read in the data. We use pandas because it simplifies a lot of tasks, like dealing
# with strings
with get_test_data('station_data.txt') as f:
    data = pd.read_csv(f, header=0, usecols=(1, 2, 3, 4, 5, 6, 7, 17, 18, 19),
                       names=['stid', 'lat', 'lon', 'slp', 'air_temperature', 'cloud_fraction',
                              'dew_point_temperature', 'weather', 'wind_dir', 'wind_speed'],
                       na_values=-99999)

    # Drop rows with missing winds
    data = data.dropna(how='any', subset=['wind_dir', 'wind_speed'])

###########################################
# This sample data has *way* too many stations to plot all of them. The number
# of stations plotted will be reduced using `reduce_point_density`.

# Set up the map projection
proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=35,
                             standard_parallels=[35])

# Use the cartopy map projection to transform station locations to the map and
# then refine the number of stations plotted by setting a 300km radius
point_locs = proj.transform_points(ccrs.PlateCarree(), data['lon'].values, data['lat'].values)
data = data[reduce_point_density(point_locs, 300000.)]

###########################################
# Now that we have the data we want, we need to perform some conversions:
#
# - Get wind components from speed and direction
# - Convert cloud fraction values to integer codes [0 - 8]
# - Map METAR weather codes to WMO codes for weather symbols

# Get the wind components, converting from m/s to knots as will be appropriate
# for the station plot.
u, v = wind_components((data['wind_speed'].values * units('m/s')).to('knots'),
                       data['wind_dir'].values * units.degree)

# Convert the fraction value into a code of 0-8 and compensate for NaN values,
# which can be used to pull out the appropriate symbol
cloud_frac = (8 * data['cloud_fraction'])
cloud_frac[np.isnan(cloud_frac)] = 10
cloud_frac = cloud_frac.astype(int)

# Map weather strings to WMO codes, which we can use to convert to symbols
# Only use the first symbol if there are multiple
wx = [wx_code_map[s.split()[0] if ' ' in s else s] for s in data['weather'].fillna('')]


###########################################
# The payoff
# ----------

# Change the DPI of the resulting figure. Higher DPI drastically improves the
# look of the text rendering.
plt.rcParams['savefig.dpi'] = 255

# Create the figure and an axes set to the projection.
fig = plt.figure(figsize=(20, 10))
add_metpy_logo(fig, 1080, 290, size='large')
ax = fig.add_subplot(1, 1, 1, projection=proj)

# Add some various map elements to the plot to make it recognizable.
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.BORDERS)

# Set plot bounds
ax.set_extent((-118, -73, 23, 50))

#
# Here's the actual station plot
#

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 12 pt.
stationplot = StationPlot(ax, data['lon'].values, data['lat'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=12)

# Plot the temperature and dew point to the upper and lower left, respectively, of
# the center point. Each one uses a different color.
stationplot.plot_parameter('NW', data['air_temperature'], color='red')
stationplot.plot_parameter('SW', data['dew_point_temperature'],
                           color='darkgreen')

# A more complex example uses a custom formatter to control how the sea-level pressure
# values are plotted. This uses the standard trailing 3-digits of the pressure value
# in tenths of millibars.
stationplot.plot_parameter('NE', data['slp'], formatter=lambda v: format(10 * v, '.0f')[-3:])

# Plot the cloud cover symbols in the center location. This uses the codes made above and
# uses the `sky_cover` mapper to convert these values to font codes for the
# weather symbol font.
stationplot.plot_symbol('C', cloud_frac, sky_cover)

# Same this time, but plot current weather to the left of center, using the
# `current_weather` mapper to convert symbols to the right glyphs.
stationplot.plot_symbol('W', wx, current_weather)

# Add wind barbs
stationplot.plot_barb(u, v)

# Also plot the actual text of the station id. Instead of cardinal directions,
# plot further out by specifying a location of 2 increments in x and 0 in y.
stationplot.plot_text((2, 0), data['stid'])

plt.show()
