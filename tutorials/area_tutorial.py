# Copyright (c) 2023 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
==================================
Using Predefined Areas with MetPy
==================================

When plotting your data on a map you want to be able to plot the data in a useful area
and with a projection that would match the areal extent well. Within MetPy we have generated
more than 400 pre-defined areas that also have associated projections, which you can use to
make quick plots over regions of interest with minimal effort on your part. While these were
intended to be used with the MetPy declarative syntax, these areas and projections are now
accessible for any use of plotting data on a map using Cartopy.
"""

from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from metpy.cbook import get_test_data
from metpy.io import metar
from metpy.plots import declarative, named_areas

###########################
# Table of Predefined Areas
# -------------------------
#
# Here is a full list of all of the areas currently in MetPy with their reference name,
# descriptive name, and extent bounds.

print('area      name                bounds')
for area in named_areas:
    print(f'{named_areas[area].name:<10s}{named_areas[area].description:<20s}{named_areas[area].bounds}')

####################################################
# Example Using Bounds and Projection for an Area
# -------------------------------------------------
#
# Each area string given in the table above have a descriptive name, extent bounds, and a
# projection associated with each entry. We can pull any of this information to help us make a
# plot over the area domain.

# Select the area string
area = 'epac'

# Get the extent and project for the selected area
extent = named_areas[area].bounds
proj = named_areas[area].projection

# Plot a simple figure for the selected area
plt.figure(1, figsize=(10, 10))
ax = plt.subplot(111, projection=proj)
ax.set_extent(extent, ccrs.PlateCarree())
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='grey', linewidth=0.75)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.1)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black')
ax.set_title(f'area={area}          name={named_areas[area].description}'
             f'\nproj={proj.coordinate_operation.method_name}')
plt.show()


########################################
# Examaple Using Declarative Syntax
# ---------------------------------
#
# Here is an example using a predefined area with the declarative plotting syntax for plotting
# some surface observations.

# Set the observation time
obs_time = datetime(2019, 7, 1, 12)

# Read in data
df = metar.parse_metar_file(get_test_data('metar_20190701_1200.txt', False), year=2019,
                            month=7)

# Plot desired data
obs = declarative.PlotObs()
obs.data = df
obs.time = obs_time
obs.time_window = timedelta(minutes=15)
obs.level = None
obs.fields = ['cloud_coverage', 'air_temperature', 'dew_point_temperature',
              'air_pressure_at_sea_level', 'current_wx1_symbol']
obs.plot_units = [None, 'degF', 'degF', None, None]
obs.locations = ['C', 'NW', 'SW', 'NE', 'W']
obs.formats = ['sky_cover', None, None, lambda v: format(v * 10, '.0f')[-3:],
               'current_weather']
obs.reduce_points = 0.75
obs.vector_field = ['eastward_wind', 'northward_wind']

# Panel for plot with Map features
panel = declarative.MapPanel()
panel.layout = (1, 1, 1)
panel.projection = 'area'
panel.area = 'in+'
panel.layers = ['states']
panel.title = f'Surface plot for {obs_time}'
panel.plots = [obs]

# Bringing it all together
pc = declarative.PanelContainer()
pc.size = (10, 10)
pc.panels = [panel]

pc.show()
