# Copyright (c) 2008-2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
Four Panel Map
===============

By reading model output data from a netCDF file, we can create a four panel plot showing:

* 300 hPa heights and winds
* 500 hPa heights and absolute vorticity
* Surface temperatures
* Precipitable water
"""

###########################################

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scipy.ndimage as ndimage

from metpy.cbook import get_test_data
from metpy.units import units

###########################################

# Make state boundaries feature
states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                name='admin_1_states_provinces_lines',
                                                scale='50m', facecolor='none')

# Make country borders feature
country_borders = cfeature.NaturalEarthFeature(category='cultural',
                                               name='admin_0_countries',
                                               scale='50m', facecolor='none')

crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

###########################################


# Function used to create the map subplots
def plot_background(ax):
    ax.set_extent([235., 290., 20., 55.])
    ax.coastlines('50m', edgecolor='black', linewidth=0.5)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.add_feature(country_borders, edgecolor='black', linewidth=0.5)
    return ax


###########################################

# Open the example netCDF data
ds = netCDF4.Dataset(get_test_data('gfs_output.nc', False))
print(ds)

###########################################

# Convert number of hours since the reference time into an actual date
time_vals = netCDF4.num2date(ds.variables['time'][:].squeeze(), ds.variables['time'].units)

###########################################

# Combine 1D latitude and longitudes into a 2D grid of locations
lon_2d, lat_2d = np.meshgrid(ds.variables['lon'][:], ds.variables['lat'][:])

###########################################

# Assign units
vort_500 = ds.variables['vort_500'][0] * units(ds.variables['vort_500'].units)
surface_temp = ds.variables['temp'][0] * units(ds.variables['temp'].units)
precip_water = ds.variables['precip_water'][0] * units(ds.variables['precip_water'].units)
winds_300 = ds.variables['winds_300'][0] * units(ds.variables['winds_300'].units)

###########################################

# Do unit conversions to what we wish to plot
vort_500 = vort_500 * 1e5
surface_temp = surface_temp.to('degF')
precip_water = precip_water.to('inches')
winds_300 = winds_300.to('knots')

###########################################

# Smooth the height data
heights_300 = ndimage.gaussian_filter(ds.variables['heights_300'][0], sigma=1.5, order=0)
heights_500 = ndimage.gaussian_filter(ds.variables['heights_500'][0], sigma=1.5, order=0)

###########################################

# Create the figure
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(5, 2, height_ratios=[1, .05, 1, .05, 0], bottom=.05, top=.95, wspace=.1)

# Upper left plot - 300-hPa winds and geopotential heights
ax1 = plt.subplot(gs[0, 0], projection=crs)
plot_background(ax1)
cf1 = ax1.contourf(lon_2d, lat_2d, winds_300, cmap='cool', transform=ccrs.PlateCarree())
c1 = ax1.contour(lon_2d, lat_2d, heights_300, colors='black', linewidth=2,
                 transform=ccrs.PlateCarree())
plt.clabel(c1, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)

ax2 = plt.subplot(gs[1, 0])
cb1 = plt.colorbar(cf1, cax=ax2, orientation='horizontal')
cb1.set_label('knots', size='x-large')
ax1.set_title('300-hPa Wind Speeds and Heights', fontsize=16)

# Upper right plot - 500mb absolute vorticity and geopotential heights
ax3 = plt.subplot(gs[0, 1], projection=crs)
plot_background(ax3)
cf2 = ax3.contourf(lon_2d, lat_2d, vort_500, cmap='BrBG', transform=ccrs.PlateCarree(),
                   zorder=0, norm=plt.Normalize(-32, 32), latlon=True)
c2 = ax3.contour(lon_2d, lat_2d, heights_500, colors='k', lw=2, transform=ccrs.PlateCarree())
plt.clabel(c2, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)

ax4 = plt.subplot(gs[1, 1])
cb2 = plt.colorbar(cf2, cax=ax4, orientation='horizontal')
cb2.set_label(r'$10^{-5}$ s$^{-1}$', size='x-large')
ax3.set_title('500-hPa Absolute Vorticity and Heights', fontsize=16)

# Lower left plot - surface temperatures
ax5 = plt.subplot(gs[2, 0], projection=crs)
plot_background(ax5)
cf3 = ax5.contourf(lon_2d, lat_2d, surface_temp, cmap='YlOrRd',
                   transform=ccrs.PlateCarree(), zorder=0)

ax6 = plt.subplot(gs[3, 0])
cb3 = plt.colorbar(cf3, cax=ax6, orientation='horizontal')
cb3.set_label(u'\N{DEGREE FAHRENHEIT}', size='x-large')
ax5.set_title('Surface Temperatures', fontsize=16)

# Lower right plot - precipitable water entire atmosphere
ax7 = plt.subplot(gs[2, 1], projection=crs)
plot_background(ax7)
cf4 = plt.contourf(lon_2d, lat_2d, precip_water, cmap='Greens',
                   transform=ccrs.PlateCarree(), zorder=0)

ax8 = plt.subplot(gs[3, 1])
cb4 = plt.colorbar(cf4, cax=ax8, orientation='horizontal')
cb4.set_label('in.', size='x-large')
ax7.set_title('Precipitable Water', fontsize=16)

fig.suptitle('{0:%d %B %Y %H:%MZ}'.format(time_vals), fontsize=24)

# Display the plot
plt.show()
