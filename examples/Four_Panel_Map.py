# Copyright (c) 2017 MetPy Developers.
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
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scipy.ndimage as ndimage

from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo
from metpy.units import units

###########################################

crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

###########################################


# Function used to create the map subplots
def plot_background(ax):
    ax.set_extent([235., 290., 20., 55.])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
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

# Create the figure and plot background on different axes
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(20, 13), constrained_layout=True,
                          subplot_kw={'projection': crs})
add_metpy_logo(fig, 140, 120, size='large')
axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

# Upper left plot - 300-hPa winds and geopotential heights
cf1 = axlist[0].contourf(lon_2d, lat_2d, winds_300, cmap='cool', transform=ccrs.PlateCarree())
c1 = axlist[0].contour(lon_2d, lat_2d, heights_300, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
axlist[0].clabel(c1, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
axlist[0].set_title('300-hPa Wind Speeds and Heights', fontsize=16)
cb1 = fig.colorbar(cf1, ax=axlist[0], orientation='horizontal', shrink=0.74, pad=0)
cb1.set_label('knots', size='x-large')

# Upper right plot - 500mb absolute vorticity and geopotential heights
cf2 = axlist[1].contourf(lon_2d, lat_2d, vort_500, cmap='BrBG', transform=ccrs.PlateCarree(),
                         zorder=0, norm=plt.Normalize(-32, 32))
c2 = axlist[1].contour(lon_2d, lat_2d, heights_500, colors='k', linewidths=2,
                       transform=ccrs.PlateCarree())
axlist[1].clabel(c2, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
axlist[1].set_title('500-hPa Absolute Vorticity and Heights', fontsize=16)
cb2 = fig.colorbar(cf2, ax=axlist[1], orientation='horizontal', shrink=0.74, pad=0)
cb2.set_label(r'$10^{-5}$ s$^{-1}$', size='x-large')

# Lower left plot - surface temperatures
cf3 = axlist[2].contourf(lon_2d, lat_2d, surface_temp, cmap='YlOrRd',
                         transform=ccrs.PlateCarree(), zorder=0)
axlist[2].set_title('Surface Temperatures', fontsize=16)
cb3 = fig.colorbar(cf3, ax=axlist[2], orientation='horizontal', shrink=0.74, pad=0)
cb3.set_label(u'\N{DEGREE FAHRENHEIT}', size='x-large')

# Lower right plot - precipitable water entire atmosphere
cf4 = axlist[3].contourf(lon_2d, lat_2d, precip_water, cmap='Greens',
                         transform=ccrs.PlateCarree(), zorder=0)
axlist[3].set_title('Precipitable Water', fontsize=16)
cb4 = fig.colorbar(cf4, ax=axlist[3], orientation='horizontal', shrink=0.74, pad=0)
cb4.set_label('in.', size='x-large')

# Set height padding for plots
fig.set_constrained_layout_pads(w_pad=0., h_pad=0.1, hspace=0., wspace=0.)

# Set figure title
fig.suptitle('{0:%d %B %Y %H:%MZ}'.format(time_vals), fontsize=24)

# Display the plot
plt.show()
