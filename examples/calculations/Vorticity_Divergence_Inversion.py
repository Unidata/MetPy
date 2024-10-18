# Copyright (c) 2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
=========
Vorticity
=========

Use `metpy.calc.vorticity`.

This example demonstrates the calculation of reconstructed wind field for
cyclone dora and plotting using matplotlib.
"""
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.ticker as mticker
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

u850 = xr.open_dataset('gfs.t12z.pgrb2.0p25.f000', engine='cfgrib',
                       backend_kwargs={'filter_by_keys':
                                       {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u',
                                        'level': 850}})
u = u850.u

v850 = xr.open_dataset('gfs.t12z.pgrb2.0p25.f000', engine='cfgrib',
                       backend_kwargs={'filter_by_keys':
                                       {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v',
                                        'level': 850}})
v = v850.v


# Compute the 850 hPa relative vorticity.


vort850 = mpcalc.vorticity(u, v)
fig = plt.figure(figsize=(12, 9), dpi=300.)
# Create a set of axes for the figure and set
# its map projection to that of the input data.
ax = plt.axes(projection=crs.PlateCarree())

# Add country borders and coastlines.
countries = NaturalEarthFeature(category='cultural', scale='50m',
                                facecolor='none',
                                name='admin_0_countries')
ax.add_feature(countries, linewidth=.5, edgecolor='black')
ax.coastlines('50m', linewidth=0.8)

plot = vort850.plot(levels=np.arange(-1.e-4, 1.e-4, 0.2e-5),
                    cmap=get_cmap('PRGn'), transform=crs.PlateCarree(), cbar_kwargs={'label':
                    'relative vorticity (x$10^{-5} s^{-1}$)', 'shrink': 0.98})

# Set the map's extent to cover just Hurricane Dora.
ax.set_extent([-180., -150., 0., 20.], crs=crs.PlateCarree())

# Add latitude/longitude gridlines.
gridlines = ax.gridlines(color='grey', linestyle='dotted', draw_labels=True)
gridlines.xlabels_top = False
gridlines.ylabels_right = False
gridlines.xlocator = mticker.FixedLocator(np.arange(-180., 149., 5.))
gridlines.ylocator = mticker.FixedLocator(np.arange(0., 21., 5.))
gridlines.xlabel_style = {'size': 12, 'color': 'black'}
gridlines.ylabel_style = {'size': 12, 'color': 'black'}
gridlines.xformatter = LONGITUDE_FORMATTER
gridlines.yformatter = LATITUDE_FORMATTER

# Add a plot title, then show the image.
plt.title('GFS 0-h 850 hPa relative vorticity (x$10^{-5} s^{-1}$) at 1200 UTC 9 August 2023')
plt.savefig('vort.png')
plt.show()

# Compute the 850 hPa divergence.

div850 = mpcalc.divergence(u, v)

# Create a figure instance.
fig = plt.figure(figsize=(12, 9), dpi=300.)

# Create a set of axes for the figure and set
# its map projection to that of the input data.
ax = plt.axes(projection=crs.PlateCarree())

# Add country borders and coastlines.
countries = NaturalEarthFeature(category='cultural', scale='50m',
                                facecolor='none',
                                name='admin_0_countries')
ax.add_feature(countries, linewidth=.5, edgecolor='black')
ax.coastlines('50m', linewidth=0.8)

# Plot the 850 hPa divergence using xarray's plot functionality.
plot = div850.plot(levels=np.arange(-1.e-4, 1.e-4, 0.2e-5),
                   cmap=get_cmap('PRGn'), transform=crs.PlateCarree(),
                   cbar_kwargs={'label': 'relative vorticity (x$10^{-5} s^{-1}$)',
                                'shrink': 0.98})

# Set the map's extent to cover just Hurricane Dora.
ax.set_extent([-180., -150., 0., 20.], crs=crs.PlateCarree())

# Add latitude/longitude gridlines.
gridlines = ax.gridlines(color='grey', linestyle='dotted', draw_labels=True)
gridlines.xlabels_top = False
gridlines.ylabels_right = False
gridlines.xlocator = mticker.FixedLocator(np.arange(-180., 149., 5.))
gridlines.ylocator = mticker.FixedLocator(np.arange(0., 21., 5.))
gridlines.xlabel_style = {'size': 12, 'color': 'black'}
gridlines.ylabel_style = {'size': 12, 'color': 'black'}
gridlines.xformatter = LONGITUDE_FORMATTER
gridlines.yformatter = LATITUDE_FORMATTER

# Add a plot title, then show the image.
plt.title('GFS 0-h 850 hPa divergence (x$10^{-5} s^{-1}$) at 1200 UTC 9 August 2023')
plt.savefig('div.png')
plt.show()

umask = mpcalc.bounding_box_mask(u, 5., 13.5, 191., 202.)

vmask = mpcalc.bounding_box_mask(v, 5., 13.5, 191., 202.)


vortmask = mpcalc.bounding_box_mask(vort850, 5., 13.5, 191., 202.)


divmask = mpcalc.bounding_box_mask(div850, 5., 13.5, 191., 202.)

i_bb_indices = mpcalc.find_bounding_box_indices(vortmask, 5., 13.5, 191., 202.)


o_bb_indices = mpcalc.find_bounding_box_indices(vortmask, 0., 30., 180., 220.)


dx, dy = mpcalc.lat_lon_grid_deltas(vortmask.longitude, vortmask.latitude)

upsi, vpsi = mpcalc.rotational_wind_from_inversion(umask, vmask, vortmask, dx, dy,
                                                   o_bb_indices, i_bb_indices)

# Create a figure instance.
fig = plt.figure(figsize=(12, 9), dpi=300.)

# Create a set of axes for the figure and set
# its map projection to that of the input data.
ax = plt.axes(projection=crs.PlateCarree())

# Add country borders and coastlines.
countries = NaturalEarthFeature(category='cultural', scale='50m',
                                facecolor='none',
                                name='admin_0_countries')
ax.add_feature(countries, linewidth=.5, edgecolor='black')
ax.coastlines('50m', linewidth=0.8)

# Compute the magnitude of the non-divergent component of the 850 hPa wind.
nd_spd = np.sqrt(upsi**2 + vpsi**2)

# Plot this using xarray's plot functionality.
plot = nd_spd.plot(levels=np.arange(0., 13., 1.),
                   cmap=get_cmap('YlGnBu'), transform=crs.PlateCarree(),
                   cbar_kwargs={'label': 'non-divergent wind ($m s^{-1}$)', 'shrink': 0.98})

# Set the map's extent to match that over which we computed the non-divergent wind.
ax.set_extent([-180., -140., 0., 30.], crs=crs.PlateCarree())

# Add latitude/longitude gridlines.
gridlines = ax.gridlines(color='grey', linestyle='dotted', draw_labels=True)
gridlines.xlabels_top = False
gridlines.ylabels_right = False
gridlines.xlocator = mticker.FixedLocator(np.arange(-180., 139., 5.))
gridlines.ylocator = mticker.FixedLocator(np.arange(0., 31., 5.))
gridlines.xlabel_style = {'size': 12, 'color': 'black'}
gridlines.ylabel_style = {'size': 12, 'color': 'black'}
gridlines.xformatter = LONGITUDE_FORMATTER
gridlines.yformatter = LATITUDE_FORMATTER

# Add a plot title, then show the image.
plt.title('850 hPa non-divergent wind magnitude  due to Dora at 1200 UTC 9 August 2023')
plt.savefig('reconstructed_rotational_wind.png')
plt.show()

uchi, vchi = mpcalc.divergent_wind_from_inversion(umask, vmask, divmask, dx, dy,
                                                  o_bb_indices, i_bb_indices)

# Create a set of axes for the figure and set
# its map projection to that of the input data.

ax = plt.axes(projection=crs.PlateCarree())

# Add country borders and coastlines.
countries = NaturalEarthFeature(category='cultural', scale='50m',
                                facecolor='none',
                                name='admin_0_countries')
ax.add_feature(countries, linewidth=.5, edgecolor='black')
ax.coastlines('50m', linewidth=0.8)

# Compute the magnitude of the non-divergent component of the 850 hPa wind.
nd_spd = np.sqrt(uchi**2 + vchi**2)

# Plot this using xarray's plot functionality.
plot = nd_spd.plot(levels=np.arange(0., 13., 1.),
                   cmap=get_cmap('YlGnBu'), transform=crs.PlateCarree(),
                   cbar_kwargs={'label': 'non-divergent wind ($m s^{-1}$)', 'shrink': 0.98})

# Set the map's extent to match that over which we computed the non-divergent wind.
ax.set_extent([-180., -140., 0., 30.], crs=crs.PlateCarree())

# Add latitude/longitude gridlines.
gridlines = ax.gridlines(color='grey', linestyle='dotted', draw_labels=True)
gridlines.top_labels = False
gridlines.right_labels = False
gridlines.xlocator = mticker.FixedLocator(np.arange(-180., 139., 5.))
gridlines.ylocator = mticker.FixedLocator(np.arange(0., 31., 5.))
gridlines.xlabel_style = {'size': 12, 'color': 'black'}
gridlines.ylabel_style = {'size': 12, 'color': 'black'}
gridlines.xformatter = LONGITUDE_FORMATTER
gridlines.yformatter = LATITUDE_FORMATTER

# Add a plot title, then show the image.
plt.title('850 hPa divergent wind magnitude due to Dora at 1200 UTC 9 August 2023')
plt.savefig('irrotational_winds.png')
plt.show()
