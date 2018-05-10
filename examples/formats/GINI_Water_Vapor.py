# Copyright (c) 2015,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
GINI Water Vapor Imagery
========================

Use MetPy's support for GINI files to read in a water vapor satellite image and plot the
data using CartoPy.
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr

from metpy.cbook import get_test_data
from metpy.io import GiniFile
from metpy.plots import add_metpy_logo, add_timestamp, colortables

###########################################

# Open the GINI file from the test data
f = GiniFile(get_test_data('WEST-CONUS_4km_WV_20151208_2200.gini'))
print(f)

###########################################

# Get a Dataset view of the data (essentially a NetCDF-like interface to the
# underlying data). Pull out the data, (x, y) coordinates, and the projection
# information.
ds = xr.open_dataset(f)
x = ds.variables['x'][:]
y = ds.variables['y'][:]
dat = ds.variables['WV']
proj_var = ds.variables[dat.attrs['grid_mapping']]
print(proj_var)

###########################################

# Create CartoPy projection information for the file
globe = ccrs.Globe(ellipse='sphere', semimajor_axis=proj_var.attrs['earth_radius'],
                   semiminor_axis=proj_var.attrs['earth_radius'])
proj = ccrs.LambertConformal(central_longitude=proj_var.attrs['longitude_of_central_meridian'],
                             central_latitude=proj_var.attrs['latitude_of_projection_origin'],
                             standard_parallels=[proj_var.attrs['standard_parallel']],
                             globe=globe)

###########################################

# Plot the image
fig = plt.figure(figsize=(10, 12))
add_metpy_logo(fig, 125, 145)
ax = fig.add_subplot(1, 1, 1, projection=proj)
wv_norm, wv_cmap = colortables.get_with_range('WVCIMSS', 100, 260)
wv_cmap.set_under('k')
im = ax.imshow(dat[:], cmap=wv_cmap, norm=wv_norm,
               extent=(x.min(), x.max(), y.min(), y.max()), origin='upper')
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
add_timestamp(ax, f.prod_desc.datetime, y=0.02, high_contrast=True)

plt.show()
