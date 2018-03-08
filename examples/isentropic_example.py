# Copyright (c) 2017,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
===================
Isentropic Analysis
===================

The MetPy function `mpcalc.isentropic_interpolation` allows for isentropic analysis from model
analysis data in isobaric coordinates.
"""

########################################
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, add_timestamp
from metpy.units import units

#######################################
# **Getting the data**
#
# In this example, NARR reanalysis data for 18 UTC 04 April 1987 from the National Centers
# for Environmental Information (https://www.ncdc.noaa.gov/data-access/model-data)
# will be used.

data = Dataset(get_test_data('narr_example.nc', False))

##########################
print(list(data.variables))

#############################
# We will reduce the dimensionality of the data as it is pulled in to remove an empty time
# dimension. Additionally, units are required for input data, so the proper units will also
# be attached.


# Assign data to variable names
dtime = data.variables['Geopotential_height'].dimensions[0]
dlev = data.variables['Geopotential_height'].dimensions[1]
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
lev = data.variables[dlev][:] * units(data.variables[dlev].units)
times = data.variables[dtime]
vtimes = num2date(times[:], times.units)

temps = data.variables['Temperature']
tmp = temps[0, :] * units.kelvin
uwnd = data.variables['u_wind'][0, :] * units(data.variables['u_wind'].units)
vwnd = data.variables['v_wind'][0, :] * units(data.variables['v_wind'].units)
hgt = data.variables['Geopotential_height'][0, :] * units.meter
spech = (data.variables['Specific_humidity'][0, :] *
         units(data.variables['Specific_humidity'].units))

#############################
# To properly interpolate to isentropic coordinates, the function must know the desired output
# isentropic levels. An array with these levels will be created below.

isentlevs = [296.] * units.kelvin

####################################
# **Conversion to Isentropic Coordinates**
#
# Once three dimensional data in isobaric coordinates has been pulled and the desired
# isentropic levels created, the conversion to isentropic coordinates can begin. Data will be
# passed to the function as below. The function requires that isentropic levels, isobaric
# levels, and temperature be input. Any additional inputs (in this case relative humidity, u,
# and v wind components) will be linearly interpolated to isentropic space.

isent_anal = mpcalc.isentropic_interpolation(isentlevs,
                                             lev,
                                             tmp,
                                             spech,
                                             uwnd,
                                             vwnd,
                                             hgt,
                                             tmpk_out=True)

#####################################
# The output is a list, so now we will separate the variables to different names before
# plotting.


isentprs, isenttmp, isentspech, isentu, isentv, isenthgt = isent_anal
isentu.ito('kt')
isentv.ito('kt')

########################################
# A quick look at the shape of these variables will show that the data is now in isentropic
# coordinates, with the number of vertical levels as specified above.

print(isentprs.shape)
print(isentspech.shape)
print(isentu.shape)
print(isentv.shape)
print(isenttmp.shape)
print(isenthgt.shape)

#################################
# **Converting to Relative Humidity**
#
# The NARR only gives specific humidity on isobaric vertical levels, so relative humidity will
# have to be calculated after the interpolation to isentropic space.

isentrh = 100 * mpcalc.relative_humidity_from_specific_humidity(isentspech, isenttmp, isentprs)

#######################################
# **Plotting the Isentropic Analysis**


# Set up our projection
crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)


# Set up our array of latitude and longitude values and transform to
# the desired projection.
tlatlons = crs.transform_points(ccrs.PlateCarree(), lon, lat)
tlons = tlatlons[:, :, 0]
tlats = tlatlons[:, :, 1]

# Coordinates to limit map area
bounds = [(-122., -75., 25., 50.)]
# Choose a level to plot, in this case 296 K
level = 0

fig = plt.figure(figsize=(17., 12.))
add_metpy_logo(fig, 120, 245, size='large')
ax = fig.add_subplot(1, 1, 1, projection=crs)
ax.set_extent(*bounds, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
ax.add_feature(cfeature.STATES, linewidth=0.5)

# Plot the surface
clevisent = np.arange(0, 1000, 25)
cs = ax.contour(tlons, tlats, isentprs[level, :, :], clevisent,
                colors='k', linewidths=1.0, linestyles='solid')
plt.clabel(cs, fontsize=10, inline=1, inline_spacing=7,
           fmt='%i', rightside_up=True, use_clabeltext=True)

# Plot RH
cf = ax.contourf(tlons, tlats, isentrh[level, :, :], range(10, 106, 5),
                 cmap=plt.cm.gist_earth_r)
cb = plt.colorbar(cf, orientation='horizontal', extend='max', aspect=65, shrink=0.5, pad=0.05,
                  extendrect='True')
cb.set_label('Relative Humidity', size='x-large')

# Transform Vectors before plotting, then plot wind barbs.
ut, vt = crs.transform_vectors(ccrs.PlateCarree(), lon, lat, isentu[level, :, :].m,
                               isentv[level, :, :].m)
ax.barbs(tlons, tlats, ut, vt, length=6, regrid_shape=20)

# Make some titles
plt.title('{:.0f} K Isentropic Pressure (hPa), Wind (kt), Relative Humidity (percent)'
          .format(isentlevs[level].m),
          loc='left')
plt.title('VALID: {:s}'.format(str(vtimes[0])), loc='right')
plt.tight_layout()
add_timestamp(ax, vtimes[0], y=0.02, high_contrast=True)

######################################
# **Montgomery Streamfunction**
#
# The Montgomery Streamfunction, :math:`{\psi} = gdz + CpT`, is often desired because its
# gradient is proportional to the geostrophic wind in isentropic space. This can be easily
# calculated with `mpcalc.montgomery_streamfunction`.


# Calculate Montgomery Streamfunction and scale by 10^-2 for plotting
msf = mpcalc.montgomery_streamfunction(isenthgt, isenttmp) / 100.

# Choose a level to plot, in this case 296 K
level = 0

fig = plt.figure(figsize=(17., 12.))
add_metpy_logo(fig, 120, 250, size='large')
ax = plt.subplot(111, projection=crs)
ax.set_extent(*bounds, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)

# Plot the surface
clevmsf = np.arange(0, 4000, 5)
cs = ax.contour(tlons, tlats, msf[level, :, :], clevmsf,
                colors='k', linewidths=1.0, linestyles='solid')
plt.clabel(cs, fontsize=10, inline=1, inline_spacing=7,
           fmt='%i', rightside_up=True, use_clabeltext=True)
# Plot RH
cf = ax.contourf(tlons, tlats, isentrh[level, :, :], range(10, 106, 5),
                 cmap=plt.cm.gist_earth_r)
cb = plt.colorbar(cf, orientation='horizontal', extend='max', aspect=65, shrink=0.5, pad=0.05,
                  extendrect='True')
cb.set_label('Relative Humidity', size='x-large')

# Transform Vectors before plotting, then plot wind barbs.
ut, vt = crs.transform_vectors(ccrs.PlateCarree(), lon, lat, isentu[level, :, :].m,
                               isentv[level, :, :].m)
ax.barbs(tlons, tlats, ut, vt, length=6, regrid_shape=20)

# Make some titles
plt.title('{:.0f} K Montgomery Streamfunction '.format(isentlevs[level].m) +
          r'($10^{-2} m^2 s^{-2}$), ' +
          'Wind (kt), Relative Humidity (percent)', loc='left')
plt.title('VALID: {:s}'.format(str(vtimes[0])), loc='right')
plt.tight_layout()
add_timestamp(ax, vtimes[0], y=0.02, high_contrast=True)

plt.show()
