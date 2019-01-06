# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
xarray with MetPy Tutorial
==========================

`xarray <http://xarray.pydata.org/>`_ is a powerful Python package that provides N-dimensional
labeled arrays and datasets following the Common Data Model. While the process of integrating
xarray features into MetPy is ongoing, this tutorial demonstrates how xarray can be used
within the current version of MetPy. MetPy's integration primarily works through accessors
which allow simplified projection handling and coordinate identification. Unit and calculation
support is currently available in a limited fashion, but should be improved in future
versions.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr

# Any import of metpy will activate the accessors
import metpy.calc as mpcalc
from metpy.testing import get_test_data
from metpy.units import units

#########################################################################
# Getting Data
# ------------
#
# While xarray can handle a wide variety of n-dimensional data (essentially anything that can
# be stored in a netCDF file), a common use case is working with model output. Such model
# data can be obtained from a THREDDS Data Server using the siphon package, but for this
# tutorial, we will use an example subset of GFS data from Hurrican Irma (September 5th,
# 2017).

# Open the netCDF file as a xarray Dataset
data = xr.open_dataset(get_test_data('irma_gfs_example.nc', False))

# View a summary of the Dataset
print(data)

#########################################################################
# Preparing Data
# --------------
#
# To make use of the data within MetPy, we need to parse the dataset for projection and
# coordinate information following the CF conventions. For this, we use the
# ``data.metpy.parse_cf()`` method, which will return a new, parsed ``DataArray`` or
# ``Dataset``.
#
# Additionally, we rename our data variables for easier reference.

# To parse the full dataset, we can call parse_cf without an argument, and assign the returned
# Dataset.
data = data.metpy.parse_cf()

# If we instead want just a single variable, we can pass that variable name to parse_cf and
# it will return just that data variable as a DataArray.
data_var = data.metpy.parse_cf('Temperature_isobaric')

# To rename variables, supply a dictionary between old and new names to the rename method
data.rename({
    'Vertical_velocity_pressure_isobaric': 'omega',
    'Relative_humidity_isobaric': 'relative_humidity',
    'Temperature_isobaric': 'temperature',
    'u-component_of_wind_isobaric': 'u',
    'v-component_of_wind_isobaric': 'v',
    'Geopotential_height_isobaric': 'height'
}, inplace=True)

#########################################################################
# Units
# -----
#
# MetPy's DataArray accessor has a ``unit_array`` property to obtain a ``pint.Quantity`` array
# of just the data from the DataArray (metadata is removed) and a ``convert_units`` method to
# convert the the data from one unit to another (keeping it as a DataArray). For now, we'll
# just use ``convert_units`` to convert our temperature to ``degC``.

data['temperature'].metpy.convert_units('degC')

#########################################################################
# Coordinates
# -----------
#
# You may have noticed how we directly accessed the vertical coordinates above using their
# names. However, in general, if we are working with a particular DataArray, we don't have to
# worry about that since MetPy is able to parse the coordinates and so obtain a particular
# coordinate type directly. There are two ways to do this:
#
# 1. Use the ``data_var.metpy.coordinates`` method
# 2. Use the ``data_var.metpy.x``, ``data_var.metpy.y``, ``data_var.metpy.vertical``,
#    ``data_var.metpy.time`` properties
#
# The valid coordinate types are:
#
# - x
# - y
# - vertical
# - time
#
# (Both approaches and all four types are shown below)

# Get multiple coordinates (for example, in just the x and y direction)
x, y = data['temperature'].metpy.coordinates('x', 'y')

# If we want to get just a single coordinate from the coordinates method, we have to use
# tuple unpacking because the coordinates method returns a generator
vertical, = data['temperature'].metpy.coordinates('vertical')

# Or, we can just get a coordinate from the property
time = data['temperature'].metpy.time

# To verify, we can inspect all their names
print([coord.name for coord in (x, y, vertical, time)])

#########################################################################
# Indexing and Selecting Data
# ---------------------------
#
# MetPy provides wrappers for the usual xarray indexing and selection routines that can handle
# quantities with units. For DataArrays, MetPy also allows using the coordinate axis types
# mentioned above as aliases for the coordinates. And so, if we wanted 850 hPa heights,
# we would take:

print(data['height'].metpy.sel(vertical=850 * units.hPa))

#########################################################################
# For full details on xarray indexing/selection, see
# `xarray's documentation <http://xarray.pydata.org/en/stable/indexing.html>`_.

#########################################################################
# Projections
# -----------
#
# Getting the cartopy coordinate reference system (CRS) of the projection of a DataArray is as
# straightforward as using the ``data_var.metpy.cartopy_crs`` property:

data_crs = data['temperature'].metpy.cartopy_crs
print(data_crs)

#########################################################################
# The cartopy ``Globe`` can similarly be accessed via the ``data_var.metpy.cartopy_globe``
# property:

data_globe = data['temperature'].metpy.cartopy_globe
print(data_globe)

#########################################################################
# Calculations
# ------------
#
# Most of the calculations in `metpy.calc` will accept DataArrays by converting them
# into their corresponding unit arrays. While this may often work without any issues, we must
# keep in mind that because the calculations are working with unit arrays and not DataArrays:
#
# - The calculations will return unit arrays rather than DataArrays
# - Broadcasting must be taken care of outside of the calculation, as it would only recognize
#   dimensions by order, not name
#
# As an example, we calculate geostropic wind at 500 hPa below:

lat, lon = xr.broadcast(y, x)
f = mpcalc.coriolis_parameter(lat)
dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat, initstring=data_crs.proj4_init)
heights = data['height'].metpy.loc[{'time': time[0], 'vertical': 500. * units.hPa}]
u_geo, v_geo = mpcalc.geostrophic_wind(heights, f, dx, dy)
print(u_geo)
print(v_geo)

#########################################################################
# Also, a limited number of calculations directly support xarray DataArrays or Datasets (they
# can accept *and* return xarray objects). Right now, this includes
#
# - Derivative functions
#     - ``first_derivative``
#     - ``second_derivative``
#     - ``gradient``
#     - ``laplacian``
# - Cross-section functions
#     - ``cross_section_components``
#     - ``normal_component``
#     - ``tangential_component``
#     - ``absolute_momentum``
#
# More details can be found by looking at the documentation for the specific function of
# interest.

#########################################################################
# There is also the special case of the helper function, ``grid_deltas_from_dataarray``, which
# takes a ``DataArray`` input, but returns unit arrays for use in other calculations. We could
# rewrite the above geostrophic wind example using this helper function as follows:

heights = data['height'].metpy.loc[{'time': time[0], 'vertical': 500. * units.hPa}]
lat, lon = xr.broadcast(y, x)
f = mpcalc.coriolis_parameter(lat)
dx, dy = mpcalc.grid_deltas_from_dataarray(heights)
u_geo, v_geo = mpcalc.geostrophic_wind(heights, f, dx, dy)
print(u_geo)
print(v_geo)

#########################################################################
# Plotting
# --------
#
# Like most meteorological data, we want to be able to plot these data. DataArrays can be used
# like normal numpy arrays in plotting code, which is the recommended process at the current
# point in time, or we can use some of xarray's plotting functionality for quick inspection of
# the data.
#
# (More detail beyond the following can be found at `xarray's plotting reference
# <http://xarray.pydata.org/en/stable/plotting.html>`_.)

# A very simple example example of a plot of 500 hPa heights
data['height'].metpy.loc[{'time': time[0], 'vertical': 500. * units.hPa}].plot()
plt.show()

#########################################################################

# Let's add a projection and coastlines to it
ax = plt.axes(projection=ccrs.LambertConformal())
data['height'].metpy.loc[{'time': time[0],
                          'vertical': 500. * units.hPa}].plot(ax=ax, transform=data_crs)
ax.coastlines()
plt.show()

#########################################################################

# Or, let's make a full 500 hPa map with heights, temperature, winds, and humidity

# Select the data for this time and level
data_level = data.metpy.loc[{time.name: time[0], vertical.name: 500. * units.hPa}]

# Create the matplotlib figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': data_crs})

# Plot RH as filled contours
rh = ax.contourf(x, y, data_level['relative_humidity'], levels=[70, 80, 90, 100],
                 colors=['#99ff00', '#00ff00', '#00cc00'])

# Plot wind barbs, but not all of them
wind_slice = slice(5, -5, 5)
ax.barbs(x[wind_slice], y[wind_slice],
         data_level['u'].metpy.unit_array[wind_slice, wind_slice].to('knots'),
         data_level['v'].metpy.unit_array[wind_slice, wind_slice].to('knots'),
         length=6)

# Plot heights and temperature as contours
h_contour = ax.contour(x, y, data_level['height'], colors='k', levels=range(5400, 6000, 60))
h_contour.clabel(fontsize=8, colors='k', inline=1, inline_spacing=8,
                 fmt='%i', rightside_up=True, use_clabeltext=True)
t_contour = ax.contour(x, y, data_level['temperature'], colors='xkcd:deep blue',
                       levels=range(-26, 4, 2), alpha=0.8, linestyles='--')
t_contour.clabel(fontsize=8, colors='xkcd:deep blue', inline=1, inline_spacing=8,
                 fmt='%i', rightside_up=True, use_clabeltext=True)

# Add geographic features
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=cfeature.COLORS['land'])
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=cfeature.COLORS['water'])
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='#c7c783', zorder=0)
ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor=cfeature.COLORS['water'],
               edgecolor='#c7c783', zorder=0)

# Set a title and show the plot
ax.set_title('500 hPa Heights (m), Temperature (\u00B0C), Humidity (%) at '
             + time[0].dt.strftime('%Y-%m-%d %H:%MZ'))
plt.show()

#########################################################################
# What Could Go Wrong?
# --------------------
#
# Depending on your dataset and what you are trying to do, you might run into problems with
# xarray and MetPy. Below are examples of some of the most common issues
#
# - ``parse_cf`` not able to parse due to conflict
# - An axis not being available
# - An axis not being interpretable
# - Arrays not broadcasting in calculations
#
# **Coordinate Conflict**
#
# Code:
#
# ::
#
#     temperature = data.metpy.parse_cf('Temperature')
#
# Error Message:
#
# ::
#
#     /home/user/env/MetPy/metpy/xarray.py:305: UserWarning: DataArray
#     of requested variable has more than one x coordinate. Specify the
#     unique axes using the coordinates argument.
#
# Fix:
#
# Specify the ``coordinates`` argument to the ``parse_cf`` method to map the ``T`` (time),
# ``Z`` (vertical), ``Y``, and ``X`` axes (as applicable to your dataset) to the corresponding
# coordinates.
#
# ::
#
#     temperature = data.metpy.parse_cf('Temperature',
#                                       coordinates={'T': 'time', 'Z': 'isobaric',
#                                                    'Y': 'y', 'X': 'x'})
#
# **Axis Unavailable**
#
# Code:
#
# ::
#
#     data['Temperature'].metpy.vertical
#
# Error Message:
#
# ::
#
#     AttributeError: vertical attribute is not available.
#
# This means that your data variable does not have the coordinate that was requested, at
# least as far as the parser can recognize. Verify that you are requesting a
# coordinate that your data actually has, and if it still is not available,
# you will need to manually specify the coordinates via the coordinates argument
# discussed above.
#
# **Axis Not Interpretable**
#
# Code:
#
# ::
#
#     x, y, ensemble = data['Temperature'].metpy.coordinates('x', 'y', 'ensemble')
#
# Error Message:
#
# ::
#
#     AttributeError: 'ensemble' is not an interpretable axis
#
# This means that you are requesting a coordinate that MetPy is (currently) unable to parse.
# While this means it cannot be recognized automatically, you can still obtain your desired
# coordinate directly by accessing it by name. If you have a need for systematic
# identification of a new coordinate type, we welcome pull requests for such new functionality
# on GitHub!
#
# **Broadcasting in Calculations**
#
# Code:
#
# ::
#
#     theta = mpcalc.potential_temperature(data['isobaric3'], data['temperature'])
#
# Error Message:
#
# ::
#
#     ValueError: operands could not be broadcast together with shapes (9,31,81,131) (31,)
#
# This is a symptom of the incomplete integration of xarray with MetPy's calculations; the
# calculations currently convert the DataArrays to unit arrays that do not recognize which
# coordinates match with which. And so, we must do some manipulations.
#
# Fix 1 (xarray broadcasting):
#
# ::
#
#     pressure, temperature = xr.broadcast(data['isobaric3'], data['temperature'])
#     theta = mpcalc.potential_temperature(pressure, temperature)
#
# Fix 2 (unit array broadcasting):
#
# ::
#
#     theta = mpcalc.potential_temperature(
#         data['isobaric3'].metpy.unit_array[None, :, None, None],
#         data['temperature'].metpy.unit_array
#     )
#
