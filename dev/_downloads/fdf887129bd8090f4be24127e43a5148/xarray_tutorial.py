# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
==========================
xarray with MetPy Tutorial
==========================

`xarray <https://docs.xarray.dev/en/stable/>`_ is a powerful Python package that provides
N-dimensional labeled arrays and datasets following the Common Data Model. MetPy's suite of
meteorological calculations are designed to integrate with xarray DataArrays as one of its two
primary data models (the other being Pint Quantities). MetPy also provides DataArray and
Dataset *accessors* (collections of methods and properties attached to the ``.metpy`` property)
for coordinate/CRS and unit operations.

Full information on MetPy's accessors is available in the :doc:`appropriate section of the
reference guide </api/generated/metpy.xarray>`, otherwise, continue on in this
tutorial for a demonstration of the three main components of MetPy's integration with xarray
(coordinates/coordinate reference systems, units, and calculations), as well as instructive
examples for both CF-compliant and non-compliant datasets.

First, some general imports...
"""

import numpy as np
import xarray as xr

# Any import of metpy will activate the accessors
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.units import units

#########################################################################
# ...and opening some sample data to work with.

# Open the netCDF file as a xarray Dataset
data = xr.open_dataset(get_test_data('irma_gfs_example.nc', False))

# View a summary of the Dataset
data

#########################################################################
# While xarray can handle a wide variety of n-dimensional data (essentially anything that can
# be stored in a netCDF file), a common use case is working with gridded model output. Such
# model data can be obtained from a THREDDS Data Server using the `siphon package
# <https://unidata.github.io/siphon/>`_, but here we've used an example subset of GFS data
# from Hurricane Irma (September 5th, 2017) included in MetPy's test suite. Generally,
# a local file (or remote file via OPeNDAP) can be opened with ``xr.open_dataset("path")``.
#
# Going back to the above object, this ``Dataset`` consists of *dimensions* and their
# associated *coordinates*, which in turn make up the axes along which the *data variables*
# are defined. The dataset also has a dictionary-like collection of *attributes*. What happens
# if we look at just a single data variable?

temperature = data['Temperature_isobaric']
temperature

#########################################################################
# This is a ``DataArray``, which stores just a single data variable with its associated
# coordinates and attributes. These individual ``DataArray``\s are the kinds of objects that
# MetPy's calculations take as input (more on that in  `Calculations`_ section below).
#
# If you are more interested in learning about xarray's terminology and data structures, see
# the `terminology section <https://docs.xarray.dev/en/stable/terminology.html>`_ of xarray's
# documentation.
#
# Coordinates and Coordinate Reference Systems
# --------------------------------------------
#
# MetPy's first set of helpers comes with identifying *coordinate types*. In a given dataset,
# coordinates can have a variety of different names and yet refer to the same type (such as
# "isobaric1" and "isobaric3" both referring to vertical isobaric coordinates). Following
# CF conventions, as well as using some fall-back regular expressions, MetPy can
# systematically identify coordinates of the following types:
#
# - time
# - vertical
# - latitude
# - y
# - longitude
# - x
#
# When identifying a single coordinate, it is best to use the property directly associated
# with that type

temperature.metpy.time

#########################################################################
# When accessing multiple coordinate types simultaneously, you can use the ``.coordinates()``
# method to yield a generator for the respective coordinates

x, y = temperature.metpy.coordinates('x', 'y')

#########################################################################
# These coordinate type aliases can also be used in MetPy's wrapped ``.sel`` and ``.loc``
# for indexing and selecting on ``DataArray``\s. For example, to access 500 hPa heights at
# 1800Z,

heights = data['Geopotential_height_isobaric'].metpy.sel(
    time='2017-09-05 18:00',
    vertical=50000.
)

#########################################################################
# (Notice how we specified 50000 here without units...we'll go over a better alternative in
# the next section on units.)
#
# One point of warning: xarray's selection and indexing only works if these coordinates are
# *dimension coordinates*, meaning that they are 1D and share the name of their associated
# dimension. In practice, this means that you can't index a dataset that has 2D latitude and
# longitude coordinates by latitudes and longitudes, instead, you must index by the 1D y and x
# dimension coordinates. (What if these coordinates are missing, you may ask? See the final
# subsection on ``.assign_y_x`` for more details.)
#
# Beyond just the coordinates themselves, a common need for both calculations with and plots
# of geospatial data is knowing the coordinate reference system (CRS) on which the horizontal
# spatial coordinates are defined. MetPy follows the `CF Conventions
# <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#grid-mappings-and-projections>`_
# for its CRS definitions, which it then caches on the ``metpy_crs`` coordinate in order for
# it to persist through calculations and other array operations. There are two ways to do so
# in MetPy:
#
# First, if your dataset is already conforming to the CF Conventions, it will have a grid
# mapping variable that is associated with the other data variables by the ``grid_mapping``
# attribute. This is automatically parsed via the ``.parse_cf()`` method:

# Parse full dataset
data_parsed = data.metpy.parse_cf()

# Parse subset of dataset
data_subset = data.metpy.parse_cf([
    'u-component_of_wind_isobaric',
    'v-component_of_wind_isobaric',
    'Vertical_velocity_pressure_isobaric'
])

# Parse single variable
relative_humidity = data.metpy.parse_cf('Relative_humidity_isobaric')

#########################################################################
# If your dataset doesn't have a CF-conforming grid mapping variable, you can manually specify
# the CRS using the ``.assign_crs()`` method:

temperature = data['Temperature_isobaric'].metpy.assign_crs(
    grid_mapping_name='latitude_longitude',
    earth_radius=6371229.0
)

temperature

#########################################################################
# Notice the newly added ``metpy_crs`` non-dimension coordinate. Now how can we use this in
# practice? For individual ``DataArrays``\s, we can access the cartopy and pyproj objects
# corresponding to this CRS:

# Cartopy CRS, useful for plotting
relative_humidity.metpy.cartopy_crs

#########################################################################

# pyproj CRS, useful for projection transformations and forward/backward azimuth and great
# circle calculations
temperature.metpy.pyproj_crs

#########################################################################
# Finally, there are times when a certain horizontal coordinate type is missing from your
# dataset, and you need the other, that is, you have latitude/longitude and need y/x, or visa
# versa. This is where the ``.assign_y_x`` and ``.assign_latitude_longitude`` methods come in
# handy. Our current GFS sample won't work to demonstrate this (since, on its
# latitude-longitude grid, y is latitude and x is longitude), so for more information, take
# a look at the `Non-Compliant Dataset Example`_ below, or view the accessor documentation.

#########################################################################
# Units
# -----
#
# Since unit-aware calculations are a major part of the MetPy library, unit support is a major
# part of MetPy's xarray integration!
#
# One very important point of consideration is that xarray data variables (in both
# ``Dataset``\s and ``DataArray``\s) can store both unit-aware and unit-naive array types.
# Unit-naive array types will be used by default in xarray, so we need to convert to a
# unit-aware type if we want to use xarray operations while preserving unit correctness. MetPy
# provides the ``.quantify()`` method for this (named since we are turning the data stored
# inside the xarray object into a Pint ``Quantity`` object)

heights = heights.metpy.quantify()
heights

#########################################################################
# Notice how the units are now represented in the data itself, rather than as a text
# attribute. Now, even if we perform some kind of xarray operation (such as taking the zonal
# mean), the units are preserved

heights_mean = heights.mean('longitude')
heights_mean

#########################################################################
# However, this "quantification" is not without its consequences. By default, xarray loads its
# data lazily to conserve memory usage. Unless your data is chunked into a Dask array (using
# the ``chunks`` argument), this ``.quantify()`` method will load data into memory, which
# could slow your script or even cause your process to run out of memory. And so, we recommend
# subsetting your data before quantifying it.
#
# Also, these Pint ``Quantity`` data objects are not properly handled by xarray when writing
# to disk. And so, if you want to safely export your data, you will need to undo the
# quantification with the ``.dequantify()`` method, which converts your data back to a
# unit-naive array with the unit as a text attribute

heights_mean_str_units = heights_mean.metpy.dequantify()
heights_mean_str_units

#########################################################################
# Other useful unit integration features include:
#
# Unit-based selection/indexing:

heights_at_45_north = data['Geopotential_height_isobaric'].metpy.sel(
    latitude=45 * units.degrees_north,
    vertical=300 * units.hPa
)
heights_at_45_north

#########################################################################
# Unit conversion:

temperature_degc = temperature[0].metpy.convert_units('degC')
temperature_degc

#########################################################################
# To base unit conversion:

temperature_degk = temperature_degc.metpy.convert_to_base_units()
temperature_degk

#########################################################################
# Unit conversion for coordinates:
heights_on_hpa_levels = heights.metpy.convert_coordinate_units('isobaric3', 'hPa')
heights_on_hpa_levels['isobaric3']

#########################################################################
# Accessing just the underlying unit array:
heights_unit_array = heights.metpy.unit_array
heights_unit_array

#########################################################################
# Accessing just the underlying units:
height_units = heights.metpy.units
height_units

#########################################################################
# Calculations
# ------------
#
# MetPy's xarray integration extends to its calculation suite as well. Most grid-capable
# calculations (such as thermodynamics, kinematics, and smoothers) fully support xarray
# ``DataArray``\s by accepting them as inputs, returning them as outputs, and automatically
# using the attached coordinate data/metadata to determine grid arguments

heights = data_parsed.metpy.parse_cf('Geopotential_height_isobaric').metpy.sel(
    time='2017-09-05 18:00',
    vertical=500 * units.hPa
)
u_g, v_g = mpcalc.geostrophic_wind(heights)
u_g

#########################################################################
# For profile-based calculations (and most remaining calculations in the ``metpy.calc``
# module), xarray ``DataArray``\s are accepted as inputs, but the outputs remain Pint
# Quantities (typically scalars). Note that MetPy's profile calculations (such as CAPE and
# CIN) require the sounding to be ordered from highest to lowest pressure. As seen earlier
# in this tutorial, this data is ordered the other way, so we need to reverse the inputs
# to ``mpcalc.surface_based_cape_cin``.

data_at_point = data.metpy.sel(
    time1='2017-09-05 12:00',
    latitude=30 * units.degrees_north,
    longitude=260 * units.degrees_east
)
dewpoint = mpcalc.dewpoint_from_relative_humidity(
    data_at_point['Temperature_isobaric'],
    data_at_point['Relative_humidity_isobaric']
)
cape, cin = mpcalc.surface_based_cape_cin(
    data_at_point['isobaric3'][::-1],
    data_at_point['Temperature_isobaric'][::-1],
    dewpoint[::-1]
)
cape

#########################################################################
# A few remaining portions of MetPy's calculations (mainly the interpolation module and a few
# other functions) do not fully support xarray, and so, use of ``.values`` may be needed to
# convert to a bare NumPy array. For full information on xarray support for your function of
# interest, see the :doc:`/api/index`.
#
# CF-Compliant Dataset Example
# ----------------------------
#
# The GFS sample used throughout this tutorial so far has been an example of a CF-compliant
# dataset. These kinds of datasets are easiest to work with it MetPy, since most of the
# "xarray magic" uses CF metadata. For this kind of dataset, a typical workflow looks like the
# following

# Load data, parse it for a CF grid mapping, and promote lat/lon data variables to coordinates
data = xr.open_dataset(
    get_test_data('narr_example.nc', False)
).metpy.parse_cf().set_coords(['lat', 'lon'])

# Subset to only the data you need to save on memory usage
subset = data.metpy.sel(isobaric=500 * units.hPa)

# Quantify if you plan on performing xarray operations that need to maintain unit correctness
subset = subset.metpy.quantify()

# Perform calculations
heights = mpcalc.smooth_gaussian(subset['Geopotential_height'], 5)
subset['u_geo'], subset['v_geo'] = mpcalc.geostrophic_wind(heights)

# Plot
heights.plot()

#########################################################################

# Save output
subset.metpy.dequantify().drop_vars('metpy_crs').to_netcdf('500hPa_analysis.nc')

#########################################################################
# Non-Compliant Dataset Example
# -----------------------------
#
# When CF metadata (such as grid mapping, coordinate attributes, etc.) are missing, a bit more
# work is required to manually supply the required information, for example,

nonstandard = xr.Dataset({
    'temperature': (('y', 'x'), np.arange(0, 9).reshape(3, 3) * units.degC),
    'y': ('y', np.arange(0, 3) * 1e5, {'units': 'km'}),
    'x': ('x', np.arange(0, 3) * 1e5, {'units': 'km'})
})

# Add both CRS and then lat/lon coords using chained methods
data = nonstandard.metpy.assign_crs(
    grid_mapping_name='lambert_conformal_conic',
    latitude_of_projection_origin=38.5,
    longitude_of_central_meridian=262.5,
    standard_parallel=38.5,
    earth_radius=6371229.0
).metpy.assign_latitude_longitude()

# Preview the changes
data

#########################################################################
# Once the CRS and additional coordinates are assigned, you can generally proceed as you would
# for a CF-compliant dataset.
#
# What Could Go Wrong?
# --------------------
#
# Depending on your dataset and what you are trying to do, you might run into problems with
# xarray and MetPy. Below are examples of some of the most common issues
#
# - Multiple coordinate conflict
# - An axis not being available
# - An axis not being interpretable
# - ``UndefinedUnitError``
#
# **Coordinate Conflict**
#
# Code:
#
# ::
#
#     x = data['Temperature'].metpy.x
#
# Error Message:
#
# ::
#
#     /home/user/env/MetPy/metpy/xarray.py:305: UserWarning: More than
#     one x coordinate present for variable "Temperature".
#
# Fix:
#
# Manually assign the coordinates using the ``assign_coordinates()`` method on your DataArray,
# or by specifying the ``coordinates`` argument to the ``parse_cf()`` method on your Dataset,
# to map the ``time``, ``vertical``, ``y``, ``latitude``, ``x``, and ``longitude`` axes (as
# applicable to your data) to the corresponding coordinates.
#
# ::
#
#     data['Temperature'].assign_coordinates({'time': 'time', 'vertical': 'isobaric',
#                                             'y': 'y', 'x': 'x'})
#     x = data['Temperature'].metpy.x
#
# or
#
# ::
#
#     temperature = data.metpy.parse_cf('Temperature',
#                                       coordinates={'time': 'time', 'vertical': 'isobaric',
#                                                    'y': 'y', 'x': 'x'})
#     x = temperature.metpy.x
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
# you will need to manually specify the coordinates as discussed above.
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
# **Undefined Unit Error**
#
# If the units attribute on your xarray data is not recognizable by Pint, you will likely
# receive an ``UndefinedUnitError``. In this case, you will likely have to update the units
# attribute to one that can be parsed properly by Pint. It is our aim to have all valid
# CF/UDUNITS unit strings be parseable, but this work is ongoing. If many variables in your
# dataset are not parseable, the ``.update_attribute`` method on the MetPy accessor may come
# in handy.
