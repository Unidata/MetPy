# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Provide accessors to enhance interoperability between XArray and MetPy."""
from __future__ import absolute_import

import functools

import xarray as xr
from xarray.core.accessors import DatetimeAccessor

from .units import DimensionalityError, units

__all__ = []


# TODO: Should we be providing renaming/mapping for easy access to x/y coordinates?
@xr.register_dataarray_accessor('metpy')
class MetPyAccessor(object):
    """Provide custom attributes and methods on XArray DataArray for MetPy functionality."""

    def __init__(self, data_array):
        """Initialize accessor with a DataArray."""
        self._data_array = data_array
        self._units = self._data_array.attrs.get('units', 'dimensionless')

    @property
    def unit_array(self):
        """Return data values as a `pint.Quantity`."""
        return self._data_array.values * units(self._units)

    @unit_array.setter
    def unit_array(self, values):
        """Set data values as a `pint.Quantity`."""
        self._data_array.values = values
        self._units = self._data_array.attrs['units'] = str(values.units)

    def convert_units(self, units):
        """Convert the data values to different units in-place."""
        self.unit_array = self.unit_array.to(units)

    @property
    def crs(self):
        """Provide easy access to the `crs` coordinate."""
        if 'crs' in self._data_array.coords:
            return self._data_array.coords['crs'].item()
        raise AttributeError('crs attribute is not available due to lack of crs coordinate.')

    @property
    def cartopy_crs(self):
        """Return the coordinate reference system (CRS) as a cartopy object."""
        return self.crs.to_cartopy()


@xr.register_dataset_accessor('metpy')
class CFConventionHandler(object):
    """Provide custom attributes and methods on XArray Dataset for MetPy functionality."""

    def __init__(self, dataset):
        """Initialize accessor with a Dataset."""
        self._dataset = dataset

    def parse_cf(self, varname):
        """Parse Climate and Forecasting (CF) convention metadata."""
        from .plots.mapping import CFProjection

        var = self._dataset[varname]
        if 'grid_mapping' in var.attrs:
            proj_name = var.attrs['grid_mapping']
            try:
                proj_var = self._dataset.variables[proj_name]
            except KeyError:
                import warnings
                warnings.warn(
                    'Could not find variable corresponding to the value of '
                    'grid_mapping: {}'.format(proj_name))
            else:
                var.coords['crs'] = CFProjection(proj_var.attrs)
                var.attrs.pop('grid_mapping')
                self._fixup_coords(var)

        # Trying to guess whether we should be adding a crs to this variable's coordinates
        # First make sure it's missing CRS but isn't lat/lon itself
        if not (self._check_lat(var) or self._check_lon(var)) and 'crs' not in var.coords:
            # Look for both lat/lon in the coordinates
            has_lat = has_lon = False
            for coord_var in var.coords.values():
                has_lat = has_lat or self._check_lat(coord_var)
                has_lon = has_lon or self._check_lon(coord_var)

            # If we found them, create a lat/lon projection as the crs coord
            if has_lat and has_lon:
                var.coords['crs'] = CFProjection({'grid_mapping_name': 'latitude_longitude'})

        return var

    @staticmethod
    def _check_lat(var):
        if var.attrs.get('standard_name') == 'latitude':
            return True

        units = var.attrs.get('units', '').replace('degrees', 'degree')
        return units in {'degree_north', 'degree_N', 'degreeN'}

    @staticmethod
    def _check_lon(var):
        if var.attrs.get('standard_name') == 'longitude':
            return True

        units = var.attrs.get('units', '').replace('degrees', 'degree')
        return units in {'degree_east', 'degree_E', 'degreeE'}

    def _fixup_coords(self, var):
        """Clean up the units on the coordinate variables."""
        for coord_name, data_array in var.coords.items():
            if data_array.attrs.get('standard_name') in ('projection_x_coordinate',
                                                         'projection_y_coordinate'):
                try:
                    var.coords[coord_name].metpy.convert_units('meters')
                except DimensionalityError:  # Radians!
                    new_data_array = data_array.copy()
                    height = var.coords['crs'].item()['perspective_point_height']
                    scaled_vals = new_data_array.metpy.unit_array * (height * units.meters)
                    new_data_array.metpy.unit_array = scaled_vals.to('meters')
                    var.coords[coord_name] = new_data_array


def preprocess_xarray(func):
    """Decorate a function to convert all DataArray arguments to pint.Quantities.

    This uses the metpy xarray accessors to do the actual conversion.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = tuple(a.metpy.unit_array if isinstance(a, xr.DataArray) else a for a in args)
        kwargs = {name: (v.metpy.unit_array if isinstance(v, xr.DataArray) else v)
                  for name, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper


# If DatetimeAccessor does not have a strftime, monkey patch one in
if not hasattr(DatetimeAccessor, 'strftime'):
    def strftime(self, date_format):
        """Format time as a string."""
        import pandas as pd
        values = self._obj.data
        values_as_series = pd.Series(values.ravel())
        strs = values_as_series.dt.strftime(date_format)
        return strs.values.reshape(values.shape)

    DatetimeAccessor.strftime = strftime
