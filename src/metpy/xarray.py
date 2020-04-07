# Copyright (c) 2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Provide accessors to enhance interoperability between xarray and MetPy.

MetPy relies upon the `CF Conventions <http://cfconventions.org/>`_. to provide helpful
attributes and methods on xarray DataArrays and Dataset for working with
coordinate-related metadata. Also included are several attributes and methods for unit
operations.

These accessors will be activated with any import of MetPy. Do not use the
``MetPyDataArrayAccessor`` or ``MetPyDatasetAccessor`` classes directly, instead, utilize the
applicable properties and methods via the ``.metpy`` attribute on an xarray DataArray or
Dataset.

See Also: :doc:`xarray with MetPy Tutorial </tutorials/xarray_tutorial>`.
"""
import functools
import logging
import re
import warnings

import numpy as np
import xarray as xr

from ._vendor.xarray import either_dict_or_kwargs, expanded_indexer, is_dict_like
from .deprecation import metpyDeprecation
from .units import DimensionalityError, UndefinedUnitError, units

__all__ = []
metpy_axes = ['time', 'vertical', 'y', 'latitude', 'x', 'longitude']
cf_to_metpy_axes = {'T': 'time', 'Z': 'vertical', 'Y': 'y', 'X': 'x'}  # TODO: Remove in 1.0

# Define the criteria for coordinate matches
coordinate_criteria = {
    'standard_name': {
        'time': 'time',
        'vertical': {'air_pressure', 'height', 'geopotential_height', 'altitude',
                     'model_level_number', 'atmosphere_ln_pressure_coordinate',
                     'atmosphere_sigma_coordinate',
                     'atmosphere_hybrid_sigma_pressure_coordinate',
                     'atmosphere_hybrid_height_coordinate', 'atmosphere_sleve_coordinate',
                     'height_above_geopotential_datum', 'height_above_reference_ellipsoid',
                     'height_above_mean_sea_level'},
        'y': 'projection_y_coordinate',
        'latitude': 'latitude',
        'x': 'projection_x_coordinate',
        'longitude': 'longitude'
    },
    '_CoordinateAxisType': {
        'time': 'Time',
        'vertical': {'GeoZ', 'Height', 'Pressure'},
        'y': 'GeoY',
        'latitude': 'Lat',
        'x': 'GeoX',
        'longitude': 'Lon'
    },
    'axis': {
        'time': 'T',
        'vertical': 'Z',
        'y': 'Y',
        'x': 'X'
    },
    'positive': {
        'vertical': {'up', 'down'}
    },
    'units': {
        'vertical': {
            'match': 'dimensionality',
            'units': 'Pa'
        },
        'latitude': {
            'match': 'name',
            'units': {'degree_north', 'degree_N', 'degreeN', 'degrees_north', 'degrees_N',
                      'degreesN'}
        },
        'longitude': {
            'match': 'name',
            'units': {'degree_east', 'degree_E', 'degreeE', 'degrees_east', 'degrees_E',
                      'degreesE'}
        },
    },
    'regular_expression': {
        'time': r'time[0-9]*',
        'vertical': (r'(lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|isobaric|pres|'
                     r'isotherm)[a-z_]*[0-9]*'),
        'y': r'y',
        'latitude': r'x?lat[a-z0-9]*',
        'x': r'x',
        'longitude': r'x?lon[a-z0-9]*'
    }
}

log = logging.getLogger(__name__)


@xr.register_dataarray_accessor('metpy')
class MetPyDataArrayAccessor:
    r"""Provide custom attributes and methods on xarray DataArrays for MetPy functionality.

    This accessor provides several convenient attributes and methods through the `.metpy`
    attribute on a DataArray. For example, MetPy can identify the coordinate corresponding
    to a particular axis (given sufficent metadata):

        >>> import xarray as xr
        >>> temperature = xr.DataArray([[0, 1], [2, 3]], dims=('lat', 'lon'),
        ...                            coords={'lat': [40, 41], 'lon': [-105, -104]},
        ...                            attrs={'units': 'degC'})
        >>> temperature.metpy.x
        <xarray.DataArray 'lon' (lon: 2)>
        array([-105, -104])
        Coordinates:
          * lon      (lon) int64 -105 -104
        Attributes:
            _metpy_axis:  x,longitude

    """

    def __init__(self, data_array):  # noqa: D107
        # Initialize accessor with a DataArray. (Do not use directly).
        self._data_array = data_array
        self._units = self._data_array.attrs.get('units', 'dimensionless')

    @property
    def units(self):
        """Return the units of this DataArray as a `pint.Quantity`."""
        if self._units != '%':
            return units(self._units)
        else:
            return units('percent')

    @property
    def unit_array(self):
        """Return the data values of this DataArray as a `pint.Quantity`."""
        return self._data_array.values * self.units

    @unit_array.setter
    def unit_array(self, values):
        """Set data values from a `pint.Quantity`."""
        try:
            self._data_array.values = values.magnitude
        except ValueError:
            raise ValueError('Unable to set unit array to a dimension coordinate. Use the '
                             'assign_coords method instead')
        self._units = self._data_array.attrs['units'] = str(values.units)

    def convert_units(self, units):
        """Convert the data values to different units in-place."""
        self.unit_array = self.unit_array.to(units)

    def convert_coordinate_units(self, coord, units):
        """Return new DataArray with coordinate converted to different units.

        Notes
        -----
        Backported from MetPy 1.0.
        """
        new_coord_var = self._data_array[coord].copy(
            data=self._data_array[coord].metpy.unit_array.m_as(units)
        )
        new_coord_var.attrs['units'] = str(units)
        return self._data_array.assign_coords(coords={coord: new_coord_var})

    @property
    def crs(self):
        """Return the coordinate reference system (CRS) as a CFProjection object."""
        if 'crs' in self._data_array.coords:
            return self._data_array.coords['crs'].item()
        raise AttributeError('crs attribute is not available.')

    @property
    def cartopy_crs(self):
        """Return the coordinate reference system (CRS) as a cartopy object."""
        return self.crs.to_cartopy()

    @property
    def cartopy_globe(self):
        """Return the globe belonging to the coordinate reference system (CRS)."""
        return self.crs.cartopy_globe

    def _fixup_coordinate_map(self, coord_map):
        """Ensure sure we have coordinate variables in map, not coordinate names."""
        for axis in coord_map:
            if coord_map[axis] is not None and not isinstance(coord_map[axis], xr.DataArray):
                coord_map[axis] = self._data_array[coord_map[axis]]

        # Adapt to old CF axis dict keys
        # TODO: Remove in v1.0
        if any(cf_axis in coord_map for cf_axis in cf_to_metpy_axes):
            warnings.warn('The use of CF axes types in the coordinate mapping has been '
                          'deprecated and will be removed in v1.0.', metpyDeprecation)
            coord_map = {(cf_to_metpy_axes[axis]
                          if axis in cf_to_metpy_axes else axis): coord
                         for axis, coord in coord_map.items()}

        return coord_map

    def assign_coordinates(self, coordinates):
        """Assign the given coordinates to the given MetPy axis types.

        Parameters
        ----------
        coordinates : dict or None
            Mapping from axis types ('time', 'vertical', 'y', 'latitude', 'x', 'longitude') to
            coordinates of this DataArray. Coordinates can either be specified directly or by
            their name. If ``None``, clears the `_metpy_axis` attribute on all coordinates,
            which will trigger reparsing of all coordinates on next access.

        Notes
        -----
        Prior to v0.12, ``coordinates`` was taken as a mapping from CF axis types
        ('T', 'Z', 'Y', 'X') to coordinates. This behavior has been deprecated and will be
        be removed in v1.0.

        """
        if coordinates:
            # Assign the _metpy_axis attributes according to supplied mapping
            coordinates = self._fixup_coordinate_map(coordinates)
            for axis in coordinates:
                if coordinates[axis] is not None:
                    _assign_axis(coordinates[axis].attrs, axis)
        else:
            # Clear _metpy_axis attribute on all coordinates
            for coord_var in self._data_array.coords.values():
                coord_var.attrs.pop('_metpy_axis', None)

    def _generate_coordinate_map(self):
        """Generate a coordinate map via CF conventions and other methods."""
        coords = self._data_array.coords.values()
        # Parse all the coordinates, attempting to identify x, longitude, y, latitude,
        # vertical, time
        coord_lists = {'time': [], 'vertical': [], 'y': [], 'latitude': [], 'x': [],
                       'longitude': []}
        for coord_var in coords:
            # Identify the coordinate type using check_axis helper
            for axis in coord_lists:
                if check_axis(coord_var, axis):
                    coord_lists[axis].append(coord_var)

        # Fill in x/y with longitude/latitude if x/y not otherwise present
        for geometric, graticule in (('y', 'latitude'), ('x', 'longitude')):
            if len(coord_lists[geometric]) == 0 and len(coord_lists[graticule]) > 0:
                coord_lists[geometric] = coord_lists[graticule]

        # Filter out multidimensional coordinates where not allowed
        require_1d_coord = ['time', 'vertical']  # TODO: Add 'y', 'x' in v1.0
        for axis in require_1d_coord:
            coord_lists[axis] = [coord for coord in coord_lists[axis] if coord.ndim <= 1]

        # Resolve any coordinate type duplication
        axis_duplicates = [axis for axis in coord_lists if len(coord_lists[axis]) > 1]
        for axis in axis_duplicates:
            self._resolve_axis_duplicates(axis, coord_lists)

        # Collapse the coord_lists to a coord_map
        return {axis: (coord_lists[axis][0] if len(coord_lists[axis]) > 0 else None)
                for axis in coord_lists}

    def _resolve_axis_duplicates(self, axis, coord_lists):
        """Handle coordinate duplication for an axis type if it arises."""
        # If one and only one of the possible axes is a dimension, use it
        dimension_coords = [coord_var for coord_var in coord_lists[axis] if
                            coord_var.name in coord_var.dims]
        if len(dimension_coords) == 1:
            coord_lists[axis] = dimension_coords
            return

        # Ambiguous axis, raise warning and do not parse
        warnings.warn('More than one ' + axis + ' coordinate present for variable "'
                      + self._data_array.name + '".')
        coord_lists[axis] = []

    def _metpy_axis_search(self, metpy_axis):
        """Search for cached _metpy_axis attribute on the coordinates, otherwise parse."""
        # Search for coord with proper _metpy_axis
        coords = self._data_array.coords.values()
        for coord_var in coords:
            if metpy_axis in coord_var.attrs.get('_metpy_axis', '').split(','):
                return coord_var

        # Opportunistically parse all coordinates, and assign if not already assigned
        coord_map = self._generate_coordinate_map()
        for axis, coord_var in coord_map.items():
            if (coord_var is not None
                and not any(axis in coord.attrs.get('_metpy_axis', '').split(',')
                            for coord in coords)):
                # Warn if assigning multidimensional coord to x/y
                # TODO: Remove in v1.0
                if coord_var.ndim > 1 and axis in ('y', 'x'):
                    warnings.warn(f'Multidimensional coordinate {coord_var.name} assigned for '
                                  f'axis "{axis}". This behavior has been deprecated and will '
                                  'be removed in v1.0 (only one-dimensional coordinates will '
                                  f'be available for the "{axis}" axis).', metpyDeprecation)

                _assign_axis(coord_var.attrs, axis)

        # Return parsed result (can be None if none found)
        return coord_map[metpy_axis]

    def _axis(self, axis):
        """Return the coordinate variable corresponding to the given individual axis type."""
        if axis in metpy_axes:
            coord_var = self._metpy_axis_search(axis)
            if coord_var is not None:
                return coord_var
            else:
                raise AttributeError(axis + ' attribute is not available.')
        else:
            raise AttributeError("'" + axis + "' is not an interpretable axis.")

    def coordinates(self, *args):
        """Return the coordinate variables corresponding to the given axes types.

        Parameters
        ----------
        args : str
            Strings describing the axes type(s) to obtain. Currently understood types are
            'time', 'vertical', 'y', 'latitude', 'x', and 'longitude'.

        Notes
        -----
        This method is designed for use with mutliple coordinates; it returns a generator. To
        access a single coordinate, use the appropriate attribute on the accessor, or use tuple
        unpacking.

        """
        for arg in args:
            yield self._axis(arg)

    @property
    def time(self):
        """Return the time coordinate."""
        return self._axis('time')

    @property
    def vertical(self):
        """Return the vertical coordinate."""
        return self._axis('vertical')

    @property
    def y(self):
        """Return the y coordinate.

        Notes
        -----
        Prior to version v0.12, this attribute allowed multidimensional coordinates (including
        latitude when another y coordinate was not available). This behavior has been
        deprecated and will be removed in v1.0.

        """
        return self._axis('y')

    @property
    def latitude(self):
        """Return the latitude coordinate (if it exists)."""
        return self._axis('latitude')

    @property
    def x(self):
        """Return the x coordinate.

        Notes
        -----
        Prior to version v0.12, this attribute allowed multidimensional coordinates (including
        longitude when another x coordinate was not available). This behavior has been
        deprecated and will be removed in v1.0.

        """
        return self._axis('x')

    @property
    def longitude(self):
        """Return the longitude coordinate (if it exists)."""
        return self._axis('longitude')

    def coordinates_identical(self, other):
        """Return whether or not the coordinates of other match this DataArray's."""
        # If the number of coordinates do not match, we know they can't match.
        if len(self._data_array.coords) != len(other.coords):
            return False

        # If same length, iterate over all of them and check
        for coord_name, coord_var in self._data_array.coords.items():
            if coord_name not in other.coords or not other[coord_name].identical(coord_var):
                return False

        # Otherwise, they match.
        return True

    def as_timestamp(self):
        """Return the data as unix timestamp (for easier time derivatives).

        Notes
        -----
        This function has been deprecated and will be removed in v1.0. For the purposes of
        time derivatives, use ``time_deltas`` instead, which allows sub-second precision.

        """
        warnings.warn('This function has been deprecated and will be removed in v1.0. Use '
                      'time_deltas instead.', metpyDeprecation)
        attrs = {key: self._data_array.attrs[key] for key in
                 {'standard_name', 'long_name', 'axis', '_metpy_axis'}
                 & set(self._data_array.attrs)}
        attrs['units'] = 'seconds'
        return xr.DataArray(self._data_array.values.astype('datetime64[s]').astype('int64'),
                            name=self._data_array.name,
                            coords=self._data_array.coords,
                            dims=self._data_array.dims,
                            attrs=attrs)

    @property
    def time_deltas(self):
        """Return the time difference of the data in seconds (to microsecond precision)."""
        return (np.diff(self._data_array.values).astype('timedelta64[us]').astype('int64')
                / 1e6 * units.s)

    def find_axis_name(self, axis):
        """Return the name of the axis corresponding to the given identifier.

        Parameters
        ----------
        axis : str or int
            Identifier for an axis. Can be the an axis number (integer), dimension coordinate
            name (string) or a standard axis type (string).

        """
        if isinstance(axis, int):
            # If an integer, use the corresponding dimension
            return self._data_array.dims[axis]
        elif axis not in self._data_array.dims and axis in metpy_axes:
            # If not a dimension name itself, but a valid axis type, get the name of the
            # coordinate corresponding to that axis type
            return self._axis(axis).name
        elif axis in self._data_array.dims and axis in self._data_array.coords:
            # If this is a dimension coordinate name, use it directly
            return axis
        else:
            # Otherwise, not valid
            raise ValueError('Given axis is not valid. Must be an axis number, a dimension '
                             'coordinate name, or a standard axis type.')

    class _LocIndexer:
        """Provide the unit-wrapped .loc indexer for data arrays."""

        def __init__(self, data_array):
            self.data_array = data_array

        def expand(self, key):
            """Parse key using xarray utils to ensure we have dimension names."""
            if not is_dict_like(key):
                labels = expanded_indexer(key, self.data_array.ndim)
                key = dict(zip(self.data_array.dims, labels))
            return key

        def __getitem__(self, key):
            key = _reassign_quantity_indexer(self.data_array, self.expand(key))
            return self.data_array.loc[key]

        def __setitem__(self, key, value):
            key = _reassign_quantity_indexer(self.data_array, self.expand(key))
            self.data_array.loc[key] = value

    @property
    def loc(self):
        """Wrap DataArray.loc with an indexer to handle units and coordinate types."""
        return self._LocIndexer(self._data_array)

    def sel(self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs):
        """Wrap DataArray.sel to handle units and coordinate types."""
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'sel')
        indexers = _reassign_quantity_indexer(self._data_array, indexers)
        return self._data_array.sel(indexers, method=method, tolerance=tolerance, drop=drop)


@xr.register_dataset_accessor('metpy')
class MetPyDatasetAccessor:
    """Provide custom attributes and methods on XArray Datasets for MetPy functionality.

    This accessor provides parsing of CF metadata and unit-/coordinate-type-aware selection.

        >>> import xarray as xr
        >>> from metpy.testing import get_test_data
        >>> ds = xr.open_dataset(get_test_data('narr_example.nc', False)).metpy.parse_cf()
        >>> print(ds['crs'].item())
        Projection: lambert_conformal_conic

    """

    def __init__(self, dataset):  # noqa: D107
        # Initialize accessor with a Dataset. (Do not use directly).
        self._dataset = dataset

    def parse_cf(self, varname=None, coordinates=None):
        """Parse Climate and Forecasting (CF) convention metadata.

        Parameters
        ----------
        varname : str or iterable of str, optional
            Name of the variable(s) to extract from the dataset while parsing for CF metadata.
            Defaults to all variables.
        coordinates : dict, optional
            Dictionary mapping CF axis types to coordinates of the variable(s). Only specify
            if you wish to override MetPy's automatic parsing of some axis type(s).

        Returns
        -------
        `xarray.DataArray` or `xarray.Dataset`
            Parsed DataArray (if varname is a string) or Dataset

        """
        from .cbook import iterable
        from .plots.mapping import CFProjection

        if varname is None:
            # If no varname is given, parse all variables in the dataset
            varname = list(self._dataset.data_vars)

        if iterable(varname) and not isinstance(varname, str):
            # If non-string iterable is given, apply recursively across the varnames
            subset = xr.merge([self.parse_cf(single_varname, coordinates=coordinates)
                               for single_varname in varname])
            subset.attrs = self._dataset.attrs
            return subset

        var = self._dataset[varname]
        if 'grid_mapping' in var.attrs:
            proj_name = var.attrs['grid_mapping']
            try:
                proj_var = self._dataset.variables[proj_name]
            except KeyError:
                log.warning(
                    'Could not find variable corresponding to the value of '
                    'grid_mapping: {}'.format(proj_name))
            else:
                var.coords['crs'] = CFProjection(proj_var.attrs)

        var = self._fixup_coords(var)

        # Trying to guess whether we should be adding a crs to this variable's coordinates
        # First make sure it's missing CRS but isn't lat/lon itself
        if not check_axis(var, 'latitude', 'longitude') and 'crs' not in var.coords:
            # Look for both lat/lon in the coordinates
            has_lat = has_lon = False
            for coord_var in var.coords.values():
                has_lat = has_lat or check_axis(coord_var, 'latitude')
                has_lon = has_lon or check_axis(coord_var, 'longitude')

            # If we found them, create a lat/lon projection as the crs coord
            if has_lat and has_lon:
                var.coords['crs'] = CFProjection({'grid_mapping_name': 'latitude_longitude'})
                log.warning('Found latitude/longitude values, assuming latitude_longitude '
                            'for projection grid_mapping variable')

        # Assign coordinates if the coordinates argument is given
        if coordinates is not None:
            var.metpy.assign_coordinates(coordinates)

        return var

    def _fixup_coords(self, var):
        """Clean up the units on the coordinate variables."""
        for coord_name, coord_var in var.coords.items():
            if (check_axis(coord_var, 'x', 'y')
                    and not check_axis(coord_var, 'longitude', 'latitude')):
                try:
                    var = var.metpy.convert_coordinate_units(coord_name, 'meter')
                except DimensionalityError:  # Radians!
                    if 'crs' in var.coords:
                        height = var.coords['crs'].item()['perspective_point_height']
                        new_coord_var = coord_var.copy(
                            data=(
                                coord_var.metpy.unit_array * (height * units.meters)
                            ).m_as('meter')
                        )
                        new_coord_var.attrs['units'] = 'meter'
                        var = var.assign_coords(coords={coord_name: new_coord_var})

        return var

    class _LocIndexer:
        """Provide the unit-wrapped .loc indexer for datasets."""

        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, key):
            parsed_key = _reassign_quantity_indexer(self.dataset, key)
            return self.dataset.loc[parsed_key]

    @property
    def loc(self):
        """Wrap Dataset.loc with an indexer to handle units and coordinate types."""
        return self._LocIndexer(self._dataset)

    def sel(self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs):
        """Wrap Dataset.sel to handle units."""
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'sel')
        indexers = _reassign_quantity_indexer(self._dataset, indexers)
        return self._dataset.sel(indexers, method=method, tolerance=tolerance, drop=drop)


def _assign_axis(attributes, axis):
    """Assign the given axis to the _metpy_axis attribute."""
    existing_axes = attributes.get('_metpy_axis', '').split(',')
    if ((axis == 'y' and 'latitude' in existing_axes)
            or (axis == 'latitude' and 'y' in existing_axes)):
        # Special case for combined y/latitude handling
        attributes['_metpy_axis'] = 'y,latitude'
    elif ((axis == 'x' and 'longitude' in existing_axes)
            or (axis == 'longitude' and 'x' in existing_axes)):
        # Special case for combined x/longitude handling
        attributes['_metpy_axis'] = 'x,longitude'
    else:
        # Simply add it/overwrite past value
        attributes['_metpy_axis'] = axis
    return attributes


def check_axis(var, *axes):
    """Check if the criteria for any of the given axes are satisfied.

    Parameters
    ----------
    var : `xarray.DataArray`
        DataArray belonging to the coordinate to be checked
    axes : str
        Axis type(s) to check for. Currently can check for 'time', 'vertical', 'y', 'latitude',
        'x', and 'longitude'.

    """
    for axis in axes:
        # Check for
        #   - standard name (CF option)
        #   - _CoordinateAxisType (from THREDDS)
        #   - axis (CF option)
        #   - positive (CF standard for non-pressure vertical coordinate)
        for criterion in ('standard_name', '_CoordinateAxisType', 'axis', 'positive'):
            if (var.attrs.get(criterion, 'absent') in
                    coordinate_criteria[criterion].get(axis, set())):
                return True

        # Check for units, either by dimensionality or name
        try:
            if (axis in coordinate_criteria['units'] and (
                    (
                        coordinate_criteria['units'][axis]['match'] == 'dimensionality'
                        and (units.get_dimensionality(var.attrs.get('units'))
                             == units.get_dimensionality(
                                 coordinate_criteria['units'][axis]['units']))
                    ) or (
                        coordinate_criteria['units'][axis]['match'] == 'name'
                        and var.attrs.get('units')
                        in coordinate_criteria['units'][axis]['units']
                    ))):
                return True
        except UndefinedUnitError:
            pass

        # Check if name matches regular expression (non-CF failsafe)
        if re.match(coordinate_criteria['regular_expression'][axis], var.name.lower()):
            return True

    # If no match has been made, return False (rather than None)
    return False


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


def check_matching_coordinates(func):
    """Decorate a function to make sure all given DataArrays have matching coordinates."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data_arrays = ([a for a in args if isinstance(a, xr.DataArray)]
                       + [a for a in kwargs.values() if isinstance(a, xr.DataArray)])
        if len(data_arrays) > 1:
            first = data_arrays[0]
            for other in data_arrays[1:]:
                if not first.metpy.coordinates_identical(other):
                    raise ValueError('Input DataArray arguments must be on same coordinates.')
        return func(*args, **kwargs)
    return wrapper


# If DatetimeAccessor does not have a strftime (xarray <0.12.2), monkey patch one in
try:
    from xarray.core.accessors import DatetimeAccessor
    if not hasattr(DatetimeAccessor, 'strftime'):
        def strftime(self, date_format):
            """Format time as a string."""
            import pandas as pd
            values = self._obj.data
            values_as_series = pd.Series(values.ravel())
            strs = values_as_series.dt.strftime(date_format)
            return strs.values.reshape(values.shape)

        DatetimeAccessor.strftime = strftime
except ImportError:
    pass


def _reassign_quantity_indexer(data, indexers):
    """Reassign a units.Quantity indexer to units of relevant coordinate."""
    def _to_magnitude(val, unit):
        try:
            return val.to(unit).m
        except AttributeError:
            return val

    # Update indexers keys for axis type -> coord name replacement
    indexers = {(key if not isinstance(data, xr.DataArray) or key in data.dims
                 or key not in metpy_axes else
                 next(data.metpy.coordinates(key)).name): indexers[key]
                for key in indexers}

    # Update indexers to handle quantities and slices of quantities
    reassigned_indexers = {}
    for coord_name in indexers:
        coord_units = data[coord_name].metpy.units
        if isinstance(indexers[coord_name], slice):
            # Handle slices of quantities
            start = _to_magnitude(indexers[coord_name].start, coord_units)
            stop = _to_magnitude(indexers[coord_name].stop, coord_units)
            step = _to_magnitude(indexers[coord_name].step, coord_units)
            reassigned_indexers[coord_name] = slice(start, stop, step)
        else:
            # Handle quantities
            reassigned_indexers[coord_name] = _to_magnitude(indexers[coord_name], coord_units)

    return reassigned_indexers


__all__ = ('MetPyDataArrayAccessor', 'MetPyDatasetAccessor')
