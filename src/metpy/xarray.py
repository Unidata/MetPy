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
from inspect import signature
import logging
import re
import warnings

import numpy as np
import xarray as xr

from ._vendor.xarray import either_dict_or_kwargs, expanded_indexer, is_dict_like
from .units import DimensionalityError, UndefinedUnitError, units

__all__ = []
metpy_axes = ["time", "vertical", "y", "latitude", "x", "longitude"]

# Define the criteria for coordinate matches
coordinate_criteria = {
    "standard_name": {
        "time": "time",
        "vertical": {
            "air_pressure",
            "height",
            "geopotential_height",
            "altitude",
            "model_level_number",
            "atmosphere_ln_pressure_coordinate",
            "atmosphere_sigma_coordinate",
            "atmosphere_hybrid_sigma_pressure_coordinate",
            "atmosphere_hybrid_height_coordinate",
            "atmosphere_sleve_coordinate",
            "height_above_geopotential_datum",
            "height_above_reference_ellipsoid",
            "height_above_mean_sea_level",
        },
        "y": "projection_y_coordinate",
        "latitude": "latitude",
        "x": "projection_x_coordinate",
        "longitude": "longitude",
    },
    "_CoordinateAxisType": {
        "time": "Time",
        "vertical": {"GeoZ", "Height", "Pressure"},
        "y": "GeoY",
        "latitude": "Lat",
        "x": "GeoX",
        "longitude": "Lon",
    },
    "axis": {"time": "T", "vertical": "Z", "y": "Y", "x": "X"},
    "positive": {"vertical": {"up", "down"}},
    "units": {
        "vertical": {"match": "dimensionality", "units": "Pa"},
        "latitude": {
            "match": "name",
            "units": {
                "degree_north",
                "degree_N",
                "degreeN",
                "degrees_north",
                "degrees_N",
                "degreesN",
            },
        },
        "longitude": {
            "match": "name",
            "units": {
                "degree_east",
                "degree_E",
                "degreeE",
                "degrees_east",
                "degrees_E",
                "degreesE",
            },
        },
    },
    "regular_expression": {
        "time": r"time[0-9]*",
        "vertical": (
            r"(lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|isobaric|pres|"
            r"isotherm)[a-z_]*[0-9]*"
        ),
        "y": r"y",
        "latitude": r"x?lat[a-z0-9]*",
        "x": r"x",
        "longitude": r"x?lon[a-z0-9]*",
    },
}

log = logging.getLogger(__name__)

_axis_identifier_error = (
    "Given axis is not valid. Must be an axis number, a dimension "
    "coordinate name, or a standard axis type."
)


@xr.register_dataarray_accessor("metpy")
class MetPyDataArrayAccessor:
    r"""Provide custom attributes and methods on xarray DataArrays for MetPy functionality.

    This accessor provides several convenient attributes and methods through the `.metpy`
    attribute on a DataArray. For example, MetPy can identify the coordinate corresponding
    to a particular axis (given sufficent metadata):

        >>> import xarray as xr
        >>> from metpy.units import units
        >>> temperature = xr.DataArray([[0, 1], [2, 3]] * units.degC, dims=('lat', 'lon'),
        ...                            coords={'lat': [40, 41], 'lon': [-105, -104]})
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

    @property
    def units(self):
        """Return the units of this DataArray as a `pint.Unit`."""
        if isinstance(self._data_array.data, units.Quantity):
            return self._data_array.data.units
        else:
            return units.parse_units(self._data_array.attrs.get("units", "dimensionless"))

    @property
    def magnitude(self):
        """Return the magnitude of the data values of this DataArray (i.e., without units)."""
        if isinstance(self._data_array.data, units.Quantity):
            return self._data_array.data.magnitude
        else:
            return self._data_array.data

    @property
    def unit_array(self):
        """Return the data values of this DataArray as a `pint.Quantity`.

        Notes
        -----
        If not already existing as a `pint.Quantity` or Dask array, the data of this DataArray
        will be loaded into memory by this operation.
        """
        if isinstance(self._data_array.data, units.Quantity):
            return self._data_array.data
        else:
            return units.Quantity(self._data_array.data, self.units)

    def convert_units(self, units):
        """Return new DataArray with values converted to different units.

        Notes
        -----
        Any cached/lazy-loaded data (except that in a Dask array) will be loaded into memory
        by this operation. Do not utilize on moderate- to large-sized remote datasets before
        subsetting!
        """
        return self.quantify().copy(data=self.unit_array.to(units))

    def convert_coordinate_units(self, coord, units):
        """Return new DataArray with coordinate converted to different units.

        Notes
        -----
        Any cached/lazy-loaded coordinate data (except that in a Dask array) will be loaded
        into memory by this operation.
        """
        new_coord_var = self._data_array[coord].copy(
            data=self._data_array[coord].metpy.unit_array.m_as(units)
        )
        new_coord_var.attrs["units"] = str(units)
        return self._data_array.assign_coords(coords={coord: new_coord_var})

    def quantify(self):
        """Return a DataArray with the data converted to a `pint.Quantity`.

        Notes
        -----
        Any cached/lazy-loaded data (except that in a Dask array) will be loaded into memory
        by this operation. Do not utilize on moderate- to large-sized remote datasets before
        subsetting!
        """
        if not isinstance(self._data_array.data, units.Quantity) and np.issubdtype(
            self._data_array.data.dtype, np.number
        ):
            # Only quantify if not already quantified and is quantifiable
            quantified_dataarray = self._data_array.copy(data=self.unit_array)
            if "units" in quantified_dataarray.attrs:
                del quantified_dataarray.attrs["units"]
        else:
            quantified_dataarray = self._data_array
        return quantified_dataarray

    def dequantify(self):
        """Return a DataArray with the data as magnitude and the units as an attribute."""
        if isinstance(self._data_array.data, units.Quantity):
            # Only dequantify if quantified
            dequantified_dataarray = self._data_array.copy(
                data=self._data_array.data.magnitude
            )
            dequantified_dataarray.attrs["units"] = str(self.units)
        else:
            dequantified_dataarray = self._data_array
        return dequantified_dataarray

    @property
    def crs(self):
        """Return the coordinate reference system (CRS) as a CFProjection object."""
        if "crs" in self._data_array.coords:
            return self._data_array.coords["crs"].item()
        raise AttributeError("crs attribute is not available.")

    @property
    def cartopy_crs(self):
        """Return the coordinate reference system (CRS) as a cartopy object."""
        return self.crs.to_cartopy()

    @property
    def cartopy_globe(self):
        """Return the globe belonging to the coordinate reference system (CRS)."""
        return self.crs.cartopy_globe

    @property
    def cartopy_geodetic(self):
        """Return the Geodetic CRS associated with the native CRS globe."""
        return self.crs.cartopy_geodetic

    def _fixup_coordinate_map(self, coord_map):
        """Ensure sure we have coordinate variables in map, not coordinate names."""
        new_coord_map = {}
        for axis in coord_map:
            if coord_map[axis] is not None and not isinstance(coord_map[axis], xr.DataArray):
                new_coord_map[axis] = self._data_array[coord_map[axis]]
            else:
                new_coord_map[axis] = coord_map[axis]

        return new_coord_map

    def assign_coordinates(self, coordinates):
        """Return new DataArray with given coordinates assigned to the given MetPy axis types.

        Parameters
        ----------
        coordinates : dict or None
            Mapping from axis types ('time', 'vertical', 'y', 'latitude', 'x', 'longitude') to
            coordinates of this DataArray. Coordinates can either be specified directly or by
            their name. If ``None``, clears the `_metpy_axis` attribute on all coordinates,
            which will trigger reparsing of all coordinates on next access.

        """
        coord_updates = {}
        if coordinates:
            # Assign the _metpy_axis attributes according to supplied mapping
            coordinates = self._fixup_coordinate_map(coordinates)
            for axis in coordinates:
                if coordinates[axis] is not None:
                    coord_updates[coordinates[axis].name] = coordinates[axis].assign_attrs(
                        _assign_axis(coordinates[axis].attrs.copy(), axis)
                    )
        else:
            # Clear _metpy_axis attribute on all coordinates
            for coord_name, coord_var in self._data_array.coords.items():
                coord_updates[coord_name] = coord_var.copy(deep=False)

                # Some coordinates remained linked in old form under other coordinates. We
                # need to remove from these.
                sub_coords = coord_updates[coord_name].coords
                for sub_coord in sub_coords:
                    coord_updates[coord_name].coords[sub_coord].attrs.pop("_metpy_axis", None)

                # Now we can remove the _metpy_axis attr from the coordinate itself
                coord_updates[coord_name].attrs.pop("_metpy_axis", None)

        return self._data_array.assign_coords(coord_updates)

    def _generate_coordinate_map(self):
        """Generate a coordinate map via CF conventions and other methods."""
        coords = self._data_array.coords.values()
        # Parse all the coordinates, attempting to identify x, longitude, y, latitude,
        # vertical, time
        coord_lists = {
            "time": [],
            "vertical": [],
            "y": [],
            "latitude": [],
            "x": [],
            "longitude": [],
        }
        for coord_var in coords:
            # Identify the coordinate type using check_axis helper
            for axis in coord_lists:
                if check_axis(coord_var, axis):
                    coord_lists[axis].append(coord_var)

        # Fill in x/y with longitude/latitude if x/y not otherwise present
        for geometric, graticule in (("y", "latitude"), ("x", "longitude")):
            if len(coord_lists[geometric]) == 0 and len(coord_lists[graticule]) > 0:
                coord_lists[geometric] = coord_lists[graticule]

        # Filter out multidimensional coordinates where not allowed
        require_1d_coord = ["time", "vertical", "y", "x"]
        for axis in require_1d_coord:
            coord_lists[axis] = [coord for coord in coord_lists[axis] if coord.ndim <= 1]

        # Resolve any coordinate type duplication
        axis_duplicates = [axis for axis in coord_lists if len(coord_lists[axis]) > 1]
        for axis in axis_duplicates:
            self._resolve_axis_duplicates(axis, coord_lists)

        # Collapse the coord_lists to a coord_map
        return {
            axis: (coord_lists[axis][0] if len(coord_lists[axis]) > 0 else None)
            for axis in coord_lists
        }

    def _resolve_axis_duplicates(self, axis, coord_lists):
        """Handle coordinate duplication for an axis type if it arises."""
        # If one and only one of the possible axes is a dimension, use it
        dimension_coords = [
            coord_var for coord_var in coord_lists[axis] if coord_var.name in coord_var.dims
        ]
        if len(dimension_coords) == 1:
            coord_lists[axis] = dimension_coords
            return

        # Ambiguous axis, raise warning and do not parse
        varname = (
            ' "' + self._data_array.name + '"' if self._data_array.name is not None else ""
        )
        warnings.warn(
            "More than one " + axis + " coordinate present for variable" + varname + "."
        )
        coord_lists[axis] = []

    def _metpy_axis_search(self, metpy_axis):
        """Search for cached _metpy_axis attribute on the coordinates, otherwise parse."""
        # Search for coord with proper _metpy_axis
        coords = self._data_array.coords.values()
        for coord_var in coords:
            if metpy_axis in coord_var.attrs.get("_metpy_axis", "").split(","):
                return coord_var

        # Opportunistically parse all coordinates, and assign if not already assigned
        # Note: since this is generally called by way of the coordinate properties, to cache
        # the coordinate parsing results in coord_map on the coordinates means modifying the
        # DataArray in-place (an exception to the usual behavior of MetPy's accessor). This is
        # considered safe because it only effects the "_metpy_axis" attribute on the
        # coordinates, and nothing else.
        coord_map = self._generate_coordinate_map()
        for axis, coord_var in coord_map.items():
            if coord_var is not None and not any(
                axis in coord.attrs.get("_metpy_axis", "").split(",") for coord in coords
            ):

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
                raise AttributeError(axis + " attribute is not available.")
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
        This method is designed for use with multiple coordinates; it returns a generator. To
        access a single coordinate, use the appropriate attribute on the accessor, or use tuple
        unpacking.

        """
        for arg in args:
            yield self._axis(arg)

    @property
    def time(self):
        """Return the time coordinate."""
        return self._axis("time")

    @property
    def vertical(self):
        """Return the vertical coordinate."""
        return self._axis("vertical")

    @property
    def y(self):
        """Return the y coordinate."""
        return self._axis("y")

    @property
    def latitude(self):
        """Return the latitude coordinate (if it exists)."""
        return self._axis("latitude")

    @property
    def x(self):
        """Return the x coordinate."""
        return self._axis("x")

    @property
    def longitude(self):
        """Return the longitude coordinate (if it exists)."""
        return self._axis("longitude")

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

    @property
    def time_deltas(self):
        """Return the time difference of the data in seconds (to microsecond precision)."""
        return (
            np.diff(self._data_array.values).astype("timedelta64[us]").astype("int64")
            / 1e6
            * units.s
        )

    def find_axis_name(self, axis):
        """Return the name of the axis corresponding to the given identifier.

        Parameters
        ----------
        axis : str or int
            Identifier for an axis. Can be an axis number (integer), dimension coordinate
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
            raise ValueError(_axis_identifier_error)

    def find_axis_number(self, axis):
        """Return the dimension number of the axis corresponding to the given identifier.

        Parameters
        ----------
        axis : str or int
            Identifier for an axis. Can be an axis number (integer), dimension coordinate
            name (string) or a standard axis type (string).

        """
        if isinstance(axis, int):
            # If an integer, use it directly
            return axis
        elif axis in self._data_array.dims:
            # Simply index into dims
            return self._data_array.dims.index(axis)
        elif axis in metpy_axes:
            # If not a dimension name itself, but a valid axis type, first determine if this
            # standard axis type is present as a dimension coordinate
            try:
                name = self._axis(axis).name
                return self._data_array.dims.index(name)
            except AttributeError as exc:
                # If x, y, or vertical requested, but not available, attempt to interpret dim
                # names using regular expressions from coordinate parsing to allow for
                # multidimensional lat/lon without y/x dimension coordinates, and basic
                # vertical dim recognition
                if axis in ("vertical", "y", "x"):
                    for i, dim in enumerate(self._data_array.dims):
                        if re.match(
                            coordinate_criteria["regular_expression"][axis], dim.lower()
                        ):
                            return i
                raise exc
            except ValueError:
                # Intercept ValueError when axis type found but not dimension coordinate
                raise AttributeError(
                    f"Requested {axis} dimension coordinate but {axis} "
                    f"coordinate {name} is not a dimension"
                )
        else:
            # Otherwise, not valid
            raise ValueError(_axis_identifier_error)

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
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        indexers = _reassign_quantity_indexer(self._data_array, indexers)
        return self._data_array.sel(indexers, method=method, tolerance=tolerance, drop=drop)

    def assign_crs(self, cf_attributes=None, **kwargs):
        """Assign a CRS to this DataArray based on CF projection attributes.

        Parameters
        ----------
        cf_attributes : dict, optional
            Dictionary of CF projection attributes
        kwargs : optional
            CF projection attributes specified as keyword arguments

        Returns
        -------
        `xarray.DataArray`
            New xarray DataArray with CRS coordinate assigned

        Notes
        -----
        CF projection arguments should be supplied as a dictionary or collection of kwargs,
        but not both.

        """
        return _assign_crs(self._data_array, cf_attributes, kwargs)

    def assign_latitude_longitude(self, force=False):
        """Assign latitude and longitude coordinates derived from y and x coordinates.

        Parameters
        ----------
        force : bool, optional
            If force is true, overwrite latitude and longitude coordinates if they exist,
            otherwise, raise a RuntimeError if such coordinates exist.

        Returns
        -------
        `xarray.DataArray`
            New xarray DataArray with latitude and longtiude auxilary coordinates assigned.

        Notes
        -----
        A valid CRS coordinate must be present. Cartopy is used for the coordinate
        transformations.

        """
        # Check for existing latitude and longitude coords
        if not force and (
            self._metpy_axis_search("latitude") is not None
            or self._metpy_axis_search("longitude")
        ):
            raise RuntimeError(
                "Latitude/longitude coordinate(s) are present. If you wish to "
                "overwrite these, specify force=True."
            )

        # Build new latitude and longitude DataArrays
        latitude, longitude = _build_latitude_longitude(self._data_array)

        # Assign new coordinates, refresh MetPy's parsed axis attribute, and return result
        new_dataarray = self._data_array.assign_coords(latitude=latitude, longitude=longitude)
        return new_dataarray.metpy.assign_coordinates(None)

    def assign_y_x(self, force=False, tolerance=None):
        """Assign y and x dimension coordinates derived from 2D latitude and longitude.

        Parameters
        ----------
        force : bool, optional
            If force is true, overwrite y and x coordinates if they exist, otherwise, raise a
            RuntimeError if such coordinates exist.
        tolerance : `pint.Quantity`
            Maximum range tolerated when collapsing projected y and x coordinates from 2D to
            1D. Defaults to 1 meter.

        Returns
        -------
        `xarray.DataArray`
            New xarray DataArray with y and x dimension coordinates assigned.

        Notes
        -----
        A valid CRS coordinate must be present. Cartopy is used for the coordinate
        transformations.

        """
        # Check for existing latitude and longitude coords
        if not force and (
            self._metpy_axis_search("y") is not None or self._metpy_axis_search("x")
        ):
            raise RuntimeError(
                "y/x coordinate(s) are present. If you wish to overwrite "
                "these, specify force=True."
            )

        # Build new y and x DataArrays
        y, x = _build_y_x(self._data_array, tolerance)

        # Assign new coordinates, refresh MetPy's parsed axis attribute, and return result
        new_dataarray = self._data_array.assign_coords(**{y.name: y, x.name: x})
        return new_dataarray.metpy.assign_coordinates(None)


@xr.register_dataset_accessor("metpy")
class MetPyDatasetAccessor:
    """Provide custom attributes and methods on XArray Datasets for MetPy functionality.

    This accessor provides parsing of CF metadata and unit-/coordinate-type-aware selection.

        >>> import xarray as xr
        >>> from metpy.cbook import get_test_data
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
        from .plots.mapping import CFProjection

        if varname is None:
            # If no varname is given, parse all variables in the dataset
            varname = list(self._dataset.data_vars)

        if np.iterable(varname) and not isinstance(varname, str):
            # If non-string iterable is given, apply recursively across the varnames
            subset = xr.merge(
                [
                    self.parse_cf(single_varname, coordinates=coordinates)
                    for single_varname in varname
                ]
            )
            subset.attrs = self._dataset.attrs
            return subset

        var = self._dataset[varname]

        # Assign coordinates if the coordinates argument is given
        if coordinates is not None:
            var = var.metpy.assign_coordinates(coordinates)

        # Attempt to build the crs coordinate
        crs = None
        if "grid_mapping" in var.attrs:
            # Use given CF grid_mapping
            proj_name = var.attrs["grid_mapping"]
            try:
                proj_var = self._dataset.variables[proj_name]
            except KeyError:
                log.warning(
                    "Could not find variable corresponding to the value of "
                    f"grid_mapping: {proj_name}"
                )
            else:
                crs = CFProjection(proj_var.attrs)

        if crs is None and not check_axis(var, "latitude", "longitude"):
            # This isn't a lat or lon coordinate itself, so determine if we need to fall back
            # to creating a latitude_longitude CRS. We do so if there exists valid coordinates
            # for latitude and longitude, even if they are not the dimension coordinates of
            # the variable.
            def _has_coord(coord_type):
                return any(
                    check_axis(coord_var, coord_type) for coord_var in var.coords.values()
                )

            if _has_coord("latitude") and _has_coord("longitude"):
                crs = CFProjection({"grid_mapping_name": "latitude_longitude"})
                log.warning(
                    "Found valid latitude/longitude coordinates, assuming "
                    "latitude_longitude for projection grid_mapping variable"
                )

        # Rebuild the coordinates of the dataarray, and return quantified DataArray
        var = self._rebuild_coords(var, crs)
        if crs is not None:
            var = var.assign_coords(coords={"crs": crs})
        return var

    def _rebuild_coords(self, var, crs):
        """Clean up the units on the coordinate variables."""
        for coord_name, coord_var in var.coords.items():
            if check_axis(coord_var, "x", "y") and not check_axis(
                coord_var, "longitude", "latitude"
            ):
                try:
                    var = var.metpy.convert_coordinate_units(coord_name, "meters")
                except DimensionalityError:
                    # Radians! Attempt to use perspective point height conversion
                    if crs is not None:
                        height = crs["perspective_point_height"]
                        new_coord_var = coord_var.copy(
                            data=(coord_var.metpy.unit_array * (height * units.meter)).m_as(
                                "meter"
                            )
                        )
                        new_coord_var.attrs["units"] = "meter"
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
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        indexers = _reassign_quantity_indexer(self._dataset, indexers)
        return self._dataset.sel(indexers, method=method, tolerance=tolerance, drop=drop)

    def assign_crs(self, cf_attributes=None, **kwargs):
        """Assign a CRS to this Datatset based on CF projection attributes.

        Parameters
        ----------
        cf_attributes : dict, optional
            Dictionary of CF projection attributes
        kwargs : optional
            CF projection attributes specified as keyword arguments

        Returns
        -------
        `xarray.Dataset`
            New xarray Dataset with CRS coordinate assigned

        Notes
        -----
        CF projection arguments should be supplied as a dictionary or collection of kwargs,
        but not both.

        """
        return _assign_crs(self._dataset, cf_attributes, kwargs)

    def assign_latitude_longitude(self, force=False):
        """Assign latitude and longitude coordinates derived from y and x coordinates.

        Parameters
        ----------
        force : bool, optional
            If force is true, overwrite latitude and longitude coordinates if they exist,
            otherwise, raise a RuntimeError if such coordinates exist.

        Returns
        -------
        `xarray.Dataset`
            New xarray Dataset with latitude and longitude coordinates assigned to all
            variables with y and x coordinates.

        Notes
        -----
        A valid CRS coordinate must be present. Cartopy is used for the coordinate
        transformations.

        """
        # Determine if there is a valid grid prototype from which to compute the coordinates,
        # while also checking for existing lat/lon coords
        grid_prototype = None
        for data_var in self._dataset.data_vars.values():
            if hasattr(data_var.metpy, "y") and hasattr(data_var.metpy, "x"):
                if grid_prototype is None:
                    grid_prototype = data_var
                if not force and (
                    hasattr(data_var.metpy, "latitude") or hasattr(data_var.metpy, "longitude")
                ):
                    raise RuntimeError(
                        "Latitude/longitude coordinate(s) are present. If you "
                        "wish to overwrite these, specify force=True."
                    )

        # Calculate latitude and longitude from grid_prototype, if it exists, and assign
        if grid_prototype is None:
            warnings.warn(
                "No latitude and longitude assigned since horizontal coordinates "
                "were not found"
            )
            return self._dataset
        else:
            latitude, longitude = _build_latitude_longitude(grid_prototype)
            return self._dataset.assign_coords(latitude=latitude, longitude=longitude)

    def assign_y_x(self, force=False, tolerance=None):
        """Assign y and x dimension coordinates derived from 2D latitude and longitude.

        Parameters
        ----------
        force : bool, optional
            If force is true, overwrite y and x coordinates if they exist, otherwise, raise a
            RuntimeError if such coordinates exist.
        tolerance : `pint.Quantity`
            Maximum range tolerated when collapsing projected y and x coordinates from 2D to
            1D. Defaults to 1 meter.

        Returns
        -------
        `xarray.Dataset`
            New xarray Dataset with y and x dimension coordinates assigned to all variables
            with valid latitude and longitude coordinates.

        Notes
        -----
        A valid CRS coordinate must be present. Cartopy is used for the coordinate
        transformations.

        """
        # Determine if there is a valid grid prototype from which to compute the coordinates,
        # while also checking for existing y and x coords
        grid_prototype = None
        for data_var in self._dataset.data_vars.values():
            if hasattr(data_var.metpy, "latitude") and hasattr(data_var.metpy, "longitude"):
                if grid_prototype is None:
                    grid_prototype = data_var
                if not force and (
                    hasattr(data_var.metpy, "y") or hasattr(data_var.metpy, "x")
                ):
                    raise RuntimeError(
                        "y/x coordinate(s) are present. If you wish to "
                        "overwrite these, specify force=True."
                    )

        # Calculate y and x from grid_prototype, if it exists, and assign
        if grid_prototype is None:
            warnings.warn(
                "No y and x coordinates assigned since horizontal coordinates "
                "were not found"
            )
            return self._dataset
        else:
            y, x = _build_y_x(grid_prototype, tolerance)
            return self._dataset.assign_coords(**{y.name: y, x.name: x})

    def update_attribute(self, attribute, mapping):
        """Return new Dataset with specified attribute updated on all Dataset variables.

        Parameters
        ----------
        attribute : str,
            Name of attribute to update
        mapping : dict or callable
            Either a dict, with keys as variable names and values as attribute values to set,
            or a callable, which must accept one positional argument (variable name) and
            arbitrary keyword arguments (all existing variable attributes). If a variable name
            is not present/the callable returns None, the attribute will not be updated.

        Returns
        -------
        `xarray.Dataset`
            New Dataset with attribute updated

        """
        # Make mapping uniform
        if not callable(mapping):
            old_mapping = mapping

            def mapping(varname, **kwargs):
                return old_mapping.get(varname, None)

        # Define mapping function for Dataset.map
        def mapping_func(da):
            new_value = mapping(da.name, **da.attrs)
            if new_value is None:
                return da
            else:
                return da.assign_attrs(**{attribute: new_value})

        # Apply across all variables and coordinates
        return self._dataset.map(mapping_func, keep_attrs=True).assign_coords(
            {
                coord_name: mapping_func(coord_var)
                for coord_name, coord_var in self._dataset.coords.items()
            }
        )

    def quantify(self):
        """Return new dataset with all numeric variables quantified and cached data loaded."""
        return self._dataset.map(lambda da: da.metpy.quantify(), keep_attrs=True)

    def dequantify(self):
        """Return new dataset with variables cast to magnitude and units on attribute."""
        return self._dataset.map(lambda da: da.metpy.dequantify(), keep_attrs=True)


def _assign_axis(attributes, axis):
    """Assign the given axis to the _metpy_axis attribute."""
    existing_axes = attributes.get("_metpy_axis", "").split(",")
    if (axis == "y" and "latitude" in existing_axes) or (
        axis == "latitude" and "y" in existing_axes
    ):
        # Special case for combined y/latitude handling
        attributes["_metpy_axis"] = "y,latitude"
    elif (axis == "x" and "longitude" in existing_axes) or (
        axis == "longitude" and "x" in existing_axes
    ):
        # Special case for combined x/longitude handling
        attributes["_metpy_axis"] = "x,longitude"
    else:
        # Simply add it/overwrite past value
        attributes["_metpy_axis"] = axis
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
        for criterion in ("standard_name", "_CoordinateAxisType", "axis", "positive"):
            if var.attrs.get(criterion, "absent") in coordinate_criteria[criterion].get(
                axis, set()
            ):
                return True

        # Check for units, either by dimensionality or name
        try:
            if axis in coordinate_criteria["units"] and (
                (
                    coordinate_criteria["units"][axis]["match"] == "dimensionality"
                    and (
                        units.get_dimensionality(var.metpy.units)
                        == units.get_dimensionality(
                            coordinate_criteria["units"][axis]["units"]
                        )
                    )
                )
                or (
                    coordinate_criteria["units"][axis]["match"] == "name"
                    and str(var.metpy.units) in coordinate_criteria["units"][axis]["units"]
                )
            ):
                return True
        except UndefinedUnitError:
            pass

        # Check if name matches regular expression (non-CF failsafe)
        if re.match(coordinate_criteria["regular_expression"][axis], var.name.lower()):
            return True

    # If no match has been made, return False (rather than None)
    return False


def _assign_crs(xarray_object, cf_attributes, cf_kwargs):
    from .plots.mapping import CFProjection

    # Handle argument options
    if cf_attributes is not None and len(cf_kwargs) > 0:
        raise ValueError("Cannot specify both attribute dictionary and kwargs.")
    elif cf_attributes is None and len(cf_kwargs) == 0:
        raise ValueError("Must specify either attribute dictionary or kwargs.")
    attrs = cf_attributes if cf_attributes is not None else cf_kwargs

    # Assign crs coordinate to xarray object
    return xarray_object.assign_coords(crs=CFProjection(attrs))


def _build_latitude_longitude(da):
    """Build latitude/longitude coordinates from DataArray's y/x coordinates."""
    y, x = da.metpy.coordinates("y", "x")
    xx, yy = np.meshgrid(x.values, y.values)
    lonlats = da.metpy.cartopy_geodetic.transform_points(da.metpy.cartopy_crs, xx, yy)
    longitude = xr.DataArray(
        lonlats[..., 0],
        dims=(y.name, x.name),
        coords={y.name: y, x.name: x},
        attrs={"units": "degrees_east", "standard_name": "longitude"},
    )
    latitude = xr.DataArray(
        lonlats[..., 1],
        dims=(y.name, x.name),
        coords={y.name: y, x.name: x},
        attrs={"units": "degrees_north", "standard_name": "latitude"},
    )
    return latitude, longitude


def _build_y_x(da, tolerance):
    """Build y/x coordinates from DataArray's latitude/longitude coordinates."""
    # Initial sanity checks
    latitude, longitude = da.metpy.coordinates("latitude", "longitude")
    if latitude.dims != longitude.dims:
        raise ValueError("Latitude and longitude must have same dimensionality")
    elif latitude.ndim != 2:
        raise ValueError(
            "To build 1D y/x coordinates via assign_y_x, latitude/longitude " "must be 2D"
        )

    # Convert to projected y/x
    xxyy = da.metpy.cartopy_crs.transform_points(
        da.metpy.cartopy_geodetic, longitude.values, latitude.values
    )

    # Handle tolerance
    tolerance = 1 if tolerance is None else tolerance.m_as("m")

    # If within tolerance, take median to collapse to 1D
    try:
        y_dim = latitude.metpy.find_axis_number("y")
        x_dim = latitude.metpy.find_axis_number("x")
    except AttributeError:
        warnings.warn(
            "y and x dimensions unable to be identified. Assuming [..., y, x] "
            "dimension order."
        )
        y_dim, x_dim = 0, 1
    if np.all(np.ptp(xxyy[..., 0], axis=y_dim) < tolerance) and np.all(
        np.ptp(xxyy[..., 1], axis=x_dim) < tolerance
    ):
        x = np.median(xxyy[..., 0], axis=y_dim)
        y = np.median(xxyy[..., 1], axis=x_dim)
        x = xr.DataArray(
            x,
            name=latitude.dims[x_dim],
            dims=(latitude.dims[x_dim],),
            coords={latitude.dims[x_dim]: x},
            attrs={"units": "meter", "standard_name": "projection_x_coordinate"},
        )
        y = xr.DataArray(
            y,
            name=latitude.dims[y_dim],
            dims=(latitude.dims[y_dim],),
            coords={latitude.dims[y_dim]: y},
            attrs={"units": "meter", "standard_name": "projection_y_coordinate"},
        )
        return y, x
    else:
        raise ValueError(
            "Projected y and x coordinates cannot be collapsed to 1D within "
            "tolerance. Verify that your latitude and longitude coordinates "
            "correpsond to your CRS coordinate."
        )


def preprocess_and_wrap(broadcast=None, wrap_like=None, match_unit=False, to_magnitude=False):
    """Return decorator to wrap array calculations for type flexibility.

    Assuming you have a calculation that works internally with `pint.Quantity` or
    `numpy.ndarray`, this will wrap the function to be able to handle `xarray.DataArray` and
    `pint.Quantity` as well (assuming appropriate match to one of the input arguments).

    Parameters
    ----------
    broadcast : iterable of str or None
        Iterable of string labels for arguments to broadcast against each other using xarray,
        assuming they are supplied as `xarray.DataArray`. No automatic broadcasting will occur
        with default of None.
    wrap_like : str or array-like or tuple of str or tuple of array-like or None
        Wrap the calculation output following a particular input argument (if str) or data
        object (if array-like). If tuple, will assume output is in the form of a tuple,
        and wrap iteratively according to the str or array-like contained within. If None,
        will not wrap output.
    match_unit : bool
        If true, force the unit of the final output to be that of wrapping object (as
        determined by wrap_like), no matter the original calculation output. Defaults to
        False.
    to_magnitude : bool
        If true, downcast xarray and Pint arguments to their magnitude. If false, downcast
        xarray arguments to Quantity, and do not change other array-like arguments.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = signature(func).bind(*args, **kwargs)

            # Auto-broadcast select xarray arguments, and update bound_args
            if broadcast is not None:
                arg_names_to_broadcast = tuple(
                    arg_name
                    for arg_name in broadcast
                    if arg_name in bound_args.arguments
                    and isinstance(bound_args.arguments[arg_name], (xr.DataArray, xr.Variable))
                )
                broadcasted_args = xr.broadcast(
                    *(bound_args.arguments[arg_name] for arg_name in arg_names_to_broadcast)
                )
                for i, arg_name in enumerate(arg_names_to_broadcast):
                    bound_args.arguments[arg_name] = broadcasted_args[i]

            # Cast all Variables to their data and warn
            # (need to do before match finding, since we don't want to rewrap as Variable)
            for arg_name in bound_args.arguments:
                if isinstance(bound_args.arguments[arg_name], xr.Variable):
                    warnings.warn(
                        f"Argument {arg_name} given as xarray Variable...casting to its data. "
                        "xarray DataArrays are recommended instead."
                    )
                    bound_args.arguments[arg_name] = bound_args.arguments[arg_name].data

            # Obtain proper match if referencing an input
            match = list(wrap_like) if isinstance(wrap_like, tuple) else wrap_like
            if isinstance(wrap_like, str):
                match = bound_args.arguments[wrap_like]
            elif isinstance(wrap_like, tuple):
                for i, arg in enumerate(wrap_like):
                    if isinstance(arg, str):
                        match[i] = bound_args.arguments[arg]

            # Cast all DataArrays to Pint Quantities
            for arg_name in bound_args.arguments:
                if isinstance(bound_args.arguments[arg_name], xr.DataArray):
                    bound_args.arguments[arg_name] = bound_args.arguments[
                        arg_name
                    ].metpy.unit_array

            # Optionally cast all Quantities to their magnitudes
            if to_magnitude:
                for arg_name in bound_args.arguments:
                    if isinstance(bound_args.arguments[arg_name], units.Quantity):
                        bound_args.arguments[arg_name] = bound_args.arguments[arg_name].m

            # Evaluate inner calculation
            result = func(*bound_args.args, **bound_args.kwargs)

            # Wrap output based on match and match_unit
            if match is None:
                return result
            else:
                if match_unit:
                    wrapping = _wrap_output_like_matching_units
                else:
                    wrapping = _wrap_output_like_not_matching_units

                if isinstance(match, list):
                    return tuple(wrapping(*args) for args in zip(result, match))
                else:
                    return wrapping(result, match)

        return wrapper

    return decorator


def _wrap_output_like_matching_units(result, match):
    """Convert result to be like match with matching units for output wrapper."""
    output_xarray = isinstance(match, xr.DataArray)
    match_units = str(match.metpy.units if output_xarray else getattr(match, "units", ""))

    if isinstance(result, xr.DataArray):
        result = result.metpy.convert_units(match_units)
        return result if output_xarray else result.metpy.unit_array
    else:
        result = (
            result.to(match_units)
            if isinstance(result, units.Quantity)
            else units.Quantity(result, match_units)
        )
        return (
            xr.DataArray(result, coords=match.coords, dims=match.dims)
            if output_xarray
            else result
        )


def _wrap_output_like_not_matching_units(result, match):
    """Convert result to be like match without matching units for output wrapper."""
    output_xarray = isinstance(match, xr.DataArray)
    if isinstance(result, xr.DataArray):
        return result if output_xarray else result.metpy.unit_array
    else:
        # Determine if need to upcast to Quantity
        if not isinstance(result, units.Quantity) and (
            isinstance(match, units.Quantity)
            or (output_xarray and isinstance(match.data, units.Quantity))
        ):
            result = units.Quantity(result)
        return (
            xr.DataArray(result, coords=match.coords, dims=match.dims)
            if output_xarray
            else result
        )


def check_matching_coordinates(func):
    """Decorate a function to make sure all given DataArrays have matching coordinates."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data_arrays = [a for a in args if isinstance(a, xr.DataArray)] + [
            a for a in kwargs.values() if isinstance(a, xr.DataArray)
        ]
        if len(data_arrays) > 1:
            first = data_arrays[0]
            for other in data_arrays[1:]:
                if not first.metpy.coordinates_identical(other):
                    raise ValueError("Input DataArray arguments must be on same coordinates.")
        return func(*args, **kwargs)

    return wrapper


def _reassign_quantity_indexer(data, indexers):
    """Reassign a units.Quantity indexer to units of relevant coordinate."""

    def _to_magnitude(val, unit):
        try:
            return val.m_as(unit)
        except AttributeError:
            return val

    # Update indexers keys for axis type -> coord name replacement
    indexers = {
        (
            key
            if not isinstance(data, xr.DataArray) or key in data.dims or key not in metpy_axes
            else next(data.metpy.coordinates(key)).name
        ): indexers[key]
        for key in indexers
    }

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


def grid_deltas_from_dataarray(f, kind="default"):
    """Calculate the horizontal deltas between grid points of a DataArray.

    Calculate the signed delta distance between grid points of a DataArray in the horizontal
    directions, using actual (real distance) or nominal (in projection space) deltas.

    Parameters
    ----------
    f : `xarray.DataArray`
        Parsed DataArray (MetPy's crs coordinate must be available for kind="actual")
    kind : str
        Type of grid delta to calculate. "actual" returns true distances as calculated from
        longitude and latitude via `lat_lon_grid_deltas`. "nominal" returns horizontal
        differences in the data's coordinate space, either in degrees (for lat/lon CRS) or
        meters (for y/x CRS). "default" behaves like "actual" for datasets with a lat/lon CRS
        and like "nominal" for all others. Defaults to "default".

    Returns
    -------
    dx, dy:
        arrays of signed deltas between grid points in the x and y directions with dimensions
        matching those of `f`.

    See Also
    --------
    lat_lon_grid_deltas

    """
    from metpy.calc import lat_lon_grid_deltas

    # Determine behavior
    if kind == "default" and f.metpy.crs["grid_mapping_name"] == "latitude_longitude":
        kind = "actual"
    elif kind == "default":
        kind = "nominal"
    elif kind not in ("actual", "nominal"):
        raise ValueError(
            '"kind" argument must be specified as "default", "actual", or ' '"nominal"'
        )

    if kind == "actual":
        # Get latitude/longitude coordinates and find dim order
        latitude, longitude = xr.broadcast(*f.metpy.coordinates("latitude", "longitude"))
        try:
            y_dim = latitude.metpy.find_axis_number("y")
            x_dim = latitude.metpy.find_axis_number("x")
        except AttributeError:
            warnings.warn(
                "y and x dimensions unable to be identified. Assuming [..., y, x] "
                "dimension order."
            )
            y_dim, x_dim = -2, -1
        # Obtain grid deltas as xarray Variables
        (dx_var, dx_units), (dy_var, dy_units) = (
            (xr.Variable(dims=latitude.dims, data=deltas.magnitude), deltas.units)
            for deltas in lat_lon_grid_deltas(
                longitude,
                latitude,
                x_dim=x_dim,
                y_dim=y_dim,
                initstring=f.metpy.cartopy_crs.proj4_init,
            )
        )
    else:
        # Obtain y/x coordinate differences
        y, x = f.metpy.coordinates("y", "x")
        dx_var = x.diff(x.dims[0]).variable
        dx_units = units(x.attrs.get("units"))
        dy_var = y.diff(y.dims[0]).variable
        dy_units = units(y.attrs.get("units"))

    # Broadcast to input and attach units
    dx = (
        dx_var.set_dims(
            f.dims, shape=[dx_var.sizes[dim] if dim in dx_var.dims else 1 for dim in f.dims]
        ).data
        * dx_units
    )
    dy = (
        dy_var.set_dims(
            f.dims, shape=[dy_var.sizes[dim] if dim in dy_var.dims else 1 for dim in f.dims]
        ).data
        * dy_units
    )

    return dx, dy


def add_grid_arguments_from_xarray(func):
    """Fill in optional arguments like dx/dy from DataArray arguments."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Search for DataArray with valid latitude and longitude coordinates to find grid
        # deltas and any other needed parameter
        dataarray_arguments = [
            value for value in bound_args.arguments.values() if isinstance(value, xr.DataArray)
        ]
        grid_prototype = None
        for da in dataarray_arguments:
            if hasattr(da.metpy, "latitude") and hasattr(da.metpy, "longitude"):
                grid_prototype = da
                break

        # Fill in x_dim/y_dim
        if (
            grid_prototype is not None
            and "x_dim" in bound_args.arguments
            and "y_dim" in bound_args.arguments
        ):
            try:
                bound_args.arguments["x_dim"] = grid_prototype.metpy.find_axis_number("x")
                bound_args.arguments["y_dim"] = grid_prototype.metpy.find_axis_number("y")
            except AttributeError:
                # If axis number not found, fall back to default but warn.
                warnings.warn(
                    "Horizontal dimension numbers not found. Defaulting to "
                    "(..., Y, X) order."
                )

        # Fill in vertical_dim
        if grid_prototype is not None and "vertical_dim" in bound_args.arguments:
            try:
                bound_args.arguments["vertical_dim"] = grid_prototype.metpy.find_axis_number(
                    "vertical"
                )
            except AttributeError:
                # If axis number not found, fall back to default but warn.
                warnings.warn(
                    "Vertical dimension number not found. Defaulting to (..., Z, Y, X) order."
                )

        # Fill in dz
        if (
            grid_prototype is not None
            and "dz" in bound_args.arguments
            and bound_args.arguments["dz"] is None
        ):
            try:
                vertical_coord = grid_prototype.metpy.vertical
                bound_args.arguments["dz"] = np.diff(vertical_coord.metpy.unit_array)
            except AttributeError:
                # Skip, since this only comes up in advection, where dz is optional (may not
                # need vertical at all)
                pass

        # Fill in dx/dy
        if (
            "dx" in bound_args.arguments
            and bound_args.arguments["dx"] is None
            and "dy" in bound_args.arguments
            and bound_args.arguments["dy"] is None
        ):
            if grid_prototype is not None:
                (
                    bound_args.arguments["dx"],
                    bound_args.arguments["dy"],
                ) = grid_deltas_from_dataarray(grid_prototype, kind="actual")
            elif "dz" in bound_args.arguments:
                # Handle advection case, allowing dx/dy to be None but dz to not be None
                if bound_args.arguments["dz"] is None:
                    raise ValueError(
                        "Must provide dx, dy, and/or dz arguments or input DataArray with "
                        "proper coordinates."
                    )
            else:
                raise ValueError(
                    "Must provide dx/dy arguments or input DataArray with "
                    "latitude/longitude coordinates."
                )

        # Fill in latitude
        if "latitude" in bound_args.arguments and bound_args.arguments["latitude"] is None:
            if grid_prototype is not None:
                bound_args.arguments["latitude"] = grid_prototype.metpy.latitude
            else:
                raise ValueError(
                    "Must provide latitude argument or input DataArray with "
                    "latitude/longitude coordinates."
                )

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def add_vertical_dim_from_xarray(func):
    """Fill in optional vertical_dim from DataArray argument."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Search for DataArray in arguments
        dataarray_arguments = [
            value for value in bound_args.arguments.values() if isinstance(value, xr.DataArray)
        ]

        # Fill in vertical_dim
        if len(dataarray_arguments) > 0 and "vertical_dim" in bound_args.arguments:
            try:
                bound_args.arguments["vertical_dim"] = dataarray_arguments[
                    0
                ].metpy.find_axis_number("vertical")
            except AttributeError:
                # If axis number not found, fall back to default but warn.
                warnings.warn(
                    "Vertical dimension number not found. Defaulting to initial dimension."
                )

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


__all__ = ("MetPyDataArrayAccessor", "MetPyDatasetAccessor", "grid_deltas_from_dataarray")
