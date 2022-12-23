# Copyright (c) 2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains a collection of generally useful calculation tools."""
import contextlib
import functools
from inspect import Parameter, signature
from operator import itemgetter
import warnings

import numpy as np
from numpy.core.numeric import normalize_axis_index
import numpy.ma as ma
from pyproj import CRS, Geod, Proj
from scipy.spatial import cKDTree
import xarray as xr

from ..cbook import broadcast_indices
from ..interpolate import interpolate_1d, log_interpolate_1d
from ..package_tools import Exporter
from ..units import check_units, concatenate, units
from ..xarray import check_axis, grid_deltas_from_dataarray, preprocess_and_wrap

exporter = Exporter(globals())

UND = 'UND'
UND_ANGLE = -999.
DIR_STRS = (
    'N', 'NNE', 'NE', 'ENE',
    'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW',
    'W', 'WNW', 'NW', 'NNW',
    UND
)  # note the order matters!

MAX_DEGREE_ANGLE = units.Quantity(360, 'degree')
BASE_DEGREE_MULTIPLIER = units.Quantity(22.5, 'degree')

DIR_DICT = {dir_str: i * BASE_DEGREE_MULTIPLIER for i, dir_str in enumerate(DIR_STRS)}
DIR_DICT[UND] = np.nan


@exporter.export
def resample_nn_1d(a, centers):
    """Return one-dimensional nearest-neighbor indexes based on user-specified centers.

    Parameters
    ----------
    a : array-like
        1-dimensional array of numeric values from which to extract indexes of
        nearest-neighbors
    centers : array-like
        1-dimensional array of numeric values representing a subset of values to approximate

    Returns
    -------
        A list of indexes (in type given by `array.argmin()`) representing values closest to
        given array values.

    """
    ix = []
    for center in centers:
        index = (np.abs(a - center)).argmin()
        if index not in ix:
            ix.append(index)
    return ix


@exporter.export
def nearest_intersection_idx(a, b):
    """Determine the index of the point just before two lines with common x values.

    Parameters
    ----------
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2

    Returns
    -------
        An array of indexes representing the index of the values
        just before the intersection(s) of the two lines.

    """
    # Difference in the two y-value sets
    difference = a - b

    # Determine the point just before the intersection of the lines
    # Will return multiple points for multiple intersections
    sign_change_idx, = np.nonzero(np.diff(np.sign(difference)))

    return sign_change_idx


@exporter.export
@preprocess_and_wrap()
@units.wraps(('=A', '=B'), ('=A', '=B', '=B', None, None))
def find_intersections(x, a, b, direction='all', log_x=False):
    """Calculate the best estimate of intersection.

    Calculates the best estimates of the intersection of two y-value
    data sets that share a common x-value set.

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric x-values
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2
    direction : str, optional
        specifies direction of crossing. 'all', 'increasing' (a becoming greater than b),
        or 'decreasing' (b becoming greater than a). Defaults to 'all'.
    log_x : bool, optional
        Use logarithmic interpolation along the `x` axis (i.e. for finding intersections
        in pressure coordinates). Default is False.

    Returns
    -------
        A tuple (x, y) of array-like with the x and y coordinates of the
        intersections of the lines.

    Notes
    -----
    This function implicitly converts `xarray.DataArray` to `pint.Quantity`, with the results
    given as `pint.Quantity`.

    """
    # Change x to logarithmic if log_x=True
    if log_x is True:
        x = np.log(x)

    # Find the index of the points just before the intersection(s)
    nearest_idx = nearest_intersection_idx(a, b)
    next_idx = nearest_idx + 1

    # Determine the sign of the change
    sign_change = np.sign(a[next_idx] - b[next_idx])

    # x-values around each intersection
    _, x0 = _next_non_masked_element(x, nearest_idx)
    _, x1 = _next_non_masked_element(x, next_idx)

    # y-values around each intersection for the first line
    _, a0 = _next_non_masked_element(a, nearest_idx)
    _, a1 = _next_non_masked_element(a, next_idx)

    # y-values around each intersection for the second line
    _, b0 = _next_non_masked_element(b, nearest_idx)
    _, b1 = _next_non_masked_element(b, next_idx)

    # Calculate the x-intersection. This comes from finding the equations of the two lines,
    # one through (x0, a0) and (x1, a1) and the other through (x0, b0) and (x1, b1),
    # finding their intersection, and reducing with a bunch of algebra.
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1
    intersect_x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)

    # Calculate the y-intersection of the lines. Just plug the x above into the equation
    # for the line through the a points. One could solve for y like x above, but this
    # causes weirder unit behavior and seems a little less good numerically.
    intersect_y = ((intersect_x - x0) / (x1 - x0)) * (a1 - a0) + a0

    # If there's no intersections, return
    if len(intersect_x) == 0:
        return intersect_x, intersect_y

    # Return x to linear if log_x is True
    if log_x is True:
        intersect_x = np.exp(intersect_x)

    # Check for duplicates
    duplicate_mask = (np.ediff1d(intersect_x, to_end=1) != 0)

    # Make a mask based on the direction of sign change desired
    if direction == 'increasing':
        mask = sign_change > 0
    elif direction == 'decreasing':
        mask = sign_change < 0
    elif direction == 'all':
        return intersect_x[duplicate_mask], intersect_y[duplicate_mask]
    else:
        raise ValueError(f'Unknown option for direction: {direction}')

    return intersect_x[mask & duplicate_mask], intersect_y[mask & duplicate_mask]


def _next_non_masked_element(a, idx):
    """Return the next non masked element of a masked array.

    If an array is masked, return the next non-masked element (if the given index is masked).
    If no other unmasked points are after the given masked point, returns none.

    Parameters
    ----------
    a : array-like
        1-dimensional array of numeric values
    idx : integer
        Index of requested element

    Returns
    -------
        Index of next non-masked element and next non-masked element

    """
    try:
        next_idx = idx + a[idx:].mask.argmin()
        if ma.is_masked(a[next_idx]):
            return None, None
        else:
            return next_idx, a[next_idx]
    except (AttributeError, TypeError, IndexError):
        return idx, a[idx]


def _delete_masked_points(*arrs):
    """Delete masked points from arrays.

    Takes arrays and removes masked points to help with calculations and plotting.

    Parameters
    ----------
    arrs : one or more array-like
        Source arrays

    Returns
    -------
    arrs : one or more array-like
        Arrays with masked elements removed

    """
    if any(hasattr(a, 'mask') for a in arrs):
        keep = ~functools.reduce(np.logical_or, (np.ma.getmaskarray(a) for a in arrs))
        return tuple(a[keep] for a in arrs)
    else:
        return arrs


@exporter.export
@preprocess_and_wrap()
def reduce_point_density(points, radius, priority=None):
    r"""Return a mask to reduce the density of points in irregularly-spaced data.

    This function is used to down-sample a collection of scattered points (e.g. surface
    data), returning a mask that can be used to select the points from one or more arrays
    (e.g. arrays of temperature and dew point). The points selected can be controlled by
    providing an array of ``priority`` values (e.g. rainfall totals to ensure that
    stations with higher precipitation remain in the mask). The points and radius can be
    specified with units. If none are provided, meters are assumed.

    Points with at least one non-finite (i.e. NaN or Inf) value are ignored and returned with
    a value of ``False`` (meaning don't keep).

    Parameters
    ----------
    points : (N, M) array-like
        N locations of the points in M dimensional space
    radius : `pint.Quantity` or float
        Minimum radius allowed between points. If units are not provided, meters is assumed.
    priority : (N, M) array-like, optional
        If given, this should have the same shape as ``points``; these values will
        be used to control selection priority for points.

    Returns
    -------
        (N,) array-like of boolean values indicating whether points should be kept. This
        can be used directly to index numpy arrays to return only the desired points.

    Examples
    --------
    >>> metpy.calc.reduce_point_density(np.array([1, 2, 3]), 1.)
    array([ True, False, True])
    >>> metpy.calc.reduce_point_density(np.array([1, 2, 3]), 1.,
    ... priority=np.array([0.1, 0.9, 0.3]))
    array([False, True, False])

    """
    # Handle input with units. Assume meters if units are not specified
    if hasattr(radius, 'units'):
        radius = radius.to('m').m

    if hasattr(points, 'units'):
        points = points.to('m').m

    # Handle 1D input
    if points.ndim < 2:
        points = points.reshape(-1, 1)

    # Identify good points--finite values (e.g. not NaN or inf). Set bad 0, but we're going to
    # ignore anyway. It's easier for managing indices to keep the original points in the group.
    good_vals = np.isfinite(points)
    points = np.where(good_vals, points, 0)

    # Make a kd-tree to speed searching of data.
    tree = cKDTree(points)

    # Need to use sorted indices rather than sorting the position
    # so that the keep mask matches *original* order.
    if priority is not None:
        # Need to sort the locations in decreasing priority.
        sorted_indices = np.argsort(priority)[::-1]
    else:
        # Take advantage of iterator nature of range here to avoid making big lists
        sorted_indices = range(len(points))

    # Keep all good points initially
    keep = np.logical_and.reduce(good_vals, axis=-1)

    # Loop over all the potential points
    for ind in sorted_indices:
        # Only proceed if we haven't already excluded this point
        if keep[ind]:
            # Find the neighbors and eliminate them
            neighbors = tree.query_ball_point(points[ind], radius)
            keep[neighbors] = False

            # We just removed ourselves, so undo that
            keep[ind] = True

    return keep


def _get_bound_pressure_height(pressure, bound, height=None, interpolate=True):
    """Calculate the bounding pressure and height in a layer.

    Given pressure, optional heights and a bound, return either the closest pressure/height
    or interpolated pressure/height. If no heights are provided, a standard atmosphere
    ([NOAA1976]_) is assumed.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressures
    bound : `pint.Quantity`
        Bound to retrieve (in pressure or height)
    height : `pint.Quantity`, optional
        Atmospheric heights associated with the pressure levels. Defaults to using
        heights calculated from ``pressure`` assuming a standard atmosphere.
    interpolate : boolean, optional
        Interpolate the bound or return the nearest. Defaults to True.

    Returns
    -------
    `pint.Quantity`
        The bound pressure and height

    """
    # avoid circular import if basic.py ever imports something from tools.py
    from .basic import height_to_pressure_std, pressure_to_height_std

    # Make sure pressure is monotonically decreasing
    sort_inds = np.argsort(pressure)[::-1]
    pressure = pressure[sort_inds]
    if height is not None:
        height = height[sort_inds]

    # Bound is given in pressure
    if bound.check('[length]**-1 * [mass] * [time]**-2'):
        # If the bound is in the pressure data, we know the pressure bound exactly
        if bound in pressure:
            # By making sure this is at least a 1D array we avoid the behavior in numpy
            # (at least up to 1.19.4) that float32 scalar * Python float -> float64, which
            # can wreak havok with floating point comparisons.
            bound_pressure = np.atleast_1d(bound)
            # If we have heights, we know the exact height value, otherwise return standard
            # atmosphere height for the pressure
            if height is not None:
                bound_height = height[pressure == bound_pressure]
            else:
                bound_height = pressure_to_height_std(bound_pressure)
        # If bound is not in the data, return the nearest or interpolated values
        else:
            if interpolate:
                bound_pressure = bound  # Use the user specified bound
                if height is not None:  # Interpolate heights from the height data
                    bound_height = log_interpolate_1d(bound_pressure, pressure, height)
                else:  # If not heights given, use the standard atmosphere
                    bound_height = pressure_to_height_std(bound_pressure)
            else:  # No interpolation, find the closest values
                idx = (np.abs(pressure - bound)).argmin()
                bound_pressure = pressure[idx]
                if height is not None:
                    bound_height = height[idx]
                else:
                    bound_height = pressure_to_height_std(bound_pressure)

    # Bound is given in height
    elif bound.check('[length]'):
        # If there is height data, see if we have the bound or need to interpolate/find nearest
        if height is not None:
            if bound in height:  # Bound is in the height data
                bound_height = bound
                bound_pressure = pressure[height == bound]
            else:  # Bound is not in the data
                if interpolate:
                    bound_height = bound

                    # Need to cast back to the input type since interp (up to at least numpy
                    # 1.13 always returns float64. This can cause upstream users problems,
                    # resulting in something like np.append() to upcast.
                    bound_pressure = np.interp(np.atleast_1d(bound),
                                               height, pressure).astype(np.result_type(bound))
                else:
                    idx = (np.abs(height - bound)).argmin()
                    bound_pressure = pressure[idx]
                    bound_height = height[idx]
        else:  # Don't have heights, so assume a standard atmosphere
            bound_height = bound
            bound_pressure = height_to_pressure_std(bound)
            # If interpolation is on, this is all we need, if not, we need to go back and
            # find the pressure closest to this and refigure the bounds
            if not interpolate:
                idx = (np.abs(pressure - bound_pressure)).argmin()
                bound_pressure = pressure[idx]
                bound_height = pressure_to_height_std(bound_pressure)

    # Bound has invalid units
    else:
        raise ValueError('Bound must be specified in units of length or pressure.')

    # If the bound is out of the range of the data, we shouldn't extrapolate
    if not (_greater_or_close(bound_pressure, np.nanmin(pressure))
            and _less_or_close(bound_pressure, np.nanmax(pressure))):
        raise ValueError('Specified bound is outside pressure range.')
    if height is not None and not (_less_or_close(bound_height, np.nanmax(height))
                                   and _greater_or_close(bound_height, np.nanmin(height))):
        raise ValueError('Specified bound is outside height range.')

    return bound_pressure, bound_height


@exporter.export
@preprocess_and_wrap()
@check_units('[length]')
def get_layer_heights(height, depth, *args, bottom=None, interpolate=True, with_agl=False):
    """Return an atmospheric layer from upper air data with the requested bottom and depth.

    This function will subset an upper air dataset to contain only the specified layer using
    the height only.

    Parameters
    ----------
    height : array-like
        Atmospheric height
    depth : `pint.Quantity`
        Thickness of the layer
    args : array-like
        Atmospheric variable(s) measured at the given pressures
    bottom : `pint.Quantity`, optional
        Bottom of the layer
    interpolate : bool, optional
        Interpolate the top and bottom points if they are not in the given data. Defaults
        to True.
    with_agl : bool, optional
        Returns the height as above ground level by subtracting the minimum height in the
        provided height. Defaults to False.

    Returns
    -------
    `pint.Quantity, pint.Quantity`
        Height and data variables of the layer

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Also, this will return Pint Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height``

    """
    # Make sure pressure and datavars are the same length
    for datavar in args:
        if len(height) != len(datavar):
            raise ValueError('Height and data variables must have the same length.')

    # If we want things in AGL, subtract the minimum height from all height values
    if with_agl:
        sfc_height = np.min(height)
        height = height - sfc_height

    # If the bottom is not specified, make it the surface
    if bottom is None:
        bottom = height[0]

    # Make heights and arguments base units
    height = height.to_base_units()
    bottom = bottom.to_base_units()

    # Calculate the top of the layer
    top = bottom + depth

    ret = []  # returned data variables in layer

    # Ensure heights are sorted in ascending order
    sort_inds = np.argsort(height)
    height = height[sort_inds]

    # Mask based on top and bottom
    inds = _greater_or_close(height, bottom) & _less_or_close(height, top)
    heights_interp = height[inds]

    # Interpolate heights at bounds if necessary and sort
    if interpolate:
        # If we don't have the bottom or top requested, append them
        if top not in heights_interp:
            heights_interp = units.Quantity(np.sort(np.append(heights_interp.m, top.m)),
                                            height.units)
        if bottom not in heights_interp:
            heights_interp = units.Quantity(np.sort(np.append(heights_interp.m, bottom.m)),
                                            height.units)

    ret.append(heights_interp)

    for datavar in args:
        # Ensure that things are sorted in ascending order
        datavar = datavar[sort_inds]

        if interpolate:
            # Interpolate for the possibly missing bottom/top values
            datavar_interp = interpolate_1d(heights_interp, height, datavar)
            datavar = datavar_interp
        else:
            datavar = datavar[inds]

        ret.append(datavar)
    return ret


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]')
def get_layer(pressure, *args, height=None, bottom=None, depth=None, interpolate=True):
    r"""Return an atmospheric layer from upper air data with the requested bottom and depth.

    This function will subset an upper air dataset to contain only the specified layer. The
    bottom of the layer can be specified with a pressure or height above the surface
    pressure. The bottom defaults to the surface pressure. The depth of the layer can be
    specified in terms of pressure or height above the bottom of the layer. If the top and
    bottom of the layer are not in the data, they are interpolated by default.

    Parameters
    ----------
    pressure : array-like
        Atmospheric pressure profile
    args : array-like
        Atmospheric variable(s) measured at the given pressures
    height: array-like, optional
        Atmospheric heights corresponding to the given pressures. Defaults to using
        heights calculated from ``pressure`` assuming a standard atmosphere [NOAA1976]_.
    bottom : `pint.Quantity`, optional
        Bottom of the layer as a pressure or height above the surface pressure. Defaults
        to the highest pressure or lowest height given.
    depth : `pint.Quantity`, optional
        Thickness of the layer as a pressure or height above the bottom of the layer.
        Defaults to 100 hPa.
    interpolate : bool, optional
        Interpolate the top and bottom points if they are not in the given data. Defaults
        to True.

    Returns
    -------
    `pint.Quantity, pint.Quantity`
        The pressure and data variables of the layer

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Also, this will return Pint Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height``

    """
    # If we get the depth kwarg, but it's None, set it to the default as well
    if depth is None:
        depth = units.Quantity(100, 'hPa')

    # Make sure pressure and datavars are the same length
    for datavar in args:
        if len(pressure) != len(datavar):
            raise ValueError('Pressure and data variables must have the same length.')

    # If the bottom is not specified, make it the surface pressure
    if bottom is None:
        bottom = np.nanmax(pressure)

    bottom_pressure, bottom_height = _get_bound_pressure_height(pressure, bottom,
                                                                height=height,
                                                                interpolate=interpolate)

    # Calculate the top in whatever units depth is in
    if depth.check('[length]**-1 * [mass] * [time]**-2'):
        top = bottom_pressure - depth
    elif depth.check('[length]'):
        top = bottom_height + depth
    else:
        raise ValueError('Depth must be specified in units of length or pressure')

    top_pressure, _ = _get_bound_pressure_height(pressure, top, height=height,
                                                 interpolate=interpolate)

    ret = []  # returned data variables in layer

    # Ensure pressures are sorted in ascending order
    sort_inds = np.argsort(pressure)
    pressure = pressure[sort_inds]

    # Mask based on top and bottom pressure
    inds = (_less_or_close(pressure, bottom_pressure)
            & _greater_or_close(pressure, top_pressure))
    p_interp = pressure[inds]

    # Interpolate pressures at bounds if necessary and sort
    if interpolate:
        # If we don't have the bottom or top requested, append them
        if not np.any(np.isclose(top_pressure, p_interp)):
            p_interp = units.Quantity(np.sort(np.append(p_interp.m, top_pressure.m)),
                                      pressure.units)
        if not np.any(np.isclose(bottom_pressure, p_interp)):
            p_interp = units.Quantity(np.sort(np.append(p_interp.m, bottom_pressure.m)),
                                      pressure.units)

    ret.append(p_interp[::-1])

    for datavar in args:
        # Ensure that things are sorted in ascending order
        datavar = datavar[sort_inds]

        if interpolate:
            # Interpolate for the possibly missing bottom/top values
            datavar_interp = log_interpolate_1d(p_interp, pressure, datavar)
            datavar = datavar_interp
        else:
            datavar = datavar[inds]

        ret.append(datavar[::-1])
    return ret


@exporter.export
@preprocess_and_wrap()
def find_bounding_indices(arr, values, axis, from_below=True):
    """Find the indices surrounding the values within arr along axis.

    Returns a set of above, below, good. Above and below are lists of arrays of indices.
    These lists are formulated such that they can be used directly to index into a numpy
    array and get the expected results (no extra slices or ellipsis necessary). `good` is
    a boolean array indicating the "columns" that actually had values to bound the desired
    value(s).

    Parameters
    ----------
    arr : array-like
        Array to search for values

    values: array-like
        One or more values to search for in `arr`

    axis : int
        Dimension of `arr` along which to search

    from_below : bool, optional
        Whether to search from "below" (i.e. low indices to high indices). If `False`,
        the search will instead proceed from high indices to low indices. Defaults to `True`.

    Returns
    -------
    above : list of arrays
        List of broadcasted indices to the location above the desired value

    below : list of arrays
        List of broadcasted indices to the location below the desired value

    good : array
        Boolean array indicating where the search found proper bounds for the desired value

    """
    # The shape of generated indices is the same as the input, but with the axis of interest
    # replaced by the number of values to search for.
    indices_shape = list(arr.shape)
    indices_shape[axis] = len(values)

    # Storage for the found indices and the mask for good locations
    indices = np.empty(indices_shape, dtype=int)
    good = np.empty(indices_shape, dtype=bool)

    # Used to put the output in the proper location
    take = make_take(arr.ndim, axis)

    # Loop over all of the values and for each, see where the value would be found from a
    # linear search
    for level_index, value in enumerate(values):
        # Look for changes in the value of the test for <= value in consecutive points
        # Taking abs() because we only care if there is a flip, not which direction.
        switches = np.abs(np.diff((arr <= value).astype(int), axis=axis))

        # Good points are those where it's not just 0's along the whole axis
        good_search = np.any(switches, axis=axis)

        if from_below:
            # Look for the first switch; need to add 1 to the index since argmax is giving the
            # index within the difference array, which is one smaller.
            index = switches.argmax(axis=axis) + 1
        else:
            # Generate a list of slices to reverse the axis of interest so that searching from
            # 0 to N is starting at the "top" of the axis.
            arr_slice = [slice(None)] * arr.ndim
            arr_slice[axis] = slice(None, None, -1)

            # Same as above, but we use the slice to come from the end; then adjust those
            # indices to measure from the front.
            index = arr.shape[axis] - 1 - switches[tuple(arr_slice)].argmax(axis=axis)

        # Set all indices where the results are not good to 0
        index[~good_search] = 0

        # Put the results in the proper slice
        store_slice = take(level_index)
        indices[store_slice] = index
        good[store_slice] = good_search

    # Create index values for broadcasting arrays
    above = broadcast_indices(indices, arr.shape, axis)
    below = broadcast_indices(indices - 1, arr.shape, axis)

    return above, below, good


def _greater_or_close(a, value, **kwargs):
    r"""Compare values for greater or close to boolean masks.

    Returns a boolean mask for values greater than or equal to a target within a specified
    absolute or relative tolerance (as in :func:`numpy.isclose`).

    Parameters
    ----------
    a : array-like
        Array of values to be compared
    value : float
        Comparison value

    Returns
    -------
    array-like
        Boolean array where values are greater than or nearly equal to value.

    """
    return (a > value) | np.isclose(a, value, **kwargs)


def _less_or_close(a, value, **kwargs):
    r"""Compare values for less or close to boolean masks.

    Returns a boolean mask for values less than or equal to a target within a specified
    absolute or relative tolerance (as in :func:`numpy.isclose`).

    Parameters
    ----------
    a : array-like
        Array of values to be compared
    value : float
        Comparison value

    Returns
    -------
    array-like
        Boolean array where values are less than or nearly equal to value

    """
    return (a < value) | np.isclose(a, value, **kwargs)


def make_take(ndims, slice_dim):
    """Generate a take function to index in a particular dimension."""
    def take(indexer):
        return tuple(indexer if slice_dim % ndims == i else slice(None)  # noqa: S001
                     for i in range(ndims))
    return take


@exporter.export
@preprocess_and_wrap()
def lat_lon_grid_deltas(longitude, latitude, x_dim=-1, y_dim=-2, geod=None):
    r"""Calculate the actual delta between grid points that are in latitude/longitude format.

    Parameters
    ----------
    longitude : array-like
        Array of longitudes defining the grid. If not a `pint.Quantity`, assumed to be in
        degrees.

    latitude : array-like
        Array of latitudes defining the grid. If not a `pint.Quantity`, assumed to be in
        degrees.

    x_dim: int
        axis number for the x dimension, defaults to -1.

    y_dim : int
        axis number for the y dimension, defaults to -2.

    geod : `pyproj.Geod` or ``None``
        PyProj Geod to use for forward azimuth and distance calculations. If ``None``, use a
        default spherical ellipsoid.

    Returns
    -------
    dx, dy:
        At least two dimensional arrays of signed deltas between grid points in the x and y
        direction

    Notes
    -----
    Accepts 1D, 2D, or higher arrays for latitude and longitude
    Assumes [..., Y, X] dimension order for input and output, unless keyword arguments `y_dim`
    and `x_dim` are otherwise specified.

    This function will only return `pint.Quantity` arrays (not `xarray.DataArray` or another
    array-like type). It will also "densify" your data if using Dask or lazy-loading.

    .. versionchanged:: 1.0
       Changed signature from ``(longitude, latitude, **kwargs)``

    """
    # Inputs must be the same number of dimensions
    if latitude.ndim != longitude.ndim:
        raise ValueError('Latitude and longitude must have the same number of dimensions.')

    # If we were given 1D arrays, make a mesh grid
    if latitude.ndim < 2:
        longitude, latitude = np.meshgrid(longitude, latitude)

    # pyproj requires ndarrays, not Quantities
    try:
        longitude = np.asarray(longitude.m_as('degrees'))
        latitude = np.asarray(latitude.m_as('degrees'))
    except AttributeError:
        longitude = np.asarray(longitude)
        latitude = np.asarray(latitude)

    # Determine dimension order for offset slicing
    take_y = make_take(latitude.ndim, y_dim)
    take_x = make_take(latitude.ndim, x_dim)

    g = Geod(ellps='sphere') if geod is None else geod
    forward_az, _, dy = g.inv(longitude[take_y(slice(None, -1))],
                              latitude[take_y(slice(None, -1))],
                              longitude[take_y(slice(1, None))],
                              latitude[take_y(slice(1, None))])
    dy[(forward_az < -90.) | (forward_az > 90.)] *= -1

    forward_az, _, dx = g.inv(longitude[take_x(slice(None, -1))],
                              latitude[take_x(slice(None, -1))],
                              longitude[take_x(slice(1, None))],
                              latitude[take_x(slice(1, None))])
    dx[(forward_az < 0.) | (forward_az > 180.)] *= -1

    return units.Quantity(dx, 'meter'), units.Quantity(dy, 'meter')


@preprocess_and_wrap()
def nominal_lat_lon_grid_deltas(longitude, latitude, geod=None):
    """Calculate the nominal deltas along axes of a latitude/longitude grid."""
    if geod is None:
        geod = CRS('+proj=latlon').get_geod()

    # This allows working with coordinates that have been manually broadcast
    longitude = longitude.squeeze()
    latitude = latitude.squeeze()

    if longitude.ndim != 1 or latitude.ndim != 1:
        raise ValueError(
            'Cannot calculate nominal grid spacing from longitude and latitude arguments '
            'that are not one dimensional.'
        )

    dx = units.Quantity(geod.a * np.diff(longitude).m_as('radian'), 'meter')
    lat = latitude.m_as('degree')
    lon_meridian_diff = np.zeros(len(lat) - 1, dtype=lat.dtype)
    forward_az, _, dy = geod.inv(lon_meridian_diff, lat[:-1], lon_meridian_diff, lat[1:],
                                 radians=False)
    dy[(forward_az < -90.) | (forward_az > 90.)] *= -1
    dy = units.Quantity(dy, 'meter')

    return dx, dy


@exporter.export
@preprocess_and_wrap()
def azimuth_range_to_lat_lon(azimuths, ranges, center_lon, center_lat, geod=None):
    """Convert azimuth and range locations in a polar coordinate system to lat/lon coordinates.

    Pole refers to the origin of the coordinate system.

    Parameters
    ----------
    azimuths : array-like
        array of azimuths defining the grid. If not a `pint.Quantity`,
        assumed to be in degrees.
    ranges : array-like
        array of range distances from the pole. Typically in meters.
    center_lat : float
        The latitude of the pole in decimal degrees
    center_lon : float
        The longitude of the pole in decimal degrees
    geod : `pyproj.Geod` or ``None``
        PyProj Geod to use for forward azimuth and distance calculations. If ``None``, use a
        default spherical ellipsoid.

    Returns
    -------
    lon, lat : 2D arrays of longitudes and latitudes corresponding to original locations

    Notes
    -----
    Credit to Brian Blaylock for the original implementation.

    """
    g = Geod(ellps='sphere') if geod is None else geod
    try:  # convert range units to meters
        ranges = ranges.m_as('meters')
    except AttributeError:  # no units associated
        warnings.warn('Range values are not a Pint-Quantity, assuming values are in meters.')
    try:  # convert azimuth units to degrees
        azimuths = azimuths.m_as('degrees')
    except AttributeError:  # no units associated
        warnings.warn(
            'Azimuth values are not a Pint-Quantity, assuming values are in degrees.'
        )
    rng2d, az2d = np.meshgrid(ranges, azimuths)
    lats = np.full(az2d.shape, center_lat)
    lons = np.full(az2d.shape, center_lon)
    lon, lat, _ = g.fwd(lons, lats, az2d, rng2d)

    return lon, lat


def xarray_derivative_wrap(func):
    """Decorate the derivative functions to make them work nicely with DataArrays.

    This will automatically determine if the coordinates can be pulled directly from the
    DataArray, or if a call to lat_lon_grid_deltas is needed.
    """
    @functools.wraps(func)
    def wrapper(f, **kwargs):
        if 'x' in kwargs or 'delta' in kwargs:
            # Use the usual DataArray to pint.Quantity preprocessing wrapper
            return preprocess_and_wrap()(func)(f, **kwargs)
        elif isinstance(f, xr.DataArray):
            # Get axis argument, defaulting to first dimension
            axis = f.metpy.find_axis_name(kwargs.get('axis', 0))

            # Initialize new kwargs with the axis number
            new_kwargs = {'axis': f.get_axis_num(axis)}

            if check_axis(f[axis], 'time'):
                # Time coordinate, need to get time deltas
                new_kwargs['delta'] = f[axis].metpy.time_deltas
            elif check_axis(f[axis], 'longitude'):
                # Longitude coordinate, need to get grid deltas
                new_kwargs['delta'], _ = grid_deltas_from_dataarray(f)
            elif check_axis(f[axis], 'latitude'):
                # Latitude coordinate, need to get grid deltas
                _, new_kwargs['delta'] = grid_deltas_from_dataarray(f)
            else:
                # General coordinate, use as is
                new_kwargs['x'] = f[axis].metpy.unit_array

            # Calculate and return result as a DataArray
            result = func(f.metpy.unit_array, **new_kwargs)
            return xr.DataArray(result, coords=f.coords, dims=f.dims)
        else:
            # Error
            raise ValueError('Must specify either "x" or "delta" for value positions when "f" '
                             'is not a DataArray.')
    return wrapper


def _add_grid_params_to_docstring(docstring: str, orig_includes: dict) -> str:
    """Add documentation for some dynamically added grid parameters to the docstring."""
    other_params = docstring.find('Other Parameters')
    blank = docstring.find('\n\n', other_params)

    entries = {
        'longitude': """
    longitude : `pint.Quantity`, optional
        Longitude of data. Optional if `xarray.DataArray` with latitude/longitude coordinates
        used as input. Also optional if parallel_scale and meridional_scale are given. If
        otherwise omitted, calculation will be carried out on a Cartesian, rather than
        geospatial, grid. Keyword-only argument.""",
        'latitude': """
    latitude : `pint.Quantity`, optional
        Latitude of data. Optional if `xarray.DataArray` with latitude/longitude coordinates
        used as input. Also optional if parallel_scale and meridional_scale are given. If
        otherwise omitted, calculation will be carried out on a Cartesian, rather than
        geospatial, grid. Keyword-only argument.""",
        'crs': """
    crs : `pyproj.crs.CRS`, optional
        Coordinate Reference System of data. Optional if `xarray.DataArray` with MetPy CRS
        used as input. Also optional if parallel_scale and meridional_scale are given. If
        otherwise omitted, calculation will be carried out on a Cartesian, rather than
        geospatial, grid. Keyword-only argument."""
    }

    return ''.join([docstring[:blank],
                    *(entries[p] for p, included in orig_includes.items() if not included),
                    docstring[blank:]])


def parse_grid_arguments(func):
    """Parse arguments to functions involving derivatives on a grid."""
    from ..xarray import dataarray_arguments

    # Dynamically add new parameters for lat, lon, and crs to the function signature
    # which is used to handle arguments inside the wrapper--but only if they're not in the
    # original signature
    sig = signature(func)
    orig_func_uses = {param: param in sig.parameters
                      for param in ('latitude', 'longitude', 'crs')}
    newsig = sig.replace(parameters=[*sig.parameters.values(),
                                     *(Parameter(name, Parameter.KEYWORD_ONLY, default=None)
                                       for name, needed in orig_func_uses.items()
                                       if not needed)])

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = newsig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        scale_lat = latitude = bound_args.arguments.pop('latitude')
        scale_lon = longitude = bound_args.arguments.pop('longitude')
        crs = bound_args.arguments.pop('crs')

        # Choose the first DataArray argument to act as grid prototype
        grid_prototype = next(dataarray_arguments(bound_args), None)

        # Fill in x_dim/y_dim
        if (
            grid_prototype is not None
            and 'x_dim' in bound_args.arguments
            and 'y_dim' in bound_args.arguments
        ):
            try:
                bound_args.arguments['x_dim'] = grid_prototype.metpy.find_axis_number('x')
                bound_args.arguments['y_dim'] = grid_prototype.metpy.find_axis_number('y')
            except AttributeError:
                # If axis number not found, fall back to default but warn.
                warnings.warn('Horizontal dimension numbers not found. Defaulting to '
                              '(..., Y, X) order.')

        # Fill in vertical_dim
        if (
            grid_prototype is not None
            and 'vertical_dim' in bound_args.arguments
        ):
            try:
                bound_args.arguments['vertical_dim'] = (
                    grid_prototype.metpy.find_axis_number('vertical')
                )
            except AttributeError:
                # If axis number not found, fall back to default but warn.
                warnings.warn(
                    'Vertical dimension number not found. Defaulting to (..., Z, Y, X) order.'
                )

        # Fill in dz
        if (
            grid_prototype is not None
            and 'dz' in bound_args.arguments
        ):
            if bound_args.arguments['dz'] is None:
                try:
                    vertical_coord = grid_prototype.metpy.vertical
                    bound_args.arguments['dz'] = np.diff(vertical_coord.metpy.unit_array)
                except (AttributeError, ValueError):
                    # Skip, since this only comes up in advection, where dz is optional
                    # (may not need vertical at all)
                    pass
            if (
                func.__name__.endswith('advection')
                and bound_args.arguments['u'] is None
                and bound_args.arguments['v'] is None
            ):
                return func(*bound_args.args, **bound_args.kwargs)

        # Fill in dx and dy
        if (
            'dx' in bound_args.arguments and bound_args.arguments['dx'] is None
            and 'dy' in bound_args.arguments and bound_args.arguments['dy'] is None
        ):
            if grid_prototype is not None:
                grid_deltas = grid_prototype.metpy.grid_deltas
                bound_args.arguments['dx'] = grid_deltas['dx']
                bound_args.arguments['dy'] = grid_deltas['dy']
            elif longitude is not None and latitude is not None and crs is not None:
                # TODO: de-duplicate .metpy.grid_deltas code
                geod = None if crs is None else crs.get_geod()
                bound_args.arguments['dx'], bound_args.arguments['dy'] = (
                    nominal_lat_lon_grid_deltas(longitude, latitude, geod)
                )
            elif 'dz' in bound_args.arguments:
                # Handle advection case, allowing dx/dy to be None but dz to not be None
                if bound_args.arguments['dz'] is None:
                    raise ValueError(
                        'Must provide dx, dy, and/or dz arguments or input DataArray with '
                        'interpretable dimension coordinates.'
                    )
            else:
                raise ValueError(
                    'Must provide dx/dy arguments, input DataArray with interpretable '
                    'dimension coordinates, or 1D longitude/latitude arguments with an '
                    'optional PyProj CRS.'
                )

        # Fill in parallel_scale and meridional_scale
        if (
            'parallel_scale' in bound_args.arguments
            and bound_args.arguments['parallel_scale'] is None
            and 'meridional_scale' in bound_args.arguments
            and bound_args.arguments['meridional_scale'] is None
        ):
            proj = None
            if grid_prototype is not None:
                # Fall back to basic cartesian calculation if we don't have a CRS or we
                # are unable to get the coordinates needed for map factor calculation
                # (either existing lat/lon or lat/lon computed from y/x)
                with contextlib.suppress(AttributeError):
                    latitude, longitude = grid_prototype.metpy.coordinates('latitude',
                                                                           'longitude')
                    scale_lat = latitude.metpy.unit_array
                    scale_lon = longitude.metpy.unit_array
                    if hasattr(grid_prototype.metpy, 'pyproj_proj'):
                        proj = grid_prototype.metpy.pyproj_proj
                    elif latitude.squeeze().ndim == 1 and longitude.squeeze().ndim == 1:
                        proj = Proj(CRS('+proj=latlon'))
            elif latitude is not None and longitude is not None:
                try:
                    proj = Proj(crs)
                except Exception as e:
                    # Whoops, intended to use
                    raise ValueError(
                        'Latitude and longitude arguments provided so as to make '
                        'calculation projection-correct, however, projection CRS is '
                        'missing or invalid.'
                    ) from e

            # Do we have everything we need to sensibly calculate the scale arrays?
            if proj is not None:
                scale_lat = scale_lat.squeeze().m_as('degrees')
                scale_lon = scale_lon.squeeze().m_as('degrees')
                if scale_lat.ndim == 1 and scale_lon.ndim == 1:
                    scale_lon, scale_lat = np.meshgrid(scale_lon, scale_lat)
                elif scale_lat.ndim != 2 or scale_lon.ndim != 2:
                    raise ValueError('Latitude and longitude must be either 1D or 2D.')
                factors = proj.get_factors(scale_lon, scale_lat)
                p_scale = factors.parallel_scale
                m_scale = factors.meridional_scale

                if grid_prototype is not None:
                    # Set the dims and coords using the original from the input lat/lon.
                    # This particular implementation relies on them being 1D/2D for the dims.
                    xr_kwargs = {'coords': {**latitude.coords, **longitude.coords},
                                 'dims': (latitude.dims[0], longitude.dims[-1])}
                    p_scale = xr.DataArray(p_scale, **xr_kwargs)
                    m_scale = xr.DataArray(m_scale, **xr_kwargs)

                bound_args.arguments['parallel_scale'] = p_scale
                bound_args.arguments['meridional_scale'] = m_scale

        # If the original function uses any of the arguments that are otherwise dynamically
        # added, be sure to pass them to the original function.
        local_namespace = vars()
        bound_args.arguments.update({param: local_namespace[param]
                                     for param, uses in orig_func_uses.items() if uses})

        return func(*bound_args.args, **bound_args.kwargs)

    # Override the wrapper function's signature with a better signature. Also add docstrings
    # for our added parameters.
    wrapper.__signature__ = newsig
    if getattr(wrapper, '__doc__', None) is not None:
        wrapper.__doc__ = _add_grid_params_to_docstring(wrapper.__doc__, orig_func_uses)

    return wrapper


@exporter.export
@xarray_derivative_wrap
def first_derivative(f, axis=None, x=None, delta=None):
    """Calculate the first derivative of a grid of values.

    Works for both regularly-spaced data and grids with varying spacing.

    Either `x` or `delta` must be specified, or `f` must be given as an `xarray.DataArray` with
    attached coordinate and projection information. If `f` is an `xarray.DataArray`, and `x` or
    `delta` are given, `f` will be converted to a `pint.Quantity` and the derivative returned
    as a `pint.Quantity`, otherwise, if neither `x` nor `delta` are given, the attached
    coordinate information belonging to `axis` will be used and the derivative will be returned
    as an `xarray.DataArray`.

    This uses 3 points to calculate the derivative, using forward or backward at the edges of
    the grid as appropriate, and centered elsewhere. The irregular spacing is handled
    explicitly, using the formulation as specified by [Bowen2005]_.

    Parameters
    ----------
    f : array-like
        Array of values of which to calculate the derivative
    axis : int or str, optional
        The array axis along which to take the derivative. If `f` is ndarray-like, must be an
        integer. If `f` is a `DataArray`, can be a string (referring to either the coordinate
        dimension name or the axis type) or integer (referring to axis number), unless using
        implicit conversion to `pint.Quantity`, in which case it must be an integer. Defaults
        to 0. For reference, the current standard axis types are 'time', 'vertical', 'y', and
        'x'.
    x : array-like, optional
        The coordinate values corresponding to the grid points in `f`
    delta : array-like, optional
        Spacing between the grid points in `f`. Should be one item less than the size
        of `f` along `axis`.

    Returns
    -------
    array-like
        The first derivative calculated along the selected axis


    .. versionchanged:: 1.0
       Changed signature from ``(f, **kwargs)``

    See Also
    --------
    second_derivative

    """
    n, axis, delta = _process_deriv_args(f, axis, x, delta)
    take = make_take(n, axis)

    # First handle centered case
    slice0 = take(slice(None, -2))
    slice1 = take(slice(1, -1))
    slice2 = take(slice(2, None))
    delta_slice0 = take(slice(None, -1))
    delta_slice1 = take(slice(1, None))

    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    delta_diff = delta[delta_slice1] - delta[delta_slice0]
    center = (- delta[delta_slice1] / (combined_delta * delta[delta_slice0]) * f[slice0]
              + delta_diff / (delta[delta_slice0] * delta[delta_slice1]) * f[slice1]
              + delta[delta_slice0] / (combined_delta * delta[delta_slice1]) * f[slice2])

    # Fill in "left" edge with forward difference
    slice0 = take(slice(None, 1))
    slice1 = take(slice(1, 2))
    slice2 = take(slice(2, 3))
    delta_slice0 = take(slice(None, 1))
    delta_slice1 = take(slice(1, 2))

    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    big_delta = combined_delta + delta[delta_slice0]
    left = (- big_delta / (combined_delta * delta[delta_slice0]) * f[slice0]
            + combined_delta / (delta[delta_slice0] * delta[delta_slice1]) * f[slice1]
            - delta[delta_slice0] / (combined_delta * delta[delta_slice1]) * f[slice2])

    # Now the "right" edge with backward difference
    slice0 = take(slice(-3, -2))
    slice1 = take(slice(-2, -1))
    slice2 = take(slice(-1, None))
    delta_slice0 = take(slice(-2, -1))
    delta_slice1 = take(slice(-1, None))

    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    big_delta = combined_delta + delta[delta_slice1]
    right = (delta[delta_slice1] / (combined_delta * delta[delta_slice0]) * f[slice0]
             - combined_delta / (delta[delta_slice0] * delta[delta_slice1]) * f[slice1]
             + big_delta / (combined_delta * delta[delta_slice1]) * f[slice2])

    return concatenate((left, center, right), axis=axis)


@exporter.export
@xarray_derivative_wrap
def second_derivative(f, axis=None, x=None, delta=None):
    """Calculate the second derivative of a grid of values.

    Works for both regularly-spaced data and grids with varying spacing.

    Either `x` or `delta` must be specified, or `f` must be given as an `xarray.DataArray` with
    attached coordinate and projection information. If `f` is an `xarray.DataArray`, and `x` or
    `delta` are given, `f` will be converted to a `pint.Quantity` and the derivative returned
    as a `pint.Quantity`, otherwise, if neither `x` nor `delta` are given, the attached
    coordinate information belonging to `axis` will be used and the derivative will be returned
    as an `xarray.DataArray`

    This uses 3 points to calculate the derivative, using forward or backward at the edges of
    the grid as appropriate, and centered elsewhere. The irregular spacing is handled
    explicitly, using the formulation as specified by [Bowen2005]_.

    Parameters
    ----------
    f : array-like
        Array of values of which to calculate the derivative
    axis : int or str, optional
        The array axis along which to take the derivative. If `f` is ndarray-like, must be an
        integer. If `f` is a `DataArray`, can be a string (referring to either the coordinate
        dimension name or the axis type) or integer (referring to axis number), unless using
        implicit conversion to `pint.Quantity`, in which case it must be an integer. Defaults
        to 0. For reference, the current standard axis types are 'time', 'vertical', 'y', and
        'x'.
    x : array-like, optional
        The coordinate values corresponding to the grid points in `f`
    delta : array-like, optional
        Spacing between the grid points in `f`. There should be one item less than the size
        of `f` along `axis`.

    Returns
    -------
    array-like
        The second derivative calculated along the selected axis


    .. versionchanged:: 1.0
       Changed signature from ``(f, **kwargs)``

    See Also
    --------
    first_derivative

    """
    n, axis, delta = _process_deriv_args(f, axis, x, delta)
    take = make_take(n, axis)

    # First handle centered case
    slice0 = take(slice(None, -2))
    slice1 = take(slice(1, -1))
    slice2 = take(slice(2, None))
    delta_slice0 = take(slice(None, -1))
    delta_slice1 = take(slice(1, None))

    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    center = 2 * (f[slice0] / (combined_delta * delta[delta_slice0])
                  - f[slice1] / (delta[delta_slice0] * delta[delta_slice1])
                  + f[slice2] / (combined_delta * delta[delta_slice1]))

    # Fill in "left" edge
    slice0 = take(slice(None, 1))
    slice1 = take(slice(1, 2))
    slice2 = take(slice(2, 3))
    delta_slice0 = take(slice(None, 1))
    delta_slice1 = take(slice(1, 2))

    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    left = 2 * (f[slice0] / (combined_delta * delta[delta_slice0])
                - f[slice1] / (delta[delta_slice0] * delta[delta_slice1])
                + f[slice2] / (combined_delta * delta[delta_slice1]))

    # Now the "right" edge
    slice0 = take(slice(-3, -2))
    slice1 = take(slice(-2, -1))
    slice2 = take(slice(-1, None))
    delta_slice0 = take(slice(-2, -1))
    delta_slice1 = take(slice(-1, None))

    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    right = 2 * (f[slice0] / (combined_delta * delta[delta_slice0])
                 - f[slice1] / (delta[delta_slice0] * delta[delta_slice1])
                 + f[slice2] / (combined_delta * delta[delta_slice1]))

    return concatenate((left, center, right), axis=axis)


@exporter.export
def gradient(f, axes=None, coordinates=None, deltas=None):
    """Calculate the gradient of a scalar quantity, assuming Cartesian coordinates.

    Works for both regularly-spaced data, and grids with varying spacing.

    Either `coordinates` or `deltas` must be specified, or `f` must be given as an
    `xarray.DataArray` with  attached coordinate and projection information. If `f` is an
    `xarray.DataArray`, and `coordinates` or `deltas` are given, `f` will be converted to a
    `pint.Quantity` and the gradient returned as a tuple of `pint.Quantity`, otherwise, if
    neither `coordinates` nor `deltas` are given, the attached coordinate information belonging
    to `axis` will be used and the gradient will be returned as a tuple of `xarray.DataArray`.

    Parameters
    ----------
    f : array-like
        Array of values of which to calculate the derivative
    axes : Sequence[str] or Sequence[int], optional
        Sequence of strings (if `f` is a `xarray.DataArray` and implicit conversion to
        `pint.Quantity` is not used) or integers that specify the array axes along which to
        take the derivatives. Defaults to all axes of `f`. If given, and used with
        `coordinates` or `deltas`, its length must be less than or equal to that of the
        `coordinates` or `deltas` given. In general, each axis can be an axis number
        (integer), dimension coordinate name (string) or a standard axis type (string). The
        current standard axis types are 'time', 'vertical', 'y', and 'x'.
    coordinates : array-like, optional
        Sequence of arrays containing the coordinate values corresponding to the
        grid points in `f` in axis order.
    deltas : array-like, optional
        Sequence of arrays or scalars that specify the spacing between the grid points in `f`
        in axis order. There should be one item less than the size of `f` along the applicable
        axis.

    Returns
    -------
    tuple of array-like
        The first derivative calculated along each specified axis of the original array

    See Also
    --------
    laplacian, first_derivative, vector_derivative, geospatial_gradient

    Notes
    -----
    If this function is used without the `axes` parameter, the length of `coordinates` or
    `deltas` (as applicable) should match the number of dimensions of `f`.

    This will not give projection-correct results for horizontal geospatial fields. Instead,
    for vector quantities, use `vector_derivative`, and for scalar quantities, use
    `geospatial_gradient`.

    .. versionchanged:: 1.0
       Changed signature from ``(f, **kwargs)``

    """
    pos_kwarg, positions, axes = _process_gradient_args(f, axes, coordinates, deltas)
    return tuple(first_derivative(f, axis=axis, **{pos_kwarg: positions[ind]})
                 for ind, axis in enumerate(axes))


@exporter.export
def laplacian(f, axes=None, coordinates=None, deltas=None):
    """Calculate the laplacian of a grid of values.

    Works for both regularly-spaced data, and grids with varying spacing.

    Either `coordinates` or `deltas` must be specified, or `f` must be given as an
    `xarray.DataArray` with  attached coordinate and projection information. If `f` is an
    `xarray.DataArray`, and `coordinates` or `deltas` are given, `f` will be converted to a
    `pint.Quantity` and the gradient returned as a tuple of `pint.Quantity`, otherwise, if
    neither `coordinates` nor `deltas` are given, the attached coordinate information belonging
    to `axis` will be used and the gradient will be returned as a tuple of `xarray.DataArray`.

    Parameters
    ----------
    f : array-like
        Array of values of which to calculate the derivative
    axes : Sequence[str] or Sequence[int], optional
        Sequence of strings (if `f` is a `xarray.DataArray` and implicit conversion to
        `pint.Quantity` is not used) or integers that specify the array axes along which to
        take the derivatives. Defaults to all axes of `f`. If given, and used with
        `coordinates` or `deltas`, its length must be less than or equal to that of the
        `coordinates` or `deltas` given. In general, each axis can be an axis number
        (integer), dimension coordinate name (string) or a standard axis type (string). The
        current standard axis types are 'time', 'vertical', 'y', and 'x'.
    coordinates : array-like, optional
        The coordinate values corresponding to the grid points in `f`
    deltas : array-like, optional
        Spacing between the grid points in `f`. There should be one item less than the size
        of `f` along the applicable axis.

    Returns
    -------
    array-like
        The laplacian

    See Also
    --------
    gradient, second_derivative

    Notes
    -----
    If this function is used without the `axes` parameter, the length of `coordinates` or
    `deltas` (as applicable) should match the number of dimensions of `f`.

    .. versionchanged:: 1.0
       Changed signature from ``(f, **kwargs)``

    """
    pos_kwarg, positions, axes = _process_gradient_args(f, axes, coordinates, deltas)
    derivs = [second_derivative(f, axis=axis, **{pos_kwarg: positions[ind]})
              for ind, axis in enumerate(axes)]
    return sum(derivs)


@exporter.export
@parse_grid_arguments
@preprocess_and_wrap(wrap_like=None,
                     broadcast=('u', 'v', 'parallel_scale', 'meridional_scale'))
@check_units(dx='[length]', dy='[length]')
def vector_derivative(u, v, *, dx=None, dy=None, x_dim=-1, y_dim=-2,
                      parallel_scale=None, meridional_scale=None, return_only=None):
    r"""Calculate the projection-correct derivative matrix of a 2D vector.

    Parameters
    ----------
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the vector
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the vector
    return_only : str or Sequence[str], optional
        Sequence of which components of the derivative matrix to compute and return. If none,
        returns the full matrix as a tuple of tuples (('du/dx', 'du/dy'), ('dv/dx', 'dv/dy')).
        Otherwise, matches the return pattern of the given strings. Only valid strings are
        'du/dx', 'du/dy', 'dv/dx', and 'dv/dy'.

    Returns
    -------
    `pint.Quantity`, tuple of `pint.Quantity`, or tuple of tuple of `pint.Quantity`
        Component(s) of vector derivative

    Other Parameters
    ----------------
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input. Also optional if one-dimensional
        longitude and latitude arguments are given for your data on a non-projected grid.
        Keyword-only argument.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input. Also optional if one-dimensional
        longitude and latitude arguments are given for your data on a non-projected grid.
        Keyword-only argument.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`. Keyword-only argument.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`. Keyword-only argument.
    parallel_scale : `pint.Quantity`, optional
        Parallel scale of map projection at data coordinate. Optional if `xarray.DataArray`
        with latitude/longitude coordinates and MetPy CRS used as input. Also optional if
        longitude, latitude, and crs are given. If otherwise omitted, calculation will be
        carried out on a Cartesian, rather than geospatial, grid. Keyword-only argument.
    meridional_scale : `pint.Quantity`, optional
        Meridional scale of map projection at data coordinate. Optional if `xarray.DataArray`
        with latitude/longitude coordinates and MetPy CRS used as input. Also optional if
        longitude, latitude, and crs are given. If otherwise omitted, calculation will be
        carried out on a Cartesian, rather than geospatial, grid. Keyword-only argument.

    See Also
    --------
    geospatial_gradient, geospatial_laplacian, first_derivative

    """
    # Determine which derivatives to calculate
    derivatives = {
        component: None
        for component in ('du/dx', 'du/dy', 'dv/dx', 'dv/dy')
        if (return_only is None or component in return_only)
    }
    map_factor_correction = parallel_scale is not None and meridional_scale is not None

    # Add in the map factor derivatives if needed
    if map_factor_correction and ('du/dx' in derivatives or 'dv/dx' in derivatives):
        derivatives['dp/dy'] = None
    if map_factor_correction and ('du/dy' in derivatives or 'dv/dy' in derivatives):
        derivatives['dm/dx'] = None

    # Compute the Cartesian derivatives
    for component in derivatives:
        scalar = {
            'du': u, 'dv': v, 'dp': parallel_scale, 'dm': meridional_scale
        }[component[:2]]
        delta, dim = (dx, x_dim) if component[-2:] == 'dx' else (dy, y_dim)
        derivatives[component] = first_derivative(scalar, delta=delta, axis=dim)

    # Apply map factor corrections
    if map_factor_correction:
        # Factor against opposite component
        if 'dp/dy' in derivatives:
            dx_correction = meridional_scale / parallel_scale * derivatives['dp/dy']
        if 'dm/dx' in derivatives:
            dy_correction = parallel_scale / meridional_scale * derivatives['dm/dx']

        # Corrected terms
        if 'du/dx' in derivatives:
            derivatives['du/dx'] = parallel_scale * derivatives['du/dx'] - v * dx_correction
        if 'du/dy' in derivatives:
            derivatives['du/dy'] = meridional_scale * derivatives['du/dy'] + v * dy_correction
        if 'dv/dx' in derivatives:
            derivatives['dv/dx'] = parallel_scale * derivatives['dv/dx'] + u * dx_correction
        if 'dv/dy' in derivatives:
            derivatives['dv/dy'] = meridional_scale * derivatives['dv/dy'] - u * dy_correction

    if return_only is None:
        return (
            derivatives['du/dx'], derivatives['du/dy'],
            derivatives['dv/dx'], derivatives['dv/dy']
        )
    elif isinstance(return_only, str):
        return derivatives[return_only]
    else:
        return tuple(derivatives[component] for component in return_only)


@exporter.export
@parse_grid_arguments
@preprocess_and_wrap(wrap_like=None,
                     broadcast=('f', 'parallel_scale', 'meridional_scale'))
@check_units(dx='[length]', dy='[length]')
def geospatial_gradient(f, *, dx=None, dy=None, x_dim=-1, y_dim=-2,
                        parallel_scale=None, meridional_scale=None, return_only=None):
    r"""Calculate the projection-correct gradient of a 2D scalar field.

    Parameters
    ----------
    f : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        scalar field for which the horizontal gradient should be calculated
    return_only : str or Sequence[str], optional
        Sequence of which components of the gradient to compute and return. If none,
        returns the gradient tuple ('df/dx', 'df/dy'). Otherwise, matches the return
        pattern of the given strings. Only valid strings are 'df/dx', 'df/dy'.

    Returns
    -------
    `pint.Quantity`, tuple of `pint.Quantity`, or tuple of pairs of `pint.Quantity`
        Component(s) of vector derivative

    Other Parameters
    ----------------
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input. Also optional if one-dimensional
        longitude and latitude arguments are given for your data on a non-projected grid.
        Keyword-only argument.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input. Also optional if one-dimensional
        longitude and latitude arguments are given for your data on a non-projected grid.
        Keyword-only argument.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`. Keyword-only argument.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`. Keyword-only argument.
    parallel_scale : `pint.Quantity`, optional
        Parallel scale of map projection at data coordinate. Optional if `xarray.DataArray`
        with latitude/longitude coordinates and MetPy CRS used as input. Also optional if
        longitude, latitude, and crs are given. If otherwise omitted, calculation will be
        carried out on a Cartesian, rather than geospatial, grid. Keyword-only argument.
    meridional_scale : `pint.Quantity`, optional
        Meridional scale of map projection at data coordinate. Optional if `xarray.DataArray`
        with latitude/longitude coordinates and MetPy CRS used as input. Also optional if
        longitude, latitude, and crs are given. If otherwise omitted, calculation will be
        carried out on a Cartesian, rather than geospatial, grid. Keyword-only argument.

    See Also
    --------
    vector_derivative, gradient, geospatial_laplacian

    """
    derivatives = {component: None
                   for component in ('df/dx', 'df/dy')
                   if (return_only is None or component in return_only)}

    scales = {'df/dx': parallel_scale, 'df/dy': meridional_scale}

    map_factor_correction = parallel_scale is not None and meridional_scale is not None

    for component in derivatives:
        delta, dim = (dx, x_dim) if component[-2:] == 'dx' else (dy, y_dim)
        derivatives[component] = first_derivative(f, delta=delta, axis=dim)

        if map_factor_correction:
            derivatives[component] *= scales[component]

    # Build return collection
    if return_only is None:
        return derivatives['df/dx'], derivatives['df/dy']
    elif isinstance(return_only, str):
        return derivatives[return_only]
    else:
        return tuple(derivatives[component] for component in return_only)


def _broadcast_to_axis(arr, axis, ndim):
    """Handle reshaping coordinate array to have proper dimensionality.

    This puts the values along the specified axis.
    """
    if arr.ndim == 1 and arr.ndim < ndim:
        new_shape = [1] * ndim
        new_shape[axis] = arr.size
        arr = arr.reshape(*new_shape)
    return arr


def _process_gradient_args(f, axes, coordinates, deltas):
    """Handle common processing of arguments for gradient and gradient-like functions."""
    axes_given = axes is not None
    axes = axes if axes_given else range(f.ndim)

    def _check_length(positions):
        if axes_given and len(positions) < len(axes):
            raise ValueError('Length of "coordinates" or "deltas" cannot be less than that '
                             'of "axes".')
        elif not axes_given and len(positions) != len(axes):
            raise ValueError('Length of "coordinates" or "deltas" must match the number of '
                             'dimensions of "f" when "axes" is not given.')

    if deltas is not None:
        if coordinates is not None:
            raise ValueError('Cannot specify both "coordinates" and "deltas".')
        _check_length(deltas)
        return 'delta', deltas, axes
    elif coordinates is not None:
        _check_length(coordinates)
        return 'x', coordinates, axes
    elif isinstance(f, xr.DataArray):
        return 'pass', axes, axes  # only the axis argument matters
    else:
        raise ValueError('Must specify either "coordinates" or "deltas" for value positions '
                         'when "f" is not a DataArray.')


def _process_deriv_args(f, axis, x, delta):
    """Handle common processing of arguments for derivative functions."""
    n = f.ndim
    axis = normalize_axis_index(axis if axis is not None else 0, n)

    if f.shape[axis] < 3:
        raise ValueError('f must have at least 3 point along the desired axis.')

    if delta is not None:
        if x is not None:
            raise ValueError('Cannot specify both "x" and "delta".')

        delta = np.atleast_1d(delta)
        if delta.size == 1:
            diff_size = list(f.shape)
            diff_size[axis] -= 1
            delta_units = getattr(delta, 'units', None)
            delta = np.broadcast_to(delta, diff_size, subok=True)
            if not hasattr(delta, 'units') and delta_units is not None:
                delta = units.Quantity(delta, delta_units)
        else:
            delta = _broadcast_to_axis(delta, axis, n)
    elif x is not None:
        x = _broadcast_to_axis(x, axis, n)
        delta = np.diff(x, axis=axis)
    else:
        raise ValueError('Must specify either "x" or "delta" for value positions.')

    return n, axis, delta


@exporter.export
@preprocess_and_wrap(wrap_like='input_dir')
def parse_angle(input_dir):
    """Calculate the meteorological angle from directional text.

    Works for abbreviations or whole words (E -> 90 | South -> 180)
    and also is able to parse 22.5 degree angles such as ESE/East South East.

    Parameters
    ----------
    input_dir : str or Sequence[str]
        Directional text such as west, [south-west, ne], etc.

    Returns
    -------
    `pint.Quantity`
        The angle in degrees

    """
    if isinstance(input_dir, str):
        # abb_dirs = abbrieviated directions
        abb_dirs = _clean_direction([_abbrieviate_direction(input_dir)])
    elif hasattr(input_dir, '__len__'):  # handle np.array, pd.Series, list, and array-like
        input_dir_str = ','.join(_clean_direction(input_dir, preprocess=True))
        abb_dir_str = _abbrieviate_direction(input_dir_str)
        abb_dirs = _clean_direction(abb_dir_str.split(','))
    else:  # handle unrecognizable scalar
        return np.nan

    return itemgetter(*abb_dirs)(DIR_DICT)


def _clean_direction(dir_list, preprocess=False):
    """Handle None if preprocess, else handles anything not in DIR_STRS."""
    if preprocess:  # primarily to remove None from list so ','.join works
        return [UND if not isinstance(the_dir, str) else the_dir
                for the_dir in dir_list]
    else:  # remove extraneous abbreviated directions
        return [UND if the_dir not in DIR_STRS else the_dir
                for the_dir in dir_list]


def _abbrieviate_direction(ext_dir_str):
    """Convert extended (non-abbreviated) directions to abbreviation."""
    return (ext_dir_str
            .upper()
            .replace('_', '')
            .replace('-', '')
            .replace(' ', '')
            .replace('NORTH', 'N')
            .replace('EAST', 'E')
            .replace('SOUTH', 'S')
            .replace('WEST', 'W')
            )


@exporter.export
@preprocess_and_wrap()
def angle_to_direction(input_angle, full=False, level=3):
    """Convert the meteorological angle to directional text.

    Works for angles greater than or equal to 360 (360 -> N | 405 -> NE)
    and rounds to the nearest angle (355 -> N | 404 -> NNE)

    Parameters
    ----------
    input_angle : float or array-like
        Angles such as 0, 25, 45, 360, 410, etc.
    full : bool
        True returns full text (South), False returns abbreviated text (S)
    level : int
        Level of detail (3 = N/NNE/NE/ENE/E... 2 = N/NE/E/SE... 1 = N/E/S/W)

    Returns
    -------
    direction
        The directional text

    """
    try:  # strip units temporarily
        origin_units = input_angle.units
        input_angle = input_angle.m
    except AttributeError:  # no units associated
        origin_units = units.degree

    if not hasattr(input_angle, '__len__') or isinstance(input_angle, str):
        input_angle = [input_angle]
        scalar = True
    else:
        scalar = False

    # clean any numeric strings, negatives, and None does not handle strings with alphabet
    input_angle = units.Quantity(np.array(input_angle).astype(float), origin_units)
    input_angle[input_angle < 0] = units.Quantity(np.nan, origin_units)

    # normalizer used for angles > 360 degree to normalize between 0 - 360
    normalizer = np.array(input_angle.m / MAX_DEGREE_ANGLE.m, dtype=int)
    norm_angles = abs(input_angle - MAX_DEGREE_ANGLE * normalizer)

    if level == 3:
        nskip = 1
    elif level == 2:
        nskip = 2
    elif level == 1:
        nskip = 4
    else:
        err_msg = 'Level of complexity cannot be less than 1 or greater than 3!'
        raise ValueError(err_msg)

    angle_dict = {i * BASE_DEGREE_MULTIPLIER.m * nskip: dir_str
                  for i, dir_str in enumerate(DIR_STRS[::nskip])}
    angle_dict[MAX_DEGREE_ANGLE.m] = 'N'  # handle edge case of 360.
    angle_dict[UND_ANGLE] = UND

    # round to the nearest angles for dict lookup
    # 0.001 is subtracted so there's an equal number of dir_str from
    # np.arange(0, 360, 22.5), or else some dir_str will be preferred

    # without the 0.001, level=2 would yield:
    # ['N', 'N', 'NE', 'E', 'E', 'E', 'SE', 'S', 'S',
    #  'S', 'SW', 'W', 'W', 'W', 'NW', 'N']

    # with the -0.001, level=2 would yield:
    # ['N', 'N', 'NE', 'NE', 'E', 'E', 'SE', 'SE',
    #  'S', 'S', 'SW', 'SW', 'W', 'W', 'NW', 'NW']

    multiplier = np.round(
        (norm_angles / BASE_DEGREE_MULTIPLIER / nskip) - 0.001).m
    round_angles = (multiplier * BASE_DEGREE_MULTIPLIER.m * nskip)
    round_angles[np.where(np.isnan(round_angles))] = UND_ANGLE

    dir_str_arr = itemgetter(*round_angles)(angle_dict)  # for array
    if not full:
        return dir_str_arr

    dir_str_arr = ','.join(dir_str_arr)
    dir_str_arr = _unabbrieviate_direction(dir_str_arr)
    return dir_str_arr.replace(',', ' ') if scalar else dir_str_arr.split(',')


def _unabbrieviate_direction(abb_dir_str):
    """Convert abbrieviated directions to non-abbrieviated direction."""
    return (abb_dir_str
            .upper()
            .replace(UND, 'Undefined ')
            .replace('N', 'North ')
            .replace('E', 'East ')
            .replace('S', 'South ')
            .replace('W', 'West ')
            .replace(' ,', ',')
            ).strip()


def _remove_nans(*variables):
    """Remove NaNs from arrays that cause issues with calculations.

    Takes a variable number of arguments and returns masked arrays in the same
    order as provided.
    """
    mask = None
    for v in variables:
        if mask is None:
            mask = np.isnan(v)
        else:
            mask |= np.isnan(v)

    # Mask everyone with that joint mask
    ret = []
    for v in variables:
        ret.append(v[~mask])
    return ret
