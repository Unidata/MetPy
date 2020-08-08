# Copyright (c) 2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains a collection of generally useful calculation tools."""
import functools
from inspect import signature
from operator import itemgetter
import warnings

import numpy as np
from numpy.core.numeric import normalize_axis_index
import numpy.ma as ma
from scipy.spatial import cKDTree
import xarray as xr

from ..cbook import broadcast_indices, result_type
from ..interpolate import interpolate_1d, log_interpolate_1d
from ..package_tools import Exporter
from ..units import check_units, concatenate, units
from ..xarray import check_axis, preprocess_xarray

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

MAX_DEGREE_ANGLE = 360 * units.degree
BASE_DEGREE_MULTIPLIER = 22.5 * units.degree

DIR_DICT = {dir_str: i * BASE_DEGREE_MULTIPLIER for i, dir_str in enumerate(DIR_STRS)}
DIR_DICT[UND] = np.nan


@exporter.export
@preprocess_xarray
def resample_nn_1d(a, centers):
    """Return one-dimensional nearest-neighbor indexes based on user-specified centers.

    Parameters
    ----------
    a : array-like
        1-dimensional array of numeric values from which to
        extract indexes of nearest-neighbors
    centers : array-like
        1-dimensional array of numeric values representing a subset of values to approximate

    Returns
    -------
        An array of indexes representing values closest to given array values

    """
    ix = []
    for center in centers:
        index = (np.abs(a - center)).argmin()
        if index not in ix:
            ix.append(index)
    return ix


@exporter.export
@preprocess_xarray
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
@preprocess_xarray
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
    direction : string, optional
        specifies direction of crossing. 'all', 'increasing' (a becoming greater than b),
        or 'decreasing' (b becoming greater than a). Defaults to 'all'.
    log_x : bool, optional
        Use logarithmic interpolation along the `x` axis (i.e. for finding intersections
        in pressure coordinates). Default is False.

    Returns
    -------
        A tuple (x, y) of array-like with the x and y coordinates of the
        intersections of the lines.

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
        raise ValueError('Unknown option for direction: {0}'.format(str(direction)))

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
        index of requested element

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
        source arrays

    Returns
    -------
    arrs : one or more array-like
        arrays with masked elements removed

    """
    if any(hasattr(a, 'mask') for a in arrs):
        keep = ~functools.reduce(np.logical_or, (np.ma.getmaskarray(a) for a in arrs))
        return tuple(ma.asarray(a[keep]) for a in arrs)
    else:
        return arrs


@exporter.export
@preprocess_xarray
def reduce_point_density(points, radius, priority=None):
    r"""Return a mask to reduce the density of points in irregularly-spaced data.

    This function is used to down-sample a collection of scattered points (e.g. surface
    data), returning a mask that can be used to select the points from one or more arrays
    (e.g. arrays of temperature and dew point). The points selected can be controlled by
    providing an array of ``priority`` values (e.g. rainfall totals to ensure that
    stations with higher precipitation remain in the mask). The points and radius can be
    specified with units. If none are provided, meters are assumed.

    Parameters
    ----------
    points : (N, K) array-like
        N locations of the points in K dimensional space
    radius : `pint.Quantity` or float
        Minimum radius allowed between points. If units are not provided, meters is assumed.
    priority : (N, K) array-like, optional
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

    # Keep all points initially
    keep = np.ones(len(points), dtype=np.bool)

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
        The bound pressure and height.

    """
    # avoid circular import if basic.py ever imports something from tools.py
    from .basic import height_to_pressure_std, pressure_to_height_std
    # Make sure pressure is monotonically decreasing
    sort_inds = np.argsort(pressure)[::-1]
    pressure = pressure[sort_inds]
    if height is not None:
        height = height[sort_inds]

    # Bound is given in pressure
    if bound.dimensionality == {'[length]': -1.0, '[mass]': 1.0, '[time]': -2.0}:
        # If the bound is in the pressure data, we know the pressure bound exactly
        if bound in pressure:
            bound_pressure = bound
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
    elif bound.dimensionality == {'[length]': 1.0}:
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
                    bound_pressure = (np.interp(np.atleast_1d(bound.m), height.m,
                                                pressure.m).astype(result_type(bound))
                                      * pressure.units)
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
    if not (_greater_or_close(bound_pressure, np.nanmin(pressure.m) * pressure.units)
            and _less_or_close(bound_pressure, np.nanmax(pressure.m) * pressure.units)):
        raise ValueError('Specified bound is outside pressure range.')
    if height is not None and not (_less_or_close(bound_height,
                                                  np.nanmax(height.m) * height.units)
                                   and _greater_or_close(bound_height,
                                                         np.nanmin(height.m) * height.units)):
        raise ValueError('Specified bound is outside height range.')

    return bound_pressure, bound_height


@exporter.export
@preprocess_xarray
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
        The thickness of the layer
    args : array-like
        Atmospheric variable(s) measured at the given pressures
    bottom : `pint.Quantity`, optional
        The bottom of the layer
    interpolate : bool, optional
        Interpolate the top and bottom points if they are not in the given data. Defaults
        to True.
    with_agl : bool, optional
        Returns the height as above ground level by subtracting the minimum height in the
        provided height. Defaults to False.

    Returns
    -------
    `pint.Quantity, pint.Quantity`
        The height and data variables of the layer

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
@preprocess_xarray
@check_units('[pressure]')
def get_layer(pressure, *args, height=None, bottom=None, depth=100 * units.hPa,
              interpolate=True):
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
        The bottom of the layer as a pressure or height above the surface pressure. Defaults
        to the highest pressure or lowest height given.
    depth : `pint.Quantity`, optional
        The thickness of the layer as a pressure or height above the bottom of the layer.
        Defaults to 100 hPa.
    interpolate : bool, optional
        Interpolate the top and bottom points if they are not in the given data. Defaults
        to True.

    Returns
    -------
    `pint.Quantity, pint.Quantity`
        The pressure and data variables of the layer

    """
    # If we get the depth kwarg, but it's None, set it to the default as well
    if depth is None:
        depth = 100 * units.hPa

    # Make sure pressure and datavars are the same length
    for datavar in args:
        if len(pressure) != len(datavar):
            raise ValueError('Pressure and data variables must have the same length.')

    # If the bottom is not specified, make it the surface pressure
    if bottom is None:
        bottom = np.nanmax(pressure.m) * pressure.units

    bottom_pressure, bottom_height = _get_bound_pressure_height(pressure, bottom,
                                                                height=height,
                                                                interpolate=interpolate)

    # Calculate the top if whatever units depth is in
    if depth.dimensionality == {'[length]': -1.0, '[mass]': 1.0, '[time]': -2.0}:
        top = bottom_pressure - depth
    elif depth.dimensionality == {'[length]': 1}:
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
            p_interp = np.sort(np.append(p_interp.m, top_pressure.m)) * pressure.units
        if not np.any(np.isclose(bottom_pressure, p_interp)):
            p_interp = np.sort(np.append(p_interp.m, bottom_pressure.m)) * pressure.units

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
@preprocess_xarray
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
        The dimension of `arr` along which to search.

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
    indices = np.empty(indices_shape, dtype=np.int)
    good = np.empty(indices_shape, dtype=np.bool)

    # Used to put the output in the proper location
    take = make_take(arr.ndim, axis)

    # Loop over all of the values and for each, see where the value would be found from a
    # linear search
    for level_index, value in enumerate(values):
        # Look for changes in the value of the test for <= value in consecutive points
        # Taking abs() because we only care if there is a flip, not which direction.
        switches = np.abs(np.diff((arr <= value).astype(np.int), axis=axis))

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
    above = broadcast_indices(arr, indices, arr.ndim, axis)
    below = broadcast_indices(arr, indices - 1, arr.ndim, axis)

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
        Boolean array where values are less than or nearly equal to value.

    """
    return (a < value) | np.isclose(a, value, **kwargs)


def make_take(ndims, slice_dim):
    """Generate a take function to index in a particular dimension."""
    def take(indexer):
        return tuple(indexer if slice_dim % ndims == i else slice(None)  # noqa: S001
                     for i in range(ndims))
    return take


@exporter.export
@preprocess_xarray
def lat_lon_grid_deltas(longitude, latitude, y_dim=-2, x_dim=-1, **kwargs):
    r"""Calculate the actual delta between grid points that are in latitude/longitude format.

    Parameters
    ----------
    longitude : array_like
        array of longitudes defining the grid. If not a `pint.Quantity`, assumed to be in
        degrees.
    latitude : array_like
        array of latitudes defining the grid. If not a `pint.Quantity`, assumed to be in
        degrees.
    y_dim : int
        axis number for the y dimesion, defaults to -2.
    x_dim: int
        axis number for the x dimension, defaults to -1.
    kwargs
        Other keyword arguments to pass to :class:`~pyproj.Geod`

    Returns
    -------
    dx, dy:
        at least two dimensional arrays of signed deltas between grid points in the x and y
        direction

    Notes
    -----
    Accepts 1D, 2D, or higher arrays for latitude and longitude
    Assumes [..., Y, X] dimension order for input and output, unless keyword arguments `y_dim`
    and `x_dim` are otherwise specified.

    """
    from pyproj import Geod

    # Inputs must be the same number of dimensions
    if latitude.ndim != longitude.ndim:
        raise ValueError('Latitude and longitude must have the same number of dimensions.')

    # If we were given 1D arrays, make a mesh grid
    if latitude.ndim < 2:
        longitude, latitude = np.meshgrid(longitude, latitude)

    # pyproj requires ndarrays, not Quantities
    try:
        longitude = longitude.m_as('degrees')
        latitude = latitude.m_as('degrees')
    except AttributeError:
        longitude = np.asarray(longitude)
        latitude = np.asarray(latitude)

    # Determine dimension order for offset slicing
    take_y = make_take(latitude.ndim, y_dim)
    take_x = make_take(latitude.ndim, x_dim)

    geod_args = {'ellps': 'sphere'}
    if kwargs:
        geod_args = kwargs

    g = Geod(**geod_args)

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

    return dx * units.meter, dy * units.meter


@exporter.export
@preprocess_xarray
def azimuth_range_to_lat_lon(azimuths, ranges, center_lon, center_lat, **kwargs):
    """Convert azimuth and range locations in a polar coordinate system to lat/lon coordinates.

    Pole refers to the origin of the coordinate system.

    Parameters
    ----------
    azimuths : array_like
        array of azimuths defining the grid. If not a `pint.Quantity`,
        assumed to be in degrees.
    ranges : array_like
        array of range distances from the pole. Typically in meters.
    center_lat : float
        The latitude of the pole in decimal degrees
    center_lon : float
        The longitude of the pole in decimal degrees
    kwargs
        arbitrary keyword arguments to pass to pyproj.Geod (e.g. 'ellps')

    Returns
    -------
    lon, lat : 2D arrays of longitudes and latitudes corresponding to original locations

    Notes
    -----
    Credit to Brian Blaylock for the original implementation.

    """
    from pyproj import Geod

    geod_args = {'ellps': 'sphere'}
    if kwargs:
        geod_args = kwargs
    g = Geod(**geod_args)

    rng2d, az2d = np.meshgrid(ranges, azimuths)
    lats = np.full(az2d.shape, center_lat)
    lons = np.full(az2d.shape, center_lon)
    lon, lat, _ = g.fwd(lons, lats, az2d, rng2d)

    return lon, lat


@exporter.export
def grid_deltas_from_dataarray(f, kind='default'):
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
    # Determine behavior
    if kind == 'default' and f.metpy.crs['grid_mapping_name'] == 'latitude_longitude':
        kind = 'actual'
    elif kind == 'default':
        kind = 'nominal'
    elif kind not in ('actual', 'nominal'):
        raise ValueError('"kind" argument must be specified as "default", "actual", or '
                         '"nominal"')

    if kind == 'actual':
        # Get latitude/longitude coordinates and find dim order
        latitude, longitude = xr.broadcast(*f.metpy.coordinates('latitude', 'longitude'))
        try:
            y_dim = latitude.metpy.find_axis_number('y')
            x_dim = latitude.metpy.find_axis_number('x')
        except AttributeError:
            warnings.warn('y and x dimensions unable to be identified. Assuming [..., y, x] '
                          'dimension order.')
            y_dim, x_dim = -2, -1
        # Obtain grid deltas as xarray Variables
        (dx_var, dx_units), (dy_var, dy_units) = (
            (xr.Variable(dims=latitude.dims, data=deltas.magnitude), deltas.units)
            for deltas in lat_lon_grid_deltas(longitude, latitude, y_dim=y_dim, x_dim=x_dim,
                                              initstring=f.metpy.cartopy_crs.proj4_init))
    else:
        # Obtain y/x coordinate differences
        y, x = f.metpy.coordinates('y', 'x')
        dx_var = x.diff(x.dims[0]).variable
        dx_units = units(x.attrs.get('units'))
        dy_var = y.diff(y.dims[0]).variable
        dy_units = units(y.attrs.get('units'))

    # Broadcast to input and attach units
    dx = dx_var.set_dims(f.dims, shape=[dx_var.sizes[dim] if dim in dx_var.dims else 1
                                        for dim in f.dims]).data * dx_units
    dy = dy_var.set_dims(f.dims, shape=[dy_var.sizes[dim] if dim in dy_var.dims else 1
                                        for dim in f.dims]).data * dy_units

    return dx, dy


def xarray_derivative_wrap(func):
    """Decorate the derivative functions to make them work nicely with DataArrays.

    This will automatically determine if the coordinates can be pulled directly from the
    DataArray, or if a call to lat_lon_grid_deltas is needed.
    """
    @functools.wraps(func)
    def wrapper(f, **kwargs):
        if 'x' in kwargs or 'delta' in kwargs:
            # Use the usual DataArray to pint.Quantity preprocessing wrapper
            return preprocess_xarray(func)(f, **kwargs)
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


@exporter.export
@xarray_derivative_wrap
def first_derivative(f, **kwargs):
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
        The coordinate values corresponding to the grid points in `f`.
    delta : array-like, optional
        Spacing between the grid points in `f`. Should be one item less than the size
        of `f` along `axis`.

    Returns
    -------
    array-like
        The first derivative calculated along the selected axis.

    See Also
    --------
    second_derivative

    """
    n, axis, delta = _process_deriv_args(f, kwargs)
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
def second_derivative(f, **kwargs):
    """Calculate the second derivative of a grid of values.

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
        The coordinate values corresponding to the grid points in `f`.
    delta : array-like, optional
        Spacing between the grid points in `f`. There should be one item less than the size
        of `f` along `axis`.

    Returns
    -------
    array-like
        The second derivative calculated along the selected axis.

    See Also
    --------
    first_derivative

    """
    n, axis, delta = _process_deriv_args(f, kwargs)
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
def gradient(f, **kwargs):
    """Calculate the gradient of a grid of values.

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
    coordinates : array-like, optional
        Sequence of arrays containing the coordinate values corresponding to the
        grid points in `f` in axis order.
    deltas : array-like, optional
        Sequence of arrays or scalars that specify the spacing between the grid points in `f`
        in axis order. There should be one item less than the size of `f` along the applicable
        axis.
    axes : sequence, optional
        Sequence of strings (if `f` is a `xarray.DataArray` and implicit conversion to
        `pint.Quantity` is not used) or integers that specify the array axes along which to
        take the derivatives. Defaults to all axes of `f`. If given, and used with
        `coordinates` or `deltas`, its length must be less than or equal to that of the
        `coordinates` or `deltas` given. In general, each axis can be an axis number
        (integer), dimension coordinate name (string) or a standard axis type (string). The
        current standard axis types are 'time', 'vertical', 'y', and 'x'.

    Returns
    -------
    tuple of array-like
        The first derivative calculated along each specified axis of the original array

    See Also
    --------
    laplacian, first_derivative

    Notes
    -----
    If this function is used without the `axes` parameter, the length of `coordinates` or
    `deltas` (as applicable) should match the number of dimensions of `f`.

    """
    pos_kwarg, positions, axes = _process_gradient_args(f, kwargs)
    return tuple(first_derivative(f, axis=axis, **{pos_kwarg: positions[ind]})
                 for ind, axis in enumerate(axes))


@exporter.export
def laplacian(f, **kwargs):
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
    coordinates : array-like, optional
        The coordinate values corresponding to the grid points in `f`
    deltas : array-like, optional
        Spacing between the grid points in `f`. There should be one item less than the size
        of `f` along the applicable axis.
    axes : sequence, optional
        Sequence of strings (if `f` is a `xarray.DataArray` and implicit conversion to
        `pint.Quantity` is not used) or integers that specify the array axes along which to
        take the derivatives. Defaults to all axes of `f`. If given, and used with
        `coordinates` or `deltas`, its length must be less than or equal to that of the
        `coordinates` or `deltas` given. In general, each axis can be an axis number
        (integer), dimension coordinate name (string) or a standard axis type (string). The
        current standard axis types are 'time', 'vertical', 'y', and 'x'.

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

    """
    pos_kwarg, positions, axes = _process_gradient_args(f, kwargs)
    derivs = [second_derivative(f, axis=axis, **{pos_kwarg: positions[ind]})
              for ind, axis in enumerate(axes)]
    return sum(derivs)


def _broadcast_to_axis(arr, axis, ndim):
    """Handle reshaping coordinate array to have proper dimensionality.

    This puts the values along the specified axis.
    """
    if arr.ndim == 1 and arr.ndim < ndim:
        new_shape = [1] * ndim
        new_shape[axis] = arr.size
        arr = arr.reshape(*new_shape)
    return arr


def _process_gradient_args(f, kwargs):
    """Handle common processing of arguments for gradient and gradient-like functions."""
    axes = kwargs.get('axes', range(f.ndim))

    def _check_length(positions):
        if 'axes' in kwargs and len(positions) < len(axes):
            raise ValueError('Length of "coordinates" or "deltas" cannot be less than that '
                             'of "axes".')
        elif 'axes' not in kwargs and len(positions) != len(axes):
            raise ValueError('Length of "coordinates" or "deltas" must match the number of '
                             'dimensions of "f" when "axes" is not given.')

    if 'deltas' in kwargs:
        if 'coordinates' in kwargs or 'x' in kwargs:
            raise ValueError('Cannot specify both "coordinates" and "deltas".')
        _check_length(kwargs['deltas'])
        return 'delta', kwargs['deltas'], axes
    elif 'coordinates' in kwargs:
        _check_length(kwargs['coordinates'])
        return 'x', kwargs['coordinates'], axes
    elif isinstance(f, xr.DataArray):
        return 'pass', axes, axes  # only the axis argument matters
    else:
        raise ValueError('Must specify either "coordinates" or "deltas" for value positions '
                         'when "f" is not a DataArray.')


def _process_deriv_args(f, kwargs):
    """Handle common processing of arguments for derivative functions."""
    n = f.ndim
    axis = normalize_axis_index(kwargs.get('axis', 0), n)

    if f.shape[axis] < 3:
        raise ValueError('f must have at least 3 point along the desired axis.')

    if 'delta' in kwargs:
        if 'x' in kwargs:
            raise ValueError('Cannot specify both "x" and "delta".')

        delta = np.atleast_1d(kwargs['delta'])
        if delta.size == 1:
            diff_size = list(f.shape)
            diff_size[axis] -= 1
            delta_units = getattr(delta, 'units', None)
            delta = np.broadcast_to(delta, diff_size, subok=True)
            if not hasattr(delta, 'units') and delta_units is not None:
                delta = delta * delta_units
        else:
            delta = _broadcast_to_axis(delta, axis, n)
    elif 'x' in kwargs:
        x = _broadcast_to_axis(kwargs['x'], axis, n)
        delta = np.diff(x, axis=axis)
    else:
        raise ValueError('Must specify either "x" or "delta" for value positions.')

    return n, axis, delta


@exporter.export
@preprocess_xarray
def parse_angle(input_dir):
    """Calculate the meteorological angle from directional text.

    Works for abbrieviations or whole words (E -> 90 | South -> 180)
    and also is able to parse 22.5 degreee angles such as ESE/East South East

    Parameters
    ----------
    input_dir : string or array-like
        Directional text such as west, [south-west, ne], etc

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
    else:  # remove extraneous abbrieviated directions
        return [UND if the_dir not in DIR_STRS else the_dir
                for the_dir in dir_list]


def _abbrieviate_direction(ext_dir_str):
    """Convert extended (non-abbrievated) directions to abbrieviation."""
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
@preprocess_xarray
def angle_to_direction(input_angle, full=False, level=3):
    """Convert the meteorological angle to directional text.

    Works for angles greater than or equal to 360 (360 -> N | 405 -> NE)
    and rounds to the nearest angle (355 -> N | 404 -> NNE)

    Parameters
    ----------
    input_angle : numeric or array-like numeric
        Angles such as 0, 25, 45, 360, 410, etc
    full : boolean
        True returns full text (South), False returns abbrieviated text (S)
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

    # clean any numeric strings, negatives, and None
    # does not handle strings with alphabet
    input_angle = np.array(input_angle).astype(float)
    with np.errstate(invalid='ignore'):  # warns about the np.nan
        input_angle[np.where(input_angle < 0)] = np.nan

    input_angle = input_angle * origin_units

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
    if full:
        dir_str_arr = ','.join(dir_str_arr)
        dir_str_arr = _unabbrieviate_direction(dir_str_arr)
        if not scalar:
            dir_str = dir_str_arr.split(',')
        else:
            dir_str = dir_str_arr.replace(',', ' ')
    else:
        dir_str = dir_str_arr

    return dir_str


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

    Takes a variable number of arguments
    Returns masked arrays in the same order as provided
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


def wrap_output_like(**wrap_kwargs):
    """Wrap the output from a function to be like some other data object type.

    Wraps given data to match the units/coordinates/object type of another array. Currently
    supports:

    - As input (output from wrapped function):

        * ``pint.Quantity``
        * ``xarray.DataArray``
        * any type wrappable by ``pint.Quantity``

    - As matched output (final returned value):

        * ``pint.Quantity``
        * ``xarray.DataArray`` wrapping a ``pint.Quantity``

    (if matched output is not one of these types, we instead treat the match as if it was a
    dimenionless Quantity.)

    This wrapping/conversion follows the following rules:

    - If match_unit is False, for output of Quantity or DataArary respectively,

        * ndarray becomes dimensionless Quantity or unitless DataArray with matching coords
        * Quantity is unchanged or becomes DataArray with input units and output coords
        * DataArray is converted to Quantity by accessor or is unchanged

    - If match_unit is True, for output of Quantity or DataArary respectively, with a given
      unit,

        * ndarray becomes Quantity or DataArray (with matching coords) with output unit
        * Quantity is converted to output unit, then returned or converted to DataArray with
          matching coords
        * DataArray is has units converted via the accessor, then converted to Quantity via
          the accessor or returned

    The output to match can be specified two ways:

    - Using the `argument` keyword argument, the output is taken from argument of that name
      from the wrapped function's signature
    - Using the `other` keyword argument, the output is given directly

    Parameters
    ----------
    argument : str
        specify the name of a single argument from the function signature from which
        to take the other data object
    other : `numpy.ndarray` or `pint.Quantity` or `xarray.DataArray`
        specify the other data object directly
    match_unit : bool
        if True and other data object has units, convert output to those units
        (defaults to False)

    Notes
    -----
    This can be extended in the future to support:

    - ``units.wraps``-like behavior
    - Python scalars vs. NumPy scalars (Issue #1209)
    - dask (and other duck array) compatibility
    - dimensionality reduction (particularly with xarray)

    See Also
    --------
    preprocess_xarray

    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine other
            if 'other' in wrap_kwargs:
                other = wrap_kwargs['other']
            elif 'argument' in wrap_kwargs:
                other = signature(func).bind(*args, **kwargs).arguments[
                    wrap_kwargs['argument']]
            else:
                raise ValueError('Must specify keyword "other" or "argument".')

            # Get result from wrapped function
            result = func(*args, **kwargs)

            # Proceed with wrapping rules
            if wrap_kwargs.get('match_unit', False):
                return _wrap_output_like_matching_units(result, other)
            else:
                return _wrap_output_like_not_matching_units(result, other)

        return wrapper
    return decorator


def _wrap_output_like_matching_units(result, match):
    """Convert result to be like match with matching units for output wrapper."""
    if isinstance(match, xr.DataArray):
        output_xarray = True
        match_units = match.metpy.units
    elif isinstance(match, units.Quantity):
        output_xarray = False
        match_units = match.units
    else:
        output_xarray = False
        match_units = ''

    if isinstance(result, xr.DataArray):
        result = result.metpy.convert_units(match_units)
        return result.metpy.quantify() if output_xarray else result.metpy.unit_array
    else:
        result = result.m_as(match_units) if isinstance(result, units.Quantity) else result
        if output_xarray:
            return xr.DataArray(
                units.Quantity(result, match_units),
                dims=match.dims,
                coords=match.coords
            )
        else:
            return units.Quantity(result, match_units)


def _wrap_output_like_not_matching_units(result, match):
    """Convert result to be like match without matching units for output wrapper."""
    output_xarray = isinstance(match, xr.DataArray)
    if isinstance(result, xr.DataArray):
        return result.metpy.quantify() if output_xarray else result.metpy.unit_array
    else:
        if isinstance(result, units.Quantity):
            result_magnitude = result.magnitude
            result_units = str(result.units)
        else:
            result_magnitude = result
            result_units = ''

        if output_xarray:
            return xr.DataArray(
                units.Quantity(result_magnitude, result_units),
                dims=match.dims,
                coords=match.coords
            )
        else:
            return units.Quantity(result_magnitude, result_units)
