# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Interpolate data along a single axis."""
from __future__ import absolute_import, division

import warnings

import numpy as np

from ..cbook import broadcast_indices
from ..package_tools import Exporter
from ..xarray import preprocess_xarray

exporter = Exporter(globals())


@exporter.export
@preprocess_xarray
def interpolate_nans_1d(x, y, kind='linear'):
    """Interpolate NaN values in y.

    Interpolate NaN values in the y dimension. Works with unsorted x values.

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric x-values
    y : array-like
        1-dimensional array of numeric y-values
    kind : string
        specifies the kind of interpolation x coordinate - 'linear' or 'log', optional.
        Defaults to 'linear'.

    Returns
    -------
        An array of the y coordinate data with NaN values interpolated.

    """
    x_sort_args = np.argsort(x)
    x = x[x_sort_args]
    y = y[x_sort_args]
    nans = np.isnan(y)
    if kind == 'linear':
        y[nans] = np.interp(x[nans], x[~nans], y[~nans])
    elif kind == 'log':
        y[nans] = np.interp(np.log(x[nans]), np.log(x[~nans]), y[~nans])
    else:
        raise ValueError('Unknown option for kind: {0}'.format(str(kind)))
    return y[x_sort_args]


@exporter.export
@preprocess_xarray
def interpolate_1d(x, xp, *args, **kwargs):
    r"""Interpolates data with any shape over a specified axis.

    Interpolation over a specified axis for arrays of any shape.

    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.

    xp : array-like
        The x-coordinates of the data points.

    args : array-like
        The data to be interpolated. Can be multiple arguments, all must be the same shape as
        xp.

    axis : int, optional
        The axis to interpolate over. Defaults to 0.

    fill_value: float, optional
        Specify handling of interpolation points out of data bounds. If None, will return
        ValueError if points are out of bounds. Defaults to nan.

    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in ascending order.

    Examples
    --------
     >>> x = np.array([1., 2., 3., 4.])
     >>> y = np.array([1., 2., 3., 4.])
     >>> x_interp = np.array([2.5, 3.5])
     >>> metpy.calc.interp(x_interp, x, y)
     array([2.5, 3.5])

    Notes
    -----
    xp and args must be the same shape.

    """
    # Pull out keyword args
    fill_value = kwargs.pop('fill_value', np.nan)
    axis = kwargs.pop('axis', 0)

    # Handle units
    x, xp = _strip_matching_units(x, xp)

    # Make x an array
    x = np.asanyarray(x).reshape(-1)

    # Save number of dimensions in xp
    ndim = xp.ndim

    # Sort input data
    sort_args = np.argsort(xp, axis=axis)
    sort_x = np.argsort(x)

    # indices for sorting
    sorter = broadcast_indices(xp, sort_args, ndim, axis)

    # sort xp
    xp = xp[sorter]
    # Ensure pressure in increasing order
    variables = [arr[sorter] for arr in args]

    # Make x broadcast with xp
    x_array = x[sort_x]
    expand = [np.newaxis] * ndim
    expand[axis] = slice(None)
    x_array = x_array[tuple(expand)]

    # Calculate value above interpolated value
    minv = np.apply_along_axis(np.searchsorted, axis, xp, x[sort_x])
    minv2 = np.copy(minv)

    # If fill_value is none and data is out of bounds, raise value error
    if ((np.max(minv) == xp.shape[axis]) or (np.min(minv) == 0)) and fill_value is None:
        raise ValueError('Interpolation point out of data bounds encountered')

    # Warn if interpolated values are outside data bounds, will make these the values
    # at end of data range.
    if np.max(minv) == xp.shape[axis]:
        warnings.warn('Interpolation point out of data bounds encountered')
        minv2[minv == xp.shape[axis]] = xp.shape[axis] - 1
    if np.min(minv) == 0:
        minv2[minv == 0] = 1

    # Get indices for broadcasting arrays
    above = broadcast_indices(xp, minv2, ndim, axis)
    below = broadcast_indices(xp, minv2 - 1, ndim, axis)

    if np.any(x_array < xp[below]):
        warnings.warn('Interpolation point out of data bounds encountered')

    # Create empty output list
    ret = []

    # Calculate interpolation for each variable
    for var in variables:
        # Var needs to be on the *left* of the multiply to ensure that if it's a pint
        # Quantity, it gets to control the operation--at least until we make sure
        # masked arrays and pint play together better. See https://github.com/hgrecco/pint#633
        var_interp = var[below] + (var[above] - var[below]) * ((x_array - xp[below])
                                                               / (xp[above] - xp[below]))

        # Set points out of bounds to fill value.
        var_interp[minv == xp.shape[axis]] = fill_value
        var_interp[x_array < xp[below]] = fill_value

        # Check for input points in decreasing order and return output to match.
        if x[0] > x[-1]:
            var_interp = np.swapaxes(np.swapaxes(var_interp, 0, axis)[::-1], 0, axis)
        # Output to list
        ret.append(var_interp)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


@exporter.export
@preprocess_xarray
def log_interpolate_1d(x, xp, *args, **kwargs):
    r"""Interpolates data with logarithmic x-scale over a specified axis.

    Interpolation on a logarithmic x-scale for interpolation values in pressure coordintates.

    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.

    xp : array-like
        The x-coordinates of the data points.

    args : array-like
        The data to be interpolated. Can be multiple arguments, all must be the same shape as
        xp.

    axis : int, optional
        The axis to interpolate over. Defaults to 0.

    fill_value: float, optional
        Specify handling of interpolation points out of data bounds. If None, will return
        ValueError if points are out of bounds. Defaults to nan.

    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in ascending order.

    Examples
    --------
     >>> x_log = np.array([1e3, 1e4, 1e5, 1e6])
     >>> y_log = np.log(x_log) * 2 + 3
     >>> x_interp = np.array([5e3, 5e4, 5e5])
     >>> metpy.calc.log_interp(x_interp, x_log, y_log)
     array([20.03438638, 24.63955657, 29.24472675])

    Notes
    -----
    xp and args must be the same shape.

    """
    # Pull out kwargs
    fill_value = kwargs.pop('fill_value', np.nan)
    axis = kwargs.pop('axis', 0)

    # Handle units
    x, xp = _strip_matching_units(x, xp)

    # Log x and xp
    log_x = np.log(x)
    log_xp = np.log(xp)
    return interpolate_1d(log_x, log_xp, *args, axis=axis, fill_value=fill_value)


def _strip_matching_units(*args):
    """Ensure arguments have same units and return with units stripped.

    Replaces `@units.wraps(None, ('=A', '=A'))`, which breaks with `*args` handling for
    pint>=0.9.
    """
    if all(hasattr(arr, 'units') for arr in args):
        return [arr.to(args[0].units).magnitude for arr in args]
    else:
        return args
