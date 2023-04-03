# Copyright (c) 2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Interpolate data along a single axis."""
import warnings

import numpy as np

from ..cbook import broadcast_indices
from ..package_tools import Exporter
from ..xarray import preprocess_and_wrap

exporter = Exporter(globals())


@exporter.export
@preprocess_and_wrap()
def interpolate_nans_1d(x, y, kind='linear'):
    """Interpolate NaN values in y.

    Interpolate NaN values in the y dimension. Works with unsorted x values.

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric x-values
    y : array-like
        1-dimensional array of numeric y-values
    kind : str
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
        raise ValueError(f'Unknown option for kind: {kind}')
    return y[x_sort_args]


@exporter.export
@preprocess_and_wrap()
def interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan, return_list_always=False):
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

    return_list_always: bool, optional
        Whether to always return a list of interpolated arrays, even when only a single
        array is passed to `args`. Defaults to ``False``.

    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in ascending order.

    Examples
    --------
     >>> import metpy.interpolate
     >>> x = np.array([1., 2., 3., 4.])
     >>> y = np.array([1., 2., 3., 4.])
     >>> x_interp = np.array([2.5, 3.5])
     >>> metpy.interpolate.interpolate_1d(x_interp, x, y)
     array([2.5, 3.5])

    Notes
    -----
    xp and args must be the same shape.

    """
    # Handle units
    x, xp = _strip_matching_units(x, xp)

    # Make x an array
    x = np.asanyarray(x).reshape(-1)

    # Sort input data
    sort_args = np.argsort(xp, axis=axis)
    sort_x = np.argsort(x)

    # The shape after all arrays are broadcast to each other
    # Can't use broadcast_shapes until numpy >=1.20 is our minimum
    final_shape = np.broadcast(xp, *args).shape

    # indices for sorting
    sorter = broadcast_indices(sort_args, final_shape, axis)

    # sort xp -- need to make sure it's been manually broadcast due to our use of indices
    # along all axes.
    xp = np.broadcast_to(xp, final_shape)
    xp = xp[sorter]

    # Ensure source arrays are also in sorted order
    variables = [arr[sorter] for arr in args]

    # Make x broadcast with xp
    x_array = x[sort_x]
    expand = [np.newaxis] * len(final_shape)
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
    above = broadcast_indices(minv2, final_shape, axis)
    below = broadcast_indices(minv2 - 1, final_shape, axis)

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

    if return_list_always or len(ret) > 1:
        return ret
    else:
        return ret[0]


@exporter.export
@preprocess_and_wrap()
def log_interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan):
    r"""Interpolates data with logarithmic x-scale over a specified axis.

    Interpolation on a logarithmic x-scale for interpolation values in pressure coordinates.

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
     >>> metpy.interpolate.log_interpolate_1d(x_interp, x_log, y_log)
     array([20.03438638, 24.63955657, 29.24472675])

    Notes
    -----
    xp and args must be the same shape.

    """
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
