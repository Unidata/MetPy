# Copyright (c) 2015,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Module to provide unit support.

This makes use of the :mod:`pint` library and sets up the default settings
for good temperature support.

Attributes
----------
units : :class:`pint.UnitRegistry`
    The unit registry used throughout the package. Any use of units in MetPy should
    import this registry and use it to grab units.

"""

from __future__ import division

import functools
import logging
import warnings

import numpy as np
import pint
import pint.unit

log = logging.getLogger(__name__)

UndefinedUnitError = pint.UndefinedUnitError
DimensionalityError = pint.DimensionalityError

units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

# For pint 0.6, this is the best way to define a dimensionless unit. See pint #185
units.define(pint.unit.UnitDefinition('percent', '%', (),
             pint.converters.ScaleConverter(0.01)))

# Define commonly encountered units not defined by pint
units.define('degrees_north = degree = degrees_N = degreesN = degree_north = degree_N '
             '= degreeN')
units.define('degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE')

# Alias geopotential meters (gpm) to just meters
try:
    units._units['meter']._aliases = ('metre', 'gpm')
    units._units['gpm'] = units._units['meter']
except AttributeError:
    log.warning('Failed to add gpm alias to meters.')

# Silence UnitStrippedWarning
if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)


def pandas_dataframe_to_unit_arrays(df, column_units=None):
    """Attach units to data in pandas dataframes and return united arrays.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data in pandas dataframe.

    column_units : dict
        Dictionary of units to attach to columns of the dataframe. Overrides
        the units attribute if it is attached to the dataframe.

    Returns
    -------
        Dictionary containing united arrays with keys corresponding to the dataframe
        column names.

    """
    if not column_units:
        try:
            column_units = df.units
        except AttributeError:
            raise ValueError('No units attribute attached to pandas '
                             'dataframe and col_units not given.')

    # Iterate through columns attaching units if we have them, if not, don't touch it
    res = {}
    for column in df:
        if column in column_units and column_units[column]:
            res[column] = df[column].values * units(column_units[column])
        else:
            res[column] = df[column].values
    return res


def concatenate(arrs, axis=0):
    r"""Concatenate multiple values into a new unitized object.

    This is essentially a unit-aware version of `numpy.concatenate`. All items
    must be able to be converted to the same units. If an item has no units, it will be given
    those of the rest of the collection, without conversion. The first units found in the
    arguments is used as the final output units.

    Parameters
    ----------
    arrs : Sequence of arrays
        The items to be joined together

    axis : integer, optional
        The array axis along which to join the arrays. Defaults to 0 (the first dimension)

    Returns
    -------
    `pint.Quantity`
        New container with the value passed in and units corresponding to the first item.

    """
    dest = 'dimensionless'
    for a in arrs:
        if hasattr(a, 'units'):
            dest = a.units
            break

    data = []
    for a in arrs:
        if hasattr(a, 'to'):
            a = a.to(dest).magnitude
        data.append(np.atleast_1d(a))

    # Use masked array concatenate to ensure masks are preserved, but convert to an
    # array if there are no masked values.
    data = np.ma.concatenate(data, axis=axis)
    if not np.any(data.mask):
        data = np.asarray(data)

    return units.Quantity(data, dest)


def diff(x, **kwargs):
    """Calculate the n-th discrete difference along given axis.

    Wraps :func:`numpy.diff` to handle units.

    Parameters
    ----------
    x : array-like
        Input data
    n : int, optional
        The number of times values are differenced.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.

    Returns
    -------
    diff : ndarray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as that of the input.

    See Also
    --------
    numpy.diff

    """
    ret = np.diff(x, **kwargs)
    if hasattr(x, 'units'):
        # Can't just use units because of how things like temperature work
        it = x.flat
        true_units = (next(it) - next(it)).units
        ret = ret * true_units
    return ret


def atleast_1d(*arrs):
    r"""Convert inputs to arrays with at least one dimension.

    Scalars are converted to 1-dimensional arrays, whilst other
    higher-dimensional inputs are preserved. This is a thin wrapper
    around `numpy.atleast_1d` to preserve units.

    Parameters
    ----------
    arrs : arbitrary positional arguments
        Input arrays to be converted if necessary

    Returns
    -------
    `pint.Quantity`
        A single quantity or a list of quantities, matching the number of inputs.

    """
    mags = [a.magnitude if hasattr(a, 'magnitude') else a for a in arrs]
    orig_units = [a.units if hasattr(a, 'units') else None for a in arrs]
    ret = np.atleast_1d(*mags)
    if len(mags) == 1:
        if orig_units[0] is not None:
            return units.Quantity(ret, orig_units[0])
        else:
            return ret
    return [units.Quantity(m, u) if u is not None else m for m, u in zip(ret, orig_units)]


def atleast_2d(*arrs):
    r"""Convert inputs to arrays with at least two dimensions.

    Scalars and 1-dimensional arrays are converted to 2-dimensional arrays,
    whilst other higher-dimensional inputs are preserved. This is a thin wrapper
    around `numpy.atleast_2d` to preserve units.

    Parameters
    ----------
    arrs : arbitrary positional arguments
        Input arrays to be converted if necessary

    Returns
    -------
    `pint.Quantity`
        A single quantity or a list of quantities, matching the number of inputs.

    """
    mags = [a.magnitude if hasattr(a, 'magnitude') else a for a in arrs]
    orig_units = [a.units if hasattr(a, 'units') else None for a in arrs]
    ret = np.atleast_2d(*mags)
    if len(mags) == 1:
        if orig_units[0] is not None:
            return units.Quantity(ret, orig_units[0])
        else:
            return ret
    return [units.Quantity(m, u) if u is not None else m for m, u in zip(ret, orig_units)]


def masked_array(data, data_units=None, **kwargs):
    """Create a :class:`numpy.ma.MaskedArray` with units attached.

    This is a thin wrapper around :func:`numpy.ma.masked_array` that ensures that
    units are properly attached to the result (otherwise units are silently lost). Units
    are taken from the ``units`` argument, or if this is ``None``, the units on ``data``
    are used.

    Parameters
    ----------
    data : array_like
        The source data. If ``units`` is `None`, this should be a `pint.Quantity` with
        the desired units.
    data_units : str or `pint.Unit`
        The units for the resulting `pint.Quantity`
    **kwargs : Arbitrary keyword arguments passed to `numpy.ma.masked_array`

    Returns
    -------
    `pint.Quantity`

    """
    if data_units is None:
        data_units = data.units
    return units.Quantity(np.ma.masked_array(data, **kwargs), data_units)


def _check_argument_units(args, dimensionality):
    """Yield arguments with improper dimensionality."""
    for arg, val in args.items():
        # Get the needed dimensionality (for printing) as well as cached, parsed version
        # for this argument.
        try:
            need, parsed = dimensionality[arg]
        except KeyError:
            # Argument did not have units specified in decorator
            continue

        # See if the value passed in is appropriate
        try:
            if val.dimensionality != parsed:
                yield arg, val.units, need
        # No dimensionality
        except AttributeError:
            # If this argument is dimensionless, don't worry
            if parsed != '':
                yield arg, 'none', need


def check_units(*units_by_pos, **units_by_name):
    """Create a decorator to check units of function arguments."""
    try:
        from inspect import signature

        def dec(func):
            # Match the signature of the function to the arguments given to the decorator
            sig = signature(func)
            bound_units = sig.bind_partial(*units_by_pos, **units_by_name)

            # Convert our specified dimensionality (e.g. "[pressure]") to one used by
            # pint directly (e.g. "[mass] / [length] / [time]**2). This is for both efficiency
            # reasons and to ensure that problems with the decorator are caught at import,
            # rather than runtime.
            dims = {name: (orig, units.get_dimensionality(orig.replace('dimensionless', '')))
                    for name, orig in bound_units.arguments.items()}

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Match all passed in value to their proper arguments so we can check units
                bound_args = sig.bind(*args, **kwargs)
                bad = list(_check_argument_units(bound_args.arguments, dims))

                # If there are any bad units, emit a proper error message making it clear
                # what went wrong.
                if bad:
                    msg = '`{0}` given arguments with incorrect units: {1}.'.format(
                        func.__name__,
                        ', '.join('`{}` requires "{}" but given "{}"'.format(arg, req, given)
                                  for arg, given, req in bad))
                    if 'none' in msg:
                        msg += ('\nAny variable `x` can be assigned a unit as follows:\n'
                                '    from metpy.units import units\n'
                                '    x = x * units.meter / units.second')
                    raise ValueError(msg)
                return func(*args, **kwargs)

            return wrapper

    # signature() only available on Python >= 3.3, so for 2.7 we just do nothing.
    except ImportError:
        def dec(func):
            return func

    return dec


try:
    # Try to enable pint's built-in support
    units.setup_matplotlib()
except (AttributeError, RuntimeError, ImportError):  # Pint's not available, try our own
    import matplotlib.units as munits

    # Inheriting from object fixes the fact that matplotlib 1.4 doesn't
    # TODO: Remove object when we drop support for matplotlib 1.4
    class PintAxisInfo(munits.AxisInfo, object):
        """Support default axis and tick labeling and default limits."""

        def __init__(self, units):
            """Set the default label to the pretty-print of the unit."""
            super(PintAxisInfo, self).__init__(label='{:P}'.format(units))

    # TODO: Remove object when we drop support for matplotlib 1.4
    class PintConverter(munits.ConversionInterface, object):
        """Implement support for pint within matplotlib's unit conversion framework."""

        def __init__(self, registry):
            """Initialize converter for pint units."""
            super(PintConverter, self).__init__()
            self._reg = registry

        def convert(self, value, unit, axis):
            """Convert :`Quantity` instances for matplotlib to use."""
            if isinstance(value, (tuple, list)):
                return [self._convert_value(v, unit, axis) for v in value]
            else:
                return self._convert_value(value, unit, axis)

        def _convert_value(self, value, unit, axis):
            """Handle converting using attached unit or falling back to axis units."""
            if hasattr(value, 'units'):
                return value.to(unit).magnitude
            else:
                return self._reg.Quantity(value, axis.get_units()).to(unit).magnitude

        @staticmethod
        def axisinfo(unit, axis):
            """Return axis information for this particular unit."""
            return PintAxisInfo(unit)

        @staticmethod
        def default_units(x, axis):
            """Get the default unit to use for the given combination of unit and axis."""
            if isinstance(x, (tuple, list)):
                return getattr(x[0], 'units', 'dimensionless')
            else:
                return getattr(x, 'units', 'dimensionless')

    # Register the class
    munits.registry[units.Quantity] = PintConverter(units)
    del munits

del pint
