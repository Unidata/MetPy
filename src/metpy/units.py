# Copyright (c) 2015,2017,2019 MetPy Developers.
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
import functools
from inspect import Parameter, signature
import logging
import re
import warnings

import numpy as np
import pint
import pint.unit

log = logging.getLogger(__name__)

UndefinedUnitError = pint.UndefinedUnitError
DimensionalityError = pint.DimensionalityError

# Create registry, with preprocessors for UDUNITS-style powers (m2 s-2) and percent signs
units = pint.UnitRegistry(
    autoconvert_offset_to_baseunit=True,
    preprocessors=[
        functools.partial(
            re.sub,
            r'(?<=[A-Za-z])(?![A-Za-z])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])',
            '**'
        ),
        lambda string: string.replace('%', 'percent')
    ]
)

# Capture v0.10 NEP 18 warning on first creation
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    units.Quantity([])

# For pint 0.6, this is the best way to define a dimensionless unit. See pint #185
units.define(pint.unit.UnitDefinition('percent', '%', (),
             pint.converters.ScaleConverter(0.01)))

# Define commonly encountered units not defined by pint
units.define('degrees_north = degree = degrees_N = degreesN = degree_north = degree_N '
             '= degreeN')
units.define('degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE')

# Alias geopotential meters (gpm) to just meters
units.define('@alias meter = gpm')

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
                             'dataframe and col_units not given.') from None

    # Iterate through columns attaching units if we have them, if not, don't touch it
    res = {}
    for column in df:
        if column in column_units and column_units[column]:
            res[column] = units.Quantity(df[column].values, column_units[column])
        else:
            res[column] = df[column].values
    return res


def concatenate(arrs, axis=0):
    r"""Concatenate multiple values into a new unitized object.

    This is essentially a scalar-/masked array-aware version of `numpy.concatenate`. All items
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


def masked_array(data, data_units=None, **kwargs):
    """Create a :class:`numpy.ma.MaskedArray` with units attached.

    This is a thin wrapper around :func:`numpy.ma.masked_array` that ensures that
    units are properly attached to the result (otherwise units are silently lost). Units
    are taken from the ``data_units`` argument, or if this is ``None``, the units on ``data``
    are used.

    Parameters
    ----------
    data : array_like
        The source data. If ``data_units`` is `None`, this should be a `pint.Quantity` with
        the desired units.
    data_units : str or `pint.Unit`, optional
        The units for the resulting `pint.Quantity`
    kwargs
        Arbitrary keyword arguments passed to `numpy.ma.masked_array`, optional

    Returns
    -------
    `pint.Quantity`

    """
    if data_units is None:
        data_units = data.units
    return units.Quantity(np.ma.masked_array(data, **kwargs), data_units)


def _check_argument_units(args, defaults, dimensionality):
    """Yield arguments with improper dimensionality."""
    for arg, val in args.items():
        # Get the needed dimensionality (for printing) as well as cached, parsed version
        # for this argument.
        try:
            need, parsed = dimensionality[arg]
        except KeyError:
            # Argument did not have units specified in decorator
            continue

        if arg in defaults and (defaults[arg] is not None or val is None):
            check = val == defaults[arg]
            if np.all(check):
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


def _get_changed_version(docstring):
    """Find the most recent version in which the docs say a function changed."""
    matches = re.findall(r'.. versionchanged:: ([\d.]+)', docstring)
    return max(matches) if matches else None


def check_units(*units_by_pos, **units_by_name):
    """Create a decorator to check units of function arguments."""
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

        defaults = {name: sig.parameters[name].default for name in sig.parameters
                    if sig.parameters[name].default is not Parameter.empty}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Match all passed in value to their proper arguments so we can check units
            bound_args = sig.bind(*args, **kwargs)
            bad = list(_check_argument_units(bound_args.arguments, defaults, dims))

            # If there are any bad units, emit a proper error message making it clear
            # what went wrong.
            if bad:
                msg = f'`{func.__name__}` given arguments with incorrect units: '
                msg += ', '.join(f'`{arg}` requires "{req}" but given "{given}"'
                                 for arg, given, req in bad)
                if 'none' in msg:
                    msg += ('\nAny variable `x` can be assigned a unit as follows:\n'
                            '    from metpy.units import units\n'
                            '    x = units.Quantity(x, "m/s")')

                # If function has changed, mention that fact
                if func.__doc__:
                    changed_version = _get_changed_version(func.__doc__)
                    if changed_version:
                        msg = (f'This function changed in {changed_version}--double check '
                               'that the function is being called properly.\n') + msg
                raise ValueError(msg)
            return func(*args, **kwargs)

        return wrapper
    return dec


# Enable pint's built-in matplotlib support
units.setup_matplotlib()

del pint
