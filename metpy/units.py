# Copyright (c) 2008-2016 MetPy Developers.
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

import matplotlib.units as munits
import numpy as np
import pint
import pint.unit

from .cbook import iterable

UndefinedUnitError = pint.UndefinedUnitError
DimensionalityError = pint.DimensionalityError

units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

# For pint 0.6, this is the best way to define a dimensionless unit. See pint #185
units.define(pint.unit.UnitDefinition('percent', '%', (),
             pint.converters.ScaleConverter(0.01)))


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

    return units.Quantity(np.concatenate(data, axis=axis), dest)


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
    mags = [a.magnitude for a in arrs]
    orig_units = [a.units for a in arrs]
    ret = np.atleast_1d(*mags)
    if len(mags) == 1:
        return units.Quantity(ret, orig_units[0])
    return [units.Quantity(m, u) for m, u in zip(ret, orig_units)]


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
    mags = [a.magnitude for a in arrs]
    orig_units = [a.units for a in arrs]
    ret = np.atleast_2d(*mags)
    if len(mags) == 1:
        return units.Quantity(ret, orig_units[0])
    return [units.Quantity(m, u) for m, u in zip(ret, orig_units)]


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


class PintConverter(munits.ConversionInterface):
    """Implement support for pint within matplotlib's unit conversion framework."""

    @staticmethod
    def convert(value, unit, axis):
        """Convert pint :`Quantity` instances for matplotlib to use.

        Currently only strips off the units to avoid matplotlib errors since we can't reliably
        have pint.Quantity instances not decay to numpy arrays.
        """
        if hasattr(value, 'magnitude'):
            return value.magnitude
        elif iterable(value):
            try:
                return [v.magnitude for v in value]
            except AttributeError:
                return value
        else:
            return value

    # TODO: Once we get things properly squared away between pint and everything else
    # these will need to be functional.
    # @staticmethod
    # def axisinfo(unit, axis):
    #     return None
    #
    # @staticmethod
    # def default_units(x, axis):
    #     return x.to_base_units()


# Register the class
munits.registry[units.Quantity] = PintConverter()

del munits, pint
