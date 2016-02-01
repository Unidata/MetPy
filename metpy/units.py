# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r'''Module to provide unit support.

This makes use of the :mod:`pint` library and sets up the default settings
for good temperature support.

Attributes
----------
units : :class:`pint.UnitRegistry`
    The unit registry used throughout the package. Any use of units in MetPy should
    import this registry and use it to grab units.
'''

from __future__ import division
import pint
import pint.unit
import numpy as np

units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

# For pint 0.6, this is the best way to define a dimensionless unit. See pint #185
units.define(pint.unit.UnitDefinition('percent', '%', (), pint.unit.ScaleConverter(0.01)))


def concatenate(arrs, axis=0):
    r'''Concatenate multiple values into a new unitized object.

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
    '''

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
    r'''Convert inputs to arrays with at least one dimension

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
    '''

    mags = [a.magnitude for a in arrs]
    orig_units = [a.units for a in arrs]
    ret = np.atleast_1d(*mags)
    if len(mags) == 1:
        return units.Quantity(ret, orig_units[0])
    return [units.Quantity(m, u) for m, u in zip(ret, orig_units)]


def atleast_2d(*arrs):
    r'''Convert inputs to arrays with at least two dimensions

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
    '''

    mags = [a.magnitude for a in arrs]
    orig_units = [a.units for a in arrs]
    ret = np.atleast_2d(*mags)
    if len(mags) == 1:
        return units.Quantity(ret, orig_units[0])
    return [units.Quantity(m, u) for m, u in zip(ret, orig_units)]

del pint
