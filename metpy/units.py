r'''Module to enable unit support.

This makes use of the `pint` library and sets up the default settings
for good temperature support.

Attributes
----------
units : `pint.UnitRegistry`
    The unit registry used throughout the package. Any use of units in MetPy should
    import this registry and use it to grab units.
'''

import pint
import numpy as np

units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


def concatenate(*args):
    r'''Concatenate multiple values into a new unitized object.

    This is essentially a unit-aware version of `numpy.concatenate`. All items
    must be able to be converted to the same units. If an item has no units, it will be given
    those of the rest of the collection, without conversion. The first units found in the
    arguments is used as the final output units.

    Parameters
    ----------
    args : arbitrary positional argument
         The items to be joined together

    Returns
    -------
    `pint.Quantity`
        New container with the value passed in and units corresponding to the first item.
    '''
    for a in args:
        if hasattr(a, 'units'):
            dest = a.units
            break

    data = []
    for a in args:
        if hasattr(a, 'to'):
            a = a.to(dest).magnitude
        data.append(np.atleast_1d(a))

    return units.Quantity(np.concatenate(data), dest)

del pint
