# Copyright (c) 2008,2015,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Collection of generally useful utility code from the cookbook."""

import os

import numpy as np
import pooch

from . import __version__

try:
    string_type = basestring
except NameError:
    string_type = str


# TODO: This can go away when we remove Python 2
def is_string_like(s):
    """Check if an object is a string."""
    return isinstance(s, string_type)


POOCH = pooch.create(
    path=pooch.os_cache('metpy'),
    base_url='https://github.com/Unidata/MetPy/raw/{version}/staticdata/',
    version='v' + __version__,
    version_dev='master',
    env='TEST_DATA_DIR')

# Check if we're running from a git clone and if so, bash the path attribute with the path
# to git's local data store (un-versioned)
# Look for the staticdata directory (i.e. this is a git checkout)
if os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'staticdata')):
    POOCH.path = os.path.join(os.path.dirname(__file__), '..', 'staticdata')

POOCH.load_registry(os.path.join(os.path.dirname(__file__), 'static-data-manifest.txt'))


def get_test_data(fname, as_file_obj=True):
    """Access a file from MetPy's collection of test data."""
    path = POOCH.fetch(fname)
    # If we want a file object, open it, trying to guess whether this should be binary mode
    # or not
    if as_file_obj:
        return open(path, 'rb')

    return path


class Registry(object):
    """Provide a generic function registry.

    This provides a class to instantiate, which then has a `register` method that can
    be used as a decorator on functions to register them under a particular name.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._registry = {}

    def register(self, name):
        """Register a callable with the registry under a particular name.

        Parameters
        ----------
        name : str
            The name under which to register a function

        Returns
        -------
        dec : callable
            A decorator that takes a function and will register it under the name.

        """
        def dec(func):
            self._registry[name] = func
            return func
        return dec

    def __getitem__(self, name):
        """Return any callable registered under name."""
        return self._registry[name]


def broadcast_indices(x, minv, ndim, axis):
    """Calculate index values to properly broadcast index array within data array.

    See usage in interp.
    """
    ret = []
    for dim in range(ndim):
        if dim == axis:
            ret.append(minv)
        else:
            broadcast_slice = [np.newaxis] * ndim
            broadcast_slice[dim] = slice(None)
            dim_inds = np.arange(x.shape[dim])
            ret.append(dim_inds[tuple(broadcast_slice)])
    return tuple(ret)


def iterable(value):
    """Determine if value can be iterated over."""
    # Special case for pint Quantities
    if hasattr(value, 'magnitude'):
        value = value.magnitude
    return np.iterable(value)


__all__ = ('Registry', 'broadcast_indices', 'get_test_data', 'is_string_like', 'iterable')
