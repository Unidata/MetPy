# Copyright (c) 2008,2015,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Collection of generally useful utility code from the cookbook."""

import os
from pathlib import Path

import numpy as np
import pooch

from . import __version__

POOCH = pooch.create(
    path=pooch.os_cache('metpy'),
    base_url='https://github.com/Unidata/MetPy/raw/{version}/staticdata/',
    version='v' + __version__,
    version_dev='main')

# Check if we have the data available directly from a git checkout, either from the
# TEST_DATA_DIR variable, or looking relative to the path of this module's file. Use this
# to override Pooch's path.
dev_data_path = os.environ.get('TEST_DATA_DIR', Path(__file__).parents[2] / 'staticdata')
if Path(dev_data_path).exists():
    POOCH.path = dev_data_path

POOCH.load_registry(Path(__file__).parent / 'static-data-manifest.txt')


def get_test_data(fname, as_file_obj=True, mode='rb'):
    """Access a file from MetPy's collection of test data."""
    path = POOCH.fetch(fname)
    # If we want a file object, open it, trying to guess whether this should be binary mode
    # or not
    if as_file_obj:
        return open(path, mode)  # noqa: SIM115

    return path


class Registry:
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


__all__ = ('Registry', 'broadcast_indices', 'get_test_data')
