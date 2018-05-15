# Copyright (c) 2008,2015,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Collection of generally useful utility code from the cookbook."""

import os
import os.path

from matplotlib.cbook import iterable


try:
    string_type = basestring
except NameError:
    string_type = str


# TODO: This can go away when we remove Python 2
def is_string_like(s):
    """Check if an object is a string."""
    return isinstance(s, string_type)


def get_test_data(fname, as_file_obj=True):
    """Access a file from MetPy's collection of test data."""
    # Look for an environment variable to point to the test data. If not, try looking at
    # the appropriate path relative to this file.
    data_dir = os.environ.get('TEST_DATA_DIR',
                              os.path.join(os.path.dirname(__file__), '..', 'staticdata'))

    # Assemble the path
    path = os.path.join(data_dir, fname)

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


__all__ = ('Registry', 'get_test_data', 'is_string_like', 'iterable')
