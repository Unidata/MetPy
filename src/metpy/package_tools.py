# Copyright (c) 2015,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Collection of tools for managing the package."""

# Used to specify functions that should be exported--i.e. added to __all__
# Inspired by David Beazley and taken from python-ideas:
# https://mail.python.org/pipermail/python-ideas/2014-May/027824.html

__all__ = ('Exporter',)


class Exporter:
    """Manages exporting of symbols from the module.

    Grabs a reference to `globals()` for a module and provides a decorator to add
    functions and classes to `__all__` rather than requiring a separately maintained list.
    Also provides a context manager to do this for instances by adding all instances added
    within a block to `__all__`.
    """

    def __init__(self, globls):
        """Initialize the Exporter."""
        self.globls = globls
        self.exports = globls.setdefault('__all__', [])

    def export(self, defn):
        """Declare a function or class as exported."""
        self.exports.append(defn.__name__)
        return defn

    def __enter__(self):
        """Start a block tracking all instances created at global scope."""
        self.start_vars = set(self.globls)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the instance tracking block."""
        self.exports.extend(set(self.globls) - self.start_vars)
        del self.start_vars


def set_module(globls):
    """Set the module for all functions in ``__all__``.

    This sets the ``__module__`` attribute of all items within the ``__all__`` list
    for the calling module.

    This supports our hoisting of functions out of individual modules, which are
    considered implementation details, into the namespace of the top-level subpackage.

    Parameters
    ----------
    globls : Dict[str, object]
        Mapping of all global variables for the module. This contains all needed
        python special ("dunder") variables needed to be modified.

    """
    for item in globls['__all__']:
        obj = globls[item]
        if hasattr(obj, '__module__'):
            obj.__module__ = globls['__name__']
