# Copyright (c) 2023 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains helpers for working with warnings."""
import inspect
import warnings


def _find_stack_level():
    """Find and return the stack level where we're outside the library."""
    import metpy

    frame = inspect.currentframe()
    n = 0
    while frame:
        if inspect.getfile(frame).startswith(metpy.__path__[0]):
            n += 1
            frame = frame.f_back
        else:
            break
    return n


def warn(*args, **kwargs):
    """Wrap `warnings.warn` and automatically set the stack level if not given."""
    level = kwargs.get('stacklevel')
    if level is None:
        level = _find_stack_level()
    warnings.warn(*args, **kwargs, stacklevel=level)
