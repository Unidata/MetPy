# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Vendor core functionality used from xarray.

This code has been reproduced with modification under the terms of the Apache License, Version
2.0 (notice included below).

    Copyright 2014-2019, xarray Developers

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


def expanded_indexer(key, ndim):
    """Expand an indexer to a tuple with length ndim.

    Given a key for indexing an ndarray, return an equivalent key which is a
    tuple with length equal to the number of dimensions.
    The expansion is done by replacing all `Ellipsis` items with the right
    number of full slices and then padding the key with full slices so that it
    reaches the appropriate dimensionality.
    """
    if not isinstance(key, tuple):
        # numpy treats non-tuple keys equivalent to tuples of length 1
        key = (key,)
    new_key = []
    # handling Ellipsis right is a little tricky, see:
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
    found_ellipsis = False
    for k in key:
        if k is Ellipsis:
            if not found_ellipsis:
                new_key.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                new_key.append(slice(None))
        else:
            new_key.append(k)
    if len(new_key) > ndim:
        raise IndexError('too many indices')
    new_key.extend((ndim - len(new_key)) * [slice(None)])
    return tuple(new_key)


def is_dict_like(value):
    """Check if value is dict-like."""
    return hasattr(value, 'keys') and hasattr(value, '__getitem__')


def either_dict_or_kwargs(pos_kwargs, kw_kwargs, func_name):
    """Ensure dict-like argument from either positional or keyword arguments."""
    if pos_kwargs is not None:
        if not is_dict_like(pos_kwargs):
            raise ValueError('the first argument to .{} must be a '
                             'dictionary'.format(func_name))
        if kw_kwargs:
            raise ValueError('cannot specify both keyword and positional arguments to '
                             '.{}'.format(func_name))
        return pos_kwargs
    else:
        return kw_kwargs
