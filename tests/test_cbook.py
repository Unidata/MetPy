# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test functionality of MetPy's utility code."""

import numpy as np
import pytest
import xarray as xr

from metpy.cbook import Registry, result_type
from metpy.units import units


def test_registry():
    """Test that the registry properly registers things."""
    reg = Registry()

    a = 'foo'
    reg.register('mine')(a)

    assert reg['mine'] is a


@pytest.mark.parametrize(
    'test_input, expected_type_match, custom_dtype',
    [(1.0, 1.0, None),
     (1, 1, None),
     (np.array(1.0), 1.0, None),
     (np.array(1), 1, None),
     (np.array([1, 2, 3], dtype=np.int32), 1, 'int32'),
     (units.Quantity(1, units.m), 1, None),
     (units.Quantity(1.0, units.m), 1.0, None),
     (units.Quantity([1, 2.0], units.m), 1.0, None),
     ([1, 2, 3] * units.m, 1, None),
     (xr.DataArray(data=[1, 2.0]), 1.0, None)])
def test_result_type(test_input, expected_type_match, custom_dtype):
    """Test result_type on the kinds of things common in MetPy."""
    assert result_type(test_input) == np.array(expected_type_match, dtype=custom_dtype).dtype


def test_result_type_failure():
    """Test result_type failure on non-numeric types."""
    with pytest.raises(TypeError):
        result_type([False])
