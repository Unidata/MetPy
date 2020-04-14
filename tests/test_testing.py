#  Copyright (c) 2019 MetPy Developers.
#  Distributed under the terms of the BSD 3-Clause License.
#  SPDX-License-Identifier: BSD-3-Clause
"""Test MetPy's testing utilities."""
import warnings

import numpy as np
import pytest
import xarray as xr

from metpy.deprecation import MetpyDeprecationWarning
from metpy.testing import (assert_array_almost_equal, check_and_drop_units,
                           check_and_silence_deprecation)


# Test #1183: numpy.testing.assert_array* ignores any masked value, so work-around
def test_masked_arrays():
    """Test that we catch masked arrays with different masks."""
    with pytest.raises(AssertionError):
        assert_array_almost_equal(np.array([10, 20]),
                                  np.ma.array([10, np.nan], mask=[False, True]), 2)


def test_masked_and_no_mask():
    """Test that we can compare a masked array with no masked values and a regular array."""
    a = np.array([10, 20])
    b = np.ma.array([10, 20], mask=[False, False])
    assert_array_almost_equal(a, b)


@check_and_silence_deprecation
def test_deprecation_decorator():
    """Make sure the deprecation checker works."""
    warnings.warn('Testing warning.', MetpyDeprecationWarning)


def test_check_and_drop_units_with_dataarray():
    """Make sure check_and_drop_units functions properly with both arguments as DataArrays."""
    var_0 = xr.DataArray([[1, 2], [3, 4]], attrs={'units': 'cm'})
    var_1 = xr.DataArray([[0.01, 0.02], [0.03, 0.04]], attrs={'units': 'm'})
    actual, desired = check_and_drop_units(var_0, var_1)
    assert isinstance(actual, np.ndarray)
    assert isinstance(desired, np.ndarray)
    np.testing.assert_array_almost_equal(actual, desired)
