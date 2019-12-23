#  Copyright (c) 2019 MetPy Developers.
#  Distributed under the terms of the BSD 3-Clause License.
#  SPDX-License-Identifier: BSD-3-Clause
"""Test MetPy's testing utilities."""

import numpy as np
import pytest

from metpy.testing import assert_array_almost_equal


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
