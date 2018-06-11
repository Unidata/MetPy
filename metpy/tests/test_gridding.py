# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test use of the deprecated gridding module."""

import pytest

from metpy.deprecation import MetpyDeprecationWarning


def test_deprecation():
    """Test deprecation warning on import, and that public API functions are available."""
    with pytest.warns(MetpyDeprecationWarning):
        from metpy.gridding import (interpolate, inverse_distance,  # noqa: F401
                                    natural_neighbor, remove_nan_observations,
                                    remove_observations_below_value, remove_repeat_coordinates)
