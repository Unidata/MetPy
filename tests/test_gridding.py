# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test use of the deprecated gridding module."""

import pytest

from metpy.deprecation import MetpyDeprecationWarning


def test_deprecation():
    """Test deprecation warning on import, and that public API functions are available."""
    with pytest.warns(MetpyDeprecationWarning):
        import metpy.gridding
        functions = ['interpolate', 'inverse_distance', 'natural_neighbor',
                     'remove_nan_observations', 'remove_observations_below_value',
                     'remove_repeat_coordinates']
        for function in functions:
            assert function in metpy.gridding.__all__
