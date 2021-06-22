# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test packaging details."""

from metpy.calc import tke
from metpy.interpolate import interpolate_to_grid
from metpy.io import Level2File
from metpy.plots import StationPlot


def test_modules_set():
    """Test that functions from each subpackage have correct module set."""
    assert Level2File.__module__ == 'metpy.io'
    assert StationPlot.__module__ == 'metpy.plots'
    assert interpolate_to_grid.__module__ == 'metpy.interpolate'
    assert tke.__module__ == 'metpy.calc'
