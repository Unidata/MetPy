# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for custom Flake8 plugin for MetPy."""

import ast

import pytest

from flake8_metpy import MetPyChecker


@pytest.mark.parametrize('source, errs', [
    ('5 * pressure.units', 1),
    ('pw = -1. * (np.trapz(w.magnitude, pressure.magnitude) * (w.units * pressure.units))', 1),
    ("""def foo():
    return ret * moist_adiabat_temperatures.units""", 1),
    ('p_interp = np.sort(np.append(p_interp.m, top_pressure.m)) * pressure.units', 1),
    ('parameter = data[ob_type][subset].values * units(self.data.units[ob_type])', 1),
    ('np.nan * pressure.units', 1),
    ('np.array([1, 2, 3]) * units.m', 1),
    ('np.arange(4) * units.s', 1),
    ('np.ma.array([1, 2, 3]) * units.hPa', 1)
])
def test_plugin(source, errs):
    """Test that the flake8 checker works correctly."""
    checker = MetPyChecker(ast.parse(source))
    assert len(list(checker.run())) == errs
