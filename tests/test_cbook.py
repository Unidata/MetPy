# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test functionality of MetPy's utility code."""

from metpy.cbook import example_data, Registry


def test_registry():
    """Test that the registry properly registers things."""
    reg = Registry()

    a = 'foo'
    reg.register('mine')(a)

    assert reg['mine'] is a


def test_example_data():
    """Test that the example data has the proper keys."""
    ds = example_data()
    var_names = list(ds.variables)

    assert 'temperature' in var_names
