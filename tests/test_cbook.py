# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test functionality of MetPy's utility code."""

import pytest

from metpy.cbook import example_data, Registry, validate_choice


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


def test_validate_choice():
    """Test that validation is functioning and error is useful."""
    with pytest.raises(ValueError, match='is not a valid option'):
        validate_choice({'red', 'yellow', 'green'}, color='blue')
