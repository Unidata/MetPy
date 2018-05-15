# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test functionality of MetPy's utility code."""

from metpy.cbook import Registry


def test_registry():
    """Test that the registry properly registers things."""
    reg = Registry()

    a = 'foo'
    reg.register('mine')(a)

    assert reg['mine'] is a
