# Copyright (c) 2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `wx_symbols` module."""
from metpy.plots import current_weather


def test_mapper():
    """Test for symbol mapping functionality."""
    assert current_weather(0) == ''
    assert current_weather(4) == u'\ue9a2'
    assert current_weather(7) == u'\ue9a5'
    assert current_weather(65) == u'\ue9e1'


def test_alt_char():
    """Test alternate character functionality for mapper."""
    assert current_weather.alt_char(7, 1) == u'\ue9a6'
    assert current_weather.alt_char(7, 2) == u'\ue9a7'


def test_mapper_len():
    """Test getting the length of the mapper."""
    assert len(current_weather) == 100
