# Copyright (c) 2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `wx_symbols` module."""
import numpy as np

from metpy.plots import current_weather, wx_code_to_numeric
from metpy.testing import assert_array_equal


def test_mapper():
    """Test for symbol mapping functionality."""
    assert current_weather(0) == ''
    assert current_weather(4) == '\ue9a2'
    assert current_weather(7) == '\ue9a5'
    assert current_weather(65) == '\ue9e1'


def test_alt_char():
    """Test alternate character functionality for mapper."""
    assert current_weather.alt_char(7, 1) == '\ue9a6'
    assert current_weather.alt_char(7, 2) == '\ue9a7'


def test_mapper_len():
    """Test getting the length of the mapper."""
    assert len(current_weather) == 100


def test_wx_code_to_numeric():
    """Test getting numeric weather codes from METAR."""
    data = ['SN', '-RA', '-SHSN', '-SHRA', 'DZ', 'RA', 'SHSN', 'TSRA', '-FZRA', '-SN', '-TSRA',
            '-RASN', '+SN', 'FG', '-SHRASN', '-DZ', 'SHRA', '-FZRASN', 'TSSN', 'MIBCFG',
            '-RAPL', 'RAPL', 'TSSNPL', '-SNPL', '+RA', '-RASNPL', '-BLSN', '-SHSNIC', '+TSRA',
            'TS', 'PL', 'SNPL', '-SHRAPL', '-SNSG', '-TSSN', 'SG', 'IC', 'FU', '+SNPL',
            'TSSNPLGR', '-TSSNPLGR', '-SHSNSG', 'SHRAPL', '-TSRASN', 'FZRA', '-TSRAPL',
            '-FZDZSN', '+TSSN', '-TSRASNPL', 'TSRAPL', 'RASN', '-SNIC', 'FZRAPL', '-FZRASNPL',
            '+RAPL', '-RASGPL', '-TSSNPL', 'FZRASN', '+TSSNGR', 'TSPLGR', '', 'RA BR', '-TSSG',
            '-TS', '-NA', 'NANA', '+NANA', 'NANANA', 'NANANANA']
    true_codes = np.array([73, 61, 85, 80, 53, 63, 86, 95, 66, 71, 95, 68, 75, 45, 83, 51, 81,
                           66, 95, 0, 79, 79, 95, 79, 65, 79, 36, 85, 97, 17, 79, 79, 80, 77,
                           95, 77, 78, 4, 79, 95, 95, 85, 81, 95, 67, 95, 56, 97, 95, 95, 69,
                           71, 79, 66, 79, 61, 95, 67, 97, 95, 0, 63, 17, 17, 0, 0, 0, 0, 0])
    wx_codes = wx_code_to_numeric(data)
    assert_array_equal(wx_codes, true_codes)
