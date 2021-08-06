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
    assert current_weather(142) == '\ue967'
    assert current_weather(1007) == '\ue9a6'
    assert current_weather(2007) == '\ue9a7'


def test_alt_char():
    """Test alternate character functionality for mapper."""
    assert current_weather.alt_char(7, 1) == '\ue9a6'
    assert current_weather.alt_char(7, 2) == '\ue9a7'


def test_mapper_len():
    """Test getting the length of the mapper."""
    assert len(current_weather) == 150


def test_wx_code_to_numeric():
    """Test getting numeric weather codes from METAR."""
    data = ['SN', '-RA', '-SHSN', '-SHRA', 'DZ', 'RA', 'SHSN', 'TSRA', '-FZRA', '-SN', '-TSRA',
            '-RASN', '+SN', 'FG', '-SHRASN', '-DZ', 'SHRA', '-FZRASN', 'TSSN', 'MIBCFG',
            '-RAPL', 'RAPL', 'TSSNPL', '-SNPL', '+RA', '-RASNPL', '-BLSN', '-SHSNIC', '+TSRA',
            'TS', 'PL', 'SNPL', '-SHRAPL', '-SNSG', '-TSSN', 'SG', 'IC', 'FU', '+SNPL',
            'TSSNPLGR', '-TSSNPLGR', '-SHSNSG', 'SHRAPL', '-TSRASN', 'FZRA', '-TSRAPL',
            '-FZDZSN', '+TSSN', '-TSRASNPL', 'TSRAPL', 'RASN', '-SNIC', 'FZRAPL', '-FZRASNPL',
            '+RAPL', '-RASGPL', '-TSSNPL', 'FZRASN', '+TSSNGR', 'TSPLGR', '', 'RA BR', '-TSSG',
            '-TS', '-NA', 'NANA', '+NANA', 'NANANA', 'NANANANA', 'TSUP', '+UP']
    true_codes = np.array([73, 61, 85, 80, 53, 63, 86, 1095, 66, 71, 1095,
                           68, 75, 45, 83, 51, 81, 66, 2095, 0,
                           79, 79, 2095, 79, 65, 79, 38, 85, 1097,
                           17, 79, 79, 80, 77, 2095, 77, 78, 4, 79,
                           2095, 2095, 85, 81, 95, 67, 1095,
                           56, 2097, 95, 1095, 69, 71, 79, 66,
                           79, 61, 2095, 67, 2097, 2095, 0, 63, 17,
                           17, 0, 0, 0, 0, 0, 17, 142])
    wx_codes = wx_code_to_numeric(data)
    assert_array_equal(wx_codes, true_codes)
