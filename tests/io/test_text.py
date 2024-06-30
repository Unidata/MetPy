# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test text handling functions."""
from datetime import datetime

import numpy as np

from metpy.cbook import get_test_data
from metpy.io import parse_wpc_surface_bulletin
from metpy.testing import needs_module


@needs_module('shapely')
def test_parse_wpc_surface_bulletin_highres():
    """Test parser reading a high res WPC coded surface bulletin into a dataframe."""
    # Get rows 17 and 47 from dataframe representing parsed text file
    # Row 17 is a pressure center and row 47 is front
    import shapely.geometry as sgeom

    input_file = get_test_data('WPC_sfc_fronts_20210628_1800.txt')
    df = parse_wpc_surface_bulletin(input_file)
    assert len(df) == 89
    assert len(df[df.feature == 'HIGH']) == 16
    assert len(df[df.feature == 'LOW']) == 24
    assert len(df[df.feature == 'TROF']) == 22

    assert df.feature[17] == 'LOW'
    assert df.strength[17] == 1002.0
    assert df.geometry[17] == sgeom.Point([-114.5, 34.4])

    assert df.feature[47] == 'STNRY'
    assert np.isnan(df.strength[47])
    assert df.geometry[47] == sgeom.LineString([[-100.5, 32.4], [-101.0, 31.9],
                                                [-101.9, 31.5], [-102.9, 31.2]])

    assert all(df.valid == datetime(2021, 6, 28, 18, 0, 0))


@needs_module('shapely')
def test_parse_wpc_surface_bulletin():
    """Test parser reading a low res WPC coded surface bulletin into a dataframe."""
    # Get rows 17 and 47 from dataframe representing parsed text file
    # Row 17 is a pressure center and row 47 is front
    import shapely.geometry as sgeom

    input_file = get_test_data('WPC_sfc_fronts_lowres_20210628_1800.txt')
    df = parse_wpc_surface_bulletin(input_file)
    assert len(df) == 89
    assert len(df[df.feature == 'HIGH']) == 16
    assert len(df[df.feature == 'LOW']) == 24
    assert len(df[df.feature == 'TROF']) == 22

    assert df.feature[17] == 'LOW'
    assert df.strength[17] == 1002.0
    assert df.geometry[17] == sgeom.Point([-115, 34])

    assert df.feature[47] == 'STNRY'
    assert df.strength[47] == 'WK'
    assert df.geometry[47] == sgeom.LineString([[-100, 32], [-101, 32],
                                                [-102, 32], [-103, 31]])

    assert all(df.valid == datetime(2021, 6, 28, 18, 0, 0))


@needs_module('shapely')
def test_negative_lat_highres():
    """Test decoding of high res coordinates with negative latitude."""
    from io import BytesIO

    import shapely.geometry as sgeom

    sample = BytesIO(b"""
178
ASUS02 KWBC 281800
CODSUS

CODED SURFACE FRONTAL POSITIONS
NWS WEATHER PREDICTION CENTER COLLEGE PARK MD
342 PM EDT MON JUN 28 2021

VALID 062818Z
HIGHS 1022 -3961069 1020 -3851069 1026 3750773 1022 4430845 1019 5520728 1018
""")
    df = parse_wpc_surface_bulletin(sample)
    assert df.geometry[0] == sgeom.Point([-106.9, -39.6])


@needs_module('shapely')
def test_negative_lat():
    """Test decoding of coordinates with negative latitude."""
    from io import BytesIO

    import shapely.geometry as sgeom

    sample = BytesIO(b"""12HR PROG VALID xxxxxxZ
HIGHS -351 -3985 -4046 -38117 -7510
 """)
    df = parse_wpc_surface_bulletin(sample)
    assert df.geometry[0] == sgeom.Point([-51, -3])
