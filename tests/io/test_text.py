# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test text handling functions."""
import geopandas as gpd
import numpy as np

from metpy.cbook import get_test_data
from metpy.io import parse_wpc_sfc_bulletin


def test_parse_wpc_sfc_bulletin():
    """Test parser reading a WPC coded surface bulletin into a geodataframe."""
    # Get rows 17 and 47 from dataframe representing parsed text file
    # Row 17 is a pressure center and row 47 is front
    input_file = get_test_data('WPC_sfc_fronts_20210628_1800.txt')
    df = parse_wpc_sfc_bulletin(input_file).loc[[17, 47]].reset_index(drop=True)

    # Expected values for rows 17 and 47 in the dataframe
    expected_json = ('{"type": "FeatureCollection", "features": [{"id": "17", "type": "Feature'
                     '", "properties": {"feature": "LOW", "strength": "1012", "valid": "062818'
                     'Z"}, "geometry": {"type": "Point", "coordinates": [-100.4, 32.6]}}, {"id'
                     '": "47", "type": "Feature", "properties": {"feature": "COLD", "strength"'
                     ': null, "valid": "062818Z"}, "geometry": {"type": "LineString", "coordin'
                     'ates": [[-83.7, 52.8], [-87.1, 50.9], [-89.0, 50.4], [-91.9, 49.8]]}}]}')
    expected_df = gpd.read_file(expected_json)[['valid', 'feature', 'strength', 'geometry']]

    # Align missing values across both dataframes
    df.replace({np.nan: 'na'}, inplace=True)
    expected_df.replace({None: 'na'}, inplace=True)

    assert df.equals(expected_df)
