# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test text handling functions."""
import geopandas as gpd
import numpy as np

from metpy.cbook import get_test_data
from metpy.io import parse_wpc_surface_bulletin
from metpy.testing import needs_geopandas


@needs_geopandas
def test_parse_wpc_surface_bulletin():
    """Test parser reading a WPC coded surface bulletin into a geodataframe."""
    # Get rows 17 and 47 from dataframe representing parsed text file
    # Row 17 is a pressure center and row 47 is front
    input_file = get_test_data('WPC_sfc_fronts_20210628_1800.txt')
    df = parse_wpc_surface_bulletin(input_file)
    assert len(df) == 89
    assert len(df[df.feature == 'HIGH']) == 16
    assert len(df[df.feature == 'LOW']) == 24
    assert len(df[df.feature == 'TROF']) == 22

    # Expected values for rows 17 and 47 in the dataframe
    expected_json = ('{"type": "FeatureCollection",'
                     ' "features": [{"id": "17", "type": "Feature",'
                     ' "properties": {"feature": "LOW", "strength": "1002.0",'
                     ' "valid": "062818Z"}, "geometry": {"type": "Point",'
                     ' "coordinates": [-114.5, 34.4]}}, {"id": "47", "type": "Feature",'
                     ' "properties": {"feature": "STNRY", "strength": null,'
                     ' "valid": "062818Z"}, "geometry": {"type": "LineString",'
                     ' "coordinates": [[-100.5, 32.4], [-101.0, 31.9],'
                     ' [-101.9, 31.5], [-102.9, 31.2]]}}]}')
    expected_df = gpd.read_file(expected_json)[['valid', 'feature', 'geometry']]

    # Align missing values across both dataframes
    subset = df.loc[[17, 47]].reset_index(drop=True)
    subset.replace({np.nan: 'na'}, inplace=True)
    expected_df.replace({None: 'na'}, inplace=True)

    assert subset[['valid', 'feature', 'geometry']].equals(expected_df)
    assert subset.strength[0] == 1002.0
