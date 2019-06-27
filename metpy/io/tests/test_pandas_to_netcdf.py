# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `pandas_to_netcdf` module."""

import logging
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from metpy.cbook import get_test_data
from metpy.io import dataframe_to_netcdf

# Turn off the warnings for tests
logging.getLogger('metpy.io.pandas_to_netcdf').setLevel(logging.CRITICAL)


@pytest.fixture
def test_df():
    """Create generic dataframe for testing."""
    return pd.DataFrame({
        'temperature': pd.Series([1, 2, 2, 3]), 'pressure': pd.Series([1, 2, 2, 3]),
        'latitude': pd.Series([4, 5, 6, 7]), 'longitude': pd.Series([1, 2, 3, 4]),
        'station_id': pd.Series(['KFNL', 'KDEN', 'KVPZ', 'KORD'])})


def test_dataframe_to_netcdf_basic(tmpdir):
    """Test dataframe conversion to netcdf."""
    df = pd.read_csv(get_test_data('station_data.txt'), usecols=[0, 1, 2, 3, 4, 5])
    df = df.rename(columns={'latitude[unit="degrees_north"]': 'latitude',
                            'longitude[unit="degrees_east"]': 'longitude',
                            'air_pressure_at_sea_level[unit="hectoPascal"]':
                                'mean_sea_level_pressure',
                            'air_temperature[unit="Celsius"]': 'temperature'})
    dataframe_to_netcdf(df, path_to_save=str(tmpdir) + '/test.nc', sampling_var='station',
                        sampling_data_vars=['station', 'latitude', 'longitude'])
    assert os.path.exists(str(tmpdir) + '/test.nc')
    data = xr.open_dataset(str(tmpdir) + '/test.nc')
    assert np.max(data['temperature']) == 27


def test_dataframe_to_netcdf_units(tmpdir):
    """Test units attached via a dictionary."""
    df = pd.read_csv(get_test_data('station_data.txt'), usecols=[0, 1, 2, 3, 4, 5])
    df = df.rename(columns={'latitude[unit="degrees_north"]': 'latitude',
                            'longitude[unit="degrees_east"]': 'longitude',
                            'air_pressure_at_sea_level[unit="hectoPascal"]':
                                'mean_sea_level_pressure',
                            'air_temperature[unit="Celsius"]': 'temperature'})
    col_units = {'samples': '', 'observations': '', 'samplingIndex': '', 'station': '',
                 'latitude': 'degrees', 'longitude': 'degrees', 'temperature': 'degC',
                 'mean_sea_level_pressure': 'hPa', 'time': ''}
    dataframe_to_netcdf(df, path_to_save=str(tmpdir) + '/test.nc', sampling_var='station',
                        sampling_data_vars=['station', 'latitude', 'longitude'],
                        column_units=col_units, dataset_type='timeSeries')
    data = xr.open_dataset(str(tmpdir) + '/test.nc')
    assert data['station'].attrs['cf_role'] == 'timeseries_id'
    assert data['temperature'].attrs['units'] == 'degC'


def test_dataframe_to_netcdf_names(test_df, tmpdir):
    """Test attachment of standard names via a dictionary."""
    long_names = {'temperature': '2-meter air temperature',
                  'pressure': 'Mean sea-level air pressure', 'latitude': 'Station latitude',
                  'longitude': 'Station longitude', 'station_id': 'Station identifier'}
    standard_names = {'temperature': 'air_temperature',
                      'pressure': 'air_pressure_at_mean_sea_level', 'latitude': 'latitude',
                      'longitude': 'longitude', 'station_id': 'platform_id'}
    dataframe_to_netcdf(test_df, path_to_save=str(tmpdir) + '/test.nc',
                        sampling_var='station_id',
                        sampling_data_vars=['station_id', 'latitude', 'longitude'],
                        standard_names=standard_names, long_names=long_names)
    data = xr.open_dataset(str(tmpdir) + '/test.nc')
    assert data['temperature'].attrs['standard_name'] == 'air_temperature'
    assert data['station_id'].attrs['long_name'] == 'Station identifier'


def test_no_dataframe(tmpdir):
    """Test error message if Pandas DataFrame is not provided."""
    array = np.arange(0, 10)
    with pytest.raises(TypeError, match='A pandas dataframe was not provided'):
        dataframe_to_netcdf(array, path_to_save=str(tmpdir) + '/test.nc', sampling_var=None,
                            sampling_data_vars=None)


def test_file_exists(test_df, tmpdir):
    """Test error message if netCDF file already exists."""
    open(str(tmpdir) + '/test.nc', 'wb')
    with pytest.raises(ValueError, match='File already exists - please delete and run again'):
        dataframe_to_netcdf(test_df, path_to_save=str(tmpdir) + '/test.nc',
                            sampling_var='station_id',
                            sampling_data_vars=['station_id', 'latitude', 'longitude'])
