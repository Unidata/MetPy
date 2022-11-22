import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pandas as pd
import pytest
import xarray as xr

from metpy.interpolate import interpolate_to_slice
from metpy.units import units

@pytest.fixture()
def test_ds_lonlat():
    """Return dataset on a lon/lat grid with no time coordinate for use in tests."""
    data_temp = np.linspace(250, 300, 5 * 6 * 7).reshape((5, 6, 7)) * units.kelvin
    data_rh = np.linspace(0, 1, 5 * 6 * 7).reshape((5, 6, 7)) * units.dimensionless
    ds = xr.Dataset(
        {
            'temperature': (['isobaric', 'lat', 'lon'], data_temp),
            'relative_humidity': (['isobaric', 'lat', 'lon'], data_rh)
        },
        coords={
            'isobaric': xr.DataArray(
                np.linspace(1000, 500, 5),
                name='isobaric',
                dims=['isobaric'],
                attrs={'units': 'hPa'}
            ),
            'lat': xr.DataArray(
                np.linspace(30, 45, 6),
                name='lat',
                dims=['lat'],
                attrs={'units': 'degrees_north'}
            ),
            'lon': xr.DataArray(
                np.linspace(255, 275, 7),
                name='lon',
                dims=['lon'],
                attrs={'units': 'degrees_east'}
            )
        }
    )
    return ds.metpy.parse_cf()


def test_interpolate_to_slice_against_selection(test_ds_lonlat):
    """Test interpolate_to_slice on a simple operation."""
    data = test_ds_lonlat['temperature']
    path = np.array([[265.0, 30.],
                     [265.0, 36.],
                     [265.0, 42.]])
    test_slice = interpolate_to_slice(data, path)
    true_slice = [0., 6., 12.]
    # Coordinates differ, so just compare the data
    assert_array_almost_equal(true_slice, test_slice["distance"], 5)


