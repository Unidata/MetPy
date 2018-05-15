# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the operation of MetPy's XArray accessors."""
from __future__ import absolute_import

import cartopy.crs as ccrs
import numpy as np
import pytest
import xarray as xr

from metpy.testing import assert_almost_equal, assert_array_equal, get_test_data
from metpy.units import units
from metpy.xarray import preprocess_xarray


@pytest.fixture
def test_ds():
    """Provide an xarray dataset for testing."""
    return xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))


@pytest.fixture
def test_var():
    """Provide a standard, parsed, variable for testing."""
    ds = test_ds()
    return ds.metpy.parse_cf('Temperature')


def test_projection(test_var):
    """Test getting the proper projection out of the variable."""
    crs = test_var.metpy.crs
    assert crs['grid_mapping_name'] == 'lambert_conformal_conic'

    assert isinstance(test_var.metpy.cartopy_crs, ccrs.LambertConformal)


def test_no_projection(test_ds):
    """Test getting the crs attribute when not available produces a sensible error."""
    var = test_ds.lat
    with pytest.raises(AttributeError) as exc:
        var.metpy.crs

    assert 'not available' in str(exc.value)


def test_units(test_var):
    """Test unit handling through the accessor."""
    arr = test_var.metpy.unit_array
    assert isinstance(arr, units.Quantity)
    assert arr.units == units.kelvin


def test_convert_units(test_var):
    """Test in-place conversion of units."""
    test_var.metpy.convert_units('degC')

    # Check that variable metadata is updated
    assert test_var.attrs['units'] == 'degC'

    # Make sure we now get an array back with properly converted values
    assert test_var.metpy.unit_array.units == units.degC
    assert_almost_equal(test_var[0, 0, 0, 0], 18.44 * units.degC, 2)


def test_radian_projection_coords():
    """Test fallback code for radian units in projection coordinate variables."""
    proj = xr.DataArray(0, attrs={'grid_mapping_name': 'geostationary',
                                  'perspective_point_height': 3})
    x = xr.DataArray(np.arange(3),
                     attrs={'standard_name': 'projection_x_coordinate', 'units': 'radians'})
    y = xr.DataArray(np.arange(2),
                     attrs={'standard_name': 'projection_y_coordinate', 'units': 'radians'})
    data = xr.DataArray(np.arange(6).reshape(2, 3), coords=(y, x), dims=('y', 'x'),
                        attrs={'grid_mapping': 'fixedgrid_projection'})
    ds = xr.Dataset({'data': data, 'fixedgrid_projection': proj})

    # Check that the coordinates in this case are properly converted
    data_var = ds.metpy.parse_cf('data')
    assert data_var.coords['x'].metpy.unit_array[1] == 3 * units.meter
    assert data_var.coords['y'].metpy.unit_array[1] == 3 * units.meter


def test_missing_grid_mapping():
    """Test falling back to implicit lat/lon projection."""
    lon = xr.DataArray(-np.arange(3),
                       attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat = xr.DataArray(np.arange(2),
                       attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    data = xr.DataArray(np.arange(6).reshape(2, 3), coords=(lat, lon), dims=('y', 'x'))
    ds = xr.Dataset({'data': data})

    data_var = ds.metpy.parse_cf('data')
    assert 'crs' in data_var.coords


def test_missing_grid_mapping_var():
    """Test behavior when we can't find the variable pointed to by grid_mapping."""
    x = xr.DataArray(np.arange(3),
                     attrs={'standard_name': 'projection_x_coordinate', 'units': 'radians'})
    y = xr.DataArray(np.arange(2),
                     attrs={'standard_name': 'projection_y_coordinate', 'units': 'radians'})
    data = xr.DataArray(np.arange(6).reshape(2, 3), coords=(y, x), dims=('y', 'x'),
                        attrs={'grid_mapping': 'fixedgrid_projection'})
    ds = xr.Dataset({'data': data})

    with pytest.warns(UserWarning, match='Could not find'):
        ds.metpy.parse_cf('data')


def test_preprocess_xarray():
    """Test xarray preprocessing decorator."""
    data = xr.DataArray(np.ones(3), attrs={'units': 'km'})
    data2 = xr.DataArray(np.ones(3), attrs={'units': 'm'})

    @preprocess_xarray
    def func(a, b):
        return a.to('m') + b

    assert_array_equal(func(data, b=data2), np.array([1001, 1001, 1001]) * units.m)
