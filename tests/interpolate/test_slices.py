# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `slices` module."""

import numpy as np
import pytest
import xarray as xr

from metpy.interpolate import cross_section, geodesic, interpolate_to_slice
from metpy.testing import assert_array_almost_equal, needs_cartopy
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


@pytest.fixture()
def test_ds_xy():
    """Return dataset on a x/y grid with a time coordinate for use in tests."""
    data_temperature = np.linspace(250, 300, 5 * 6 * 7).reshape((1, 5, 6, 7)) * units.kelvin
    ds = xr.Dataset(
        {
            'temperature': (['time', 'isobaric', 'y', 'x'], data_temperature),
            'lambert_conformal': ([], '')
        },
        coords={
            'time': xr.DataArray(
                np.array([np.datetime64('2018-07-01T00:00')]),
                name='time',
                dims=['time']
            ),
            'isobaric': xr.DataArray(
                np.linspace(1000, 500, 5),
                name='isobaric',
                dims=['isobaric'],
                attrs={'units': 'hPa'}
            ),
            'y': xr.DataArray(
                np.linspace(-1500, 0, 6),
                name='y',
                dims=['y'],
                attrs={'units': 'km'}
            ),
            'x': xr.DataArray(
                np.linspace(-500, 3000, 7),
                name='x',
                dims=['x'],
                attrs={'units': 'km'}
            )
        }
    )
    ds['temperature'].attrs = {
        'grid_mapping': 'lambert_conformal'
    }
    ds['lambert_conformal'].attrs = {
        'grid_mapping_name': 'lambert_conformal_conic',
        'standard_parallel': 50.0,
        'longitude_of_central_meridian': -107.0,
        'latitude_of_projection_origin': 50.0,
        'earth_shape': 'spherical',
        'earth_radius': 6367470.21484375
    }
    return ds.metpy.parse_cf()


def test_interpolate_to_slice_against_selection(test_ds_lonlat):
    """Test interpolate_to_slice on a simple operation."""
    data = test_ds_lonlat['temperature']
    path = np.array([[265.0, 30.],
                     [265.0, 36.],
                     [265.0, 42.]])
    test_slice = interpolate_to_slice(data, path)
    true_slice = data.sel({'lat': [30., 36., 42.], 'lon': 265.0})
    # Coordinates differ, so just compare the data
    assert_array_almost_equal(true_slice.metpy.unit_array, test_slice.metpy.unit_array, 5)


@needs_cartopy
def test_geodesic(test_ds_xy):
    """Test the geodesic construction."""
    crs = test_ds_xy['temperature'].metpy.pyproj_crs
    path = geodesic(crs, (36.46, -112.45), (42.95, -68.74), 7)
    truth = np.array([[-4.99495719e+05, -1.49986599e+06],
                      [9.84044354e+04, -1.26871737e+06],
                      [6.88589099e+05, -1.02913966e+06],
                      [1.27269045e+06, -7.82037603e+05],
                      [1.85200974e+06, -5.28093957e+05],
                      [2.42752546e+06, -2.67710326e+05],
                      [2.99993290e+06, -9.39107692e+02]])
    assert_array_almost_equal(path, truth, 0)


@needs_cartopy
def test_cross_section_dataarray_and_linear_interp(test_ds_xy):
    """Test the cross_section function with a data array and linear interpolation."""
    data = test_ds_xy['temperature']
    start, end = ((36.46, -112.45), (42.95, -68.74))
    data_cross = cross_section(data, start, end, steps=7)
    truth_values = np.array([[[250.00095489, 251.53646673, 253.11586664, 254.73477364,
                               256.38991013, 258.0794356, 259.80334269],
                              [260.04880178, 261.58431362, 263.16371353, 264.78262053,
                               266.43775702, 268.12728249, 269.85118958],
                              [270.09664867, 271.63216051, 273.21156042, 274.83046742,
                               276.48560391, 278.17512938, 279.89903647],
                              [280.14449556, 281.6800074, 283.25940731, 284.87831431,
                               286.5334508, 288.22297627, 289.94688336],
                              [290.19234245, 291.72785429, 293.3072542, 294.9261612,
                               296.58129769, 298.27082316, 299.99473025]]])
    truth_values_x = np.array([-499495.71907062, 98404.43537514, 688589.09865512,
                               1272690.44926197, 1852009.73516881, 2427525.45740665,
                               2999932.89862589])
    truth_values_y = np.array([-1499865.98780602, -1268717.36799267, -1029139.66048478,
                               -782037.60343652, -528093.95678826, -267710.32566917,
                               -939.10769171])
    index = xr.DataArray(range(7), name='index', dims=['index'])

    data_truth_x = xr.DataArray(
        truth_values_x,
        name='x',
        coords={
            'metpy_crs': data['metpy_crs'],
            'y': (['index'], truth_values_y),
            'x': (['index'], truth_values_x),
            'index': index,
        },
        dims=['index']
    )
    data_truth_y = xr.DataArray(
        truth_values_y,
        name='y',
        coords={
            'metpy_crs': data['metpy_crs'],
            'y': (['index'], truth_values_y),
            'x': (['index'], truth_values_x),
            'index': index,
        },
        dims=['index']
    )
    data_truth = xr.DataArray(
        truth_values * units.kelvin,
        name='temperature',
        coords={
            'time': data['time'],
            'isobaric': data['isobaric'],
            'index': index,
            'metpy_crs': data['metpy_crs'],
            'y': data_truth_y,
            'x': data_truth_x
        },
        dims=['time', 'isobaric', 'index']
    )

    xr.testing.assert_allclose(data_truth, data_cross)


def test_cross_section_dataarray_projection_noop(test_ds_xy):
    """Test the cross_section function with a projection dataarray."""
    data = test_ds_xy['lambert_conformal']
    start, end = ((36.46, -112.45), (42.95, -68.74))
    data_cross = cross_section(data, start, end, steps=7)
    xr.testing.assert_identical(data, data_cross)


@needs_cartopy
def test_cross_section_dataset_and_nearest_interp(test_ds_lonlat):
    """Test the cross_section function with a dataset and nearest interpolation."""
    start, end = (30.5, 255.5), (44.5, 274.5)
    data_cross = cross_section(test_ds_lonlat, start, end, steps=7, interp_type='nearest')
    nearest_values = test_ds_lonlat.isel(lat=xr.DataArray([0, 1, 2, 3, 3, 4, 5], dims='index'),
                                         lon=xr.DataArray(range(7), dims='index'))
    truth_temp = nearest_values['temperature'].metpy.unit_array
    truth_rh = nearest_values['relative_humidity'].metpy.unit_array
    truth_values_lon = np.array([255.5, 258.20305939, 261.06299342, 264.10041516,
                                 267.3372208, 270.7961498, 274.5])
    truth_values_lat = np.array([30.5, 33.02800969, 35.49306226, 37.88512911, 40.19271688,
                                 42.40267088, 44.5])
    index = xr.DataArray(range(7), name='index', dims=['index'])

    data_truth = xr.Dataset(
        {
            'temperature': (['isobaric', 'index'], truth_temp),
            'relative_humidity': (['isobaric', 'index'], truth_rh)
        },
        coords={
            'isobaric': test_ds_lonlat['isobaric'],
            'index': index,
            'metpy_crs': test_ds_lonlat['metpy_crs'],
            'lat': (['index'], truth_values_lat),
            'lon': (['index'], truth_values_lon)
        },
    )

    xr.testing.assert_allclose(data_truth, data_cross)


def test_interpolate_to_slice_error_on_missing_coordinate(test_ds_lonlat):
    """Test that the proper error is raised with missing coordinate."""
    # Use a variable with a coordinate removed
    data_bad = test_ds_lonlat['temperature'].copy()
    del data_bad['lat']
    path = np.array([[265.0, 30.],
                     [265.0, 36.],
                     [265.0, 42.]])

    with pytest.raises(ValueError):
        interpolate_to_slice(data_bad, path)


def test_cross_section_error_on_missing_coordinate(test_ds_lonlat):
    """Test that the proper error is raised with missing coordinate."""
    # Use a variable with no crs coordinate
    data_bad = test_ds_lonlat['temperature'].copy()
    del data_bad['metpy_crs']
    start, end = (30.5, 255.5), (44.5, 274.5)

    with pytest.raises(ValueError):
        cross_section(data_bad, start, end)
