# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `cross_sections` module."""

import numpy as np
import pytest
import xarray as xr

from metpy.calc import (absolute_momentum, cross_section_components, normal_component,
                        tangential_component, unit_vectors_from_cross_section)
from metpy.calc.cross_sections import distances_from_cross_section, latitude_from_cross_section
from metpy.cbook import get_test_data
from metpy.interpolate import cross_section
from metpy.testing import assert_array_almost_equal, assert_xarray_allclose, needs_cartopy
from metpy.units import units


@pytest.fixture()
@needs_cartopy
def test_cross_lonlat():
    """Return cross section on a lon/lat grid with no time coordinate for use in tests."""
    data_u = np.linspace(-40, 40, 5 * 6 * 7).reshape((5, 6, 7)) * units.knots
    data_v = np.linspace(40, -40, 5 * 6 * 7).reshape((5, 6, 7)) * units.knots
    ds = xr.Dataset(
        {
            'u_wind': (['isobaric', 'lat', 'lon'], data_u),
            'v_wind': (['isobaric', 'lat', 'lon'], data_v)
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

    start, end = (30.5, 255.5), (44.5, 274.5)
    return cross_section(ds.metpy.parse_cf(), start, end, steps=7, interp_type='nearest')


@pytest.fixture()
@needs_cartopy
def test_cross_xy():
    """Return cross section on a x/y grid with a time coordinate for use in tests."""
    data_u = np.linspace(-25, 25, 5 * 6 * 7).reshape((1, 5, 6, 7)) * units('m/s')
    data_v = np.linspace(25, -25, 5 * 6 * 7).reshape((1, 5, 6, 7)) * units('m/s')
    ds = xr.Dataset(
        {
            'u_wind': (['time', 'isobaric', 'y', 'x'], data_u),
            'v_wind': (['time', 'isobaric', 'y', 'x'], data_v),
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
    ds['u_wind'].attrs = ds['v_wind'].attrs = {
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

    start, end = ((36.46, -112.45), (42.95, -68.74))
    return cross_section(ds.metpy.parse_cf(), start, end, steps=7)


def test_distances_from_cross_section_given_lonlat(test_cross_lonlat):
    """Test distances from cross section with lat/lon grid."""
    x, y = distances_from_cross_section(test_cross_lonlat['u_wind'])

    true_x_values = np.array([-0., 252585.3108187, 505170.6216374, 757755.93245611,
                              1010341.24327481, 1262926.55409352, 1515511.86491222])
    true_y_values = np.array([-0., 283412.80349716, 566825.60699432, 850238.41049148,
                              1133651.21398864, 1417064.0174858, 1700476.82098296])
    index = xr.DataArray(range(7), name='index', dims=['index'])
    true_x = xr.DataArray(
        true_x_values * units.meters,
        coords={
            'metpy_crs': test_cross_lonlat['metpy_crs'],
            'lat': test_cross_lonlat['lat'],
            'lon': test_cross_lonlat['lon'],
            'index': index,
        },
        dims=['index']
    )
    true_y = xr.DataArray(
        true_y_values * units.meters,
        coords={
            'metpy_crs': test_cross_lonlat['metpy_crs'],
            'lat': test_cross_lonlat['lat'],
            'lon': test_cross_lonlat['lon'],
            'index': index,
        },
        dims=['index']
    )
    assert_xarray_allclose(x, true_x)
    assert_xarray_allclose(y, true_y)


def test_distances_from_cross_section_given_xy(test_cross_xy):
    """Test distances from cross section with x/y grid."""
    x, y = distances_from_cross_section(test_cross_xy['u_wind'])
    xr.testing.assert_identical(test_cross_xy['x'], x)
    xr.testing.assert_identical(test_cross_xy['y'], y)


def test_distances_from_cross_section_given_bad_coords(test_cross_xy):
    """Ensure an AttributeError is raised when the cross section lacks need coordinates."""
    with pytest.raises(AttributeError):
        distances_from_cross_section(test_cross_xy['u_wind'].drop_vars('x'))


def test_latitude_from_cross_section_given_lat(test_cross_lonlat):
    """Test latitude from cross section with latitude given."""
    latitude = latitude_from_cross_section(test_cross_lonlat['v_wind'])
    xr.testing.assert_identical(test_cross_lonlat['lat'], latitude)


def test_latitude_from_cross_section_given_y(test_cross_xy):
    """Test latitude from cross section with y given."""
    latitude = latitude_from_cross_section(test_cross_xy['v_wind'])
    true_latitude_values = np.array([36.46, 38.64829115, 40.44833152, 41.81277354, 42.7011178,
                                     43.0845549, 42.95])
    index = xr.DataArray(range(7), name='index', dims=['index'])
    true_latitude = xr.DataArray(
        true_latitude_values * units.degrees_north,
        coords={
            'metpy_crs': test_cross_xy['metpy_crs'],
            'y': test_cross_xy['y'],
            'x': test_cross_xy['x'],
            'index': index,
        },
        dims=['index']
    )
    assert_xarray_allclose(latitude, true_latitude)


def test_unit_vectors_from_cross_section_given_lonlat(test_cross_lonlat):
    """Test unit vector calculation from cross section with lat/lon grid."""
    unit_tangent, unit_normal = unit_vectors_from_cross_section(test_cross_lonlat['u_wind'])
    true_unit_tangent = np.array([[0.66533859, 0.66533859, 0.66533859, 0.66533859, 0.66533859,
                                   0.66533859, 0.66533859],
                                  [0.74654173, 0.74654173, 0.74654173, 0.74654173, 0.74654173,
                                   0.74654173, 0.74654173]])
    true_unit_normal = np.array([[-0.74654173, -0.74654173, -0.74654173, -0.74654173,
                                  -0.74654173, -0.74654173, -0.74654173],
                                 [0.66533859, 0.66533859, 0.66533859, 0.66533859,
                                  0.66533859, 0.66533859, 0.66533859]])
    assert_array_almost_equal(true_unit_tangent, unit_tangent, 7)
    assert_array_almost_equal(true_unit_normal, unit_normal, 7)


def test_unit_vectors_from_cross_section_given_xy(test_cross_xy):
    """Test unit vector calculation from cross section with x/y grid."""
    unit_tangent, unit_normal = unit_vectors_from_cross_section(test_cross_xy['u_wind'])
    true_unit_tangent = np.array([[0.93567585, 0.929688, 0.92380315, 0.91844706, 0.91349795,
                                   0.90875771, 0.90400673],
                                 [0.35286074, 0.36834796, 0.3828678, 0.39554392, 0.40684333,
                                  0.41732413, 0.42751822]])
    true_unit_normal = np.array([[-0.35286074, -0.36834796, -0.3828678, -0.39554392,
                                  -0.40684333, -0.41732413, -0.42751822],
                                 [0.93567585, 0.929688, 0.92380315, 0.91844706, 0.91349795,
                                  0.90875771, 0.90400673]])
    assert_array_almost_equal(true_unit_tangent, unit_tangent, 7)
    assert_array_almost_equal(true_unit_normal, unit_normal, 7)


def test_cross_section_components(test_cross_lonlat):
    """Test getting cross section components of a 2D vector field."""
    tang_wind, norm_wind = cross_section_components(test_cross_lonlat['u_wind'],
                                                    test_cross_lonlat['v_wind'])
    true_tang_wind_values = np.array([[3.24812563, 2.9994653, 2.75080496, 2.50214463,
                                       2.47106208, 2.22240175, 1.97374141],
                                      [1.94265887, 1.69399854, 1.4453382, 1.19667786,
                                       1.16559532, 0.91693499, 0.66827465],
                                      [0.63719211, 0.38853177, 0.13987144, -0.1087889,
                                       -0.13987144, -0.38853177, -0.63719211],
                                      [-0.66827465, -0.91693499, -1.16559532, -1.41425566,
                                       -1.4453382, -1.69399854, -1.94265887],
                                      [-1.97374141, -2.22240175, -2.47106208, -2.71972242,
                                       -2.75080496, -2.9994653, -3.24812563]])
    true_norm_wind_values = np.array([[56.47521297, 52.15175169, 47.82829041, 43.50482913,
                                       42.96439647, 38.64093519, 34.31747391],
                                      [33.77704125, 29.45357997, 25.13011869, 20.80665741,
                                       20.26622475, 15.94276347, 11.61930219],
                                      [11.07886953, 6.75540825, 2.43194697, -1.89151431,
                                       -2.43194697, -6.75540825, -11.07886953],
                                      [-11.61930219, -15.94276347, -20.26622475, -24.58968603,
                                       -25.13011869, -29.45357997, -33.77704125],
                                      [-34.31747391, -38.64093519, -42.96439647, -47.28785775,
                                       -47.82829041, -52.15175169, -56.47521297]])
    true_tang_wind = xr.DataArray(true_tang_wind_values * units.knots,
                                  coords=test_cross_lonlat['u_wind'].coords,
                                  dims=test_cross_lonlat['u_wind'].dims,
                                  attrs=test_cross_lonlat['u_wind'].attrs)
    true_norm_wind = xr.DataArray(true_norm_wind_values * units.knots,
                                  coords=test_cross_lonlat['u_wind'].coords,
                                  dims=test_cross_lonlat['u_wind'].dims,
                                  attrs=test_cross_lonlat['u_wind'].attrs)
    assert_xarray_allclose(tang_wind, true_tang_wind)
    assert_xarray_allclose(norm_wind, true_norm_wind)


def test_tangential_component(test_cross_xy):
    """Test getting cross section tangential component of a 2D vector field."""
    tang_wind = tangential_component(test_cross_xy['u_wind'], test_cross_xy['v_wind'])
    true_tang_wind_values = np.array([[[-14.56982141, -13.17102075, -11.83790134,
                                        -10.59675064, -9.42888813, -8.31533355, -7.2410326],
                                       [-8.71378435, -7.53076196, -6.40266576, -5.34269988,
                                        -4.33810002, -3.37748418, -2.45334901],
                                       [-2.85774728, -1.89050316, -0.96743019, -0.08864912,
                                        0.7526881, 1.5603652, 2.33433459],
                                       [2.99828978, 3.74975563, 4.46780539, 5.16540164,
                                        5.84347621, 6.49821458, 7.12201819],
                                       [8.85432685, 9.39001443, 9.90304096, 10.41945241,
                                        10.93426433, 11.43606396, 11.90970179]]])
    true_tang_wind = xr.DataArray(true_tang_wind_values * units('m/s'),
                                  coords=test_cross_xy['u_wind'].coords,
                                  dims=test_cross_xy['u_wind'].dims,
                                  attrs=test_cross_xy['u_wind'].attrs)
    assert_xarray_allclose(tang_wind, true_tang_wind)


def test_normal_component(test_cross_xy):
    """Test getting cross section normal component of a 2D vector field."""
    norm_wind = normal_component(test_cross_xy['u_wind'], test_cross_xy['v_wind'])
    true_norm_wind_values = np.array([[[32.21218429, 30.45650997, 28.59536112, 26.62832466,
                                        24.57166983, 22.43805311, 20.2347284],
                                       [19.26516594, 17.41404337, 15.46613157, 13.42554447,
                                        11.30508283, 9.11378585, 6.85576955],
                                       [6.3181476, 4.37157677, 2.33690202, 0.22276428,
                                        -1.96150417, -4.2104814, -6.52318931],
                                       [-6.62887075, -8.67088982, -10.79232752, -12.98001592,
                                        -15.22809117, -17.53474865, -19.90214816],
                                       [-19.5758891, -21.71335642, -23.92155707, -26.18279611,
                                        -28.49467817, -30.8590159, -33.28110701]]])
    true_norm_wind = xr.DataArray(true_norm_wind_values * units('m/s'),
                                  coords=test_cross_xy['u_wind'].coords,
                                  dims=test_cross_xy['u_wind'].dims,
                                  attrs=test_cross_xy['u_wind'].attrs)
    assert_xarray_allclose(norm_wind, true_norm_wind)


def test_absolute_momentum_given_lonlat(test_cross_lonlat):
    """Test absolute momentum calculation."""
    momentum = absolute_momentum(test_cross_lonlat['u_wind'], test_cross_lonlat['v_wind'])
    true_momentum_values = np.array([[29.05335956, 57.00676169, 88.89733786, 124.37969813,
                                      165.02883664, 206.55775948, 250.49678829],
                                     [17.37641122, 45.32981335, 77.22038952, 112.70274979,
                                      153.3518883, 194.88081114, 238.81983995],
                                     [5.69946288, 33.65286501, 65.54344118, 101.02580145,
                                      141.67493996, 183.2038628, 227.14289161],
                                     [-5.97748546, 21.97591667, 53.86649284, 89.34885311,
                                      129.99799162, 171.52691446, 215.46594327],
                                     [-17.6544338, 10.29896833, 42.1895445, 77.67190477,
                                      118.32104328, 159.84996612, 203.78899492]])

    true_momentum = xr.DataArray(true_momentum_values * units('m/s'),
                                 coords=test_cross_lonlat['u_wind'].coords,
                                 dims=test_cross_lonlat['u_wind'].dims)
    assert_xarray_allclose(momentum, true_momentum)


def test_absolute_momentum_given_xy(test_cross_xy):
    """Test absolute momentum calculation."""
    momentum = absolute_momentum(test_cross_xy['u_wind'], test_cross_xy['v_wind'])
    true_momentum_values = np.array([[[169.22222693, 146.36354006, 145.75559124, 171.8710635,
                                       215.04876817, 265.73797007, 318.34138347],
                                      [156.27520858, 133.32107346, 132.62636169, 158.66828331,
                                       201.78218117, 252.41370282, 304.96242462],
                                      [143.32819023, 120.27860686, 119.49713214, 145.46550311,
                                       188.51559418, 239.08943557, 291.58346576],
                                      [130.38117188, 107.23614026, 106.36790259, 132.26272292,
                                       175.24900718, 225.76516831, 278.20450691],
                                      [117.43415353, 94.19367366, 93.23867305, 119.05994273,
                                       161.98242018, 212.44090106, 264.82554806]]])
    true_momentum = xr.DataArray(true_momentum_values * units('m/s'),
                                 coords=test_cross_xy['u_wind'].coords,
                                 dims=test_cross_xy['u_wind'].dims)
    assert_xarray_allclose(momentum, true_momentum)


def test_absolute_momentum_xarray_units_attr():
    """Test absolute momentum when `u` and `v` are DataArrays with a `units` attribute."""
    data = xr.open_dataset(get_test_data('narr_example.nc', False))
    data = data.metpy.parse_cf().squeeze()

    start = (37.0, -105.0)
    end = (35.5, -65.0)
    cross = cross_section(data, start, end)

    u = cross['u_wind'][0].sel(index=slice(0, 2))
    v = cross['v_wind'][0].sel(index=slice(0, 2))

    momentum = absolute_momentum(u, v)
    true_momentum_values = np.array([137.46164031, 134.11450232, 133.85196023])
    true_momentum = xr.DataArray(units.Quantity(true_momentum_values, 'm/s'),
                                 coords=u.coords)

    assert_xarray_allclose(momentum, true_momentum)
