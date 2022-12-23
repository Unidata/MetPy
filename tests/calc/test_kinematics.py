# Copyright (c) 2008,2015,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `kinematics` module."""

import numpy as np
import pytest
import xarray as xr

from metpy.calc import (absolute_vorticity, advection, ageostrophic_wind, coriolis_parameter,
                        divergence, first_derivative, frontogenesis, geospatial_laplacian,
                        geostrophic_wind, inertial_advective_wind, lat_lon_grid_deltas,
                        montgomery_streamfunction, potential_temperature,
                        potential_vorticity_baroclinic, potential_vorticity_barotropic,
                        q_vector, shearing_deformation, static_stability,
                        storm_relative_helicity, stretching_deformation, total_deformation,
                        vorticity, wind_components)
from metpy.constants import g, Re
from metpy.testing import (assert_almost_equal, assert_array_almost_equal, assert_array_equal,
                           get_test_data)
from metpy.units import concatenate, units


@pytest.fixture()
def basic_dataset():
    """Fixture to create a dataset for use in basic tests using xarray integration."""
    lon = xr.DataArray([-100, -96, -90],
                       attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat = xr.DataArray([30, 31, 33],
                       attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    u = xr.DataArray(np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s'),
                     coords=(lat, lon), dims=('lat', 'lon'))
    v = xr.DataArray(np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s'),
                     coords=(lat, lon), dims=('lat', 'lon'))
    z = xr.DataArray(np.array([[1, 2, 4], [4, 8, 4], [8, 6, 4]]) * 20. * units.meter,
                     coords=(lat, lon), dims=('lat', 'lon'))
    t = xr.DataArray(np.arange(9).reshape(3, 3) * units.kelvin, coords=(lat, lon),
                     dims=('lat', 'lon'))
    return xr.Dataset({'u': u, 'v': v, 'height': z, 'temperature': t}).metpy.parse_cf()


def test_default_order():
    """Test using the default array ordering."""
    u = np.ones((3, 3)) * units('m/s')
    v = vorticity(u, u, dx=1 * units.meter, dy=1 * units.meter)
    true_v = np.zeros_like(u) / units.sec
    assert_array_equal(v, true_v)


def test_zero_vorticity():
    """Test vorticity calculation when zeros should be returned."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    v = vorticity(u.T, u, dx=1 * units.meter, dy=1 * units.meter)
    true_v = np.zeros_like(u) / units.sec
    assert_array_equal(v, true_v)


def test_vorticity():
    """Test vorticity for simple case."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    v = vorticity(u.T, u.T, dx=1 * units.meter, dy=1 * units.meter)
    true_v = np.ones_like(u) / units.sec
    assert_array_equal(v, true_v)


@pytest.mark.parametrize('geog_data', ('+proj=lcc lat_1=25', '+proj=latlon', '+proj=stere'),
                         indirect=True)
def test_vorticity_geographic(geog_data):
    """Test vorticity for simple case on geographic coordinates."""
    crs, lons, lats, u, v, mx, my, dx, dy = geog_data
    vort = vorticity(u, v, longitude=lons, latitude=lats, crs=crs)

    # Calculate the true field using known map-correct approach
    truth = (mx * first_derivative(v, delta=dx, axis=1)
             - my * first_derivative(u, delta=dy, axis=0)
             - (v * mx / my) * first_derivative(my, delta=dx, axis=1)
             + (u * my / mx) * first_derivative(mx, delta=dy, axis=0))

    assert_array_almost_equal(vort, truth, 12)


@pytest.mark.parametrize('geog_data', ('+proj=lcc lat_1=25', '+proj=latlon', '+proj=stere'),
                         indirect=True)
def test_abs_vorticity_geographic(geog_data):
    """Test absolute_vorticity for simple case on geographic coordinates."""
    # Generate a field of u and v on a lat/lon grid
    crs, lons, lats, u, v, mx, my, dx, dy = geog_data
    vort = absolute_vorticity(u, v, longitude=lons, latitude=lats[:, None], crs=crs)

    # Calculate the true field using known map-correct approach
    truth = ((mx * first_derivative(v, delta=dx, axis=1)
              - my * first_derivative(u, delta=dy, axis=0)
              - (v * mx / my) * first_derivative(my, delta=dx, axis=1)
              + (u * my / mx) * first_derivative(mx, delta=dy, axis=0)
              )
             + coriolis_parameter(lats[:, None]))

    assert_array_almost_equal(vort, truth, 12)


def test_vorticity_asym():
    """Test vorticity calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    vort = vorticity(u, v, dx=1 * units.meters, dy=2 * units.meters)
    true_vort = np.array([[-2.5, 3.5, 13.], [8.5, -1.5, -11.], [-5.5, -1.5, 0.]]) / units.sec
    assert_array_equal(vort, true_vort)


def test_vorticity_positional_grid_args_failure():
    """Test that old API of positional grid arguments to vorticity fails."""
    # pylint: disable=too-many-function-args
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    with pytest.raises(TypeError, match='too many positional arguments'):
        vorticity(u.T, u, 1 * units.meter, 1 * units.meter)


def test_vorticity_xarray(basic_dataset):
    """Test vorticity calculation using xarray support."""
    d = vorticity(basic_dataset.u, basic_dataset.v)
    truth = np.array([[2.03538383e-5, 3.02085059e-5, 9.64086237e-5],
                      [2.48885712e-5, 8.32454044e-6, 4.33065628e-6],
                      [-4.46930721e-5, -3.87823454e-5, -6.92512555e-5]]) / units.sec
    truth = xr.DataArray(truth, coords=basic_dataset.coords)
    assert_array_almost_equal(d, truth)


def test_vorticity_grid_pole():
    """Test vorticity consistency at a pole (#2582)."""
    xy = [-25067.525, 0., 25067.525]
    us = np.ones((len(xy), len(xy)))
    vs = us * np.linspace(-1, 0, len(xy))[None, :]
    grid = {'grid_mapping_name': 'lambert_azimuthal_equal_area',
            'longitude_of_projection_origin': 0, 'latitude_of_projection_origin': 90,
            'false_easting': 0, 'false_northing': 0}

    x = xr.DataArray(
        xy, name='x', attrs={'standard_name': 'projection_x_coordinate', 'units': 'm'})
    y = xr.DataArray(
        xy, name='y', attrs={'standard_name': 'projection_y_coordinate', 'units': 'm'})
    u = xr.DataArray(us, name='u', coords=(y, x), dims=('y', 'x'), attrs={'units': 'm/s'})
    v = xr.DataArray(vs, name='v', coords=(y, x), dims=('y', 'x'), attrs={'units': 'm/s'})

    ds = xr.merge((u, v)).metpy.assign_crs(grid)

    vort = vorticity(ds.u, ds.v)

    assert_array_almost_equal(vort.isel(y=0), vort.isel(y=-1), decimal=9)


def test_zero_divergence():
    """Test divergence calculation when zeros should be returned."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    c = divergence(u.T, u, dx=1 * units.meter, dy=1 * units.meter)
    true_c = 2. * np.ones_like(u) / units.sec
    assert_array_equal(c, true_c)


def test_divergence():
    """Test divergence for simple case."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    c = divergence(u.T, u.T, dx=1 * units.meter, dy=1 * units.meter)
    true_c = np.ones_like(u) / units.sec
    assert_array_equal(c, true_c)


@pytest.mark.parametrize('geog_data', ('+proj=lcc lat_1=25', '+proj=latlon', '+proj=stere'),
                         indirect=True)
def test_divergence_geographic(geog_data):
    """Test divergence for simple case on geographic coordinates."""
    # Generate a field of u and v on a lat/lon grid
    crs, lons, lats, u, v, mx, my, dx, dy = geog_data
    div = divergence(u, v, longitude=lons, latitude=lats, crs=crs)

    # Calculate the true field using known map-correct approach
    truth = (mx * first_derivative(u, delta=dx, axis=1)
             + my * first_derivative(v, delta=dy, axis=0)
             - (u * mx / my) * first_derivative(my, delta=dx, axis=1)
             - (v * my / mx) * first_derivative(mx, delta=dy, axis=0))

    assert_array_almost_equal(div, truth, 12)


def test_horizontal_divergence():
    """Test taking the horizontal divergence of a 3D field."""
    u = np.array([[[1., 1., 1.],
                   [1., 0., 1.],
                   [1., 1., 1.]],
                  [[0., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 0.]]]) * units('m/s')
    c = divergence(u, u, dx=1 * units.meter, dy=1 * units.meter)
    true_c = np.array([[[0., -2., 0.],
                        [-2., 0., 2.],
                        [0., 2., 0.]],
                       [[0., 2., 0.],
                        [2., 0., -2.],
                        [0., -2., 0.]]]) * units('s^-1')
    assert_array_equal(c, true_c)


def test_divergence_asym():
    """Test divergence calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    c = divergence(u, v, dx=1 * units.meters, dy=2 * units.meters)
    true_c = np.array([[-2, 5.5, -2.5], [2., 0.5, -1.5], [3., -1.5, 8.5]]) / units.sec
    assert_array_equal(c, true_c)


def test_divergence_positional_grid_args_failure():
    """Test that old API of positional grid arguments to divergence fails."""
    # pylint: disable=too-many-function-args
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    with pytest.raises(TypeError, match='too many positional arguments'):
        divergence(u, u, 1 * units.meter, 1 * units.meter)


def test_divergence_xarray(basic_dataset):
    """Test divergence calculation using xarray support."""
    d = divergence(basic_dataset.u, basic_dataset.v)
    truth = np.array([[-4.361528313e-5, 3.593794959e-5, -9.728275045e-5],
                      [-1.672604193e-5, 9.158127775e-6, -4.223658565e-5],
                      [3.011996772e-5, -3.745237046e-5, 9.570189256e-5]]) / units.sec
    truth = xr.DataArray(truth, coords=basic_dataset.coords)
    assert_array_almost_equal(d, truth, 4)


@pytest.mark.parametrize('geog_data', ('+proj=lcc lat_1=25', '+proj=latlon', '+proj=stere'),
                         indirect=True)
def test_shearing_deformation_geographic(geog_data):
    """Test shearing deformation for simple case on geographic coordinates."""
    # Generate a field of u and v on a lat/lon grid
    crs, lons, lats, u, v, mx, my, dx, dy = geog_data
    shear = shearing_deformation(u, v, longitude=lons, latitude=lats, crs=crs)

    # Calculate the true field using known map-correct approach
    truth = (mx * first_derivative(v, delta=dx, axis=1)
             + my * first_derivative(u, delta=dy, axis=0)
             + (v * mx / my) * first_derivative(my, delta=dx, axis=1)
             + (u * my / mx) * first_derivative(mx, delta=dy, axis=0))

    assert_array_almost_equal(shear, truth, 12)


def test_shearing_deformation_asym():
    """Test shearing deformation calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    sh = shearing_deformation(u, v, 1 * units.meters, 2 * units.meters)
    true_sh = np.array([[-7.5, -1.5, 1.], [9.5, -0.5, -11.], [1.5, 5.5, 12.]]) / units.sec
    assert_array_equal(sh, true_sh)


@pytest.mark.parametrize('geog_data', ('+proj=lcc lat_1=25', '+proj=latlon', '+proj=stere'),
                         indirect=True)
def test_stretching_deformation_geographic(geog_data):
    """Test divergence for simple case on geographic coordinates."""
    # Generate a field of u and v on a lat/lon grid
    crs, lons, lats, u, v, mx, my, dx, dy = geog_data
    stretch = stretching_deformation(u, v, longitude=lons, latitude=lats, crs=crs)

    # Calculate the true field using known map-correct approach
    truth = (mx * first_derivative(u, delta=dx, axis=1)
             - my * first_derivative(v, delta=dy, axis=0)
             + (u * mx / my) * first_derivative(my, delta=dx, axis=1)
             - (v * my / mx) * first_derivative(mx, delta=dy, axis=0))

    assert_array_almost_equal(stretch, truth, 12)


def test_stretching_deformation_asym():
    """Test stretching deformation calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    st = stretching_deformation(u, v, 1 * units.meters, 2 * units.meters)
    true_st = np.array([[4., 0.5, 12.5], [4., 1.5, -0.5], [1., 5.5, -4.5]]) / units.sec
    assert_array_equal(st, true_st)


def test_total_deformation_asym():
    """Test total deformation calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    tdef = total_deformation(u, v, 1 * units.meters, 2 * units.meters)
    true_tdef = np.array([[8.5, 1.58113883, 12.5399362], [10.30776406, 1.58113883, 11.0113578],
                          [1.80277562, 7.7781746, 12.8160056]]) / units.sec
    assert_almost_equal(tdef, true_tdef)


def test_frontogenesis_asym():
    """Test frontogenesis calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    theta = np.array([[303, 295, 305], [308, 310, 312], [299, 293, 289]]) * units('K')
    fronto = frontogenesis(theta, u, v, 1 * units.meters, 2 * units.meters)
    true_fronto = np.array([[-52.4746386, -37.3658646, -50.3996939],
                            [3.5777088, -2.1221867, -16.9941166],
                            [-23.1417334, 26.0499143, -158.4839684]]
                           ) * units.K / units.meter / units.sec
    assert_almost_equal(fronto, true_fronto)


def test_advection_uniform():
    """Test advection calculation for a uniform 1D field."""
    u = np.ones((3,)) * units('m/s')
    s = np.ones_like(u) * units.kelvin
    a = advection(s.T, u.T, dx=1 * units.meter)
    truth = np.zeros_like(u) * units('K/sec')
    assert_array_equal(a, truth)


def test_advection_1d_uniform_wind():
    """Test advection for simple 1D case with uniform wind."""
    u = np.ones((3,)) * units('m/s')
    s = np.array([1, 2, 3]) * units('kg')
    a = advection(s.T, u.T, dx=1 * units.meter)
    truth = -np.ones_like(u) * units('kg/sec')
    assert_array_equal(a, truth)


def test_advection_1d():
    """Test advection calculation with varying wind and field."""
    u = np.array([1, 2, 3]) * units('m/s')
    s = np.array([1, 2, 3]) * units('Pa')
    a = advection(s.T, u.T, dx=1 * units.meter)
    truth = np.array([-1, -2, -3]) * units('Pa/sec')
    assert_array_equal(a, truth)


def test_advection_2d_uniform():
    """Test advection for uniform 2D field."""
    u = np.ones((3, 3)) * units('m/s')
    s = np.ones_like(u) * units.kelvin
    a = advection(s.T, u.T, u.T, dx=1 * units.meter, dy=1 * units.meter)
    truth = np.zeros_like(u) * units('K/sec')
    assert_array_equal(a, truth)


def test_advection_2d():
    """Test advection in varying 2D field."""
    u = np.ones((3, 3)) * units('m/s')
    v = 2 * np.ones((3, 3)) * units('m/s')
    s = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * units.kelvin
    a = advection(s.T, v.T, u.T, dx=1 * units.meter, dy=1 * units.meter)
    truth = np.array([[-6, -4, 2], [-8, 0, 8], [-2, 4, 6]]) * units('K/sec')
    assert_array_equal(a, truth)


def test_advection_z_y():
    """Test advection in varying 2D z-y field."""
    v = 2 * np.ones((3, 3)) * units('m/s')
    w = np.ones((3, 3)) * units('m/s')
    s = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * units.kelvin
    a = advection(s.T, v=v.T, w=w.T,
                  dy=1 * units.meter, dz=1 * units.meter,
                  y_dim=-1, vertical_dim=-2)
    truth = np.array([[-6, -4, 2], [-8, 0, 8], [-2, 4, 6]]) * units('K/sec')
    assert_array_equal(a, truth)


def test_advection_4d_vertical(data_4d):
    """Test 4-d vertical advection with parsed dims."""
    data_4d['w'] = -abs(data_4d['u'])
    data_4d['w'].attrs['units'] = 'Pa/s'

    a = advection(data_4d.temperature, w=data_4d.w)

    assert (a < 0).sum() == 0
    assert a.data.units == units.Unit('K/sec')


def test_advection_1d_vertical():
    """Test 1-d vertical advection with parsed dims."""
    pressure = xr.DataArray(
        np.array([1000., 950., 900.]), dims='pressure', attrs={'units': 'hPa'})
    omega = xr.DataArray(
        np.array([20., 30., 40.]),
        coords=[pressure], dims=['pressure'], attrs={'units': 'hPa/sec'})
    s = xr.DataArray(
        np.array([25., 20., 15.]),
        coords=[pressure], dims=['pressure'], attrs={'units': 'degC'})
    a = advection(s, w=omega)
    truth = xr.DataArray(
        -np.array([2, 3, 4]) * units('K/sec'), coords=[pressure], dims=['pressure'])

    assert_array_almost_equal(a, truth)


def test_advection_2d_asym():
    """Test advection in asymmetric varying 2D field."""
    u = np.arange(9).reshape(3, 3) * units('m/s')
    v = 2 * u
    s = np.array([[1, 2, 4], [4, 8, 4], [8, 6, 4]]) * units.kelvin
    a = advection(s, u, v, dx=2 * units.meter, dy=1 * units.meter)
    truth = np.array([[0, -20.75, -2.5], [-33., -16., 20.], [-48, 91., 8]]) * units('K/sec')
    assert_array_equal(a, truth)


def test_advection_xarray(basic_dataset):
    """Test advection calculation using xarray support."""
    a = advection(basic_dataset.temperature, basic_dataset.u, basic_dataset.v)
    truth = np.array([[-0.0001953019, -0.000135269, -0.000262247],
                      [-4.51008510e-5, -0.000139840, -2.44370835e-6],
                      [-2.11362160e-5, -2.29201236e-5, -3.70146905e-5]]) * units('K/sec')
    truth = xr.DataArray(truth, coords=basic_dataset.coords)
    assert_array_almost_equal(a, truth, 4)


def test_geostrophic_wind():
    """Test geostrophic wind calculation with basic conditions."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 10. * units.meter
    latitude = 30 * units.degrees
    ug, vg = geostrophic_wind(
        z.T,
        100. * units.kilometer,
        100. * units.kilometer,
        latitude
    )
    true_u = np.array([[-26.897, 0, 26.897]] * 3) * units('m/s')
    true_v = -true_u.T
    assert_array_almost_equal(ug, true_u.T, 2)
    assert_array_almost_equal(vg, true_v.T, 2)


def test_geostrophic_wind_asym():
    """Test geostrophic wind calculation with a complicated field."""
    z = np.array([[1, 2, 4], [4, 8, 4], [8, 6, 4]]) * 20. * units.meter
    latitude = 30 * units.degrees
    ug, vg = geostrophic_wind(
        z,
        2000. * units.kilometer,
        1000. * units.kilometer,
        latitude
    )
    true_u = np.array(
        [[-6.724, -26.897, 0], [-9.414, -5.379, 0], [-12.103, 16.138, 0]]
    ) * units('m/s')
    true_v = np.array(
        [[0.672, 2.017, 3.362], [10.759, 0, -10.759], [-2.690, -2.690, -2.690]]
    ) * units('m/s')
    assert_array_almost_equal(ug, true_u, 2)
    assert_array_almost_equal(vg, true_v, 2)


def test_geostrophic_geopotential():
    """Test geostrophic wind calculation with geopotential."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
    latitude = 30 * units.degrees
    ug, vg = geostrophic_wind(
        z.T,
        100. * units.kilometer,
        100. * units.kilometer,
        latitude
    )
    true_u = np.array([[-27.427, 0, 27.427]] * 3) * units('m/s')
    true_v = -true_u.T
    assert_array_almost_equal(ug, true_u.T, 2)
    assert_array_almost_equal(vg, true_v.T, 2)


def test_geostrophic_3d():
    """Test geostrophic wind calculation with 3D array."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 10.
    latitude = 30 * units.degrees
    z3d = np.dstack((z, z)) * units.meter
    ug, vg = geostrophic_wind(
        z3d.T,
        100. * units.kilometer,
        100. * units.kilometer,
        latitude
    )
    true_u = np.array([[-26.897, 0, 26.897]] * 3) * units('m/s')
    true_v = -true_u.T

    true_u = concatenate((true_u[..., None], true_u[..., None]), axis=2)
    true_v = concatenate((true_v[..., None], true_v[..., None]), axis=2)
    assert_array_almost_equal(ug, true_u.T, 2)
    assert_array_almost_equal(vg, true_v.T, 2)


def test_geostrophic_gempak():
    """Test of geostrophic wind calculation against gempak values."""
    z = np.array([[5586387.00, 5584467.50, 5583147.50],
                  [5594407.00, 5592487.50, 5591307.50],
                  [5604707.50, 5603247.50, 5602527.50]]).T * (9.80616 * units('m/s^2')) * 1e-3
    dx = np.deg2rad(0.25) * Re * np.cos(np.deg2rad(44))
    dy = -np.deg2rad(0.25) * Re
    lat = 44 * units.degrees
    ug, vg = geostrophic_wind(z.T * units.m, dx.T, dy.T, lat)
    true_u = np.array([[21.97512, 21.97512, 22.08005],
                       [31.89402, 32.69477, 33.73863],
                       [38.43922, 40.18805, 42.14609]])
    true_v = np.array([[-10.93621, -7.83859, -4.54839],
                       [-10.74533, -7.50152, -3.24262],
                       [-8.66612, -5.27816, -1.45282]])
    assert_almost_equal(ug[1, 1], true_u[1, 1] * units('m/s'), 2)
    assert_almost_equal(vg[1, 1], true_v[1, 1] * units('m/s'), 2)


def test_no_ageostrophic_geopotential():
    """Test the updated ageostrophic wind function."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
    u = np.array([[-27.427, 0, 27.427]] * 3) * units('m/s')
    v = -u.T
    latitude = 30 * units.degrees
    uag, vag = ageostrophic_wind(z.T, u.T, v.T, 100. * units.kilometer,
                                 100. * units.kilometer, latitude)
    true = np.array([[0, 0, 0]] * 3) * units('m/s')
    assert_array_almost_equal(uag, true.T, 2)
    assert_array_almost_equal(vag, true.T, 2)


def test_ageostrophic_geopotential():
    """Test ageostrophic wind calculation with future input variable order."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
    u = v = np.array([[0, 0, 0]] * 3) * units('m/s')
    latitude = 30 * units.degrees
    uag, vag = ageostrophic_wind(z.T, u.T, v.T, 100. * units.kilometer,
                                 100. * units.kilometer, latitude)
    u_true = np.array([[27.427, 0, -27.427]] * 3) * units('m/s')
    v_true = -u_true.T
    assert_array_almost_equal(uag, u_true.T, 2)
    assert_array_almost_equal(vag, v_true.T, 2)


def test_streamfunc():
    """Test of Montgomery Streamfunction calculation."""
    t = 287. * units.kelvin
    hgt = 5000. * units.meter
    msf = montgomery_streamfunction(hgt, t)
    assert_almost_equal(msf, 337372.45469 * units('m^2 s^-2'), 4)


def test_storm_relative_helicity_no_storm_motion():
    """Test storm relative helicity with no storm motion and differing input units."""
    u = np.array([0, 20, 10, 0]) * units('m/s')
    v = np.array([20, 0, 0, 10]) * units('m/s')
    u = u.to('knots')
    heights = np.array([0, 250, 500, 750]) * units.m

    positive_srh, negative_srh, total_srh = storm_relative_helicity(heights, u, v,
                                                                    depth=750 * units.meters)

    assert_almost_equal(positive_srh, 400. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(negative_srh, -100. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(total_srh, 300. * units('meter ** 2 / second ** 2 '), 6)


def test_storm_relative_helicity_storm_motion():
    """Test storm relative helicity with storm motion and differing input units."""
    u = np.array([5, 25, 15, 5]) * units('m/s')
    v = np.array([30, 10, 10, 20]) * units('m/s')
    u = u.to('knots')
    heights = np.array([0, 250, 500, 750]) * units.m

    pos_srh, neg_srh, total_srh = storm_relative_helicity(heights, u, v,
                                                          depth=750 * units.meters,
                                                          storm_u=5 * units('m/s'),
                                                          storm_v=10 * units('m/s'))

    assert_almost_equal(pos_srh, 400. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(neg_srh, -100. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(total_srh, 300. * units('meter ** 2 / second ** 2 '), 6)


def test_storm_relative_helicity_with_interpolation():
    """Test storm relative helicity with interpolation."""
    u = np.array([-5, 15, 25, 15, -5]) * units('m/s')
    v = np.array([40, 20, 10, 10, 30]) * units('m/s')
    u = u.to('knots')
    heights = np.array([0, 100, 200, 300, 400]) * units.m

    pos_srh, neg_srh, total_srh = storm_relative_helicity(heights, u, v,
                                                          bottom=50 * units.meters,
                                                          depth=300 * units.meters,
                                                          storm_u=5 * units('m/s'),
                                                          storm_v=10 * units('m/s'))

    assert_almost_equal(pos_srh, 400. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(neg_srh, -100. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(total_srh, 300. * units('meter ** 2 / second ** 2 '), 6)


def test_storm_relative_helicity():
    """Test function for SRH calculations on an eigth-circle hodograph."""
    # Create larger arrays for everything except pressure to make a smoother graph
    hgt_int = np.arange(0, 2050, 50)
    hgt_int = hgt_int * units('meter')
    dir_int = np.arange(180, 272.25, 2.25)
    spd_int = np.zeros(hgt_int.shape[0])
    spd_int[:] = 2.
    u_int, v_int = wind_components(spd_int * units('m/s'), dir_int * units.degree)

    # Put in the correct value of SRH for a eighth-circle, 2 m/s hodograph
    # (SRH = 2 * area under hodo, in this case...)
    srh_true_p = (.25 * np.pi * (2 ** 2)) * units('m^2/s^2')

    # Since there's only positive SRH in this case, total SRH will be equal to positive SRH and
    # negative SRH will be zero.
    srh_true_t = srh_true_p
    srh_true_n = 0 * units('m^2/s^2')
    p_srh, n_srh, t_srh = storm_relative_helicity(hgt_int, u_int, v_int,
                                                  1000 * units('meter'),
                                                  bottom=0 * units('meter'),
                                                  storm_u=0 * units.knot,
                                                  storm_v=0 * units.knot)
    assert_almost_equal(p_srh, srh_true_p, 2)
    assert_almost_equal(n_srh, srh_true_n, 2)
    assert_almost_equal(t_srh, srh_true_t, 2)


def test_storm_relative_helicity_agl():
    """Test storm relative helicity with heights above ground."""
    u = np.array([-5, 15, 25, 15, -5]) * units('m/s')
    v = np.array([40, 20, 10, 10, 30]) * units('m/s')
    u = u.to('knots')
    heights = np.array([100, 200, 300, 400, 500]) * units.m

    pos_srh, neg_srh, total_srh = storm_relative_helicity(heights, u, v,
                                                          bottom=50 * units.meters,
                                                          depth=300 * units.meters,
                                                          storm_u=5 * units('m/s'),
                                                          storm_v=10 * units('m/s'))

    # Check that heights isn't modified--checks for regression of #789
    assert_almost_equal(heights[0], 100 * units.m, 6)
    assert_almost_equal(pos_srh, 400. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(neg_srh, -100. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(total_srh, 300. * units('meter ** 2 / second ** 2 '), 6)


def test_storm_relative_helicity_masked():
    """Test that srh does not return masked values."""
    h = units.Quantity(np.ma.array([20.72, 234.85, 456.69, 683.21]), units.meter)
    u = units.Quantity(np.ma.array(np.zeros((4,))), units.knot)
    v = units.Quantity(np.zeros_like(u), units.knot)
    pos, neg, com = storm_relative_helicity(h, u, v, depth=500 * units.meter,
                                            storm_u=15.77463015050421 * units('m/s'),
                                            storm_v=21.179437759755647 * units('m/s'))

    assert not np.ma.is_masked(pos)
    assert not np.ma.is_masked(neg)
    assert not np.ma.is_masked(com)


def test_absolute_vorticity_asym():
    """Test absolute vorticity calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    lats = np.array([[30, 30, 30], [20, 20, 20], [10, 10, 10]]) * units.degrees
    vort = absolute_vorticity(u, v, 1 * units.meters, 2 * units.meters, lats)
    true_vort = np.array([[-2.499927, 3.500073, 13.00007],
                          [8.500050, -1.499950, -10.99995],
                          [-5.499975, -1.499975, 2.532525e-5]]) / units.sec
    assert_almost_equal(vort, true_vort, 5)


@pytest.fixture
def pv_data():
    """Test data for all PV testing."""
    u = np.array([[[100, 90, 80, 70],
                   [90, 80, 70, 60],
                   [80, 70, 60, 50],
                   [70, 60, 50, 40]],
                  [[100, 90, 80, 70],
                   [90, 80, 70, 60],
                   [80, 70, 60, 50],
                   [70, 60, 50, 40]],
                  [[100, 90, 80, 70],
                   [90, 80, 70, 60],
                   [80, 70, 60, 50],
                   [70, 60, 50, 40]]]) * units('m/s')

    v = np.zeros_like(u) * units('m/s')

    lats = np.array([[40, 40, 40, 40],
                     [40.1, 40.1, 40.1, 40.1],
                     [40.2, 40.2, 40.2, 40.2],
                     [40.3, 40.3, 40.3, 40.3]]) * units.degrees

    lons = np.array([[40, 39.9, 39.8, 39.7],
                     [40, 39.9, 39.8, 39.7],
                     [40, 39.9, 39.8, 39.7],
                     [40, 39.9, 39.8, 39.7]]) * units.degrees

    dx, dy = lat_lon_grid_deltas(lons, lats)

    return u, v, lats, lons, dx, dy


def test_potential_vorticity_baroclinic_unity_axis0(pv_data):
    """Test potential vorticity calculation with unity stability and height on axis 0."""
    u, v, lats, _, dx, dy = pv_data

    potential_temperature = np.ones((3, 4, 4)) * units.kelvin
    potential_temperature[0] = 200 * units.kelvin
    potential_temperature[1] = 300 * units.kelvin
    potential_temperature[2] = 400 * units.kelvin

    pressure = np.ones((3, 4, 4)) * units.hPa
    pressure[2] = 1000 * units.hPa
    pressure[1] = 900 * units.hPa
    pressure[0] = 800 * units.hPa

    pvor = potential_vorticity_baroclinic(potential_temperature, pressure,
                                          u, v, dx[None, :, :], dy[None, :, :],
                                          lats[None, :, :])

    abs_vorticity = absolute_vorticity(u, v, dx[None, :, :], dy[None, :, :],
                                       lats[None, :, :])

    vort_difference = pvor - (abs_vorticity * g * (-1 * (units.kelvin / units.hPa)))

    true_vort = np.zeros_like(u) * (units.kelvin * units.meter**2
                                    / (units.second * units.kilogram))

    assert_almost_equal(vort_difference, true_vort, 10)


def test_potential_vorticity_baroclinic_non_unity_derivative(pv_data):
    """Test potential vorticity calculation with unity stability and height on axis 0."""
    u, v, lats, _, dx, dy = pv_data

    potential_temperature = np.ones((3, 4, 4)) * units.kelvin
    potential_temperature[0] = 200 * units.kelvin
    potential_temperature[1] = 300 * units.kelvin
    potential_temperature[2] = 400 * units.kelvin

    pressure = np.ones((3, 4, 4)) * units.hPa
    pressure[2] = 1000 * units.hPa
    pressure[1] = 999 * units.hPa
    pressure[0] = 998 * units.hPa

    pvor = potential_vorticity_baroclinic(potential_temperature, pressure,
                                          u, v, dx[None, :, :], dy[None, :, :],
                                          lats[None, :, :])

    abs_vorticity = absolute_vorticity(u, v, dx[None, :, :], dy[None, :, :],
                                       lats[None, :, :])

    vort_difference = pvor - (abs_vorticity * g * (-100 * (units.kelvin / units.hPa)))

    true_vort = np.zeros_like(u) * (units.kelvin * units.meter ** 2
                                    / (units.second * units.kilogram))

    assert_almost_equal(vort_difference, true_vort, 10)


def test_potential_vorticity_baroclinic_wrong_number_of_levels_axis_0(pv_data):
    """Test that potential vorticity calculation errors without 3 levels on axis 0."""
    u, v, lats, _, dx, dy = pv_data

    potential_temperature = np.ones((3, 4, 4)) * units.kelvin
    potential_temperature[0] = 200 * units.kelvin
    potential_temperature[1] = 300 * units.kelvin
    potential_temperature[2] = 400 * units.kelvin

    pressure = np.ones((3, 4, 4)) * units.hPa
    pressure[2] = 1000 * units.hPa
    pressure[1] = 900 * units.hPa
    pressure[0] = 800 * units.hPa

    with pytest.raises(ValueError):
        potential_vorticity_baroclinic(potential_temperature[:1, :, :], pressure, u, v,
                                       dx[None, :, :], dy[None, :, :],
                                       lats[None, :, :])

    with pytest.raises(ValueError):
        potential_vorticity_baroclinic(u, v, dx[None, :, :], dy[None, :, :],
                                       lats[None, :, :], potential_temperature,
                                       pressure[:1, :, :])


def test_potential_vorticity_baroclinic_isentropic_real_data():
    """Test potential vorticity calculation with real isentropic data."""
    isentlevs = [328, 330, 332] * units.K
    isentprs = np.array([[[245.88052, 245.79416, 245.68776, 245.52525, 245.31844],
                          [245.97734, 245.85878, 245.74838, 245.61089, 245.4683],
                          [246.4308, 246.24358, 246.08649, 245.93279, 245.80148],
                          [247.14348, 246.87215, 246.64842, 246.457, 246.32005],
                          [248.05727, 247.72388, 247.44029, 247.19205, 247.0112]],
                        [[239.66074, 239.60431, 239.53738, 239.42496, 239.27725],
                         [239.5676, 239.48225, 239.4114, 239.32259, 239.23781],
                         [239.79681, 239.6465, 239.53227, 239.43031, 239.35794],
                         [240.2442, 240.01723, 239.84442, 239.71255, 239.64021],
                         [240.85277, 240.57112, 240.34885, 240.17174, 240.0666]],
                        [[233.63297, 233.60493, 233.57542, 233.51053, 233.41898],
                         [233.35995, 233.3061, 233.27275, 233.23009, 233.2001],
                         [233.37685, 233.26152, 233.18793, 233.13496, 233.11841],
                         [233.57312, 233.38823, 233.26366, 233.18817, 233.17694],
                         [233.89297, 233.66039, 233.49615, 233.38635, 233.35281]]]) * units.hPa
    isentu = np.array([[[28.94226812, 28.53362902, 27.98145564, 27.28696092, 26.46488305],
                        [28.15024259, 28.12645242, 27.95788749, 27.62007338, 27.10351611],
                        [26.27821641, 26.55765132, 26.7329775, 26.77170719, 26.64779014],
                        [24.07215607, 24.48837805, 24.86738637, 25.17622757, 25.38030319],
                        [22.25524153, 22.65568001, 23.07333679, 23.48542321, 23.86341343]],
                       [[28.50078095, 28.12605738, 27.6145395, 26.96565679, 26.1919881],
                        [27.73718892, 27.73189078, 27.58886228, 27.28329365, 26.80468118],
                        [25.943111, 26.23034592, 26.41833632, 26.47466534, 26.37320009],
                        [23.82858821, 24.24937503, 24.63505859, 24.95235053, 25.16669265],
                        [22.09498322, 22.5008718, 22.9247538, 23.34295878, 23.72623895]],
                       [[28.05929378, 27.71848573, 27.24762337, 26.64435265, 25.91909314],
                        [27.32413525, 27.33732915, 27.21983708, 26.94651392, 26.50584625],
                        [25.60800559, 25.90304052, 26.10369515, 26.17762349, 26.09861004],
                        [23.58502035, 24.01037201, 24.4027308, 24.72847348, 24.95308212],
                        [21.9347249, 22.34606359, 22.77617081, 23.20049435,
                         23.58906447]]]) * units('m/s')
    isentv = np.array([[[-2.22336191, -2.82451946, -3.27190475, -3.53076527, -3.59311591],
                        [-2.12438321, -2.98895919, -3.73633746, -4.32254411, -4.70849598],
                        [-1.24050415, -2.31904635, -3.32284815, -4.20895826, -4.93036136],
                        [0.32254009, -0.89843808, -2.09621275, -3.2215678, -4.2290825],
                        [2.14238865, 0.88207403, -0.40652485, -1.67244834, -2.86837275]],
                       [[-1.99024801, -2.59146057, -3.04973279, -3.3296825, -3.42137476],
                        [-1.8711102, -2.71865804, -3.45952099, -4.05064148, -4.45309013],
                        [-0.99367383, -2.04299168, -3.02642031, -3.90252563, -4.62540783],
                        [0.547778, -0.63635567, -1.80391109, -2.90776869, -3.90375721],
                        [2.33967328, 1.12072805, -0.13066324, -1.3662872, -2.5404749]],
                       [[-1.75713411, -2.35840168, -2.82756083, -3.12859972, -3.24963361],
                        [-1.6178372, -2.44835688, -3.18270452, -3.77873886, -4.19768429],
                        [-0.7468435, -1.76693701, -2.72999246, -3.596093, -4.32045429],
                        [0.7730159, -0.37427326, -1.51160943, -2.59396958, -3.57843192],
                        [2.53695791, 1.35938207, 0.14519838, -1.06012605,
                         -2.21257705]]]) * units('m/s')
    lats = np.array([57.5, 57., 56.5, 56., 55.5]) * units.degrees
    lons = np.array([227.5, 228., 228.5, 229., 229.5]) * units.degrees

    dx, dy = lat_lon_grid_deltas(lons, lats)

    pvor = potential_vorticity_baroclinic(isentlevs[:, None, None], isentprs,
                                          isentu, isentv, dx[None, :, :], dy[None, :, :],
                                          lats[None, :, None])

    true_pv = np.array([[[2.97116898e-06, 3.38486331e-06, 3.81432403e-06, 4.24722471e-06,
                          4.64995688e-06],
                         [2.04235589e-06, 2.35739554e-06, 2.71138003e-06, 3.11803005e-06,
                          3.54655984e-06],
                         [1.41179481e-06, 1.60663306e-06, 1.85439220e-06, 2.17827401e-06,
                          2.55309150e-06],
                         [1.25933892e-06, 1.31915377e-06, 1.43444064e-06, 1.63067920e-06,
                          1.88631658e-06],
                         [1.37533104e-06, 1.31658998e-06, 1.30424716e-06, 1.36777872e-06,
                          1.49289942e-06]],
                        [[3.07674708e-06, 3.48172482e-06, 3.90371030e-06, 4.33207155e-06,
                         4.73253199e-06],
                         [2.16369614e-06, 2.47112604e-06, 2.81747901e-06, 3.21722053e-06,
                          3.63944011e-06],
                         [1.53925419e-06, 1.72853221e-06, 1.97026966e-06, 2.28774012e-06,
                          2.65577906e-06],
                         [1.38675388e-06, 1.44221972e-06, 1.55296146e-06, 1.74439951e-06,
                          1.99486345e-06],
                         [1.50312413e-06, 1.44039769e-06, 1.42422805e-06, 1.48403040e-06,
                          1.60544869e-06]],
                        [[3.17979446e-06, 3.57430736e-06, 3.98713951e-06, 4.40950119e-06,
                         4.80650246e-06],
                         [2.28618901e-06, 2.58455503e-06, 2.92172357e-06, 3.31292186e-06,
                          3.72721632e-06],
                         [1.67022518e-06, 1.85294576e-06, 2.08747504e-06, 2.39710083e-06,
                          2.75677598e-06],
                         [1.51817109e-06, 1.56879550e-06, 1.67430213e-06, 1.85997008e-06,
                          2.10409000e-06],
                         [1.63449148e-06, 1.56773336e-06, 1.54753266e-06, 1.60313832e-06,
                         1.72018062e-06]]]) * (units.kelvin * units.meter ** 2
                                               / (units.second * units.kilogram))

    assert_almost_equal(pvor, true_pv, 10)


def test_potential_vorticity_baroclinic_isobaric_real_data():
    """Test potential vorticity calculation with real isentropic data."""
    pressure = [20000., 25000., 30000.] * units.Pa
    theta = np.array([[[344.45776, 344.5063, 344.574, 344.6499, 344.735],
                       [343.98444, 344.02536, 344.08682, 344.16284, 344.2629],
                       [343.58792, 343.60876, 343.65628, 343.72818, 343.82834],
                       [343.21542, 343.2204, 343.25833, 343.32935, 343.43414],
                       [342.85272, 342.84982, 342.88556, 342.95645, 343.0634]],
                      [[326.70923, 326.67603, 326.63416, 326.57153, 326.49155],
                       [326.77695, 326.73468, 326.6931, 326.6408, 326.58405],
                       [326.95062, 326.88986, 326.83627, 326.78134, 326.7308],
                       [327.1913, 327.10928, 327.03894, 326.97546, 326.92587],
                       [327.47235, 327.3778, 327.29468, 327.2188, 327.15973]],
                      [[318.47897, 318.30374, 318.1081, 317.8837, 317.63837],
                       [319.155, 318.983, 318.79745, 318.58905, 318.36212],
                       [319.8042, 319.64206, 319.4669, 319.2713, 319.0611],
                       [320.4621, 320.3055, 320.13373, 319.9425, 319.7401],
                       [321.1375, 320.98648, 320.81473, 320.62186, 320.4186]]]) * units.K
    uwnd = np.array([[[25.309322, 25.169882, 24.94082, 24.61212, 24.181437],
                      [24.849028, 24.964956, 24.989666, 24.898415, 24.673553],
                      [23.666418, 24.003235, 24.269922, 24.435743, 24.474638],
                      [22.219162, 22.669518, 23.09492, 23.460283, 23.731855],
                      [21.065105, 21.506243, 21.967466, 22.420042, 22.830257]],
                     [[29.227198, 28.803436, 28.23203, 27.516447, 26.670708],
                      [28.402836, 28.376076, 28.199024, 27.848948, 27.315084],
                      [26.454042, 26.739328, 26.916056, 26.952703, 26.822044],
                      [24.17064, 24.59482, 24.979027, 25.290913, 25.495026],
                      [22.297522, 22.70384, 23.125736, 23.541069, 23.921045]],
                     [[27.429195, 26.97554, 26.360558, 25.594944, 24.7073],
                      [26.959536, 26.842077, 26.56688, 26.118752, 25.50171],
                      [25.460867, 25.599699, 25.62171, 25.50819, 25.249628],
                      [23.6418, 23.920736, 24.130007, 24.255558, 24.28613],
                      [21.915337, 22.283215, 22.607704, 22.879448,
                       23.093569]]]) * units('m/s')
    vwnd = np.array([[[-0.3050951, -0.90105104, -1.4307652, -1.856761, -2.156073],
                      [-0.10017005, -0.82312256, -1.5097888, -2.1251845, -2.631675],
                      [0.6832816, -0.16461015, -1.0023694, -1.7991445, -2.5169075],
                      [2.0360851, 1.0960612, 0.13380499, -0.81640035, -1.718524],
                      [3.6074955, 2.654059, 1.6466523, 0.61709386, -0.39874703]],
                     [[-2.3738103, -2.9788015, -3.423631, -3.6743853, -3.7226477],
                      [-2.2792664, -3.159968, -3.917221, -4.507328, -4.8893175],
                      [-1.3700132, -2.4722757, -3.4953287, -4.3956766, -5.123884],
                      [0.2314668, -1.0151587, -2.2366724, -3.382317, -4.403803],
                      [2.0903401, 0.8078297, -0.5038105, -1.7920332, -3.0061343]],
                     [[-1.4415079, -1.7622383, -1.9080431, -1.8903408, -1.7376306],
                      [-1.5708634, -2.288579, -2.823628, -3.1583376, -3.285275],
                      [-0.9814599, -1.999404, -2.8674111, -3.550859, -4.0168552],
                      [0.07641177, -1.1033016, -2.1928647, -3.1449537, -3.9159832],
                      [1.2759045, 0.05043932, -1.1469103, -2.264961,
                       -3.2550638]]]) * units('m/s')
    lats = np.array([57.5, 57., 56.5, 56., 55.5]) * units.degrees
    lons = np.array([227.5, 228., 228.5, 229., 229.5]) * units.degrees

    dx, dy = lat_lon_grid_deltas(lons, lats)

    pvor = potential_vorticity_baroclinic(theta, pressure[:, None, None],
                                          uwnd, vwnd, dx[None, :, :], dy[None, :, :],
                                          lats[None, :, None])

    true_pv = np.array([[[4.29013406e-06, 4.61736108e-06, 4.97453387e-06, 5.36730237e-06,
                          5.75500645e-06],
                         [3.48415057e-06, 3.72492697e-06, 4.00658450e-06, 4.35128065e-06,
                          4.72701041e-06],
                         [2.87775662e-06, 3.01866087e-06, 3.21074864e-06, 3.47971854e-06,
                          3.79924194e-06],
                         [2.70274738e-06, 2.71627883e-06, 2.78699880e-06, 2.94197238e-06,
                          3.15685712e-06],
                         [2.81293318e-06, 2.70649941e-06, 2.65188277e-06, 2.68109532e-06,
                          2.77737801e-06]],
                        [[2.43090597e-06, 2.79248225e-06, 3.16783697e-06, 3.54497301e-06,
                         3.89481001e-06],
                         [1.61968826e-06, 1.88924405e-06, 2.19296648e-06, 2.54191855e-06,
                          2.91119712e-06],
                         [1.09089606e-06, 1.25384007e-06, 1.46192044e-06, 1.73476959e-06,
                          2.05268876e-06],
                         [9.72047256e-07, 1.02016741e-06, 1.11466014e-06, 1.27721014e-06,
                          1.49122340e-06],
                         [1.07501523e-06, 1.02474621e-06, 1.01290749e-06, 1.06385170e-06,
                          1.16674712e-06]],
                        [[6.10254835e-07, 7.31519400e-07, 8.55731472e-07, 9.74301226e-07,
                         1.08453329e-06],
                         [3.17052987e-07, 3.98799900e-07, 4.91789955e-07, 5.96021549e-07,
                          7.10773939e-07],
                         [1.81983099e-07, 2.26503437e-07, 2.83058115e-07, 3.56549337e-07,
                          4.47098851e-07],
                         [1.54729567e-07, 1.73825926e-07, 2.01823376e-07, 2.44513805e-07,
                          3.02525735e-07],
                         [1.55220676e-07, 1.63334569e-07, 1.76335524e-07, 1.98346439e-07,
                          2.30155553e-07]]]) * (units.kelvin * units.meter ** 2
                                                / (units.second * units.kilogram))

    assert_almost_equal(pvor, true_pv, 10)


def test_potential_vorticity_baroclinic_4d(data_4d):
    """Test potential vorticity calculation with latlon+xarray spatial handling."""
    theta = potential_temperature(data_4d.pressure, data_4d.temperature)
    pvor = potential_vorticity_baroclinic(theta, data_4d.pressure, data_4d.u, data_4d.v)

    truth = np.array([
        [[[2.02341517e-07, 1.08253899e-06, 5.07866020e-07, 7.59602062e-07],
          [5.10389680e-07, 6.85689387e-07, 8.21670367e-07, 7.07634816e-07],
          [1.32493368e-06, 7.42556664e-07, 6.56995963e-07, 1.42860463e-06],
          [3.98119942e-07, 1.44178504e-06, 1.00098404e-06, 1.32741769e-07]],
         [[3.78824281e-07, 8.69275146e-07, 8.85194259e-07, 6.71317237e-07],
          [6.98417346e-07, 9.07612472e-07, 9.43897715e-07, 7.86981464e-07],
          [1.14118467e-06, 5.46283726e-07, 8.51417036e-07, 1.47484547e-06],
          [6.09694315e-07, 8.92755943e-07, 8.21736234e-07, 2.19146777e-07]],
         [[5.45372476e-07, 8.65038943e-07, 1.02542271e-06, 7.01655222e-07],
          [9.09010760e-07, 1.14690318e-06, 9.52200248e-07, 8.39364616e-07],
          [1.30601001e-06, 5.13731599e-07, 9.45482183e-07, 1.12678378e-06],
          [1.41700436e-06, 5.34416471e-07, 5.77202761e-07, 8.00215780e-07]]],
        [[[4.89875284e-07, 7.41732002e-07, 4.00156659e-07, 4.51659753e-07],
          [4.92109734e-07, 5.00766168e-07, 4.65459579e-07, 6.57429624e-07],
          [5.25432209e-07, 4.65439077e-07, 5.95175649e-07, 6.15264682e-07],
          [5.31988096e-07, 6.02477834e-07, 5.69272740e-07, 4.23351696e-07]],
         [[5.14269220e-07, 7.78503321e-07, 6.11304383e-07, 5.15249894e-07],
          [4.46066171e-07, 5.87690456e-07, 5.40874995e-07, 5.20729202e-07],
          [5.54138102e-07, 4.80436803e-07, 5.44944125e-07, 7.67293518e-07],
          [5.50869543e-07, 5.67508510e-07, 6.15430155e-07, 7.11393271e-07]],
         [[4.62763045e-07, 7.58095696e-07, 5.71561539e-07, 5.09461534e-07],
          [4.00198925e-07, 5.65386246e-07, 6.59228506e-07, 5.21051149e-07],
          [4.86756849e-07, 4.51122732e-07, 5.54841504e-07, 6.37263135e-07],
          [4.97103017e-07, 3.76458794e-07, 3.84346823e-07, 6.33177143e-07]]],
        [[[3.67414624e-07, 3.11634409e-07, 4.63243895e-07, 3.57094992e-07],
          [3.09361430e-07, 3.77719588e-07, 2.44198465e-07, 4.83354174e-07],
          [5.69920205e-08, 4.16754253e-07, 6.39950078e-07, 1.01328837e-07],
          [2.56285156e-07, 2.35613341e-07, 4.95745172e-07, 5.31565087e-07]],
         [[4.91680068e-07, 4.55365178e-07, 4.76828376e-07, 4.27773462e-07],
          [3.43227964e-07, 3.21022454e-07, 2.81916434e-07, 4.21074000e-07],
          [2.65819971e-07, 5.26528676e-07, 4.79102139e-07, 2.74517652e-07],
          [2.22251840e-07, 3.44727929e-07, 7.41995750e-07, 4.76425941e-07]],
         [[3.16830323e-07, 4.45198415e-07, 4.82149658e-07, 4.92118755e-07],
          [2.47719020e-07, 1.13643951e-07, 4.11871361e-07, 4.19639595e-07],
          [1.14884698e-07, 4.59177263e-07, 3.22239409e-07, 3.14475957e-07],
          [6.39184081e-08, 3.11908917e-07, 6.38295102e-07, 4.58138799e-07]]]]
    ) * units('K * m ** 2 / s / kg')

    assert_array_almost_equal(pvor, truth, 10)


def test_potential_vorticity_barotropic(pv_data):
    """Test the barotopic (Rossby) potential vorticity."""
    u, v, lats, _, dx, dy = pv_data

    u = u[0]
    v = v[0]

    heights = np.ones_like(u) * 3 * units.km
    pv = potential_vorticity_barotropic(heights, u, v, dx, dy, lats)
    avor = absolute_vorticity(u, v, dx, dy, lats)
    truth = avor / heights
    assert_almost_equal(pv, truth, 10)


def test_inertial_advective_wind_diffluent():
    """Test inertial advective wind with a diffluent flow."""
    lats = np.array([[50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.],
                     [48., 48., 48., 48., 48., 48., 48., 48., 48., 48., 48.],
                     [46., 46., 46., 46., 46., 46., 46., 46., 46., 46., 46.],
                     [44., 44., 44., 44., 44., 44., 44., 44., 44., 44., 44.],
                     [42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.],
                     [40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40.],
                     [38., 38., 38., 38., 38., 38., 38., 38., 38., 38., 38.],
                     [36., 36., 36., 36., 36., 36., 36., 36., 36., 36., 36.],
                     [34., 34., 34., 34., 34., 34., 34., 34., 34., 34., 34.],
                     [32., 32., 32., 32., 32., 32., 32., 32., 32., 32., 32.],
                     [30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30.]]) * units.degrees

    lons = np.array([[250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286., 290.],
                     [250., 254., 258., 262., 266., 270., 274., 278., 282., 286.,
                      290.]]) * units.degrees

    ug = np.array([[23.68206888, 23.28736773, 22.49796543, 21.70856314, 19.7350574,
                    17.36685051, 14.20924133, 10.26222985, 7.49932181, 4.34171263,
                    2.36820689],
                   [24.4118194, 24.00495574, 23.19122843, 22.37750111, 20.34318283,
                    17.90200089, 14.64709164, 10.57845507, 7.73040948, 4.47550022,
                    2.44118194],
                   [25.21967679, 24.79934884, 23.95869295, 23.11803706, 21.01639732,
                    18.49442965, 15.13180607, 10.92852661, 7.98623098, 4.62360741,
                    2.52196768],
                   [26.11573982, 25.68047749, 24.80995283, 23.93942817, 21.76311652,
                    19.15154253, 15.66944389, 11.31682059, 8.26998428, 4.78788563,
                    2.61157398],
                   [27.11207213, 26.66020426, 25.75646853, 24.85273279, 22.59339344,
                    19.88218623, 16.26724328, 11.74856459, 8.58548951, 4.97054656,
                    2.71120721],
                   [28.22319067, 27.75280415, 26.81203113, 25.87125811, 23.51932555,
                    20.69700649, 16.9339144, 12.23004929, 8.93734371, 5.17425162,
                    2.82231907],
                   [29.46670856, 28.97559675, 27.99337313, 27.01114951, 24.55559047,
                    21.60891961, 17.68002514, 12.76890704, 9.33112438, 5.4022299,
                    2.94667086],
                   [30.86419265, 30.34978944, 29.32098302, 28.2921766, 25.72016054,
                    22.63374128, 18.51851559, 13.37448348, 9.77366101, 5.65843532,
                    3.08641927],
                   [32.44232384, 31.90161845, 30.82020765, 29.73879686, 27.03526987,
                    23.79103749, 19.46539431, 14.05834033, 10.27340255, 5.94775937,
                    3.24423238],
                   [34.23449286, 33.66391798, 32.52276821, 31.38161845, 28.52874405,
                    25.10529476, 20.54069571, 14.8349469, 10.84092274, 6.27632369, 3.42344929],
                   [36.28303453, 35.67831729, 34.46888281, 33.25944832, 30.23586211,
                    26.60755866, 21.76982072, 15.7226483, 11.4896276, 6.65188966,
                    3.62830345]]) * units('m/s')

    vg = np.array([[7.67648972e-01, 2.30294692e+00, 3.07059589e+00,
                    5.37354281e+00, 8.44413870e+00, 1.07470856e+01,
                    1.38176815e+01, 1.30500325e+01, 1.15147346e+01,
                    9.97943664e+00, 5.37354281e+00],
                   [6.08116408e-01, 1.82434923e+00, 2.43246563e+00,
                    4.25681486e+00, 6.68928049e+00, 8.51362972e+00,
                    1.09460954e+01, 1.03379789e+01, 9.12174613e+00,
                    7.90551331e+00, 4.25681486e+00],
                   [4.53862086e-01, 1.36158626e+00, 1.81544834e+00,
                    3.17703460e+00, 4.99248295e+00, 6.35406920e+00,
                    8.16951755e+00, 7.71565546e+00, 6.80793129e+00,
                    5.90020712e+00, 3.17703460e+00],
                   [3.02572579e-01, 9.07717738e-01, 1.21029032e+00,
                    2.11800806e+00, 3.32829837e+00, 4.23601611e+00,
                    5.44630643e+00, 5.14373385e+00, 4.53858869e+00,
                    3.93344353e+00, 2.11800806e+00],
                   [1.52025875e-01, 4.56077624e-01, 6.08103499e-01,
                    1.06418112e+00, 1.67228462e+00, 2.12836225e+00,
                    2.73646575e+00, 2.58443987e+00, 2.28038812e+00,
                    1.97633637e+00, 1.06418112e+00],
                   [-5.44403782e-13, 0.00000000e+00, 0.00000000e+00,
                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                    0.00000000e+00, 3.62935855e-13],
                   [-1.55819455e-01, -4.67458366e-01, -6.23277822e-01,
                    -1.09073619e+00, -1.71401401e+00, -2.18147238e+00,
                    -2.80475020e+00, -2.64893074e+00, -2.33729183e+00,
                    -2.02565292e+00, -1.09073619e+00],
                   [-3.17940982e-01, -9.53822947e-01, -1.27176393e+00,
                    -2.22558688e+00, -3.49735080e+00, -4.45117375e+00,
                    -5.72293768e+00, -5.40499670e+00, -4.76911473e+00,
                    -4.13323277e+00, -2.22558688e+00],
                   [-4.89187491e-01, -1.46756247e+00, -1.95674996e+00,
                    -3.42431243e+00, -5.38106240e+00, -6.84862487e+00,
                    -8.80537483e+00, -8.31618734e+00, -7.33781236e+00,
                    -6.35943738e+00, -3.42431243e+00],
                   [-6.72847961e-01, -2.01854388e+00, -2.69139184e+00,
                    -4.70993572e+00, -7.40132757e+00, -9.41987145e+00,
                    -1.21112633e+01, -1.14384153e+01, -1.00927194e+01,
                    -8.74702349e+00, -4.70993572e+00],
                   [-8.72878488e-01, -2.61863546e+00, -3.49151395e+00,
                    -6.11014941e+00, -9.60166336e+00, -1.22202988e+01,
                    -1.57118128e+01, -1.48389343e+01, -1.30931773e+01,
                    -1.13474203e+01, -6.11014941e+00]]) * units('m/s')

    uiaw_truth = np.array([[-1.42807415e+00, -8.84702475e-01, -1.16169714e+00,
                            -2.07178191e+00, -2.26651744e+00, -2.44307980e+00,
                            -2.13572115e+00, -1.07805246e+00, -7.66864343e-01,
                            -4.29350989e-01, 2.09863394e-01],
                           [-1.15466056e+00, -7.14539881e-01, -9.37868053e-01,
                            -1.67069607e+00, -1.82145232e+00, -1.95723406e+00,
                            -1.69677456e+00, -8.44795197e-01, -5.99128909e-01,
                            -3.31430392e-01, 1.74263065e-01],
                           [-8.85879800e-01, -5.47662808e-01, -7.18560490e-01,
                            -1.27868851e+00, -1.38965979e+00, -1.48894561e+00,
                            -1.28074427e+00, -6.29324678e-01, -4.45008769e-01,
                            -2.43265738e-01, 1.36907150e-01],
                           [-6.11536708e-01, -3.77851670e-01, -4.95655649e-01,
                            -8.81515449e-01, -9.56332337e-01, -1.02300759e+00,
                            -8.76092304e-01, -4.27259697e-01, -3.01610757e-01,
                            -1.63732062e-01, 9.57321746e-02],
                           [-3.20542252e-01, -1.98032517e-01, -2.59762867e-01,
                            -4.61930812e-01, -5.00960635e-01, -5.35715150e-01,
                            -4.58376210e-01, -2.23205436e-01, -1.57510625e-01,
                            -8.53847182e-02, 5.03061160e-02],
                           [-7.17595005e-13, -2.36529156e-13, -0.00000000e+00,
                            -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
                            -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
                            -2.93991041e-14, -6.68646808e-14],
                           [3.66014877e-01, 2.26381410e-01, 2.97076576e-01,
                            5.28911772e-01, 5.75670994e-01, 6.17640324e-01,
                            5.33241822e-01, 2.63664097e-01, 1.86703719e-01,
                            1.02644563e-01, -5.59431414e-02],
                           [7.98146331e-01, 4.94284987e-01, 6.48956331e-01,
                            1.15693361e+00, 1.26429103e+00, 1.36142948e+00,
                            1.18700769e+00, 5.96585261e-01, 4.23976442e-01,
                            2.36489429e-01, -1.18303960e-01],
                           [1.32422955e+00, 8.21548216e-01, 1.07935767e+00,
                            1.92781420e+00, 2.11849275e+00, 2.29274320e+00,
                            2.02576077e+00, 1.04018545e+00, 7.42658541e-01,
                            4.21848012e-01, -1.87693140e-01],
                           [1.98305622e+00, 1.23316526e+00, 1.62158055e+00,
                            2.90329131e+00, 3.21355659e+00, 3.50025687e+00,
                            3.14455187e+00, 1.65685097e+00, 1.18935849e+00,
                            6.89756719e-01, -2.64167230e-01],
                           [2.83017758e+00, 1.76405766e+00, 2.32173329e+00,
                            4.16683333e+00, 4.64487467e+00, 5.09076175e+00,
                            4.64597480e+00, 2.50595982e+00, 1.80748965e+00,
                            1.06712544e+00, -3.52915793e-01]]) * units('m/s')

    viaw_truth = np.array([[-0.16767916, -0.49465351, -0.63718079, -1.07594125, -1.53705893,
                            -1.721506, -1.81093489, -1.23523645, -0.79647599, -0.39963532,
                            -0.11737541],
                           [-0.17337355, -0.51145198, -0.6588195, -1.1124803, -1.58925758,
                            -1.77996849, -1.87243438, -1.27718518, -0.82352438, -0.41320697,
                            -0.12136149],
                           [-0.18010801, -0.53131862, -0.68441043, -1.15569305, -1.65099008,
                            -1.84910889, -1.94516649, -1.32679566, -0.85551304, -0.42925742,
                            -0.12607561],
                           [-0.18806768, -0.55479966, -0.71465719, -1.20676763, -1.72395376,
                            -1.93082821, -2.03113097, -1.38543193, -0.89332149, -0.44822798,
                            -0.13164738],
                           [-0.19730148, -0.58203938, -0.74974564, -1.26601785, -1.80859693,
                            -2.02562856, -2.13085602, -1.45345426, -0.93718205, -0.4702352,
                            -0.13811104],
                           [-0.2078345, -0.61311178, -0.78977111, -1.33360472, -1.90514961,
                            -2.13376756, -2.24461263, -1.5310475, -0.98721389, -0.4953389,
                            -0.14548415],
                           [-0.21963486, -0.64792283, -0.83461247, -1.40932368, -2.01331954,
                            -2.25491789, -2.37205648, -1.6179768, -1.04326558, -0.52346308,
                            -0.1537444],
                           [-0.2325551, -0.68603755, -0.88370939, -1.49222857, -2.1317551,
                            -2.38756571, -2.5115951, -1.71315592, -1.10463673, -0.55425633,
                            -0.16278857],
                           [-0.24622751, -0.72637116, -0.93566454, -1.57995986, -2.25708551,
                            -2.52793577, -2.65925711, -1.81387599, -1.16958067, -0.58684223,
                            -0.17235926],
                           [-0.25987451, -0.76662981, -0.98752314, -1.66752812, -2.38218302,
                            -2.66804499, -2.80664473, -1.9144089, -1.23440393, -0.61936759,
                            -0.18191216],
                           [-0.27342538, -0.80660487, -1.03901645, -1.75447953, -2.50639932,
                            -2.80716724, -2.95299411, -2.01423364, -1.29877056, -0.65166382,
                            -0.19139777]]) * units('m/s')

    dx, dy = lat_lon_grid_deltas(lons, lats)
    uiaw, viaw = inertial_advective_wind(ug, vg, ug, vg, dx, dy, lats)
    assert_almost_equal(uiaw, uiaw_truth, 5)
    assert_almost_equal(viaw, viaw_truth, 5)


@pytest.fixture
def q_vector_data():
    """Define data for use in Q-vector tests."""
    speed = np.ones((4, 4)) * 50. * units('knots')
    wdir = np.array([[210., 190., 170., 150.],
                     [190., 180., 180., 170.],
                     [170., 180., 180., 190.],
                     [150., 170., 190., 210.]]) * units('degrees')
    u, v = wind_components(speed, wdir)

    temp = np.array([[[18., 18., 18., 18.],
                      [17., 17., 17., 17.],
                      [17., 17., 17., 17.],
                      [16., 16., 16., 16.]],
                     [[12., 11., 10., 9.],
                      [11., 10.5, 10.5, 10.],
                      [10., 10.5, 10.5, 11.],
                      [9., 10., 11., 12.]],
                     [[-10., -10., -10., -10.],
                      [-10., -10., -10., -10.],
                      [-11., -11., -11., -11.],
                      [-11., -11., -11., -11.]]]) * units('degC')

    p = np.array([850., 700., 500.]) * units('hPa')

    lats = np.linspace(35., 40., 4) * units('degrees')
    lons = np.linspace(-100., -90., 4) * units('degrees')
    dx, dy = lat_lon_grid_deltas(lons, lats)

    return u, v, temp, p, dx, dy


def test_q_vector_without_static_stability(q_vector_data):
    """Test the Q-vector function without using static stability."""
    u, v, temp, p, dx, dy = q_vector_data

    # Treating as 700 hPa data
    q1, q2 = q_vector(u, v, temp[1], p[1], dx, dy)

    q1_truth = (np.array([[-2.7454089e-14, -3.0194267e-13, -3.0194267e-13, -2.7454089e-14],
                          [-1.8952185e-13, -2.2269905e-14, -2.2269905e-14, -1.8952185e-13],
                          [-1.9918390e-13, -2.3370829e-14, -2.3370829e-14, -1.9918390e-13],
                          [-5.6160772e-14, -3.5145951e-13, -3.5145951e-13, -5.6160772e-14]])
                * units('m^2 kg^-1 s^-1'))
    q2_truth = (np.array([[-4.4976059e-14, -4.3582378e-13, 4.3582378e-13, 4.4976059e-14],
                          [-3.0124244e-13, -3.5724617e-14, 3.5724617e-14, 3.0124244e-13],
                          [3.1216232e-13, 3.6662900e-14, -3.6662900e-14, -3.1216232e-13],
                          [8.6038280e-14, 4.6968342e-13, -4.6968342e-13, -8.6038280e-14]])
                * units('m^2 kg^-1 s^-1'))

    assert_almost_equal(q1, q1_truth, 16)
    assert_almost_equal(q2, q2_truth, 16)


def test_q_vector_with_static_stability(q_vector_data):
    """Test the Q-vector function using static stability."""
    u, v, temp, p, dx, dy = q_vector_data

    sigma = static_stability(p[:, np.newaxis, np.newaxis], temp)

    # Treating as 700 hPa data
    q1, q2 = q_vector(u, v, temp[1], p[1], dx, dy, sigma[1])

    q1_truth = (np.array([[-1.4158140e-08, -1.6197987e-07, -1.6875014e-07, -1.6010616e-08],
                          [-9.3971386e-08, -1.1252476e-08, -1.1252476e-08, -9.7617234e-08],
                          [-1.0785670e-07, -1.2403513e-08, -1.2403513e-08, -1.0364793e-07],
                          [-2.9186946e-08, -1.7577703e-07, -1.6937879e-07, -2.6112047e-08]])
                * units('kg m^-2 s^-3'))
    q2_truth = (np.array([[-2.31770213e-08, -2.33621439e-07, 2.43378967e-07, 2.62072251e-08],
                          [-1.4936626e-07, -1.8050836e-08, 1.8050836e-08, 1.5516129e-07],
                          [1.6903373e-07, 1.9457964e-08, -1.9457964e-08, -1.6243771e-07],
                          [4.46812456e-08, 2.34736724e-07, -2.26197708e-07, -3.99768328e-08]])
                * units('kg m^-2 s^-3'))

    assert_almost_equal(q1, q1_truth, 10)
    assert_almost_equal(q2, q2_truth, 10)


@pytest.fixture
def data_4d():
    """Define 4D data (extracted from Irma GFS example) for testing kinematics functions."""
    data = xr.open_dataset(get_test_data('irma_gfs_example.nc', False))
    data = data.metpy.parse_cf()
    data['Geopotential_height_isobaric'].attrs['units'] = 'm'
    subset = data.drop_vars((
        'LatLon_361X720-0p25S-180p00E', 'Vertical_velocity_pressure_isobaric', 'isobaric1',
        'Relative_humidity_isobaric', 'reftime'

    )).sel(
        latitude=[46., 44., 42., 40.],
        longitude=[262., 267., 272., 277.],
        isobaric3=[50000., 70000., 85000.]
    ).isel(time1=[0, 1, 2])
    return subset.rename({
        'Geopotential_height_isobaric': 'height',
        'Temperature_isobaric': 'temperature',
        'isobaric3': 'pressure',
        'u-component_of_wind_isobaric': 'u',
        'v-component_of_wind_isobaric': 'v'
    })


true_vort4d = np.array([[[[-5.72939079e-05, 3.36008149e-05, 4.80394116e-05, 2.24754927e-05],
                          [2.28437884e-05, 2.16350819e-05, 4.40912008e-05, 7.21109010e-05],
                          [6.56150935e-05, 7.12554707e-05, 8.63437939e-05, 8.77146299e-05],
                          [-4.12479588e-05, 1.60707608e-04, 1.47465661e-04, -5.63406909e-05]],
                         [[1.22453259e-07, 4.40258958e-05, -8.22480293e-06, 1.54493600e-05],
                          [1.29420183e-05, 1.25760315e-05, 2.98881935e-05, 6.40671857e-05],
                          [1.96998118e-05, 8.78628308e-06, 3.96330962e-05, 4.88475149e-05],
                          [3.37810678e-05, 2.94602756e-05, 3.98077989e-05, 5.71554040e-05]],
                         [[-2.76821428e-05, 5.08462417e-06, 5.55937962e-06, 5.23436098e-05],
                          [-2.11754797e-05, 6.40524521e-06, 2.11226065e-05, 3.52627761e-05],
                          [-1.92494063e-05, -1.43439529e-06, 2.99489927e-05, 3.13418677e-05],
                          [-2.32787494e-05, -1.76993463e-05, 3.10941039e-05, 3.53835159e-05]]],
                        [[[-3.57414525e-05, 2.61424456e-05, 8.46799855e-05, 2.62297854e-05],
                          [3.41307192e-05, 2.48272187e-05, 4.93974252e-05, 7.85589219e-05],
                          [7.95962242e-05, 5.17417889e-05, 6.89810168e-05, 1.03949044e-04],
                          [-6.39992725e-07, 9.11570311e-05, 1.15816379e-04, -4.01350495e-05]],
                         [[-1.85416639e-06, 4.06009696e-05, 3.90917706e-05, 3.92211904e-05],
                          [-3.72155456e-06, 2.21444097e-05, 3.05974559e-05, 3.17910074e-05],
                          [1.64244406e-05, 9.33099989e-06, 2.59450976e-05, 7.20713763e-05],
                          [2.19198952e-05, 2.29714884e-05, 3.55228162e-05, 9.42695439e-05]],
                         [[-6.29026250e-06, -1.66926104e-06, 2.06531086e-05, 6.30024082e-05],
                          [-1.71967796e-05, 8.10200354e-06, 1.52458021e-05, 1.94769674e-05],
                          [-2.22495255e-06, 3.57057325e-06, 2.35516080e-05, 3.85710155e-05],
                          [-1.44681821e-05, -5.45860797e-06, 3.80976184e-05, 1.24881360e-05]]],
                        [[[-2.07301156e-05, 3.23990819e-05, 9.57142159e-05, 6.38114024e-05],
                          [2.92811973e-05, 2.88056901e-05, 4.70659778e-05, 8.20235562e-05],
                          [7.50632852e-05, 3.26235585e-05, 3.92811088e-05, 8.12137436e-05],
                          [7.16082561e-05, 2.43401051e-05, 7.43764563e-05, 7.33103146e-05]],
                         [[1.28299480e-08, 5.67151478e-05, 3.02790507e-05, 3.75851668e-05],
                          [-5.47604749e-06, 2.78629076e-05, 3.41596648e-05, 3.01239273e-05],
                          [9.66906328e-06, 7.80152347e-06, 2.20928721e-05, 5.18810534e-05],
                          [1.64696390e-05, 2.44849598e-06, -5.61052143e-06, 6.28005847e-05]],
                         [[3.76422464e-06, 3.03913454e-05, 3.42662513e-05, 4.60870862e-05],
                          [-2.50531945e-06, 9.38416716e-06, 1.46413567e-05, 1.94701388e-05],
                          [-5.24048728e-06, 3.21705642e-07, 7.17758181e-06, 1.95403688e-05],
                          [-2.47265560e-06, 4.73080463e-06, 6.29036551e-06,
                           2.84689950e-05]]]]) * units('s^-1')


def test_vorticity_4d(data_4d):
    """Test vorticity on a 4D (time, pressure, y, x) grid."""
    vort = vorticity(data_4d.u, data_4d.v)
    assert_array_almost_equal(vort.data, true_vort4d, 12)


def test_absolute_vorticity_4d(data_4d):
    """Test absolute_vorticity on a 4D (time, pressure, y, x) grid."""
    vort = absolute_vorticity(data_4d.u, data_4d.v)
    f = coriolis_parameter(data_4d.latitude).broadcast_like(vort)
    truth = true_vort4d + f.data
    assert_array_almost_equal(vort.data, truth, 12)


def test_divergence_4d(data_4d):
    """Test divergence on a 4D (time, pressure, y, x) grid."""
    div = divergence(data_4d.u, data_4d.v)
    truth = np.array([[[[-5.69109693e-06, -1.97918528e-06, 1.47453542e-05, 2.69697704e-05],
                        [3.20267932e-05, -6.19720681e-06, 1.25570333e-05, -7.10519011e-06],
                        [-4.28862128e-05, -1.91207500e-05, 1.39780734e-05, 4.55906339e-05],
                        [-3.10230392e-05, 1.65756168e-05, -4.24591337e-05, 3.82235500e-05]],
                       [[1.98223791e-05, 4.33913279e-06, 4.19627202e-05, -5.09003830e-05],
                        [2.56348274e-05, -1.55289420e-06, 6.96268077e-06, 1.70390048e-05],
                        [-3.52183670e-06, 3.78206345e-07, -1.34093219e-05, 2.90710519e-06],
                        [-2.84461615e-05, -6.53843845e-06, -2.54072285e-05, -2.81482021e-05]],
                       [[-1.21348077e-05, -1.93861224e-05, -1.93459201e-05, -3.87356806e-05],
                        [4.81616405e-06, -7.66038273e-06, 1.88430179e-07, 4.20022198e-06],
                        [1.66798208e-05, 5.65659378e-06, 6.33736697e-06, 5.67003948e-06],
                        [1.30753908e-05, 1.80197572e-05, 1.26966380e-05, 4.18043296e-06]]],
                      [[[-7.75829235e-07, -5.08426457e-06, 7.57910544e-06, 4.72124287e-05],
                        [2.57585409e-05, 1.71301607e-06, 5.83802681e-06, -3.33138015e-05],
                        [-2.91819759e-05, -7.49775551e-06, 2.63853084e-06, 2.33586676e-05],
                        [-6.76888907e-05, 1.76394873e-05, -5.08169287e-05, 2.85916802e-05]],
                       [[1.93044895e-05, 1.51461678e-05, -2.09465009e-05, -1.91221470e-05],
                        [2.02601342e-05, 7.55251174e-07, 4.86519855e-06, -5.99451216e-06],
                        [6.46768008e-06, -3.39133854e-06, 4.95963402e-06, 3.75958887e-06],
                        [-1.45155227e-06, 5.60979108e-06, -2.09967347e-05, 2.89704581e-05]],
                       [[-4.39050924e-06, 2.12833521e-06, -4.50196821e-05, 6.49783523e-06],
                        [8.22035480e-07, -4.71231966e-06, 2.45757249e-06, 2.41048520e-06],
                        [7.57532808e-06, -7.32507793e-07, 1.78057678e-05, 1.29309987e-05],
                        [-2.29661166e-06, 1.96837178e-05, 1.45078799e-06, 1.41496820e-05]]],
                      [[[3.16631969e-07, -1.24957659e-05, 1.23451304e-05, 9.09226076e-06],
                        [2.53440942e-05, 3.33772853e-07, 4.20355495e-06, -1.38016966e-05],
                        [1.66685173e-06, -1.25348400e-05, -7.29217984e-07, 1.40404816e-05],
                        [-7.16330286e-05, -2.04996415e-05, -6.39953567e-06, -1.13599582e-05]],
                       [[-6.14675217e-07, 2.05951752e-05, -1.43773812e-05, 5.83203981e-06],
                        [2.44795938e-05, 4.42280257e-06, -7.63592160e-06, -4.90036880e-06],
                        [9.02514162e-06, 6.51518845e-08, 5.88086792e-06, -8.59999454e-06],
                        [-3.99115438e-06, 2.05745950e-07, 1.42084579e-05, 2.83814897e-05]],
                       [[5.23848091e-06, -1.63679904e-06, -1.97566839e-05, 9.19774945e-06],
                        [1.32383435e-05, 1.42742942e-06, -8.96735083e-06, -7.41887021e-06],
                        [3.32715273e-06, 9.54519710e-07, 7.33022680e-06, -9.09165376e-06],
                        [2.24746232e-06, -1.69640129e-06, 1.80208289e-05,
                         -5.73083897e-06]]]]) * units('s^-1')
    assert_array_almost_equal(div.data, truth, 12)


def test_shearing_deformation_4d(data_4d):
    """Test shearing_deformation on a 4D (time, pressure, y, x) grid."""
    shdef = shearing_deformation(data_4d.u, data_4d.v)
    truth = np.array([[[[-2.22216294e-05, 5.27319738e-06, 2.91543418e-05, 1.30329364e-05],
                        [-6.25886934e-05, 1.21925428e-05, 1.98103919e-05, -2.09655345e-05],
                        [4.46342492e-06, -1.68748849e-05, -1.12290966e-05, 4.23005194e-05],
                        [6.66667593e-05, -1.03683458e-04, -9.12956532e-05, 7.72037279e-05]],
                       [[-2.32590651e-05, -8.58252633e-06, 5.74233121e-05, 1.06072378e-06],
                        [-1.44661146e-06, 1.12270967e-05, 2.17945891e-05, 3.52899261e-05],
                        [-5.92993188e-06, 1.86784643e-05, -3.53279109e-06, -1.09552232e-05],
                        [-2.33237922e-05, 1.05752016e-05, 2.39065363e-07, -5.03096678e-05]],
                       [[-2.49842754e-05, 1.13796431e-05, 4.69266814e-05, 1.53376235e-06],
                        [-1.48804543e-05, 7.30453364e-06, 2.47197602e-05, 2.67195275e-05],
                        [-2.16290910e-06, 1.25045903e-05, 9.26533963e-06, 3.17915141e-05],
                        [1.17935334e-05, 2.77147641e-05, -3.81014510e-07, 1.15523534e-05]]],
                      [[[6.21038574e-06, 1.20236024e-05, -1.57256625e-05, 3.85950088e-05],
                        [-3.56990901e-05, 9.80909130e-06, 5.73692616e-06, -2.15769374e-05],
                        [3.02174095e-06, 1.60641317e-06, 3.91744031e-06, 3.55580983e-05],
                        [2.10778238e-05, -2.83135572e-05, -4.87985007e-05, 6.74649144e-05]],
                       [[-2.47860362e-05, -1.78528251e-05, 8.96558477e-06, 9.09500677e-06],
                        [-1.49626706e-05, 1.04536473e-05, 2.11549168e-05, 7.95984267e-06],
                        [3.83438985e-06, 1.15792231e-05, 1.65025584e-05, 1.31679266e-05],
                        [-5.05877901e-06, 6.33465037e-06, 5.39663041e-06, -4.10734948e-05]],
                       [[-1.88803090e-05, 9.12220863e-06, 2.06531065e-05, 1.26422093e-05],
                        [-1.71967796e-05, 8.10200354e-06, 1.70443811e-05, -5.70312992e-06],
                        [1.12643867e-05, 4.46986382e-06, 8.26368981e-06, 4.30674619e-05],
                        [1.34097890e-05, 8.03073340e-06, -1.31618753e-05, 5.11575682e-05]]],
                      [[[-7.69041420e-06, 4.07147725e-06, -2.25423126e-05, 1.92965675e-05],
                        [-1.79314900e-05, 4.97452325e-06, 1.60404993e-05, 1.50265065e-05],
                        [2.67050064e-06, 2.04831498e-05, 1.90470999e-05, 1.33174097e-05],
                        [9.10766558e-06, 3.10847747e-05, -1.15056632e-05, 2.60976273e-05]],
                       [[-9.42970708e-06, -4.53541805e-05, 9.59539764e-06, 3.57865814e-05],
                        [-1.58178729e-05, -3.16257088e-06, 8.08027693e-06, 3.14524883e-06],
                        [-3.37063173e-06, 1.63447699e-05, 1.98446489e-05, 7.36623139e-06],
                        [-1.06650676e-06, 1.90853425e-05, 4.51993196e-05, 8.39356857e-06]],
                       [[-2.00669443e-05, -2.94113884e-05, -2.60460413e-06, 2.81012941e-05],
                        [-3.85425208e-06, 2.63949754e-06, 8.34633349e-06, 6.88009010e-06],
                        [6.45027402e-06, 1.15628217e-05, 1.66201167e-05, 1.68425036e-05],
                        [1.28152573e-05, -1.11457227e-06, 1.66321845e-05,
                         4.01597531e-05]]]]) * units('s^-1')
    assert_array_almost_equal(shdef.data, truth, 12)


def test_stretching_deformation_4d(data_4d):
    """Test stretching_deformation on a 4D (time, pressure, y, x) grid."""
    stdef = stretching_deformation(data_4d.u, data_4d.v)
    truth = np.array([[[[3.74747989e-05, 2.58987773e-05, -5.48865461e-06, -2.92358076e-05],
                        [-7.54193179e-06, 2.70764949e-05, 5.81236371e-06, 2.93160254e-05],
                        [-5.59259142e-05, 6.05935158e-06, 4.00574629e-05, 5.32345919e-05],
                        [9.17299272e-05, 2.01727791e-05, 3.57790347e-05, -1.04313800e-04]],
                       [[5.88340213e-06, 1.64795501e-05, -4.25704731e-05, 1.78952481e-05],
                        [2.15880187e-05, 7.88964498e-06, -7.42594688e-06, 7.59646711e-06],
                        [2.77318655e-06, 1.47668340e-05, 2.84076306e-05, -3.38792021e-06],
                        [-1.13596428e-05, 2.04402443e-05, 5.86763185e-05, 5.00899658e-05]],
                       [[3.14807173e-05, 3.63698156e-05, 1.75249482e-05, 5.77913742e-06],
                        [1.87551496e-05, 1.75197146e-05, 6.48345772e-06, 1.18441808e-05],
                        [1.03847975e-05, 9.70339384e-06, -3.55481427e-06, -1.59129031e-05],
                        [-4.01111070e-06, 1.03758035e-05, 1.00587992e-06, -3.89854532e-05]]],
                      [[[4.00968607e-05, 2.17145390e-05, -1.86342944e-06, -3.61966573e-05],
                        [-3.60325721e-06, 2.94111210e-05, 7.90639233e-06, 5.31067383e-06],
                        [-3.65111798e-05, 6.84591093e-06, 4.82774623e-05, 6.04093871e-05],
                        [3.92815820e-05, 4.37497554e-06, 7.04522343e-05, -5.95386733e-05]],
                       [[-2.72810481e-06, 1.65683717e-06, 2.71654705e-05, 6.95724033e-06],
                        [1.53140500e-05, 1.60431672e-05, -1.87947106e-06, 3.89766962e-06],
                        [1.09641265e-05, 1.68426660e-05, 7.65750143e-06, 8.70568056e-06],
                        [4.84344526e-06, 6.95872586e-06, 5.54428478e-05, 4.02115752e-05]],
                       [[1.94406533e-05, 8.42336276e-06, 4.31106607e-05, -2.90240924e-05],
                        [2.17096597e-06, 1.23741775e-05, 6.05472618e-06, 7.35657635e-06],
                        [7.12568173e-06, 8.26038502e-06, -6.47504105e-06, 1.02331313e-05],
                        [1.61388203e-05, 1.69793215e-06, 5.94724298e-06, -4.43041213e-05]]],
                      [[[3.35903423e-05, 3.02204835e-05, -3.39244060e-06, 1.89794480e-06],
                        [1.06327675e-06, 3.13592514e-05, 1.27468014e-05, -4.80880378e-06],
                        [-1.04735570e-05, 1.80409922e-05, 3.16451980e-05, 5.00120508e-05],
                        [-1.93809208e-06, 2.08676689e-05, 4.93564018e-05, 6.23817547e-05]],
                       [[-1.63522334e-05, -9.08137285e-06, 3.10367228e-05, -8.10694416e-06],
                        [1.86342126e-05, 1.07178258e-05, 8.10163869e-06, 7.24003618e-06],
                        [1.12733648e-05, 1.85005839e-05, 6.33051213e-06, 9.83543530e-06],
                        [4.55210065e-06, 6.95042414e-06, 1.37588137e-05, 3.33275803e-05]],
                       [[-2.40546855e-06, 1.23021865e-05, 2.47581446e-05, -2.49752420e-05],
                        [1.00908318e-05, 1.08699686e-05, 7.66950217e-06, -9.21744893e-06],
                        [8.27324121e-06, 1.08467010e-05, -7.63377597e-07, 4.84732988e-06],
                        [1.88843132e-05, 1.35915105e-05, -1.16557148e-05,
                         7.30885665e-06]]]]) * units('s^-1')
    assert_array_almost_equal(stdef.data, truth, 10)


def test_total_deformation_4d(data_4d):
    """Test total_deformation on a 4D (time, pressure, y, x) grid."""
    totdef = total_deformation(data_4d.u, data_4d.v)
    truth = np.array([[[[4.35678937e-05, 2.64301585e-05, 2.96664959e-05, 3.20092155e-05],
                        [6.30414568e-05, 2.96950278e-05, 2.06454644e-05, 3.60414065e-05],
                        [5.61037436e-05, 1.79297931e-05, 4.16015979e-05, 6.79945271e-05],
                        [1.13396809e-04, 1.05627651e-04, 9.80562880e-05, 1.29775901e-04]],
                       [[2.39916345e-05, 1.85805094e-05, 7.14820393e-05, 1.79266572e-05],
                        [2.16364331e-05, 1.37220333e-05, 2.30249604e-05, 3.60982714e-05],
                        [6.54634675e-06, 2.38105946e-05, 2.86264578e-05, 1.14671234e-05],
                        [2.59430292e-05, 2.30138757e-05, 5.86768055e-05, 7.09934317e-05]],
                       [[4.01901677e-05, 3.81085262e-05, 5.00922872e-05, 5.97920197e-06],
                        [2.39412522e-05, 1.89814807e-05, 2.55558559e-05, 2.92270041e-05],
                        [1.06076480e-05, 1.58278435e-05, 9.92387137e-06, 3.55516645e-05],
                        [1.24569836e-05, 2.95933345e-05, 1.07562376e-06, 4.06610677e-05]]],
                      [[[4.05749569e-05, 2.48211245e-05, 1.58356822e-05, 5.29128784e-05],
                        [3.58804752e-05, 3.10037467e-05, 9.76848819e-06, 2.22208794e-05],
                        [3.66360092e-05, 7.03186033e-06, 4.84361405e-05, 7.00975920e-05],
                        [4.45793376e-05, 2.86495712e-05, 8.57018727e-05, 8.99798216e-05]],
                       [[2.49357203e-05, 1.79295419e-05, 2.86067212e-05, 1.14508664e-05],
                        [2.14103162e-05, 1.91484192e-05, 2.12382418e-05, 8.86289591e-06],
                        [1.16152751e-05, 2.04390265e-05, 1.81926293e-05, 1.57855366e-05],
                        [7.00358530e-06, 9.41018921e-06, 5.57048740e-05, 5.74804554e-05]],
                       [[2.70999090e-05, 1.24164299e-05, 4.78025091e-05, 3.16579120e-05],
                        [1.73332721e-05, 1.47906299e-05, 1.80878588e-05, 9.30832458e-06],
                        [1.33289815e-05, 9.39221184e-06, 1.04983202e-05, 4.42665026e-05],
                        [2.09829446e-05, 8.20826733e-06, 1.44431528e-05, 6.76753422e-05]]],
                      [[[3.44594481e-05, 3.04935165e-05, 2.27961512e-05, 1.93896806e-05],
                        [1.79629867e-05, 3.17513547e-05, 2.04884983e-05, 1.57772143e-05],
                        [1.08086526e-05, 2.72953627e-05, 3.69352213e-05, 5.17547932e-05],
                        [9.31159348e-06, 3.74395890e-05, 5.06797266e-05, 6.76207769e-05]],
                       [[1.88763056e-05, 4.62544379e-05, 3.24861481e-05, 3.66933503e-05],
                        [2.44425650e-05, 1.11746877e-05, 1.14423522e-05, 7.89371358e-06],
                        [1.17664741e-05, 2.46864965e-05, 2.08299178e-05, 1.22880899e-05],
                        [4.67536705e-06, 2.03115410e-05, 4.72470469e-05, 3.43682935e-05]],
                       [[2.02106045e-05, 3.18806142e-05, 2.48947723e-05, 3.75958169e-05],
                        [1.08018585e-05, 1.11858466e-05, 1.13350142e-05, 1.15020435e-05],
                        [1.04905936e-05, 1.58540142e-05, 1.66376388e-05, 1.75261671e-05],
                        [2.28220968e-05, 1.36371342e-05, 2.03097329e-05,
                         4.08194213e-05]]]]) * units('s^-1')
    assert_array_almost_equal(totdef.data, truth, 12)


def test_frontogenesis_4d(data_4d):
    """Test frontogenesis on a 4D (time, pressure, y, x) grid."""
    theta = potential_temperature(data_4d.pressure, data_4d.temperature)
    frnt = frontogenesis(theta, data_4d.u, data_4d.v).transpose(
        'time1',
        'pressure',
        'latitude',
        'longitude'
    )

    truth = np.array([[[[4.03241166e-10, -1.66671969e-11, -2.21695296e-10, -3.92052431e-10],
                        [-5.25244095e-10, -1.12160993e-11, -4.89245961e-11, 3.12029903e-10],
                        [6.64291257e-10, 3.35294072e-10, 4.40696926e-11, -3.80990599e-10],
                        [-3.81175558e-10, 2.01421545e-10, 1.69276538e-09, -1.46727967e-09]],
                       [[-1.14725815e-10, 1.00977715e-10, -1.36697064e-10, 3.42060878e-10],
                        [-6.07354303e-11, 1.47758507e-11, -4.61570931e-11, 1.07716080e-10],
                        [4.52780481e-11, 6.99776255e-11, 3.54918971e-10, -7.12011926e-11],
                        [5.82577150e-10, 1.41596752e-10, 8.60051223e-10, 1.39190722e-09]],
                       [[-2.76524144e-13, 3.17817935e-10, 5.61139182e-10, 1.41251234e-10],
                        [9.13538909e-11, 1.07222839e-10, 5.84889489e-11, -7.04354673e-12],
                        [-7.63814267e-11, -2.61136261e-11, -9.38996785e-12, 2.25155943e-10],
                        [-8.07125189e-11, -4.09501260e-11, -1.68556325e-10, -1.68395224e-10]]],
                      [[[1.70912241e-10, -3.59604596e-11, -5.63440060e-11, -3.72141552e-10],
                        [-1.88604573e-10, -2.84591125e-11, 8.04708643e-12, 3.35465874e-10],
                        [4.05009495e-10, 4.99273109e-11, 2.70840073e-10, 3.53292290e-10],
                        [7.61811501e-10, -1.15049239e-10, 1.39114133e-09, -2.13934119e-11]],
                       [[-7.11061577e-11, -5.56233487e-11, 1.38759396e-10, 2.10158880e-10],
                        [-9.85704771e-11, 2.43793585e-11, 2.41161028e-11, 8.41366288e-11],
                        [2.07079174e-11, 9.67316909e-11, 3.79484230e-11, 1.00231778e-10],
                        [9.09016673e-11, 4.70716770e-12, 7.27411299e-10, -3.68904210e-10]],
                       [[1.48452574e-11, 1.64659568e-12, 7.71858317e-10, 1.74891129e-10],
                        [7.51825243e-11, 6.34791773e-11, 6.26549997e-11, -2.97116232e-11],
                        [-9.19046148e-11, 3.17048878e-11, -6.59923945e-11, 2.25154449e-10],
                        [-3.68975988e-12, -1.20891474e-10, -3.53749041e-11, 2.42234202e-10]]],
                      [[[-1.34106978e-10, 1.19278109e-10, -1.70196541e-10, 2.48281391e-11],
                        [-4.99795205e-11, -2.30130765e-11, 4.96545465e-11, 3.90460132e-11],
                        [-6.23025651e-12, -2.90005871e-11, 5.57986734e-11, 3.82595360e-10],
                        [1.33830354e-09, -8.27063507e-11, -2.04614424e-10, 4.66009647e-10]],
                       [[1.13855928e-10, -3.71418369e-10, 2.37111014e-10, 1.60355663e-10],
                        [-3.01604394e-10, 3.21033959e-11, 9.52301632e-11, 4.26592524e-11],
                        [6.25482337e-12, 7.81804086e-11, 2.58199246e-11, 1.74886075e-10],
                        [4.73684042e-11, 1.42713420e-11, 2.25862198e-10, 3.35966198e-11]],
                       [[3.63828967e-11, 1.41447035e-10, 9.83470917e-11, 1.37432553e-10],
                        [-7.52505235e-11, 7.47348135e-12, 8.59892617e-11, 1.09800029e-10],
                        [-7.58453531e-11, -4.69966422e-12, 1.14342322e-11, 1.81473021e-10],
                        [-2.97566390e-11, 9.55288188e-11, 1.90872070e-12,
                         5.32192321e-10]]]]) * units('K/m/s')
    assert_array_almost_equal(frnt.data, truth, 13)


def test_geostrophic_wind_4d(data_4d):
    """Test geostrophic_wind on a 4D (time, pressure, y, x) grid."""
    u_g, v_g = geostrophic_wind(data_4d.height)
    u_g_truth = np.array([[[[4.40486857, 12.51692362, 20.63729052, 3.17769103],
                            [14.10194385, 17.12263527, 22.04954906, 28.25627455],
                            [24.44520744, 22.83658981, 31.70185785, 41.43475568],
                            [35.55079058, 29.81196157, 50.61168553, 41.3453152]],
                           [[7.35973026, 11.15080483, 15.35393153, 8.90224492],
                            [8.36112125, 12.51333666, 13.38382965, 14.31962023],
                            [10.36996866, 13.03590323, 16.55132073, 20.5818555],
                            [13.5135907, 12.61987724, 25.47981975, 27.81300618]],
                           [[5.7532349, 8.87025457, 12.11513303, 6.95699048],
                            [5.63036393, 9.22723096, 9.46050119, 9.63463697],
                            [5.15111753, 8.92136337, 10.13229436, 10.02026917],
                            [4.27093407, 7.87208545, 14.52880097, 7.84194092]]],
                          [[[2.5637431, 12.12175173, 18.88903199, 9.31429705],
                            [11.13363928, 16.0692665, 22.88529458, 23.2247996],
                            [21.17380737, 18.19154369, 27.45449837, 37.89231093],
                            [32.89749798, 18.27860794, 32.68137607, 53.46238172]],
                           [[5.88868723, 10.23886179, 13.99207128, 7.62863391],
                            [7.72562524, 12.48283965, 13.87130359, 12.97472345],
                            [9.38948632, 12.47561185, 15.29521563, 18.71570682],
                            [10.86569541, 9.9484405, 18.45258492, 24.92010765]],
                           [[5.37666204, 9.31750379, 9.01145336, 3.68871571],
                            [5.42142755, 8.93123996, 9.3456061, 9.00788096],
                            [4.94868897, 8.34298027, 9.29367749, 11.09021722],
                            [3.89473037, 7.52596886, 8.80903478, 9.55782485]]],
                          [[[4.07701238, 9.91100559, 14.63521328, 11.44931302],
                            [9.21849096, 15.3989699, 20.84826449, 20.35213024],
                            [17.27879494, 16.28474382, 23.2252306, 32.43391015],
                            [28.63615274, 12.02290076, 21.31740598, 48.11881923]],
                           [[4.67797945, 7.67496476, 7.67070623, 7.43540912],
                            [6.36765831, 10.59388475, 12.09551703, 11.52096191],
                            [7.77187799, 11.17427747, 14.91109777, 16.17178096],
                            [8.86174464, 9.13936139, 15.93606235, 21.47254981]],
                           [[4.06859791, 6.49637561, 4.98326026, 5.11096512],
                            [4.19923606, 6.75503407, 8.50298015, 8.50994027],
                            [3.85339598, 6.92959314, 9.8142002, 10.51547453],
                            [2.97279588, 7.0103826, 8.65854182, 10.96893324]]]]) * units('m/s')
    v_g_truth = np.array([[[[-2.34958057e+01, -1.94104519e+01, -7.44959497e+00,
                             1.23868322e+01],
                            [-2.05867367e+01, -1.59688225e+01, -7.24619436e+00,
                             5.58114910e+00],
                            [-2.13003979e+01, -1.50644426e+01, -1.26465809e+00,
                             2.00990219e+01],
                            [-2.83335381e+01, -1.22608318e+01, 2.75571752e+00,
                             1.67161713e+01]],
                           [[-2.12135105e+01, -1.57486000e+01, -7.18331385e+00,
                             4.48243952e+00],
                            [-1.85706921e+01, -1.38995152e+01, -7.25590754e+00,
                             1.36025941e+00],
                            [-1.48431730e+01, -1.30190716e+01, -6.20916080e+00,
                             5.58656025e+00],
                            [-1.64091930e+01, -1.07454290e+01, -3.26166773e+00,
                             6.04215336e+00]],
                           [[-1.84210243e+01, -1.51837034e+01, -8.32569885e+00,
                             2.15305471e+00],
                            [-1.60743446e+01, -1.37354202e+01, -8.54446602e+00,
                             -5.01543939e-01],
                            [-1.26119165e+01, -1.31178055e+01, -8.13879681e+00,
                             2.32514095e+00],
                            [-1.08224831e+01, -1.12312374e+01, -8.07368088e+00,
                             -1.34987926e+00]]],
                          [[[-2.47784901e+01, -2.06641865e+01, -7.55605650e+00,
                             1.45456514e+01],
                            [-2.05139866e+01, -1.66804104e+01, -6.96553278e+00,
                             8.63076687e+00],
                            [-2.04345818e+01, -1.41986904e+01, -3.59461641e+00,
                             1.13773890e+01],
                            [-3.07159233e+01, -1.35134182e+01, 3.63993049e+00,
                             2.07441883e+01]],
                           [[-2.20703144e+01, -1.61019173e+01, -6.81787109e+00,
                             5.78179121e+00],
                            [-1.89408665e+01, -1.40810776e+01, -7.12525749e+00,
                             1.92659533e+00],
                            [-1.49793730e+01, -1.27466383e+01, -6.57639217e+00,
                             3.53139591e+00],
                            [-1.57215986e+01, -1.10794334e+01, -3.83887053e+00,
                             6.00018406e+00]],
                           [[-1.89922485e+01, -1.49378052e+01, -8.35085773e+00,
                             7.68607914e-01],
                            [-1.58400993e+01, -1.38690310e+01, -9.15049839e+00,
                             -1.68443954e+00],
                            [-1.34329786e+01, -1.28181543e+01, -8.34892273e+00,
                             -2.52818279e-02],
                            [-1.10563183e+01, -1.17126417e+01, -7.79271078e+00,
                             7.03427792e-01]]],
                          [[[-2.87962914e+01, -2.08093758e+01, -7.41080666e+00,
                             1.13992844e+01],
                            [-2.51369133e+01, -1.76726551e+01, -6.50083351e+00,
                             8.37874126e+00],
                            [-2.16215363e+01, -1.44126577e+01, -4.67937374e+00,
                             7.57850361e+00],
                            [-3.09025593e+01, -1.47021618e+01, 2.18099499e+00,
                             1.97469769e+01]],
                           [[-2.14603043e+01, -1.55501490e+01, -7.21480083e+00,
                             3.54577303e+00],
                            [-1.86117344e+01, -1.43230457e+01, -7.12040138e+00,
                             2.99635530e+00],
                            [-1.53198442e+01, -1.24255934e+01, -6.73208141e+00,
                             1.76072347e+00],
                            [-1.53703299e+01, -1.06545277e+01, -4.50935888e+00,
                             3.06527138e+00]],
                           [[-1.62525253e+01, -1.41536722e+01, -9.22987461e+00,
                             -1.48113370e+00],
                            [-1.41632900e+01, -1.34236937e+01, -9.18536949e+00,
                             -1.44826770e+00],
                            [-1.30243769e+01, -1.18180895e+01, -8.29443932e+00,
                             -2.45343924e+00],
                            [-1.09246559e+01, -1.03824110e+01, -7.37222433e+00,
                             -1.89407897e+00]]]]) * units('m/s')
    assert_array_almost_equal(u_g.data, u_g_truth, 4)
    assert_array_almost_equal(v_g.data, v_g_truth, 4)


def test_inertial_advective_wind_4d(data_4d):
    """Test inertial_advective_wind on a 4D (time, pressure, y, x) grid."""
    u_g, v_g = geostrophic_wind(data_4d.height)
    u_i, v_i = inertial_advective_wind(u_g, v_g, u_g, v_g)
    u_i_truth = np.array([[[[-4.76966186, -6.39706038, -7.24003746, -11.13794333],
                            [-1.89586566, -4.35883424, -6.85805714, -9.4212875],
                            [2.31372726, -6.96059926, -14.11458588, -20.68380008],
                            [-0.92883306, -13.81354883, -17.96354053, -23.79779997]],
                           [[-2.62082254, -3.50555985, -3.63842481, -4.20932821],
                            [-3.38566969, -2.58907076, -2.67708014, -3.36021786],
                            [-0.56713627, -2.3416945, -4.39000187, -6.69093397],
                            [1.70678751, -3.60860629, -5.96627063, -7.52914321]],
                           [[-1.61492016, -2.31780373, -2.40235407, -2.60787441],
                            [-2.19903344, -1.48707548, -1.58037953, -2.25343451],
                            [-1.11096954, -1.25163409, -2.02857574, -3.32734329],
                            [-0.26020197, -1.62905796, -1.75707467, -1.2223621]]],
                          [[[-6.72701434, -6.76960203, -7.94802076, -12.50171137],
                            [-2.22284799, -5.07983672, -7.76025363, -11.23189296],
                            [2.67509705, -4.83471753, -9.58547825, -12.94725576],
                            [8.58545145, -7.72587914, -12.41979585, -10.25605548]],
                           [[-3.19317899, -3.55857747, -3.56352137, -4.31615186],
                            [-3.70727146, -2.8684896, -2.7782166, -3.33031965],
                            [-1.17242459, -2.18140469, -3.58528354, -5.27404394],
                            [1.42344232, -2.45475499, -4.65221513, -6.1169067]],
                           [[-3.23907889, -1.91350728, -1.17379843, -1.09402307],
                            [-2.0340837, -1.38963467, -1.40556307, -1.93552382],
                            [-1.31936373, -1.1627646, -1.73546489, -2.82082041],
                            [-0.96507328, -0.94398947, -1.53168307, -2.57261637]]],
                          [[[-5.13667819, -5.35808776, -5.96105057, -8.09779516],
                            [-5.27868329, -6.04992134, -7.09615152, -9.11538451],
                            [0.32367483, -4.40754181, -7.26937211, -8.89052436],
                            [11.86601164, -3.52532263, -8.2149503, -3.91397366]],
                           [[-2.95853902, -1.94361543, -1.79128105, -2.22848035],
                            [-2.98114417, -2.49536376, -2.66131831, -3.4095258],
                            [-1.43210061, -2.24010995, -3.02803196, -3.96476269],
                            [0.38124008, -2.11580893, -3.41706461, -4.07935491]],
                           [[-1.85523484, -0.74020207, -0.62945585, -1.19060464],
                            [-0.90996905, -1.11068858, -1.44720476, -1.96113271],
                            [-0.97632032, -1.23447402, -1.48613628, -1.80024482],
                            [-1.30046767, -0.98449831, -1.25199805,
                             -1.96583328]]]]) * units('m/s')
    v_i_truth = np.array([[[[1.03212922e+01, 5.87785876e+00, -3.24290351e+00, -1.88453875e+01],
                            [9.87498125e+00, 5.33624247e+00, 4.80855268e+00, 3.62780511e-02],
                            [6.37519841e+00, 6.45883096e+00, 8.14332496e+00, 4.38659798e+00],
                            [-1.31389541e+00, 1.00955857e+01, 4.19848197e+00,
                             -1.97713955e+01]],
                           [[1.10365470e+00, 2.30316727e+00, -1.82344497e+00, -3.54754121e+00],
                            [2.43595083e+00, 1.35702893e+00, 4.91118248e-01, -1.03105842e-02],
                            [2.33831643e+00, 1.03116363e+00, 3.27903073e+00, 4.52178657e-01],
                            [2.90828402e-01, 1.43477414e+00, 6.69517000e+00, -4.27716340e+00]],
                           [[4.77177073e-01, 1.14435024e+00, -1.82680726e+00, -1.95986760e+00],
                            [5.18719070e-01, 4.51688547e-01, -3.28412094e-01, 6.84697225e-02],
                            [2.50141134e-01, 1.41518671e-01, 1.08838497e+00, -9.61933095e-02],
                            [-3.39178295e-01, 2.45727962e-01, 2.41825249e+00,
                             -2.84771923e+00]]],
                          [[[9.01360331e+00, 6.74640647e+00, 5.47040255e-01, -1.25154925e+01],
                            [9.56977790e+00, 4.57707018e+00, 3.34473925e+00, -7.13502610e+00],
                            [5.46464641e+00, 2.13949666e+00, 7.51823914e+00, 2.43533142e+00],
                            [-5.48839487e+00, -6.52611598e-01, 1.34292069e+01,
                             1.61544754e+01]],
                           [[2.49507477e+00, 3.34927241e+00, -7.11661027e-01, -3.42627695e+00],
                            [2.69966530e+00, 1.64559616e+00, 2.90248174e-01, -1.12696139e+00],
                            [1.83330337e+00, 1.69378198e-01, 1.87762364e+00, 7.55276554e-01],
                            [-4.89132896e-01, -1.06737759e+00, 4.20052028e+00,
                             1.54873202e+00]],
                           [[1.05176368e+00, 2.35279690e-01, -4.37230320e-01, -9.41455734e-01],
                            [5.26256702e-01, 1.32552797e-01, 6.61475967e-02, 1.17988702e-01],
                            [9.40681182e-02, 3.45287932e-02, 2.13397644e-01, 6.10768896e-01],
                            [-2.44304796e-01, -6.00961285e-02, -3.78761065e-02,
                             2.27978276e-01]]],
                          [[[5.18728227e+00, 8.23825046e+00, 2.86046723e+00, -5.59088886e+00],
                            [8.85355614e+00, 4.70956220e+00, 2.51349179e+00, -5.64414232e+00],
                            [7.54622775e+00, 7.98092891e-02, 4.70152506e+00, 3.47162602e+00],
                            [-1.92789744e+00, -5.92225638e+00, 1.00594741e+01,
                             2.62864566e+01]],
                           [[2.20468164e+00, 3.00812312e+00, 1.59439971e+00, -6.42312367e-01],
                            [2.15609133e+00, 1.86103734e+00, 1.28243894e+00, -1.03944156e+00],
                            [1.50383253e+00, 5.72866867e-01, 1.51969207e+00, -3.94601885e-01],
                            [2.57841077e-02, -8.98532915e-01, 2.48926548e+00, 1.81145651e+00]],
                           [[6.98587642e-01, 2.55740716e-01, 1.74401316e+00, 3.79592864e-01],
                            [2.39095399e-01, 4.87795233e-01, 1.16885491e+00, -7.66586054e-03],
                            [-6.48645993e-02, 5.81727905e-01, 4.66123480e-01, 3.71778788e-02],
                            [-2.11967488e-01, 5.16025460e-01, -4.15578572e-01,
                             6.96366806e-01]]]]) * units('m/s')
    assert_array_almost_equal(u_i.data, u_i_truth, 4)
    assert_array_almost_equal(v_i.data, v_i_truth, 4)


def test_q_vector_4d(data_4d):
    """Test q_vector on a 4D (time, pressure, y, x) grid."""
    u_g, v_g = geostrophic_wind(data_4d.height)
    q1, q2 = q_vector(u_g, v_g, data_4d.temperature, data_4d.pressure)
    q1_truth = np.array([[[[-1.11399407e-12, 2.50794237e-13, 3.16093168e-12, 2.32093331e-12],
                           [5.65649869e-13, 1.45620779e-13, 1.71343752e-12, 4.35800011e-12],
                           [-4.96503931e-13, 2.62116549e-12, 3.96540726e-12, 4.08996874e-12],
                           [-1.31411324e-12, 8.91776830e-12, 9.28518964e-12, -2.68490726e-12]],
                          [[7.62925715e-13, 1.16785318e-12, -2.01309755e-13, 1.26529742e-12],
                           [3.47435346e-13, 6.73455725e-13, 6.90294419e-13, 8.12267467e-13],
                           [3.45704077e-14, 3.82817753e-13, 1.54656386e-12, 3.07185369e-12],
                           [6.11434780e-13, 6.23632300e-13, 1.40617773e-12, 5.34947219e-12]],
                          [[3.06414278e-13, 6.53804262e-13, 1.75404505e-12, 5.51976164e-13],
                           [3.28229719e-13, 2.75782033e-13, 4.00407507e-13, 7.84750100e-13],
                           [1.32588098e-13, 2.51525423e-13, 5.49106514e-13, 1.78892467e-12],
                           [7.88840796e-14, 1.60673966e-13, 1.19208617e-12, 2.05418653e-12]]],
                         [[[-3.34132897e-13, 1.53374763e-12, 4.49316053e-12, 1.64643286e-12],
                           [1.07061926e-13, 1.48351071e-13, 2.40731954e-12, 5.09815184e-12],
                           [-6.72234608e-13, 1.09871184e-12, 1.78399997e-12, 2.83734147e-12],
                           [-2.04431842e-12, 6.47809851e-12, 4.82039700e-12, -2.09744034e-12]],
                          [[5.49758402e-13, 1.84138510e-13, 6.32622851e-13, 2.64607266e-12],
                           [6.72993111e-13, 6.48589900e-13, 8.38201872e-13, 1.02446030e-12],
                           [6.71507328e-14, 4.68905430e-13, 1.21606351e-12, 2.11104302e-12],
                           [3.15734101e-13, 1.95983121e-13, 1.20260143e-12, 4.19732652e-12]],
                          [[6.33871103e-13, 5.90709910e-13, 1.21586844e-12, 1.06166654e-12],
                           [3.43322382e-13, 2.76046202e-13, 4.90662239e-13, 5.62988991e-13],
                           [2.09877678e-13, 2.37809232e-13, 3.65590502e-13, 8.07598362e-13],
                           [2.35522003e-14, 2.21315437e-13, 5.14061506e-13, 1.17222164e-12]]],
                         [[[8.22178446e-13, 2.09477989e-12, 5.54298564e-12, 2.21511518e-12],
                           [1.54450727e-12, 5.45033765e-13, 2.37288896e-12, 3.30727156e-12],
                           [2.94161134e-13, 3.03848451e-13, 1.47235183e-13, 2.95945450e-12],
                           [-6.20823394e-12, 9.77981323e-13, -2.06881609e-12,
                            -3.58251099e-12]],
                          [[7.61196804e-13, -6.76343613e-13, 9.48323229e-13, 6.33365711e-13],
                           [1.14599786e-12, 6.99199729e-13, 7.41681860e-13, 9.28590425e-13],
                           [1.03166054e-13, 6.13187200e-13, 8.39627802e-13, 1.48207669e-12],
                           [3.22870872e-13, -2.18606955e-13, 9.98812765e-13, 1.91778451e-12]],
                          [[5.55381844e-13, 2.79538040e-13, 3.30236669e-13, -4.91571259e-14],
                           [2.50227841e-13, 2.70855069e-13, 4.03362348e-13, 5.22065702e-13],
                           [3.37119836e-13, 3.17667714e-13, 2.25387106e-13, 6.46265259e-13],
                           [2.05548507e-13, 3.55426850e-13, -1.74728156e-14,
                            5.04028133e-13]]]]) * units('m^2 kg^-1 s^-1')
    q2_truth = np.array([[[[3.34318820e-12, -1.32561232e-13, 1.01510711e-12, 6.03331800e-12],
                           [2.51737448e-13, -1.71044158e-13, -8.25290924e-13, 1.68843717e-13],
                           [-3.50533924e-12, -1.68864979e-12, 7.74026063e-13, 1.53811977e-12],
                           [-1.75456351e-12, -3.86555813e-12, -1.89153040e-12,
                            -5.15241976e-12]],
                          [[-2.09823428e-13, -6.26774796e-13, -1.40145242e-13, 1.09119884e-12],
                           [-2.58383579e-13, -2.67580088e-13, -6.44081099e-14, 5.90687237e-13],
                           [-2.73791441e-13, -2.28454021e-13, -4.76780567e-13,
                            -8.48612071e-13],
                           [1.21028867e-12, -5.10570435e-13, 6.32933788e-14, 2.44873356e-12]],
                          [[-6.72615385e-14, -3.57492837e-13, -4.18039453e-14, 3.81431652e-13],
                           [-3.56221252e-13, -1.23227388e-13, -3.21550682e-14,
                            -4.69364017e-14],
                           [-2.82392676e-13, -1.20969658e-13, 1.13815813e-13, -6.93334063e-14],
                           [5.19714375e-14, -4.61213207e-13, 5.33263529e-13, 1.28188808e-12]]],
                         [[[1.72090175e-12, -1.35788214e-12, 1.48184196e-13, 3.22826220e-12],
                           [-2.13531998e-13, -1.17527920e-13, -6.94495723e-13, 1.76851853e-12],
                           [-2.67906083e-12, -3.78055500e-13, -9.90177606e-13, 2.87945080e-12],
                           [1.48308333e-12, 2.15094364e-13, -4.84492586e-12, 2.77186124e-12]],
                          [[-3.09679995e-13, -2.52107306e-13, 4.57596343e-14, 2.03410764e-12],
                           [-3.95694028e-13, -3.00120864e-13, 1.05194228e-14, 1.06118650e-12],
                           [-2.46886775e-13, -2.43771482e-13, -3.81434550e-13,
                            -1.70381858e-13],
                           [8.12779739e-13, -1.38604573e-13, -8.06018823e-13,
                            -7.80874251e-13]],
                          [[-2.19867650e-13, -1.53226258e-13, 4.07751663e-13, 1.52430853e-12],
                           [-2.56504387e-13, -1.21082762e-13, 6.29695620e-15, 3.49769107e-13],
                           [-2.44113387e-13, -1.21986494e-13, -9.12718030e-14,
                            -1.60274595e-13],
                           [-2.47713487e-13, -1.77307889e-13, -1.13295694e-13,
                            -6.07631206e-13]]],
                         [[[-6.49198232e-13, -1.97145105e-12, -5.54605298e-13, 1.94723486e-12],
                           [-2.00875018e-12, -3.72744112e-13, -4.59665809e-13, 1.12359119e-13],
                           [-3.83508199e-12, 1.18439125e-13, -4.24891463e-13, -5.88482481e-13],
                           [-1.84276119e-12, 1.55112390e-12, -7.38034738e-13, 1.03676199e-13]],
                          [[-4.58813210e-13, -1.88617051e-13, 2.58369931e-13, 8.15071067e-13],
                           [-6.09657914e-13, -3.51811097e-13, 2.39365587e-13, 5.80541301e-13],
                           [-1.69115858e-13, -3.49864908e-13, -2.26620147e-13, 7.79560474e-13],
                           [2.23317058e-13, 1.20352864e-13, -1.01565643e-12, -2.16675768e-13]],
                          [[-1.68140233e-13, -5.07963999e-14, 2.77741196e-13, 8.37842279e-13],
                           [-1.39578146e-13, -1.36744814e-13, 3.12352497e-14, 4.55339789e-13],
                           [-1.06614836e-13, -2.19878930e-13, -8.37992151e-14, 1.87868902e-13],
                           [-2.27057581e-13, -2.74474045e-13, -1.10759455e-13,
                            -3.90242255e-13]]]]) * units('m^2 kg^-1 s^-1')
    assert_array_almost_equal(q1.data, q1_truth, 15)
    assert_array_almost_equal(q2.data, q2_truth, 15)


@pytest.mark.parametrize('geog_data', ('+proj=lcc lat_1=25', '+proj=latlon', '+proj=stere'),
                         indirect=True)
def test_geospatial_laplacian_geographic(geog_data):
    """Test geospatial_laplacian across projections."""
    crs, lons, lats, _, arr, mx, my, dx, dy = geog_data
    laplac = geospatial_laplacian(arr, longitude=lons, latitude=lats, crs=crs)

    # Calculate the true fields using known map-correct approach
    u = mx * first_derivative(arr, delta=dx, axis=1)
    v = my * first_derivative(arr, delta=dy, axis=0)

    truth = (mx * first_derivative(u, delta=dx, axis=1)
             + my * first_derivative(v, delta=dy, axis=0)
             - (u * mx / my) * first_derivative(my, delta=dx, axis=1)
             - (v * my / mx) * first_derivative(mx, delta=dy, axis=0))

    assert_array_almost_equal(laplac, truth)
