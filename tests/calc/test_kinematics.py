# Copyright (c) 2008,2015,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `kinematics` module."""

import numpy as np
import pytest
import xarray as xr

from metpy.calc import (absolute_vorticity, advection, ageostrophic_wind, divergence,
                        frontogenesis, geostrophic_wind, inertial_advective_wind,
                        lat_lon_grid_deltas, montgomery_streamfunction, potential_temperature,
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
    truth = np.array([[2.004485646e-5, 2.971929112e-5, 9.534206801e-5],
                      [2.486578545e-5, 8.110461115e-6, 4.029608124e-6],
                      [-4.494362040e-5, -3.923563259e-5, -6.976113764e-5]]) / units.sec
    truth = xr.DataArray(truth, coords=basic_dataset.coords)
    assert_array_almost_equal(d, truth, 4)


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


def test_shearing_deformation_asym():
    """Test shearing deformation calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    sh = shearing_deformation(u, v, 1 * units.meters, 2 * units.meters)
    true_sh = np.array([[-7.5, -1.5, 1.], [9.5, -0.5, -11.], [1.5, 5.5, 12.]]) / units.sec
    assert_array_equal(sh, true_sh)


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


def test_vorticity_4d(data_4d):
    """Test vorticity on a 4D (time, pressure, y, x) grid."""
    vort = vorticity(data_4d.u, data_4d.v)
    truth = np.array([[[[-5.84515167e-05, 3.17729585e-05, 4.58261458e-05, 2.01292844e-05],
                        [2.13352387e-05, 1.96004423e-05, 4.16486823e-05, 6.90445310e-05],
                        [6.19222139e-05, 6.93601354e-05, 8.36426564e-05, 8.27371956e-05],
                        [-4.43297956e-05, 1.56381845e-04, 1.42510285e-04, -6.03146997e-05]],
                       [[-8.68712685e-07, 4.24890902e-05, -1.01652245e-05, 1.40567079e-05],
                        [1.16955768e-05, 1.06772721e-05, 2.85536762e-05, 6.24020996e-05],
                        [1.83821861e-05, 7.39881835e-06, 3.76797354e-05, 4.67243318e-05],
                        [3.19572986e-05, 2.81106117e-05, 3.73957581e-05, 5.39673545e-05]],
                       [[-2.85395302e-05, 3.84435833e-06, 4.04684576e-06, 5.14302137e-05],
                        [-2.19027713e-05, 5.30196965e-06, 2.00825604e-05, 3.39049359e-05],
                        [-1.97989026e-05, -2.49143814e-06, 2.87381603e-05, 3.02743602e-05],
                        [-2.34066064e-05, -1.82455025e-05, 2.95918539e-05, 3.42233420e-05]]],
                      [[[-3.66558283e-05, 2.45520394e-05, 8.31163955e-05, 2.44003406e-05],
                        [3.30393774e-05, 2.30884339e-05, 4.67122293e-05, 7.61002310e-05],
                        [7.66048678e-05, 4.98785325e-05, 6.62382288e-05, 9.91998869e-05],
                        [-3.81435328e-06, 8.81953022e-05, 1.11601055e-04, -4.42546076e-05]],
                       [[-2.51779702e-06, 3.95852220e-05, 3.77219249e-05, 3.79808820e-05],
                        [-4.63231122e-06, 2.05995207e-05, 2.89780728e-05, 3.01707041e-05],
                        [1.54931892e-05, 8.07405448e-06, 2.44489167e-05, 7.02383317e-05],
                        [2.07687143e-05, 2.17030773e-05, 3.38485776e-05, 9.11533757e-05]],
                       [[-7.17079917e-06, -2.80615398e-06, 1.94218575e-05, 6.22111037e-05],
                        [-1.81300845e-05, 7.12699895e-06, 1.41049190e-05, 1.80915929e-05],
                        [-2.99278303e-06, 2.57606747e-06, 2.25304657e-05, 3.70860448e-05],
                        [-1.48782578e-05, -6.27503290e-06, 3.66662188e-05, 1.14265141e-05]]],
                      [[[-2.14099419e-05, 3.11255562e-05, 9.43465637e-05, 6.21369629e-05],
                        [2.83576779e-05, 2.71698609e-05, 4.45208755e-05, 7.95114352e-05],
                        [7.29890624e-05, 3.07600211e-05, 3.71063624e-05, 7.76394608e-05],
                        [6.87038548e-05, 2.25751156e-05, 7.15889230e-05, 6.91611419e-05]],
                       [[-5.87372735e-07, 5.63433054e-05, 2.92457668e-05, 3.63765486e-05],
                        [-6.20728063e-06, 2.63907904e-05, 3.27952358e-05, 2.87436180e-05],
                        [8.81938816e-06, 6.49991429e-06, 2.03676451e-05, 4.99729397e-05],
                        [1.54604020e-05, 1.42654642e-06, -6.85425498e-06, 6.03129247e-05]],
                       [[2.98152979e-06, 2.95129295e-05, 3.30665600e-05, 4.51175504e-05],
                        [-3.45169373e-06, 7.99863229e-06, 1.31503178e-05, 1.82983904e-05],
                        [-5.96640905e-06, -6.58389207e-07, 5.92924140e-06, 1.82929244e-05],
                        [-2.94758645e-06, 3.86196289e-06, 5.27851664e-06,
                         2.73814460e-05]]]]) * units('s^-1')
    assert_array_almost_equal(vort.data, truth, 12)


def test_divergence_4d(data_4d):
    """Test divergence on a 4D (time, pressure, y, x) grid."""
    div = divergence(data_4d.u, data_4d.v)
    truth = np.array([[[[-8.37608702e-06, -5.39336397e-06, 1.42603335e-05, 2.81027853e-05],
                        [2.95370985e-05, -8.88467654e-06, 1.18747200e-05, -6.20723902e-06],
                        [-4.64906358e-05, -2.10620465e-05, 1.33445184e-05, 4.77361610e-05],
                        [-3.35504768e-05, 1.49909469e-05, -4.22891356e-05, 3.92159433e-05]],
                       [[1.69393752e-05, 1.56905500e-06, 4.11012845e-05, -5.08416562e-05],
                        [2.27792270e-05, -3.97105867e-06, 5.31878772e-06, 1.75998805e-05],
                        [-6.17322438e-06, -1.74803086e-06, -1.46149939e-05, 2.66339797e-06],
                        [-3.07336112e-05, -8.20592276e-06, -2.56044746e-05, -2.78477951e-05]],
                       [[-1.46864164e-05, -2.25693290e-05, -2.10250150e-05, -3.88307255e-05],
                        [2.90827814e-06, -9.96431223e-06, -1.02191776e-06, 4.55537027e-06],
                        [1.48872763e-05, 3.66494030e-06, 5.06711902e-06, 5.82930016e-06],
                        [1.12436632e-05, 1.61499510e-05, 1.13636149e-05, 3.85414061e-06]]],
                      [[[-3.88892057e-06, -8.14255274e-06, 6.65050488e-06, 4.82691326e-05],
                        [2.29332150e-05, -6.95362743e-07, 4.90498323e-06, -3.27034929e-05],
                        [-3.28375966e-05, -9.30193297e-06, 1.89455614e-06, 2.55048066e-05],
                        [-7.03452094e-05, 1.59780333e-05, -5.02908248e-05, 3.01991417e-05]],
                       [[1.67133745e-05, 1.24417427e-05, -2.22432790e-05, -1.89957283e-05],
                        [1.76022853e-05, -1.76730982e-06, 3.99751017e-06, -5.57126626e-06],
                        [4.04526025e-06, -5.27586187e-06, 3.61323452e-06, 4.18352106e-06],
                        [-3.61713767e-06, 4.02190371e-06, -2.16650827e-05, 2.94848150e-05]],
                       [[-6.47746172e-06, -2.98975077e-07, -4.67842312e-05, 6.49628794e-06],
                        [-9.24534558e-07, -6.79275746e-06, 1.57736990e-06, 2.15325190e-06],
                        [5.79213453e-06, -2.31793357e-06, 1.63764017e-05, 1.30886984e-05],
                        [-3.81688990e-06, 1.81508167e-05, -1.74511070e-08, 1.38462853e-05]]],
                      [[[-3.04155688e-06, -1.57905256e-05, 1.11057616e-05, 1.02005872e-05],
                        [2.22682490e-05, -2.13398663e-06, 2.98041246e-06, -1.27487570e-05],
                        [-2.03870559e-06, -1.44407167e-05, -1.54275894e-06, 1.52970546e-05],
                        [-7.46492084e-05, -2.17403382e-05, -6.50948158e-06, -9.39708598e-06]],
                       [[-2.87798173e-06, 1.83342079e-05, -1.59741420e-05, 5.93365940e-06],
                        [2.21780902e-05, 2.10396122e-06, -8.61898987e-06, -4.81985056e-06],
                        [6.86376938e-06, -1.71394859e-06, 4.97940406e-06, -8.12845764e-06],
                        [-5.91535752e-06, -1.25897243e-06, 1.33746621e-05, 2.89926447e-05]],
                       [[3.56301627e-06, -3.60303780e-06, -2.14805401e-05, 9.05879471e-06],
                        [1.15765605e-05, -2.20007656e-07, -1.00689171e-05, -7.85316340e-06],
                        [1.76295477e-06, -4.68035973e-07, 6.34634343e-06, -9.26903305e-06],
                        [9.56906212e-07, -2.83017535e-06, 1.68342294e-05,
                         -5.69798533e-06]]]]) * units('s^-1')
    assert_array_almost_equal(div.data, truth, 12)


def test_shearing_deformation_4d(data_4d):
    """Test shearing_deformation on a 4D (time, pressure, y, x) grid."""
    shdef = shearing_deformation(data_4d.u, data_4d.v)
    truth = np.array([[[[-2.33792381e-05, 3.44534094e-06, 2.69410760e-05, 1.06867281e-05],
                        [-6.40972431e-05, 1.01579031e-05, 1.73678734e-05, -2.40319045e-05],
                        [7.70545354e-07, -1.87702202e-05, -1.39302341e-05, 3.73230852e-05],
                        [6.35849225e-05, -1.08009221e-04, -9.62510298e-05, 7.32297192e-05]],
                       [[-2.42502310e-05, -1.01193319e-05, 5.54828905e-05, -3.31928326e-07],
                        [-2.69305297e-06, 9.32833730e-06, 2.04600718e-05, 3.36248400e-05],
                        [-7.24755760e-06, 1.72909996e-05, -5.48615182e-06, -1.30784063e-05],
                        [-2.51475614e-05, 9.22553765e-06, -2.17297542e-06, -5.34977173e-05]],
                       [[-2.58416628e-05, 1.01393773e-05, 4.54141476e-05, 6.20366322e-07],
                        [-1.56077459e-05, 6.20125807e-06, 2.36797141e-05, 2.53616873e-05],
                        [-2.71240538e-06, 1.14475474e-05, 8.05450723e-06, 3.07240065e-05],
                        [1.16656764e-05, 2.71686080e-05, -1.88326452e-06, 1.03921795e-05]]],
                      [[[5.29600994e-06, 1.04331961e-05, -1.72892524e-05, 3.67655639e-05],
                        [-3.67904320e-05, 8.07030650e-06, 3.05173020e-06, -2.40356283e-05],
                        [3.03845109e-08, -2.56843275e-07, 1.17465234e-06, 3.08089412e-05],
                        [1.79034632e-05, -3.12752861e-05, -5.30138255e-05, 6.33453564e-05]],
                       [[-2.54496668e-05, -1.88685727e-05, 7.59573914e-06, 7.85469836e-06],
                        [-1.58734272e-05, 8.90875832e-06, 1.95355336e-05, 6.33953947e-06],
                        [2.90313838e-06, 1.03222777e-05, 1.50063775e-05, 1.13348820e-05],
                        [-6.20995986e-06, 5.06623932e-06, 3.72239179e-06, -4.41896630e-05]],
                       [[-1.97608457e-05, 7.98531569e-06, 1.94218554e-05, 1.18509048e-05],
                        [-1.81300845e-05, 7.12699895e-06, 1.59034980e-05, -7.08850441e-06],
                        [1.04965562e-05, 3.47535804e-06, 7.24254745e-06, 4.15824912e-05],
                        [1.29997134e-05, 7.21430847e-06, -1.45932750e-05, 5.00959463e-05]]],
                      [[[-8.37024044e-06, 2.79795154e-06, -2.39099649e-05, 1.76221280e-05],
                        [-1.88550094e-05, 3.33869412e-06, 1.34953970e-05, 1.25143854e-05],
                        [5.96277806e-07, 1.86196124e-05, 1.68723536e-05, 9.74312685e-06],
                        [6.20326426e-06, 2.93197852e-05, -1.42931965e-05, 2.19484546e-05]],
                       [[-1.00299098e-05, -4.57260229e-05, 8.56211376e-06, 3.45779631e-05],
                        [-1.65491061e-05, -4.63468810e-06, 6.71584791e-06, 1.76493950e-06],
                        [-4.22030685e-06, 1.50431608e-05, 1.81194219e-05, 5.45811766e-06],
                        [-2.07574370e-06, 1.80633930e-05, 4.39555860e-05, 5.90590854e-06]],
                       [[-2.08496392e-05, -3.02898043e-05, -3.80429538e-06, 2.71317584e-05],
                        [-4.80062637e-06, 1.25396267e-06, 6.85529455e-06, 5.70834171e-06],
                        [5.72435226e-06, 1.05827268e-05, 1.53717763e-05, 1.55950591e-05],
                        [1.23403264e-05, -1.98341401e-06, 1.56203357e-05,
                         3.90722041e-05]]]]) * units('s^-1')
    assert_array_almost_equal(shdef.data, truth, 12)


def test_stretching_deformation_4d(data_4d):
    """Test stretching_deformation on a 4D (time, pressure, y, x) grid."""
    stdef = stretching_deformation(data_4d.u, data_4d.v)
    truth = np.array([[[[3.47898088e-05, 2.24845986e-05, -5.97367530e-06, -2.81027927e-05],
                        [-1.00316265e-05, 2.43890252e-05, 5.13005043e-06, 3.02139765e-05],
                        [-5.95303373e-05, 4.11805509e-06, 3.94239079e-05, 5.53801191e-05],
                        [8.92024896e-05, 1.85881092e-05, 3.59490328e-05, -1.03321407e-04]],
                       [[3.00039817e-06, 1.37094723e-05, -4.34319088e-05, 1.79539749e-05],
                        [1.87324184e-05, 5.47148050e-06, -9.06983993e-06, 8.15734277e-06],
                        [1.21798873e-07, 1.26405968e-05, 2.72019585e-05, -3.63162743e-06],
                        [-1.36470926e-05, 1.87727600e-05, 5.84790724e-05, 5.03903728e-05]],
                       [[2.89291086e-05, 3.31866090e-05, 1.58458533e-05, 5.68409251e-06],
                        [1.68472637e-05, 1.52157851e-05, 5.27310978e-06, 1.21993291e-05],
                        [8.59225306e-06, 7.71174035e-06, -4.82506223e-06, -1.57536424e-05],
                        [-5.84283826e-06, 8.50599727e-06, -3.27143224e-07, -3.93117456e-05]]],
                      [[[3.69837694e-05, 1.86562509e-05, -2.79203000e-06, -3.51399535e-05],
                        [-6.42858314e-06, 2.70027422e-05, 6.97334875e-06, 5.92098244e-06],
                        [-4.01668004e-05, 5.04173347e-06, 4.75334876e-05, 6.25555261e-05],
                        [3.66252634e-05, 2.71352154e-06, 7.09783382e-05, -5.79312118e-05]],
                       [[-5.31921974e-06, -1.04758793e-06, 2.58686924e-05, 7.08365906e-06],
                        [1.26562011e-05, 1.35206063e-05, -2.74715944e-06, 4.32091552e-06],
                        [8.54170666e-06, 1.49581427e-05, 6.31110194e-06, 9.12961275e-06],
                        [2.67785986e-06, 5.37083849e-06, 5.47744998e-05, 4.07259321e-05]],
                       [[1.73537008e-05, 5.99605247e-06, 4.13461116e-05, -2.90256397e-05],
                        [4.24395934e-07, 1.02937398e-05, 5.17452359e-06, 7.09934306e-06],
                        [5.34248818e-06, 6.67495925e-06, -7.90440717e-06, 1.03908310e-05],
                        [1.46185421e-05, 1.65031056e-07, 4.47900388e-06, -4.46075180e-05]]],
                      [[[3.02321534e-05, 2.69257238e-05, -4.63180943e-06, 3.00627122e-06],
                        [-2.01256850e-06, 2.88914919e-05, 1.15236589e-05, -3.75586415e-06],
                        [-1.41791143e-05, 1.61351154e-05, 3.08316570e-05, 5.12686237e-05],
                        [-4.95427192e-06, 1.96269721e-05, 4.92464559e-05, 6.43446270e-05]],
                       [[-1.86155399e-05, -1.13423401e-05, 2.94399620e-05, -8.00532458e-06],
                        [1.63327091e-05, 8.39898448e-06, 7.11857042e-06, 7.32055442e-06],
                        [9.11199258e-06, 1.67214834e-05, 5.42904828e-06, 1.03069722e-05],
                        [2.62789752e-06, 5.48570575e-06, 1.29250179e-05, 3.39387353e-05]],
                       [[-4.08093319e-06, 1.03359478e-05, 2.30342884e-05, -2.51141968e-05],
                        [8.42904887e-06, 9.22253152e-06, 6.56793595e-06, -9.65174212e-06],
                        [6.70904325e-06, 9.42414527e-06, -1.74726096e-06, 4.66995059e-06],
                        [1.75937571e-05, 1.24577364e-05, -1.28423144e-05,
                         7.34171029e-06]]]]) * units('s^-1')
    assert_array_almost_equal(stdef.data, truth, 10)


def test_total_deformation_4d(data_4d):
    """Test total_deformation on a 4D (time, pressure, y, x) grid."""
    totdef = total_deformation(data_4d.u, data_4d.v)
    truth = np.array([[[[4.19156244e-05, 2.27470339e-05, 2.75954049e-05, 3.00661456e-05],
                        [6.48775008e-05, 2.64198324e-05, 1.81096782e-05, 3.86059168e-05],
                        [5.95353239e-05, 1.92166476e-05, 4.18126289e-05, 6.67830089e-05],
                        [1.09545089e-04, 1.09597033e-04, 1.02745286e-04, 1.26640850e-04]],
                       [[2.44351405e-05, 1.70396746e-05, 7.04604984e-05, 1.79570429e-05],
                        [1.89250108e-05, 1.08145724e-05, 2.23802711e-05, 3.46001750e-05],
                        [7.24858097e-06, 2.14187617e-05, 2.77496740e-05, 1.35732616e-05],
                        [2.86119377e-05, 2.09171476e-05, 5.85194303e-05, 7.34928257e-05]],
                       [[3.87902676e-05, 3.47009797e-05, 4.80992294e-05, 5.71784592e-06],
                        [2.29658884e-05, 1.64309378e-05, 2.42597310e-05, 2.81431841e-05],
                        [9.01021396e-06, 1.38027998e-05, 9.38915929e-06, 3.45274069e-05],
                        [1.30470980e-05, 2.84690226e-05, 1.91146748e-06, 4.06621536e-05]]],
                      [[[3.73610348e-05, 2.13753895e-05, 1.75132430e-05, 5.08578708e-05],
                        [3.73478589e-05, 2.81829369e-05, 7.61187559e-06, 2.47541807e-05],
                        [4.01668119e-05, 5.04827148e-06, 4.75479995e-05, 6.97308017e-05],
                        [4.07669464e-05, 3.13927813e-05, 8.85911406e-05, 8.58408963e-05]],
                       [[2.59996085e-05, 1.88976315e-05, 2.69607956e-05, 1.05770748e-05],
                        [2.03013575e-05, 1.61917500e-05, 1.97277459e-05, 7.67203178e-06],
                        [9.02158329e-06, 1.81740323e-05, 1.62794771e-05, 1.45543595e-05],
                        [6.76273132e-06, 7.38327075e-06, 5.49008382e-05, 6.00943247e-05]],
                       [[2.62990866e-05, 9.98588563e-06, 4.56805146e-05, 3.13517416e-05],
                        [1.81350510e-05, 1.25201914e-05, 1.67241425e-05, 1.00323261e-05],
                        [1.17779401e-05, 7.52550294e-06, 1.07207344e-05, 4.28610889e-05],
                        [1.95625745e-05, 7.21619581e-06, 1.52651613e-05, 6.70778242e-05]]],
                      [[[3.13694760e-05, 2.70707062e-05, 2.43544673e-05, 1.78767184e-05],
                        [1.89621152e-05, 2.90837615e-05, 1.77459983e-05, 1.30658470e-05],
                        [1.41916465e-05, 2.46380177e-05, 3.51463709e-05, 5.21862079e-05],
                        [7.93884738e-06, 3.52826847e-05, 5.12787372e-05, 6.79850401e-05]],
                       [[2.11456240e-05, 4.71117592e-05, 3.06597644e-05, 3.54925451e-05],
                        [2.32514580e-05, 9.59287621e-06, 9.78655496e-06, 7.53030733e-06],
                        [1.00418822e-05, 2.24923252e-05, 1.89152853e-05, 1.16629638e-05],
                        [3.34881431e-06, 1.88780066e-05, 4.58164777e-05, 3.44487664e-05]],
                       [[2.12452693e-05, 3.20047506e-05, 2.33463296e-05, 3.69710047e-05],
                        [9.70025146e-06, 9.30739007e-06, 9.49383200e-06, 1.12134424e-05],
                        [8.81926698e-06, 1.41706959e-05, 1.54707604e-05, 1.62792600e-05],
                        [2.14900894e-05, 1.26146394e-05, 2.02217686e-05,
                         3.97559787e-05]]]]) * units('s^-1')
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

    truth = np.array([[[[4.23682388e-10, -6.60428594e-12, -2.16700227e-10, -3.80960666e-10],
                        [-5.28427593e-10, -7.11496293e-12, -4.77951513e-11, 2.94985981e-10],
                        [7.86953679e-10, 3.54196972e-10, 2.07842740e-11, -5.25487973e-10],
                        [-3.52111258e-10, 2.06421077e-10, 1.67986422e-09, -1.45950592e-09]],
                       [[-7.31728965e-11, 1.06892315e-10, -1.33453527e-10, 3.42647921e-10],
                        [-5.05805666e-11, 2.12238918e-11, -4.71306612e-11, 9.62250022e-11],
                        [4.76933273e-11, 6.94586917e-11, 3.53139630e-10, -7.14834221e-11],
                        [6.14587969e-10, 1.41091788e-10, 8.42714362e-10, 1.36031856e-09]],
                       [[2.05113794e-11, 3.21339794e-10, 5.56947831e-10, 1.43142115e-10],
                        [9.85782985e-11, 1.06721561e-10, 5.73106405e-11, -5.03368922e-12],
                        [-6.43122987e-11, -2.12772736e-11, -1.17352480e-11, 2.13297934e-10],
                        [-6.97155996e-11, -4.10739462e-11, -1.75156002e-10, -1.76167917e-10]]],
                      [[[1.74719456e-10, -1.35620544e-11, -5.23975776e-11, -3.77740716e-10],
                        [-1.89498320e-10, -2.40570704e-11, 1.09765802e-11, 3.26582884e-10],
                        [5.05760395e-10, 5.96930313e-11, 2.51806496e-10, 2.62326483e-10],
                        [8.55597272e-10, -1.03839677e-10, 1.36437001e-09, -2.55279252e-11]],
                       [[-4.68143046e-11, -4.29566800e-11, 1.37326379e-10, 2.00212822e-10],
                        [-7.60292021e-11, 3.13481943e-11, 2.02636812e-11, 7.07310188e-11],
                        [2.07073318e-11, 9.74536122e-11, 3.64495220e-11, 9.11599007e-11],
                        [1.07707226e-10, 4.27961436e-12, 7.17400120e-10, -4.07742791e-10]],
                       [[3.51033086e-11, 6.86914537e-12, 7.68630167e-10, 1.73824937e-10],
                        [8.63644951e-11, 6.43950959e-11, 6.01335884e-11, -3.49684748e-11],
                        [-8.06772168e-11, 3.34221310e-11, -6.70871076e-11, 2.13933933e-10],
                        [2.77857293e-12, -1.19419804e-10, -3.88340891e-11, 2.35051688e-10]]],
                      [[[-1.06920260e-10, 1.42163009e-10, -1.67670634e-10, 7.77738130e-12],
                        [-2.14431980e-11, -1.40383248e-11, 5.12326588e-11, 4.47136472e-11],
                        [9.29690678e-11, -1.91237280e-11, 5.11911088e-11, 3.57423744e-10],
                        [1.48172065e-09, -6.47936247e-11, -2.02021163e-10, 3.76309534e-10]],
                       [[1.40697485e-10, -3.68197137e-10, 2.35522920e-10, 1.53804948e-10],
                        [-2.61409796e-10, 3.88149869e-11, 9.17155132e-11, 3.56335985e-11],
                        [6.05095218e-12, 8.10937994e-11, 2.38586262e-11, 1.57114763e-10],
                        [5.98536934e-11, 1.42709122e-11, 2.20296991e-10, 6.13222348e-12]],
                       [[5.77582222e-11, 1.50846336e-10, 9.79419525e-11, 1.38512768e-10],
                        [-5.73091526e-11, 1.59416672e-11, 8.32303219e-11, 1.08035832e-10],
                        [-5.84859130e-11, 7.43545248e-13, 9.37957614e-12, 1.74102020e-10],
                        [-2.38469755e-11, 1.01414977e-10, 4.18826651e-12,
                         5.18914848e-10]]]]) * units('K/m/s')
    assert_array_almost_equal(frnt.data, truth, 13)


def test_geostrophic_wind_4d(data_4d):
    """Test geostrophic_wind on a 4D (time, pressure, y, x) grid."""
    u_g, v_g = geostrophic_wind(data_4d.height)
    u_g_truth = np.array([[[[4.4048682, 12.51692258, 20.6372888, 3.17769076],
                            [14.10194272, 17.12263389, 22.04954728, 28.25627227],
                            [24.44520364, 22.83658626, 31.70185292, 41.43474924],
                            [35.55078527, 29.81195711, 50.61167797, 41.34530902]],
                           [[7.35972965, 11.1508039, 15.35393025, 8.90224418],
                            [8.36112058, 12.51333565, 13.38382857, 14.31961908],
                            [10.36996705, 13.0359012, 16.55131816, 20.5818523, ],
                            [13.51358869, 12.61987535, 25.47981594, 27.81300202]],
                           [[5.75323442, 8.87025383, 12.11513202, 6.9569899],
                            [5.63036347, 9.22723021, 9.46050042, 9.6346362],
                            [5.15111673, 8.92136198, 10.13229278, 10.02026762],
                            [4.27093343, 7.87208428, 14.5287988, 7.84193975]]],
                          [[[2.56374289, 12.12175071, 18.88903041, 9.31429628],
                            [11.13363838, 16.0692652, 22.88529273, 23.22479772],
                            [21.17380408, 18.19154086, 27.4544941, 37.89230504],
                            [32.89749307, 18.27860521, 32.68137119, 53.46237373]],
                           [[5.88868673, 10.23886093, 13.99207011, 7.62863328],
                            [7.72562462, 12.48283865, 13.87130247, 12.9747224],
                            [9.38948486, 12.47560991, 15.29521325, 18.71570391],
                            [10.86569379, 9.94843902, 18.45258217, 24.92010393]],
                           [[5.37666159, 9.31750301, 9.01145261, 3.6887154],
                            [5.42142711, 8.93123924, 9.34560535, 9.00788023],
                            [4.9486882, 8.34297898, 9.29367604, 11.09021549],
                            [3.89472979, 7.52596773, 8.80903347, 9.55782342]]],
                          [[[4.07701203, 9.91100477, 14.63521206, 11.44931207],
                            [9.21849021, 15.39896866, 20.84826281, 20.3521286],
                            [17.27879226, 16.28474129, 23.22522698, 32.4339051],
                            [28.63614846, 12.02289896, 21.31740279, 48.11881204]],
                           [[4.67797906, 7.67496412, 7.67070558, 7.4354085],
                            [6.3676578, 10.5938839, 12.09551605, 11.52096098],
                            [7.77187678, 11.17427574, 14.91109545, 16.17177845],
                            [8.86174332, 9.13936002, 15.93605997, 21.47254661]],
                           [[4.06859757, 6.49637507, 4.98325985, 5.1109647],
                            [4.19923572, 6.75503352, 8.50297947, 8.50993959],
                            [3.85339539, 6.92959206, 9.81419868, 10.5154729],
                            [2.97279544, 7.01038155, 8.65854052, 10.9689316]]]]) * units('m/s')
    v_g_truth = np.array([[[[-2.34997753e+01, -1.94136235e+01, -7.45077637e+00,
                             1.23887662e+01],
                            [-2.05898579e+01, -1.59712848e+01, -7.24733971e+00,
                             5.58197747e+00],
                            [-2.13032949e+01, -1.50665793e+01, -1.26486198e+00,
                             2.01018571e+01],
                            [-2.83372497e+01, -1.22624731e+01, 2.75609237e+00,
                             1.67184466e+01]],
                           [[-2.12169685e+01, -1.57511747e+01, -7.18451047e+00,
                             4.48302414e+00],
                            [-1.85734872e+01, -1.39016674e+01, -7.25703167e+00,
                             1.36042011e+00],
                            [-1.48452478e+01, -1.30209105e+01, -6.21005126e+00,
                             5.58732988e+00],
                            [-1.64113345e+01, -1.07468232e+01, -3.26209862e+00,
                             6.04283912e+00]],
                           [[-1.84240576e+01, -1.51861981e+01, -8.32705150e+00,
                             2.15338222e+00],
                            [-1.60768326e+01, -1.37375247e+01, -8.54578152e+00,
                             -5.01603207e-01],
                            [-1.26137008e+01, -1.31196694e+01, -8.13994713e+00,
                             2.32546588e+00],
                            [-1.08239460e+01, -1.12327091e+01, -8.07473534e+00,
                             -1.35002468e+00]]],
                          [[[-2.47825558e+01, -2.06675642e+01, -7.55733001e+00,
                             1.45481469e+01],
                            [-2.05171683e+01, -1.66829347e+01, -6.96656838e+00,
                             8.63193062e+00],
                            [-2.04375067e+01, -1.42006723e+01, -3.59516781e+00,
                             1.13790069e+01],
                            [-3.07199620e+01, -1.35152096e+01, 3.64042638e+00,
                             2.07469460e+01]],
                           [[-2.20738890e+01, -1.61045805e+01, -6.81898954e+00,
                             5.78288395e+00],
                            [-1.89437910e+01, -1.40832144e+01, -7.12633797e+00,
                             1.92683830e+00],
                            [-1.49814792e+01, -1.27484476e+01, -6.57732385e+00,
                             3.53189205e+00],
                            [-1.57235558e+01, -1.10808922e+01, -3.83938054e+00,
                             6.00097928e+00]],
                           [[-1.89953281e+01, -1.49402619e+01, -8.35222723e+00,
                             7.68775922e-01],
                            [-1.58424970e+01, -1.38711585e+01, -9.15189832e+00,
                             -1.68471661e+00],
                            [-1.34349198e+01, -1.28199780e+01, -8.35009927e+00,
                             -2.52835808e-02],
                            [-1.10578184e+01, -1.17141722e+01, -7.79372570e+00,
                             7.03521108e-01]]],
                          [[[-2.88009221e+01, -2.08127679e+01, -7.41206720e+00,
                             1.14011801e+01],
                            [-2.51405873e+01, -1.76754149e+01, -6.50182713e+00,
                             8.38017608e+00],
                            [-2.16245136e+01, -1.44146994e+01, -4.68003089e+00,
                             7.57949195e+00],
                            [-3.09065921e+01, -1.47040769e+01, 2.18126927e+00,
                             1.97494465e+01]],
                           [[-2.14639093e+01, -1.55526942e+01, -7.21598014e+00,
                             3.54623269e+00],
                            [-1.86145303e+01, -1.43252474e+01, -7.12149199e+00,
                             2.99673603e+00],
                            [-1.53220281e+01, -1.24273773e+01, -6.73303389e+00,
                             1.76100214e+00],
                            [-1.53722451e+01, -1.06559370e+01, -4.50997751e+00,
                             3.06563326e+00]],
                           [[-1.62551769e+01, -1.41559875e+01, -9.23139816e+00,
                             -1.48140877e+00],
                            [-1.41654778e+01, -1.34257568e+01, -9.18676573e+00,
                             -1.44850466e+00],
                            [-1.30262107e+01, -1.18197548e+01, -8.29562748e+00,
                             -2.45382867e+00],
                            [-1.09261218e+01, -1.03837731e+01, -7.37319328e+00,
                             -1.89438246e+00]]]]) * units('m/s')
    assert_array_almost_equal(u_g.data, u_g_truth, 4)
    assert_array_almost_equal(v_g.data, v_g_truth, 4)


def test_inertial_advective_wind_4d(data_4d):
    """Test inertial_advective_wind on a 4D (time, pressure, y, x) grid."""
    u_g, v_g = geostrophic_wind(data_4d.height)
    u_i, v_i = inertial_advective_wind(u_g, v_g, u_g, v_g)
    u_i_truth = np.array([[[[-4.77165787, -6.39928757, -7.24239774, -11.14139847],
                            [-1.8967587, -4.36028755, -6.86016435, -9.424228],
                            [2.31421679, -6.96263439, -14.11859275, -20.68976199],
                            [-0.92900951, -13.81722973, -17.96832023, -23.80435234]],
                           [[-2.62194257, -3.50676725, -3.63961746, -4.21059159],
                            [-3.38684408, -2.58995365, -2.67792148, -3.36122749],
                            [-0.56740802, -2.34244481, -4.39126012, -6.69284736],
                            [1.70715454, -3.60961021, -5.96780511, -7.53107716]],
                           [[-1.61558735, -2.31867093, -2.40316115, -2.60870259],
                            [-2.19984407, -1.48762908, -1.58089856, -2.2541336],
                            [-1.11136338, -1.25207315, -2.02918744, -3.32828099],
                            [-0.26028196, -1.62956357, -1.75756959, -1.22270124]]],
                          [[[-6.72938857, -6.77202159, -7.95073037, -12.50625533],
                            [-2.22377841, -5.0815521, -7.76259189, -11.23523285],
                            [2.67551814, -4.83617581, -9.58820051, -12.95106032],
                            [8.58739912, -7.72793742, -12.42304341, -10.25891257]],
                           [[-3.19431927, -3.55990592, -3.56474965, -4.31772693],
                            [-3.70858471, -2.86947801, -2.77907873, -3.331319],
                            [-1.17292465, -2.182095, -3.58631575, -5.27553824],
                            [1.4236791, -2.45544962, -4.65344893, -6.11853894]],
                           [[-3.24030343, -1.91423726, -1.1742268, -1.09439772],
                            [-2.03479751, -1.39015234, -1.40603089, -1.93610702],
                            [-1.31981448, -1.16318518, -1.73599486, -2.82161648],
                            [-0.96540565, -0.94432034, -1.53211138, -2.57328907]]],
                          [[[-5.13892702, -5.35990209, -5.96305829, -8.10039371],
                            [-5.28049715, -6.05189422, -7.09840362, -9.11834812],
                            [0.32358269, -4.40891596, -7.27141143, -8.89305721],
                            [11.86892255, -3.52631413, -8.21707342, -3.9149252]],
                           [[-2.95997348, -1.94436814, -1.79187921, -2.22918106],
                            [-2.98223302, -2.49621136, -2.66214712, -3.41052605],
                            [-1.43265094, -2.2408268, -3.02891598, -3.9658998],
                            [0.38112998, -2.11641585, -3.417963, -4.08044633]],
                           [[-1.85590971, -0.74052267, -0.62971895, -1.19099569],
                            [-0.91035149, -1.11111857, -1.44768616, -1.96172425],
                            [-0.97667565, -1.23489465, -1.48658447, -1.80074616],
                            [-1.30083552, -0.98479841, -1.25235639,
                             -1.96633294]]]]) * units('m/s')
    v_i_truth = np.array([[[[1.03230312e+01, 5.87882109e+00, -3.24343027e+00, -1.88483470e+01],
                            [9.87647721e+00, 5.33706213e+00, 4.80929670e+00, 3.63063183e-02],
                            [6.37603821e+00, 6.45974507e+00, 8.14449487e+00, 4.38722620e+00],
                            [-1.31406689e+00, 1.00969188e+01, 4.19901525e+00,
                             -1.97739544e+01]],
                           [[1.10383561e+00, 2.30354462e+00, -1.82374723e+00, -3.54809094e+00],
                            [2.43631993e+00, 1.35723724e+00, 4.91193534e-01, -1.02997771e-02],
                            [2.33864366e+00, 1.03130947e+00, 3.27949769e+00, 4.52250225e-01],
                            [2.90865168e-01, 1.43496262e+00, 6.69604741e+00, -4.27768358e+00]],
                           [[4.77255548e-01, 1.14453826e+00, -1.82710412e+00, -1.96018490e+00],
                            [5.18797941e-01, 4.51757453e-01, -3.28462782e-01, 6.84789970e-02],
                            [2.50176678e-01, 1.41538500e-01, 1.08853845e+00, -9.62071225e-02],
                            [-3.39224824e-01, 2.45760327e-01, 2.41856776e+00,
                             -2.84808630e+00]]],
                          [[[9.01508187e+00, 6.74751069e+00, 5.47135566e-01, -1.25176087e+01],
                            [9.57125782e+00, 4.57776586e+00, 3.34524473e+00, -7.13601695e+00],
                            [5.46543202e+00, 2.13979774e+00, 7.51931363e+00, 2.43567533e+00],
                            [-5.48910344e+00, -6.52697336e-01, 1.34309575e+01,
                             1.61565561e+01]],
                           [[2.49548039e+00, 3.34982501e+00, -7.11777553e-01, -3.42687086e+00],
                            [2.70007988e+00, 1.64584666e+00, 2.90292095e-01, -1.12712093e+00],
                            [1.83356146e+00, 1.69401994e-01, 1.87788933e+00, 7.55385123e-01],
                            [-4.89203395e-01, -1.06751808e+00, 4.20107093e+00,
                             1.54893157e+00]],
                           [[1.05193589e+00, 2.35318468e-01, -4.37301952e-01, -9.41622628e-01],
                            [5.26337352e-01, 1.32572812e-01, 6.61575719e-02, 1.18009862e-01],
                            [9.40801497e-02, 3.45333939e-02, 2.13427873e-01, 6.10855423e-01],
                            [-2.44339907e-01, -6.01035575e-02, -3.78806842e-02,
                             2.28008249e-01]]],
                          [[[5.18811867e+00, 8.23959428e+00, 2.86095202e+00, -5.59181418e+00],
                            [8.85485851e+00, 4.71028978e+00, 2.51387570e+00, -5.64507599e+00],
                            [7.54725519e+00, 7.98206363e-02, 4.70219106e+00, 3.47217441e+00],
                            [-1.92815930e+00, -5.92302637e+00, 1.00607869e+01,
                             2.62899914e+01]],
                           [[2.20504999e+00, 3.00861548e+00, 1.59466025e+00, -6.42397860e-01],
                            [2.15641722e+00, 1.86132244e+00, 1.28263500e+00, -1.03958535e+00],
                            [1.50404596e+00, 5.72947187e-01, 1.51990698e+00, -3.94664336e-01],
                            [2.57832794e-02, -8.98652226e-01, 2.48959124e+00, 1.81170400e+00]],
                           [[6.98702092e-01, 2.55782733e-01, 1.74430100e+00, 3.79660759e-01],
                            [2.39131800e-01, 4.87869781e-01, 1.16903247e+00, -7.66523806e-03],
                            [-6.48734332e-02, 5.81810137e-01, 4.66189458e-01, 3.71854388e-02],
                            [-2.11996986e-01, 5.16093087e-01, -4.15633085e-01,
                             6.96457035e-01]]]]) * units('m/s')
    assert_array_almost_equal(u_i.data, u_i_truth, 4)
    assert_array_almost_equal(v_i.data, v_i_truth, 4)


def test_q_vector_4d(data_4d):
    """Test q_vector on a 4D (time, pressure, y, x) grid."""
    u_g, v_g = geostrophic_wind(data_4d.height)
    q1, q2 = q_vector(u_g, v_g, data_4d.temperature, data_4d.pressure)
    q1_truth = np.array([[[[-9.02684270e-13, 2.04906965e-13, 2.90366741e-12, 2.19304520e-12],
                           [4.39259469e-13, 1.21664810e-13, 1.52570637e-12, 3.84499568e-12],
                           [-1.20961682e-12, 2.28334568e-12, 3.48876764e-12, 3.04353683e-12],
                           [-1.52298016e-12, 8.05872598e-12, 7.74115167e-12, -2.23036948e-12]],
                          [[5.76052684e-13, 1.04797925e-12, -1.76250215e-13, 1.21374024e-12],
                           [2.96159390e-13, 5.82994320e-13, 6.26425486e-13, 7.35027599e-13],
                           [-4.05458639e-14, 3.26100111e-13, 1.40096964e-12, 2.83322883e-12],
                           [3.28501677e-13, 5.63278420e-13, 1.13853072e-12, 4.65264045e-12]],
                          [[2.25120252e-13, 5.80121595e-13, 1.62940948e-12, 5.46851323e-13],
                           [2.66540809e-13, 2.42144848e-13, 3.74380714e-13, 7.39064640e-13],
                           [6.59374356e-14, 2.00559760e-13, 5.22478916e-13, 1.70369853e-12],
                           [4.17124258e-14, 1.20339974e-13, 1.04017356e-12, 2.00285625e-12]]],
                         [[[-2.70235442e-13, 1.36656387e-12, 4.19692633e-12, 1.51457512e-12],
                           [4.64050107e-14, 1.26573416e-13, 2.16942269e-12, 4.72745728e-12],
                           [-1.25821951e-12, 9.58231778e-13, 1.49027307e-12, 2.13360636e-12],
                           [-2.94458687e-12, 6.08808030e-12, 4.22668460e-12, -2.13178006e-12]],
                          [[4.25758843e-13, 1.40346565e-13, 5.92764154e-13, 2.56329392e-12],
                           [5.75744868e-13, 5.63479482e-13, 7.68528359e-13, 9.54169673e-13],
                           [2.17989465e-14, 3.98672857e-13, 1.09630992e-12, 1.91458466e-12],
                           [1.45472393e-13, 1.80092943e-13, 1.03379050e-12, 3.63517612e-12]],
                          [[5.35452418e-13, 5.16110645e-13, 1.16167143e-12, 1.05362388e-12],
                           [2.78553810e-13, 2.34739333e-13, 4.61792157e-13, 5.36758839e-13],
                           [1.47975123e-13, 1.96526691e-13, 3.47222939e-13, 7.50706946e-13],
                           [-2.14929942e-14, 1.75848584e-13, 4.87164306e-13, 1.11205431e-12]]],
                         [[[6.44882146e-13, 1.89203548e-12, 5.23849406e-12, 2.08882463e-12],
                           [1.31350246e-12, 4.79492775e-13, 2.16235780e-12, 3.08756991e-12],
                           [-2.30468810e-13, 2.50903749e-13, 8.88048363e-14, 2.47021777e-12],
                           [-7.28610348e-12, 8.50128944e-13, -2.07551923e-12,
                            -4.41869472e-12]],
                          [[6.06909454e-13, -6.74302573e-13, 9.15261604e-13, 5.85674558e-13],
                           [9.58472982e-13, 6.14413858e-13, 6.99238896e-13, 8.71127400e-13],
                           [7.89621985e-14, 5.29651074e-13, 7.48742263e-13, 1.34494118e-12],
                           [2.23181345e-13, -1.99880485e-13, 8.59667557e-13, 1.60364542e-12]],
                          [[4.49485378e-13, 2.19403888e-13, 3.32682107e-13, -4.06788877e-14],
                           [1.71334107e-13, 2.19234639e-13, 3.80362792e-13, 5.06206489e-13],
                           [2.60485983e-13, 2.66689509e-13, 2.05507268e-13, 6.05236747e-13],
                           [1.69953806e-13, 2.92743451e-13, -1.21101740e-14,
                            4.45255696e-13]]]]) * units('m^2 kg^-1 s^-1')
    q2_truth = np.array([[[[3.34398414e-12, -1.32578962e-13, 1.01530245e-12, 6.03460412e-12],
                           [2.51655551e-13, -1.71080181e-13, -8.25450865e-13, 1.68941987e-13],
                           [-3.50610571e-12, -1.68900418e-12, 7.74142051e-13, 1.53842636e-12],
                           [-1.75486540e-12, -3.86629371e-12, -1.89184780e-12,
                            -5.15338594e-12]],
                          [[-2.09878631e-13, -6.26922694e-13, -1.40170277e-13, 1.09139148e-12],
                           [-2.58443408e-13, -2.67657189e-13, -6.44319215e-14, 5.90804763e-13],
                           [-2.73875193e-13, -2.28517322e-13, -4.76883863e-13,
                            -8.48746443e-13],
                           [1.21044640e-12, -5.10676858e-13, 6.32812733e-14, 2.44933519e-12]],
                          [[-6.72809694e-14, -3.57593424e-13, -4.18326571e-14, 3.81509257e-13],
                           [-3.56312152e-13, -1.23269564e-13, -3.21698576e-14,
                            -4.69401174e-14],
                           [-2.82461704e-13, -1.21007762e-13, 1.13823760e-13, -6.93485412e-14],
                           [5.19806694e-14, -4.61314808e-13, 5.33326094e-13, 1.28209513e-12]]],
                         [[[1.72127539e-12, -1.35818611e-12, 1.48111017e-13, 3.22882115e-12],
                           [-2.13631818e-13, -1.17550571e-13, -6.94644658e-13, 1.76893456e-12],
                           [-2.67966931e-12, -3.78148042e-13, -9.90360068e-13, 2.87997878e-12],
                           [1.48322304e-12, 2.15101840e-13, -4.84581616e-12, 2.77231259e-12]],
                          [[-3.09742331e-13, -2.52155554e-13, 4.57591777e-14, 2.03457093e-12],
                           [-3.95777463e-13, -3.00202455e-13, 1.05082591e-14, 1.06140347e-12],
                           [-2.46969363e-13, -2.43836368e-13, -3.81515859e-13,
                            -1.70412444e-13],
                           [8.12844940e-13, -1.38633850e-13, -8.06173908e-13,
                            -7.80955396e-13]],
                          [[-2.19923258e-13, -1.53282385e-13, 4.07809333e-13, 1.52462097e-12],
                           [-2.56567476e-13, -1.21124223e-13, 6.28491470e-15, 3.49835200e-13],
                           [-2.44172367e-13, -1.22026447e-13, -9.12989545e-14,
                            -1.60305594e-13],
                           [-2.47776092e-13, -1.77361553e-13, -1.13326400e-13,
                            -6.07726254e-13]]],
                         [[[-6.49332840e-13, -1.97186697e-12, -5.54805109e-13, 1.94760968e-12],
                           [-2.00917113e-12, -3.72825112e-13, -4.59780632e-13, 1.12445112e-13],
                           [-3.83584827e-12, 1.18455212e-13, -4.24969207e-13, -5.88484873e-13],
                           [-1.84313287e-12, 1.55136757e-12, -7.38157445e-13, 1.03689734e-13]],
                          [[-4.58924792e-13, -1.88627007e-13, 2.58408535e-13, 8.15237426e-13],
                           [-6.09787117e-13, -3.51901418e-13, 2.39399294e-13, 5.80646992e-13],
                           [-1.69168847e-13, -3.49955041e-13, -2.26671298e-13, 7.79694360e-13],
                           [2.23293556e-13, 1.20382150e-13, -1.01583327e-12, -2.16626822e-13]],
                          [[-1.68178414e-13, -5.08196191e-14, 2.77786052e-13, 8.38012650e-13],
                           [-1.39619960e-13, -1.36786251e-13, 3.12305194e-14, 4.55426142e-13],
                           [-1.06649917e-13, -2.19937033e-13, -8.38223242e-14, 1.87904895e-13],
                           [-2.27100932e-13, -2.74536001e-13, -1.10779552e-13,
                            -3.90314768e-13]]]]) * units('m^2 kg^-1 s^-1')
    assert_array_almost_equal(q1.data, q1_truth, 15)
    assert_array_almost_equal(q2.data, q2_truth, 15)
