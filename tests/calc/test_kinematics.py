# Copyright (c) 2008,2015,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `kinematics` module."""

from collections import namedtuple

import numpy as np
import pytest
import xarray as xr

from metpy.calc import (absolute_vorticity, advection, ageostrophic_wind, coriolis_parameter,
                        divergence, frontogenesis, geostrophic_wind, inertial_advective_wind,
                        lat_lon_grid_deltas, montgomery_streamfunction,
                        potential_temperature, potential_vorticity_baroclinic,
                        potential_vorticity_barotropic, q_vector, shearing_deformation,
                        static_stability, storm_relative_helicity, stretching_deformation,
                        total_deformation, vorticity, wind_components)
from metpy.constants import g, omega, Re
from metpy.future import ageostrophic_wind as ageostrophic_wind_future
from metpy.testing import (assert_almost_equal, assert_array_almost_equal, assert_array_equal,
                           check_and_silence_warning, get_test_data)
from metpy.units import concatenate, units


def test_default_order():
    """Test using the default array ordering."""
    u = np.ones((3, 3)) * units('m/s')
    v = vorticity(u, u, 1 * units.meter, 1 * units.meter)
    true_v = np.zeros_like(u) / units.sec
    assert_array_equal(v, true_v)


def test_zero_vorticity():
    """Test vorticity calculation when zeros should be returned."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    v = vorticity(u, u.T, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_v = np.zeros_like(u) / units.sec
    assert_array_equal(v, true_v)


def test_vorticity():
    """Test vorticity for simple case."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    v = vorticity(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_v = np.ones_like(u) / units.sec
    assert_array_equal(v, true_v)


def test_vorticity_asym():
    """Test vorticity calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    vort = vorticity(u, v, 1 * units.meters, 2 * units.meters, dim_order='yx')
    true_vort = np.array([[-2.5, 3.5, 13.], [8.5, -1.5, -11.], [-5.5, -1.5, 0.]]) / units.sec
    assert_array_equal(vort, true_vort)

    # Now try for xy ordered
    vort = vorticity(u.T, v.T, 1 * units.meters, 2 * units.meters, dim_order='xy')
    assert_array_equal(vort, true_vort.T)


def test_zero_divergence():
    """Test divergence calculation when zeros should be returned."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    c = divergence(u, u.T, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_c = 2. * np.ones_like(u) / units.sec
    assert_array_equal(c, true_c)


def test_divergence():
    """Test divergence for simple case."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    c = divergence(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
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
    c = divergence(u, u, 1 * units.meter, 1 * units.meter)
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
    c = divergence(u, v, 1 * units.meters, 2 * units.meters, dim_order='yx')
    true_c = np.array([[-2, 5.5, -2.5], [2., 0.5, -1.5], [3., -1.5, 8.5]]) / units.sec
    assert_array_equal(c, true_c)

    # Now try for xy ordered
    c = divergence(u.T, v.T, 1 * units.meters, 2 * units.meters, dim_order='xy')
    assert_array_equal(c, true_c.T)


def test_shearing_deformation_asym():
    """Test shearing deformation calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    sh = shearing_deformation(u, v, 1 * units.meters, 2 * units.meters, dim_order='yx')
    true_sh = np.array([[-7.5, -1.5, 1.], [9.5, -0.5, -11.], [1.5, 5.5, 12.]]) / units.sec
    assert_array_equal(sh, true_sh)

    # Now try for yx ordered
    sh = shearing_deformation(u.T, v.T, 1 * units.meters, 2 * units.meters,
                              dim_order='xy')
    assert_array_equal(sh, true_sh.T)


def test_stretching_deformation_asym():
    """Test stretching deformation calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    st = stretching_deformation(u, v, 1 * units.meters, 2 * units.meters, dim_order='yx')
    true_st = np.array([[4., 0.5, 12.5], [4., 1.5, -0.5], [1., 5.5, -4.5]]) / units.sec
    assert_array_equal(st, true_st)

    # Now try for yx ordered
    st = stretching_deformation(u.T, v.T, 1 * units.meters, 2 * units.meters,
                                dim_order='xy')
    assert_array_equal(st, true_st.T)


def test_total_deformation_asym():
    """Test total deformation calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    tdef = total_deformation(u, v, 1 * units.meters, 2 * units.meters,
                             dim_order='yx')
    true_tdef = np.array([[8.5, 1.58113883, 12.5399362], [10.30776406, 1.58113883, 11.0113578],
                          [1.80277562, 7.7781746, 12.8160056]]) / units.sec
    assert_almost_equal(tdef, true_tdef)

    # Now try for xy ordered
    tdef = total_deformation(u.T, v.T, 1 * units.meters, 2 * units.meters,
                             dim_order='xy')
    assert_almost_equal(tdef, true_tdef.T)


def test_frontogenesis_asym():
    """Test frontogensis calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    theta = np.array([[303, 295, 305], [308, 310, 312], [299, 293, 289]]) * units('K')
    fronto = frontogenesis(theta, u, v, 1 * units.meters, 2 * units.meters,
                           dim_order='yx')
    true_fronto = np.array([[-52.4746386, -37.3658646, -50.3996939],
                            [3.5777088, -2.1221867, -16.9941166],
                            [-23.1417334, 26.0499143, -158.4839684]]
                           ) * units.K / units.meter / units.sec
    assert_almost_equal(fronto, true_fronto)

    # Now try for xy ordered
    fronto = frontogenesis(theta.T, u.T, v.T, 1 * units.meters, 2 * units.meters,
                           dim_order='xy')
    assert_almost_equal(fronto, true_fronto.T)


def test_advection_uniform():
    """Test advection calculation for a uniform 1D field."""
    u = np.ones((3,)) * units('m/s')
    s = np.ones_like(u) * units.kelvin
    a = advection(s, u, (1 * units.meter,), dim_order='xy')
    truth = np.zeros_like(u) * units('K/sec')
    assert_array_equal(a, truth)


def test_advection_1d_uniform_wind():
    """Test advection for simple 1D case with uniform wind."""
    u = np.ones((3,)) * units('m/s')
    s = np.array([1, 2, 3]) * units('kg')
    a = advection(s, u, (1 * units.meter,), dim_order='xy')
    truth = -np.ones_like(u) * units('kg/sec')
    assert_array_equal(a, truth)


def test_advection_1d():
    """Test advection calculation with varying wind and field."""
    u = np.array([1, 2, 3]) * units('m/s')
    s = np.array([1, 2, 3]) * units('Pa')
    a = advection(s, u, (1 * units.meter,), dim_order='xy')
    truth = np.array([-1, -2, -3]) * units('Pa/sec')
    assert_array_equal(a, truth)


def test_advection_2d_uniform():
    """Test advection for uniform 2D field."""
    u = np.ones((3, 3)) * units('m/s')
    s = np.ones_like(u) * units.kelvin
    a = advection(s, [u, u], (1 * units.meter, 1 * units.meter), dim_order='xy')
    truth = np.zeros_like(u) * units('K/sec')
    assert_array_equal(a, truth)


def test_advection_2d():
    """Test advection in varying 2D field."""
    u = np.ones((3, 3)) * units('m/s')
    v = 2 * np.ones((3, 3)) * units('m/s')
    s = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * units.kelvin
    a = advection(s, [u, v], (1 * units.meter, 1 * units.meter), dim_order='xy')
    truth = np.array([[-6, -4, 2], [-8, 0, 8], [-2, 4, 6]]) * units('K/sec')
    assert_array_equal(a, truth)


def test_advection_2d_asym():
    """Test advection in asymmetric varying 2D field."""
    u = np.arange(9).reshape(3, 3) * units('m/s')
    v = 2 * u
    s = np.array([[1, 2, 4], [4, 8, 4], [8, 6, 4]]) * units.kelvin
    a = advection(s, [u, v], (2 * units.meter, 1 * units.meter), dim_order='yx')
    truth = np.array([[0, -20.75, -2.5], [-33., -16., 20.], [-48, 91., 8]]) * units('K/sec')
    assert_array_equal(a, truth)

    # Now try xy ordered
    a = advection(s.T, [u.T, v.T], (2 * units.meter, 1 * units.meter), dim_order='xy')
    assert_array_equal(a, truth.T)


def test_geostrophic_wind():
    """Test geostrophic wind calculation with basic conditions."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units.meter
    # Using g as the value for f allows it to cancel out
    ug, vg = geostrophic_wind(z, g.magnitude / units.sec,
                              100. * units.meter, 100. * units.meter, dim_order='xy')
    true_u = np.array([[-2, 0, 2]] * 3) * units('m/s')
    true_v = -true_u.T
    assert_array_equal(ug, true_u)
    assert_array_equal(vg, true_v)


def test_geostrophic_wind_asym():
    """Test geostrophic wind calculation with a complicated field."""
    z = np.array([[1, 2, 4], [4, 8, 4], [8, 6, 4]]) * 200. * units.meter
    # Using g as the value for f allows it to cancel out
    ug, vg = geostrophic_wind(z, g.magnitude / units.sec,
                              200. * units.meter, 100. * units.meter, dim_order='yx')
    true_u = -np.array([[5, 20, 0], [7, 4, 0], [9, -12, 0]]) * units('m/s')
    true_v = np.array([[0.5, 1.5, 2.5], [8, 0, -8], [-2, -2, -2]]) * units('m/s')
    assert_array_equal(ug, true_u)
    assert_array_equal(vg, true_v)

    # Now try for xy ordered
    ug, vg = geostrophic_wind(z.T, g.magnitude / units.sec,
                              200. * units.meter, 100. * units.meter, dim_order='xy')
    assert_array_equal(ug, true_u.T)
    assert_array_equal(vg, true_v.T)


def test_geostrophic_geopotential():
    """Test geostrophic wind calculation with geopotential."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
    ug, vg = geostrophic_wind(z, 1 / units.sec, 100. * units.meter, 100. * units.meter,
                              dim_order='xy')
    true_u = np.array([[-2, 0, 2]] * 3) * units('m/s')
    true_v = -true_u.T
    assert_array_equal(ug, true_u)
    assert_array_equal(vg, true_v)


def test_geostrophic_3d():
    """Test geostrophic wind calculation with 3D array."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100.
    # Using g as the value for f allows it to cancel out
    z3d = np.dstack((z, z)) * units.meter
    ug, vg = geostrophic_wind(z3d, g.magnitude / units.sec,
                              100. * units.meter, 100. * units.meter, dim_order='xy')
    true_u = np.array([[-2, 0, 2]] * 3) * units('m/s')
    true_v = -true_u.T

    true_u = concatenate((true_u[..., None], true_u[..., None]), axis=2)
    true_v = concatenate((true_v[..., None], true_v[..., None]), axis=2)
    assert_array_equal(ug, true_u)
    assert_array_equal(vg, true_v)


def test_geostrophic_gempak():
    """Test of geostrophic wind calculation against gempak values."""
    z = np.array([[5586387.00, 5584467.50, 5583147.50],
                  [5594407.00, 5592487.50, 5591307.50],
                  [5604707.50, 5603247.50, 5602527.50]]).T \
        * (9.80616 * units('m/s^2')) * 1e-3
    dx = np.deg2rad(0.25) * Re * np.cos(np.deg2rad(44))
    # Inverting dy since latitudes in array increase as you go up
    dy = -np.deg2rad(0.25) * Re
    f = (2 * omega * np.sin(np.deg2rad(44))).to('1/s')
    ug, vg = geostrophic_wind(z * units.m, f, dx, dy, dim_order='xy')
    true_u = np.array([[21.97512, 21.97512, 22.08005],
                       [31.89402, 32.69477, 33.73863],
                       [38.43922, 40.18805, 42.14609]])
    true_v = np.array([[-10.93621, -7.83859, -4.54839],
                       [-10.74533, -7.50152, -3.24262],
                       [-8.66612, -5.27816, -1.45282]])
    assert_almost_equal(ug[1, 1], true_u[1, 1] * units('m/s'), 2)
    assert_almost_equal(vg[1, 1], true_v[1, 1] * units('m/s'), 2)


def test_no_ageostrophic_geopotential():
    """Test ageostrophic wind calculation with geopotential and no ageostrophic wind."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
    u = np.array([[-2, 0, 2]] * 3) * units('m/s')
    v = -u.T
    with pytest.warns(FutureWarning):
        uag, vag = ageostrophic_wind(z, 1 / units.sec, 100. * units.meter, 100. * units.meter,
                                     u, v, dim_order='xy')
    true = np.array([[0, 0, 0]] * 3) * units('m/s')
    assert_array_equal(uag, true)
    assert_array_equal(vag, true)


def test_ageostrophic_geopotential():
    """Test ageostrophic wind calculation with geopotential and ageostrophic wind."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
    u = v = np.array([[0, 0, 0]] * 3) * units('m/s')
    with pytest.warns(FutureWarning):
        uag, vag = ageostrophic_wind(z, 1 / units.sec, 100. * units.meter, 100. * units.meter,
                                     u, v, dim_order='xy')

    u_true = np.array([[2, 0, -2]] * 3) * units('m/s')
    v_true = -u_true.T

    assert_array_equal(uag, u_true)
    assert_array_equal(vag, v_true)


def test_no_ageostrophic_geopotential_future():
    """Test the updated ageostrophic wind function."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
    u = np.array([[-2, 0, 2]] * 3) * units('m/s')
    v = -u.T
    uag, vag = ageostrophic_wind_future(z, u, v, 1 / units.sec, 100. * units.meter,
                                        100. * units.meter, dim_order='xy')
    true = np.array([[0, 0, 0]] * 3) * units('m/s')
    assert_array_equal(uag, true)
    assert_array_equal(vag, true)


def test_ageostrophic_geopotential_future():
    """Test ageostrophic wind calculation with future input variable order."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
    u = v = np.array([[0, 0, 0]] * 3) * units('m/s')
    uag, vag = ageostrophic_wind_future(z, u, v, 1 / units.sec, 100. * units.meter,
                                        100. * units.meter, dim_order='xy')
    u_true = np.array([[2, 0, -2]] * 3) * units('m/s')
    v_true = -u_true.T
    assert_array_equal(uag, u_true)
    assert_array_equal(vag, v_true)


def test_streamfunc():
    """Test of Montgomery Streamfunction calculation."""
    t = 287. * units.kelvin
    hgt = 5000. * units.meter
    msf = montgomery_streamfunction(hgt, t)
    assert_almost_equal(msf, 337468.2500 * units('m^2 s^-2'), 4)


@check_and_silence_warning(FutureWarning)
def test_storm_relative_helicity_no_storm_motion():
    """Test storm relative helicity with no storm motion and differing input units."""
    u = np.array([0, 20, 10, 0]) * units('m/s')
    v = np.array([20, 0, 0, 10]) * units('m/s')
    u = u.to('knots')
    heights = np.array([0, 250, 500, 750]) * units.m

    positive_srh, negative_srh, total_srh = storm_relative_helicity(u, v, heights,
                                                                    depth=750 * units.meters)

    assert_almost_equal(positive_srh, 400. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(negative_srh, -100. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(total_srh, 300. * units('meter ** 2 / second ** 2 '), 6)


@check_and_silence_warning(FutureWarning)
def test_storm_relative_helicity_storm_motion():
    """Test storm relative helicity with storm motion and differing input units."""
    u = np.array([5, 25, 15, 5]) * units('m/s')
    v = np.array([30, 10, 10, 20]) * units('m/s')
    u = u.to('knots')
    heights = np.array([0, 250, 500, 750]) * units.m

    pos_srh, neg_srh, total_srh = storm_relative_helicity(u, v, heights,
                                                          depth=750 * units.meters,
                                                          storm_u=5 * units('m/s'),
                                                          storm_v=10 * units('m/s'))

    assert_almost_equal(pos_srh, 400. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(neg_srh, -100. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(total_srh, 300. * units('meter ** 2 / second ** 2 '), 6)


@check_and_silence_warning(FutureWarning)
def test_storm_relative_helicity_with_interpolation():
    """Test storm relative helicity with interpolation."""
    u = np.array([-5, 15, 25, 15, -5]) * units('m/s')
    v = np.array([40, 20, 10, 10, 30]) * units('m/s')
    u = u.to('knots')
    heights = np.array([0, 100, 200, 300, 400]) * units.m

    pos_srh, neg_srh, total_srh = storm_relative_helicity(u, v, heights,
                                                          bottom=50 * units.meters,
                                                          depth=300 * units.meters,
                                                          storm_u=5 * units('m/s'),
                                                          storm_v=10 * units('m/s'))

    assert_almost_equal(pos_srh, 400. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(neg_srh, -100. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(total_srh, 300. * units('meter ** 2 / second ** 2 '), 6)


@check_and_silence_warning(FutureWarning)
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
    p_srh, n_srh, t_srh = storm_relative_helicity(u_int, v_int,
                                                  hgt_int, 1000 * units('meter'),
                                                  bottom=0 * units('meter'),
                                                  storm_u=0 * units.knot,
                                                  storm_v=0 * units.knot)
    assert_almost_equal(p_srh, srh_true_p, 2)
    assert_almost_equal(n_srh, srh_true_n, 2)
    assert_almost_equal(t_srh, srh_true_t, 2)


@check_and_silence_warning(FutureWarning)
def test_storm_relative_helicity_agl():
    """Test storm relative helicity with heights above ground."""
    u = np.array([-5, 15, 25, 15, -5]) * units('m/s')
    v = np.array([40, 20, 10, 10, 30]) * units('m/s')
    u = u.to('knots')
    heights = np.array([100, 200, 300, 400, 500]) * units.m

    pos_srh, neg_srh, total_srh = storm_relative_helicity(u, v, heights,
                                                          bottom=50 * units.meters,
                                                          depth=300 * units.meters,
                                                          storm_u=5 * units('m/s'),
                                                          storm_v=10 * units('m/s'))

    # Check that heights isn't modified--checks for regression of #789
    assert_almost_equal(heights[0], 100 * units.m, 6)
    assert_almost_equal(pos_srh, 400. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(neg_srh, -100. * units('meter ** 2 / second ** 2 '), 6)
    assert_almost_equal(total_srh, 300. * units('meter ** 2 / second ** 2 '), 6)


@check_and_silence_warning(FutureWarning)
def test_storm_relative_helicity_masked():
    """Test that srh does not return masked values."""
    h = units.Quantity(np.ma.array([20.72, 234.85, 456.69, 683.21]), units.meter)
    u = units.Quantity(np.ma.array(np.zeros((4,))), units.knot)
    v = units.Quantity(np.zeros_like(u), units.knot)
    pos, neg, com = storm_relative_helicity(u, v, h, depth=500 * units.meter,
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
    vort = absolute_vorticity(u, v, 1 * units.meters, 2 * units.meters, lats, dim_order='yx')
    true_vort = np.array([[-2.499927, 3.500073, 13.00007],
                          [8.500050, -1.499950, -10.99995],
                          [-5.499975, -1.499975, 2.532525e-5]]) / units.sec
    assert_almost_equal(vort, true_vort, 5)

    # Now try for xy ordered
    vort = absolute_vorticity(u.T, v.T, 1 * units.meters, 2 * units.meters,
                              lats.T, dim_order='xy')
    assert_almost_equal(vort, true_vort.T, 5)


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
                        [21.9347249, 22.34606359, 22.77617081, 23.20049435, 23.58906447]]])\
        * (units.meters / units.sec)
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
                        [2.53695791, 1.35938207, 0.14519838, -1.06012605, -2.21257705]]])\
        * (units.meters / units.sec)
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

    assert_almost_equal(pvor, true_pv, 14)


def test_potential_vorticity_baroclinic_isobaric_real_data():
    """Test potential vorticity calculation with real isentropic data."""
    pres = [20000., 25000., 30000.] * units.Pa
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
                      [21.915337, 22.283215, 22.607704, 22.879448, 23.093569]]])\
        * (units.meters / units.sec)
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
                      [1.2759045, 0.05043932, -1.1469103, -2.264961, -3.2550638]]])\
        * (units.meters / units.sec)
    lats = np.array([57.5, 57., 56.5, 56., 55.5]) * units.degrees
    lons = np.array([227.5, 228., 228.5, 229., 229.5]) * units.degrees

    dx, dy = lat_lon_grid_deltas(lons, lats)

    pvor = potential_vorticity_baroclinic(theta, pres[:, None, None],
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

    # Now try for xy ordered
    pv = potential_vorticity_barotropic(heights.T, u.T, v.T, dx.T, dy.T, lats.T,
                                        dim_order='xy')
    avor = absolute_vorticity(u.T, v.T, dx.T, dy.T, lats.T, dim_order='xy')
    truth = avor / heights.T
    assert_almost_equal(pv, truth, 10)


def test_lat_lon_grid_deltas_geod_kwargs():
    """Test that geod kwargs are overridden by users #774."""
    lat = np.arange(40, 50, 2.5)
    lon = np.arange(-100, -90, 2.5)
    dx, dy = lat_lon_grid_deltas(lon, lat, a=4370997)
    dx_truth = np.array([[146095.76101984, 146095.76101984, 146095.76101984],
                         [140608.9751528, 140608.9751528, 140608.9751528],
                         [134854.56713287, 134854.56713287, 134854.56713287],
                         [128843.49645823, 128843.49645823, 128843.49645823]]) * units.meter
    dy_truth = np.array([[190720.72311199, 190720.72311199, 190720.72311199, 190720.72311199],
                         [190720.72311199, 190720.72311199, 190720.72311199, 190720.72311199],
                         [190720.72311199, 190720.72311199, 190720.72311199,
                          190720.72311199]]) * units.meter
    assert_almost_equal(dx, dx_truth, 4)
    assert_almost_equal(dy, dy_truth, 4)


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
    assert_almost_equal(uiaw, uiaw_truth, 7)
    assert_almost_equal(viaw, viaw_truth, 7)


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

    assert_almost_equal(q1, q1_truth, 18)
    assert_almost_equal(q2, q2_truth, 18)


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
    q2_truth = (np.array([[-2.3194263e-08, -2.3380160e-07, 2.4357380e-07, 2.6229040e-08],
                          [-1.4936626e-07, -1.8050836e-08, 1.8050836e-08, 1.5516129e-07],
                          [1.6903373e-07, 1.9457964e-08, -1.9457964e-08, -1.6243771e-07],
                          [4.4714390e-08, 2.3490489e-07, -2.2635441e-07, -4.0003646e-08]])
                * units('kg m^-2 s^-3'))

    assert_almost_equal(q1, q1_truth, 12)
    assert_almost_equal(q2, q2_truth, 12)


@pytest.fixture
def data_4d():
    """Define 4D data (extracted from Irma GFS example) for testing kinematics functions."""
    data = xr.open_dataset(get_test_data('irma_gfs_example.nc', False))
    data = data.metpy.parse_cf()
    data['Geopotential_height_isobaric'].attrs['units'] = 'm'
    subset = data.drop((
        'LatLon_361X720-0p25S-180p00E', 'Vertical_velocity_pressure_isobaric', 'isobaric1',
        'Relative_humidity_isobaric', 'reftime'
    )).sel(
        latitude=[46., 44., 42., 40.],
        longitude=[262., 267., 272., 277.],
        isobaric3=[50000., 70000., 85000.]
    ).isel(time1=[0, 1, 2])
    dx, dy = lat_lon_grid_deltas(subset['longitude'].values,
                                 subset['latitude'].values,
                                 initstring=subset['longitude'].metpy.cartopy_crs.proj4_init)
    return namedtuple('D_4D_Test_Data',
                      'height temperature pressure u v dx dy latitude')(
        subset['Geopotential_height_isobaric'].metpy.unit_array,
        subset['Temperature_isobaric'].metpy.unit_array,
        subset['isobaric3'].metpy.unit_array[None, :, None, None],
        subset['u-component_of_wind_isobaric'].metpy.unit_array,
        subset['v-component_of_wind_isobaric'].metpy.unit_array,
        dx[None, None],
        dy[None, None],
        subset['latitude'].values[None, None, :, None] * units('degrees')
    )


def test_vorticity_4d(data_4d):
    """Test vorticity on a 4D (time, pressure, y, x) grid."""
    vort = vorticity(data_4d.u, data_4d.v, data_4d.dx, data_4d.dy)
    truth = np.array([[[[-5.83650490e-05, 3.17327814e-05, 4.57268332e-05, 2.00732350e-05],
                        [2.14368312e-05, 1.95623237e-05, 4.15790182e-05, 6.90274641e-05],
                        [6.18610861e-05, 6.93600880e-05, 8.36201998e-05, 8.25922654e-05],
                        [-4.44038452e-05, 1.56487106e-04, 1.42605312e-04, -6.03981765e-05]],
                       [[-8.26772499e-07, 4.24638141e-05, -1.02560273e-05, 1.40379447e-05],
                        [1.16882545e-05, 1.06463071e-05, 2.84971990e-05, 6.22850560e-05],
                        [1.83850591e-05, 7.36387780e-06, 3.76622760e-05, 4.67188878e-05],
                        [3.19856719e-05, 2.80735317e-05, 3.73822586e-05, 5.40379931e-05]],
                       [[-2.84629423e-05, 3.82238141e-06, 3.96173636e-06, 5.13752737e-05],
                        [-2.18549443e-05, 5.28657636e-06, 2.00254459e-05, 3.38246076e-05],
                        [-1.97810827e-05, -2.51363102e-06, 2.87033130e-05, 3.01975044e-05],
                        [-2.34149501e-05, -1.82846160e-05, 2.95791089e-05, 3.41817364e-05]]],
                      [[[-3.66403309e-05, 2.45056689e-05, 8.30552352e-05, 2.42918324e-05],
                        [3.30814959e-05, 2.30523398e-05, 4.66571426e-05, 7.60789420e-05],
                        [7.65406561e-05, 4.98489001e-05, 6.61967180e-05, 9.90553670e-05],
                        [-3.83060455e-06, 8.82014475e-05, 1.11633279e-04, -4.43270102e-05]],
                       [[-2.47146999e-06, 3.95768075e-05, 3.76682359e-05, 3.79239346e-05],
                        [-4.60129429e-06, 2.05601660e-05, 2.89144970e-05, 3.01301961e-05],
                        [1.54778233e-05, 8.05069277e-06, 2.44051429e-05, 7.01730409e-05],
                        [2.07682219e-05, 2.16790897e-05, 3.38209456e-05, 9.11823021e-05]],
                       [[-7.12798691e-06, -2.81765143e-06, 1.93675069e-05, 6.21220857e-05],
                        [-1.80822794e-05, 7.10872010e-06, 1.40635809e-05, 1.80843580e-05],
                        [-3.01135264e-06, 2.56664766e-06, 2.25038301e-05, 3.69789825e-05],
                        [-1.48940627e-05, -6.28397440e-06, 3.66706625e-05, 1.13280233e-05]]],
                      [[[-2.13814161e-05, 3.10846718e-05, 9.42880991e-05, 6.20302960e-05],
                        [2.83685685e-05, 2.71376067e-05, 4.44470499e-05, 7.94154059e-05],
                        [7.29341555e-05, 3.07015029e-05, 3.70538789e-05, 7.75632608e-05],
                        [6.86595116e-05, 2.25094524e-05, 7.15703850e-05, 6.90873953e-05]],
                       [[-5.70566101e-07, 5.63627148e-05, 2.91960395e-05, 3.62726492e-05],
                        [-6.17247194e-06, 2.63672993e-05, 3.27525843e-05, 2.87151996e-05],
                        [8.82121811e-06, 6.46657237e-06, 2.03146030e-05, 4.99274322e-05],
                        [1.54560972e-05, 1.39161983e-06, -6.92832423e-06, 6.02698395e-05]],
                       [[3.01573325e-06, 2.95361596e-05, 3.30386503e-05, 4.50206712e-05],
                        [-3.44201203e-06, 7.98411843e-06, 1.31230998e-05, 1.82704434e-05],
                        [-5.97302093e-06, -6.76058488e-07, 5.89633276e-06, 1.82494546e-05],
                        [-2.96985363e-06, 3.86098537e-06, 5.24525482e-06,
                         2.72933874e-05]]]]) * units('s^-1')
    assert_array_almost_equal(vort, truth, 12)


def test_divergence_4d(data_4d):
    """Test divergence on a 4D (time, pressure, y, x) grid."""
    div = divergence(data_4d.u, data_4d.v, data_4d.dx, data_4d.dy)
    truth = np.array([[[[-8.43705083e-06, -5.42243991e-06, 1.42553766e-05, 2.81311077e-05],
                        [2.95334911e-05, -8.91904163e-06, 1.18532270e-05, -6.26196756e-06],
                        [-4.63583096e-05, -2.10525265e-05, 1.32571075e-05, 4.76118929e-05],
                        [-3.36862002e-05, 1.49431136e-05, -4.23301144e-05, 3.93742169e-05]],
                       [[1.69160375e-05, 1.54447811e-06, 4.11350021e-05, -5.08238612e-05],
                        [2.27239404e-05, -3.97652811e-06, 5.32329400e-06, 1.75756955e-05],
                        [-6.16991733e-06, -1.77132521e-06, -1.46585782e-05, 2.66081211e-06],
                        [-3.06896618e-05, -8.23671871e-06, -2.56998533e-05, -2.79187158e-05]],
                       [[-1.47210200e-05, -2.26015888e-05, -2.10309987e-05, -3.88000930e-05],
                        [2.87880179e-06, -9.97852896e-06, -1.02741993e-06, 4.53302920e-06],
                        [1.48614763e-05, 3.64899207e-06, 5.07255670e-06, 5.85619901e-06],
                        [1.12477665e-05, 1.61231699e-05, 1.13583495e-05, 3.92603086e-06]]],
                      [[[-3.95659837e-06, -8.16293111e-06, 6.64912076e-06, 4.82899740e-05],
                        [2.29285761e-05, -7.41730432e-07, 4.88714205e-06, -3.26931553e-05],
                        [-3.27492305e-05, -9.30212918e-06, 1.79834485e-06, 2.53811573e-05],
                        [-7.03628352e-05, 1.59599812e-05, -5.03928715e-05, 3.02766722e-05]],
                       [[1.67050619e-05, 1.24336555e-05, -2.22683301e-05, -1.89873955e-05],
                        [1.75618966e-05, -1.79165561e-06, 4.00327550e-06, -5.57201491e-06],
                        [4.02631582e-06, -5.29814574e-06, 3.59245019e-06, 4.16299189e-06],
                        [-3.62032526e-06, 4.00602251e-06, -2.17495860e-05, 2.93910418e-05]],
                       [[-6.50182516e-06, -3.06444044e-07, -4.68103153e-05, 6.54271734e-06],
                        [-9.22409986e-07, -6.80509227e-06, 1.57428914e-06, 2.13528516e-06],
                        [5.77636627e-06, -2.32628120e-06, 1.63766780e-05, 1.30647979e-05],
                        [-3.84054963e-06, 1.81368329e-05, -2.12456769e-08, 1.39177255e-05]]],
                      [[[-3.09706856e-06, -1.58174014e-05, 1.11042898e-05, 1.01863872e-05],
                        [2.22554541e-05, -2.18115261e-06, 2.95538179e-06, -1.27314237e-05],
                        [-2.01806928e-06, -1.44611342e-05, -1.60090851e-06, 1.51875027e-05],
                        [-7.45883555e-05, -2.17664690e-05, -6.59935118e-06, -9.51280586e-06]],
                       [[-2.84262294e-06, 1.83370481e-05, -1.60080375e-05, 5.94414530e-06],
                        [2.21275530e-05, 2.08417698e-06, -8.62049532e-06, -4.83025078e-06],
                        [6.84132907e-06, -1.74271778e-06, 4.96579520e-06, -8.14051216e-06],
                        [-5.91652815e-06, -1.27122109e-06, 1.33424166e-05, 2.89090443e-05]],
                       [[3.56617289e-06, -3.61628942e-06, -2.14971623e-05, 9.09440121e-06],
                        [1.15504071e-05, -2.35438670e-07, -1.00682327e-05, -7.83169489e-06],
                        [1.74820166e-06, -4.85659616e-07, 6.34687163e-06, -9.27089944e-06],
                        [9.23766788e-07, -2.85241737e-06, 1.68475020e-05,
                         -5.70982211e-06]]]]) * units('s^-1')
    assert_array_almost_equal(div, truth, 12)


def test_shearing_deformation_4d(data_4d):
    """Test shearing_deformation on a 4D (time, pressure, y, x) grid."""
    shdef = shearing_deformation(data_4d.u, data_4d.v, data_4d.dx, data_4d.dy)
    truth = np.array([[[[-2.32353766e-05, 3.38638896e-06, 2.68355706e-05, 1.06560395e-05],
                        [-6.40834716e-05, 1.01157390e-05, 1.72783215e-05, -2.41362735e-05],
                        [6.69848680e-07, -1.89007571e-05, -1.40877214e-05, 3.71581119e-05],
                        [6.36114984e-05, -1.08233745e-04, -9.64601102e-05, 7.32813538e-05]],
                       [[-2.42214688e-05, -1.01851671e-05, 5.54461375e-05, -3.51796268e-07],
                        [-2.71001778e-06, 9.30533079e-06, 2.03843188e-05, 3.34828205e-05],
                        [-7.27917172e-06, 1.72622100e-05, -5.55179147e-06, -1.31598103e-05],
                        [-2.51927242e-05, 9.17056498e-06, -2.24631811e-06, -5.35695138e-05]],
                       [[-2.57651839e-05, 1.01219942e-05, 4.53600390e-05, 5.28799494e-07],
                        [-1.55543764e-05, 6.18561775e-06, 2.36187660e-05, 2.52821250e-05],
                        [-2.67211842e-06, 1.14466225e-05, 7.99438411e-06, 3.06434113e-05],
                        [1.17029675e-05, 2.71857215e-05, -1.93883537e-06, 1.03237850e-05]]],
                      [[[5.36878109e-06, 1.03810936e-05, -1.74133759e-05, 3.67019082e-05],
                        [-3.68224309e-05, 8.02234002e-06, 2.97256283e-06, -2.41548002e-05],
                        [-1.03217116e-07, -3.55295991e-07, 1.04215491e-06, 3.06178781e-05],
                        [1.78849776e-05, -3.14217684e-05, -5.31904971e-05, 6.33706906e-05]],
                       [[-2.54172738e-05, -1.89184694e-05, 7.52187239e-06, 7.78263224e-06],
                        [-1.58491234e-05, 8.86850501e-06, 1.94682285e-05, 6.28154949e-06],
                        [2.87101909e-06, 1.02981542e-05, 1.49483081e-05, 1.11896135e-05],
                        [-6.24536223e-06, 5.02422684e-06, 3.65739211e-06, -4.43343790e-05]],
                       [[-1.97278534e-05, 7.98223535e-06, 1.93668724e-05, 1.17314603e-05],
                        [-1.80800662e-05, 7.10682312e-06, 1.58638789e-05, -7.11095379e-06],
                        [1.04957568e-05, 3.46915772e-06, 7.19233197e-06, 4.14864918e-05],
                        [1.30201499e-05, 7.22093417e-06, -1.46521398e-05, 5.00427528e-05]]],
                      [[[-8.31494391e-06, 2.74335132e-06, -2.40497584e-05, 1.75042968e-05],
                        [-1.88915366e-05, 3.28864180e-06, 1.34127052e-05, 1.23621458e-05],
                        [4.61243898e-07, 1.85506704e-05, 1.67855055e-05, 9.59377214e-06],
                        [6.06292303e-06, 2.92575008e-05, -1.44159217e-05, 2.17975694e-05]],
                       [[-1.00168319e-05, -4.57753168e-05, 8.50542316e-06, 3.44821467e-05],
                        [-1.65225222e-05, -4.66989095e-06, 6.65190573e-06, 1.71105129e-06],
                        [-4.23400810e-06, 1.50208950e-05, 1.80731201e-05, 5.36054473e-06],
                        [-2.10443897e-06, 1.80502669e-05, 4.39381815e-05, 5.78573040e-06]],
                       [[-2.08335492e-05, -3.03108476e-05, -3.85875023e-06, 2.70252773e-05],
                        [-4.78804481e-06, 1.24351472e-06, 6.82854110e-06, 5.67152289e-06],
                        [5.73158894e-06, 1.05747791e-05, 1.53497021e-05, 1.55510561e-05],
                        [1.23394357e-05, -1.98706807e-06, 1.56020711e-05,
                         3.89964205e-05]]]]) * units('s^-1')
    assert_array_almost_equal(shdef, truth, 12)


def test_stretching_deformation_4d(data_4d):
    """Test stretching_deformation on a 4D (time, pressure, y, x) grid."""
    stdef = stretching_deformation(data_4d.u, data_4d.v, data_4d.dx, data_4d.dy)
    truth = np.array([[[[3.47764258e-05, 2.24655678e-05, -5.99204286e-06, -2.81311151e-05],
                        [-1.00806414e-05, 2.43815624e-05, 5.10566770e-06, 3.02039392e-05],
                        [-5.93889988e-05, 4.15227142e-06, 3.93751112e-05, 5.52382202e-05],
                        [8.92010023e-05, 1.85531529e-05, 3.60056433e-05, -1.03321628e-04]],
                       [[2.96761128e-06, 1.36910447e-05, -4.34590663e-05, 1.80287489e-05],
                        [1.86757141e-05, 5.47290208e-06, -9.06422655e-06, 8.11203944e-06],
                        [1.34111946e-07, 1.26357749e-05, 2.72130530e-05, -3.62654235e-06],
                        [-1.35816208e-05, 1.87775033e-05, 5.84933984e-05, 5.04057146e-05]],
                       [[2.89236233e-05, 3.31889843e-05, 1.58664147e-05, 5.74675794e-06],
                        [1.68234432e-05, 1.52158343e-05, 5.26714302e-06, 1.21764691e-05],
                        [8.55744700e-06, 7.69832273e-06, -4.83112472e-06, -1.57549242e-05],
                        [-5.86025733e-06, 8.47198887e-06, -3.49088418e-07, -3.92962147e-05]]],
                      [[[3.69582950e-05, 1.86470363e-05, -2.80150634e-06, -3.51977497e-05],
                        [-6.46847071e-06, 2.69781373e-05, 6.95914436e-06, 5.98289917e-06],
                        [-4.00667252e-05, 5.05292624e-06, 4.75021137e-05, 6.24518714e-05],
                        [3.67260262e-05, 2.68548943e-06, 7.10293796e-05, -5.79403685e-05]],
                       [[-5.34297827e-06, -1.07157183e-06, 2.58835391e-05, 7.10885516e-06],
                        [1.26149579e-05, 1.35132438e-05, -2.75629799e-06, 4.32503739e-06],
                        [8.52816121e-06, 1.49554343e-05, 6.30626925e-06, 9.11577765e-06],
                        [2.68336326e-06, 5.36356177e-06, 5.47773716e-05, 4.06465999e-05]],
                       [[1.73474509e-05, 5.98748586e-06, 4.13875024e-05, -2.90086556e-05],
                        [4.23620645e-07, 1.02966295e-05, 5.15938898e-06, 7.09234799e-06],
                        [5.32951540e-06, 6.67206038e-06, -7.92655166e-06, 1.03541254e-05],
                        [1.46155702e-05, 1.33856872e-07, 4.47179844e-06, -4.46031161e-05]]],
                      [[[3.02111317e-05, 2.69212573e-05, -4.64855995e-06, 2.98329788e-06],
                        [-2.05441981e-06, 2.88664710e-05, 1.15095603e-05, -3.72867092e-06],
                        [-1.41578887e-05, 1.61511593e-05, 3.08142052e-05, 5.12063542e-05],
                        [-4.81886157e-06, 1.96583134e-05, 4.92309570e-05, 6.43248731e-05]],
                       [[-1.85904008e-05, -1.13648603e-05, 2.94359552e-05, -8.00997936e-06],
                        [1.62793512e-05, 8.39043368e-06, 7.12412372e-06, 7.32420819e-06],
                        [9.09319575e-06, 1.67115147e-05, 5.41579051e-06, 1.03134035e-05],
                        [2.63717343e-06, 5.48753335e-06, 1.28924212e-05, 3.38671775e-05]],
                       [[-4.08263202e-06, 1.03302483e-05, 2.30465372e-05, -2.51046121e-05],
                        [8.40123079e-06, 9.21367536e-06, 6.57669664e-06, -9.62598559e-06],
                        [6.70192818e-06, 9.41865112e-06, -1.75966046e-06, 4.68368828e-06],
                        [1.75811596e-05, 1.24562416e-05, -1.28654291e-05,
                         7.34949445e-06]]]]) * units('s^-1')
    assert_array_almost_equal(stdef, truth, 12)


def test_total_deformation_4d(data_4d):
    """Test total_deformation on a 4D (time, pressure, y, x) grid."""
    totdef = total_deformation(data_4d.u, data_4d.v, data_4d.dx, data_4d.dy)
    truth = np.array([[[[4.18244250e-05, 2.27193611e-05, 2.74964075e-05, 3.00817356e-05],
                        [6.48714934e-05, 2.63967566e-05, 1.80168876e-05, 3.86631303e-05],
                        [5.93927763e-05, 1.93514851e-05, 4.18194127e-05, 6.65731647e-05],
                        [1.09559306e-04, 1.09812399e-04, 1.02960960e-04, 1.26670895e-04]],
                       [[2.44025874e-05, 1.70640656e-05, 7.04483116e-05, 1.80321809e-05],
                        [1.88713140e-05, 1.07954545e-05, 2.23087574e-05, 3.44514797e-05],
                        [7.28040706e-06, 2.13926787e-05, 2.77735961e-05, 1.36503632e-05],
                        [2.86205132e-05, 2.08972221e-05, 5.85365151e-05, 7.35556175e-05]],
                       [[3.87352641e-05, 3.46981764e-05, 4.80549296e-05, 5.77103594e-06],
                        [2.29121554e-05, 1.64250869e-05, 2.41989442e-05, 2.80615795e-05],
                        [8.96493815e-06, 1.37945402e-05, 9.34076781e-06, 3.44562954e-05],
                        [1.30882414e-05, 2.84752182e-05, 1.97001150e-06, 4.06297062e-05]]],
                      [[[3.73462098e-05, 2.13419556e-05, 1.76372928e-05, 5.08518598e-05],
                        [3.73862612e-05, 2.81456539e-05, 7.56741832e-06, 2.48847233e-05],
                        [4.00668582e-05, 5.06540214e-06, 4.75135443e-05, 6.95535096e-05],
                        [4.08493993e-05, 3.15363185e-05, 8.87378259e-05, 8.58657716e-05]],
                       [[2.59727785e-05, 1.89487929e-05, 2.69543347e-05, 1.05406445e-05],
                        [2.02566502e-05, 1.61634816e-05, 1.96623777e-05, 7.62652034e-06],
                        [8.99846010e-06, 1.81581110e-05, 1.62240854e-05, 1.44327701e-05],
                        [6.79742509e-06, 7.34919385e-06, 5.48993347e-05, 6.01471798e-05]],
                       [[2.62701780e-05, 9.97827981e-06, 4.56946507e-05, 3.12910413e-05],
                        [1.80850283e-05, 1.25110957e-05, 1.66817850e-05, 1.00432596e-05],
                        [1.17713485e-05, 7.52006948e-06, 1.07032640e-05, 4.27590565e-05],
                        [1.95739418e-05, 7.22217474e-06, 1.53193401e-05, 6.70351779e-05]]],
                      [[[3.13344980e-05, 2.70606739e-05, 2.44948972e-05, 1.77567021e-05],
                        [1.90029154e-05, 2.90531980e-05, 1.76740102e-05, 1.29122281e-05],
                        [1.41654000e-05, 2.45964900e-05, 3.50894348e-05, 5.20973241e-05],
                        [7.74470545e-06, 3.52484133e-05, 5.12982059e-05, 6.79177688e-05]],
                       [[2.11172897e-05, 4.71650260e-05, 3.06401319e-05, 3.54002572e-05],
                        [2.31950645e-05, 9.60246108e-06, 9.74684506e-06, 7.52141756e-06],
                        [1.00306048e-05, 2.24700247e-05, 1.88671263e-05, 1.16233270e-05],
                        [3.37392161e-06, 1.88659788e-05, 4.57905920e-05, 3.43578287e-05]],
                       [[2.12298059e-05, 3.20228280e-05, 2.33673454e-05, 3.68864089e-05],
                        [9.66985273e-06, 9.29721155e-06, 9.48060716e-06, 1.11725454e-05],
                        [8.81855731e-06, 1.41611066e-05, 1.54502349e-05, 1.62410677e-05],
                        [2.14792655e-05, 1.26137383e-05, 2.02223611e-05,
                         3.96829419e-05]]]]) * units('s^-1')
    assert_array_almost_equal(totdef, truth, 12)


def test_frontogenesis_4d(data_4d):
    """Test frontogenesis on a 4D (time, pressure, y, x) grid."""
    thta = potential_temperature(data_4d.pressure, data_4d.temperature)
    frnt = frontogenesis(thta, data_4d.u, data_4d.v, data_4d.dx, data_4d.dy)
    truth = np.array([[[[4.23682195e-10, -6.42818314e-12, -2.16491106e-10, -3.81845902e-10],
                        [-5.28632893e-10, -6.99413155e-12, -4.77775880e-11, 2.95949984e-10],
                        [7.82193227e-10, 3.55234312e-10, 2.14592821e-11, -5.20704165e-10],
                        [-3.51045184e-10, 2.06780694e-10, 1.68485199e-09, -1.46174872e-09]],
                       [[-7.24768625e-11, 1.07136516e-10, -1.33696585e-10, 3.43097590e-10],
                        [-5.01911031e-11, 2.14208730e-11, -4.73005145e-11, 9.54558115e-11],
                        [4.78125962e-11, 6.95838402e-11, 3.53993328e-10, -7.14986278e-11],
                        [6.14372837e-10, 1.41441177e-10, 8.45358549e-10, 1.36583089e-09]],
                       [[2.06344624e-11, 3.21877016e-10, 5.56930831e-10, 1.42479782e-10],
                        [9.85250671e-11, 1.06837011e-10, 5.71410578e-11, -4.75266666e-12],
                        [-6.42402291e-11, -2.11239598e-11, -1.21400141e-11, 2.11343115e-10],
                        [-6.98848390e-11, -4.11958693e-11, -1.75826411e-10, -1.78597026e-10]]],
                      [[[1.75135966e-10, -1.28928980e-11, -5.23466009e-11, -3.77702045e-10],
                        [-1.89751794e-10, -2.39181519e-11, 1.11100290e-11, 3.27299708e-10],
                        [5.03778532e-10, 6.01896046e-11, 2.52803022e-10, 2.63665975e-10],
                        [8.58126215e-10, -1.03984888e-10, 1.36888801e-09, -2.55913184e-11]],
                       [[-4.65433709e-11, -4.29058715e-11, 1.37345453e-10, 1.99587747e-10],
                        [-7.54157254e-11, 3.17296645e-11, 1.98329738e-11, 7.03907102e-11],
                        [2.07597572e-11, 9.76879251e-11, 3.64976958e-11, 9.05618888e-11],
                        [1.08070144e-10, 4.28175872e-12, 7.19436313e-10, -4.06647884e-10]],
                       [[3.53005192e-11, 6.97543696e-12, 7.69214961e-10, 1.72736149e-10],
                        [8.60386822e-11, 6.45426820e-11, 5.98938946e-11, -3.48393954e-11],
                        [-8.04261595e-11, 3.35202076e-11, -6.74886256e-11, 2.13321809e-10],
                        [3.19647867e-12, -1.19646868e-10, -3.89001592e-11, 2.32044147e-10]]],
                      [[[-1.06347076e-10, 1.43182380e-10, -1.67871828e-10, 7.55627348e-12],
                        [-2.10533376e-11, -1.36653603e-11, 5.14204938e-11, 4.52818256e-11],
                        [9.35607979e-11, -1.90340153e-11, 5.14752944e-11, 3.57906394e-10],
                        [1.47969288e-09, -6.40884777e-11, -2.00932232e-10, 3.79333810e-10]],
                       [[1.39959107e-10, -3.68301267e-10, 2.35651001e-10, 1.53240049e-10],
                        [-2.59723933e-10, 3.90319800e-11, 9.15095885e-11, 3.55206479e-11],
                        [6.07203827e-12, 8.14448434e-11, 2.37589779e-11, 1.56707972e-10],
                        [6.01077213e-11, 1.43024438e-11, 2.19842020e-10, 6.06611960e-12]],
                       [[5.76777660e-11, 1.50880981e-10, 9.80083831e-11, 1.37735770e-10],
                        [-5.69004435e-11, 1.61334326e-11, 8.32223545e-11, 1.07345248e-10],
                        [-5.82285173e-11, 1.03267739e-12, 9.19171693e-12, 1.73823741e-10],
                        [-2.33302976e-11, 1.01795295e-10, 4.19754683e-12,
                         5.18286088e-10]]]]) * units('K/m/s')
    assert_array_almost_equal(frnt, truth, 16)


def test_geostrophic_wind_4d(data_4d):
    """Test geostrophic_wind on a 4D (time, pressure, y, x) grid."""
    f = coriolis_parameter(data_4d.latitude)
    u_g, v_g = geostrophic_wind(data_4d.height, f, data_4d.dx, data_4d.dy)
    u_g_truth = np.array([[[[4.40351577, 12.52087174, 20.6458988, 3.17057524],
                            [14.11461945, 17.13672114, 22.06686549, 28.28270102],
                            [24.47454294, 22.86342357, 31.74065923, 41.48130088],
                            [35.59988608, 29.85398309, 50.68045123, 41.40714946]],
                           [[7.36263117, 11.15525254, 15.36136167, 8.9043257],
                            [8.36777239, 12.52326604, 13.39382587, 14.3316852],
                            [10.38214971, 13.05048176, 16.57141998, 20.60619861],
                            [13.53264547, 12.63889256, 25.51465438, 27.85194657]],
                           [[5.7558101, 8.87403945, 12.12136937, 6.95914488],
                            [5.63469768, 9.23443489, 9.46733739, 9.64257854],
                            [5.15675792, 8.93121196, 10.1444189, 10.03116286],
                            [4.2776387, 7.88423455, 14.54891694, 7.85445617]]],
                          [[[2.56196227, 12.12574931, 18.89599304, 9.31367555],
                            [11.14381317, 16.08241758, 22.90372798, 23.24530036],
                            [21.19957068, 18.21200016, 27.48622742, 37.93750724],
                            [32.9424023, 18.30589852, 32.72832352, 53.53662491]],
                           [[5.89065665, 10.24260864, 13.9982738, 7.63017804],
                            [7.7319368, 12.49290588, 13.8820136, 12.98573492],
                            [9.40028538, 12.48920597, 15.31293349, 18.73778037],
                            [10.88139947, 9.96423153, 18.47901734, 24.95509777]],
                           [[5.3790145, 9.32173797, 9.01530817, 3.68853401],
                            [5.42563016, 8.9380796, 9.35289746, 9.01581846],
                            [4.95407268, 8.35221797, 9.30403934, 11.10242176],
                            [3.90093595, 7.537516, 8.82237876, 9.57266969]]],
                          [[[4.077062, 9.91350661, 14.63954845, 11.45133198],
                            [9.22655905, 15.4118828, 20.8655254, 20.36949692],
                            [17.30011986, 16.30232673, 23.25100815, 32.47299215],
                            [28.67482075, 12.0424244, 21.34996542, 48.18503321]],
                           [[4.67946573, 7.67735341, 7.67258169, 7.43729616],
                            [6.37289243, 10.60261239, 12.10568057, 11.53061917],
                            [7.78079442, 11.18649202, 14.92802562, 16.19084404],
                            [8.87459397, 9.15376375, 15.95950978, 21.50271574]],
                           [[4.07034519, 6.49914852, 4.9842596, 5.1120617],
                            [4.20250871, 6.7603074, 8.51019946, 8.51714298],
                            [3.85757805, 6.9373935, 9.8250342, 10.52736698],
                            [2.97756024, 7.02083208, 8.67190524,
                             10.98516411]]]]) * units('m/s')
    v_g_truth = np.array([[[[-2.34336304e+01, -1.93589800e+01, -7.42980465e+00,
                             1.23538955e+01],
                            [-2.05343103e+01, -1.59281972e+01, -7.22778771e+00,
                             5.56691831e+00],
                            [-2.12483061e+01, -1.50276890e+01, -1.26159708e+00,
                             2.00499696e+01],
                            [-2.82673839e+01, -1.22322398e+01, 2.74929719e+00,
                             1.66772271e+01]],
                           [[-2.11572490e+01, -1.57068398e+01, -7.16428821e+00,
                             4.47040576e+00],
                            [-1.85233793e+01, -1.38641633e+01, -7.23745352e+00,
                             1.35674995e+00],
                            [-1.48069287e+01, -1.29873005e+01, -6.19402168e+00,
                             5.57290769e+00],
                            [-1.63708722e+01, -1.07203268e+01, -3.25405588e+00,
                             6.02794043e+00]],
                           [[-1.83721994e+01, -1.51434535e+01, -8.30361332e+00,
                             2.14732109e+00],
                            [-1.60334603e+01, -1.37004633e+01, -8.52272656e+00,
                             -5.00249972e-01],
                            [-1.25811419e+01, -1.30858045e+01, -8.11893604e+00,
                             2.31946331e+00],
                            [-1.07972595e+01, -1.12050147e+01, -8.05482698e+00,
                             -1.34669618e+00]]],
                          [[[-2.47128002e+01, -2.06093912e+01, -7.53605837e+00,
                             1.45071982e+01],
                            [-2.04618167e+01, -1.66379272e+01, -6.94777385e+00,
                             8.60864325e+00],
                            [-2.03847527e+01, -1.41640171e+01, -3.58588785e+00,
                             1.13496351e+01],
                            [-3.06442215e+01, -1.34818877e+01, 3.63145087e+00,
                             2.06957942e+01]],
                           [[-2.20117576e+01, -1.60592509e+01, -6.79979611e+00,
                             5.76660686e+00],
                            [-1.88926841e+01, -1.40452204e+01, -7.10711240e+00,
                             1.92164004e+00],
                            [-1.49428085e+01, -1.27155409e+01, -6.56034626e+00,
                             3.52277542e+00],
                            [-1.56847892e+01, -1.10535722e+01, -3.82991450e+00,
                             5.98618380e+00]],
                           [[-1.89418619e+01, -1.48982095e+01, -8.32871820e+00,
                             7.66612047e-01],
                            [-1.57997569e+01, -1.38337366e+01, -9.12720817e+00,
                             -1.68017155e+00],
                            [-1.34002412e+01, -1.27868867e+01, -8.32854573e+00,
                             -2.52183180e-02],
                            [-1.10305552e+01, -1.16852908e+01, -7.77451018e+00,
                             7.01786569e-01]]],
                          [[[-2.87198561e+01, -2.07541862e+01, -7.39120444e+00,
                             1.13690891e+01],
                            [-2.50727626e+01, -1.76277299e+01, -6.48428638e+00,
                             8.35756789e+00],
                            [-2.15686958e+01, -1.43774917e+01, -4.66795064e+00,
                             7.55992752e+00],
                            [-3.08303915e+01, -1.46678239e+01, 2.17589132e+00,
                             1.97007540e+01]],
                           [[-2.14034948e+01, -1.55089179e+01, -7.19566930e+00,
                             3.53625110e+00],
                            [-1.85643117e+01, -1.42866005e+01, -7.10227950e+00,
                             2.98865138e+00],
                            [-1.52824784e+01, -1.23952994e+01, -6.71565437e+00,
                             1.75645659e+00],
                            [-1.53343446e+01, -1.06296646e+01, -4.49885811e+00,
                             3.05807491e+00]],
                           [[-1.62094234e+01, -1.41161427e+01, -9.20541452e+00,
                             -1.47723905e+00],
                            [-1.41272620e+01, -1.33895366e+01, -9.16198151e+00,
                             -1.44459685e+00],
                            [-1.29925870e+01, -1.17892453e+01, -8.27421454e+00,
                             -2.44749477e+00],
                            [-1.08991833e+01, -1.03581717e+01, -7.35501458e+00,
                             -1.88971184e+00]]]]) * units('m/s')
    assert_array_almost_equal(u_g, u_g_truth, 6)
    assert_array_almost_equal(v_g, v_g_truth, 6)


def test_intertial_advective_wind_4d(data_4d):
    """Test intertial_advective_wind on a 4D (time, pressure, y, x) grid."""
    f = coriolis_parameter(data_4d.latitude)
    u_g, v_g = geostrophic_wind(data_4d.height, f, data_4d.dx, data_4d.dy)
    u_i, v_i = inertial_advective_wind(u_g, v_g, u_g, v_g, data_4d.dx, data_4d.dy,
                                       data_4d.latitude)
    u_i_truth = np.array([[[[-4.74579332, -6.36486064, -7.20354171, -11.08307751],
                            [-1.88515129, -4.33855679, -6.82871465, -9.38096911],
                            [2.308649, -6.93391208, -14.06293133, -20.60786775],
                            [-0.92388354, -13.76737076, -17.9039117, -23.71419254]],
                           [[-2.60558413, -3.48755492, -3.62050089, -4.18871134],
                            [-3.36965812, -2.57689219, -2.66529828, -3.34582207],
                            [-0.56309499, -2.3322732, -4.37379768, -6.6663065],
                            [1.70092943, -3.59623514, -5.94640587, -7.50380432]],
                           [[-1.60508844, -2.30572073, -2.39044749, -2.59511279],
                            [-2.18854472, -1.47967397, -1.57319604, -2.24386278],
                            [-1.10582176, -1.24627092, -2.02075175, -3.314856],
                            [-0.25911941, -1.62294229, -1.75103256, -1.21885814]]],
                          [[[-6.69345313, -6.73506869, -7.9082287, -12.43972804],
                            [-2.21048835, -5.05651724, -7.72691754, -11.18333726],
                            [2.66904547, -4.81530785, -9.54984823, -12.89835729],
                            [8.55752862, -7.70089375, -12.37978952, -10.22208691]],
                           [[-3.17485999, -3.54021424, -3.54593593, -4.29515483],
                            [-3.68981249, -2.85516457, -2.76603925, -3.31604629],
                            [-1.16624451, -2.17242275, -3.57186768, -5.25444633],
                            [1.41851647, -2.44637201, -4.63693023, -6.09680756]],
                           [[-3.2219496, -1.90321215, -1.16750878, -1.08832287],
                            [-2.0239913, -1.38273223, -1.39926438, -1.92743159],
                            [-1.31353175, -1.15761322, -1.72857968, -2.81015813],
                            [-0.96137414, -0.94030556, -1.52657711, -2.56457651]]],
                          [[[-5.10794084, -5.32937859, -5.93090309, -8.05663994],
                            [-5.25295525, -6.02259284, -7.06582462, -9.0763472],
                            [0.32747247, -4.38931301, -7.24210551, -8.856658],
                            [11.82591067, -3.51482111, -8.18935835, -3.90270871]],
                           [[-2.9420404, -1.93269048, -1.78193608, -2.21710641],
                            [-2.96678921, -2.48380116, -2.64978243, -3.39496054],
                            [-1.42507824, -2.23090734, -3.01660858, -3.95003961],
                            [0.38000295, -2.10863221, -3.40584443, -4.06614801]],
                           [[-1.84525414, -0.73542408, -0.62568812, -1.18458192],
                            [-0.90497548, -1.10518325, -1.44073904, -1.95278103],
                            [-0.97196521, -1.22914653, -1.48019684, -1.79349709],
                            [-1.29544691, -0.9808466, -1.24778616,
                             -1.95945874]]]]) * units('m/s')
    v_i_truth = np.array([[[[1.03108918e+01, 5.87304544e+00, -3.23865690e+00,
                             -1.88225987e+01],
                            [9.87187503e+00, 5.33610060e+00, 4.80874417e+00,
                             3.92484555e-02],
                            [6.37856912e+00, 6.46296166e+00, 8.14267044e+00,
                             4.37232518e+00],
                            [-1.30385124e+00, 1.01032585e+01, 4.20243238e+00,
                             -1.97934081e+01]],
                           [[1.10360108e+00, 2.30280536e+00, -1.82078930e+00,
                             -3.54284012e+00],
                            [2.43663102e+00, 1.35818636e+00, 4.92919838e-01,
                             -9.85544117e-03],
                            [2.33985677e+00, 1.03370035e+00, 3.28069921e+00,
                             4.50046765e-01],
                            [2.93689077e-01, 1.43848430e+00, 6.69758269e+00,
                             -4.27897434e+00]],
                           [[4.77869846e-01, 1.14482717e+00, -1.82404796e+00,
                             -1.95731131e+00],
                            [5.19464097e-01, 4.52949199e-01, -3.26412809e-01,
                             6.88744088e-02],
                            [2.51097720e-01, 1.43448773e-01, 1.08982754e+00,
                             -9.69963394e-02],
                            [-3.37860948e-01, 2.48187099e-01, 2.41935519e+00,
                             -2.84847302e+00]]],
                          [[[9.00342804e+00, 6.74193832e+00, 5.48141003e-01,
                             -1.25005172e+01],
                            [9.56628265e+00, 4.57654669e+00, 3.34479904e+00,
                             -7.13103555e+00],
                            [5.46655351e+00, 2.14241047e+00, 7.51934330e+00,
                             2.43229680e+00],
                            [-5.48082957e+00, -6.46852260e-01, 1.34334674e+01,
                             1.61485491e+01]],
                           [[2.49375451e+00, 3.34815514e+00, -7.09673457e-01,
                             -3.42185701e+00],
                            [2.69963182e+00, 1.64621317e+00, 2.91799176e-01,
                             -1.12584231e+00],
                            [1.83462164e+00, 1.71608154e-01, 1.87927013e+00,
                             7.54482898e-01],
                            [-4.86175507e-01, -1.06374611e+00, 4.20283383e+00,
                             1.54789418e+00]],
                           [[1.05175282e+00, 2.36715709e-01, -4.35406547e-01,
                             -9.39935118e-01],
                            [5.26821709e-01, 1.34167595e-01, 6.74485663e-02,
                             1.18351992e-01],
                            [9.51152970e-02, 3.63519903e-02, 2.14587938e-01,
                             6.10557463e-01],
                            [-2.42904366e-01, -5.80309556e-02, -3.63185957e-02,
                             2.28010678e-01]]],
                          [[[5.18112516e+00, 8.23347995e+00, 2.85922078e+00,
                             -5.58457816e+00],
                            [8.85157651e+00, 4.70839103e+00, 2.51314815e+00,
                             -5.64246393e+00],
                            [7.54770787e+00, 8.21372199e-02, 4.70293099e+00,
                             3.47174970e+00],
                            [-1.92174464e+00, -5.91657547e+00, 1.00629730e+01,
                             2.62854305e+01]],
                           [[2.20347520e+00, 3.00714687e+00, 1.59377661e+00,
                             -6.41826692e-01],
                            [2.15604582e+00, 1.86128202e+00, 1.28260457e+00,
                             -1.03918888e+00],
                            [1.50501488e+00, 5.74547239e-01, 1.52092784e+00,
                             -3.94591487e-01],
                            [2.83614456e-02, -8.95222937e-01, 2.49176874e+00,
                             1.81097696e+00]],
                           [[6.98668139e-01, 2.56635250e-01, 1.74332893e+00,
                             3.79321436e-01],
                            [2.39593746e-01, 4.88748160e-01, 1.16884612e+00,
                             -7.54110131e-03],
                            [-6.40285805e-02, 5.82931602e-01, 4.67005716e-01,
                             3.76288542e-02],
                            [-2.10896883e-01, 5.17706856e-01, -4.13562541e-01,
                             6.96975860e-01]]]]) * units('m/s')
    assert_array_almost_equal(u_i, u_i_truth, 6)
    assert_array_almost_equal(v_i, v_i_truth, 6)


def test_q_vector_4d(data_4d):
    """Test q_vector on a 4D (time, pressure, y, x) grid."""
    f = coriolis_parameter(data_4d.latitude)
    u_g, v_g = geostrophic_wind(data_4d.height, f, data_4d.dx, data_4d.dy)
    q1, q2 = q_vector(u_g, v_g, data_4d.temperature, data_4d.pressure, data_4d.dx, data_4d.dy)
    q1_truth = np.array([[[[-8.98245364e-13, 2.03803219e-13, 2.88874668e-12, 2.18043424e-12],
                           [4.37446820e-13, 1.21145200e-13, 1.51859353e-12, 3.82803347e-12],
                           [-1.20538030e-12, 2.27477298e-12, 3.47570178e-12, 3.03123012e-12],
                           [-1.51597275e-12, 8.02915408e-12, 7.71292472e-12, -2.22078527e-12]],
                          [[5.72960497e-13, 1.04264321e-12, -1.75695523e-13, 1.20745997e-12],
                           [2.94807953e-13, 5.80261767e-13, 6.23668595e-13, 7.31474131e-13],
                           [-4.04218965e-14, 3.24794013e-13, 1.39539675e-12, 2.82242029e-12],
                           [3.27509076e-13, 5.61307677e-13, 1.13454829e-12, 4.63551274e-12]],
                          [[2.23877015e-13, 5.77177907e-13, 1.62133659e-12, 5.43858376e-13],
                           [2.65333917e-13, 2.41006445e-13, 3.72510595e-13, 7.35822030e-13],
                           [6.56644633e-14, 1.99773842e-13, 5.20573457e-13, 1.69706608e-12],
                           [4.15915138e-14, 1.19910880e-13, 1.03632944e-12, 1.99561829e-12]]],
                         [[[-2.68870846e-13, 1.35977736e-12, 4.17548337e-12, 1.50465522e-12],
                           [4.62457018e-14, 1.25888111e-13, 2.15928418e-12, 4.70656495e-12],
                           [-1.25393137e-12, 9.54737370e-13, 1.48443002e-12, 2.12375621e-12],
                           [-2.93284658e-12, 6.06555344e-12, 4.21151397e-12, -2.12250513e-12]],
                          [[4.23461674e-13, 1.39393686e-13, 5.89509120e-13, 2.55041326e-12],
                           [5.73125714e-13, 5.60965341e-13, 7.65040451e-13, 9.49571939e-13],
                           [2.17153819e-14, 3.97023968e-13, 1.09194718e-12, 1.90731542e-12],
                           [1.45101233e-13, 1.79588608e-13, 1.03018848e-12, 3.62186462e-12]],
                          [[5.32674437e-13, 5.13465061e-13, 1.15582657e-12, 1.04827520e-12],
                           [2.77261345e-13, 2.33645555e-13, 4.59592371e-13, 5.34293340e-13],
                           [1.47376125e-13, 1.95746242e-13, 3.45854003e-13, 7.47741411e-13],
                           [-2.14078421e-14, 1.75226662e-13, 4.85424103e-13, 1.10808035e-12]]],
                         [[[6.41348753e-13, 1.88256910e-12, 5.21213092e-12, 2.07707653e-12],
                           [1.30753737e-12, 4.77125469e-13, 2.15204760e-12, 3.07374453e-12],
                           [-2.30546806e-13, 2.49929428e-13, 8.82215204e-14, 2.45990265e-12],
                           [-7.25812141e-12, 8.47072439e-13, -2.06762495e-12,
                            -4.40132129e-12]],
                          [[6.03705941e-13, -6.71320661e-13, 9.10543636e-13, 5.82480651e-13],
                           [9.54081741e-13, 6.11781160e-13, 6.95995265e-13, 8.67169047e-13],
                           [7.86580678e-14, 5.27405484e-13, 7.45800341e-13, 1.33965768e-12],
                           [2.22480631e-13, -1.98920384e-13, 8.56608245e-13, 1.59793218e-12]],
                          [[4.47195537e-13, 2.18235390e-13, 3.30926531e-13, -4.06675908e-14],
                           [1.70517246e-13, 2.18234962e-13, 3.78622612e-13, 5.03962144e-13],
                           [2.59462161e-13, 2.65626826e-13, 2.04642555e-13, 6.02812047e-13],
                           [1.69339642e-13, 2.91716502e-13, -1.20043003e-14,
                            4.43770388e-13]]]]) * units('m^2 kg^-1 s^-1')
    q2_truth = np.array([[[[3.33980776e-12, -1.32969763e-13, 1.01454470e-12, 6.02652581e-12],
                           [2.52898242e-13, -1.71069245e-13, -8.24708561e-13, 1.66384429e-13],
                           [-3.50646511e-12, -1.68929195e-12, 7.76215111e-13, 1.54486058e-12],
                           [-1.75492099e-12, -3.86524071e-12, -1.89368596e-12,
                            -5.14689517e-12]],
                          [[-2.09848775e-13, -6.25683634e-13, -1.40009292e-13, 1.08972893e-12],
                           [-2.58259284e-13, -2.67211578e-13, -6.41928957e-14, 5.90625597e-13],
                           [-2.73346325e-13, -2.28248227e-13, -4.76577835e-13,
                            -8.48559875e-13],
                           [1.21003124e-12, -5.10541546e-13, 6.35947149e-14, 2.44893915e-12]],
                          [[-6.72309334e-14, -3.56791270e-13, -4.13553842e-14, 3.81212108e-13],
                           [-3.55860413e-13, -1.22880634e-13, -3.19443665e-14,
                            -4.71232601e-14],
                           [-2.82289531e-13, -1.20965929e-13, 1.14160715e-13, -6.85113982e-14],
                           [5.17465531e-14, -4.61129211e-13, 5.33793701e-13, 1.28285338e-12]]],
                         [[[1.71894904e-12, -1.35675428e-12, 1.48328005e-13, 3.22454170e-12],
                           [-2.12666583e-13, -1.17551681e-13, -6.93968059e-13, 1.76659826e-12],
                           [-2.67906914e-12, -3.78250861e-13, -9.88730956e-13, 2.88200442e-12],
                           [1.48225123e-12, 2.15004833e-13, -4.84554577e-12, 2.77242999e-12]],
                          [[-3.09626209e-13, -2.52138997e-13, 4.58311589e-14, 2.03206766e-12],
                           [-3.95662347e-13, -2.99828956e-13, 1.08715446e-14, 1.06096030e-12],
                           [-2.46502471e-13, -2.43524217e-13, -3.81250581e-13,
                            -1.70270366e-13],
                           [8.12479206e-13, -1.38629628e-13, -8.05591138e-13,
                            -7.80286006e-13]],
                          [[-2.19626566e-13, -1.52852503e-13, 4.07706963e-13, 1.52323163e-12],
                           [-2.56235985e-13, -1.20817691e-13, 6.51260820e-15, 3.49591511e-13],
                           [-2.44063890e-13, -1.21871642e-13, -9.09798480e-14,
                            -1.59903476e-13],
                           [-2.47929201e-13, -1.77269110e-13, -1.12991330e-13,
                            -6.06795348e-13]]],
                         [[[-6.48288201e-13, -1.96951432e-12, -5.53508048e-13, 1.94507133e-12],
                           [-2.00769011e-12, -3.72469047e-13, -4.59116219e-13, 1.11322705e-13],
                           [-3.83507643e-12, 1.18054543e-13, -4.24001455e-13, -5.88688871e-13],
                           [-1.84528711e-12, 1.54974343e-12, -7.36123184e-13, 1.06256777e-13]],
                          [[-4.58487019e-13, -1.89124158e-13, 2.58416604e-13, 8.14652306e-13],
                           [-6.09664269e-13, -3.51509413e-13, 2.39576397e-13, 5.80539044e-13],
                           [-1.68850738e-13, -3.49553817e-13, -2.26470205e-13, 7.79989044e-13],
                           [2.23081718e-13, 1.20195366e-13, -1.01508013e-12, -2.15527487e-13]],
                          [[-1.68054338e-13, -5.06878852e-14, 2.77697698e-13, 8.37521961e-13],
                           [-1.39462599e-13, -1.36628363e-13, 3.13920124e-14, 4.55413406e-13],
                           [-1.06658890e-13, -2.19817426e-13, -8.35968065e-14, 1.88190788e-13],
                           [-2.27182863e-13, -2.74607819e-13, -1.10587309e-13,
                            -3.88915866e-13]]]]) * units('m^2 kg^-1 s^-1')
    assert_array_almost_equal(q1, q1_truth, 18)
    assert_array_almost_equal(q2, q2_truth, 18)
