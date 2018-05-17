# Copyright (c) 2008,2015,2017,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `kinematics` module."""

import numpy as np
import pytest

from metpy.calc import (absolute_vorticity, advection, ageostrophic_wind,
                        convergence_vorticity, divergence,
                        frontogenesis, geostrophic_wind, get_wind_components, h_convergence,
                        inertial_advective_wind, lat_lon_grid_deltas, lat_lon_grid_spacing,
                        montgomery_streamfunction, potential_vorticity_baroclinic,
                        potential_vorticity_barotropic, q_vector, shearing_deformation,
                        shearing_stretching_deformation, static_stability,
                        storm_relative_helicity, stretching_deformation, total_deformation,
                        v_vorticity, vorticity)
from metpy.constants import g, omega, Re
from metpy.deprecation import MetpyDeprecationWarning
from metpy.testing import assert_almost_equal, assert_array_equal
from metpy.units import concatenate, units


def test_default_order_warns():
    """Test that using the default array ordering issues a warning."""
    u = np.ones((3, 3)) * units('m/s')
    with pytest.warns(UserWarning):
        vorticity(u, u, 1 * units.meter, 1 * units.meter)


def test_zero_gradient():
    """Test divergence_vorticity when there is no gradient in the field."""
    u = np.ones((3, 3)) * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        c, v = convergence_vorticity(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    truth = np.zeros_like(u) / units.sec
    assert_array_equal(c, truth)
    assert_array_equal(v, truth)


def test_cv_zero_vorticity():
    """Test divergence_vorticity when there is only divergence."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        c, v = convergence_vorticity(u, u.T, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_c = 2. * np.ones_like(u) / units.sec
    true_v = np.zeros_like(u) / units.sec
    assert_array_equal(c, true_c)
    assert_array_equal(v, true_v)


def test_divergence_vorticity():
    """Test of vorticity and divergence calculation for basic case."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        c, v = convergence_vorticity(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_c = np.ones_like(u) / units.sec
    true_v = np.ones_like(u) / units.sec
    assert_array_equal(c, true_c)
    assert_array_equal(v, true_v)


def test_vorticity_divergence_asym():
    """Test vorticity and divergence calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        c, vort = convergence_vorticity(u, v, 1 * units.meters, 2 * units.meters,
                                        dim_order='yx')
    true_c = np.array([[-2, 5.5, -2.5], [2., 0.5, -1.5], [3., -1.5, 8.5]]) / units.sec
    true_vort = np.array([[-2.5, 3.5, 13.], [8.5, -1.5, -11.], [-5.5, -1.5, 0.]]) / units.sec
    assert_array_equal(c, true_c)
    assert_array_equal(vort, true_vort)

    # Now try for xy ordered
    with pytest.warns(MetpyDeprecationWarning):
        c, vort = convergence_vorticity(u.T, v.T, 1 * units.meters, 2 * units.meters,
                                        dim_order='xy')
    assert_array_equal(c, true_c.T)
    assert_array_equal(vort, true_vort.T)


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


def test_shst_zero_gradient():
    """Test shear_stretching_deformation when there is zero gradient."""
    u = np.ones((3, 3)) * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        sh, st = shearing_stretching_deformation(u, u, 1 * units.meter, 1 * units.meter,
                                                 dim_order='xy')
    truth = np.zeros_like(u) / units.sec
    assert_array_equal(sh, truth)
    assert_array_equal(st, truth)


def test_shst_zero_stretching():
    """Test shear_stretching_deformation when there is only shearing."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        sh, st = shearing_stretching_deformation(u, u.T, 1 * units.meter, 1 * units.meter,
                                                 dim_order='yx')
    true_sh = 2. * np.ones_like(u) / units.sec
    true_st = np.zeros_like(u) / units.sec
    assert_array_equal(sh, true_sh)
    assert_array_equal(st, true_st)


def test_shst_deformation():
    """Test of shearing and stretching deformation calculation for basic case."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        sh, st = shearing_stretching_deformation(u, u, 1 * units.meter, 1 * units.meter,
                                                 dim_order='xy')
    true_sh = np.ones_like(u) / units.sec
    true_st = np.ones_like(u) / units.sec
    assert_array_equal(sh, true_st)
    assert_array_equal(st, true_sh)


def test_shst_deformation_asym():
    """Test shearing and stretching deformation calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        sh, st = shearing_stretching_deformation(u, v, 1 * units.meters, 2 * units.meters,
                                                 dim_order='yx')
    true_sh = np.array([[-7.5, -1.5, 1.], [9.5, -0.5, -11.], [1.5, 5.5, 12.]]) / units.sec
    true_st = np.array([[4., 0.5, 12.5], [4., 1.5, -0.5], [1., 5.5, -4.5]]) / units.sec
    assert_array_equal(sh, true_sh)
    assert_array_equal(st, true_st)

    # Now try for yx ordered
    with pytest.warns(MetpyDeprecationWarning):
        sh, st = shearing_stretching_deformation(u.T, v.T, 1 * units.meters, 2 * units.meters,
                                                 dim_order='xy')
    assert_array_equal(sh, true_sh.T)
    assert_array_equal(st, true_st.T)


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
    uag, vag = ageostrophic_wind(z, 1 / units.sec, 100. * units.meter, 100. * units.meter,
                                 u, v, dim_order='xy')
    true = np.array([[0, 0, 0]] * 3) * units('m/s')
    assert_array_equal(uag, true)
    assert_array_equal(vag, true)


def test_ageostrophic_geopotential():
    """Test ageostrophic wind calculation with geopotential and ageostrophic wind."""
    z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
    u = v = np.array([[0, 0, 0]] * 3) * units('m/s')
    uag, vag = ageostrophic_wind(z, 1 / units.sec, 100. * units.meter, 100. * units.meter,
                                 u, v, dim_order='xy')

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


def test_storm_relative_helicity():
    """Test function for SRH calculations on an eigth-circle hodograph."""
    # Create larger arrays for everything except pressure to make a smoother graph
    hgt_int = np.arange(0, 2050, 50)
    hgt_int = hgt_int * units('meter')
    dir_int = np.arange(180, 272.25, 2.25)
    spd_int = np.zeros((hgt_int.shape[0]))
    spd_int[:] = 2.
    u_int, v_int = get_wind_components(spd_int * units('m/s'), dir_int * units.degree)

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


def test_lat_lon_grid_spacing_1d():
    """Test for lat_lon_grid_spacing for variable grid."""
    lat = np.arange(40, 50, 2.5)
    lon = np.arange(-100, -90, 2.5)
    dx, dy = lat_lon_grid_spacing(lon, lat)
    dx_truth = np.array([[212943.5585, 212943.5585, 212943.5585],
                         [204946.2305, 204946.2305, 204946.2305],
                         [196558.8269, 196558.8269, 196558.8269],
                         [187797.3216, 187797.3216, 187797.3216]]) * units.meter
    dy_truth = np.array([[277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857]]) * units.meter
    assert_almost_equal(dx, dx_truth, 4)
    assert_almost_equal(dy, dy_truth, 4)


def test_lat_lon_grid_spacing_2d():
    """Test for lat_lon_grid_spacing for variable grid."""
    lat = np.arange(40, 50, 2.5)
    lon = np.arange(-100, -90, 2.5)
    lon, lat = np.meshgrid(lon, lat)
    dx, dy = lat_lon_grid_spacing(lon, lat)
    dx_truth = np.array([[212943.5585, 212943.5585, 212943.5585],
                         [204946.2305, 204946.2305, 204946.2305],
                         [196558.8269, 196558.8269, 196558.8269],
                         [187797.3216, 187797.3216, 187797.3216]]) * units.meter
    dy_truth = np.array([[277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857]]) * units.meter
    assert_almost_equal(dx, dx_truth, 4)
    assert_almost_equal(dy, dy_truth, 4)


def test_lat_lon_grid_spacing_mismatched_shape():
    """Test for lat_lon_grid_spacing for variable grid."""
    lat = np.arange(40, 50, 2.5)
    lon = np.array([[-100., -97.5, -95., -92.5],
                    [-100., -97.5, -95., -92.5],
                    [-100., -97.5, -95., -92.5],
                    [-100., -97.5, -95., -92.5]])
    with pytest.raises(ValueError):
        dx, dy = lat_lon_grid_spacing(lon, lat)


def test_v_vorticity():
    """Test that v_vorticity wrapper works (deprecated in 0.7)."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        v = v_vorticity(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_v = np.ones_like(u) / units.sec
    assert_array_equal(v, true_v)


def test_convergence():
    """Test that convergence wrapper works (deprecated in 0.7)."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        c = h_convergence(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_c = np.ones_like(u) / units.sec
    assert_array_equal(c, true_c)


def test_convergence_vorticity():
    """Test that convergence_vorticity wrapper works (deprecated in 0.7)."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    with pytest.warns(MetpyDeprecationWarning):
        c, v = convergence_vorticity(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_c = np.ones_like(u) / units.sec
    true_v = np.ones_like(u) / units.sec
    assert_array_equal(c, true_c)
    assert_array_equal(v, true_v)


def test_lat_lon_grid_deltas_1d():
    """Test for lat_lon_grid_spacing for variable grid."""
    lat = np.arange(40, 50, 2.5)
    lon = np.arange(-100, -90, 2.5)
    dx, dy = lat_lon_grid_deltas(lon, lat)
    dx_truth = np.array([[212943.5585, 212943.5585, 212943.5585],
                         [204946.2305, 204946.2305, 204946.2305],
                         [196558.8269, 196558.8269, 196558.8269],
                         [187797.3216, 187797.3216, 187797.3216]]) * units.meter
    dy_truth = np.array([[277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857]]) * units.meter
    assert_almost_equal(dx, dx_truth, 4)
    assert_almost_equal(dy, dy_truth, 4)


@pytest.mark.parametrize('flip_order', [(False, True)])
def test_lat_lon_grid_deltas_2d(flip_order):
    """Test for lat_lon_grid_spacing for variable grid with negative delta distances."""
    lat = np.arange(40, 50, 2.5)
    lon = np.arange(-100, -90, 2.5)
    dx_truth = np.array([[212943.5585, 212943.5585, 212943.5585],
                         [204946.2305, 204946.2305, 204946.2305],
                         [196558.8269, 196558.8269, 196558.8269],
                         [187797.3216, 187797.3216, 187797.3216]]) * units.meter
    dy_truth = np.array([[277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857]]) * units.meter
    if flip_order:
        lon = lon[::-1]
        lat = lat[::-1]
        dx_truth = -1 * dx_truth[::-1]
        dy_truth = -1 * dy_truth[::-1]

    lon, lat = np.meshgrid(lon, lat)
    dx, dy = lat_lon_grid_deltas(lon, lat)
    assert_almost_equal(dx, dx_truth, 4)
    assert_almost_equal(dy, dy_truth, 4)


def test_lat_lon_grid_deltas_mismatched_shape():
    """Test for lat_lon_grid_spacing for variable grid."""
    lat = np.arange(40, 50, 2.5)
    lon = np.array([[-100., -97.5, -95., -92.5],
                    [-100., -97.5, -95., -92.5],
                    [-100., -97.5, -95., -92.5],
                    [-100., -97.5, -95., -92.5]])
    with pytest.raises(ValueError):
        lat_lon_grid_deltas(lon, lat)


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
    u = np.array([[100, 90, 80, 70],
                  [90, 80, 70, 60],
                  [80, 70, 60, 50],
                  [70, 60, 50, 40]]) * units('m/s')

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

    pvor = potential_vorticity_baroclinic(potential_temperature, pressure, u, v, dx, dy, lats)

    abs_vorticity = absolute_vorticity(u, v, dx, dy, lats)

    vort_difference = pvor - (abs_vorticity * g * (-1 * (units.kelvin / units.hPa)))

    true_vort = np.zeros_like(u) * (units.kelvin * units.meter**2 /
                                    (units.second * units.kilogram))

    assert_almost_equal(vort_difference, true_vort, 10)

    # Now try for xy ordered
    pvor = potential_vorticity_baroclinic(potential_temperature, pressure, u.T, v.T, dx.T,
                                          dy.T, lats.T, dim_order='xy')
    abs_vorticity = absolute_vorticity(u.T, v.T, dx.T, dy.T, lats.T, dim_order='xy')
    vort_difference = pvor - (abs_vorticity * g * (-1 * (units.kelvin / units.hPa)))
    assert_almost_equal(vort_difference, true_vort, 10)


def test_potential_vorticity_baroclinic_unity_axis2(pv_data):
    """Test potential vorticity calculation with unity stability and height on axis 2."""
    u, v, lats, _, dx, dy = pv_data

    potential_temperature = np.ones((4, 4, 3)) * units.kelvin
    potential_temperature[..., 0] = 200 * units.kelvin
    potential_temperature[..., 1] = 300 * units.kelvin
    potential_temperature[..., 2] = 400 * units.kelvin

    pressure = np.ones((4, 4, 3)) * units.hPa
    pressure[..., 2] = 1000 * units.hPa
    pressure[..., 1] = 900 * units.hPa
    pressure[..., 0] = 800 * units.hPa

    pvor = potential_vorticity_baroclinic(potential_temperature, pressure, u, v,
                                          dx, dy, lats, axis=2)

    abs_vorticity = absolute_vorticity(u, v, dx, dy, lats)

    vort_difference = pvor - (abs_vorticity * g * (-1 * (units.kelvin / units.hPa)))

    true_vort = np.zeros_like(u) * (units.kelvin * units.meter ** 2 /
                                    (units.second * units.kilogram))

    assert_almost_equal(vort_difference, true_vort, 10)

    # Now try for xy ordered
    pvor = potential_vorticity_baroclinic(potential_temperature, pressure, u.T, v.T, dx.T,
                                          dy.T, lats.T, axis=2, dim_order='xy')
    abs_vorticity = absolute_vorticity(u.T, v.T, dx.T, dy.T, lats.T, dim_order='xy')
    vort_difference = pvor - (abs_vorticity * g * (-1 * (units.kelvin / units.hPa)))
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

    pvor = potential_vorticity_baroclinic(potential_temperature, pressure, u, v, dx, dy, lats)

    abs_vorticity = absolute_vorticity(u, v, dx, dy, lats)

    vort_difference = pvor - (abs_vorticity * g * (-100 * (units.kelvin / units.hPa)))

    true_vort = np.zeros_like(u) * (units.kelvin * units.meter ** 2 /
                                    (units.second * units.kilogram))

    assert_almost_equal(vort_difference, true_vort, 10)

    # Now try for xy ordered
    pvor = potential_vorticity_baroclinic(potential_temperature, pressure, u.T, v.T, dx.T,
                                          dy.T, lats.T, dim_order='xy')
    abs_vorticity = absolute_vorticity(u.T, v.T, dx.T, dy.T, lats.T, dim_order='xy')
    vort_difference = pvor - (abs_vorticity * g * (-100 * (units.kelvin / units.hPa)))
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
                                       dx, dy, lats)

    with pytest.raises(ValueError):
        potential_vorticity_baroclinic(u, v, dx, dy, lats,
                                       potential_temperature, pressure[:1, :, :])


def test_potential_vorticity_baroclinic_wrong_number_of_levels_axis_2(pv_data):
    """Test that potential vorticity calculation errors without 3 levels on axis 2."""
    u, v, lats, _, dx, dy = pv_data

    potential_temperature = np.ones((4, 4, 3)) * units.kelvin
    potential_temperature[..., 0] = 200 * units.kelvin
    potential_temperature[..., 1] = 300 * units.kelvin
    potential_temperature[..., 1] = 400 * units.kelvin

    pressure = np.ones((4, 4, 3)) * units.hPa
    pressure[..., 2] = 1000 * units.hPa
    pressure[..., 1] = 900 * units.hPa
    pressure[..., 0] = 800 * units.hPa

    with pytest.raises(ValueError):
        potential_vorticity_baroclinic(potential_temperature[..., :1], pressure, u, v,
                                       dx, dy, lats, axis=2)

    with pytest.raises(ValueError):
        potential_vorticity_baroclinic(potential_temperature, pressure[..., :1], u, v,
                                       dx, dy, lats, axis=1)


def test_potential_vorticity_barotropic(pv_data):
    """Test the barotopic (Rossby) potential vorticity."""
    u, v, lats, _, dx, dy = pv_data

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
    u, v = get_wind_components(speed, wdir)

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
                          [-5.6160772e-14, -3.5145951e-13, -3.5145951e-13, -5.6160772e-14]]) *
                units('m^2 kg^-1 s^-1'))
    q2_truth = (np.array([[-4.4976059e-14, -4.3582378e-13, 4.3582378e-13, 4.4976059e-14],
                          [-3.0124244e-13, -3.5724617e-14, 3.5724617e-14, 3.0124244e-13],
                          [3.1216232e-13, 3.6662900e-14, -3.6662900e-14, -3.1216232e-13],
                          [8.6038280e-14, 4.6968342e-13, -4.6968342e-13, -8.6038280e-14]]) *
                units('m^2 kg^-1 s^-1'))

    assert_almost_equal(q1, q1_truth, 20)
    assert_almost_equal(q2, q2_truth, 20)


def test_q_vector_with_static_stability(q_vector_data):
    """Test the Q-vector function using static stability."""
    u, v, temp, p, dx, dy = q_vector_data

    sigma = static_stability(p[:, np.newaxis, np.newaxis], temp)

    # Treating as 700 hPa data
    q1, q2 = q_vector(u, v, temp[1], p[1], dx, dy, sigma[1])

    q1_truth = (np.array([[-1.4158140e-08, -1.6197987e-07, -1.6875014e-07, -1.6010616e-08],
                          [-9.3971386e-08, -1.1252476e-08, -1.1252476e-08, -9.7617234e-08],
                          [-1.0785670e-07, -1.2403513e-08, -1.2403513e-08, -1.0364793e-07],
                          [-2.9186946e-08, -1.7577703e-07, -1.6937879e-07, -2.6112047e-08]]) *
                units('kg m^-2 s^-3'))
    q2_truth = (np.array([[-2.3194263e-08, -2.3380160e-07, 2.4357380e-07, 2.6229040e-08],
                          [-1.4936626e-07, -1.8050836e-08, 1.8050836e-08, 1.5516129e-07],
                          [1.6903373e-07, 1.9457964e-08, -1.9457964e-08, -1.6243771e-07],
                          [4.4714390e-08, 2.3490489e-07, -2.2635441e-07, -4.0003646e-08]]) *
                units('kg m^-2 s^-3'))

    assert_almost_equal(q1, q1_truth, 14)
    assert_almost_equal(q2, q2_truth, 14)
