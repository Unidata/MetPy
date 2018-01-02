# Copyright (c) 2008,2015,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `kinematics` module."""

import numpy as np
import pytest

from metpy.calc import (advection, h_convergence, convergence_vorticity, divergence,
                        divergence_vorticity, frontogenesis, geostrophic_wind,
                        get_wind_components, lat_lon_grid_spacing, montgomery_streamfunction,
                        shearing_deformation, shearing_stretching_deformation,
                        storm_relative_helicity, stretching_deformation, total_deformation,
                        v_vorticity, vorticity)
from metpy.constants import g, omega, Re
from metpy.testing import assert_almost_equal, assert_array_equal
from metpy.units import concatenate, units


def test_default_order_warns():
    """Test that using the default array ordering issues a warning."""
    u = np.ones((3, 3)) * units('m/s')
    with pytest.warns(FutureWarning):
        divergence_vorticity(u, u, 1 * units.meter, 1 * units.meter)


def test_zero_gradient():
    """Test divergence_vorticity when there is no gradient in the field."""
    u = np.ones((3, 3)) * units('m/s')
    c, v = divergence_vorticity(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    truth = np.zeros_like(u) / units.sec
    assert_array_equal(c, truth)
    assert_array_equal(v, truth)


def test_cv_zero_vorticity():
    """Test divergence_vorticity when there is only divergence."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    c, v = divergence_vorticity(u, u.T, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_c = 2. * np.ones_like(u) / units.sec
    true_v = np.zeros_like(u) / units.sec
    assert_array_equal(c, true_c)
    assert_array_equal(v, true_v)


def test_divergence_vorticity():
    """Test of vorticity and divergence calculation for basic case."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    c, v = divergence_vorticity(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_c = np.ones_like(u) / units.sec
    true_v = np.ones_like(u) / units.sec
    assert_array_equal(c, true_c)
    assert_array_equal(v, true_v)


def test_vorticity_divergence_asym():
    """Test vorticity and divergence calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    c, vort = divergence_vorticity(u, v, 1 * units.meters, 2 * units.meters, dim_order='yx')
    true_c = np.array([[0., 4., 0.], [1., 0.5, -0.5], [2., 0., 5.]]) / units.sec
    true_vort = np.array([[-1., 2., 7.], [3.5, -1.5, -6.], [-2., 0., 1.]]) / units.sec
    assert_array_equal(c, true_c)
    assert_array_equal(vort, true_vort)

    # Now try for xy ordered
    c, vort = divergence_vorticity(u.T, v.T, 1 * units.meters, 2 * units.meters,
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
    true_vort = np.array([[-1., 2., 7.], [3.5, -1.5, -6.], [-2., 0., 1.]]) / units.sec
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
    true_c = np.array([[0., 4., 0.], [1., 0.5, -0.5], [2., 0., 5.]]) / units.sec
    assert_array_equal(c, true_c)

    # Now try for xy ordered
    c = divergence(u.T, v.T, 1 * units.meters, 2 * units.meters, dim_order='xy')
    assert_array_equal(c, true_c.T)


def test_shst_zero_gradient():
    """Test shear_stretching_deformation when there is zero gradient."""
    u = np.ones((3, 3)) * units('m/s')
    sh, st = shearing_stretching_deformation(u, u, 1 * units.meter, 1 * units.meter,
                                             dim_order='xy')
    truth = np.zeros_like(u) / units.sec
    assert_array_equal(sh, truth)
    assert_array_equal(st, truth)


def test_shst_zero_stretching():
    """Test shear_stretching_deformation when there is only shearing."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
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
    sh, st = shearing_stretching_deformation(u, v, 1 * units.meters, 2 * units.meters,
                                             dim_order='yx')
    true_sh = np.array([[-3., 0., 1.], [4.5, -0.5, -6.], [2., 4., 7.]]) / units.sec
    true_st = np.array([[4., 2., 8.], [3., 1.5, 0.5], [2., 4., -1.]]) / units.sec
    assert_array_equal(sh, true_sh)
    assert_array_equal(st, true_st)

    # Now try for yx ordered
    sh, st = shearing_stretching_deformation(u.T, v.T, 1 * units.meters, 2 * units.meters,
                                             dim_order='xy')
    assert_array_equal(sh, true_sh.T)
    assert_array_equal(st, true_st.T)


def test_shearing_deformation_asym():
    """Test shearing deformation calculation with a complicated field."""
    u = np.array([[2, 4, 8], [0, 2, 2], [4, 6, 8]]) * units('m/s')
    v = np.array([[6, 4, 8], [2, 6, 0], [2, 2, 6]]) * units('m/s')
    sh = shearing_deformation(u, v, 1 * units.meters, 2 * units.meters, dim_order='yx')
    true_sh = np.array([[-3., 0., 1.], [4.5, -0.5, -6.], [2., 4., 7.]]) / units.sec
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
    true_st = np.array([[4., 2., 8.], [3., 1.5, 0.5], [2., 4., -1.]]) / units.sec
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
    true_tdef = np.array([[5., 2., 8.06225775], [5.40832691, 1.58113883, 6.02079729],
                          [2.82842712, 5.65685425, 7.07106781]]) / units.sec
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
    true_fronto = np.array([[-20.93890452, -7.83070042, -36.43293256],
                            [0.89442719, -2.12218672, -8.94427191],
                            [-16.8, -7.65600391, -61.65921479]]
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
    truth = np.array([[-3, -2, 1], [-4, 0, 4], [-1, 2, 3]]) * units('K/sec')
    assert_array_equal(a, truth)


def test_advection_2d_asym():
    """Test advection in asymmetric varying 2D field."""
    u = np.arange(9).reshape(3, 3) * units('m/s')
    v = 2 * u
    s = np.array([[1, 2, 4], [4, 8, 4], [8, 6, 4]]) * units.kelvin
    a = advection(s, [u, v], (2 * units.meter, 1 * units.meter), dim_order='yx')
    truth = np.array([[0, -12.75, -2], [-27., -16., 10.], [-42, 35, 8]]) * units('K/sec')
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
    true_u = np.array([[-1, 0, 1]] * 3) * units('m/s')
    true_v = -true_u.T
    assert_array_equal(ug, true_u)
    assert_array_equal(vg, true_v)


def test_geostrophic_wind_asym():
    """Test geostrophic wind calculation with a complicated field."""
    z = np.array([[1, 2, 4], [4, 8, 4], [8, 6, 4]]) * 200. * units.meter
    # Using g as the value for f allows it to cancel out
    ug, vg = geostrophic_wind(z, g.magnitude / units.sec,
                              200. * units.meter, 100. * units.meter, dim_order='yx')
    true_u = -np.array([[6, 12, 0], [7, 4, 0], [8, -4, 0]]) * units('m/s')
    true_v = np.array([[1, 1.5, 2], [4, 0, -4], [-2, -2, -2]]) * units('m/s')
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
    true_u = np.array([[-1, 0, 1]] * 3) * units('m/s')
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
    true_u = np.array([[-1, 0, 1]] * 3) * units('m/s')
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
    p_srh, n_srh, T_srh = storm_relative_helicity(u_int, v_int,
                                                  hgt_int, 1000 * units('meter'),
                                                  bottom=0 * units('meter'),
                                                  storm_u=0 * units.knot,
                                                  storm_v=0 * units.knot)
    assert_almost_equal(p_srh, srh_true_p, 2)
    assert_almost_equal(n_srh, srh_true_n, 2)
    assert_almost_equal(T_srh, srh_true_t, 2)


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
    v = v_vorticity(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_v = np.ones_like(u) / units.sec
    assert_array_equal(v, true_v)


def test_convergence():
    """Test that convergence wrapper works (deprecated in 0.7)."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    c = h_convergence(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_c = np.ones_like(u) / units.sec
    assert_array_equal(c, true_c)


def test_convergence_vorticity():
    """Test that convergence_vorticity wrapper works (deprecated in 0.7)."""
    a = np.arange(3)
    u = np.c_[a, a, a] * units('m/s')
    c, v = convergence_vorticity(u, u, 1 * units.meter, 1 * units.meter, dim_order='xy')
    true_c = np.ones_like(u) / units.sec
    true_v = np.ones_like(u) / units.sec
    assert_array_equal(c, true_c)
    assert_array_equal(v, true_v)
