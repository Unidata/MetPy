# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from metpy.testing import assert_array_equal
import numpy as np
from metpy.calc.kinematics import *  # noqa
from metpy.constants import g
from metpy.units import units, concatenate


class TestGradients(object):
    def test_basic(self):
        'Basic braindead test of vorticity and divergence calculation'
        u = np.ones((3, 3)) * units('m/s')
        c, v = convergence_vorticity(u, u, 1 * units.meter, 1 * units.meter)
        truth = np.zeros_like(u) / units.sec
        assert_array_equal(c, truth)
        assert_array_equal(v, truth)

    def test_basic2(self):
        'Basic test of vorticity and divergence calculation'
        a = np.arange(3)
        u = np.c_[a, a, a] * units('m/s')
        c, v = convergence_vorticity(u, u.T, 1 * units.meter, 1 * units.meter)
        true_c = 2. * np.ones_like(u) / units.sec
        true_v = np.zeros_like(u) / units.sec
        assert_array_equal(c, true_c)
        assert_array_equal(v, true_v)

    def test_basic3(self):
        'Basic test of vorticity and divergence calculation'
        a = np.arange(3)
        u = np.c_[a, a, a] * units('m/s')
        c, v = convergence_vorticity(u, u, 1 * units.meter, 1 * units.meter)
        true_c = np.ones_like(u) / units.sec
        true_v = np.ones_like(u) / units.sec
        assert_array_equal(c, true_c)
        assert_array_equal(v, true_v)


class TestVort(object):
    def test_basic(self):
        'Simple test of only vorticity'
        a = np.arange(3)
        u = np.c_[a, a, a] * units('m/s')
        v = v_vorticity(u, u.T, 1 * units.meter, 1 * units.meter)
        true_v = np.zeros_like(u) / units.sec
        assert_array_equal(v, true_v)

    def test_basic3(self):
        'Basic test of vorticity and divergence calculation'
        a = np.arange(3)
        u = np.c_[a, a, a] * units('m/s')
        v = v_vorticity(u, u, 1 * units.meter, 1 * units.meter)
        true_v = np.ones_like(u) / units.sec
        assert_array_equal(v, true_v)


class TestConv(object):
    def test_basic(self):
        'Simple test of only vorticity'
        a = np.arange(3)
        u = np.c_[a, a, a] * units('m/s')
        c = h_convergence(u, u.T, 1 * units.meter, 1 * units.meter)
        true_c = 2. * np.ones_like(u) / units.sec
        assert_array_equal(c, true_c)

    def test_basic3(self):
        'Basic test of vorticity and divergence calculation'
        a = np.arange(3)
        u = np.c_[a, a, a] * units('m/s')
        c = h_convergence(u, u, 1 * units.meter, 1 * units.meter)
        true_c = np.ones_like(u) / units.sec
        assert_array_equal(c, true_c)


class TestAdvection(object):
    def test_basic(self):
        'Basic braindead test of advection'
        u = np.ones((3,)) * units('m/s')
        s = np.ones_like(u) * units.kelvin
        a = advection(s, u, (1 * units.meter,))
        truth = np.zeros_like(u) * units('K/sec')
        assert_array_equal(a, truth)

    def test_basic2(self):
        'Basic test of advection'
        u = np.ones((3,)) * units('m/s')
        s = np.array([1, 2, 3]) * units('kg')
        a = advection(s, u, (1 * units.meter,))
        truth = -np.ones_like(u) * units('kg/sec')
        assert_array_equal(a, truth)

    def test_basic3(self):
        'Basic test of advection'
        u = np.array([1, 2, 3]) * units('m/s')
        s = np.array([1, 2, 3]) * units('Pa')
        a = advection(s, u, (1 * units.meter,))
        truth = np.array([-1, -2, -3]) * units('Pa/sec')
        assert_array_equal(a, truth)

    def test_2dbasic(self):
        'Basic 2D braindead test of advection'
        u = np.ones((3, 3)) * units('m/s')
        s = np.ones_like(u) * units.kelvin
        a = advection(s, [u, u], (1 * units.meter, 1 * units.meter))
        truth = np.zeros_like(u) * units('K/sec')
        assert_array_equal(a, truth)

    def test_2dbasic2(self):
        'Basic 2D test of advection'
        u = np.ones((3, 3)) * units('m/s')
        v = 2 * np.ones((3, 3)) * units('m/s')
        s = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * units.kelvin
        a = advection(s, [u, v], (1 * units.meter, 1 * units.meter))
        truth = np.array([[-3, -2, 1], [-4, 0, 4], [-1, 2, 3]]) * units('K/sec')
        assert_array_equal(a, truth)


class TestGeos(object):
    def test_basic(self):
        'Basic test of geostrophic wind calculation'
        z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units.meter
        # Using g as the value for f allows it to cancel out
        ug, vg = geostrophic_wind(z, g.magnitude / units.sec,
                                  100. * units.meter, 100. * units.meter)
        true_u = np.array([[-1, 0, 1]] * 3) * units('m/s')
        true_v = -true_u.T
        assert_array_equal(ug, true_u)
        assert_array_equal(vg, true_v)

    def test_geopotential(self):
        'Test of geostrophic wind calculation with geopotential'
        z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100. * units('m^2/s^2')
        ug, vg = geostrophic_wind(z, 1 / units.sec, 100. * units.meter, 100. * units.meter)
        true_u = np.array([[-1, 0, 1]] * 3) * units('m/s')
        true_v = -true_u.T
        assert_array_equal(ug, true_u)
        assert_array_equal(vg, true_v)

    def test_3d(self):
        'Test of geostrophic wind calculation with 3D array'
        z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100.
        # Using g as the value for f allows it to cancel out
        z3d = np.dstack((z, z)) * units.meter
        ug, vg = geostrophic_wind(z3d, g.magnitude / units.sec,
                                  100. * units.meter, 100. * units.meter)
        true_u = np.array([[-1, 0, 1]] * 3) * units('m/s')
        true_v = -true_u.T

        true_u = concatenate((true_u[..., None], true_u[..., None]), axis=2)
        true_v = concatenate((true_v[..., None], true_v[..., None]), axis=2)
        assert_array_equal(ug, true_u)
        assert_array_equal(vg, true_v)
