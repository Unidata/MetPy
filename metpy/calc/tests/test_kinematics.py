from numpy.testing import assert_array_equal
import numpy as np
from metpy.calc.kinematics import *  # noqa
from metpy.constants import g


class TestGradients(object):
    def test_basic(self):
        'Basic braindead test of vorticity and divergence calculation'
        u = np.ones((3, 3))
        c, v = convergence_vorticity(u, u, 1, 1)
        truth = np.zeros_like(u)
        assert_array_equal(c, truth)
        assert_array_equal(v, truth)

    def test_basic2(self):
        'Basic test of vorticity and divergence calculation'
        a = np.arange(3)
        u = np.c_[a, a, a]
        c, v = convergence_vorticity(u, u.T, 1, 1)
        true_c = 2. * np.ones_like(u)
        true_v = np.zeros_like(u)
        assert_array_equal(c, true_c)
        assert_array_equal(v, true_v)

    def test_basic3(self):
        'Basic test of vorticity and divergence calculation'
        a = np.arange(3)
        u = np.c_[a, a, a]
        c, v = convergence_vorticity(u, u, 1, 1)
        true_c = np.ones_like(u)
        true_v = np.ones_like(u)
        assert_array_equal(c, true_c)
        assert_array_equal(v, true_v)


class TestVort(object):
    def test_basic(self):
        'Simple test of only vorticity'
        a = np.arange(3)
        u = np.c_[a, a, a]
        v = v_vorticity(u, u.T, 1, 1)
        true_v = np.zeros_like(u)
        assert_array_equal(v, true_v)

    def test_basic3(self):
        'Basic test of vorticity and divergence calculation'
        a = np.arange(3)
        u = np.c_[a, a, a]
        v = v_vorticity(u, u, 1, 1)
        true_v = np.ones_like(u)
        assert_array_equal(v, true_v)


class TestConv(object):
    def test_basic(self):
        'Simple test of only vorticity'
        a = np.arange(3)
        u = np.c_[a, a, a]
        c = h_convergence(u, u.T, 1, 1)
        true_c = 2. * np.ones_like(u)
        assert_array_equal(c, true_c)

    def test_basic3(self):
        'Basic test of vorticity and divergence calculation'
        a = np.arange(3)
        u = np.c_[a, a, a]
        c = h_convergence(u, u, 1, 1)
        true_c = np.ones_like(u)
        assert_array_equal(c, true_c)


class TestAdvection(object):
    def test_basic(self):
        'Basic braindead test of advection'
        u = np.ones((3,))
        s = np.ones_like(u)
        a = advection(s, u, (1,))
        truth = np.zeros_like(u)
        assert_array_equal(a, truth)

    def test_basic2(self):
        'Basic test of advection'
        u = np.ones((3,))
        s = np.array([1, 2, 3])
        a = advection(s, u, (1,))
        truth = -np.ones_like(u)
        assert_array_equal(a, truth)

    def test_basic3(self):
        'Basic test of advection'
        u = np.array([1, 2, 3])
        s = np.array([1, 2, 3])
        a = advection(s, u, (1,))
        truth = np.array([-1, -2, -3])
        assert_array_equal(a, truth)

    def test_2dbasic(self):
        'Basic 2D braindead test of advection'
        u = np.ones((3, 3))
        s = np.ones_like(u)
        a = advection(s, u, (1,))
        truth = np.zeros_like(u)
        assert_array_equal(a, truth)

    def test_2dbasic2(self):
        'Basic 2D test of advection'
        u = np.ones((3, 3))
        v = 2 * np.ones((3, 3))
        s = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        a = advection(s, [u, v], (1, 1))
        truth = np.array([[-3, -2, 1], [-4, 0, 4], [-1, 2, 3]])
        assert_array_equal(a, truth)


class TestGeos(object):
    def test_basic(self):
        'Basic test of geostrophic wind calculation'
        z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100.
        # Using g as the value for f allows it to cancel out
        ug, vg = geostrophic_wind(z, g, 100., 100.)
        true_u = np.array([[-1, 0, 1]] * 3)
        true_v = -true_u.T
        assert_array_equal(ug, true_u)
        assert_array_equal(vg, true_v)

    def test_geopotential(self):
        'Test of geostrophic wind calculation with geopotential'
        z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100.
        ug, vg = geostrophic_wind(z, 1, 100., 100., geopotential=True)
        true_u = np.array([[-1, 0, 1]] * 3)
        true_v = -true_u.T
        assert_array_equal(ug, true_u)
        assert_array_equal(vg, true_v)

    def test_3d(self):
        'Test of geostrophic wind calculation with 3D array'
        z = np.array([[48, 49, 48], [49, 50, 49], [48, 49, 48]]) * 100.
        # Using g as the value for f allows it to cancel out
        z3d = np.dstack((z, z))
        ug, vg = geostrophic_wind(z3d, g, 100., 100.)
        true_u = np.array([[-1, 0, 1]] * 3)
        true_v = -true_u.T

        true_u = np.dstack((true_u, true_u))
        true_v = np.dstack((true_v, true_v))
        assert_array_equal(ug, true_u)
        assert_array_equal(vg, true_v)
