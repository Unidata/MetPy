import numpy as np
from numpy.testing import assert_array_equal
from metpy.calc.turbulence import *  # noqa


class TestTurbulenceKineticEnergy(object):
    def get_uvw_and_known_tke(self):
        u = np.array([-2, -1, 0, 1, 2])
        v = -u
        w = 2 * u
        #  0.5 * sqrt(2 + 2 + 8)
        e_true = np.sqrt(12) / 2.
        return u, v, w, e_true

    def test_no_tke_1d(self):
        observations = 5
        # given all the values are the same, there should not be any tke
        u = np.ones(observations)
        v = np.ones(observations)
        w = np.ones(observations)
        e_zero = 0
        assert_array_equal(e_zero, tke(u, v, w))

    def test_no_tke_2d_axis_last(self):
        observations = 5
        instruments = 2
        # given all the values are the same, there should not be any tke
        u = np.ones((instruments, observations))
        v = np.ones((instruments, observations))
        w = np.ones((instruments, observations))
        e_zero = np.zeros(instruments)
        assert_array_equal(e_zero, tke(u, v, w, axis=-1))

    def test_no_tke_2d_axis_first(self):
        observations = 5
        instruments = 2
        # given all the values are the same, there should not be any tke
        u = np.ones((observations, instruments))
        v = np.ones((observations, instruments))
        w = np.ones((observations, instruments))
        e_zero = np.zeros(instruments)
        assert_array_equal(e_zero, tke(u, v, w, axis=0))

    def test_known_tke(self):
        u, v, w, e_true = self.get_uvw_and_known_tke()
        assert_array_equal(e_true, tke(u, v, w))

    def test_known_tke_2d_axis_last(self):
        '''test array with shape (3, 5) [pretend time axis is -1]'''
        u, v, w, e_true = self.get_uvw_and_known_tke()
        u = np.array([u, u, u])
        v = np.array([v, v, v])
        w = np.array([w, w, w])
        e_true = e_true * np.ones(3)
        assert_array_equal(e_true, tke(u, v, w, axis=-1))

    def test_known_tke_2d_axis_first(self):
        '''test array with shape (5, 3) [pretend time axis is 0]'''
        u, v, w, e_true = self.get_uvw_and_known_tke()
        u = np.array([u, u, u]).transpose()
        v = np.array([v, v, v]).transpose()
        w = np.array([w, w, w]).transpose()
        e_true = e_true * np.ones(3).transpose()
        assert_array_equal(e_true, tke(u, v, w, axis=0))
        assert_array_equal(e_true, tke(u, v, w, axis=0, perturbation=True))


class TestGetPerturbation(object):
    def get_pert_from_zero_mean(self):
        ts = np.array([-2, -1, 0, 1, 2])
        #  ts.meam() = 0
        pert_true = ts.copy()
        return ts, pert_true

    def get_pert_from_non_zero_mean(self):
        ts = np.array([-2, 0, 2, 4, 6])
        # ts.mean() = 2
        pert_true = np.array([-4, -2, 0, 2, 4])
        return ts, pert_true

    def test_no_perturbation_1d(self):
        observations = 5
        # given all the values are the same, there should not be perturbations
        ts = np.ones(observations)
        pert_zero = 0
        assert_array_equal(pert_zero, get_perturbation(ts))

    def test_no_perturbation_2d_axis_last(self):
        observations = 5
        instruments = 2
        # given all the values are the same, there should not be perturbations
        ts = np.ones((instruments, observations))
        pert_zero = np.zeros((instruments, observations))
        assert_array_equal(pert_zero, get_perturbation(ts, axis=-1))

    def test_no_tke_2d_axis_first(self):
        observations = 5
        instruments = 2
        # given all the values are the same, there should not be perturbations
        ts = np.ones((observations, instruments))
        pert_zero = np.zeros((observations, instruments))
        assert_array_equal(pert_zero, get_perturbation(ts, axis=0))

    def test_known_perturbation_zero_mean_1d(self):
        ts, pert_known = self.get_pert_from_zero_mean()
        assert_array_equal(pert_known, get_perturbation(ts))

    def test_known_perturbation_zero_mean_2d_axis_last(self):
        ts, pert_known = self.get_pert_from_zero_mean()
        ts = np.array([ts, ts, ts])
        pert_known = np.array([pert_known, pert_known, pert_known])
        assert_array_equal(pert_known, get_perturbation(ts, axis=-1))

    def test_known_perturbation_zero_mean_2d_axis_first(self):
        ts, pert_known = self.get_pert_from_zero_mean()
        ts = np.array([ts, ts, ts]).transpose()
        pert_known = np.array([pert_known, pert_known, pert_known]).transpose()
        assert_array_equal(pert_known, get_perturbation(ts, axis=0))

    def test_known_perturbation_non_zero_mean_1d(self):
        ts, pert_known = self.get_pert_from_non_zero_mean()
        assert_array_equal(pert_known, get_perturbation(ts))

    def test_known_perturbation_non_zero_mean_2d_axis_last(self):
        ts, pert_known = self.get_pert_from_non_zero_mean()
        ts = np.array([ts, ts, ts])
        pert_known = np.array([pert_known, pert_known, pert_known])
        assert_array_equal(pert_known, get_perturbation(ts, axis=-1))

    def test_known_perturbation_non_zero_mean_2d_axis_first(self):
        ts, pert_known = self.get_pert_from_non_zero_mean()
        ts = np.array([ts, ts, ts]).transpose()
        pert_known = np.array([pert_known, pert_known, pert_known]).transpose()
        assert_array_equal(pert_known, get_perturbation(ts, axis=0))
