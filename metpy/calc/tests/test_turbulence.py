# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
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


class TestKinematicFlux(object):
    def get_uvw_and_known_kf_zero_mean(self):
        u = np.array([-2, -1, 0, 1, 2])
        v = -u
        w = 2 * u
        kf_true = {'uv': -2, 'uw': 4, 'vw': -4}
        return u, v, w, kf_true

    def get_uvw_and_known_kf_non_zero_mean(self):
        u = np.array([-2, -1, 0, 1, 5])
        v = -u
        w = 2 * u
        kf_true = {'uv': -5.84, 'uw': 11.68, 'vw': -11.68}
        return u, v, w, kf_true

    def test_kf_1d(self):
        u, v, w, kf_true = self.get_uvw_and_known_kf_zero_mean()
        assert_array_equal(kinematic_flux(u, v, perturbation=False),
                           kf_true['uv'])
        assert_array_equal(kinematic_flux(u, w, perturbation=False),
                           kf_true['uw'])
        assert_array_equal(kinematic_flux(v, w, perturbation=False),
                           kf_true['vw'])
        # given u, v, and w have a zero mean, the kf computed with
        # perturbation=True and perturbation=False should be the same
        assert_array_equal(kinematic_flux(u, v, perturbation=False),
                           kinematic_flux(u, v, perturbation=True))
        assert_array_equal(kinematic_flux(u, w, perturbation=False),
                           kinematic_flux(u, w, perturbation=True))
        assert_array_equal(kinematic_flux(v, w, perturbation=False),
                           kinematic_flux(v, w, perturbation=True))
        # now use a non-zero mean
        u, v, w, kf_true = self.get_uvw_and_known_kf_non_zero_mean()
        assert_array_equal(kinematic_flux(u, v, perturbation=False),
                           kf_true['uv'])
        assert_array_equal(kinematic_flux(u, w, perturbation=False),
                           kf_true['uw'])
        assert_array_equal(kinematic_flux(v, w, perturbation=False),
                           kf_true['vw'])

    def test_kf_2d_axis_last(self):
        u, v, w, kf_true = self.get_uvw_and_known_kf_zero_mean()
        u = np.array([u, u, u])
        v = np.array([v, v, v])
        w = np.array([w, w, w])
        for key in kf_true.keys():
            tmp = kf_true[key]
            kf_true[key] = np.array([tmp, tmp, tmp])
        assert_array_equal(kinematic_flux(u, v, perturbation=False, axis=-1),
                           kf_true['uv'])
        assert_array_equal(kinematic_flux(u, w, perturbation=False, axis=-1),
                           kf_true['uw'])
        assert_array_equal(kinematic_flux(v, w, perturbation=False, axis=-1),
                           kf_true['vw'])
        # given u, v, and w have a zero mean, the kf computed with
        # perturbation=True and perturbation=False should be the same
        assert_array_equal(kinematic_flux(u, v, perturbation=False, axis=-1),
                           kinematic_flux(u, v, perturbation=True, axis=-1))
        assert_array_equal(kinematic_flux(u, w, perturbation=False, axis=-1),
                           kinematic_flux(u, w, perturbation=True, axis=-1))
        assert_array_equal(kinematic_flux(v, w, perturbation=False, axis=-1),
                           kinematic_flux(v, w, perturbation=True, axis=-1))
        # now use a non-zero mean
        u, v, w, kf_true = self.get_uvw_and_known_kf_non_zero_mean()
        u = np.array([u, u, u])
        v = np.array([v, v, v])
        w = np.array([w, w, w])
        for key in kf_true.keys():
            tmp = kf_true[key]
            kf_true[key] = np.array([tmp, tmp, tmp])
        assert_array_equal(kinematic_flux(u, v, perturbation=False, axis=-1),
                           kf_true['uv'])
        assert_array_equal(kinematic_flux(u, w, perturbation=False, axis=-1),
                           kf_true['uw'])
        assert_array_equal(kinematic_flux(v, w, perturbation=False, axis=-1),
                           kf_true['vw'])

    def test_kf_2d_axis_first(self):
        u, v, w, kf_true = self.get_uvw_and_known_kf_zero_mean()
        u = np.array([u, u, u]).transpose()
        v = np.array([v, v, v]).transpose()
        w = np.array([w, w, w]).transpose()
        for key in kf_true.keys():
            tmp = kf_true[key]
            kf_true[key] = np.array([tmp, tmp, tmp]).transpose()
        assert_array_equal(kinematic_flux(u, v, perturbation=False, axis=0),
                           kf_true['uv'])
        assert_array_equal(kinematic_flux(u, w, perturbation=False, axis=0),
                           kf_true['uw'])
        assert_array_equal(kinematic_flux(v, w, perturbation=False, axis=0),
                           kf_true['vw'])
        # given u, v, and w have a zero mean, the kf computed with
        # perturbation=True and perturbation=False should be the same
        assert_array_equal(kinematic_flux(u, v, perturbation=False, axis=0),
                           kinematic_flux(u, v, perturbation=True, axis=0))
        assert_array_equal(kinematic_flux(u, w, perturbation=False, axis=0),
                           kinematic_flux(u, w, perturbation=True, axis=0))
        assert_array_equal(kinematic_flux(v, w, perturbation=False, axis=0),
                           kinematic_flux(v, w, perturbation=True, axis=0))
        # non use a non-zero mean
        u, v, w, kf_true = self.get_uvw_and_known_kf_non_zero_mean()
        u = np.array([u, u, u]).transpose()
        v = np.array([v, v, v]).transpose()
        w = np.array([w, w, w]).transpose()
        for key in kf_true.keys():
            tmp = kf_true[key]
            kf_true[key] = np.array([tmp, tmp, tmp]).transpose()
        assert_array_equal(kinematic_flux(u, v, perturbation=False, axis=0),
                           kf_true['uv'])
        assert_array_equal(kinematic_flux(u, w, perturbation=False, axis=0),
                           kf_true['uw'])
        assert_array_equal(kinematic_flux(v, w, perturbation=False, axis=0),
                           kf_true['vw'])


class TestFrictionVelocity(object):
    def get_uvw_and_known_u_star_zero_mean(self):
        u = np.array([-2, -1, 0, 1, 2])
        v = -u
        w = 2 * u
        u_star_true = {'uw': 2.0, 'uwvw': 2.3784142300054421}
        return u, v, w, u_star_true

    def get_uvw_and_known_u_star_non_zero_mean(self):
        u = np.array([-2, -1, 0, 1, 5])
        v = -u
        w = 2 * u
        u_star_true = {'uw': 3.4176014981270124, 'uwvw': 4.0642360178166017}
        return u, v, w, u_star_true

    def test_u_star_1d(self):
        u, v, w, u_star_true = self.get_uvw_and_known_u_star_zero_mean()
        assert_almost_equal(friction_velocity(u, w, perturbation=False),
                            u_star_true['uw'])
        assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False),
                            u_star_true['uwvw'])
        # now use a non-zero mean
        u, v, w, u_star_true = self.get_uvw_and_known_u_star_non_zero_mean()
        assert_almost_equal(friction_velocity(u, w, perturbation=False),
                            u_star_true['uw'])
        assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False),
                            u_star_true['uwvw'])

    def test_u_star_2d_axis_last(self):
        u, v, w, u_star_true = self.get_uvw_and_known_u_star_zero_mean()
        u = np.array([u, u, u])
        v = np.array([v, v, v])
        w = np.array([w, w, w])
        for key in u_star_true.keys():
            tmp = u_star_true[key]
            u_star_true[key] = np.array([tmp, tmp, tmp])
        assert_almost_equal(friction_velocity(u, w, perturbation=False,
                            axis=-1), u_star_true['uw'])
        assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False,
                            axis=-1), u_star_true['uwvw'])
        # now use a non-zero mean
        u, v, w, u_star_true = self.get_uvw_and_known_u_star_non_zero_mean()
        u = np.array([u, u, u])
        v = np.array([v, v, v])
        w = np.array([w, w, w])
        for key in u_star_true.keys():
            tmp = u_star_true[key]
            u_star_true[key] = np.array([tmp, tmp, tmp])
        assert_almost_equal(friction_velocity(u, w, perturbation=False,
                            axis=-1), u_star_true['uw'])
        assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False,
                            axis=-1), u_star_true['uwvw'])

    def test_u_star_2d_axis_first(self):
        u, v, w, u_star_true = self.get_uvw_and_known_u_star_zero_mean()
        u = np.array([u, u, u]).transpose()
        v = np.array([v, v, v]).transpose()
        w = np.array([w, w, w]).transpose()
        for key in u_star_true.keys():
            tmp = u_star_true[key]
            u_star_true[key] = np.array([tmp, tmp, tmp]).transpose()
        assert_almost_equal(friction_velocity(u, w, perturbation=False,
                            axis=0), u_star_true['uw'])
        assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False,
                            axis=0), u_star_true['uwvw'])
        # now use a non-zero mean
        u, v, w, u_star_true = self.get_uvw_and_known_u_star_non_zero_mean()
        u = np.array([u, u, u]).transpose()
        v = np.array([v, v, v]).transpose()
        w = np.array([w, w, w]).transpose()
        for key in u_star_true.keys():
            tmp = u_star_true[key]
            u_star_true[key] = np.array([tmp, tmp, tmp]).transpose()
        assert_almost_equal(friction_velocity(u, w, perturbation=False,
                            axis=0), u_star_true['uw'])
        assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False,
                            axis=0), u_star_true['uwvw'])
