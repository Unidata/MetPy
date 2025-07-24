# Copyright (c) 2008,2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `turbulence` module."""

import numpy as np
import pytest
import xarray as xr

from metpy.calc.turbulence import friction_velocity, get_perturbation, kinematic_flux, tke
from metpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from metpy.units import units


#
# TKE Tests
#
@pytest.fixture()
def uvw_and_known_tke():
    """Provide a set of u,v,w with a known tke value."""
    u = np.array([-7, -1, 0, 1, 2])
    v = -u
    w = 2 * u
    # average u: -1
    # average v: 1
    # average w: -2
    # average(u'^2): 10
    # average(v'^2): 10
    # average(w'^2): 40
    #  0.5 * (10 + 10 + 40) = 30
    e_true = 30
    return u, v, w, e_true


@pytest.fixture()
def uvw_and_known_tke_xarray():
    """Provide a set of u, v, w with a known tke value as an xarray."""
    # Define coordinate values
    pressure = [1000, 850]  # hPa
    lat = [10, 20]          # degrees North
    lon = [30, 40]          # degrees East
    time = np.array(['2025-01-01T00:00', '2025-01-01T06:00'], dtype='datetime64')

    # Define dimensions
    dims = ('pressure', 'lat', 'lon', 'time')

    # Generate 16 linearly spaced values between -30 and 30
    uwind_values = np.linspace(0, 30, num=16).reshape(2, 2, 2, 2)
    vwind_values = np.linspace(-30, 0, num=16).reshape(2, 2, 2, 2)
    wwind_values = np.linspace(-1, 2, num=16).reshape(2, 2, 2, 2)

    # Apply units
    u = uwind_values * units('m/s')
    v = vwind_values * units('m/s')
    w = wwind_values * units('m/s')

    # Create the Dataset
    ds = xr.Dataset(
        {
            'uwind': (dims, u),
            'vwind': (dims, v),
            'wwind': (dims, w),
        },
        coords={
            'pressure': pressure,
            'lat': lat,
            'lon': lon,
            'time': time
        }
    )

    e_true = np.full((2, 2, 2), 1.005) * units('m^2/s^2')
    return ds, e_true


def test_no_tke_1d():
    """Test tke calculation where the expected value is 0."""
    observations = 5
    # given all the values are the same, there should not be any tke
    u = np.ones(observations)
    v = np.ones(observations)
    w = np.ones(observations)
    e_zero = 0
    assert_array_equal(e_zero, tke(u, v, w))


def test_no_tke_2d_axis_last():
    """Test 0 tke calculation with 2D arrays; calculation axis is last."""
    observations = 5
    instruments = 2
    # given all the values are the same, there should not be any tke
    u = np.ones((instruments, observations))
    v = np.ones((instruments, observations))
    w = np.ones((instruments, observations))
    e_zero = np.zeros(instruments)
    assert_array_equal(e_zero, tke(u, v, w, axis=-1))


def test_no_tke_2d_axis_first():
    """Test 0 tke calculation with 2D arrays; calculation axis is first."""
    observations = 5
    instruments = 2
    # given all the values are the same, there should not be any tke
    u = np.ones((observations, instruments))
    v = np.ones((observations, instruments))
    w = np.ones((observations, instruments))
    e_zero = np.zeros(instruments)
    assert_array_equal(e_zero, tke(u, v, w, axis=0))


def test_known_tke(uvw_and_known_tke):
    """Test basic behavior of tke with known values."""
    u, v, w, e_true = uvw_and_known_tke
    assert_array_equal(e_true, tke(u, v, w))


def test_known_tke_xarray(uvw_and_known_tke_xarray):
    """Test basic behavior of tke with known xarray values."""
    data, e_true = uvw_and_known_tke_xarray
    assert_array_almost_equal(tke(data.uwind, data.vwind, data.wwind), e_true)


def test_known_tke_using_perturbation_velocities(uvw_and_known_tke):
    """Test basic behavior of tke with known values."""
    u, v, w, e_true = uvw_and_known_tke
    assert_array_equal(e_true, tke(u - u.mean(), v - v.mean(), w - w.mean(),
                                   perturbation=True))


def test_known_tke_2d_axis_last(uvw_and_known_tke):
    """Test array with shape (3, 5) [pretend time axis is -1]."""
    u, v, w, e_true = uvw_and_known_tke
    u = np.array([u, u, u])
    v = np.array([v, v, v])
    w = np.array([w, w, w])
    e_true = e_true * np.ones(3)
    assert_array_equal(e_true, tke(u, v, w, axis=-1))


def test_known_tke_2d_axis_first(uvw_and_known_tke):
    """Test array with shape (5, 3) [pretend time axis is 0]."""
    u, v, w, e_true = uvw_and_known_tke
    u = np.array([u, u, u]).transpose()
    v = np.array([v, v, v]).transpose()
    w = np.array([w, w, w]).transpose()
    e_true = e_true * np.ones(3).transpose()
    assert_array_equal(e_true, tke(u, v, w, axis=0))


#
# Perturbation tests
#
@pytest.fixture()
def pert_zero_mean():
    """Return time series with zero-mean and perturbations."""
    ts = np.array([-2, -1, 0, 1, 2])
    pert_true = ts.copy()
    return ts, pert_true


@pytest.fixture()
def pert_nonzero_mean():
    """Return time seres with non-zero-mean and perturbations."""
    ts = np.array([-2, 0, 2, 4, 6])
    # ts.mean() = 2
    pert_true = np.array([-4, -2, 0, 2, 4])
    return ts, pert_true


def test_no_perturbation_1d():
    """Test with uniform data in 1D."""
    observations = 5
    # given all the values are the same, there should not be perturbations
    ts = np.ones(observations)
    pert_zero = 0
    assert_array_equal(pert_zero, get_perturbation(ts))


def test_no_perturbation_2d_axis_last():
    """Test with uniform data in 2D along the last axis."""
    observations = 5
    instruments = 2
    # given all the values are the same, there should not be perturbations
    ts = np.ones((instruments, observations))
    pert_zero = np.zeros((instruments, observations))
    assert_array_equal(pert_zero, get_perturbation(ts, axis=-1))


def test_no_perturbation_2d_axis_first():
    """Test with uniform data in 2D along the first axis."""
    observations = 5
    instruments = 2
    # given all the values are the same, there should not be perturbations
    ts = np.ones((observations, instruments))
    pert_zero = np.zeros((observations, instruments))
    assert_array_equal(pert_zero, get_perturbation(ts, axis=0))


def test_known_perturbation_zero_mean_1d(pert_zero_mean):
    """Test with zero-mean data in 1D."""
    ts, pert_known = pert_zero_mean
    assert_array_equal(pert_known, get_perturbation(ts))


def test_known_perturbation_zero_mean_2d_axis_last(pert_zero_mean):
    """Test with zero-mean data in 2D along the last axis."""
    ts, pert_known = pert_zero_mean
    ts = np.array([ts, ts, ts])
    pert_known = np.array([pert_known, pert_known, pert_known])
    assert_array_equal(pert_known, get_perturbation(ts, axis=-1))


def test_known_perturbation_zero_mean_2d_axis_first(pert_zero_mean):
    """Test with zero-mean data in 2D along the first axis."""
    ts, pert_known = pert_zero_mean
    ts = np.array([ts, ts, ts]).transpose()
    pert_known = np.array([pert_known, pert_known, pert_known]).transpose()
    assert_array_equal(pert_known, get_perturbation(ts, axis=0))


def test_known_perturbation_nonzero_mean_1d(pert_nonzero_mean):
    """Test with non-zero-mean data in 1D."""
    ts, pert_known = pert_nonzero_mean
    assert_array_equal(pert_known, get_perturbation(ts))


def test_known_perturbation_nonzero_mean_2d_axis_last(pert_nonzero_mean):
    """Test with non-zero-mean data in 2D along the last axis."""
    ts, pert_known = pert_nonzero_mean
    ts = np.array([ts, ts, ts])
    pert_known = np.array([pert_known, pert_known, pert_known])
    assert_array_equal(pert_known, get_perturbation(ts, axis=-1))


def test_known_perturbation_nonzero_mean_2d_axis_first(pert_nonzero_mean):
    """Test with non-zero-mean data in 2D along the first axis."""
    ts, pert_known = pert_nonzero_mean
    ts = np.array([ts, ts, ts]).transpose()
    pert_known = np.array([pert_known, pert_known, pert_known]).transpose()
    assert_array_equal(pert_known, get_perturbation(ts, axis=0))


#
# Kinematic Flux Tests
#
@pytest.fixture()
def uvw_and_known_kf_zero_mean():
    """Return components and kinematic flux for zero-mean time series."""
    u = np.array([-2, -1, 0, 1, 2])
    v = -u
    w = 2 * u
    kf_true = {'uv': -2, 'uw': 4, 'vw': -4}
    return u, v, w, kf_true


@pytest.fixture()
def uvw_and_known_kf_nonzero_mean():
    """Return components and kinematic flux for non-zero-mean time series."""
    u = np.array([-2, -1, 0, 1, 5])
    v = -u
    w = 2 * u
    kf_true = {'uv': -5.84, 'uw': 11.68, 'vw': -11.68}
    return u, v, w, kf_true


def test_kf_1d_zero_mean(uvw_and_known_kf_zero_mean):
    """Test kinematic flux calculation in 1D with zero-mean time series."""
    u, v, w, kf_true = uvw_and_known_kf_zero_mean
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


def test_kf_1d_nonzero_mean(uvw_and_known_kf_nonzero_mean):
    """Test kinematic flux calculation in 1D with non-zero-mean time series."""
    u, v, w, kf_true = uvw_and_known_kf_nonzero_mean
    assert_array_equal(kinematic_flux(u, v, perturbation=False),
                       kf_true['uv'])
    assert_array_equal(kinematic_flux(u, w, perturbation=False),
                       kf_true['uw'])
    assert_array_equal(kinematic_flux(v, w, perturbation=False),
                       kf_true['vw'])


def test_kf_2d_axis_last_zero_mean(uvw_and_known_kf_zero_mean):
    """Test kinematic flux calculation in 2D with zero-mean time series along last axis."""
    u, v, w, kf_true = uvw_and_known_kf_zero_mean
    u = np.array([u, u, u])
    v = np.array([v, v, v])
    w = np.array([w, w, w])

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


def test_kf_2d_axis_last_nonzero_mean(uvw_and_known_kf_nonzero_mean):
    """Test kinematic flux calculation in 2D with non-zero-mean time series along last axis."""
    u, v, w, kf_true = uvw_and_known_kf_nonzero_mean
    u = np.array([u, u, u])
    v = np.array([v, v, v])
    w = np.array([w, w, w])

    assert_array_equal(kinematic_flux(u, v, perturbation=False, axis=-1),
                       kf_true['uv'])
    assert_array_equal(kinematic_flux(u, w, perturbation=False, axis=-1),
                       kf_true['uw'])
    assert_array_equal(kinematic_flux(v, w, perturbation=False, axis=-1),
                       kf_true['vw'])


def test_kf_2d_axis_first_zero_mean(uvw_and_known_kf_zero_mean):
    """Test kinematic flux calculation in 2D with zero-mean time series along first axis."""
    u, v, w, kf_true = uvw_and_known_kf_zero_mean
    u = np.array([u, u, u]).transpose()
    v = np.array([v, v, v]).transpose()
    w = np.array([w, w, w]).transpose()

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


def test_kf_2d_axis_first_nonzero_mean(uvw_and_known_kf_nonzero_mean):
    """Test kinematic flux in 2D with non-zero-mean time series along first axis."""
    u, v, w, kf_true = uvw_and_known_kf_nonzero_mean
    u = np.array([u, u, u]).transpose()
    v = np.array([v, v, v]).transpose()
    w = np.array([w, w, w]).transpose()

    assert_array_equal(kinematic_flux(u, v, perturbation=False, axis=0),
                       kf_true['uv'])
    assert_array_equal(kinematic_flux(u, w, perturbation=False, axis=0),
                       kf_true['uw'])
    assert_array_equal(kinematic_flux(v, w, perturbation=False, axis=0),
                       kf_true['vw'])


#
# Friction Velocity Tests
#
@pytest.fixture()
def uvw_and_known_u_star_zero_mean():
    """Return components and friction velocity for a zero-mean time series."""
    u = np.array([-2, -1, 0, 1, 2])
    v = -u
    w = 2 * u
    u_star_true = {'uw': 2.0, 'uwvw': 2.3784142300054421}
    return u, v, w, u_star_true


@pytest.fixture()
def uvw_and_known_friction_velocity_xarray():
    """Provide a set of u, v, w with a known tke value as an xarray."""
    # Define coordinate values
    pressure = [1000, 850]  # hPa
    lat = [10, 20]          # degrees North
    lon = [30, 40]          # degrees East
    time = np.array(['2025-01-01T00:00', '2025-01-01T06:00'], dtype='datetime64')

    # Define dimensions
    dims = ('pressure', 'lat', 'lon', 'time')

    # Generate 16 linearly spaced values between -30 and 30
    uwind_values = np.linspace(0, 30, num=16).reshape(2, 2, 2, 2)
    vwind_values = np.linspace(-30, 0, num=16).reshape(2, 2, 2, 2)
    wwind_values = np.linspace(-1, 2, num=16).reshape(2, 2, 2, 2)

    # Apply units
    u = uwind_values * units('m/s')
    v = vwind_values * units('m/s')
    w = wwind_values * units('m/s')

    # Create the Dataset
    ds = xr.Dataset(
        {
            'uwind': (dims, u),
            'vwind': (dims, v),
            'wwind': (dims, w),
        },
        coords={
            'pressure': pressure,
            'lat': lat,
            'lon': lon,
            'time': time
        }
    )
    expected = np.full((2, 2, 2), .3760603) * units('meter / second')
    return ds, expected


@pytest.fixture()
def uvw_and_known_u_star_nonzero_mean():
    """Return components and friction velocity for a non-zero-mean time series."""
    u = np.array([-2, -1, 0, 1, 5])
    v = -u
    w = 2 * u
    u_star_true = {'uw': 3.4176014981270124, 'uwvw': 4.0642360178166017}
    return u, v, w, u_star_true


def test_u_star_1d_zero_mean(uvw_and_known_u_star_zero_mean):
    """Test friction velocity in 1D with a zero-mean time series."""
    u, v, w, u_star_true = uvw_and_known_u_star_zero_mean
    assert_almost_equal(friction_velocity(u, w, perturbation=False),
                        u_star_true['uw'])
    assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False),
                        u_star_true['uwvw'])


def test_u_star_1d_nonzero_mean(uvw_and_known_u_star_nonzero_mean):
    """Test friction velocity in 1D with a non-zero-mean time series."""
    u, v, w, u_star_true = uvw_and_known_u_star_nonzero_mean
    assert_almost_equal(friction_velocity(u, w, perturbation=False),
                        u_star_true['uw'])
    assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False),
                        u_star_true['uwvw'])


def test_friction_velocity_nonzero_xarray(uvw_and_known_friction_velocity_xarray):
    """Test friction velocity in 1d with an xarray."""
    data, expected = uvw_and_known_friction_velocity_xarray
    assert_array_almost_equal(friction_velocity(data.uwind, data.wwind, data.vwind), expected)


def test_u_star_2d_axis_last_zero_mean(uvw_and_known_u_star_zero_mean):
    """Test friction velocity in 2D with a zero-mean time series along the last axis."""
    u, v, w, u_star_true = uvw_and_known_u_star_zero_mean
    u = np.array([u, u, u])
    v = np.array([v, v, v])
    w = np.array([w, w, w])

    assert_almost_equal(friction_velocity(u, w, perturbation=False,
                        axis=-1), u_star_true['uw'])
    assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False,
                        axis=-1), u_star_true['uwvw'])


def test_u_star_2d_axis_last_nonzero_mean(uvw_and_known_u_star_nonzero_mean):
    """Test friction velocity in 2D with a non-zero-mean time series along the last axis."""
    u, v, w, u_star_true = uvw_and_known_u_star_nonzero_mean
    u = np.array([u, u, u])
    v = np.array([v, v, v])
    w = np.array([w, w, w])

    assert_almost_equal(friction_velocity(u, w, perturbation=False,
                        axis=-1), u_star_true['uw'])
    assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False,
                        axis=-1), u_star_true['uwvw'])


def test_u_star_2d_axis_first_zero_mean(uvw_and_known_u_star_zero_mean):
    """Test friction velocity in 2D with a zero-mean time series along the first axis."""
    u, v, w, u_star_true = uvw_and_known_u_star_zero_mean
    u = np.array([u, u, u]).transpose()
    v = np.array([v, v, v]).transpose()
    w = np.array([w, w, w]).transpose()

    assert_almost_equal(friction_velocity(u, w, perturbation=False,
                        axis=0), u_star_true['uw'])
    assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False,
                        axis=0), u_star_true['uwvw'])


def test_u_star_2d_axis_first_nonzero_mean(uvw_and_known_u_star_nonzero_mean):
    """Test friction velocity in 2D with a non-zero-mean time series along the first axis."""
    u, v, w, u_star_true = uvw_and_known_u_star_nonzero_mean
    u = np.array([u, u, u]).transpose()
    v = np.array([v, v, v]).transpose()
    w = np.array([w, w, w]).transpose()

    assert_almost_equal(friction_velocity(u, w, perturbation=False,
                        axis=0), u_star_true['uw'])
    assert_almost_equal(friction_velocity(u, w, v=v, perturbation=False,
                        axis=0), u_star_true['uwvw'])
