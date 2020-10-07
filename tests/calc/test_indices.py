# Copyright (c) 2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `indices` module."""

from datetime import datetime

import numpy as np

from metpy.calc import (
    bulk_shear,
    bunkers_storm_motion,
    critical_angle,
    mean_pressure_weighted,
    precipitable_water,
    significant_tornado,
    supercell_composite,
)
from metpy.testing import assert_almost_equal, assert_array_equal, get_upper_air_data
from metpy.units import concatenate, units


def test_precipitable_water():
    """Test precipitable water with observed sounding."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    pw = precipitable_water(data["pressure"], data["dewpoint"], top=400 * units.hPa)
    truth = (0.8899441949243486 * units("inches")).to("millimeters")
    assert_array_equal(pw, truth)


def test_precipitable_water_no_bounds():
    """Test precipitable water with observed sounding and no bounds given."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    dewpoint = data["dewpoint"]
    pressure = data["pressure"]
    inds = pressure >= 400 * units.hPa
    pw = precipitable_water(pressure[inds], dewpoint[inds])
    truth = (0.8899441949243486 * units("inches")).to("millimeters")
    assert_array_equal(pw, truth)


def test_precipitable_water_bound_error():
    """Test with no top bound given and data that produced floating point issue #596."""
    pressure = (
        np.array(
            [
                993.0,
                978.0,
                960.5,
                927.6,
                925.0,
                895.8,
                892.0,
                876.0,
                45.9,
                39.9,
                36.0,
                36.0,
                34.3,
            ]
        )
        * units.hPa
    )
    dewpoint = (
        np.array(
            [25.5, 24.1, 23.1, 21.2, 21.1, 19.4, 19.2, 19.2, -87.1, -86.5, -86.5, -86.5, -88.1]
        )
        * units.degC
    )
    pw = precipitable_water(pressure, dewpoint)
    truth = 89.86955998646951 * units("millimeters")
    assert_almost_equal(pw, truth, 8)


def test_precipitable_water_nans():
    """Test that PW returns appropriate number if NaNs are present."""
    pressure = (
        np.array(
            [
                1001,
                1000,
                997,
                977.9,
                977,
                957,
                937.8,
                925,
                906,
                899.3,
                887,
                862.5,
                854,
                850,
                800,
                793.9,
                785,
                777,
                771,
                762,
                731.8,
                726,
                703,
                700,
                655,
                630,
                621.2,
                602,
                570.7,
                548,
                546.8,
                539,
                513,
                511,
                485,
                481,
                468,
                448,
                439,
                424,
                420,
                412,
            ]
        )
        * units.hPa
    )
    dewpoint = (
        np.array(
            [
                -25.1,
                -26.1,
                -26.8,
                np.nan,
                -27.3,
                -28.2,
                np.nan,
                -27.2,
                -26.6,
                np.nan,
                -27.4,
                np.nan,
                -23.5,
                -23.5,
                -25.1,
                np.nan,
                -22.9,
                -17.8,
                -16.6,
                np.nan,
                np.nan,
                -16.4,
                np.nan,
                -18.5,
                -21.0,
                -23.7,
                np.nan,
                -28.3,
                np.nan,
                -32.6,
                np.nan,
                -33.8,
                -35.0,
                -35.1,
                -38.1,
                -40.0,
                -43.3,
                -44.6,
                -46.4,
                -47.0,
                -49.2,
                -50.7,
            ]
        )
        * units.degC
    )
    pw = precipitable_water(pressure, dewpoint)
    truth = 4.003709214463873 * units.mm
    assert_almost_equal(pw, truth, 8)


def test_mean_pressure_weighted():
    """Test pressure-weighted mean wind function with vertical interpolation."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    u, v = mean_pressure_weighted(
        data["pressure"],
        data["u_wind"],
        data["v_wind"],
        height=data["height"],
        depth=6000 * units("meter"),
    )
    assert_almost_equal(u, 6.0208700094534775 * units("m/s"), 7)
    assert_almost_equal(v, 7.966031839967931 * units("m/s"), 7)


def test_mean_pressure_weighted_elevated():
    """Test pressure-weighted mean wind function with a base above the surface."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    u, v = mean_pressure_weighted(
        data["pressure"],
        data["u_wind"],
        data["v_wind"],
        height=data["height"],
        depth=3000 * units("meter"),
        bottom=data["height"][0] + 3000 * units("meter"),
    )
    assert_almost_equal(u, 8.270829843626476 * units("m/s"), 7)
    assert_almost_equal(v, 1.7392601775853547 * units("m/s"), 7)


def test_bunkers_motion():
    """Test Bunkers storm motion with observed sounding."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    motion = concatenate(
        bunkers_storm_motion(data["pressure"], data["u_wind"], data["v_wind"], data["height"])
    )
    truth = [
        1.4537892577864744,
        2.0169333025630616,
        10.587950761120482,
        13.915130377372801,
        6.0208700094534775,
        7.9660318399679308,
    ] * units("m/s")
    assert_almost_equal(motion.flatten(), truth, 8)


def test_bulk_shear():
    """Test bulk shear with observed sounding."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    u, v = bulk_shear(
        data["pressure"],
        data["u_wind"],
        data["v_wind"],
        height=data["height"],
        depth=6000 * units("meter"),
    )
    truth = [29.899581266946115, -14.389225800205509] * units("knots")
    assert_almost_equal(u.to("knots"), truth[0], 8)
    assert_almost_equal(v.to("knots"), truth[1], 8)


def test_bulk_shear_no_depth():
    """Test bulk shear with observed sounding and no depth given. Issue #568."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    u, v = bulk_shear(data["pressure"], data["u_wind"], data["v_wind"], height=data["height"])
    truth = [20.225018939, 22.602359692] * units("knots")
    assert_almost_equal(u.to("knots"), truth[0], 8)
    assert_almost_equal(v.to("knots"), truth[1], 8)


def test_bulk_shear_elevated():
    """Test bulk shear with observed sounding and a base above the surface."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    u, v = bulk_shear(
        data["pressure"],
        data["u_wind"],
        data["v_wind"],
        height=data["height"],
        bottom=data["height"][0] + 3000 * units("meter"),
        depth=3000 * units("meter"),
    )
    truth = [0.9655943923302139, -3.8405428777944466] * units("m/s")
    assert_almost_equal(u, truth[0], 8)
    assert_almost_equal(v, truth[1], 8)


def test_supercell_composite():
    """Test supercell composite function."""
    mucape = [2000.0, 1000.0, 500.0, 2000.0] * units("J/kg")
    esrh = [400.0, 150.0, 45.0, 45.0] * units("m^2/s^2")
    ebwd = [30.0, 15.0, 5.0, 5.0] * units("m/s")
    truth = [16.0, 2.25, 0.0, 0.0]
    supercell_comp = supercell_composite(mucape, esrh, ebwd)
    assert_array_equal(supercell_comp, truth)


def test_supercell_composite_scalar():
    """Test supercell composite function with a single value."""
    mucape = 2000.0 * units("J/kg")
    esrh = 400.0 * units("m^2/s^2")
    ebwd = 30.0 * units("m/s")
    truth = 16.0
    supercell_comp = supercell_composite(mucape, esrh, ebwd)
    assert_almost_equal(supercell_comp, truth, 6)


def test_sigtor():
    """Test significant tornado parameter function."""
    sbcape = [2000.0, 2000.0, 2000.0, 2000.0, 3000, 4000] * units("J/kg")
    sblcl = [3000.0, 1500.0, 500.0, 1500.0, 1500, 800] * units("meter")
    srh1 = [200.0, 200.0, 200.0, 200.0, 300, 400] * units("m^2/s^2")
    shr6 = [20.0, 5.0, 20.0, 35.0, 20.0, 35] * units("m/s")
    truth = [0.0, 0, 1.777778, 1.333333, 2.0, 10.666667]
    sigtor = significant_tornado(sbcape, sblcl, srh1, shr6)
    assert_almost_equal(sigtor, truth, 6)


def test_sigtor_scalar():
    """Test significant tornado parameter function with a single value."""
    sbcape = 4000 * units("J/kg")
    sblcl = 800 * units("meter")
    srh1 = 400 * units("m^2/s^2")
    shr6 = 35 * units("m/s")
    truth = 10.666667
    sigtor = significant_tornado(sbcape, sblcl, srh1, shr6)
    assert_almost_equal(sigtor, truth, 6)


def test_critical_angle():
    """Test critical angle with observed sounding."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    ca = critical_angle(
        data["pressure"],
        data["u_wind"],
        data["v_wind"],
        data["height"],
        u_storm=0 * units("m/s"),
        v_storm=0 * units("m/s"),
    )
    truth = [140.0626637513269] * units("degrees")
    assert_almost_equal(ca, truth, 8)


def test_critical_angle_units():
    """Test critical angle with observed sounding and different storm motion units."""
    data = get_upper_air_data(datetime(2016, 5, 22, 0), "DDC")
    # Set storm motion in m/s
    ca_ms = critical_angle(
        data["pressure"],
        data["u_wind"],
        data["v_wind"],
        data["height"],
        u_storm=10 * units("m/s"),
        v_storm=10 * units("m/s"),
    )
    # Set same storm motion in kt and m/s
    ca_kt_ms = critical_angle(
        data["pressure"],
        data["u_wind"],
        data["v_wind"],
        data["height"],
        u_storm=10 * units("m/s"),
        v_storm=19.4384449244 * units("kt"),
    )
    # Make sure the resulting critical angles are equal
    assert_almost_equal(ca_ms, ca_kt_ms, 8)
