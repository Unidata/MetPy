# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from metpy.calc.thermo import *  # noqa: F403
from metpy.units import units
from metpy.testing import assert_almost_equal, assert_array_almost_equal


def test_potential_temperature():
    'Basic test of potential_temperature calculation'
    temp = np.array([278., 283., 291., 298.]) * units.kelvin
    pres = np.array([900., 500., 300., 100.]) * units.mbar
    real_th = np.array([286.493, 344.961, 410.4335, 575.236]) * units.kelvin
    assert_array_almost_equal(potential_temperature(pres, temp), real_th, 3)


def test_scalar():
    'Test potential_temperature accepts scalar values'
    assert_almost_equal(potential_temperature(1000. * units.mbar, 293. * units.kelvin),
                        293. * units.kelvin, 4)
    assert_almost_equal(potential_temperature(800. * units.mbar, 293. * units.kelvin),
                        312.2828 * units.kelvin, 4)


def test_fahrenheit():
    'Test that potential_temperature handles temperature values in Fahrenheit'
    assert_almost_equal(potential_temperature(800. * units.mbar, 68. * units.degF),
                        (312.444 * units.kelvin).to(units.degF), 2)


def test_pot_temp_inhg():
    'Tests that potential_temperature can handle pressure not in mb (issue #165)'
    assert_almost_equal(potential_temperature(29.92 * units.inHg, 29 * units.degC),
                        301.019735 * units.kelvin, 4)


def test_dry_lapse():
    'Basic test of dry_lapse calculation'
    levels = np.array([1000, 900, 864.89]) * units.mbar
    temps = dry_lapse(levels, 303.15 * units.kelvin)
    assert_array_almost_equal(temps,
                              np.array([303.15, 294.16, 290.83]) * units.kelvin, 2)


def test_2_levels():
    'Test dry_lapse calculation when given only two levels'
    temps = dry_lapse(np.array([1000., 500.]) * units.mbar, 293. * units.kelvin)
    assert_array_almost_equal(temps, [293., 240.3723] * units.kelvin, 4)


def test_moist_lapse():
    'Basic test of moist_lapse calculation'
    temp = moist_lapse(np.array([1000., 800., 600., 500., 400.]) * units.mbar,
                       293. * units.kelvin)
    true_temp = np.array([293, 284.64, 272.81, 264.42, 252.91]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_moist_lapse_degc():
    'Test moist_lapse with Celsius temperatures'
    temp = moist_lapse(np.array([1000., 800., 600., 500., 400.]) * units.mbar,
                       19.85 * units.degC)
    true_temp = np.array([293, 284.64, 272.81, 264.42, 252.91]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_parcel_profile():
    'Basic test of parcel profile'
    levels = np.array([1000., 900., 800., 700., 600., 500., 400.]) * units.mbar
    true_prof = np.array([303.15, 294.16, 288.026, 283.073, 277.058, 269.402,
                          258.966]) * units.kelvin

    prof = parcel_profile(levels, 30. * units.degC, 20. * units.degC)
    assert_array_almost_equal(prof, true_prof, 2)


def test_sat_vapor_pressure():
    'Basic test of saturation_vapor_pressure calculation'
    temp = np.array([5., 10., 18., 25.]) * units.degC
    real_es = np.array([8.72, 12.27, 20.63, 31.67]) * units.mbar
    assert_array_almost_equal(saturation_vapor_pressure(temp), real_es, 2)


def test_sat_vapor_pressure_scalar():
    'Test saturation_vapor_pressure handles scalar values'
    es = saturation_vapor_pressure(0 * units.degC)
    assert_almost_equal(es, 6.112 * units.mbar, 3)


def test_sat_vapor_pressure_fahrenheit():
    'Test saturation_vapor_pressure handles temperature in Fahrenheit'
    temp = np.array([50., 68.]) * units.degF
    real_es = np.array([12.2717, 23.3695]) * units.mbar
    assert_array_almost_equal(saturation_vapor_pressure(temp), real_es, 4)


def test_basic_dewpoint_rh():
    'Basic test of dewpoint_rh function'
    temp = np.array([30., 25., 10., 20., 25.]) * units.degC
    rh = np.array([30., 45., 55., 80., 85.]) / 100.

    real_td = np.array([11, 12, 1, 16, 22]) * units.degC
    assert_array_almost_equal(real_td, dewpoint_rh(temp, rh), 0)


def test_scalar_dewpoint_rh():
    'Test dewpoint_rh with scalar values'
    td = dewpoint_rh(10.6 * units.degC, 0.37)
    assert_almost_equal(td, 26. * units.degF, 0)


def test_dewpoint():
    'Basic test of dewpoint calculation'
    assert_almost_equal(dewpoint(6.112 * units.mbar), 0. * units.degC, 2)


def test_dewpoint_weird_units():
    """Test of dewpoint revealed from odd dimensionless units and ending up using
    numpy.ma math functions instead of numpy ones."""
    assert_almost_equal(dewpoint(15825.6 * units('g * mbar / kg')),
                        13.8564 * units.degC, 4)


def test_mixing_ratio():
    'Basic test of mixing ratio calculation'
    p = 998. * units.mbar
    e = 73.75 * units.mbar
    assert_almost_equal(mixing_ratio(e, p), 0.04963, 2)


def test_vapor_pressure():
    'Basic test of vapor pressure calculation'
    assert_almost_equal(vapor_pressure(998. * units.mbar, 0.04963),
                        73.74925 * units.mbar, 5)


def test_lcl():
    'Simple test of LCL calculation.'
    l = lcl(1000. * units.mbar, 30. * units.degC, 20. * units.degC)
    assert_almost_equal(l, 864.761 * units.mbar, 2)


def test_lfc_basic():
    'Simple test of LFC calculation.'
    levels = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12., 9.4, 7., -49.]) * units.celsius
    dewpoints = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    l = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(l[0], 727.468 * units.mbar, 2)
    assert_almost_equal(l[1], 9.705 * units.celsius, 2)


def test_saturation_mixing_ratio():
    'Simple test of saturation mixing ratio calculation.'
    p = 999. * units.mbar
    t = 288. * units.kelvin
    assert_almost_equal(saturation_mixing_ratio(p, t), .01068, 3)


def test_equivalent_potential_temperature():
    'Simple test of equivalent potential temperature calculation.'
    p = 999. * units.mbar
    t = 288. * units.kelvin
    ept = equivalent_potential_temperature(p, t)
    assert_almost_equal(ept, 315.9548 * units.kelvin, 3)
