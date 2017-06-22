# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `thermo` module."""

import numpy as np
import pytest

from metpy.calc import (cape_cin, density, dewpoint, dewpoint_rh, dry_lapse, el,
                        equivalent_potential_temperature, lcl, lfc, mixing_ratio,
                        mixing_ratio_from_specific_humidity, moist_lapse,
                        parcel_profile, potential_temperature,
                        psychrometric_vapor_pressure_wet,
                        relative_humidity_from_mixing_ratio,
                        relative_humidity_from_specific_humidity,
                        relative_humidity_wet_psychrometric,
                        saturation_mixing_ratio,
                        saturation_vapor_pressure, vapor_pressure,
                        virtual_potential_temperature, virtual_temperature)

from metpy.calc.thermo import _find_append_zero_crossings
from metpy.testing import assert_almost_equal, assert_array_almost_equal, assert_nan
from metpy.units import units


def test_potential_temperature():
    """Test potential_temperature calculation."""
    temp = np.array([278., 283., 291., 298.]) * units.kelvin
    pres = np.array([900., 500., 300., 100.]) * units.mbar
    real_th = np.array([286.493, 344.961, 410.4335, 575.236]) * units.kelvin
    assert_array_almost_equal(potential_temperature(pres, temp), real_th, 3)


def test_scalar():
    """Test potential_temperature accepts scalar values."""
    assert_almost_equal(potential_temperature(1000. * units.mbar, 293. * units.kelvin),
                        293. * units.kelvin, 4)
    assert_almost_equal(potential_temperature(800. * units.mbar, 293. * units.kelvin),
                        312.2828 * units.kelvin, 4)


def test_fahrenheit():
    """Test that potential_temperature handles temperature values in Fahrenheit."""
    assert_almost_equal(potential_temperature(800. * units.mbar, 68. * units.degF),
                        (312.444 * units.kelvin).to(units.degF), 2)


def test_pot_temp_inhg():
    """Test that potential_temperature can handle pressure not in mb (issue #165)."""
    assert_almost_equal(potential_temperature(29.92 * units.inHg, 29 * units.degC),
                        301.019735 * units.kelvin, 4)


def test_dry_lapse():
    """Test dry_lapse calculation."""
    levels = np.array([1000, 900, 864.89]) * units.mbar
    temps = dry_lapse(levels, 303.15 * units.kelvin)
    assert_array_almost_equal(temps,
                              np.array([303.15, 294.16, 290.83]) * units.kelvin, 2)


def test_dry_lapse_2_levels():
    """Test dry_lapse calculation when given only two levels."""
    temps = dry_lapse(np.array([1000., 500.]) * units.mbar, 293. * units.kelvin)
    assert_array_almost_equal(temps, [293., 240.3723] * units.kelvin, 4)


def test_moist_lapse():
    """Test moist_lapse calculation."""
    temp = moist_lapse(np.array([1000., 800., 600., 500., 400.]) * units.mbar,
                       293. * units.kelvin)
    true_temp = np.array([293, 284.64, 272.81, 264.42, 252.91]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_moist_lapse_degc():
    """Test moist_lapse with Celsius temperatures."""
    temp = moist_lapse(np.array([1000., 800., 600., 500., 400.]) * units.mbar,
                       19.85 * units.degC)
    true_temp = np.array([293, 284.64, 272.81, 264.42, 252.91]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_parcel_profile():
    """Test parcel profile calculation."""
    levels = np.array([1000., 900., 800., 700., 600., 500., 400.]) * units.mbar
    true_prof = np.array([303.15, 294.16, 288.026, 283.073, 277.058, 269.402,
                          258.966]) * units.kelvin

    prof = parcel_profile(levels, 30. * units.degC, 20. * units.degC)
    assert_array_almost_equal(prof, true_prof, 2)


def test_parcel_profile_saturated():
    """Test parcel_profile works when LCL in levels (issue #232)."""
    levels = np.array([1000., 700., 500.]) * units.mbar
    true_prof = np.array([296.95, 284.381, 271.123]) * units.kelvin

    prof = parcel_profile(levels, 23.8 * units.degC, 23.8 * units.degC)
    assert_array_almost_equal(prof, true_prof, 2)


def test_sat_vapor_pressure():
    """Test saturation_vapor_pressure calculation."""
    temp = np.array([5., 10., 18., 25.]) * units.degC
    real_es = np.array([8.72, 12.27, 20.63, 31.67]) * units.mbar
    assert_array_almost_equal(saturation_vapor_pressure(temp), real_es, 2)


def test_sat_vapor_pressure_scalar():
    """Test saturation_vapor_pressure handles scalar values."""
    es = saturation_vapor_pressure(0 * units.degC)
    assert_almost_equal(es, 6.112 * units.mbar, 3)


def test_sat_vapor_pressure_fahrenheit():
    """Test saturation_vapor_pressure handles temperature in Fahrenheit."""
    temp = np.array([50., 68.]) * units.degF
    real_es = np.array([12.2717, 23.3695]) * units.mbar
    assert_array_almost_equal(saturation_vapor_pressure(temp), real_es, 4)


def test_basic_dewpoint_rh():
    """Test dewpoint_rh function."""
    temp = np.array([30., 25., 10., 20., 25.]) * units.degC
    rh = np.array([30., 45., 55., 80., 85.]) / 100.

    real_td = np.array([11, 12, 1, 16, 22]) * units.degC
    assert_array_almost_equal(real_td, dewpoint_rh(temp, rh), 0)


def test_scalar_dewpoint_rh():
    """Test dewpoint_rh with scalar values."""
    td = dewpoint_rh(10.6 * units.degC, 0.37)
    assert_almost_equal(td, 26. * units.degF, 0)


def test_dewpoint():
    """Test dewpoint calculation."""
    assert_almost_equal(dewpoint(6.112 * units.mbar), 0. * units.degC, 2)


def test_dewpoint_weird_units():
    """Test dewpoint using non-standard units.

    Revealed from odd dimensionless units and ending up using numpy.ma math
    functions instead of numpy ones.
    """
    assert_almost_equal(dewpoint(15825.6 * units('g * mbar / kg')),
                        13.8564 * units.degC, 4)


def test_mixing_ratio():
    """Test mixing ratio calculation."""
    p = 998. * units.mbar
    e = 73.75 * units.mbar
    assert_almost_equal(mixing_ratio(e, p), 0.04963, 2)


def test_vapor_pressure():
    """Test vapor pressure calculation."""
    assert_almost_equal(vapor_pressure(998. * units.mbar, 0.04963),
                        73.74925 * units.mbar, 5)


def test_lcl():
    """Test LCL calculation."""
    lcl_pressure, lcl_temperature = lcl(1000. * units.mbar, 30. * units.degC, 20. * units.degC)
    assert_almost_equal(lcl_pressure, 864.761 * units.mbar, 2)
    assert_almost_equal(lcl_temperature, 17.676 * units.degC, 2)


def test_lcl_convergence():
    """Test LCL calculation convergence failure."""
    with pytest.raises(RuntimeError):
        lcl(1000. * units.mbar, 30. * units.degC, 20. * units.degC, max_iters=2)


def test_lfc_basic():
    """Test LFC calculation."""
    levels = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12., 9.4, 7., -49.]) * units.celsius
    dewpoints = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    l = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(l[0], 727.468 * units.mbar, 2)
    assert_almost_equal(l[1], 9.705 * units.celsius, 2)


def test_no_lfc():
    """Test LFC calculation when there is no LFC in the data."""
    levels = np.array([959., 867.9, 779.2, 647.5, 472.5, 321.9, 251.]) * units.mbar
    temperatures = np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5]) * units.celsius
    dewpoints = np.array([9., 4.3, -21.2, -26.7, -31., -53.3, -66.7]) * units.celsius
    lfc_pressure, lfc_temperature = lfc(levels, temperatures, dewpoints)
    assert assert_nan(lfc_pressure, levels.units)
    assert assert_nan(lfc_temperature, temperatures.units)


def test_lfc_inversion():
    """Test LFC when there is an inversion to be sure we don't pick that."""
    levels = np.array([963., 789., 782.3, 754.8, 728.1, 727., 700.,
                       571., 450., 300., 248.]) * units.mbar
    temperatures = np.array([25.4, 18.4, 17.8, 15.4, 12.9, 12.8,
                             10., -3.9, -16.3, -41.1, -51.5]) * units.celsius
    dewpoints = np.array([20.4, 0.4, -0.5, -4.3, -8., -8.2, -9.,
                          -23.9, -33.3, -54.1, -63.5]) * units.celsius
    l = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(l[0], 706.0103 * units.mbar, 2)
    assert_almost_equal(l[1], 10.6232 * units.celsius, 2)


def test_lfc_equals_lcl():
    """Test LFC when there is no cap and the lfc is equal to the lcl."""
    levels = np.array([912., 905.3, 874.4, 850., 815.1, 786.6, 759.1,
                       748., 732.2, 700., 654.8]) * units.mbar
    temperatures = np.array([29.4, 28.7, 25.2, 22.4, 19.4, 16.8,
                             14.3, 13.2, 12.6, 11.4, 7.1]) * units.celsius
    dewpoints = np.array([18.4, 18.1, 16.6, 15.4, 13.2, 11.4, 9.6,
                          8.8, 0., -18.6, -22.9]) * units.celsius
    l = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(l[0], 777.0333 * units.mbar, 2)
    assert_almost_equal(l[1], 15.8714 * units.celsius, 2)


def test_saturation_mixing_ratio():
    """Test saturation mixing ratio calculation."""
    p = 999. * units.mbar
    t = 288. * units.kelvin
    assert_almost_equal(saturation_mixing_ratio(p, t), .01068, 3)


def test_equivalent_potential_temperature():
    """Test equivalent potential temperature calculation."""
    p = 999. * units.mbar
    t = 288. * units.kelvin
    ept = equivalent_potential_temperature(p, t)
    assert_almost_equal(ept, 315.9548 * units.kelvin, 3)


def test_virtual_temperature():
    """Test virtual temperature calculation."""
    t = 288. * units.kelvin
    qv = .0016  # kg/kg
    tv = virtual_temperature(t, qv)
    assert_almost_equal(tv, 288.2796 * units.kelvin, 3)


def test_virtual_potential_temperature():
    """Test virtual potential temperature calculation."""
    p = 999. * units.mbar
    t = 288. * units.kelvin
    qv = .0016  # kg/kg
    theta_v = virtual_potential_temperature(p, t, qv)
    assert_almost_equal(theta_v, 288.3620 * units.kelvin, 3)


def test_density():
    """Test density calculation."""
    p = 999. * units.mbar
    t = 288. * units.kelvin
    qv = .0016  # kg/kg
    rho = density(p, t, qv).to(units.kilogram / units.meter ** 3)
    assert_almost_equal(rho, 1.2072 * (units.kilogram / units.meter ** 3), 3)


def test_el():
    """Test equilibrium layer calculation."""
    levels = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.celsius
    dewpoints = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_almost_equal(el_pressure, 520.8700 * units.mbar, 3)
    assert_almost_equal(el_temperature, -11.7027 * units.degC, 3)


def test_no_el():
    """Test equilibrium layer calculation when there is no EL in the data."""
    levels = np.array([959., 867.9, 779.2, 647.5, 472.5, 321.9, 251.]) * units.mbar
    temperatures = np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5]) * units.celsius
    dewpoints = np.array([19., 14.3, -11.2, -16.7, -21., -43.3, -56.7]) * units.celsius
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert assert_nan(el_pressure, levels.units)
    assert assert_nan(el_temperature, temperatures.units)


def test_el_lfc_equals_lcl():
    """Test equilibrium layer calculation when the lfc equals the lcl."""
    levels = np.array([912., 905.3, 874.4, 850., 815.1, 786.6, 759.1, 748.,
                       732.3, 700., 654.8, 606.8, 562.4, 501.8, 500., 482.,
                       400., 393.3, 317.1, 307., 300., 252.7, 250., 200.,
                       199.3, 197., 190., 172., 156.6, 150., 122.9, 112.,
                       106.2, 100.]) * units.mbar
    temperatures = np.array([29.4, 28.7, 25.2, 22.4, 19.4, 16.8, 14.3,
                             13.2, 12.6, 11.4, 7.1, 2.2, -2.7, -10.1,
                             -10.3, -12.4, -23.3, -24.4, -38., -40.1, -41.1,
                             -49.8, -50.3, -59.1, -59.1, -59.3, -59.7, -56.3,
                             -56.9, -57.1, -59.1, -60.1, -58.6, -56.9]) * units.celsius
    dewpoints = np.array([18.4, 18.1, 16.6, 15.4, 13.2, 11.4, 9.6, 8.8, 0.,
                          -18.6, -22.9, -27.8, -32.7, -40.1, -40.3, -42.4, -53.3,
                          -54.4, -68., -70.1, -70., -70., -70., -70., -70., -70.,
                          -70., -70., -70., -70., -70., -70., -70., -70.]) * units.celsius
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_almost_equal(el_pressure, 175.8684 * units.mbar, 3)
    assert_almost_equal(el_temperature, -57.0307 * units.degC, 3)


def test_wet_psychrometric_vapor_pressure():
    """Test calculation of vapor pressure from wet and dry bulb temperatures."""
    p = 1013.25 * units.mbar
    dry_bulb_temperature = 20. * units.degC
    wet_bulb_temperature = 18. * units.degC
    psychrometric_vapor_pressure = psychrometric_vapor_pressure_wet(dry_bulb_temperature,
                                                                    wet_bulb_temperature, p)
    assert_almost_equal(psychrometric_vapor_pressure, 19.3673 * units.mbar, 3)


def test_wet_psychrometric_rh():
    """Test calculation of relative humidity from wet and dry bulb temperatures."""
    p = 1013.25 * units.mbar
    dry_bulb_temperature = 20. * units.degC
    wet_bulb_temperature = 18. * units.degC
    psychrometric_rh = relative_humidity_wet_psychrometric(dry_bulb_temperature,
                                                           wet_bulb_temperature, p)
    assert_almost_equal(psychrometric_rh, 82.8747 * units.percent, 3)


def test_wet_psychrometric_rh_kwargs():
    """Test calculation of relative humidity from wet and dry bulb temperatures."""
    p = 1013.25 * units.mbar
    dry_bulb_temperature = 20. * units.degC
    wet_bulb_temperature = 18. * units.degC
    coeff = 6.1e-4 / units.kelvin
    psychrometric_rh = relative_humidity_wet_psychrometric(dry_bulb_temperature,
                                                           wet_bulb_temperature, p,
                                                           psychrometer_coefficient=coeff)
    assert_almost_equal(psychrometric_rh, 82.9701 * units.percent, 3)


def test_rh_mixing_ratio():
    """Tests relative humidity from mixing ratio."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    w = 0.012
    rh = relative_humidity_from_mixing_ratio(w, temperature, p)
    assert_almost_equal(rh, 81.7219 * units.percent, 3)


def test_mixing_ratio_from_specific_humidity():
    """Tests mixing ratio from specific humidity."""
    q = 0.012
    w = mixing_ratio_from_specific_humidity(q)
    assert_almost_equal(w, 0.01215, 3)


def test_rh_specific_humidity():
    """Tests relative humidity from specific humidity."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    q = 0.012
    rh = relative_humidity_from_specific_humidity(q, temperature, p)
    assert_almost_equal(rh, 82.7145 * units.percent, 3)


def test_cape_cin():
    """Tests the basic CAPE and CIN calculation."""
    p = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]).to('degC')
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 58.0368212 * units('joule / kilogram'), 6)
    assert_almost_equal(cin, -89.8073512 * units('joule / kilogram'), 6)


def test_cape_cin_no_el():
    """Tests that CAPE works with no EL."""
    p = np.array([959., 779.2, 751.3, 724.3]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]).to('degC')
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 0.08750805 * units('joule / kilogram'), 6)
    assert_almost_equal(cin, -89.8073512 * units('joule / kilogram'), 6)


def test_cape_cin_no_lfc():
    """Tests that CAPE is zero with no LFC."""
    p = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 24.6, 22., 20.4, 18., -10.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]).to('degC')
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 0.0 * units('joule / kilogram'), 6)
    assert_almost_equal(cin, 0.0 * units('joule / kilogram'), 6)


def test_find_append_zero_crossings():
    """Tests finding and appending zero crossings of an x, y series."""
    x = np.arange(11) * units.hPa
    y = np.array([3, 2, 1, -1, 2, 2, 0, 1, 0, -1, 2]) * units.degC
    x2, y2 = _find_append_zero_crossings(x, y)

    x_truth = np.array([0., 1., 2., 2.5, 3., 3.33333333, 4., 5.,
                        6., 7., 8., 9., 9.33333333, 10.]) * units.hPa
    y_truth = np.array([3, 2, 1, 0, -1, 0, 2, 2, 0, 1, 0, -1, 0, 2]) * units.degC
    assert_array_almost_equal(x2, x_truth, 6)
    assert_almost_equal(y2, y_truth, 6)
