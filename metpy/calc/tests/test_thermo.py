# Copyright (c) 2008,2015,2016,2017,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `thermo` module."""

import numpy as np
import pytest

from metpy.calc import (brunt_vaisala_frequency, brunt_vaisala_frequency_squared,
                        brunt_vaisala_period, cape_cin, density, dewpoint,
                        dewpoint_from_specific_humidity, dewpoint_rh,
                        downdraft_cape, dry_lapse, dry_static_energy, el,
                        equivalent_potential_temperature,
                        exner_function, isentropic_interpolation, lcl, lfc, mixed_layer,
                        mixed_parcel, mixing_ratio, mixing_ratio_from_relative_humidity,
                        mixing_ratio_from_specific_humidity, moist_lapse,
                        moist_static_energy, most_unstable_cape_cin, most_unstable_parcel,
                        parcel_profile, potential_temperature,
                        psychrometric_vapor_pressure_wet,
                        relative_humidity_from_dewpoint,
                        relative_humidity_from_mixing_ratio,
                        relative_humidity_from_specific_humidity,
                        relative_humidity_wet_psychrometric,
                        saturation_equivalent_potential_temperature,
                        saturation_mixing_ratio,
                        saturation_vapor_pressure,
                        specific_humidity_from_mixing_ratio, static_stability,
                        surface_based_cape_cin, temperature_from_potential_temperature,
                        thickness_hydrostatic,
                        thickness_hydrostatic_from_relative_humidity, vapor_pressure,
                        virtual_potential_temperature, virtual_temperature,
                        wet_bulb_temperature)
from metpy.calc.thermo import _find_append_zero_crossings
from metpy.testing import assert_almost_equal, assert_array_almost_equal, assert_nan
from metpy.units import units


def test_relative_humidity_from_dewpoint():
    """Test Relative Humidity calculation."""
    assert_almost_equal(relative_humidity_from_dewpoint(25. * units.degC, 15. * units.degC),
                        53.80 * units.percent, 2)


def test_relative_humidity_from_dewpoint_with_f():
    """Test Relative Humidity accepts temperature in Fahrenheit."""
    assert_almost_equal(relative_humidity_from_dewpoint(70. * units.degF, 55. * units.degF),
                        58.935 * units.percent, 3)


def test_exner_function():
    """Test Exner function calculation."""
    pres = np.array([900., 500., 300., 100.]) * units.mbar
    truth = np.array([0.9703542, 0.8203834, 0.7090065, 0.518048]) * units.dimensionless
    assert_array_almost_equal(exner_function(pres), truth, 6)


def test_potential_temperature():
    """Test potential_temperature calculation."""
    temp = np.array([278., 283., 291., 298.]) * units.kelvin
    pres = np.array([900., 500., 300., 100.]) * units.mbar
    real_th = np.array([286.493, 344.961, 410.4335, 575.236]) * units.kelvin
    assert_array_almost_equal(potential_temperature(pres, temp), real_th, 3)


def test_temperature_from_potential_temperature():
    """Test temperature_from_potential_temperature calculation."""
    theta = np.array([286.12859679, 288.22362587, 290.31865495, 292.41368403]) * units.kelvin
    pres = np.array([850] * 4) * units.mbar
    real_t = np.array([273.15, 275.15, 277.15, 279.15]) * units.kelvin
    assert_array_almost_equal(temperature_from_potential_temperature(pres, theta),
                              real_t, 2)


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


def test_percent_dewpoint_rh():
    """Test dewpoint_rh with rh in percent."""
    td = dewpoint_rh(10.6 * units.degC, 37 * units.percent)
    assert_almost_equal(td, 26. * units.degF, 0)


def test_warning_dewpoint_rh():
    """Test that warning is raised for >120% RH."""
    with pytest.warns(UserWarning):
        dewpoint_rh(10.6 * units.degC, 50)


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
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(lfc_pressure, 727.468 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 9.705 * units.celsius, 2)


def test_lfc_ml():
    """Test Mixed-Layer LFC calculation."""
    levels = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12., 9.4, 7., -49.]) * units.celsius
    dewpoints = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    __, T_mixed, Td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = parcel_profile(levels, T_mixed, Td_mixed)
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints, mixed_parcel_prof)
    assert_almost_equal(lfc_pressure, 631.794 * units.mbar, 2)
    assert_almost_equal(lfc_temp, -1.862 * units.degC, 2)


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
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(lfc_pressure, 706.0103 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 10.6232 * units.celsius, 2)


def test_lfc_equals_lcl():
    """Test LFC when there is no cap and the lfc is equal to the lcl."""
    levels = np.array([912., 905.3, 874.4, 850., 815.1, 786.6, 759.1,
                       748., 732.2, 700., 654.8]) * units.mbar
    temperatures = np.array([29.4, 28.7, 25.2, 22.4, 19.4, 16.8,
                             14.3, 13.2, 12.6, 11.4, 7.1]) * units.celsius
    dewpoints = np.array([18.4, 18.1, 16.6, 15.4, 13.2, 11.4, 9.6,
                          8.8, 0., -18.6, -22.9]) * units.celsius
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(lfc_pressure, 777.0333 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 15.8714 * units.celsius, 2)


def test_lfc_sfc_precision():
    """Test LFC when there are precision issues with the parcel path."""
    levels = np.array([839., 819.4, 816., 807., 790.7, 763., 736.2,
                       722., 710.1, 700.]) * units.mbar
    temperatures = np.array([20.6, 22.3, 22.6, 22.2, 20.9, 18.7, 16.4,
                             15.2, 13.9, 12.8]) * units.celsius
    dewpoints = np.array([10.6, 8., 7.6, 6.2, 5.7, 4.7, 3.7, 3.2, 3., 2.8]) * units.celsius
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert assert_nan(lfc_pressure, levels.units)
    assert assert_nan(lfc_temp, temperatures.units)


def test_saturation_mixing_ratio():
    """Test saturation mixing ratio calculation."""
    p = 999. * units.mbar
    t = 288. * units.kelvin
    assert_almost_equal(saturation_mixing_ratio(p, t), .01068, 3)


def test_equivalent_potential_temperature():
    """Test equivalent potential temperature calculation."""
    p = 1000 * units.mbar
    t = 293. * units.kelvin
    td = 280. * units.kelvin
    ept = equivalent_potential_temperature(p, t, td)
    assert_almost_equal(ept, 311.18586467284007 * units.kelvin, 3)


def test_saturation_equivalent_potential_temperature():
    """Test saturation equivalent potential temperature calculation."""
    p = 700 * units.mbar
    t = 263.15 * units.kelvin
    s_ept = saturation_equivalent_potential_temperature(p, t)
    # 299.096584 comes from equivalent_potential_temperature(p,t,t)
    # where dewpoint and temperature are equal, which means saturations.
    assert_almost_equal(s_ept, 299.096584 * units.kelvin, 3)


def test_virtual_temperature():
    """Test virtual temperature calculation."""
    t = 288. * units.kelvin
    qv = .0016 * units.dimensionless  # kg/kg
    tv = virtual_temperature(t, qv)
    assert_almost_equal(tv, 288.2796 * units.kelvin, 3)


def test_virtual_potential_temperature():
    """Test virtual potential temperature calculation."""
    p = 999. * units.mbar
    t = 288. * units.kelvin
    qv = .0016 * units.dimensionless  # kg/kg
    theta_v = virtual_potential_temperature(p, t, qv)
    assert_almost_equal(theta_v, 288.3620 * units.kelvin, 3)


def test_density():
    """Test density calculation."""
    p = 999. * units.mbar
    t = 288. * units.kelvin
    qv = .0016 * units.dimensionless  # kg/kg
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


def test_el_ml():
    """Test equilibrium layer calculation for a mixed parcel."""
    levels = np.array([959., 779.2, 751.3, 724.3, 700., 400., 269.]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12., 9.4, 7., -25., -35.]) * units.celsius
    dewpoints = np.array([19., -11.2, -10.8, -10.4, -10., -35., -53.2]) * units.celsius
    __, T_mixed, Td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = parcel_profile(levels, T_mixed, Td_mixed)
    el_pressure, el_temperature = el(levels, temperatures, dewpoints, mixed_parcel_prof)
    assert_almost_equal(el_pressure, 355.834 * units.mbar, 3)
    assert_almost_equal(el_temperature, -28.371 * units.degC, 3)


def test_no_el():
    """Test equilibrium layer calculation when there is no EL in the data."""
    levels = np.array([959., 867.9, 779.2, 647.5, 472.5, 321.9, 251.]) * units.mbar
    temperatures = np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5]) * units.celsius
    dewpoints = np.array([19., 14.3, -11.2, -16.7, -21., -43.3, -56.7]) * units.celsius
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert assert_nan(el_pressure, levels.units)
    assert assert_nan(el_temperature, temperatures.units)


def test_no_el_multi_crossing():
    """Test el calculation with no el and severel parcel path-profile crossings."""
    levels = np.array([918., 911., 880., 873.9, 850., 848., 843.5, 818., 813.8, 785.,
                       773., 763., 757.5, 730.5, 700., 679., 654.4, 645.,
                       643.9]) * units.mbar
    temperatures = np.array([24.2, 22.8, 19.6, 19.1, 17., 16.8, 16.5, 15., 14.9, 14.4, 16.4,
                             16.2, 15.7, 13.4, 10.6, 8.4, 5.7, 4.6, 4.5]) * units.celsius
    dewpoints = np.array([19.5, 17.8, 16.7, 16.5, 15.8, 15.7, 15.3, 13.1, 12.9, 11.9, 6.4,
                          3.2, 2.6, -0.6, -4.4, -6.6, -9.3, -10.4, -10.5]) * units.celsius
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


def test_el_small_surface_instability():
    """Test that no EL is found when there is a small pocket of instability at the sfc."""
    levels = np.array([959., 931.3, 925., 899.3, 892., 867.9, 850., 814.,
                       807.9, 790., 779.2, 751.3, 724.3, 700., 655., 647.5,
                       599.4, 554.7, 550., 500.]) * units.mbar
    temperatures = np.array([22.2, 20.2, 19.8, 18.4, 18., 17.4, 17., 15.4, 15.4,
                             15.6, 14.6, 12., 9.4, 7., 2.2, 1.4, -4.2, -9.7,
                             -10.3, -14.9]) * units.degC
    dewpoints = np.array([20., 18.5, 18.1, 17.9, 17.8, 15.3, 13.5, 6.4, 2.2,
                          -10.4, -10.2, -9.8, -9.4, -9., -15.8, -15.7, -14.8, -14.,
                          -13.9, -17.9]) * units.degC
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert assert_nan(el_pressure, levels.units)
    assert assert_nan(el_temperature, temperatures.units)


def test_no_el_parcel_colder():
    """Test no EL when parcel stays colder than environment. INL 20170925-12Z."""
    levels = np.array([974., 946., 925., 877.2, 866., 850., 814.6, 785.,
                       756.6, 739., 729.1, 700., 686., 671., 641., 613.,
                       603., 586., 571., 559.3, 539., 533., 500., 491.,
                       477.9, 413., 390., 378., 345., 336.]) * units.mbar
    temperatures = np.array([10., 8.4, 7.6, 5.9, 7.2, 7.6, 6.8, 7.1, 7.7,
                             7.8, 7.7, 5.6, 4.6, 3.4, 0.6, -0.9, -1.1, -3.1,
                             -4.7, -4.7, -6.9, -7.5, -11.1, -10.9, -12.1, -20.5, -23.5,
                             -24.7, -30.5, -31.7]) * units.celsius
    dewpoints = np.array([8.9, 8.4, 7.6, 5.9, 7.2, 7., 5., 3.6, 0.3,
                          -4.2, -12.8, -12.4, -8.4, -8.6, -6.4, -7.9, -11.1, -14.1,
                          -8.8, -28.1, -18.9, -14.5, -15.2, -15.1, -21.6, -41.5, -45.5,
                          -29.6, -30.6, -32.1]) * units.celsius
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert assert_nan(el_pressure, levels.units)
    assert assert_nan(el_temperature, temperatures.units)


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


def test_mixing_ratio_from_relative_humidity():
    """Test relative humidity from mixing ratio."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    rh = 81.7219 * units.percent
    w = mixing_ratio_from_relative_humidity(rh, temperature, p)
    assert_almost_equal(w, 0.012 * units.dimensionless, 3)


def test_rh_mixing_ratio():
    """Test relative humidity from mixing ratio."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    w = 0.012 * units.dimensionless
    rh = relative_humidity_from_mixing_ratio(w, temperature, p)
    assert_almost_equal(rh, 81.7219 * units.percent, 3)


def test_mixing_ratio_from_specific_humidity():
    """Test mixing ratio from specific humidity."""
    q = 0.012 * units.dimensionless
    w = mixing_ratio_from_specific_humidity(q)
    assert_almost_equal(w, 0.01215, 3)


def test_specific_humidity_from_mixing_ratio():
    """Test specific humidity from mixing ratio."""
    w = 0.01215 * units.dimensionless
    q = specific_humidity_from_mixing_ratio(w)
    assert_almost_equal(q, 0.01200, 5)


def test_rh_specific_humidity():
    """Test relative humidity from specific humidity."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    q = 0.012 * units.dimensionless
    rh = relative_humidity_from_specific_humidity(q, temperature, p)
    assert_almost_equal(rh, 82.7145 * units.percent, 3)


def test_cape_cin():
    """Test the basic CAPE and CIN calculation."""
    p = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0])
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 58.0368212 * units('joule / kilogram'), 6)
    assert_almost_equal(cin, -89.8073512 * units('joule / kilogram'), 6)


def test_cape_cin_no_el():
    """Test that CAPE works with no EL."""
    p = np.array([959., 779.2, 751.3, 724.3]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]).to('degC')
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 0.08750805 * units('joule / kilogram'), 6)
    assert_almost_equal(cin, -89.8073512 * units('joule / kilogram'), 6)


def test_cape_cin_no_lfc():
    """Test that CAPE is zero with no LFC."""
    p = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 24.6, 22., 20.4, 18., -10.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]).to('degC')
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 0.0 * units('joule / kilogram'), 6)
    assert_almost_equal(cin, 0.0 * units('joule / kilogram'), 6)


def test_find_append_zero_crossings():
    """Test finding and appending zero crossings of an x, y series."""
    x = np.arange(11) * units.hPa
    y = np.array([3, 2, 1, -1, 2, 2, 0, 1, 0, -1, 2]) * units.degC
    x2, y2 = _find_append_zero_crossings(x, y)

    x_truth = np.array([0., 1., 2., 2.5, 3., 3.33333333, 4., 5.,
                        6., 7., 8., 9., 9.33333333, 10.]) * units.hPa
    y_truth = np.array([3, 2, 1, 0, -1, 0, 2, 2, 0, 1, 0, -1, 0, 2]) * units.degC
    assert_array_almost_equal(x2, x_truth, 6)
    assert_almost_equal(y2, y_truth, 6)


def test_most_unstable_parcel():
    """Test calculating the most unstable parcel."""
    levels = np.array([1000., 959., 867.9]) * units.mbar
    temperatures = np.array([18.2, 22.2, 17.4]) * units.celsius
    dewpoints = np.array([19., 19., 14.3]) * units.celsius
    ret = most_unstable_parcel(levels, temperatures, dewpoints, depth=100 * units.hPa)
    assert_almost_equal(ret[0], 959.0 * units.hPa, 6)
    assert_almost_equal(ret[1], 22.2 * units.degC, 6)
    assert_almost_equal(ret[2], 19.0 * units.degC, 6)


def test_isentropic_pressure():
    """Test calculation of isentropic pressure function."""
    lev = [100000., 95000., 90000., 85000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.
    tmp[1, :] = 292.
    tmp[2, :] = 290
    tmp[3, :] = 288.
    tmpk = tmp * units.kelvin
    isentlev = [296.] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk)
    trueprs = 1000. * units.hPa
    assert_almost_equal(isentprs[0].shape, (1, 5, 5), 3)
    assert_almost_equal(isentprs[0], trueprs, 3)


def test_isentropic_pressure_p_increase():
    """Test calculation of isentropic pressure function, p increasing order."""
    lev = [85000, 90000., 95000., 100000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 288.
    tmp[1, :] = 290.
    tmp[2, :] = 292.
    tmp[3, :] = 296.
    tmpk = tmp * units.kelvin
    isentlev = [296.] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk)
    trueprs = 1000. * units.hPa
    assert_almost_equal(isentprs[0], trueprs, 3)


def test_isentropic_pressure_adition_args():
    """Test calculation of isentropic pressure function, additional args."""
    lev = [100000., 95000., 90000., 85000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.
    tmp[1, :] = 292.
    tmp[2, :] = 290.
    tmp[3, :] = 288.
    rh = np.ones((4, 5, 5))
    rh[0, :] = 100.
    rh[1, :] = 80.
    rh[2, :] = 40.
    rh[3, :] = 20.
    relh = rh * units.percent
    tmpk = tmp * units.kelvin
    isentlev = [296.] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, relh)
    truerh = 100. * units.percent
    assert_almost_equal(isentprs[1], truerh, 3)


def test_isentropic_pressure_tmp_out():
    """Test calculation of isentropic pressure function, temperature output."""
    lev = [100000., 95000., 90000., 85000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.
    tmp[1, :] = 292.
    tmp[2, :] = 290.
    tmp[3, :] = 288.
    tmpk = tmp * units.kelvin
    isentlev = [296.] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, tmpk_out=True)
    truetmp = 296. * units.kelvin
    assert_almost_equal(isentprs[1], truetmp, 3)


def test_isentropic_pressure_p_increase_rh_out():
    """Test calculation of isentropic pressure function, p increasing order."""
    lev = [85000., 90000., 95000., 100000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 288.
    tmp[1, :] = 290.
    tmp[2, :] = 292.
    tmp[3, :] = 296.
    tmpk = tmp * units.kelvin
    rh = np.ones((4, 5, 5))
    rh[0, :] = 20.
    rh[1, :] = 40.
    rh[2, :] = 80.
    rh[3, :] = 100.
    relh = rh * units.percent
    isentlev = 296. * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, relh)
    truerh = 100. * units.percent
    assert_almost_equal(isentprs[1], truerh, 3)


def test_isentropic_pressure_interp():
    """Test calculation of isentropic pressure function."""
    lev = [100000., 95000., 90000., 85000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.
    tmp[1, :] = 292.
    tmp[2, :] = 290
    tmp[3, :] = 288.
    tmpk = tmp * units.kelvin
    isentlev = [296., 297] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk)
    trueprs = 936.18057 * units.hPa
    assert_almost_equal(isentprs[0][1], trueprs, 3)


def test_isentropic_pressure_adition_args_interp():
    """Test calculation of isentropic pressure function, additional args."""
    lev = [100000., 95000., 90000., 85000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.
    tmp[1, :] = 292.
    tmp[2, :] = 290.
    tmp[3, :] = 288.
    rh = np.ones((4, 5, 5))
    rh[0, :] = 100.
    rh[1, :] = 80.
    rh[2, :] = 40.
    rh[3, :] = 20.
    relh = rh * units.percent
    tmpk = tmp * units.kelvin
    isentlev = [296., 297.] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, relh)
    truerh = 69.171 * units.percent
    assert_almost_equal(isentprs[1][1], truerh, 3)


def test_isentropic_pressure_tmp_out_interp():
    """Test calculation of isentropic pressure function, temperature output."""
    lev = [100000., 95000., 90000., 85000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.
    tmp[1, :] = 292.
    tmp[2, :] = 290.
    tmp[3, :] = 288.
    tmpk = tmp * units.kelvin
    isentlev = [296., 297.] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, tmpk_out=True)
    truetmp = 291.4579 * units.kelvin
    assert_almost_equal(isentprs[1][1], truetmp, 3)


def test_isentropic_pressure_data_bounds_error():
    """Test calculation of isentropic pressure function, error for data out of bounds."""
    lev = [100000., 95000., 90000., 85000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.
    tmp[1, :] = 292.
    tmp[2, :] = 290.
    tmp[3, :] = 288.
    tmpk = tmp * units.kelvin
    isentlev = [296., 350.] * units.kelvin
    with pytest.raises(ValueError):
        isentropic_interpolation(isentlev, lev, tmpk)


def test_isentropic_pressure_4d():
    """Test calculation of isentropic pressure function."""
    lev = [100000., 95000., 90000., 85000.] * units.Pa
    tmp = np.ones((3, 4, 5, 5))
    tmp[:, 0, :] = 296.
    tmp[:, 1, :] = 292.
    tmp[:, 2, :] = 290
    tmp[:, 3, :] = 288.
    tmpk = tmp * units.kelvin
    rh = np.ones((3, 4, 5, 5))
    rh[:, 0, :] = 100.
    rh[:, 1, :] = 80.
    rh[:, 2, :] = 40.
    rh[:, 3, :] = 20.
    relh = rh * units.percent
    isentlev = [296., 297., 300.] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, relh, axis=1)
    trueprs = 1000. * units.hPa
    trueprs2 = 936.18057 * units.hPa
    trueprs3 = 879.446 * units.hPa
    truerh = 69.171 * units.percent
    assert_almost_equal(isentprs[0].shape, (3, 3, 5, 5), 3)
    assert_almost_equal(isentprs[0][:, 0, :], trueprs, 3)
    assert_almost_equal(isentprs[0][:, 1, :], trueprs2, 3)
    assert_almost_equal(isentprs[0][:, 2, :], trueprs3, 3)
    assert_almost_equal(isentprs[1][:, 1, ], truerh, 3)


def test_surface_based_cape_cin():
    """Test the surface-based CAPE and CIN calculation."""
    p = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    cape, cin = surface_based_cape_cin(p, temperature, dewpoint)
    assert_almost_equal(cape, 58.0368212 * units('joule / kilogram'), 6)
    assert_almost_equal(cin, -89.8073512 * units('joule / kilogram'), 6)


def test_most_unstable_cape_cin_surface():
    """Test the most unstable CAPE/CIN calculation when surface is most unstable."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    assert_almost_equal(mucape, 58.0368212 * units('joule / kilogram'), 6)
    assert_almost_equal(mucin, -89.8073512 * units('joule / kilogram'), 6)


def test_most_unstable_cape_cin():
    """Test the most unstable CAPE/CIN calculation."""
    pressure = np.array([1000., 959., 867.9, 850., 825., 800.]) * units.mbar
    temperature = np.array([18.2, 22.2, 17.4, 10., 0., 15]) * units.celsius
    dewpoint = np.array([19., 19., 14.3, 0., -10., 0.]) * units.celsius
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    assert_almost_equal(mucape, 157.07111 * units('joule / kilogram'), 4)
    assert_almost_equal(mucin, -15.74772 * units('joule / kilogram'), 4)


def test_mixed_parcel():
    """Test the mixed parcel calculation."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.degC
    parcel_pressure, parcel_temperature, parcel_dewpoint = mixed_parcel(pressure, temperature,
                                                                        dewpoint,
                                                                        depth=250 * units.hPa)
    assert_almost_equal(parcel_pressure, 959. * units.hPa, 6)
    assert_almost_equal(parcel_temperature, 28.7363771 * units.degC, 6)
    assert_almost_equal(parcel_dewpoint, 7.1534658 * units.degC, 6)


def test_mixed_layer():
    """Test the mixed layer calculation."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    mixed_layer_temperature = mixed_layer(pressure, temperature, depth=250 * units.hPa)[0]
    assert_almost_equal(mixed_layer_temperature, 16.4024930 * units.degC, 6)


def test_dry_static_energy():
    """Test the dry static energy calculation."""
    dse = dry_static_energy(1000 * units.m, 25 * units.degC)
    assert_almost_equal(dse, 309.4474 * units('kJ/kg'), 6)


def test_moist_static_energy():
    """Test the moist static energy calculation."""
    mse = moist_static_energy(1000 * units.m, 25 * units.degC, 0.012 * units.dimensionless)
    assert_almost_equal(mse, 339.4594 * units('kJ/kg'), 6)


def test_thickness_hydrostatic():
    """Test the thickness calculation for a moist layer."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    mixing = np.array([0.01458, 0.00209, 0.00224, 0.00240, 0.00256, 0.00010])
    thickness = thickness_hydrostatic(pressure, temperature, mixing=mixing)
    assert_almost_equal(thickness, 9892.07 * units.m, 2)


def test_thickness_hydrostatic_subset():
    """Test the thickness calculation with a subset of the moist layer."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    mixing = np.array([0.01458, 0.00209, 0.00224, 0.00240, 0.00256, 0.00010])
    thickness = thickness_hydrostatic(pressure, temperature, mixing=mixing,
                                      bottom=850 * units.hPa, depth=150 * units.hPa)
    assert_almost_equal(thickness, 1630.81 * units.m, 2)


def test_thickness_hydrostatic_isothermal():
    """Test the thickness calculation for a dry isothermal layer at 0 degC."""
    pressure = np.arange(1000, 500 - 1e-10, -10) * units.hPa
    temperature = np.zeros_like(pressure) * units.degC
    thickness = thickness_hydrostatic(pressure, temperature)
    assert_almost_equal(thickness, 5542.12 * units.m, 2)


def test_thickness_hydrostatic_isothermal_subset():
    """Test the thickness calculation for a dry isothermal layer subset at 0 degC."""
    pressure = np.arange(1000, 500 - 1e-10, -10) * units.hPa
    temperature = np.zeros_like(pressure) * units.degC
    thickness = thickness_hydrostatic(pressure, temperature, bottom=850 * units.hPa,
                                      depth=350 * units.hPa)
    assert_almost_equal(thickness, 4242.68 * units.m, 2)


def test_thickness_hydrostatic_from_relative_humidity():
    """Test the thickness calculation for a moist layer using RH data."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    relative_humidity = np.array([81.69, 15.43, 18.95, 23.32, 28.36, 18.55]) * units.percent
    thickness = thickness_hydrostatic_from_relative_humidity(pressure, temperature,
                                                             relative_humidity)
    assert_almost_equal(thickness, 9892.07 * units.m, 2)


def test_mixing_ratio_dimensions():
    """Verify mixing ratio returns a dimensionless number."""
    p = 998. * units.mbar
    e = 73.75 * units.hPa
    assert str(mixing_ratio(e, p).units) == 'dimensionless'


def test_saturation_mixing_ratio_dimensions():
    """Verify saturation mixing ratio returns a dimensionless number."""
    p = 998. * units.mbar
    temp = 20 * units.celsius
    assert str(saturation_mixing_ratio(p, temp).units) == 'dimensionless'


def test_mixing_ratio_from_rh_dimensions():
    """Verify mixing ratio from RH returns a dimensionless number."""
    p = 1000. * units.mbar
    temperature = 0. * units.degC
    rh = 100. * units.percent
    assert (str(mixing_ratio_from_relative_humidity(rh, temperature, p).units) ==
            'dimensionless')


@pytest.fixture
def bv_data():
    """Return height and potential temperature data for testing Brunt-Vaisala functions."""
    heights = [1000., 1500., 2000., 2500.] * units('m')
    potential_temperatures = [[290., 290., 290., 290.],
                              [292., 293., 293., 292.],
                              [294., 296., 293., 293.],
                              [296., 295., 293., 296.]] * units('K')
    return heights, potential_temperatures


def test_brunt_vaisala_frequency_squared():
    """Test Brunt-Vaisala frequency squared function."""
    truth = [[1.35264138e-04, 2.02896207e-04, 3.04344310e-04, 1.69080172e-04],
             [1.34337671e-04, 2.00818771e-04, 1.00409386e-04, 1.00753253e-04],
             [1.33423810e-04, 6.62611486e-05, 0, 1.33879181e-04],
             [1.32522297e-04, -1.99457288e-04, 0., 2.65044595e-04]] * units('s^-2')
    bv_freq_sqr = brunt_vaisala_frequency_squared(bv_data()[0], bv_data()[1])
    assert_almost_equal(bv_freq_sqr, truth, 6)


def test_brunt_vaisala_frequency():
    """Test Brunt-Vaisala frequency function."""
    truth = [[0.01163031, 0.01424416, 0.01744547, 0.01300308],
             [0.01159041, 0.01417105, 0.01002045, 0.01003759],
             [0.01155092, 0.00814010, 0., 0.01157062],
             [0.01151183, np.nan, 0., 0.01628019]] * units('s^-1')
    bv_freq = brunt_vaisala_frequency(bv_data()[0], bv_data()[1])
    assert_almost_equal(bv_freq, truth, 6)


def test_brunt_vaisala_period():
    """Test Brunt-Vaisala period function."""
    truth = [[540.24223556, 441.10593821, 360.16149037, 483.20734521],
             [542.10193894, 443.38165033, 627.03634320, 625.96540075],
             [543.95528431, 771.88106656, np.nan, 543.02940230],
             [545.80233643, np.nan, np.nan, 385.94053328]] * units('s')
    bv_period = brunt_vaisala_period(bv_data()[0], bv_data()[1])
    assert_almost_equal(bv_period, truth, 6)


def test_wet_bulb_temperature():
    """Test wet bulb calculation with scalars."""
    val = wet_bulb_temperature(1000 * units.hPa, 25 * units.degC, 15 * units.degC)
    truth = 18.34345936 * units.degC  # 18.59 from NWS calculator
    assert_almost_equal(val, truth)


def test_wet_bulb_temperature_1d():
    """Test wet bulb calculation with 1d list."""
    pressures = [1013, 1000, 990] * units.hPa
    temperatures = [25, 20, 15] * units.degC
    dewpoints = [20, 15, 10] * units.degC
    val = wet_bulb_temperature(pressures, temperatures, dewpoints)
    truth = [21.4449794, 16.7368576, 12.0656909] * units.degC
    # 21.58, 16.86, 12.18 from NWS Calculator
    assert_array_almost_equal(val, truth)


def test_wet_bulb_temperature_2d():
    """Test wet bulb calculation with 2d list."""
    pressures = [[1013, 1000, 990],
                 [1012, 999, 989]] * units.hPa
    temperatures = [[25, 20, 15],
                    [24, 19, 14]] * units.degC
    dewpoints = [[20, 15, 10],
                 [19, 14, 9]] * units.degC
    val = wet_bulb_temperature(pressures, temperatures, dewpoints)
    truth = [[21.4449794, 16.7368576, 12.0656909],
             [20.5021631, 15.801218, 11.1361878]] * units.degC
    # 21.58, 16.86, 12.18
    # 20.6, 15.9, 11.2 from NWS Calculator
    assert_array_almost_equal(val, truth)


def test_static_stability_adiabatic():
    """Test static stability calculation with a dry adiabatic profile."""
    pressures = [1000., 900., 800., 700., 600., 500.] * units.hPa
    temperature_start = 20 * units.degC
    temperatures = dry_lapse(pressures, temperature_start)
    sigma = static_stability(pressures, temperatures)
    truth = np.zeros_like(pressures) * units('J kg^-1 hPa^-2')
    # Should be zero with a dry adiabatic profile
    assert_almost_equal(sigma, truth, 6)


def test_static_stability_cross_section():
    """Test static stability calculation with a 2D cross-section."""
    pressures = [[850., 700., 500.],
                 [850., 700., 500.],
                 [850., 700., 500.]] * units.hPa
    temperatures = [[17., 11., -10.],
                    [16., 10., -11.],
                    [11., 6., -12.]] * units.degC
    sigma = static_stability(pressures, temperatures, axis=1)
    truth = [[0.02819452, 0.02016804, 0.00305262],
             [0.02808841, 0.01999462, 0.00274956],
             [0.02840196, 0.02366708, 0.0131604]] * units('J kg^-1 hPa^-2')
    assert_almost_equal(sigma, truth, 6)


def test_dewpoint_specific_humidity():
    """Test relative humidity from specific humidity."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    q = 0.012 * units.dimensionless
    td = dewpoint_from_specific_humidity(q, temperature, p)
    assert_almost_equal(td, 16.973 * units.degC, 3)


def test_lfc_not_below_lcl():
    """Test sounding where LFC appears to be (but isn't) below LCL."""
    levels = np.array([1002.5, 1001.7, 1001., 1000.3, 999.7, 999., 998.2, 977.9,
                       966.2, 952.3, 940.6, 930.5, 919.8, 909.1, 898.9, 888.4,
                       878.3, 868.1, 858., 848., 837.2, 827., 816.7, 805.4]) * units.hPa
    temperatures = np.array([17.9, 17.9, 17.8, 17.7, 17.7, 17.6, 17.5, 16.,
                             15.2, 14.5, 13.8, 13., 12.5, 11.9, 11.4, 11.,
                             10.3, 9.7, 9.2, 8.7, 8., 7.4, 6.8, 6.1]) * units.degC
    dewpoints = np.array([13.6, 13.6, 13.5, 13.5, 13.5, 13.5, 13.4, 12.5,
                          12.1, 11.8, 11.4, 11.3, 11., 9.3, 10., 8.7, 8.9,
                          8.6, 8.1, 7.6, 7., 6.5, 6., 5.4]) * units.degC
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    # Before patch, LFC pressure would show 1000.5912165339967 hPa
    assert_almost_equal(lfc_pressure, 811.8456357 * units.mbar, 6)
    assert_almost_equal(lfc_temp, 6.4992871 * units.celsius, 6)


def test_dcape_defaults():
    """Test DCAPE with the default behavior."""
    pressure = np.array([973, 943.5, 925, 910.6, 878.4, 865, 850, 848, 847.2, 816.9,
                        793, 787.8, 759.6, 759, 732.2, 700, 654.5]) * units.hPa
    temperature = np.array([23.4, 20.4, 18.4, 17.1, 14.1, 12.8, 13.2, 13.4, 13.4,
                           13.1, 12.8, 12.7, 12.4, 12.4, 10, 7, 3.4]) * units.degC
    dewpoint = np.array([5.4, 3.6, 2.4, 1.8, 0.4, -0.2, 0.2, 0.4, 0.2, -5.5, -10.2,
                         -11.9, -21.4, -21.6, -21.8, -22, -21.4]) * units.degC
    dcape, dcape_pressure, dcape_temperature = downdraft_cape(pressure, temperature, dewpoint)
    dcape_truth = 74.11506371089433 * units.joule / units.kilogram
    dcape_pressure_truth = np.array([973, 943.5, 925, 910.6, 878.4]) * units.hPa
    dcape_temperature_truth = np.array([17.95718657, 16.80836487,
                                        16.06406211, 15.47121721, 14.1]) * units.degC
    assert_almost_equal(dcape, dcape_truth, 6)
    assert_almost_equal(dcape_pressure, dcape_pressure_truth, 6)
    assert_almost_equal(dcape_temperature, dcape_temperature_truth, 6)


def test_dcape_custom_parcel_start():
    """Test DCAPE with a custom parcel starting point."""
    pressure = np.array([973, 943.5, 925, 910.6, 878.4, 865, 850, 848, 847.2, 816.9,
                        793, 787.8, 759.6, 759, 732.2, 700, 654.5]) * units.hPa
    temperature = np.array([23.4, 20.4, 18.4, 17.1, 14.1, 12.8, 13.2, 13.4, 13.4,
                           13.1, 12.8, 12.7, 12.4, 12.4, 10, 7, 3.4]) * units.degC
    dewpoint = np.array([5.4, 3.6, 2.4, 1.8, 0.4, -0.2, 0.2, 0.4, 0.2, -5.5, -10.2,
                         -11.9, -21.4, -21.6, -21.8, -22, -21.4]) * units.degC
    custom_parcel = (670 * units.hPa, 3.5 * units.degC)
    dcape, dcape_pressure, dcape_temperature = downdraft_cape(pressure, temperature, dewpoint,
                                                              parcel=custom_parcel)
    dcape_truth = 101.56717405359117 * units.joule / units.kilogram
    dcape_pressure_truth = np.array([973, 943.5, 925, 910.6, 878.4, 865, 850, 848, 847.2,
                                     816.9, 793, 787.8, 759.6, 759, 732.2, 700, 670]) * units.hPa
    dcape_temperature_truth = np.array([19.0633538, 17.93772489, 17.20885559, 16.62853588,
                                        15.2870874, 14.70992377, 14.04982066, 13.96065181,
                                        13.92490675, 12.53726347, 11.39336396, 11.13832389,
                                        9.71426456, 9.68318326, 8.25933429, 6.44940391,
                                        4.65373642]) * units.degC
    assert_almost_equal(dcape, dcape_truth, 6)
    assert_almost_equal(dcape_pressure, dcape_pressure_truth, 6)
    assert_almost_equal(dcape_temperature, dcape_temperature_truth, 6)
