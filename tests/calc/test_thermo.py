# Copyright (c) 2008,2015,2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `thermo` module."""

import warnings

import numpy as np
import pytest
import xarray as xr

from metpy.calc import (brunt_vaisala_frequency, brunt_vaisala_frequency_squared,
                        brunt_vaisala_period, cape_cin, ccl, cross_totals, density, dewpoint,
                        dewpoint_from_relative_humidity, dewpoint_from_specific_humidity,
                        dry_lapse, dry_static_energy, el, equivalent_potential_temperature,
                        exner_function, gradient_richardson_number, InvalidSoundingError,
                        isentropic_interpolation, isentropic_interpolation_as_dataset, k_index,
                        lcl, lfc, lifted_index, mixed_layer, mixed_layer_cape_cin,
                        mixed_parcel, mixing_ratio, mixing_ratio_from_relative_humidity,
                        mixing_ratio_from_specific_humidity, moist_lapse, moist_static_energy,
                        most_unstable_cape_cin, most_unstable_parcel, parcel_profile,
                        parcel_profile_with_lcl, parcel_profile_with_lcl_as_dataset,
                        potential_temperature, psychrometric_vapor_pressure_wet,
                        relative_humidity_from_dewpoint, relative_humidity_from_mixing_ratio,
                        relative_humidity_from_specific_humidity,
                        relative_humidity_wet_psychrometric,
                        saturation_equivalent_potential_temperature, saturation_mixing_ratio,
                        saturation_vapor_pressure, showalter_index,
                        specific_humidity_from_dewpoint, specific_humidity_from_mixing_ratio,
                        static_stability, surface_based_cape_cin, sweat_index,
                        temperature_from_potential_temperature, thickness_hydrostatic,
                        thickness_hydrostatic_from_relative_humidity, total_totals_index,
                        vapor_pressure, vertical_totals, vertical_velocity,
                        vertical_velocity_pressure, virtual_potential_temperature,
                        virtual_temperature, wet_bulb_temperature)
from metpy.calc.thermo import _find_append_zero_crossings
from metpy.testing import assert_almost_equal, assert_array_almost_equal, assert_nan
from metpy.units import is_quantity, masked_array, units


def test_relative_humidity_from_dewpoint():
    """Test Relative Humidity calculation."""
    assert_almost_equal(relative_humidity_from_dewpoint(25. * units.degC, 15. * units.degC),
                        53.80 * units.percent, 2)


def test_relative_humidity_from_dewpoint_with_f():
    """Test Relative Humidity accepts temperature in Fahrenheit."""
    assert_almost_equal(relative_humidity_from_dewpoint(70. * units.degF, 55. * units.degF),
                        58.935 * units.percent, 3)


def test_relative_humidity_from_dewpoint_xarray():
    """Test Relative Humidity with xarray data arrays (quantified and unquantified)."""
    temp = xr.DataArray(25., attrs={'units': 'degC'})
    dewp = xr.DataArray([15.] * units.degC)
    assert_almost_equal(relative_humidity_from_dewpoint(temp, dewp), 53.80 * units.percent, 2)


def test_exner_function():
    """Test Exner function calculation."""
    pressure = np.array([900., 500., 300., 100.]) * units.mbar
    truth = np.array([0.97034558, 0.82033536, 0.70893444, 0.51794747]) * units.dimensionless
    assert_array_almost_equal(exner_function(pressure), truth, 5)


def test_potential_temperature():
    """Test potential_temperature calculation."""
    temp = np.array([278., 283., 291., 298.]) * units.kelvin
    pressure = np.array([900., 500., 300., 100.]) * units.mbar
    real_th = np.array([286.496, 344.981, 410.475, 575.348]) * units.kelvin
    assert_array_almost_equal(potential_temperature(pressure, temp), real_th, 3)


def test_temperature_from_potential_temperature():
    """Test temperature_from_potential_temperature calculation."""
    theta = np.array([286.12859679, 288.22362587, 290.31865495, 292.41368403]) * units.kelvin
    pressure = np.array([850] * 4) * units.mbar
    real_t = np.array([273.15, 275.15, 277.15, 279.15]) * units.kelvin
    assert_array_almost_equal(temperature_from_potential_temperature(pressure, theta),
                              real_t, 2)


def test_pot_temp_scalar():
    """Test potential_temperature accepts scalar values."""
    assert_almost_equal(potential_temperature(1000. * units.mbar, 293. * units.kelvin),
                        293. * units.kelvin, 4)
    assert_almost_equal(potential_temperature(800. * units.mbar, 293. * units.kelvin),
                        312.2886 * units.kelvin, 4)


def test_pot_temp_fahrenheit():
    """Test that potential_temperature handles temperature values in Fahrenheit."""
    assert_almost_equal(potential_temperature(800. * units.mbar, 68. * units.degF),
                        (312.444 * units.kelvin).to(units.degF), 2)


def test_pot_temp_inhg():
    """Test that potential_temperature can handle pressure not in mb (issue #165)."""
    assert_almost_equal(potential_temperature(29.92 * units.inHg, 29 * units.degC),
                        301.019 * units.kelvin, 3)


def test_dry_lapse():
    """Test dry_lapse calculation."""
    levels = np.array([1000, 900, 864.89]) * units.mbar
    temps = dry_lapse(levels, 303.15 * units.kelvin)
    assert_array_almost_equal(temps,
                              np.array([303.15, 294.16, 290.83]) * units.kelvin, 2)


def test_dry_lapse_2_levels():
    """Test dry_lapse calculation when given only two levels."""
    temps = dry_lapse(np.array([1000., 500.]) * units.mbar, 293. * units.kelvin)
    assert_array_almost_equal(temps, [293., 240.3583] * units.kelvin, 4)


@pytest.mark.parametrize('temp_units', ['degF', 'degC', 'K'])
def test_moist_lapse(temp_units):
    """Test moist_lapse with various temperature units."""
    starting_temp = 19.85 * units.degC
    temp = moist_lapse(np.array([1000., 800., 600., 500., 400.]) * units.mbar,
                       starting_temp.to(temp_units))
    true_temp = np.array([293, 284.64, 272.81, 264.42, 252.91]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_moist_lapse_ref_pressure():
    """Test moist_lapse with a reference pressure."""
    temp = moist_lapse(np.array([1050., 800., 600., 500., 400.]) * units.mbar,
                       19.85 * units.degC, 1000. * units.mbar)
    true_temp = np.array([294.76, 284.64, 272.81, 264.42, 252.91]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_moist_lapse_multiple_temps():
    """Test moist_lapse with multiple starting temperatures."""
    temp = moist_lapse(np.array([1050., 800., 600., 500., 400.]) * units.mbar,
                       np.array([19.85, np.nan, 19.85]) * units.degC, 1000. * units.mbar)
    true_temp = np.array([[294.76, 284.64, 272.81, 264.42, 252.91],
                          [np.nan, np.nan, np.nan, np.nan, np.nan],
                          [294.76, 284.64, 272.81, 264.42, 252.91]]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_moist_lapse_scalar():
    """Test moist_lapse when given a scalar desired pressure and a reference pressure."""
    temp = moist_lapse(np.array([800.]) * units.mbar, 19.85 * units.degC, 1000. * units.mbar)
    assert_almost_equal(temp, 284.64 * units.kelvin, 2)


def test_moist_lapse_close_start():
    """Test that we behave correctly with a reference pressure close to an actual pressure."""
    with warnings.catch_warnings(record=True) as record:
        temp = moist_lapse(units.Quantity(1000, 'hPa'), 0 * units.degC,
                           units.Quantity(1000., 'mbar'))
        assert len(record) == 0
    assert_almost_equal(temp, units.Quantity(0., 'degC'))


def test_moist_lapse_uniform():
    """Test moist_lapse when given a uniform array of pressures."""
    temp = moist_lapse(np.array([900., 900., 900.]) * units.hPa, 20. * units.degC)
    assert_almost_equal(temp, np.array([20., 20., 20.]) * units.degC, 7)


def test_moist_lapse_nan_temp():
    """Test moist_lapse when given nan for temperature."""
    temp = moist_lapse(40 * units.hPa, np.nan * units.degC, 400 * units.hPa)
    assert_nan(temp, units.degC)


def test_moist_lapse_nan_ref_press():
    """Test moist_lapse when given nans for reference pressure."""
    temp = moist_lapse(40 * units.hPa, -20 * units.degC, np.nan * units.hPa)
    assert_nan(temp, units.degC)


def test_moist_lapse_downwards():
    """Test moist_lapse when integrating downwards (#2128)."""
    temp = moist_lapse(units.Quantity([600, 700], 'mbar'), units.Quantity(0, 'degC'))
    assert_almost_equal(temp, units.Quantity([0, 6.47748353], units.degC), 4)


@pytest.mark.parametrize('direction', (1, -1))
@pytest.mark.parametrize('start', list(range(5)))
def test_moist_lapse_starting_points(start, direction):
    """Test moist_lapse with a variety of reference points."""
    truth = units.Quantity([20.0804315, 17.2333509, 14.0752659, 6.4774835, 0.0],
                           'degC')[::direction]
    pressure = units.Quantity([1000, 925, 850, 700, 600], 'hPa')[::direction]
    temp = moist_lapse(pressure, truth[start], pressure[start])
    assert_almost_equal(temp, truth, 4)


def test_parcel_profile():
    """Test parcel profile calculation."""
    levels = np.array([1000., 900., 800., 700., 600., 500., 400.]) * units.mbar
    true_prof = np.array([303.15, 294.16, 288.026, 283.073, 277.058, 269.402,
                          258.966]) * units.kelvin

    prof = parcel_profile(levels, 30. * units.degC, 20. * units.degC)
    assert_array_almost_equal(prof, true_prof, 2)


def test_parcel_profile_lcl():
    """Test parcel profile with lcl calculation."""
    p = np.array([1004., 1000., 943., 928., 925., 850., 839., 749., 700., 699.]) * units.hPa
    t = np.array([24.2, 24., 20.2, 21.6, 21.4, 20.4, 20.2, 14.4, 13.2, 13.]) * units.degC
    td = np.array([21.9, 22.1, 19.2, 20.5, 20.4, 18.4, 17.4, 8.4, -2.8, -3.0]) * units.degC

    true_prof = np.array([297.35, 297.01, 294.5, 293.48, 292.92, 292.81, 289.79, 289.32,
                          285.15, 282.59, 282.53]) * units.kelvin
    true_p = np.insert(p.m, 2, 970.711) * units.mbar
    true_t = np.insert(t.m, 2, 22.047) * units.degC
    true_td = np.insert(td.m, 2, 20.609) * units.degC

    pressure, temp, dewp, prof = parcel_profile_with_lcl(p, t, td)
    assert_almost_equal(pressure, true_p, 3)
    assert_almost_equal(temp, true_t, 3)
    assert_almost_equal(dewp, true_td, 3)
    assert_array_almost_equal(prof, true_prof, 2)


def test_parcel_profile_lcl_not_monotonic():
    """Test parcel profile with lcl calculation."""
    with pytest.raises(InvalidSoundingError):
        p = np.array([1004., 1000., 943., 925., 928., 850., 839., 749., 700.]) * units.hPa
        t = np.array([24.2, 24., 20.2, 21.6, 21.4, 20.4, 20.2, 14.4, 13.2]) * units.degC
        td = np.array([21.9, 22.1, 19.2, 20.5, 20.4, 18.4, 17.4, 8.4, -2.8]) * units.degC

        _ = parcel_profile_with_lcl(p, t, td)


def test_parcel_profile_with_lcl_as_dataset():
    """Test parcel profile with lcl calculation with xarray."""
    p = np.array([1004., 1000., 943., 928., 925., 850., 839., 749., 700., 699.]) * units.hPa
    t = np.array([24.2, 24., 20.2, 21.6, 21.4, 20.4, 20.2, 14.4, 13.2, 13.]) * units.degC
    td = np.array([21.9, 22.1, 19.2, 20.5, 20.4, 18.4, 17.4, 8.4, -2.8, -3.0]) * units.degC

    result = parcel_profile_with_lcl_as_dataset(p, t, td)

    expected = xr.Dataset(
        {
            'ambient_temperature': (
                ('isobaric',),
                np.insert(t.m, 2, 22.047) * units.degC,
                {'standard_name': 'air_temperature'}
            ),
            'ambient_dew_point': (
                ('isobaric',),
                np.insert(td.m, 2, 20.609) * units.degC,
                {'standard_name': 'dew_point_temperature'}
            ),
            'parcel_temperature': (
                ('isobaric',),
                [
                    297.35, 297.01, 294.5, 293.48, 292.92, 292.81, 289.79, 289.32, 285.15,
                    282.59, 282.53
                ] * units.kelvin,
                {'long_name': 'air_temperature_of_lifted_parcel'}
            )
        },
        coords={
            'isobaric': (
                'isobaric',
                np.insert(p.m, 2, 970.699),
                {'units': 'hectopascal', 'standard_name': 'air_pressure'}
            )
        }
    )
    xr.testing.assert_allclose(result, expected, atol=1e-2)
    for field in (
        'ambient_temperature',
        'ambient_dew_point',
        'parcel_temperature',
        'isobaric'
    ):
        assert result[field].attrs == expected[field].attrs


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


def test_basic_dewpoint_from_relative_humidity():
    """Test dewpoint_from_relative_humidity function."""
    temp = np.array([30., 25., 10., 20., 25.]) * units.degC
    rh = np.array([30., 45., 55., 80., 85.]) / 100.

    real_td = np.array([11, 12, 1, 16, 22]) * units.degC
    assert_array_almost_equal(real_td, dewpoint_from_relative_humidity(temp, rh), 0)


def test_scalar_dewpoint_from_relative_humidity():
    """Test dewpoint_from_relative_humidity with scalar values."""
    td = dewpoint_from_relative_humidity(10.6 * units.degC, 0.37)
    assert_almost_equal(td, 26. * units.degF, 0)


def test_percent_dewpoint_from_relative_humidity():
    """Test dewpoint_from_relative_humidity with rh in percent."""
    td = dewpoint_from_relative_humidity(10.6 * units.degC, 37 * units.percent)
    assert_almost_equal(td, 26. * units.degF, 0)


def test_warning_dewpoint_from_relative_humidity():
    """Test that warning is raised for >120% RH."""
    with pytest.warns(UserWarning):
        dewpoint_from_relative_humidity(10.6 * units.degC, 50)


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
                        73.75179 * units.mbar, 5)


def test_lcl():
    """Test LCL calculation."""
    lcl_pressure, lcl_temperature = lcl(1000. * units.mbar, 30. * units.degC, 20. * units.degC)
    assert_almost_equal(lcl_pressure, 864.806 * units.mbar, 2)
    assert_almost_equal(lcl_temperature, 17.676 * units.degC, 2)


def test_lcl_kelvin():
    """Test LCL temperature is returned as Kelvin, if temperature is Kelvin."""
    temperature = 273.09723 * units.kelvin
    lcl_pressure, lcl_temperature = lcl(1017.16 * units.mbar, temperature,
                                        264.5351 * units.kelvin)
    assert_almost_equal(lcl_pressure, 889.459 * units.mbar, 2)
    assert_almost_equal(lcl_temperature, 262.827 * units.kelvin, 2)
    assert lcl_temperature.units == temperature.units


def test_lcl_convergence():
    """Test LCL calculation convergence failure."""
    with pytest.raises(RuntimeError):
        lcl(1000. * units.mbar, 30. * units.degC, 20. * units.degC, max_iters=2)


def test_lcl_nans():
    """Test LCL calculation on data with nans."""
    press = np.array([900., 900., 900., 900.]) * units.hPa
    temp = np.array([np.nan, 25., 25., 25.]) * units.degC
    dewp = np.array([20., 20., np.nan, 20.]) * units.degC
    lcl_press, lcl_temp = lcl(press, temp, dewp)

    assert_array_almost_equal(lcl_press, np.array([np.nan, 836.4098648012595,
                                                   np.nan, 836.4098648012595]) * units.hPa)
    assert_array_almost_equal(lcl_temp, np.array([np.nan, 18.82281982535794,
                                                  np.nan, 18.82281982535794]) * units.degC)


def test_ccl_basic():
    """First test of CCL calculation. Data: ILX, June 17 2022 00Z."""
    pressure = np.array([993.0, 984.0, 957.0, 948.0, 925.0, 917.0, 886.0, 868.0, 850.0,
                         841.0, 813.0, 806.0, 798.0, 738.0, 732.0, 723.0, 716.0, 711.0,
                         700.0, 623.0, 621.0, 582.0, 541.0, 500.0, 468.0]) * units.mbar
    temperature = np.array([34.6, 33.7, 31.1, 30.1, 27.8, 27.1, 24.3, 22.6, 21.4,
                            20.8, 19.6, 19.4, 18.7, 13.0, 13.0, 13.4, 13.5, 13.6,
                            13.0, 5.2, 5.0, 1.5, -2.4, -6.7, -10.7]) * units.degC
    dewpoint = np.array([19.6, 19.4, 18.7, 18.4, 17.8, 17.5, 16.3, 15.6, 12.4, 10.8,
                         -0.4, -3.6, -3.8, -5.0, -6.0, -15.6, -13.2, -11.4, -11.0,
                         -5.8, -6.2, -14.8, -24.3, -34.7, -38.1]) * units.degC
    ccl_p, ccl_t, t_c = ccl(pressure, temperature, dewpoint)
    assert_almost_equal(ccl_p, 763.006048 * units.mbar, 5)
    assert_almost_equal(ccl_t, 15.429946 * units.degC, 5)
    assert_almost_equal(t_c, 37.991498 * units.degC, 5)


def test_ccl_nans():
    """Tests CCL handles nans."""
    pressure = np.array([993.0, 984.0, 957.0, np.nan, 925.0, 917.0, np.nan, 868.0, 850.0,
                         841.0, 813.0, 806.0, 798.0, 738.0, 732.0, 723.0, 716.0, 711.0,
                         700.0, 623.0, 621.0, 582.0, 541.0, 500.0, 468.0]) * units.mbar
    temperature = np.array([34.6, np.nan, 31.1, np.nan, 27.8, 27.1, 24.3, 22.6, 21.4,
                            20.8, 19.6, 19.4, 18.7, 13.0, 13.0, 13.4, 13.5, 13.6,
                            13.0, 5.2, 5.0, 1.5, -2.4, -6.7, -10.7]) * units.degC
    dewpoint = np.array([19.6, 19.4, 18.7, np.nan, 17.8, 17.5, 16.3, 15.6, 12.4, 10.8,
                         -0.4, -3.6, -3.8, -5.0, -6.0, -15.6, -13.2, -11.4, -11.0,
                         -5.8, -6.2, -14.8, -24.3, -34.7, -38.1]) * units.degC
    ccl_p, ccl_t, t_c = ccl(pressure, temperature, dewpoint)
    assert_almost_equal(ccl_p, 763.006048 * units.mbar, 5)
    assert_almost_equal(ccl_t, 15.429946 * units.degC, 5)
    assert_almost_equal(t_c, 37.991498 * units.degC, 5)


def test_ccl_unit():
    """Tests CCL pressure and temperature is returned in the correct unit."""
    pressure = (np.array([993.0, 984.0, 957.0, 948.0, 925.0, 917.0, 886.0, 868.0, 850.0,
                         841.0, 813.0, 806.0, 798.0, 738.0, 732.0, 723.0, 716.0, 711.0,
                         700.0, 623.0, 621.0, 582.0, 541.0, 500.0, 468.0]) * 100) * units.Pa
    temperature = (np.array([34.6, 33.7, 31.1, 30.1, 27.8, 27.1, 24.3, 22.6, 21.4,
                             20.8, 19.6, 19.4, 18.7, 13.0, 13.0, 13.4, 13.5, 13.6,
                             13.0, 5.2, 5.0, 1.5, -2.4, -6.7, -10.7]) + 273.15) * units.kelvin
    dewpoint = (np.array([19.6, 19.4, 18.7, 18.4, 17.8, 17.5, 16.3, 15.6, 12.4, 10.8,
                          -0.4, -3.6, -3.8, -5.0, -6.0, -15.6, -13.2, -11.4, -11.0,
                          -5.8, -6.2, -14.8, -24.3, -34.7, -38.1]) + 273.15) * units.kelvin

    ccl_p, ccl_t, t_c = ccl(pressure, temperature, dewpoint)
    assert_almost_equal(ccl_p, (763.006048 * 100) * units.Pa, 3)
    assert_almost_equal(ccl_t, (15.429946 + 273.15) * units.kelvin, 3)
    assert_almost_equal(t_c, (37.991498 + 273.15) * units.kelvin, 3)

    assert ccl_p.units == pressure.units
    assert ccl_t.units == temperature.units
    assert t_c.units == temperature.units


def test_multiple_ccl():
    """Tests the case where there are multiple CCLs. Data: BUF, May 18 2022 12Z."""
    pressure = np.array([992.0, 990.0, 983.0, 967.0, 950.0, 944.0, 928.0, 925.0, 922.0,
                         883.0, 877.7, 858.0, 853.0, 850.0, 835.0, 830.0, 827.0, 826.0,
                         813.6, 808.0, 799.0, 784.0, 783.3, 769.0, 760.0, 758.0, 754.0,
                         753.0, 738.0, 725.7, 711.0, 704.0, 700.0, 685.0, 672.0, 646.6,
                         598.6, 596.0, 587.0, 582.0, 567.0, 560.0, 555.0, 553.3, 537.0,
                         526.0, 521.0, 519.0, 515.0, 500.0]) * units.mbar
    temperature = np.array([6.8, 6.2, 7.8, 7.6, 7.2, 7.6, 6.6, 6.4, 6.2, 3.2, 2.8, 1.2,
                            1.0, 0.8, -0.3, -0.1, 0.4, 0.6, 0.9, 1.0, 0.6, -0.3, -0.3,
                            -0.7, -1.5, -1.3, 0.2, 0.2, -1.1, -2.1, -3.3, -2.3, -1.7, 0.2,
                            -0.9, -3.0, -7.3, -7.5, -8.1, -8.3, -9.5, -10.1, -10.7,
                            -10.8, -12.1, -12.5, -12.7, -12.9, -13.5, -15.5]) * units.degC
    dewpoint = np.array([5.1, 5.0, 4.2, 2.7, 2.2, 0.6, -2.4, -2.6, -2.8, -3.8, -3.6,
                        -3.1, -5.0, -4.2, -1.8, -4.3, -7.6, -6.4, -8.2, -9.0, -10.4,
                        -9.3, -9.6, -14.7, -11.5, -12.3, -25.8, -25.8, -19.1, -19.6,
                        -20.3, -42.3, -39.7, -46.8, -46.8, -46.7, -46.5, -46.5,
                        -52.1, -36.3, -47.5, -30.1, -29.7, -30.4, -37.1, -49.5,
                        -36.7, -28.9, -28.5, -22.5]) * units.degC

    ccl_p, ccl_t, t_c = ccl(pressure, temperature, dewpoint)
    assert_almost_equal(ccl_p, 680.191653 * units.mbar, 5)
    assert_almost_equal(ccl_t, -0.204408 * units.degC, 5)
    assert_almost_equal(t_c, 30.8678258 * units.degC, 5)

    ccl_p, ccl_t, t_c = ccl(pressure, temperature, dewpoint, which='bottom')
    assert_almost_equal(ccl_p, 886.835325 * units.mbar, 5)
    assert_almost_equal(ccl_t, 3.500840 * units.degC, 5)
    assert_almost_equal(t_c, 12.5020423 * units.degC, 5)

    ccl_p, ccl_t, t_c = ccl(pressure, temperature, dewpoint, which='all')
    assert_array_almost_equal(ccl_p, np.array([886.835325, 680.191653]) * units.mbar, 5)
    assert_array_almost_equal(ccl_t, np.array([3.500840, -0.204408]) * units.degC, 5)
    assert_array_almost_equal(t_c, np.array([12.5020423, 30.8678258]) * units.degC, 5)


def test_ccl_with_ml():
    """Test CCL calculation with a specified mixed-layer depth."""
    pressure = np.array([992.0, 990.0, 983.0, 967.0, 950.0, 944.0, 928.0, 925.0, 922.0,
                         883.0, 877.7, 858.0, 853.0, 850.0, 835.0, 830.0, 827.0, 826.0,
                         813.6, 808.0, 799.0, 784.0, 783.3, 769.0, 760.0, 758.0, 754.0,
                         753.0, 738.0, 725.7, 711.0, 704.0, 700.0, 685.0, 672.0, 646.6,
                         598.6, 596.0, 587.0, 582.0, 567.0, 560.0, 555.0, 553.3, 537.0,
                         526.0, 521.0, 519.0, 515.0, 500.0]) * units.mbar
    temperature = np.array([6.8, 6.2, 7.8, 7.6, 7.2, 7.6, 6.6, 6.4, 6.2, 3.2, 2.8, 1.2,
                            1.0, 0.8, -0.3, -0.1, 0.4, 0.6, 0.9, 1.0, 0.6, -0.3, -0.3,
                            -0.7, -1.5, -1.3, 0.2, 0.2, -1.1, -2.1, -3.3, -2.3, -1.7, 0.2,
                            -0.9, -3.0, -7.3, -7.5, -8.1, -8.3, -9.5, -10.1, -10.7,
                            -10.8, -12.1, -12.5, -12.7, -12.9, -13.5, -15.5]) * units.degC
    dewpoint = np.array([5.1, 5.0, 4.2, 2.7, 2.2, 0.6, -2.4, -2.6, -2.8, -3.8, -3.6,
                        -3.1, -5.0, -4.2, -1.8, -4.3, -7.6, -6.4, -8.2, -9.0, -10.4,
                        -9.3, -9.6, -14.7, -11.5, -12.3, -25.8, -25.8, -19.1, -19.6,
                        -20.3, -42.3, -39.7, -46.8, -46.8, -46.7, -46.5, -46.5,
                        -52.1, -36.3, -47.5, -30.1, -29.7, -30.4, -37.1, -49.5,
                        -36.7, -28.9, -28.5, -22.5]) * units.degC

    ccl_p, ccl_t, t_c = ccl(pressure, temperature, dewpoint,
                            mixed_layer_depth=500 * units.m, which='all')

    assert_array_almost_equal(ccl_p, np.array(
        [850.600930, 784.325312, 737.767377, 648.076147]) * units.mbar, 5)
    assert_array_almost_equal(ccl_t, np.array(
        [0.840118, -0.280299, -1.118757, -2.875716]) * units.degC, 5)
    assert_array_almost_equal(t_c, np.array(
        [13.146845, 18.661621, 22.896152, 32.081388]) * units.degC, 5)


def test_lfc_basic():
    """Test LFC calculation."""
    levels = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12., 9.4, 7., -49.]) * units.celsius
    dewpoints = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(lfc_pressure, 727.371 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 9.705 * units.celsius, 2)


def test_lfc_kelvin():
    """Test that LFC temperature returns Kelvin if Kelvin is provided."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = (np.array([22.2, 14.6, 12., 9.4, 7., -49.]
                            ) + 273.15) * units.kelvin
    dewpoint = (np.array([19., -11.2, -10.8, -10.4, -10., -53.2]
                         ) + 273.15) * units.kelvin
    lfc_pressure, lfc_temp = lfc(pressure, temperature, dewpoint)
    assert_almost_equal(lfc_pressure, 727.371 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 9.705 * units.degC, 2)
    assert lfc_temp.units == temperature.units


def test_lfc_ml():
    """Test Mixed-Layer LFC calculation."""
    levels = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12., 9.4, 7., -49.]) * units.celsius
    dewpoints = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    __, t_mixed, td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = parcel_profile(levels, t_mixed, td_mixed)
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints, mixed_parcel_prof)
    assert_almost_equal(lfc_pressure, 601.225 * units.mbar, 2)
    assert_almost_equal(lfc_temp, -1.90688 * units.degC, 2)


def test_lfc_ml2():
    """Test a mixed-layer LFC calculation that previously crashed."""
    levels = np.array([1024.95703125, 1016.61474609, 1005.33056641, 991.08544922, 973.4163208,
                       951.3381958, 924.82836914, 898.25482178, 873.46124268, 848.69830322,
                       823.92553711, 788.49304199, 743.44580078, 700.50970459, 659.62017822,
                       620.70861816, 583.69421387, 548.49719238, 515.03826904, 483.24401855,
                       453.0418396, 424.36477661, 397.1505127, 371.33441162, 346.85922241,
                       323.66995239, 301.70935059, 280.92651367, 261.27053833, 242.69168091,
                       225.14237976, 208.57781982, 192.95333862, 178.22599792, 164.39630127,
                       151.54336548, 139.68635559, 128.74923706, 118.6588974, 109.35111237,
                       100.76405334, 92.84288025, 85.53556824, 78.79430389, 72.57549286,
                       66.83885193, 61.54678726, 56.66480637, 52.16108322]) * units.mbar
    temperatures = np.array([6.00750732, 5.14892578, 4.177948, 3.00268555, 1.55535889,
                             -0.25527954, -1.93988037, -3.57766724, -4.40600586, -4.19238281,
                             -3.71185303, -4.47943115, -6.81280518, -8.08685303, -8.41287231,
                             -10.79302979, -14.13262939, -16.85784912, -19.51675415,
                             -22.28689575, -24.99938965, -27.79664612, -30.90414429,
                             -34.49435425, -38.438797, -42.27981567, -45.99230957,
                             -49.75340271, -53.58230591, -57.30686951, -60.76026917,
                             -63.92070007, -66.72470093, -68.97846985, -70.4264679,
                             -71.16407776, -71.53797913, -71.64375305, -71.52735901,
                             -71.53523254, -71.61097717, -71.92687988, -72.68682861,
                             -74.129776, -76.02471924, -76.88977051, -76.26008606,
                             -75.90351868, -76.15809631]) * units.celsius
    dewpoints = np.array([4.50012302, 3.42483997, 2.78102994, 2.24474645, 1.593485, -0.9440815,
                          -3.8044982, -3.55629468, -9.7376976, -10.2950449, -9.67498302,
                          -10.30486488, -8.70559597, -8.71669006, -12.66509628, -18.6697197,
                          -23.00351334, -29.46240425, -36.82178497, -41.68824768, -44.50320816,
                          -48.54426575, -52.50753403, -51.09564209, -48.92690659, -49.97380829,
                          -51.57516098, -52.62096405, -54.24332809, -57.09109879, -60.5596199,
                          -63.93486404, -67.07530212, -70.01263428, -72.9258728, -76.12271881,
                          -79.49847412, -82.2350769, -83.91127014, -84.95665741, -85.61238861,
                          -86.16391754, -86.7653656, -87.34436035, -87.87495422, -88.34281921,
                          -88.74453735, -89.04680634, -89.26436615]) * units.celsius
    __, t_mixed, td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = parcel_profile(levels, t_mixed, td_mixed)
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints, mixed_parcel_prof, td_mixed)
    assert_almost_equal(lfc_pressure, 962.34 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 0.767 * units.degC, 2)


def test_lfc_intersection():
    """Test LFC calculation when LFC is below a tricky intersection."""
    p = np.array([1024.957, 930., 924.828, 898.255, 873.461, 848.698, 823.926,
                  788.493]) * units('hPa')
    t = np.array([6.008, -10., -6.94, -8.58, -4.41, -4.19, -3.71, -4.48]) * units('degC')
    td = np.array([5., -10., -7., -9., -4.5, -4.2, -3.8, -4.5]) * units('degC')
    _, mlt, mltd = mixed_parcel(p, t, td)
    ml_profile = parcel_profile(p, mlt, mltd)
    mllfc_p, mllfc_t = lfc(p, t, td, ml_profile, mltd)
    assert_almost_equal(mllfc_p, 981.620 * units.hPa, 2)
    assert_almost_equal(mllfc_t, 272.045 * units.kelvin, 2)


def test_no_lfc():
    """Test LFC calculation when there is no LFC in the data."""
    levels = np.array([959., 867.9, 779.2, 647.5, 472.5, 321.9, 251.]) * units.mbar
    temperatures = np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5]) * units.celsius
    dewpoints = np.array([9., 4.3, -21.2, -26.7, -31., -53.3, -66.7]) * units.celsius
    lfc_pressure, lfc_temperature = lfc(levels, temperatures, dewpoints)
    assert_nan(lfc_pressure, levels.units)
    assert_nan(lfc_temperature, temperatures.units)


def test_lfc_inversion():
    """Test LFC when there is an inversion to be sure we don't pick that."""
    levels = np.array([963., 789., 782.3, 754.8, 728.1, 727., 700.,
                       571., 450., 300., 248.]) * units.mbar
    temperatures = np.array([25.4, 18.4, 17.8, 15.4, 12.9, 12.8,
                             10., -3.9, -16.3, -41.1, -51.5]) * units.celsius
    dewpoints = np.array([20.4, 0.4, -0.5, -4.3, -8., -8.2, -9.,
                          -23.9, -33.3, -54.1, -63.5]) * units.celsius
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(lfc_pressure, 705.8806 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 10.6232 * units.celsius, 2)


def test_lfc_equals_lcl():
    """Test LFC when there is no cap and the lfc is equal to the lcl."""
    levels = np.array([912., 905.3, 874.4, 850., 815.1, 786.6, 759.1,
                       748., 732.2, 700., 654.8]) * units.mbar
    temperatures = np.array([29.4, 28.7, 25.2, 22.4, 19.4, 16.8,
                             14.0, 13.2, 12.6, 11.4, 7.1]) * units.celsius
    dewpoints = np.array([18.4, 18.1, 16.6, 15.4, 13.2, 11.4, 9.6,
                          8.8, 0., -18.6, -22.9]) * units.celsius
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(lfc_pressure, 777.0786 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 15.8714 * units.celsius, 2)


def test_sensitive_sounding():
    """Test quantities for a sensitive sounding (#902)."""
    # This sounding has a very small positive area in the low level. It's only captured
    # properly if the parcel profile includes the LCL, otherwise it breaks LFC and CAPE
    p = units.Quantity([1004., 1000., 943., 928., 925., 850., 839., 749., 700., 699.,
                        603., 500., 404., 400., 363., 306., 300., 250., 213., 200.,
                        176., 150.], 'hectopascal')
    t = units.Quantity([24.2, 24., 20.2, 21.6, 21.4, 20.4, 20.2, 14.4, 13.2, 13., 6.8, -3.3,
                        -13.1, -13.7, -17.9, -25.5, -26.9, -37.9, -46.7, -48.7, -52.1, -58.9],
                       'degC')
    td = units.Quantity([21.9, 22.1, 19.2, 20.5, 20.4, 18.4, 17.4, 8.4, -2.8, -3.0, -15.2,
                         -20.3, -29.1, -27.7, -24.9, -39.5, -41.9, -51.9, -60.7, -62.7, -65.1,
                         -71.9], 'degC')
    lfc_pressure, lfc_temp = lfc(p, t, td)
    assert_almost_equal(lfc_pressure, 947.422 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 20.498 * units.degC, 2)

    pos, neg = surface_based_cape_cin(p, t, td)
    assert_almost_equal(pos, 0.1115 * units('J/kg'), 3)
    assert_almost_equal(neg, -6.0866 * units('J/kg'), 3)


def test_lfc_sfc_precision():
    """Test LFC when there are precision issues with the parcel path."""
    levels = np.array([839., 819.4, 816., 807., 790.7, 763., 736.2,
                       722., 710.1, 700.]) * units.mbar
    temperatures = np.array([20.6, 22.3, 22.6, 22.2, 20.9, 18.7, 16.4,
                             15.2, 13.9, 12.8]) * units.celsius
    dewpoints = np.array([10.6, 8., 7.6, 6.2, 5.7, 4.7, 3.7, 3.2, 3., 2.8]) * units.celsius
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_nan(lfc_pressure, levels.units)
    assert_nan(lfc_temp, temperatures.units)


def test_lfc_pos_area_below_lcl():
    """Test LFC when there is positive area below the LCL (#1003)."""
    p = [902.1554, 897.9034, 893.6506, 889.4047, 883.063, 874.6284, 866.2387, 857.887,
         849.5506, 841.2686, 833.0042, 824.7891, 812.5049, 796.2104, 776.0027, 751.9025,
         727.9612, 704.1409, 680.4028, 656.7156, 629.077, 597.4286, 565.6315, 533.5961,
         501.2452, 468.493, 435.2486, 401.4239, 366.9387, 331.7026, 295.6319, 258.6428,
         220.9178, 182.9384, 144.959, 106.9778, 69.00213] * units.hPa
    t = [-3.039381, -3.703779, -4.15996, -4.562574, -5.131827, -5.856229, -6.568434,
         -7.276881, -7.985013, -8.670911, -8.958063, -7.631381, -6.05927, -5.083627,
         -5.11576, -5.687552, -5.453021, -4.981445, -5.236665, -6.324916, -8.434324,
         -11.58795, -14.99297, -18.45947, -21.92021, -25.40522, -28.914, -32.78637,
         -37.7179, -43.56836, -49.61077, -54.24449, -56.16666, -57.03775, -58.28041,
         -60.86264, -64.21677] * units.degC
    td = [-22.08774, -22.18181, -22.2508, -22.31323, -22.4024, -22.51582, -22.62526,
          -22.72919, -22.82095, -22.86173, -22.49489, -21.66936, -21.67332, -21.94054,
          -23.63561, -27.17466, -31.87395, -38.31725, -44.54717, -46.99218, -43.17544,
          -37.40019, -34.3351, -36.42896, -42.1396, -46.95909, -49.36232, -48.94634,
          -47.90178, -49.97902, -55.02753, -63.06276, -72.53742, -88.81377, -93.54573,
          -92.92464, -91.57479] * units.degC
    prof = parcel_profile(p, t[0], td[0]).to('degC')
    lfc_p, lfc_t = lfc(p, t, td, prof)
    assert_nan(lfc_p, p.units)
    assert_nan(lfc_t, t.units)


def test_saturation_mixing_ratio():
    """Test saturation mixing ratio calculation."""
    p = 999. * units.mbar
    t = 288. * units.kelvin
    assert_almost_equal(saturation_mixing_ratio(p, t), .01068, 3)


def test_saturation_mixing_ratio_with_xarray():
    """Test saturation mixing ratio calculation with xarray."""
    temperature = xr.DataArray(
        np.arange(10, 18).reshape((2, 2, 2)) * units.degC,
        dims=('isobaric', 'y', 'x'),
        coords={
            'isobaric': (('isobaric',), [700., 850.], {'units': 'hPa'}),
            'y': (('y',), [0., 100.], {'units': 'kilometer'}),
            'x': (('x',), [0., 100.], {'units': 'kilometer'})
        }
    )
    result = saturation_mixing_ratio(temperature.metpy.vertical, temperature)
    expected_values = [[[0.011098, 0.011879], [0.012708, 0.013589]],
                       [[0.011913, 0.012724], [0.013586, 0.014499]]]
    assert_array_almost_equal(result.data, expected_values, 5)
    xr.testing.assert_identical(result['isobaric'], temperature['isobaric'])
    xr.testing.assert_identical(result['y'], temperature['y'])
    xr.testing.assert_identical(result['x'], temperature['x'])


def test_equivalent_potential_temperature():
    """Test equivalent potential temperature calculation."""
    p = 1000 * units.mbar
    t = 293. * units.kelvin
    td = 280. * units.kelvin
    ept = equivalent_potential_temperature(p, t, td)
    assert_almost_equal(ept, 311.18586467284007 * units.kelvin, 3)


def test_equivalent_potential_temperature_masked():
    """Test equivalent potential temperature calculation with masked arrays."""
    p = 1000 * units.mbar
    t = units.Quantity(np.ma.array([293., 294., 295.]), units.kelvin)
    td = units.Quantity(
        np.ma.array([280., 281., 282.], mask=[False, True, False]),
        units.kelvin
    )
    ept = equivalent_potential_temperature(p, t, td)
    expected = units.Quantity(
        np.ma.array([311.18586, 313.51781, 315.93971], mask=[False, True, False]),
        units.kelvin
    )
    assert is_quantity(ept)
    assert isinstance(ept.m, np.ma.MaskedArray)
    assert_array_almost_equal(ept, expected, 3)


def test_saturation_equivalent_potential_temperature():
    """Test saturation equivalent potential temperature calculation."""
    p = 700 * units.mbar
    t = 263.15 * units.kelvin
    s_ept = saturation_equivalent_potential_temperature(p, t)
    # 299.096584 comes from equivalent_potential_temperature(p,t,t)
    # where dewpoint and temperature are equal, which means saturations.
    assert_almost_equal(s_ept, 299.10542 * units.kelvin, 3)


def test_saturation_equivalent_potential_temperature_masked():
    """Test saturation equivalent potential temperature calculation with masked arrays."""
    p = 1000 * units.mbar
    t = units.Quantity(np.ma.array([293., 294., 295.]), units.kelvin)
    s_ept = saturation_equivalent_potential_temperature(p, t)
    expected = units.Quantity(
        np.ma.array([335.02750, 338.95813, 343.08740]),
        units.kelvin
    )
    assert is_quantity(s_ept)
    assert isinstance(s_ept.m, np.ma.MaskedArray)
    assert_array_almost_equal(s_ept, expected, 3)


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
    assert_almost_equal(el_pressure, 471.83286 * units.mbar, 3)
    assert_almost_equal(el_temperature, -11.5603 * units.degC, 3)


def test_el_kelvin():
    """Test that EL temperature returns Kelvin if Kelvin is provided."""
    levels = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperatures = (np.array([22.2, 14.6, 12., 9.4, 7., -38.]) + 273.15) * units.kelvin
    dewpoints = (np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) + 273.15) * units.kelvin
    el_pressure, el_temp = el(levels, temperatures, dewpoints)
    assert_almost_equal(el_pressure, 471.8329 * units.mbar, 3)
    assert_almost_equal(el_temp, -11.5603 * units.degC, 3)
    assert el_temp.units == temperatures.units


def test_el_ml():
    """Test equilibrium layer calculation for a mixed parcel."""
    levels = np.array([959., 779.2, 751.3, 724.3, 700., 400., 269.]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12., 9.4, 7., -25., -35.]) * units.celsius
    dewpoints = np.array([19., -11.2, -10.8, -10.4, -10., -35., -53.2]) * units.celsius
    __, t_mixed, td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = parcel_profile(levels, t_mixed, td_mixed)
    el_pressure, el_temperature = el(levels, temperatures, dewpoints, mixed_parcel_prof)
    assert_almost_equal(el_pressure, 350.0561 * units.mbar, 3)
    assert_almost_equal(el_temperature, -28.36156 * units.degC, 3)


def test_no_el():
    """Test equilibrium layer calculation when there is no EL in the data."""
    levels = np.array([959., 867.9, 779.2, 647.5, 472.5, 321.9, 251.]) * units.mbar
    temperatures = np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5]) * units.celsius
    dewpoints = np.array([19., 14.3, -11.2, -16.7, -21., -43.3, -56.7]) * units.celsius
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


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
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


def test_lfc_and_el_below_lcl():
    """Test that LFC and EL are returned as NaN if both are below LCL."""
    dewpoint = [264.5351, 261.13443, 259.0122, 252.30063, 248.58017, 242.66582] * units.kelvin
    temperature = [273.09723, 268.40173, 263.56207, 260.257, 256.63538,
                   252.91345] * units.kelvin
    pressure = [1017.16, 950, 900, 850, 800, 750] * units.hPa
    el_pressure, el_temperature = el(pressure, temperature, dewpoint)
    lfc_pressure, lfc_temperature = lfc(pressure, temperature, dewpoint)
    assert_nan(lfc_pressure, pressure.units)
    assert_nan(lfc_temperature, temperature.units)
    assert_nan(el_pressure, pressure.units)
    assert_nan(el_temperature, temperature.units)


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
    assert_almost_equal(el_pressure, 175.7663 * units.mbar, 3)
    assert_almost_equal(el_temperature, -57.03994 * units.degC, 3)


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
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


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
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


def test_el_below_lcl():
    """Test LFC when there is positive area below the LCL (#1003)."""
    p = [902.1554, 897.9034, 893.6506, 889.4047, 883.063, 874.6284, 866.2387, 857.887,
         849.5506, 841.2686, 833.0042, 824.7891, 812.5049, 796.2104, 776.0027, 751.9025,
         727.9612, 704.1409, 680.4028, 656.7156, 629.077, 597.4286, 565.6315, 533.5961,
         501.2452, 468.493, 435.2486, 401.4239, 366.9387, 331.7026, 295.6319, 258.6428,
         220.9178, 182.9384, 144.959, 106.9778, 69.00213] * units.hPa
    t = [-3.039381, -3.703779, -4.15996, -4.562574, -5.131827, -5.856229, -6.568434,
         -7.276881, -7.985013, -8.670911, -8.958063, -7.631381, -6.05927, -5.083627,
         -5.11576, -5.687552, -5.453021, -4.981445, -5.236665, -6.324916, -8.434324,
         -11.58795, -14.99297, -18.45947, -21.92021, -25.40522, -28.914, -32.78637,
         -37.7179, -43.56836, -49.61077, -54.24449, -56.16666, -57.03775, -58.28041,
         -60.86264, -64.21677] * units.degC
    td = [-22.08774, -22.18181, -22.2508, -22.31323, -22.4024, -22.51582, -22.62526,
          -22.72919, -22.82095, -22.86173, -22.49489, -21.66936, -21.67332, -21.94054,
          -23.63561, -27.17466, -31.87395, -38.31725, -44.54717, -46.99218, -43.17544,
          -37.40019, -34.3351, -36.42896, -42.1396, -46.95909, -49.36232, -48.94634,
          -47.90178, -49.97902, -55.02753, -63.06276, -72.53742, -88.81377, -93.54573,
          -92.92464, -91.57479] * units.degC
    prof = parcel_profile(p, t[0], td[0]).to('degC')
    el_p, el_t = el(p, t, td, prof)
    assert_nan(el_p, p.units)
    assert_nan(el_t, t.units)


def test_wet_psychrometric_vapor_pressure():
    """Test calculation of vapor pressure from wet and dry bulb temperatures."""
    p = 1013.25 * units.mbar
    dry_bulb_temperature = 20. * units.degC
    wet_bulb_temperature = 18. * units.degC
    psychrometric_vapor_pressure = psychrometric_vapor_pressure_wet(p, dry_bulb_temperature,
                                                                    wet_bulb_temperature)
    assert_almost_equal(psychrometric_vapor_pressure, 19.3673 * units.mbar, 3)


def test_wet_psychrometric_rh():
    """Test calculation of relative humidity from wet and dry bulb temperatures."""
    p = 1013.25 * units.mbar
    dry_bulb_temperature = 20. * units.degC
    wet_bulb_temperature = 18. * units.degC
    psychrometric_rh = relative_humidity_wet_psychrometric(p, dry_bulb_temperature,
                                                           wet_bulb_temperature)
    assert_almost_equal(psychrometric_rh, 82.8747 * units.percent, 3)


def test_wet_psychrometric_rh_kwargs():
    """Test calculation of relative humidity from wet and dry bulb temperatures."""
    p = 1013.25 * units.mbar
    dry_bulb_temperature = 20. * units.degC
    wet_bulb_temperature = 18. * units.degC
    coeff = 6.1e-4 / units.kelvin
    psychrometric_rh = relative_humidity_wet_psychrometric(p, dry_bulb_temperature,
                                                           wet_bulb_temperature,
                                                           psychrometer_coefficient=coeff)
    assert_almost_equal(psychrometric_rh, 82.9701 * units.percent, 3)


def test_mixing_ratio_from_relative_humidity():
    """Test relative humidity from mixing ratio."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    rh = 81.7219 * units.percent
    w = mixing_ratio_from_relative_humidity(p, temperature, rh)
    assert_almost_equal(w, 0.012 * units.dimensionless, 3)


def test_rh_mixing_ratio():
    """Test relative humidity from mixing ratio."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    w = 0.012 * units.dimensionless
    rh = relative_humidity_from_mixing_ratio(p, temperature, w)
    assert_almost_equal(rh, 81.72498 * units.percent, 3)


def test_mixing_ratio_from_specific_humidity():
    """Test mixing ratio from specific humidity."""
    q = 0.012 * units.dimensionless
    w = mixing_ratio_from_specific_humidity(q)
    assert_almost_equal(w, 0.01215, 3)


def test_mixing_ratio_from_specific_humidity_no_units():
    """Test mixing ratio from specific humidity works without units."""
    q = 0.012
    w = mixing_ratio_from_specific_humidity(q)
    assert_almost_equal(w, 0.01215, 3)


def test_specific_humidity_from_mixing_ratio():
    """Test specific humidity from mixing ratio."""
    w = 0.01215 * units.dimensionless
    q = specific_humidity_from_mixing_ratio(w)
    assert_almost_equal(q, 0.01200, 5)


def test_specific_humidity_from_mixing_ratio_no_units():
    """Test specific humidity from mixing ratio works without units."""
    w = 0.01215
    q = specific_humidity_from_mixing_ratio(w)
    assert_almost_equal(q, 0.01200, 5)


def test_rh_specific_humidity():
    """Test relative humidity from specific humidity."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    q = 0.012 * units.dimensionless
    rh = relative_humidity_from_specific_humidity(p, temperature, q)
    assert_almost_equal(rh, 82.71759 * units.percent, 3)


def test_cape_cin():
    """Test the basic CAPE and CIN calculation."""
    p = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0])
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 75.05354 * units('joule / kilogram'), 2)
    assert_almost_equal(cin, -89.890078 * units('joule / kilogram'), 2)


def test_cape_cin_no_el():
    """Test that CAPE works with no EL."""
    p = np.array([959., 779.2, 751.3, 724.3]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]).to('degC')
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 0.08610409 * units('joule / kilogram'), 2)
    assert_almost_equal(cin, -89.8900784 * units('joule / kilogram'), 2)


def test_cape_cin_no_lfc():
    """Test that CAPE is zero with no LFC."""
    p = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 24.6, 22., 20.4, 18., -10.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]).to('degC')
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 0.0 * units('joule / kilogram'), 2)
    assert_almost_equal(cin, 0.0 * units('joule / kilogram'), 2)


def test_find_append_zero_crossings():
    """Test finding and appending zero crossings of an x, y series."""
    x = np.arange(11) * units.hPa
    y = np.array([3, 2, 1, -1, 2, 2, 0, 1, 0, -1, 2]) * units.degC
    x2, y2 = _find_append_zero_crossings(x, y)

    x_truth = np.array([0., 1., 2., 2.4494897, 3., 3.3019272, 4., 5.,
                        6., 7., 8., 9., 9.3216975, 10.]) * units.hPa
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


@pytest.mark.filterwarnings('ignore:invalid value:RuntimeWarning')
def test_isentropic_pressure():
    """Test calculation of isentropic pressure function."""
    lev = [100000., 95000., 90000., 85000.] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.
    tmp[1, :] = 292.
    tmp[2, :] = 290
    tmp[3, :] = 288.
    tmp[:, :, -1] = np.nan
    tmpk = tmp * units.kelvin
    isentlev = [296.] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk)
    trueprs = np.ones((1, 5, 5)) * (1000. * units.hPa)
    trueprs[:, :, -1] = np.nan
    assert isentprs[0].shape == (1, 5, 5)
    assert_almost_equal(isentprs[0], trueprs, 3)


def test_isentropic_pressure_masked_column():
    """Test calculation of isentropic pressure function with a masked column (#769)."""
    lev = [100000., 95000.] * units.Pa
    tmp = np.ma.ones((len(lev), 5, 5))
    tmp[0, :] = 296.
    tmp[1, :] = 292.
    tmp[:, :, -1] = np.ma.masked
    tmp = units.Quantity(tmp, units.kelvin)
    isentprs = isentropic_interpolation([296.] * units.kelvin, lev, tmp)
    trueprs = np.ones((1, 5, 5)) * (1000. * units.hPa)
    trueprs[:, :, -1] = np.nan
    assert isentprs[0].shape == (1, 5, 5)
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


def test_isentropic_pressure_additional_args():
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
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, temperature_out=True)
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
    trueprs = 936.213 * units.hPa
    assert_almost_equal(isentprs[0][1], trueprs, 3)


def test_isentropic_pressure_addition_args_interp():
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
    truerh = 69.197 * units.percent
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
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, temperature_out=True)
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
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, relh, vertical_dim=1)
    trueprs = 1000. * units.hPa
    trueprs2 = 936.213 * units.hPa
    trueprs3 = 879.50375588 * units.hPa
    truerh = 69.19706 * units.percent
    assert isentprs[0].shape == (3, 3, 5, 5)
    assert_almost_equal(isentprs[0][:, 0, :], trueprs, 3)
    assert_almost_equal(isentprs[0][:, 1, :], trueprs2, 3)
    assert_almost_equal(isentprs[0][:, 2, :], trueprs3, 3)
    assert_almost_equal(isentprs[1][:, 1, ], truerh, 3)


def test_isentropic_interpolation_dataarray():
    """Test calculation of isentropic interpolation with xarray dataarrays."""
    temp = xr.DataArray([[[296.]], [[292.]], [[290.]], [[288.]]] * units.K,
                        dims=('isobaric', 'y', 'x'),
                        coords={'isobaric': (('isobaric',), [1000., 950., 900., 850.],
                                             {'units': 'hPa'}),
                                'time': '2020-01-01T00:00Z'})

    rh = xr.DataArray([[[100.]], [[80.]], [[40.]], [[20.]]] * units.percent,
                      dims=('isobaric', 'y', 'x'), coords={
        'isobaric': (('isobaric',), [1000., 950., 900., 850.], {'units': 'hPa'}),
        'time': '2020-01-01T00:00Z'})

    isentlev = [296., 297.] * units.kelvin
    press, rh_interp = isentropic_interpolation(isentlev, temp.isobaric, temp, rh)

    assert_array_almost_equal(press, np.array([[[1000.]], [[936.213]]]) * units.hPa, 3)
    assert_array_almost_equal(rh_interp, np.array([[[100.]], [[69.19706]]]) * units.percent, 3)


def test_isentropic_interpolation_as_dataset():
    """Test calculation of isentropic interpolation with xarray."""
    data = xr.Dataset(
        {
            'temperature': (
                ('isobaric', 'y', 'x'),
                [[[296.]], [[292.]], [[290.]], [[288.]]] * units.K
            ),
            'rh': (
                ('isobaric', 'y', 'x'),
                [[[100.]], [[80.]], [[40.]], [[20.]]] * units.percent
            )
        },
        coords={
            'isobaric': (('isobaric',), [1000., 950., 900., 850.], {'units': 'hPa'}),
            'time': '2020-01-01T00:00Z'
        }
    )
    isentlev = [296., 297.] * units.kelvin
    result = isentropic_interpolation_as_dataset(isentlev, data['temperature'], data['rh'])
    expected = xr.Dataset(
        {
            'pressure': (
                ('isentropic_level', 'y', 'x'),
                [[[1000.]], [[936.213]]] * units.hPa,
                {'standard_name': 'air_pressure'}
            ),
            'temperature': (
                ('isentropic_level', 'y', 'x'),
                [[[296.]], [[291.4579]]] * units.K,
                {'standard_name': 'air_temperature'}
            ),
            'rh': (
                ('isentropic_level', 'y', 'x'),
                [[[100.]], [[69.19706]]] * units.percent
            )
        },
        coords={
            'isentropic_level': (
                ('isentropic_level',),
                [296., 297.],
                {'units': 'kelvin', 'positive': 'up'}
            ),
            'time': '2020-01-01T00:00Z'
        }
    )
    xr.testing.assert_allclose(result, expected)
    assert result['pressure'].attrs == expected['pressure'].attrs
    assert result['temperature'].attrs == expected['temperature'].attrs
    assert result['isentropic_level'].attrs == expected['isentropic_level'].attrs


@pytest.mark.parametrize('array_class', (units.Quantity, masked_array))
def test_surface_based_cape_cin(array_class):
    """Test the surface-based CAPE and CIN calculation."""
    p = array_class([959., 779.2, 751.3, 724.3, 700., 269.], units.mbar)
    temperature = array_class([22.2, 14.6, 12., 9.4, 7., -38.], units.celsius)
    dewpoint = array_class([19., -11.2, -10.8, -10.4, -10., -53.2], units.celsius)
    cape, cin = surface_based_cape_cin(p, temperature, dewpoint)
    assert_almost_equal(cape, 75.0535446 * units('joule / kilogram'), 2)
    assert_almost_equal(cin, -136.685967 * units('joule / kilogram'), 2)


def test_surface_based_cape_cin_with_xarray():
    """Test the surface-based CAPE and CIN calculation with xarray."""
    data = xr.Dataset(
        {
            'temperature': (('isobaric',), [22.2, 14.6, 12., 9.4, 7., -38.] * units.degC),
            'dewpoint': (('isobaric',), [19., -11.2, -10.8, -10.4, -10., -53.2] * units.degC)
        },
        coords={
            'isobaric': (
                ('isobaric',),
                [959., 779.2, 751.3, 724.3, 700., 269.],
                {'units': 'hPa'}
            )
        }
    )
    cape, cin = surface_based_cape_cin(
        data['isobaric'],
        data['temperature'],
        data['dewpoint']
    )
    assert_almost_equal(cape, 75.0535446 * units('joule / kilogram'), 2)
    assert_almost_equal(cin, -136.685967 * units('joule / kilogram'), 2)


def test_profile_with_nans():
    """Test a profile with nans to make sure it calculates functions appropriately (#1187)."""
    pressure = np.array([1001, 1000, 997, 977.9, 977, 957, 937.8, 925, 906, 899.3, 887, 862.5,
                         854, 850, 800, 793.9, 785, 777, 771, 762, 731.8, 726, 703, 700, 655,
                         630, 621.2, 602, 570.7, 548, 546.8, 539, 513, 511, 485, 481, 468,
                         448, 439, 424, 420, 412]) * units.hPa
    temperature = np.array([-22.5, -22.7, -23.1, np.nan, -24.5, -25.1, np.nan, -24.5, -23.9,
                            np.nan, -24.7, np.nan, -21.3, -21.3, -22.7, np.nan, -20.7, -16.3,
                            -15.5, np.nan, np.nan, -15.3, np.nan, -17.3, -20.9, -22.5,
                            np.nan, -25.5, np.nan, -31.5, np.nan, -31.5, -34.1, -34.3,
                            -37.3, -37.7, -39.5, -42.1, -43.1, -45.1, -45.7, -46.7]
                           ) * units.degC
    dewpoint = np.array([-25.1, -26.1, -26.8, np.nan, -27.3, -28.2, np.nan, -27.2, -26.6,
                         np.nan, -27.4, np.nan, -23.5, -23.5, -25.1, np.nan, -22.9, -17.8,
                         -16.6, np.nan, np.nan, -16.4, np.nan, -18.5, -21, -23.7, np.nan,
                         -28.3, np.nan, -32.6, np.nan, -33.8, -35, -35.1, -38.1, -40,
                         -43.3, -44.6, -46.4, -47, -49.2, -50.7]) * units.degC
    lfc_p, _ = lfc(pressure, temperature, dewpoint)
    profile = parcel_profile(pressure, temperature[0], dewpoint[0])
    cape, cin = cape_cin(pressure, temperature, dewpoint, profile)
    sbcape, sbcin = surface_based_cape_cin(pressure, temperature, dewpoint)
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    assert_nan(lfc_p, units.hPa)
    assert_almost_equal(cape, 0 * units('J/kg'), 0)
    assert_almost_equal(cin, 0 * units('J/kg'), 0)
    assert_almost_equal(sbcape, 0 * units('J/kg'), 0)
    assert_almost_equal(sbcin, 0 * units('J/kg'), 0)
    assert_almost_equal(mucape, 0 * units('J/kg'), 0)
    assert_almost_equal(mucin, 0 * units('J/kg'), 0)


def test_most_unstable_cape_cin_surface():
    """Test the most unstable CAPE/CIN calculation when surface is most unstable."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    assert_almost_equal(mucape, 75.0535446 * units('joule / kilogram'), 2)
    assert_almost_equal(mucin, -136.685967 * units('joule / kilogram'), 2)


def test_most_unstable_cape_cin():
    """Test the most unstable CAPE/CIN calculation."""
    pressure = np.array([1000., 959., 867.9, 850., 825., 800.]) * units.mbar
    temperature = np.array([18.2, 22.2, 17.4, 10., 0., 15]) * units.celsius
    dewpoint = np.array([19., 19., 14.3, 0., -10., 0.]) * units.celsius
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    assert_almost_equal(mucape, 157.11404 * units('joule / kilogram'), 4)
    assert_almost_equal(mucin, -31.8406578 * units('joule / kilogram'), 4)


def test_mixed_parcel():
    """Test the mixed parcel calculation."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.degC
    parcel_pressure, parcel_temperature, parcel_dewpoint = mixed_parcel(pressure, temperature,
                                                                        dewpoint,
                                                                        depth=250 * units.hPa)
    assert_almost_equal(parcel_pressure, 959. * units.hPa, 6)
    assert_almost_equal(parcel_temperature, 28.7401463 * units.degC, 6)
    assert_almost_equal(parcel_dewpoint, 7.1534658 * units.degC, 6)


def test_mixed_layer_cape_cin(multiple_intersections):
    """Test the calculation of mixed layer cape/cin."""
    pressure, temperature, dewpoint = multiple_intersections
    mlcape, mlcin = mixed_layer_cape_cin(pressure, temperature, dewpoint)
    assert_almost_equal(mlcape, 987.7323 * units('joule / kilogram'), 2)
    assert_almost_equal(mlcin, -20.6727628 * units('joule / kilogram'), 2)


def test_mixed_layer():
    """Test the mixed layer calculation."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    mixed_layer_temperature = mixed_layer(pressure, temperature, depth=250 * units.hPa)[0]
    assert_almost_equal(mixed_layer_temperature, 16.4024930 * units.degC, 6)


def test_dry_static_energy():
    """Test the dry static energy calculation."""
    dse = dry_static_energy(1000 * units.m, 25 * units.degC)
    assert_almost_equal(dse, 309.3479 * units('kJ/kg'), 4)


def test_moist_static_energy():
    """Test the moist static energy calculation."""
    mse = moist_static_energy(1000 * units.m, 25 * units.degC, 0.012 * units.dimensionless)
    assert_almost_equal(mse, 339.35796 * units('kJ/kg'), 4)


def test_thickness_hydrostatic():
    """Test the thickness calculation for a moist layer."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    mixing = np.array([0.01458, 0.00209, 0.00224, 0.00240, 0.00256, 0.00010])
    thickness = thickness_hydrostatic(pressure, temperature, mixing_ratio=mixing)
    assert_almost_equal(thickness, 9891.706 * units.m, 2)


def test_thickness_hydrostatic_subset():
    """Test the thickness calculation with a subset of the moist layer."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    mixing = np.array([0.01458, 0.00209, 0.00224, 0.00240, 0.00256, 0.00010])
    thickness = thickness_hydrostatic(pressure, temperature, mixing_ratio=mixing,
                                      bottom=850 * units.hPa, depth=150 * units.hPa)
    assert_almost_equal(thickness, 1630.752 * units.m, 2)


def test_thickness_hydrostatic_isothermal():
    """Test the thickness calculation for a dry isothermal layer at 0 degC."""
    pressure = np.arange(1000, 500 - 1e-10, -10) * units.hPa
    temperature = np.zeros_like(pressure) * units.degC
    thickness = thickness_hydrostatic(pressure, temperature)
    assert_almost_equal(thickness, 5541.91 * units.m, 2)


def test_thickness_hydrostatic_isothermal_subset():
    """Test the thickness calculation for a dry isothermal layer subset at 0 degC."""
    pressure = np.arange(1000, 500 - 1e-10, -10) * units.hPa
    temperature = np.zeros_like(pressure) * units.degC
    thickness = thickness_hydrostatic(pressure, temperature, bottom=850 * units.hPa,
                                      depth=350 * units.hPa)
    assert_almost_equal(thickness, 4242.527 * units.m, 2)


def test_thickness_hydrostatic_from_relative_humidity():
    """Test the thickness calculation for a moist layer using RH data."""
    pressure = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.hPa
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.degC
    relative_humidity = np.array([81.69, 15.43, 18.95, 23.32, 28.36, 18.55]) * units.percent
    thickness = thickness_hydrostatic_from_relative_humidity(pressure, temperature,
                                                             relative_humidity)
    assert_almost_equal(thickness, 9891.71 * units.m, 2)


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
    assert (str(mixing_ratio_from_relative_humidity(p, temperature, rh).units)
            == 'dimensionless')


@pytest.fixture
def bv_data():
    """Return height and potential temperature data for testing Brunt-Vaisala functions."""
    heights = [1000., 1500., 2000., 2500.] * units('m')
    potential_temperatures = [[290., 290., 290., 290.],
                              [292., 293., 293., 292.],
                              [294., 296., 293., 293.],
                              [296., 295., 293., 296.]] * units('K')
    return heights, potential_temperatures


def test_brunt_vaisala_frequency_squared(bv_data):
    """Test Brunt-Vaisala frequency squared function."""
    truth = [[1.35264138e-04, 2.02896207e-04, 3.04344310e-04, 1.69080172e-04],
             [1.34337671e-04, 2.00818771e-04, 1.00409386e-04, 1.00753253e-04],
             [1.33423810e-04, 6.62611486e-05, 0, 1.33879181e-04],
             [1.32522297e-04, -1.99457288e-04, 0., 2.65044595e-04]] * units('s^-2')
    bv_freq_sqr = brunt_vaisala_frequency_squared(bv_data[0], bv_data[1])
    assert_almost_equal(bv_freq_sqr, truth, 6)


def test_brunt_vaisala_frequency(bv_data):
    """Test Brunt-Vaisala frequency function."""
    truth = [[0.01163031, 0.01424416, 0.01744547, 0.01300308],
             [0.01159041, 0.01417105, 0.01002045, 0.01003759],
             [0.01155092, 0.00814010, 0., 0.01157062],
             [0.01151183, np.nan, 0., 0.01628019]] * units('s^-1')
    bv_freq = brunt_vaisala_frequency(bv_data[0], bv_data[1])
    assert_almost_equal(bv_freq, truth, 6)


def test_brunt_vaisala_period(bv_data):
    """Test Brunt-Vaisala period function."""
    truth = [[540.24223556, 441.10593821, 360.16149037, 483.20734521],
             [542.10193894, 443.38165033, 627.03634320, 625.96540075],
             [543.95528431, 771.88106656, np.nan, 543.02940230],
             [545.80233643, np.nan, np.nan, 385.94053328]] * units('s')
    bv_period = brunt_vaisala_period(bv_data[0], bv_data[1])
    assert_almost_equal(bv_period, truth, 6)


@pytest.mark.parametrize('temp_units', ['degF', 'degC', 'K'])
def test_wet_bulb_temperature(temp_units):
    """Test wet bulb calculation with scalars."""
    temp = 25 * units.degC
    dewp = 15 * units.degC
    val = wet_bulb_temperature(1000 * units.hPa, temp.to(temp_units), dewp.to(temp_units))
    truth = 18.3432116 * units.degC  # 18.59 from NWS calculator
    assert_almost_equal(val, truth, 5)


def test_wet_bulb_temperature_saturated():
    """Test wet bulb calculation works properly with saturated conditions."""
    val = wet_bulb_temperature(850. * units.hPa, 17.6 * units.degC, 17.6 * units.degC)
    assert_almost_equal(val, 17.6 * units.degC, 7)


def test_wet_bulb_temperature_numpy_scalars():
    """Test wet bulb calculation with NumPy scalars, which have a shape attribute."""
    pressure = units.Quantity(np.float32(1000), 'hPa')
    temperature = units.Quantity(np.float32(25), 'degC')
    dewpoint = units.Quantity(np.float32(15), 'degC')
    val = wet_bulb_temperature(pressure, temperature, dewpoint)
    truth = 18.3432116 * units.degC
    assert_almost_equal(val, truth, 5)


def test_wet_bulb_temperature_1d():
    """Test wet bulb calculation with 1d list."""
    pressures = [1013, 1000, 990] * units.hPa
    temperatures = [25, 20, 15] * units.degC
    dewpoints = [20, 15, 10] * units.degC
    val = wet_bulb_temperature(pressures, temperatures, dewpoints)
    truth = [21.44487, 16.73673, 12.06554] * units.degC
    # 21.58, 16.86, 12.18 from NWS Calculator
    assert_array_almost_equal(val, truth, 5)


def test_wet_bulb_temperature_2d():
    """Test wet bulb calculation with 2d list."""
    pressures = [[1013, 1000, 990],
                 [1012, 999, 989]] * units.hPa
    temperatures = [[25, 20, 15],
                    [24, 19, 14]] * units.degC
    dewpoints = [[20, 15, 10],
                 [19, 14, 9]] * units.degC
    val = wet_bulb_temperature(pressures, temperatures, dewpoints)
    truth = [[21.44487, 16.73673, 12.06554],
             [20.50205, 15.80108, 11.13603]] * units.degC
    # 21.58, 16.86, 12.18
    # 20.6, 15.9, 11.2 from NWS Calculator
    assert_array_almost_equal(val, truth, 5)


@pytest.mark.parametrize('temp_units', ['degF', 'degC', 'K'])
def test_wet_bulb_nan(temp_units):
    """Test wet bulb calculation with nans."""
    pressure = [1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0,
                550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0, 150.0, 100.0, 70.0,
                50.0, 40.0, 30.0, 20.0, 15.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.4] * units.hPa
    dewpoint = [18.26819029083491, 17.461145433226555, 16.336553308346822, 13.115820973342743,
                6.143452735189838, -0.2623966010678873, -3.3370926082535926,
                -6.550658968811679, -9.53219938820244, -14.238884651017894, -22.66240089860936,
                -30.84751911458423, -33.68444429930677, -33.758997581695766,
                -44.84378577284087, -43.464276398501696, -60.91091595765517,
                -59.34560239336541, -77.78116414272493, -77.3133221938806, -82.27649829610058,
                -83.63567550304444, -84.52241968178798, np.nan, -90.08663879090643,
                -88.4437696852317, np.nan, -89.92561301616831, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan] * units.degC
    temperature = [21.38790283, 19.25753174, 17.29430542, 16.52999878, 16.88999023,
                   15.28999634, 12.13189697, 8.73193359, 5.1416748, 1.52350464, -2.15553589,
                   -6.14998169, -11.08999634, -17.96802979, -25.26040649, -32.41000061,
                   -41.09917908, -45.71267395, -48.46999512, -59.97120056, -74.6106781,
                   -79.78999329, -74.53684082, -63.49831848, -58.1831604, -53.39454041,
                   -48.2122467, -39.66942444, -34.30999451, -33.35002747, -23.31001892,
                   -14.55002441, -11.74678955, -25.58999634] * units.degC

    val = wet_bulb_temperature(pressure, temperature.to(temp_units), dewpoint.to(temp_units))
    truth = [19.238071735308814, 18.033294139060633, 16.65179610640866, 14.341431051131467,
             10.713278865013098, 7.2703265785039, 4.63234087372236, 1.8324379627895773,
             -1.0154897545814394, -4.337334561885717, -8.23596994210175, -11.902727397896111,
             -15.669313544076992, -20.78875735056887, -27.342629368898884, -33.42946313024462,
             -41.8026422159221, -46.17279371976847, -48.99179569697857, -60.13602538549741,
             -74.63516192059605, -79.80028104362006, -74.59295016865613, np.nan,
             -59.20059644897026, -55.76040402608365, np.nan, -49.692666440433335, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan] * units.degC
    assert_array_almost_equal(val, truth, 5)


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
    sigma = static_stability(pressures, temperatures, vertical_dim=1)
    truth = [[0.028203, 0.020182, 0.003077],
             [0.028097, 0.020008, 0.002774],
             [0.02841, 0.02368, 0.013184]] * units('J kg^-1 hPa^-2')
    assert_almost_equal(sigma, truth, 6)


def test_dewpoint_specific_humidity():
    """Test dewpoint from specific humidity."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    q = 0.012 * units.dimensionless
    td = dewpoint_from_specific_humidity(p, temperature, q)
    assert_almost_equal(td, 16.973 * units.degC, 3)


def test_dewpoint_specific_humidity_old_signature():
    """Test dewpoint from specific humidity using old signature issues specific error."""
    p = 1013.25 * units.mbar
    temperature = 20. * units.degC
    q = 0.012 * units.dimensionless
    with pytest.raises(ValueError, match='changed in 1.0'):
        dewpoint_from_specific_humidity(q, temperature, p)


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
    assert_almost_equal(lfc_pressure, 811.618879 * units.mbar, 3)
    assert_almost_equal(lfc_temp, 6.48644650 * units.celsius, 3)


@pytest.fixture
def multiple_intersections():
    """Create profile with multiple LFCs and ELs for testing."""
    levels = np.array([966., 937.2, 925., 904.6, 872.6, 853., 850., 836., 821., 811.6, 782.3,
                       754.2, 726.9, 700., 648.9, 624.6, 601.1, 595., 587., 576., 555.7,
                       534.2, 524., 500., 473.3, 400., 384.5, 358., 343., 308.3, 300., 276.,
                       273., 268.5, 250., 244.2, 233., 200.]) * units.mbar
    temperatures = np.array([18.2, 16.8, 16.2, 15.1, 13.3, 12.2, 12.4, 14., 14.4,
                             13.7, 11.4, 9.1, 6.8, 4.4, -1.4, -4.4, -7.3, -8.1,
                             -7.9, -7.7, -8.7, -9.8, -10.3, -13.5, -17.1, -28.1, -30.7,
                             -35.3, -37.1, -43.5, -45.1, -49.9, -50.4, -51.1, -54.1, -55.,
                             -56.7, -57.5]) * units.degC
    dewpoints = np.array([16.9, 15.9, 15.5, 14.2, 12.1, 10.8, 8.6, 0., -3.6, -4.4,
                          -6.9, -9.5, -12., -14.6, -15.8, -16.4, -16.9, -17.1, -27.9, -42.7,
                          -44.1, -45.6, -46.3, -45.5, -47.1, -52.1, -50.4, -47.3, -57.1,
                          -57.9, -58.1, -60.9, -61.4, -62.1, -65.1, -65.6,
                          -66.7, -70.5]) * units.degC
    return levels, temperatures, dewpoints


def test_multiple_lfcs_simple(multiple_intersections):
    """Test sounding with multiple LFCs.

    If which='top', return lowest-pressure LFC.
    If which='bottom', return the highest-pressure LFC.
    If which='all', return all LFCs

    """
    levels, temperatures, dewpoints = multiple_intersections
    lfc_pressure_top, lfc_temp_top = lfc(levels, temperatures, dewpoints)
    lfc_pressure_bottom, lfc_temp_bottom = lfc(levels, temperatures, dewpoints,
                                               which='bottom')
    lfc_pressure_all, _ = lfc(levels, temperatures, dewpoints, which='all')
    assert_almost_equal(lfc_pressure_top, 705.3534595 * units.mbar, 3)
    assert_almost_equal(lfc_temp_top, 4.8848999 * units.degC, 3)
    assert_almost_equal(lfc_pressure_bottom, 884.14790 * units.mbar, 3)
    assert_almost_equal(lfc_temp_bottom, 13.95707016 * units.degC, 3)
    assert_almost_equal(len(lfc_pressure_all), 2, 0)


def test_multiple_lfs_wide(multiple_intersections):
    """Test 'wide' LFC for sounding with multiple LFCs."""
    levels, temperatures, dewpoints = multiple_intersections
    lfc_pressure_wide, lfc_temp_wide = lfc(levels, temperatures, dewpoints, which='wide')
    assert_almost_equal(lfc_pressure_wide, 705.3534595 * units.hPa, 3)
    assert_almost_equal(lfc_temp_wide, 4.8848999 * units.degC, 3)


def test_invalid_which(multiple_intersections):
    """Test error message for invalid which option for LFC and EL."""
    levels, temperatures, dewpoints = multiple_intersections
    with pytest.raises(ValueError):
        lfc(levels, temperatures, dewpoints, which='test')
    with pytest.raises(ValueError):
        el(levels, temperatures, dewpoints, which='test')


def test_multiple_els_simple(multiple_intersections):
    """Test sounding with multiple ELs.

    If which='top', return lowest-pressure EL.
    If which='bottom', return the highest-pressure EL.
    If which='all', return all ELs

    """
    levels, temperatures, dewpoints = multiple_intersections
    el_pressure_top, el_temp_top = el(levels, temperatures, dewpoints)
    el_pressure_bottom, el_temp_bottom = el(levels, temperatures, dewpoints, which='bottom')
    el_pressure_all, _ = el(levels, temperatures, dewpoints, which='all')
    assert_almost_equal(el_pressure_top, 228.151466 * units.mbar, 3)
    assert_almost_equal(el_temp_top, -56.81015490 * units.degC, 3)
    assert_almost_equal(el_pressure_bottom, 849.7998947 * units.mbar, 3)
    assert_almost_equal(el_temp_bottom, 12.4233265 * units.degC, 3)
    assert_almost_equal(len(el_pressure_all), 2, 0)


def test_multiple_el_wide(multiple_intersections):
    """Test 'wide' EL for sounding with multiple ELs."""
    levels, temperatures, dewpoints = multiple_intersections
    el_pressure_wide, el_temp_wide = el(levels, temperatures, dewpoints, which='wide')
    assert_almost_equal(el_pressure_wide, 228.151466 * units.hPa, 3)
    assert_almost_equal(el_temp_wide, -56.81015490 * units.degC, 3)


def test_muliple_el_most_cape(multiple_intersections):
    """Test 'most_cape' EL for sounding with multiple ELs."""
    levels, temperatures, dewpoints = multiple_intersections
    el_pressure_wide, el_temp_wide = el(levels, temperatures, dewpoints, which='most_cape')
    assert_almost_equal(el_pressure_wide, 228.151466 * units.hPa, 3)
    assert_almost_equal(el_temp_wide, -56.81015490 * units.degC, 3)


def test_muliple_lfc_most_cape(multiple_intersections):
    """Test 'most_cape' LFC for sounding with multiple LFCs."""
    levels, temperatures, dewpoints = multiple_intersections
    lfc_pressure_wide, lfc_temp_wide = lfc(levels, temperatures, dewpoints, which='most_cape')
    assert_almost_equal(lfc_pressure_wide, 705.3534595 * units.hPa, 3)
    assert_almost_equal(lfc_temp_wide, 4.8848999 * units.degC, 3)


def test_el_lfc_most_cape_bottom():
    """Test 'most_cape' LFC/EL when the bottom combination produces the most CAPE."""
    levels = np.array([966., 937.2, 904.6, 872.6, 853., 850., 836., 821., 811.6, 782.3,
                       754.2, 726.9, 700., 648.9]) * units.mbar
    temperatures = np.array([18.2, 16.5, 15.1, 11.5, 11.0, 12.4, 14., 14.4,
                             13.7, 11.4, 9.1, 6.8, 3.8, 1.5]) * units.degC
    dewpoints = np.array([16.9, 15.9, 14.2, 11, 9.5, 8.6, 0., -3.6, -4.4,
                          -6.9, -9.5, -12., -14.6, -15.8]) * units.degC
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints, which='most_cape')
    el_pressure, el_temp = el(levels, temperatures, dewpoints, which='most_cape')
    assert_almost_equal(lfc_pressure, 900.73235 * units.hPa, 3)
    assert_almost_equal(lfc_temp, 14.672512 * units.degC, 3)
    assert_almost_equal(el_pressure, 849.7998947 * units.hPa, 3)
    assert_almost_equal(el_temp, 12.4233265 * units.degC, 3)


def test_cape_cin_top_el_lfc(multiple_intersections):
    """Test using LFC/EL options for CAPE/CIN."""
    levels, temperatures, dewpoints = multiple_intersections
    parcel_prof = parcel_profile(levels, temperatures[0], dewpoints[0]).to('degC')
    cape, cin = cape_cin(levels, temperatures, dewpoints, parcel_prof, which_lfc='top')
    assert_almost_equal(cape, 1258.94592 * units('joule / kilogram'), 3)
    assert_almost_equal(cin, -97.752333 * units('joule / kilogram'), 3)


def test_cape_cin_bottom_el_lfc(multiple_intersections):
    """Test using LFC/EL options for CAPE/CIN."""
    levels, temperatures, dewpoints = multiple_intersections
    parcel_prof = parcel_profile(levels, temperatures[0], dewpoints[0]).to('degC')
    cape, cin = cape_cin(levels, temperatures, dewpoints, parcel_prof, which_el='bottom')
    assert_almost_equal(cape, 2.18798 * units('joule / kilogram'), 3)
    assert_almost_equal(cin, -8.16221 * units('joule / kilogram'), 3)


def test_cape_cin_wide_el_lfc(multiple_intersections):
    """Test using LFC/EL options for CAPE/CIN."""
    levels, temperatures, dewpoints = multiple_intersections
    parcel_prof = parcel_profile(levels, temperatures[0], dewpoints[0]).to('degC')
    cape, cin = cape_cin(levels, temperatures, dewpoints, parcel_prof, which_lfc='wide',
                         which_el='wide')
    assert_almost_equal(cape, 1258.9459 * units('joule / kilogram'), 3)
    assert_almost_equal(cin, -97.752333 * units('joule / kilogram'), 3)


def test_cape_cin_custom_profile():
    """Test the CAPE and CIN calculation with a custom profile passed to LFC and EL."""
    p = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
    temperature = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.celsius
    dewpoint = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]) + 5 * units.delta_degC
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 1440.463208696 * units('joule / kilogram'), 2)
    assert_almost_equal(cin, 0.0 * units('joule / kilogram'), 2)


def test_parcel_profile_below_lcl():
    """Test parcel profile calculation when pressures do not reach LCL (#827)."""
    pressure = np.array([981, 949.2, 925., 913.9, 903, 879.4, 878, 864, 855,
                         850, 846.3, 838, 820, 814.5, 799, 794]) * units.hPa
    truth = np.array([276.35, 273.760341, 271.747753, 270.812026, 269.885225,
                      267.850849, 267.728946, 266.502214, 265.706084, 265.261201,
                      264.930782, 264.185801, 262.551884, 262.047526, 260.61294,
                      260.145932]) * units.kelvin
    profile = parcel_profile(pressure, 3.2 * units.degC, -10.8 * units.degC)
    assert_almost_equal(profile, truth, 6)


def test_vertical_velocity_pressure_dry_air():
    """Test conversion of w to omega assuming dry air."""
    w = 1 * units('cm/s')
    omega_truth = -1.25073619 * units('microbar/second')
    omega_test = vertical_velocity_pressure(w, 1000. * units.mbar, 273.15 * units.K)
    assert_almost_equal(omega_test, omega_truth, 6)


def test_vertical_velocity_dry_air():
    """Test conversion of w to omega assuming dry air."""
    omega = 1 * units('microbar/second')
    w_truth = -0.7995291 * units('cm/s')
    w_test = vertical_velocity(omega, 1000. * units.mbar, 273.15 * units.K)
    assert_almost_equal(w_test, w_truth, 6)


def test_vertical_velocity_pressure_moist_air():
    """Test conversion of w to omega assuming moist air."""
    w = -1 * units('cm/s')
    omega_truth = 1.032138077 * units('microbar/second')
    omega_test = vertical_velocity_pressure(w, 850. * units.mbar, 280. * units.K,
                                            8 * units('g/kg'))
    assert_almost_equal(omega_test, omega_truth, 6)


def test_vertical_velocity_moist_air():
    """Test conversion of w to omega assuming moist air."""
    omega = -1 * units('microbar/second')
    w_truth = 0.9688626185 * units('cm/s')
    w_test = vertical_velocity(omega, 850. * units.mbar, 280. * units.K, 8 * units('g/kg'))
    assert_almost_equal(w_test, w_truth, 6)


def test_specific_humidity_from_dewpoint():
    """Specific humidity from dewpoint."""
    p = 1013.25 * units.mbar
    q = specific_humidity_from_dewpoint(p, 16.973 * units.degC)
    assert_almost_equal(q, 0.012 * units.dimensionless, 3)


def test_lcl_convergence_issue():
    """Test profile where LCL wouldn't converge (#1187)."""
    pressure = np.array([990, 973, 931, 925, 905]) * units.hPa
    temperature = np.array([14.4, 14.2, 13, 12.6, 11.4]) * units.degC
    dewpoint = np.array([14.4, 11.7, 8.2, 7.8, 7.6]) * units.degC
    lcl_pressure, _ = lcl(pressure[0], temperature[0], dewpoint[0])
    assert_almost_equal(lcl_pressure, 990 * units.hPa, 0)


def test_cape_cin_value_error():
    """Test a profile that originally caused a ValueError in #1190."""
    pressure = np.array([1012.0, 1009.0, 1002.0, 1000.0, 925.0, 896.0, 855.0, 850.0, 849.0,
                         830.0, 775.0, 769.0, 758.0, 747.0, 741.0, 731.0, 712.0, 700.0, 691.0,
                         671.0, 636.0, 620.0, 610.0, 601.0, 594.0, 587.0, 583.0, 580.0, 571.0,
                         569.0, 554.0, 530.0, 514.0, 506.0, 502.0, 500.0, 492.0, 484.0, 475.0,
                         456.0, 449.0, 442.0, 433.0, 427.0, 400.0, 395.0, 390.0, 351.0, 300.0,
                         298.0, 294.0, 274.0, 250.0]) * units.hPa
    temperature = np.array([27.8, 25.8, 24.2, 24, 18.8, 16, 13, 12.6, 12.6, 11.6, 9.2, 8.6,
                            8.4, 9.2, 10, 9.4, 7.4, 6.2, 5.2, 3.2, -0.3, -2.3, -3.3, -4.5,
                            -5.5, -6.1, -6.1, -6.1, -6.3, -6.3, -7.7, -9.5, -9.9, -10.3,
                            -10.9, -11.1, -11.9, -12.7, -13.7, -16.1, -16.9, -17.9, -19.1,
                            -19.9, -23.9, -24.7, -25.3, -29.5, -39.3, -39.7, -40.5, -44.3,
                            -49.3]) * units.degC
    dewpoint = np.array([19.8, 16.8, 16.2, 16, 13.8, 12.8, 10.1, 9.7, 9.7,
                         8.6, 4.2, 3.9, 0.4, -5.8, -32, -34.6, -35.6, -34.8,
                         -32.8, -10.8, -9.3, -10.3, -9.3, -10.5, -10.5, -10, -16.1,
                         -19.1, -23.3, -18.3, -17.7, -20.5, -27.9, -32.3, -33.9, -34.1,
                         -35.9, -26.7, -37.7, -43.1, -33.9, -40.9, -46.1, -34.9, -33.9,
                         -33.7, -33.3, -42.5, -50.3, -49.7, -49.5, -58.3, -61.3]) * units.degC
    cape, cin = surface_based_cape_cin(pressure, temperature, dewpoint)
    expected_cape, expected_cin = 2007.040698 * units('joules/kg'), 0.0 * units('joules/kg')
    assert_almost_equal(cape, expected_cape, 3)
    assert_almost_equal(cin, expected_cin, 3)


def test_lcl_grid_surface_lcls():
    """Test surface grid where some values have LCLs at the surface."""
    pressure = np.array([1000, 990, 1010]) * units.hPa
    temperature = np.array([15, 14, 13]) * units.degC
    dewpoint = np.array([15, 10, 13]) * units.degC
    lcl_pressure, lcl_temperature = lcl(pressure, temperature, dewpoint)
    pres_truth = np.array([1000, 932.1719, 1010]) * units.hPa
    temp_truth = np.array([15, 9.10424, 13]) * units.degC
    assert_array_almost_equal(lcl_pressure, pres_truth, 4)
    assert_array_almost_equal(lcl_temperature, temp_truth, 4)


@pytest.fixture()
def index_xarray_data():
    """Create data for testing that index calculations work with xarray data."""
    pressure = xr.DataArray([850., 700., 500.], dims=('isobaric',), attrs={'units': 'hPa'})
    temp = xr.DataArray([[[[296., 295., 294.], [293., 292., 291.]],
                          [[286., 285., 284.], [283., 282., 281.]],
                          [[276., 275., 274.], [273., 272., 271.]]]] * units.K,
                        dims=('time', 'isobaric', 'y', 'x'))

    profile = xr.DataArray([[[[289., 288., 287.], [286., 285., 284.]],
                             [[279., 278., 277.], [276., 275., 274.]],
                             [[269., 268., 267.], [266., 265., 264.]]]] * units.K,
                           dims=('time', 'isobaric', 'y', 'x'))

    dewp = xr.DataArray([[[[294., 293., 292.], [291., 290., 289.]],
                          [[284., 283., 282.], [281., 280., 279.]],
                          [[274., 273., 272.], [271., 270., 269.]]]] * units.K,
                        dims=('time', 'isobaric', 'y', 'x'))

    dirw = xr.DataArray([[[[180., 180., 180.], [180., 180., 180.]],
                          [[225., 225., 225.], [225., 225., 225.]],
                          [[270., 270., 270.], [270., 270., 270.]]]] * units.degree,
                        dims=('time', 'isobaric', 'y', 'x'))

    speed = xr.DataArray([[[[20., 20., 20.], [20., 20., 20.]],
                           [[25., 25., 25.], [25., 25., 25.]],
                           [[50., 50., 50.], [50., 50., 50.]]]] * units.knots,
                         dims=('time', 'isobaric', 'y', 'x'))

    return xr.Dataset({'temperature': temp, 'profile': profile, 'dewpoint': dewp,
                       'wind_direction': dirw, 'wind_speed': speed},
                      coords={'isobaric': pressure, 'time': ['2020-01-01T00:00Z']})


def test_lifted_index():
    """Test the Lifted Index calculation."""
    pressure = np.array([1014., 1000., 997., 981.2, 947.4, 925., 914.9, 911.,
                         902., 883., 850., 822.3, 816., 807., 793.2, 770.,
                         765.1, 753., 737.5, 737., 713., 700., 688., 685.,
                         680., 666., 659.8, 653., 643., 634., 615., 611.8,
                         566.2, 516., 500., 487., 484.2, 481., 475., 460.,
                         400.]) * units.hPa
    temperature = np.array([24.2, 24.2, 24., 23.1, 21., 19.6, 18.7, 18.4,
                            19.2, 19.4, 17.2, 15.3, 14.8, 14.4, 13.4, 11.6,
                            11.1, 10., 8.8, 8.8, 8.2, 7., 5.6, 5.6,
                            5.6, 4.4, 3.8, 3.2, 3., 3.2, 1.8, 1.5,
                            -3.4, -9.3, -11.3, -13.1, -13.1, -13.1, -13.7, -15.1,
                            -23.5]) * units.degC
    dewpoint = np.array([23.2, 23.1, 22.8, 22., 20.2, 19., 17.6, 17.,
                         16.8, 15.5, 14., 11.7, 11.2, 8.4, 7., 4.6,
                         5., 6., 4.2, 4.1, -1.8, -2., -1.4, -0.4,
                         -3.4, -5.6, -4.3, -2.8, -7., -25.8, -31.2, -31.4,
                         -34.1, -37.3, -32.3, -34.1, -37.3, -41.1, -37.7, -58.1,
                         -57.5]) * units.degC
    parcel_prof = parcel_profile(pressure, temperature[0], dewpoint[0])
    li = lifted_index(pressure, temperature, parcel_prof)
    assert_almost_equal(li, -7.9176350 * units.delta_degree_Celsius, 2)


def test_lifted_index_500hpa_missing():
    """Test the Lifted Index calculation when data at 500 hpa is missing."""
    pressure = np.array([1014., 1000., 997., 981.2, 947.4, 925., 914.9, 911.,
                         902., 883., 850., 822.3, 816., 807., 793.2, 770.,
                         765.1, 753., 737.5, 737., 713., 700., 688., 685.,
                         680., 666., 659.8, 653., 643., 634., 615., 611.8,
                         566.2, 516., 487., 484.2, 481., 475., 460.,
                         400.]) * units.hPa
    temperature = np.array([24.2, 24.2, 24., 23.1, 21., 19.6, 18.7, 18.4,
                            19.2, 19.4, 17.2, 15.3, 14.8, 14.4, 13.4, 11.6,
                            11.1, 10., 8.8, 8.8, 8.2, 7., 5.6, 5.6,
                            5.6, 4.4, 3.8, 3.2, 3., 3.2, 1.8, 1.5,
                            -3.4, -9.3, -13.1, -13.1, -13.1, -13.7, -15.1,
                            -23.5]) * units.degC
    dewpoint = np.array([23.2, 23.1, 22.8, 22., 20.2, 19., 17.6, 17.,
                         16.8, 15.5, 14., 11.7, 11.2, 8.4, 7., 4.6,
                         5., 6., 4.2, 4.1, -1.8, -2., -1.4, -0.4,
                         -3.4, -5.6, -4.3, -2.8, -7., -25.8, -31.2, -31.4,
                         -34.1, -37.3, -34.1, -37.3, -41.1, -37.7, -58.1,
                         -57.5]) * units.degC
    parcel_prof = parcel_profile(pressure, temperature[0], dewpoint[0])
    li = lifted_index(pressure, temperature, parcel_prof)
    assert_almost_equal(li, -7.9176350 * units.delta_degree_Celsius, 1)


def test_lifted_index_xarray(index_xarray_data):
    """Test lifted index with a grid of xarray data."""
    result = lifted_index(index_xarray_data.isobaric, index_xarray_data.temperature,
                          index_xarray_data.profile)
    assert_array_almost_equal(result, np.full((1, 1, 2, 3), 7) * units.delta_degC)


def test_k_index():
    """Test the K Index calculation."""
    pressure = np.array([1014., 1000., 997., 981.2, 947.4, 925., 914.9, 911.,
                         902., 883., 850., 822.3, 816., 807., 793.2, 770.,
                         765.1, 753., 737.5, 737., 713., 700., 688., 685.,
                         680., 666., 659.8, 653., 643., 634., 615., 611.8,
                         566.2, 516., 500., 487., 484.2, 481., 475., 460.,
                         400.]) * units.hPa
    temperature = np.array([24.2, 24.2, 24., 23.1, 21., 19.6, 18.7, 18.4,
                            19.2, 19.4, 17.2, 15.3, 14.8, 14.4, 13.4, 11.6,
                            11.1, 10., 8.8, 8.8, 8.2, 7., 5.6, 5.6,
                            5.6, 4.4, 3.8, 3.2, 3., 3.2, 1.8, 1.5,
                            -3.4, -9.3, -11.3, -13.1, -13.1, -13.1, -13.7, -15.1,
                            -23.5]) * units.degC
    dewpoint = np.array([23.2, 23.1, 22.8, 22., 20.2, 19., 17.6, 17.,
                         16.8, 15.5, 14., 11.7, 11.2, 8.4, 7., 4.6,
                         5., 6., 4.2, 4.1, -1.8, -2., -1.4, -0.4,
                         -3.4, -5.6, -4.3, -2.8, -7., -25.8, -31.2, -31.4,
                         -34.1, -37.3, -32.3, -34.1, -37.3, -41.1, -37.7, -58.1,
                         -57.5]) * units.degC
    ki = k_index(pressure, temperature, dewpoint)
    assert_almost_equal(ki, 33.5 * units.degC, 2)


def test_k_index_xarray(index_xarray_data):
    """Test the K index calculation with a grid of xarray data."""
    result = k_index(index_xarray_data.isobaric, index_xarray_data.temperature,
                     index_xarray_data.dewpoint)
    assert_array_almost_equal(result,
                              np.array([[[312., 311., 310.], [309., 308., 307.]]]) * units.K)


def test_gradient_richardson_number():
    """Test gradient Richardson number calculation."""
    theta = units('K') * np.asarray([254.5, 258.3, 262.2])
    u_wnd = units('m/s') * np.asarray([-2., -1.1, 0.23])
    v_wnd = units('m/s') * np.asarray([3.3, 4.2, 5.2])
    height = units('km') * np.asarray([0.2, 0.4, 0.6])

    result = gradient_richardson_number(height, theta, u_wnd, v_wnd)
    expected = np.asarray([24.2503551, 13.6242603, 8.4673744])

    assert_array_almost_equal(result, expected, 4)


def test_gradient_richardson_number_with_xarray():
    """Test gradient Richardson number calculation using xarray."""
    data = xr.Dataset(
        {
            'theta': (('height',), [254.5, 258.3, 262.2] * units.K),
            'u_wind': (('height',), [-2., -1.1, 0.23] * units('m/s')),
            'v_wind': (('height',), [3.3, 4.2, 5.2] * units('m/s')),
            'Ri_g': (('height',), [24.2503551, 13.6242603, 8.4673744])
        },
        coords={'height': (('height',), [0.2, 0.4, 0.6], {'units': 'kilometer'})}
    )

    result = gradient_richardson_number(
        data['height'],
        data['theta'],
        data['u_wind'],
        data['v_wind']
    )

    assert isinstance(result, xr.DataArray)
    xr.testing.assert_identical(result['height'], data['Ri_g']['height'])
    assert_array_almost_equal(result.data.m_as(''), data['Ri_g'].data)


def test_showalter_index():
    """Test the Showalter index calculation."""
    pressure = units.Quantity(np.array([931.0, 925.0, 911.0, 891.0, 886.9, 855.0, 850.0, 825.6,
                                        796.3, 783.0, 768.0, 759.0, 745.0, 740.4, 733.0, 715.0,
                                        700.0, 695.0, 687.2, 684.0, 681.0, 677.0, 674.0, 661.9,
                                        657.0, 639.0, 637.6, 614.0, 592.0, 568.9, 547.4, 526.8,
                                        500.0, 487.5, 485.0]), 'hPa')
    temps = units.Quantity(np.array([18.4, 19.8, 20.0, 19.6, 19.3, 16.8, 16.4, 15.1, 13.4,
                                     12.6, 11.2, 10.4, 8.6, 8.3, 7.8, 5.8, 4.6, 4.2, 3.4, 3.0,
                                     3.0, 4.4, 5.0, 5.1, 5.2, 3.4, 3.3, 2.4, 1.4, -0.4, -2.2,
                                     -3.9, -6.3, -7.6, -7.9]), 'degC')
    dewp = units.Quantity(np.array([9.4, 8.8, 6.0, 8.6, 8.4, 6.8, 6.4, 4.0, 1.0, -0.4, -1.1,
                                    -1.6, 1.6, -0.2, -3.2, -3.2, -4.4, -2.8, -3.6, -4.0, -6.0,
                                    -17.6, -25.0, -31.2, -33.8, -29.6, -30.1, -39.0, -47.6,
                                    -48.9, -50.2, -51.5, -53.3, -55.5, -55.9]), 'degC')

    result = showalter_index(pressure, temps, dewp)
    assert_almost_equal(result, units.Quantity(7.6024, 'delta_degC'), 4)


def test_total_totals_index():
    """Test the Total Totals Index calculation."""
    pressure = np.array([1008., 1000., 947., 925., 921., 896., 891., 889., 866.,
                         858., 850., 835., 820., 803., 733., 730., 700., 645.,
                         579., 500., 494., 466., 455., 441., 433., 410., 409.,
                         402., 400., 390., 388., 384., 381., 349., 330., 320.,
                         306., 300., 278., 273., 250., 243., 208., 200., 196.,
                         190., 179., 159., 151., 150., 139.]) * units.hPa
    temperature = np.array([27.4, 26.4, 22.9, 21.4, 21.2, 20.7, 20.6, 21.2, 19.4,
                            19.1, 18.8, 17.8, 17.4, 16.3, 11.4, 11.2, 10.2, 6.1,
                            0.6, -4.9, -5.5, -8.5, -9.9, -11.7, -12.3, -13.7, -13.8,
                            -14.9, -14.9, -16.1, -16.1, -16.9, -17.3, -21.7, -24.5, -26.1,
                            -28.3, -29.5, -33.1, -34.2, -39.3, -41., -50.2, -52.5, -53.5,
                            -55.2, -58.6, -65.2, -68.1, -68.5, -72.5]) * units.degC
    dewpoint = np.array([24.9, 24.6, 22., 20.9, 20.7, 14.8, 13.6, 12.2, 16.8,
                         16.6, 16.5, 15.9, 13.6, 13.2, 11.3, 11.2, 8.6, 4.5,
                         -0.8, -8.1, -9.5, -12.7, -12.7, -12.8, -13.1, -24.7, -24.4,
                         -21.9, -24.9, -36.1, -31.1, -26.9, -27.4, -33., -36.5, -47.1,
                         -31.4, -33.5, -40.1, -40.8, -44.1, -45.6, -54., -56.1, -56.9,
                         -58.6, -61.9, -68.4, -71.2, -71.6, -77.2]) * units.degC

    tt = total_totals_index(pressure, temperature, dewpoint)
    assert_almost_equal(tt, 45.10 * units.delta_degC, 2)


def test_total_totals_index_xarray(index_xarray_data):
    """Test the total totals index calculation with a grid of xarray data."""
    result = total_totals_index(index_xarray_data.isobaric, index_xarray_data.temperature,
                                index_xarray_data.dewpoint)
    assert_array_almost_equal(result, np.full((1, 2, 3), 38.) * units.K)


def test_vertical_totals():
    """Test the Vertical Totals calculation."""
    pressure = np.array([1008., 1000., 947., 925., 921., 896., 891., 889., 866.,
                         858., 850., 835., 820., 803., 733., 730., 700., 645.,
                         579., 500., 494., 466., 455., 441., 433., 410., 409.,
                         402., 400., 390., 388., 384., 381., 349., 330., 320.,
                         306., 300., 278., 273., 250., 243., 208., 200., 196.,
                         190., 179., 159., 151., 150., 139.]) * units.hPa
    temperature = np.array([27.4, 26.4, 22.9, 21.4, 21.2, 20.7, 20.6, 21.2, 19.4,
                            19.1, 18.8, 17.8, 17.4, 16.3, 11.4, 11.2, 10.2, 6.1,
                            0.6, -4.9, -5.5, -8.5, -9.9, -11.7, -12.3, -13.7, -13.8,
                            -14.9, -14.9, -16.1, -16.1, -16.9, -17.3, -21.7, -24.5, -26.1,
                            -28.3, -29.5, -33.1, -34.2, -39.3, -41., -50.2, -52.5, -53.5,
                            -55.2, -58.6, -65.2, -68.1, -68.5, -72.5]) * units.degC

    vt = vertical_totals(pressure, temperature)
    assert_almost_equal(vt, 23.70 * units.delta_degC, 2)


def test_vertical_totals_index_xarray(index_xarray_data):
    """Test the vertical totals index calculation with a grid of xarray data."""
    result = vertical_totals(index_xarray_data.isobaric, index_xarray_data.temperature)
    assert_array_almost_equal(result, np.full((1, 2, 3), 20.) * units.K)


def test_cross_totals():
    """Test the Cross Totals calculation."""
    pressure = np.array([1008., 1000., 947., 925., 921., 896., 891., 889., 866.,
                         858., 850., 835., 820., 803., 733., 730., 700., 645.,
                         579., 500., 494., 466., 455., 441., 433., 410., 409.,
                         402., 400., 390., 388., 384., 381., 349., 330., 320.,
                         306., 300., 278., 273., 250., 243., 208., 200., 196.,
                         190., 179., 159., 151., 150., 139.]) * units.hPa
    temperature = np.array([27.4, 26.4, 22.9, 21.4, 21.2, 20.7, 20.6, 21.2, 19.4,
                            19.1, 18.8, 17.8, 17.4, 16.3, 11.4, 11.2, 10.2, 6.1,
                            0.6, -4.9, -5.5, -8.5, -9.9, -11.7, -12.3, -13.7, -13.8,
                            -14.9, -14.9, -16.1, -16.1, -16.9, -17.3, -21.7, -24.5, -26.1,
                            -28.3, -29.5, -33.1, -34.2, -39.3, -41., -50.2, -52.5, -53.5,
                            -55.2, -58.6, -65.2, -68.1, -68.5, -72.5]) * units.degC
    dewpoint = np.array([24.9, 24.6, 22., 20.9, 20.7, 14.8, 13.6, 12.2, 16.8,
                         16.6, 16.5, 15.9, 13.6, 13.2, 11.3, 11.2, 8.6, 4.5,
                         -0.8, -8.1, -9.5, -12.7, -12.7, -12.8, -13.1, -24.7, -24.4,
                         -21.9, -24.9, -36.1, -31.1, -26.9, -27.4, -33., -36.5, -47.1,
                         -31.4, -33.5, -40.1, -40.8, -44.1, -45.6, -54., -56.1, -56.9,
                         -58.6, -61.9, -68.4, -71.2, -71.6, -77.2]) * units.degC

    ct = cross_totals(pressure, temperature, dewpoint)
    assert_almost_equal(ct, 21.40 * units.delta_degC, 2)


def test_cross_totals_index_xarray(index_xarray_data):
    """Test the cross totals index calculation with a grid of xarray data."""
    result = cross_totals(index_xarray_data.isobaric, index_xarray_data.temperature,
                          index_xarray_data.dewpoint)
    assert_array_almost_equal(result, np.full((1, 2, 3), 18.) * units.K)


def test_parcel_profile_drop_duplicates():
    """Test handling repeat pressures in moist region of profile."""
    pressure = np.array([962., 951., 937.9, 925., 908., 905.7, 894., 875.,
                         41.3, 40.8, 37., 36.8, 32., 30., 27.7, 27.7, 26.4]) * units.hPa

    temperature = units.Quantity(19.6, 'degC')

    dewpoint = units.Quantity(18.6, 'degC')

    truth = np.array([292.75, 291.78965331, 291.12778784, 290.61996294,
                      289.93681828, 289.84313902, 289.36183185, 288.5626898,
                      135.46280886, 134.99220142, 131.27369084, 131.07055878,
                      125.93977169, 123.63877507, 120.85291224, 120.85291224,
                      119.20448296]) * units.kelvin

    with pytest.warns(UserWarning, match='Duplicate pressure'):
        profile = parcel_profile(pressure, temperature, dewpoint)
        assert_almost_equal(profile, truth, 5)


def test_parcel_profile_with_lcl_as_dataset_duplicates():
    """Test that parcel profile dataset creation works with duplicate pressures in profile."""
    pressure = np.array(
        [951., 951., 937.9, 925., 908., 30., 27.7, 27.7, 26.4, 25.1]
    ) * units.hPa

    temperature = np.array(
        [20., 20., 19.5, 19., 18.6, -58.5, -58.1, -58.1, -57.2, -56.2]
    ) * units.degC

    dewpoint = np.array(
        [19.4, 19.4, 19., 18.6, 18.3, -73.5, -75.1, -75.1, -77., -78.8]
    ) * units.degC

    truth = xr.Dataset(
        {
            'ambient_temperature': (
                ('isobaric',),
                np.insert(temperature.m, 2, 19.679237747615478) * units.degC
            ),
            'ambient_dew_point': (
                ('isobaric',),
                np.insert(dewpoint.m, 2, 19.143390198092384) * units.degC
            ),
            'parcel_temperature': (
                ('isobaric',),
                [
                    293.15, 293.15, 292.40749167, 292.22841462, 291.73069653, 291.06139433,
                    125.22698955, 122.40534065, 122.40534065, 120.73573642, 119.0063293
                ] * units.kelvin
            )
        },
        coords={
            'isobaric': (
                'isobaric',
                np.insert(pressure.m, 2, 942.6)
            )
        }
    )

    profile = parcel_profile_with_lcl_as_dataset(pressure, temperature, dewpoint)

    xr.testing.assert_allclose(profile, truth, atol=1e-5)


def test_sweat_index():
    """Test the SWEAT Index calculation."""
    pressure = np.array([1008., 1000., 947., 925., 921., 896., 891., 889., 866.,
                         858., 850., 835., 820., 803., 733., 730., 700., 645.,
                         579., 500., 494., 466., 455., 441., 433., 410., 409.,
                         402., 400., 390., 388., 384., 381., 349., 330., 320.,
                         306., 300., 278., 273., 250., 243., 208., 200., 196.,
                         190., 179., 159., 151., 150., 139.]) * units.hPa
    temperature = np.array([27.4, 26.4, 22.9, 21.4, 21.2, 20.7, 20.6, 21.2, 19.4,
                            19.1, 18.8, 17.8, 17.4, 16.3, 11.4, 11.2, 10.2, 6.1,
                            0.6, -4.9, -5.5, -8.5, -9.9, -11.7, -12.3, -13.7, -13.8,
                            -14.9, -14.9, -16.1, -16.1, -16.9, -17.3, -21.7, -24.5, -26.1,
                            -28.3, -29.5, -33.1, -34.2, -39.3, -41., -50.2, -52.5, -53.5,
                            -55.2, -58.6, -65.2, -68.1, -68.5, -72.5]) * units.degC
    dewpoint = np.array([24.9, 24.6, 22., 20.9, 20.7, 14.8, 13.6, 12.2, 16.8,
                         16.6, 16.5, 15.9, 13.6, 13.2, 11.3, 11.2, 8.6, 4.5,
                         -0.8, -8.1, -9.5, -12.7, -12.7, -12.8, -13.1, -24.7, -24.4,
                         -21.9, -24.9, -36.1, -31.1, -26.9, -27.4, -33., -36.5, -47.1,
                         -31.4, -33.5, -40.1, -40.8, -44.1, -45.6, -54., -56.1, -56.9,
                         -58.6, -61.9, -68.4, -71.2, -71.6, -77.2]) * units.degC
    speed = np.array([0., 3., 10., 12., 12., 14., 14., 14., 12.,
                      12., 12., 12., 11., 11., 12., 12., 10., 10.,
                      8., 5., 4., 1., 0., 3., 5., 10., 10.,
                      11., 11., 13., 14., 14., 15., 23., 23., 24.,
                      24., 24., 26., 27., 28., 30., 25., 24., 26.,
                      28., 33., 29., 32., 26., 26.]) * units.knot
    direction = np.array([0., 170., 200., 205., 204., 200., 197., 195., 180.,
                          175., 175., 178., 181., 185., 160., 160., 165., 165.,
                          203., 255., 268., 333., 0., 25., 40., 83., 85.,
                          89., 90., 100., 103., 107., 110., 90., 88., 87.,
                          86., 85., 85., 85., 60., 55., 60., 50., 46.,
                          40., 45., 35., 50., 50., 50.]) * units.degree

    sweat = sweat_index(pressure, temperature, dewpoint, speed, direction)
    assert_almost_equal(sweat, 227., 2)


def test_sweat_index_xarray(index_xarray_data):
    """Test the SWEAT index calculation with a grid of xarray data."""
    result = sweat_index(index_xarray_data.isobaric, index_xarray_data.temperature,
                         index_xarray_data.dewpoint, index_xarray_data.wind_speed,
                         index_xarray_data.wind_direction)
    assert_array_almost_equal(result, np.array([[[[490.2, 478.2, 466.2],
                                                  [454.2, 442.2, 430.2]]]]))
