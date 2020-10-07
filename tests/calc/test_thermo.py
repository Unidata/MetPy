# Copyright (c) 2008,2015,2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `thermo` module."""

import numpy as np
import pytest
import xarray as xr

from metpy.calc import (
    brunt_vaisala_frequency,
    brunt_vaisala_frequency_squared,
    brunt_vaisala_period,
    cape_cin,
    density,
    dewpoint,
    dewpoint_from_relative_humidity,
    dewpoint_from_specific_humidity,
    dry_lapse,
    dry_static_energy,
    el,
    equivalent_potential_temperature,
    exner_function,
    gradient_richardson_number,
    isentropic_interpolation,
    isentropic_interpolation_as_dataset,
    lcl,
    lfc,
    lifted_index,
    mixed_layer,
    mixed_layer_cape_cin,
    mixed_parcel,
    mixing_ratio,
    mixing_ratio_from_relative_humidity,
    mixing_ratio_from_specific_humidity,
    moist_lapse,
    moist_static_energy,
    most_unstable_cape_cin,
    most_unstable_parcel,
    parcel_profile,
    parcel_profile_with_lcl,
    parcel_profile_with_lcl_as_dataset,
    potential_temperature,
    psychrometric_vapor_pressure_wet,
    relative_humidity_from_dewpoint,
    relative_humidity_from_mixing_ratio,
    relative_humidity_from_specific_humidity,
    relative_humidity_wet_psychrometric,
    saturation_equivalent_potential_temperature,
    saturation_mixing_ratio,
    saturation_vapor_pressure,
    specific_humidity_from_dewpoint,
    specific_humidity_from_mixing_ratio,
    static_stability,
    surface_based_cape_cin,
    temperature_from_potential_temperature,
    thickness_hydrostatic,
    thickness_hydrostatic_from_relative_humidity,
    vapor_pressure,
    vertical_velocity,
    vertical_velocity_pressure,
    virtual_potential_temperature,
    virtual_temperature,
    wet_bulb_temperature,
)
from metpy.calc.thermo import _find_append_zero_crossings
from metpy.testing import assert_almost_equal, assert_array_almost_equal, assert_nan
from metpy.units import masked_array, units


def test_relative_humidity_from_dewpoint():
    """Test Relative Humidity calculation."""
    assert_almost_equal(
        relative_humidity_from_dewpoint(25.0 * units.degC, 15.0 * units.degC),
        53.80 * units.percent,
        2,
    )


def test_relative_humidity_from_dewpoint_with_f():
    """Test Relative Humidity accepts temperature in Fahrenheit."""
    assert_almost_equal(
        relative_humidity_from_dewpoint(70.0 * units.degF, 55.0 * units.degF),
        58.935 * units.percent,
        3,
    )


def test_relative_humidity_from_dewpoint_xarray():
    """Test Relative Humidity with xarray data arrays (quantified and unquantified)."""
    temp = xr.DataArray(25.0, attrs={"units": "degC"})
    dewp = xr.DataArray([15.0] * units.degC)
    assert_almost_equal(relative_humidity_from_dewpoint(temp, dewp), 53.80 * units.percent, 2)


def test_exner_function():
    """Test Exner function calculation."""
    pres = np.array([900.0, 500.0, 300.0, 100.0]) * units.mbar
    truth = np.array([0.9703542, 0.8203834, 0.7090065, 0.518048]) * units.dimensionless
    assert_array_almost_equal(exner_function(pres), truth, 6)


def test_potential_temperature():
    """Test potential_temperature calculation."""
    temp = np.array([278.0, 283.0, 291.0, 298.0]) * units.kelvin
    pres = np.array([900.0, 500.0, 300.0, 100.0]) * units.mbar
    real_th = np.array([286.493, 344.961, 410.4335, 575.236]) * units.kelvin
    assert_array_almost_equal(potential_temperature(pres, temp), real_th, 3)


def test_temperature_from_potential_temperature():
    """Test temperature_from_potential_temperature calculation."""
    theta = np.array([286.12859679, 288.22362587, 290.31865495, 292.41368403]) * units.kelvin
    pres = np.array([850] * 4) * units.mbar
    real_t = np.array([273.15, 275.15, 277.15, 279.15]) * units.kelvin
    assert_array_almost_equal(temperature_from_potential_temperature(pres, theta), real_t, 2)


def test_scalar():
    """Test potential_temperature accepts scalar values."""
    assert_almost_equal(
        potential_temperature(1000.0 * units.mbar, 293.0 * units.kelvin),
        293.0 * units.kelvin,
        4,
    )
    assert_almost_equal(
        potential_temperature(800.0 * units.mbar, 293.0 * units.kelvin),
        312.2828 * units.kelvin,
        4,
    )


def test_fahrenheit():
    """Test that potential_temperature handles temperature values in Fahrenheit."""
    assert_almost_equal(
        potential_temperature(800.0 * units.mbar, 68.0 * units.degF),
        (312.444 * units.kelvin).to(units.degF),
        2,
    )


def test_pot_temp_inhg():
    """Test that potential_temperature can handle pressure not in mb (issue #165)."""
    assert_almost_equal(
        potential_temperature(29.92 * units.inHg, 29 * units.degC),
        301.019735 * units.kelvin,
        4,
    )


def test_dry_lapse():
    """Test dry_lapse calculation."""
    levels = np.array([1000, 900, 864.89]) * units.mbar
    temps = dry_lapse(levels, 303.15 * units.kelvin)
    assert_array_almost_equal(temps, np.array([303.15, 294.16, 290.83]) * units.kelvin, 2)


def test_dry_lapse_2_levels():
    """Test dry_lapse calculation when given only two levels."""
    temps = dry_lapse(np.array([1000.0, 500.0]) * units.mbar, 293.0 * units.kelvin)
    assert_array_almost_equal(temps, [293.0, 240.3723] * units.kelvin, 4)


def test_moist_lapse():
    """Test moist_lapse calculation."""
    temp = moist_lapse(
        np.array([1000.0, 800.0, 600.0, 500.0, 400.0]) * units.mbar, 293.0 * units.kelvin
    )
    true_temp = np.array([293, 284.64, 272.81, 264.42, 252.91]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_moist_lapse_degc():
    """Test moist_lapse with Celsius temperatures."""
    temp = moist_lapse(
        np.array([1000.0, 800.0, 600.0, 500.0, 400.0]) * units.mbar, 19.85 * units.degC
    )
    true_temp = np.array([293, 284.64, 272.81, 264.42, 252.91]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_moist_lapse_ref_pres():
    """Test moist_lapse with a reference pressure."""
    temp = moist_lapse(
        np.array([1050.0, 800.0, 600.0, 500.0, 400.0]) * units.mbar,
        19.85 * units.degC,
        1000.0 * units.mbar,
    )
    true_temp = np.array([294.76, 284.64, 272.81, 264.42, 252.91]) * units.kelvin
    assert_array_almost_equal(temp, true_temp, 2)


def test_parcel_profile():
    """Test parcel profile calculation."""
    levels = np.array([1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0]) * units.mbar
    true_prof = (
        np.array([303.15, 294.16, 288.026, 283.073, 277.058, 269.402, 258.966]) * units.kelvin
    )

    prof = parcel_profile(levels, 30.0 * units.degC, 20.0 * units.degC)
    assert_array_almost_equal(prof, true_prof, 2)


def test_parcel_profile_lcl():
    """Test parcel profile with lcl calculation."""
    p = (
        np.array([1004.0, 1000.0, 943.0, 928.0, 925.0, 850.0, 839.0, 749.0, 700.0, 699.0])
        * units.hPa
    )
    t = np.array([24.2, 24.0, 20.2, 21.6, 21.4, 20.4, 20.2, 14.4, 13.2, 13.0]) * units.degC
    td = np.array([21.9, 22.1, 19.2, 20.5, 20.4, 18.4, 17.4, 8.4, -2.8, -3.0]) * units.degC

    true_prof = (
        np.array(
            [
                297.35,
                297.01,
                294.5,
                293.48,
                292.92,
                292.81,
                289.79,
                289.32,
                285.15,
                282.59,
                282.53,
            ]
        )
        * units.kelvin
    )
    true_p = np.insert(p.m, 2, 970.699) * units.mbar
    true_t = np.insert(t.m, 2, 22.047) * units.degC
    true_td = np.insert(td.m, 2, 20.609) * units.degC

    pressure, temp, dewp, prof = parcel_profile_with_lcl(p, t, td)
    assert_almost_equal(pressure, true_p, 3)
    assert_almost_equal(temp, true_t, 3)
    assert_almost_equal(dewp, true_td, 3)
    assert_array_almost_equal(prof, true_prof, 2)


def test_parcel_profile_with_lcl_as_dataset():
    """Test parcel profile with lcl calculation with xarray."""
    p = (
        np.array([1004.0, 1000.0, 943.0, 928.0, 925.0, 850.0, 839.0, 749.0, 700.0, 699.0])
        * units.hPa
    )
    t = np.array([24.2, 24.0, 20.2, 21.6, 21.4, 20.4, 20.2, 14.4, 13.2, 13.0]) * units.degC
    td = np.array([21.9, 22.1, 19.2, 20.5, 20.4, 18.4, 17.4, 8.4, -2.8, -3.0]) * units.degC

    result = parcel_profile_with_lcl_as_dataset(p, t, td)

    expected = xr.Dataset(
        {
            "ambient_temperature": (
                ("isobaric",),
                np.insert(t.m, 2, 22.047) * units.degC,
                {"standard_name": "air_temperature"},
            ),
            "ambient_dew_point": (
                ("isobaric",),
                np.insert(td.m, 2, 20.609) * units.degC,
                {"standard_name": "dew_point_temperature"},
            ),
            "parcel_temperature": (
                ("isobaric",),
                [
                    297.35,
                    297.01,
                    294.5,
                    293.48,
                    292.92,
                    292.81,
                    289.79,
                    289.32,
                    285.15,
                    282.59,
                    282.53,
                ]
                * units.kelvin,
                {"long_name": "air_temperature_of_lifted_parcel"},
            ),
        },
        coords={
            "isobaric": (
                "isobaric",
                np.insert(p.m, 2, 970.699),
                {"units": "hectopascal", "standard_name": "air_pressure"},
            )
        },
    )
    xr.testing.assert_allclose(result, expected, atol=1e-2)
    for field in (
        "ambient_temperature",
        "ambient_dew_point",
        "parcel_temperature",
        "isobaric",
    ):
        assert result[field].attrs == expected[field].attrs


def test_parcel_profile_saturated():
    """Test parcel_profile works when LCL in levels (issue #232)."""
    levels = np.array([1000.0, 700.0, 500.0]) * units.mbar
    true_prof = np.array([296.95, 284.381, 271.123]) * units.kelvin

    prof = parcel_profile(levels, 23.8 * units.degC, 23.8 * units.degC)
    assert_array_almost_equal(prof, true_prof, 2)


def test_sat_vapor_pressure():
    """Test saturation_vapor_pressure calculation."""
    temp = np.array([5.0, 10.0, 18.0, 25.0]) * units.degC
    real_es = np.array([8.72, 12.27, 20.63, 31.67]) * units.mbar
    assert_array_almost_equal(saturation_vapor_pressure(temp), real_es, 2)


def test_sat_vapor_pressure_scalar():
    """Test saturation_vapor_pressure handles scalar values."""
    es = saturation_vapor_pressure(0 * units.degC)
    assert_almost_equal(es, 6.112 * units.mbar, 3)


def test_sat_vapor_pressure_fahrenheit():
    """Test saturation_vapor_pressure handles temperature in Fahrenheit."""
    temp = np.array([50.0, 68.0]) * units.degF
    real_es = np.array([12.2717, 23.3695]) * units.mbar
    assert_array_almost_equal(saturation_vapor_pressure(temp), real_es, 4)


def test_basic_dewpoint_from_relative_humidity():
    """Test dewpoint_from_relative_humidity function."""
    temp = np.array([30.0, 25.0, 10.0, 20.0, 25.0]) * units.degC
    rh = np.array([30.0, 45.0, 55.0, 80.0, 85.0]) / 100.0

    real_td = np.array([11, 12, 1, 16, 22]) * units.degC
    assert_array_almost_equal(real_td, dewpoint_from_relative_humidity(temp, rh), 0)


def test_scalar_dewpoint_from_relative_humidity():
    """Test dewpoint_from_relative_humidity with scalar values."""
    td = dewpoint_from_relative_humidity(10.6 * units.degC, 0.37)
    assert_almost_equal(td, 26.0 * units.degF, 0)


def test_percent_dewpoint_from_relative_humidity():
    """Test dewpoint_from_relative_humidity with rh in percent."""
    td = dewpoint_from_relative_humidity(10.6 * units.degC, 37 * units.percent)
    assert_almost_equal(td, 26.0 * units.degF, 0)


def test_warning_dewpoint_from_relative_humidity():
    """Test that warning is raised for >120% RH."""
    with pytest.warns(UserWarning):
        dewpoint_from_relative_humidity(10.6 * units.degC, 50)


def test_dewpoint():
    """Test dewpoint calculation."""
    assert_almost_equal(dewpoint(6.112 * units.mbar), 0.0 * units.degC, 2)


def test_dewpoint_weird_units():
    """Test dewpoint using non-standard units.

    Revealed from odd dimensionless units and ending up using numpy.ma math
    functions instead of numpy ones.
    """
    assert_almost_equal(dewpoint(15825.6 * units("g * mbar / kg")), 13.8564 * units.degC, 4)


def test_mixing_ratio():
    """Test mixing ratio calculation."""
    p = 998.0 * units.mbar
    e = 73.75 * units.mbar
    assert_almost_equal(mixing_ratio(e, p), 0.04963, 2)


def test_vapor_pressure():
    """Test vapor pressure calculation."""
    assert_almost_equal(vapor_pressure(998.0 * units.mbar, 0.04963), 73.74925 * units.mbar, 5)


def test_lcl():
    """Test LCL calculation."""
    lcl_pressure, lcl_temperature = lcl(
        1000.0 * units.mbar, 30.0 * units.degC, 20.0 * units.degC
    )
    assert_almost_equal(lcl_pressure, 864.761 * units.mbar, 2)
    assert_almost_equal(lcl_temperature, 17.676 * units.degC, 2)


def test_lcl_kelvin():
    """Test LCL temperature is returned as Kelvin, if temperature is Kelvin."""
    temperature = 273.09723 * units.kelvin
    lcl_pressure, lcl_temperature = lcl(
        1017.16 * units.mbar, temperature, 264.5351 * units.kelvin
    )
    assert_almost_equal(lcl_pressure, 889.416 * units.mbar, 2)
    assert_almost_equal(lcl_temperature, 262.827 * units.kelvin, 2)
    assert lcl_temperature.units == temperature.units


def test_lcl_convergence():
    """Test LCL calculation convergence failure."""
    with pytest.raises(RuntimeError):
        lcl(1000.0 * units.mbar, 30.0 * units.degC, 20.0 * units.degC, max_iters=2)


def test_lfc_basic():
    """Test LFC calculation."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -49.0]) * units.celsius
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * units.celsius
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(lfc_pressure, 727.415 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 9.705 * units.celsius, 2)


def test_lfc_kelvin():
    """Test that LFC temperature returns Kelvin if Kelvin is provided."""
    pressure = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.mbar
    temperature = (np.array([22.2, 14.6, 12.0, 9.4, 7.0, -49.0]) + 273.15) * units.kelvin
    dewpoint = (np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) + 273.15) * units.kelvin
    lfc_pressure, lfc_temp = lfc(pressure, temperature, dewpoint)
    assert_almost_equal(lfc_pressure, 727.415 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 9.705 * units.degC, 2)
    assert lfc_temp.units == temperature.units


def test_lfc_ml():
    """Test Mixed-Layer LFC calculation."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -49.0]) * units.celsius
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * units.celsius
    __, t_mixed, td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = parcel_profile(levels, t_mixed, td_mixed)
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints, mixed_parcel_prof)
    assert_almost_equal(lfc_pressure, 601.685 * units.mbar, 2)
    assert_almost_equal(lfc_temp, -1.862 * units.degC, 2)


def test_lfc_ml2():
    """Test a mixed-layer LFC calculation that previously crashed."""
    levels = (
        np.array(
            [
                1024.95703125,
                1016.61474609,
                1005.33056641,
                991.08544922,
                973.4163208,
                951.3381958,
                924.82836914,
                898.25482178,
                873.46124268,
                848.69830322,
                823.92553711,
                788.49304199,
                743.44580078,
                700.50970459,
                659.62017822,
                620.70861816,
                583.69421387,
                548.49719238,
                515.03826904,
                483.24401855,
                453.0418396,
                424.36477661,
                397.1505127,
                371.33441162,
                346.85922241,
                323.66995239,
                301.70935059,
                280.92651367,
                261.27053833,
                242.69168091,
                225.14237976,
                208.57781982,
                192.95333862,
                178.22599792,
                164.39630127,
                151.54336548,
                139.68635559,
                128.74923706,
                118.6588974,
                109.35111237,
                100.76405334,
                92.84288025,
                85.53556824,
                78.79430389,
                72.57549286,
                66.83885193,
                61.54678726,
                56.66480637,
                52.16108322,
            ]
        )
        * units.mbar
    )
    temperatures = (
        np.array(
            [
                6.00750732,
                5.14892578,
                4.177948,
                3.00268555,
                1.55535889,
                -0.25527954,
                -1.93988037,
                -3.57766724,
                -4.40600586,
                -4.19238281,
                -3.71185303,
                -4.47943115,
                -6.81280518,
                -8.08685303,
                -8.41287231,
                -10.79302979,
                -14.13262939,
                -16.85784912,
                -19.51675415,
                -22.28689575,
                -24.99938965,
                -27.79664612,
                -30.90414429,
                -34.49435425,
                -38.438797,
                -42.27981567,
                -45.99230957,
                -49.75340271,
                -53.58230591,
                -57.30686951,
                -60.76026917,
                -63.92070007,
                -66.72470093,
                -68.97846985,
                -70.4264679,
                -71.16407776,
                -71.53797913,
                -71.64375305,
                -71.52735901,
                -71.53523254,
                -71.61097717,
                -71.92687988,
                -72.68682861,
                -74.129776,
                -76.02471924,
                -76.88977051,
                -76.26008606,
                -75.90351868,
                -76.15809631,
            ]
        )
        * units.celsius
    )
    dewpoints = (
        np.array(
            [
                4.50012302,
                3.42483997,
                2.78102994,
                2.24474645,
                1.593485,
                -0.9440815,
                -3.8044982,
                -3.55629468,
                -9.7376976,
                -10.2950449,
                -9.67498302,
                -10.30486488,
                -8.70559597,
                -8.71669006,
                -12.66509628,
                -18.6697197,
                -23.00351334,
                -29.46240425,
                -36.82178497,
                -41.68824768,
                -44.50320816,
                -48.54426575,
                -52.50753403,
                -51.09564209,
                -48.92690659,
                -49.97380829,
                -51.57516098,
                -52.62096405,
                -54.24332809,
                -57.09109879,
                -60.5596199,
                -63.93486404,
                -67.07530212,
                -70.01263428,
                -72.9258728,
                -76.12271881,
                -79.49847412,
                -82.2350769,
                -83.91127014,
                -84.95665741,
                -85.61238861,
                -86.16391754,
                -86.7653656,
                -87.34436035,
                -87.87495422,
                -88.34281921,
                -88.74453735,
                -89.04680634,
                -89.26436615,
            ]
        )
        * units.celsius
    )
    __, t_mixed, td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = parcel_profile(levels, t_mixed, td_mixed)
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints, mixed_parcel_prof, td_mixed)
    assert_almost_equal(lfc_pressure, 962.34 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 0.767 * units.degC, 2)


def test_lfc_intersection():
    """Test LFC calculation when LFC is below a tricky intersection."""
    p = np.array(
        [1024.957, 930.0, 924.828, 898.255, 873.461, 848.698, 823.926, 788.493]
    ) * units("hPa")
    t = np.array([6.008, -10.0, -6.94, -8.58, -4.41, -4.19, -3.71, -4.48]) * units("degC")
    td = np.array([5.0, -10.0, -7.0, -9.0, -4.5, -4.2, -3.8, -4.5]) * units("degC")
    _, mlt, mltd = mixed_parcel(p, t, td)
    ml_profile = parcel_profile(p, mlt, mltd)
    mllfc_p, mllfc_t = lfc(p, t, td, ml_profile, mltd)
    assert_almost_equal(mllfc_p, 981.620 * units.hPa, 2)
    assert_almost_equal(mllfc_t, 272.045 * units.kelvin, 2)


def test_no_lfc():
    """Test LFC calculation when there is no LFC in the data."""
    levels = np.array([959.0, 867.9, 779.2, 647.5, 472.5, 321.9, 251.0]) * units.mbar
    temperatures = np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5]) * units.celsius
    dewpoints = np.array([9.0, 4.3, -21.2, -26.7, -31.0, -53.3, -66.7]) * units.celsius
    lfc_pressure, lfc_temperature = lfc(levels, temperatures, dewpoints)
    assert_nan(lfc_pressure, levels.units)
    assert_nan(lfc_temperature, temperatures.units)


def test_lfc_inversion():
    """Test LFC when there is an inversion to be sure we don't pick that."""
    levels = (
        np.array([963.0, 789.0, 782.3, 754.8, 728.1, 727.0, 700.0, 571.0, 450.0, 300.0, 248.0])
        * units.mbar
    )
    temperatures = (
        np.array([25.4, 18.4, 17.8, 15.4, 12.9, 12.8, 10.0, -3.9, -16.3, -41.1, -51.5])
        * units.celsius
    )
    dewpoints = (
        np.array([20.4, 0.4, -0.5, -4.3, -8.0, -8.2, -9.0, -23.9, -33.3, -54.1, -63.5])
        * units.celsius
    )
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(lfc_pressure, 705.9214 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 10.6232 * units.celsius, 2)


def test_lfc_equals_lcl():
    """Test LFC when there is no cap and the lfc is equal to the lcl."""
    levels = (
        np.array([912.0, 905.3, 874.4, 850.0, 815.1, 786.6, 759.1, 748.0, 732.2, 700.0, 654.8])
        * units.mbar
    )
    temperatures = (
        np.array([29.4, 28.7, 25.2, 22.4, 19.4, 16.8, 14.0, 13.2, 12.6, 11.4, 7.1])
        * units.celsius
    )
    dewpoints = (
        np.array([18.4, 18.1, 16.6, 15.4, 13.2, 11.4, 9.6, 8.8, 0.0, -18.6, -22.9])
        * units.celsius
    )
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_almost_equal(lfc_pressure, 777.0333 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 15.8714 * units.celsius, 2)


def test_sensitive_sounding():
    """Test quantities for a sensitive sounding (#902)."""
    # This sounding has a very small positive area in the low level. It's only captured
    # properly if the parcel profile includes the LCL, otherwise it breaks LFC and CAPE
    p = units.Quantity(
        [
            1004.0,
            1000.0,
            943.0,
            928.0,
            925.0,
            850.0,
            839.0,
            749.0,
            700.0,
            699.0,
            603.0,
            500.0,
            404.0,
            400.0,
            363.0,
            306.0,
            300.0,
            250.0,
            213.0,
            200.0,
            176.0,
            150.0,
        ],
        "hectopascal",
    )
    t = units.Quantity(
        [
            24.2,
            24.0,
            20.2,
            21.6,
            21.4,
            20.4,
            20.2,
            14.4,
            13.2,
            13.0,
            6.8,
            -3.3,
            -13.1,
            -13.7,
            -17.9,
            -25.5,
            -26.9,
            -37.9,
            -46.7,
            -48.7,
            -52.1,
            -58.9,
        ],
        "degC",
    )
    td = units.Quantity(
        [
            21.9,
            22.1,
            19.2,
            20.5,
            20.4,
            18.4,
            17.4,
            8.4,
            -2.8,
            -3.0,
            -15.2,
            -20.3,
            -29.1,
            -27.7,
            -24.9,
            -39.5,
            -41.9,
            -51.9,
            -60.7,
            -62.7,
            -65.1,
            -71.9,
        ],
        "degC",
    )
    lfc_pressure, lfc_temp = lfc(p, t, td)
    assert_almost_equal(lfc_pressure, 947.422 * units.mbar, 2)
    assert_almost_equal(lfc_temp, 20.498 * units.degC, 2)

    pos, neg = surface_based_cape_cin(p, t, td)
    assert_almost_equal(pos, 0.1115 * units("J/kg"), 3)
    assert_almost_equal(neg, -6.0806 * units("J/kg"), 3)


def test_lfc_sfc_precision():
    """Test LFC when there are precision issues with the parcel path."""
    levels = (
        np.array([839.0, 819.4, 816.0, 807.0, 790.7, 763.0, 736.2, 722.0, 710.1, 700.0])
        * units.mbar
    )
    temperatures = (
        np.array([20.6, 22.3, 22.6, 22.2, 20.9, 18.7, 16.4, 15.2, 13.9, 12.8]) * units.celsius
    )
    dewpoints = np.array([10.6, 8.0, 7.6, 6.2, 5.7, 4.7, 3.7, 3.2, 3.0, 2.8]) * units.celsius
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    assert_nan(lfc_pressure, levels.units)
    assert_nan(lfc_temp, temperatures.units)


def test_lfc_pos_area_below_lcl():
    """Test LFC when there is positive area below the LCL (#1003)."""
    p = [
        902.1554,
        897.9034,
        893.6506,
        889.4047,
        883.063,
        874.6284,
        866.2387,
        857.887,
        849.5506,
        841.2686,
        833.0042,
        824.7891,
        812.5049,
        796.2104,
        776.0027,
        751.9025,
        727.9612,
        704.1409,
        680.4028,
        656.7156,
        629.077,
        597.4286,
        565.6315,
        533.5961,
        501.2452,
        468.493,
        435.2486,
        401.4239,
        366.9387,
        331.7026,
        295.6319,
        258.6428,
        220.9178,
        182.9384,
        144.959,
        106.9778,
        69.00213,
    ] * units.hPa
    t = [
        -3.039381,
        -3.703779,
        -4.15996,
        -4.562574,
        -5.131827,
        -5.856229,
        -6.568434,
        -7.276881,
        -7.985013,
        -8.670911,
        -8.958063,
        -7.631381,
        -6.05927,
        -5.083627,
        -5.11576,
        -5.687552,
        -5.453021,
        -4.981445,
        -5.236665,
        -6.324916,
        -8.434324,
        -11.58795,
        -14.99297,
        -18.45947,
        -21.92021,
        -25.40522,
        -28.914,
        -32.78637,
        -37.7179,
        -43.56836,
        -49.61077,
        -54.24449,
        -56.16666,
        -57.03775,
        -58.28041,
        -60.86264,
        -64.21677,
    ] * units.degC
    td = [
        -22.08774,
        -22.18181,
        -22.2508,
        -22.31323,
        -22.4024,
        -22.51582,
        -22.62526,
        -22.72919,
        -22.82095,
        -22.86173,
        -22.49489,
        -21.66936,
        -21.67332,
        -21.94054,
        -23.63561,
        -27.17466,
        -31.87395,
        -38.31725,
        -44.54717,
        -46.99218,
        -43.17544,
        -37.40019,
        -34.3351,
        -36.42896,
        -42.1396,
        -46.95909,
        -49.36232,
        -48.94634,
        -47.90178,
        -49.97902,
        -55.02753,
        -63.06276,
        -72.53742,
        -88.81377,
        -93.54573,
        -92.92464,
        -91.57479,
    ] * units.degC
    prof = parcel_profile(p, t[0], td[0]).to("degC")
    lfc_p, lfc_t = lfc(p, t, td, prof)
    assert_nan(lfc_p, p.units)
    assert_nan(lfc_t, t.units)


def test_saturation_mixing_ratio():
    """Test saturation mixing ratio calculation."""
    p = 999.0 * units.mbar
    t = 288.0 * units.kelvin
    assert_almost_equal(saturation_mixing_ratio(p, t), 0.01068, 3)


def test_saturation_mixing_ratio_with_xarray():
    """Test saturation mixing ratio calculation with xarray."""
    temperature = xr.DataArray(
        np.arange(10, 18).reshape((2, 2, 2)) * units.degC,
        dims=("isobaric", "y", "x"),
        coords={
            "isobaric": (("isobaric",), [700.0, 850.0], {"units": "hPa"}),
            "y": (("y",), [0.0, 100.0], {"units": "kilometer"}),
            "x": (("x",), [0.0, 100.0], {"units": "kilometer"}),
        },
    )
    result = saturation_mixing_ratio(temperature.metpy.vertical, temperature)
    expected_values = [
        [[0.011098, 0.011879], [0.012708, 0.013589]],
        [[0.011913, 0.012724], [0.013586, 0.014499]],
    ]
    assert_array_almost_equal(result.data, expected_values, 5)
    xr.testing.assert_identical(result["isobaric"], temperature["isobaric"])
    xr.testing.assert_identical(result["y"], temperature["y"])
    xr.testing.assert_identical(result["x"], temperature["x"])


def test_equivalent_potential_temperature():
    """Test equivalent potential temperature calculation."""
    p = 1000 * units.mbar
    t = 293.0 * units.kelvin
    td = 280.0 * units.kelvin
    ept = equivalent_potential_temperature(p, t, td)
    assert_almost_equal(ept, 311.18586467284007 * units.kelvin, 3)


def test_equivalent_potential_temperature_masked():
    """Test equivalent potential temperature calculation with masked arrays."""
    p = 1000 * units.mbar
    t = units.Quantity(np.ma.array([293.0, 294.0, 295.0]), units.kelvin)
    td = units.Quantity(
        np.ma.array([280.0, 281.0, 282.0], mask=[False, True, False]), units.kelvin
    )
    ept = equivalent_potential_temperature(p, t, td)
    expected = units.Quantity(
        np.ma.array([311.18586, 313.51781, 315.93971], mask=[False, True, False]), units.kelvin
    )
    assert isinstance(ept, units.Quantity)
    assert isinstance(ept.m, np.ma.MaskedArray)
    assert_array_almost_equal(ept, expected, 3)


def test_saturation_equivalent_potential_temperature():
    """Test saturation equivalent potential temperature calculation."""
    p = 700 * units.mbar
    t = 263.15 * units.kelvin
    s_ept = saturation_equivalent_potential_temperature(p, t)
    # 299.096584 comes from equivalent_potential_temperature(p,t,t)
    # where dewpoint and temperature are equal, which means saturations.
    assert_almost_equal(s_ept, 299.096584 * units.kelvin, 3)


def test_saturation_equivalent_potential_temperature_masked():
    """Test saturation equivalent potential temperature calculation with masked arrays."""
    p = 1000 * units.mbar
    t = units.Quantity(np.ma.array([293.0, 294.0, 295.0]), units.kelvin)
    s_ept = saturation_equivalent_potential_temperature(p, t)
    expected = units.Quantity(np.ma.array([335.02750, 338.95813, 343.08740]), units.kelvin)
    assert isinstance(s_ept, units.Quantity)
    assert isinstance(s_ept.m, np.ma.MaskedArray)
    assert_array_almost_equal(s_ept, expected, 3)


def test_virtual_temperature():
    """Test virtual temperature calculation."""
    t = 288.0 * units.kelvin
    qv = 0.0016 * units.dimensionless  # kg/kg
    tv = virtual_temperature(t, qv)
    assert_almost_equal(tv, 288.2796 * units.kelvin, 3)


def test_virtual_potential_temperature():
    """Test virtual potential temperature calculation."""
    p = 999.0 * units.mbar
    t = 288.0 * units.kelvin
    qv = 0.0016 * units.dimensionless  # kg/kg
    theta_v = virtual_potential_temperature(p, t, qv)
    assert_almost_equal(theta_v, 288.3620 * units.kelvin, 3)


def test_density():
    """Test density calculation."""
    p = 999.0 * units.mbar
    t = 288.0 * units.kelvin
    qv = 0.0016 * units.dimensionless  # kg/kg
    rho = density(p, t, qv).to(units.kilogram / units.meter ** 3)
    assert_almost_equal(rho, 1.2072 * (units.kilogram / units.meter ** 3), 3)


def test_el():
    """Test equilibrium layer calculation."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * units.celsius
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * units.celsius
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_almost_equal(el_pressure, 470.4075 * units.mbar, 3)
    assert_almost_equal(el_temperature, -11.7027 * units.degC, 3)


def test_el_kelvin():
    """Test that EL temperature returns Kelvin if Kelvin is provided."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.mbar
    temperatures = (np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) + 273.15) * units.kelvin
    dewpoints = (np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) + 273.15) * units.kelvin
    el_pressure, el_temp = el(levels, temperatures, dewpoints)
    assert_almost_equal(el_pressure, 470.4075 * units.mbar, 3)
    assert_almost_equal(el_temp, -11.7027 * units.degC, 3)
    assert el_temp.units == temperatures.units


def test_el_ml():
    """Test equilibrium layer calculation for a mixed parcel."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 400.0, 269.0]) * units.mbar
    temperatures = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -25.0, -35.0]) * units.celsius
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -35.0, -53.2]) * units.celsius
    __, t_mixed, td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = parcel_profile(levels, t_mixed, td_mixed)
    el_pressure, el_temperature = el(levels, temperatures, dewpoints, mixed_parcel_prof)
    assert_almost_equal(el_pressure, 349.919 * units.mbar, 3)
    assert_almost_equal(el_temperature, -28.371 * units.degC, 3)


def test_no_el():
    """Test equilibrium layer calculation when there is no EL in the data."""
    levels = np.array([959.0, 867.9, 779.2, 647.5, 472.5, 321.9, 251.0]) * units.mbar
    temperatures = np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5]) * units.celsius
    dewpoints = np.array([19.0, 14.3, -11.2, -16.7, -21.0, -43.3, -56.7]) * units.celsius
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


def test_no_el_multi_crossing():
    """Test el calculation with no el and severel parcel path-profile crossings."""
    levels = (
        np.array(
            [
                918.0,
                911.0,
                880.0,
                873.9,
                850.0,
                848.0,
                843.5,
                818.0,
                813.8,
                785.0,
                773.0,
                763.0,
                757.5,
                730.5,
                700.0,
                679.0,
                654.4,
                645.0,
                643.9,
            ]
        )
        * units.mbar
    )
    temperatures = (
        np.array(
            [
                24.2,
                22.8,
                19.6,
                19.1,
                17.0,
                16.8,
                16.5,
                15.0,
                14.9,
                14.4,
                16.4,
                16.2,
                15.7,
                13.4,
                10.6,
                8.4,
                5.7,
                4.6,
                4.5,
            ]
        )
        * units.celsius
    )
    dewpoints = (
        np.array(
            [
                19.5,
                17.8,
                16.7,
                16.5,
                15.8,
                15.7,
                15.3,
                13.1,
                12.9,
                11.9,
                6.4,
                3.2,
                2.6,
                -0.6,
                -4.4,
                -6.6,
                -9.3,
                -10.4,
                -10.5,
            ]
        )
        * units.celsius
    )
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


def test_lfc_and_el_below_lcl():
    """Test that LFC and EL are returned as NaN if both are below LCL."""
    dewpoint = [264.5351, 261.13443, 259.0122, 252.30063, 248.58017, 242.66582] * units.kelvin
    temperature = [
        273.09723,
        268.40173,
        263.56207,
        260.257,
        256.63538,
        252.91345,
    ] * units.kelvin
    pressure = [1017.16, 950, 900, 850, 800, 750] * units.hPa
    el_pressure, el_temperature = el(pressure, temperature, dewpoint)
    lfc_pressure, lfc_temperature = lfc(pressure, temperature, dewpoint)
    assert_nan(lfc_pressure, pressure.units)
    assert_nan(lfc_temperature, temperature.units)
    assert_nan(el_pressure, pressure.units)
    assert_nan(el_temperature, temperature.units)


def test_el_lfc_equals_lcl():
    """Test equilibrium layer calculation when the lfc equals the lcl."""
    levels = (
        np.array(
            [
                912.0,
                905.3,
                874.4,
                850.0,
                815.1,
                786.6,
                759.1,
                748.0,
                732.3,
                700.0,
                654.8,
                606.8,
                562.4,
                501.8,
                500.0,
                482.0,
                400.0,
                393.3,
                317.1,
                307.0,
                300.0,
                252.7,
                250.0,
                200.0,
                199.3,
                197.0,
                190.0,
                172.0,
                156.6,
                150.0,
                122.9,
                112.0,
                106.2,
                100.0,
            ]
        )
        * units.mbar
    )
    temperatures = (
        np.array(
            [
                29.4,
                28.7,
                25.2,
                22.4,
                19.4,
                16.8,
                14.3,
                13.2,
                12.6,
                11.4,
                7.1,
                2.2,
                -2.7,
                -10.1,
                -10.3,
                -12.4,
                -23.3,
                -24.4,
                -38.0,
                -40.1,
                -41.1,
                -49.8,
                -50.3,
                -59.1,
                -59.1,
                -59.3,
                -59.7,
                -56.3,
                -56.9,
                -57.1,
                -59.1,
                -60.1,
                -58.6,
                -56.9,
            ]
        )
        * units.celsius
    )
    dewpoints = (
        np.array(
            [
                18.4,
                18.1,
                16.6,
                15.4,
                13.2,
                11.4,
                9.6,
                8.8,
                0.0,
                -18.6,
                -22.9,
                -27.8,
                -32.7,
                -40.1,
                -40.3,
                -42.4,
                -53.3,
                -54.4,
                -68.0,
                -70.1,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
                -70.0,
            ]
        )
        * units.celsius
    )
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_almost_equal(el_pressure, 175.7187 * units.mbar, 3)
    assert_almost_equal(el_temperature, -57.0307 * units.degC, 3)


def test_el_small_surface_instability():
    """Test that no EL is found when there is a small pocket of instability at the sfc."""
    levels = (
        np.array(
            [
                959.0,
                931.3,
                925.0,
                899.3,
                892.0,
                867.9,
                850.0,
                814.0,
                807.9,
                790.0,
                779.2,
                751.3,
                724.3,
                700.0,
                655.0,
                647.5,
                599.4,
                554.7,
                550.0,
                500.0,
            ]
        )
        * units.mbar
    )
    temperatures = (
        np.array(
            [
                22.2,
                20.2,
                19.8,
                18.4,
                18.0,
                17.4,
                17.0,
                15.4,
                15.4,
                15.6,
                14.6,
                12.0,
                9.4,
                7.0,
                2.2,
                1.4,
                -4.2,
                -9.7,
                -10.3,
                -14.9,
            ]
        )
        * units.degC
    )
    dewpoints = (
        np.array(
            [
                20.0,
                18.5,
                18.1,
                17.9,
                17.8,
                15.3,
                13.5,
                6.4,
                2.2,
                -10.4,
                -10.2,
                -9.8,
                -9.4,
                -9.0,
                -15.8,
                -15.7,
                -14.8,
                -14.0,
                -13.9,
                -17.9,
            ]
        )
        * units.degC
    )
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


def test_no_el_parcel_colder():
    """Test no EL when parcel stays colder than environment. INL 20170925-12Z."""
    levels = (
        np.array(
            [
                974.0,
                946.0,
                925.0,
                877.2,
                866.0,
                850.0,
                814.6,
                785.0,
                756.6,
                739.0,
                729.1,
                700.0,
                686.0,
                671.0,
                641.0,
                613.0,
                603.0,
                586.0,
                571.0,
                559.3,
                539.0,
                533.0,
                500.0,
                491.0,
                477.9,
                413.0,
                390.0,
                378.0,
                345.0,
                336.0,
            ]
        )
        * units.mbar
    )
    temperatures = (
        np.array(
            [
                10.0,
                8.4,
                7.6,
                5.9,
                7.2,
                7.6,
                6.8,
                7.1,
                7.7,
                7.8,
                7.7,
                5.6,
                4.6,
                3.4,
                0.6,
                -0.9,
                -1.1,
                -3.1,
                -4.7,
                -4.7,
                -6.9,
                -7.5,
                -11.1,
                -10.9,
                -12.1,
                -20.5,
                -23.5,
                -24.7,
                -30.5,
                -31.7,
            ]
        )
        * units.celsius
    )
    dewpoints = (
        np.array(
            [
                8.9,
                8.4,
                7.6,
                5.9,
                7.2,
                7.0,
                5.0,
                3.6,
                0.3,
                -4.2,
                -12.8,
                -12.4,
                -8.4,
                -8.6,
                -6.4,
                -7.9,
                -11.1,
                -14.1,
                -8.8,
                -28.1,
                -18.9,
                -14.5,
                -15.2,
                -15.1,
                -21.6,
                -41.5,
                -45.5,
                -29.6,
                -30.6,
                -32.1,
            ]
        )
        * units.celsius
    )
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


def test_el_below_lcl():
    """Test LFC when there is positive area below the LCL (#1003)."""
    p = [
        902.1554,
        897.9034,
        893.6506,
        889.4047,
        883.063,
        874.6284,
        866.2387,
        857.887,
        849.5506,
        841.2686,
        833.0042,
        824.7891,
        812.5049,
        796.2104,
        776.0027,
        751.9025,
        727.9612,
        704.1409,
        680.4028,
        656.7156,
        629.077,
        597.4286,
        565.6315,
        533.5961,
        501.2452,
        468.493,
        435.2486,
        401.4239,
        366.9387,
        331.7026,
        295.6319,
        258.6428,
        220.9178,
        182.9384,
        144.959,
        106.9778,
        69.00213,
    ] * units.hPa
    t = [
        -3.039381,
        -3.703779,
        -4.15996,
        -4.562574,
        -5.131827,
        -5.856229,
        -6.568434,
        -7.276881,
        -7.985013,
        -8.670911,
        -8.958063,
        -7.631381,
        -6.05927,
        -5.083627,
        -5.11576,
        -5.687552,
        -5.453021,
        -4.981445,
        -5.236665,
        -6.324916,
        -8.434324,
        -11.58795,
        -14.99297,
        -18.45947,
        -21.92021,
        -25.40522,
        -28.914,
        -32.78637,
        -37.7179,
        -43.56836,
        -49.61077,
        -54.24449,
        -56.16666,
        -57.03775,
        -58.28041,
        -60.86264,
        -64.21677,
    ] * units.degC
    td = [
        -22.08774,
        -22.18181,
        -22.2508,
        -22.31323,
        -22.4024,
        -22.51582,
        -22.62526,
        -22.72919,
        -22.82095,
        -22.86173,
        -22.49489,
        -21.66936,
        -21.67332,
        -21.94054,
        -23.63561,
        -27.17466,
        -31.87395,
        -38.31725,
        -44.54717,
        -46.99218,
        -43.17544,
        -37.40019,
        -34.3351,
        -36.42896,
        -42.1396,
        -46.95909,
        -49.36232,
        -48.94634,
        -47.90178,
        -49.97902,
        -55.02753,
        -63.06276,
        -72.53742,
        -88.81377,
        -93.54573,
        -92.92464,
        -91.57479,
    ] * units.degC
    prof = parcel_profile(p, t[0], td[0]).to("degC")
    el_p, el_t = el(p, t, td, prof)
    assert_nan(el_p, p.units)
    assert_nan(el_t, t.units)


def test_wet_psychrometric_vapor_pressure():
    """Test calculation of vapor pressure from wet and dry bulb temperatures."""
    p = 1013.25 * units.mbar
    dry_bulb_temperature = 20.0 * units.degC
    wet_bulb_temperature = 18.0 * units.degC
    psychrometric_vapor_pressure = psychrometric_vapor_pressure_wet(
        p, dry_bulb_temperature, wet_bulb_temperature
    )
    assert_almost_equal(psychrometric_vapor_pressure, 19.3673 * units.mbar, 3)


def test_wet_psychrometric_rh():
    """Test calculation of relative humidity from wet and dry bulb temperatures."""
    p = 1013.25 * units.mbar
    dry_bulb_temperature = 20.0 * units.degC
    wet_bulb_temperature = 18.0 * units.degC
    psychrometric_rh = relative_humidity_wet_psychrometric(
        p, dry_bulb_temperature, wet_bulb_temperature
    )
    assert_almost_equal(psychrometric_rh, 82.8747 * units.percent, 3)


def test_wet_psychrometric_rh_kwargs():
    """Test calculation of relative humidity from wet and dry bulb temperatures."""
    p = 1013.25 * units.mbar
    dry_bulb_temperature = 20.0 * units.degC
    wet_bulb_temperature = 18.0 * units.degC
    coeff = 6.1e-4 / units.kelvin
    psychrometric_rh = relative_humidity_wet_psychrometric(
        p, dry_bulb_temperature, wet_bulb_temperature, psychrometer_coefficient=coeff
    )
    assert_almost_equal(psychrometric_rh, 82.9701 * units.percent, 3)


def test_mixing_ratio_from_relative_humidity():
    """Test relative humidity from mixing ratio."""
    p = 1013.25 * units.mbar
    temperature = 20.0 * units.degC
    rh = 81.7219 * units.percent
    w = mixing_ratio_from_relative_humidity(p, temperature, rh)
    assert_almost_equal(w, 0.012 * units.dimensionless, 3)


def test_rh_mixing_ratio():
    """Test relative humidity from mixing ratio."""
    p = 1013.25 * units.mbar
    temperature = 20.0 * units.degC
    w = 0.012 * units.dimensionless
    rh = relative_humidity_from_mixing_ratio(p, temperature, w)
    assert_almost_equal(rh, 81.7219 * units.percent, 3)


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
    temperature = 20.0 * units.degC
    q = 0.012 * units.dimensionless
    rh = relative_humidity_from_specific_humidity(p, temperature, q)
    assert_almost_equal(rh, 82.7145 * units.percent, 3)


def test_cape_cin():
    """Test the basic CAPE and CIN calculation."""
    p = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.mbar
    temperature = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * units.celsius
    dewpoint = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0])
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 75.7340825 * units("joule / kilogram"), 2)
    assert_almost_equal(cin, -89.8179205 * units("joule / kilogram"), 2)


def test_cape_cin_no_el():
    """Test that CAPE works with no EL."""
    p = np.array([959.0, 779.2, 751.3, 724.3]) * units.mbar
    temperature = np.array([22.2, 14.6, 12.0, 9.4]) * units.celsius
    dewpoint = np.array([19.0, -11.2, -10.8, -10.4]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]).to("degC")
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 0.08610409 * units("joule / kilogram"), 2)
    assert_almost_equal(cin, -89.8179205 * units("joule / kilogram"), 2)


def test_cape_cin_no_lfc():
    """Test that CAPE is zero with no LFC."""
    p = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.mbar
    temperature = np.array([22.2, 24.6, 22.0, 20.4, 18.0, -10.0]) * units.celsius
    dewpoint = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]).to("degC")
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 0.0 * units("joule / kilogram"), 2)
    assert_almost_equal(cin, 0.0 * units("joule / kilogram"), 2)


def test_find_append_zero_crossings():
    """Test finding and appending zero crossings of an x, y series."""
    x = np.arange(11) * units.hPa
    y = np.array([3, 2, 1, -1, 2, 2, 0, 1, 0, -1, 2]) * units.degC
    x2, y2 = _find_append_zero_crossings(x, y)

    x_truth = (
        np.array(
            [
                0.0,
                1.0,
                2.0,
                2.4494897,
                3.0,
                3.3019272,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                9.3216975,
                10.0,
            ]
        )
        * units.hPa
    )
    y_truth = np.array([3, 2, 1, 0, -1, 0, 2, 2, 0, 1, 0, -1, 0, 2]) * units.degC
    assert_array_almost_equal(x2, x_truth, 6)
    assert_almost_equal(y2, y_truth, 6)


def test_most_unstable_parcel():
    """Test calculating the most unstable parcel."""
    levels = np.array([1000.0, 959.0, 867.9]) * units.mbar
    temperatures = np.array([18.2, 22.2, 17.4]) * units.celsius
    dewpoints = np.array([19.0, 19.0, 14.3]) * units.celsius
    ret = most_unstable_parcel(levels, temperatures, dewpoints, depth=100 * units.hPa)
    assert_almost_equal(ret[0], 959.0 * units.hPa, 6)
    assert_almost_equal(ret[1], 22.2 * units.degC, 6)
    assert_almost_equal(ret[2], 19.0 * units.degC, 6)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_isentropic_pressure():
    """Test calculation of isentropic pressure function."""
    lev = [100000.0, 95000.0, 90000.0, 85000.0] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.0
    tmp[1, :] = 292.0
    tmp[2, :] = 290
    tmp[3, :] = 288.0
    tmp[:, :, -1] = np.nan
    tmpk = tmp * units.kelvin
    isentlev = [296.0] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk)
    trueprs = np.ones((1, 5, 5)) * (1000.0 * units.hPa)
    trueprs[:, :, -1] = np.nan
    assert isentprs[0].shape == (1, 5, 5)
    assert_almost_equal(isentprs[0], trueprs, 3)


def test_isentropic_pressure_masked_column():
    """Test calculation of isentropic pressure function with a masked column (#769)."""
    lev = [100000.0, 95000.0] * units.Pa
    tmp = np.ma.ones((len(lev), 5, 5))
    tmp[0, :] = 296.0
    tmp[1, :] = 292.0
    tmp[:, :, -1] = np.ma.masked
    tmp = units.Quantity(tmp, units.kelvin)
    isentprs = isentropic_interpolation([296.0] * units.kelvin, lev, tmp)
    trueprs = np.ones((1, 5, 5)) * (1000.0 * units.hPa)
    trueprs[:, :, -1] = np.nan
    assert isentprs[0].shape == (1, 5, 5)
    assert_almost_equal(isentprs[0], trueprs, 3)


def test_isentropic_pressure_p_increase():
    """Test calculation of isentropic pressure function, p increasing order."""
    lev = [85000, 90000.0, 95000.0, 100000.0] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 288.0
    tmp[1, :] = 290.0
    tmp[2, :] = 292.0
    tmp[3, :] = 296.0
    tmpk = tmp * units.kelvin
    isentlev = [296.0] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk)
    trueprs = 1000.0 * units.hPa
    assert_almost_equal(isentprs[0], trueprs, 3)


def test_isentropic_pressure_additional_args():
    """Test calculation of isentropic pressure function, additional args."""
    lev = [100000.0, 95000.0, 90000.0, 85000.0] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.0
    tmp[1, :] = 292.0
    tmp[2, :] = 290.0
    tmp[3, :] = 288.0
    rh = np.ones((4, 5, 5))
    rh[0, :] = 100.0
    rh[1, :] = 80.0
    rh[2, :] = 40.0
    rh[3, :] = 20.0
    relh = rh * units.percent
    tmpk = tmp * units.kelvin
    isentlev = [296.0] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, relh)
    truerh = 100.0 * units.percent
    assert_almost_equal(isentprs[1], truerh, 3)


def test_isentropic_pressure_tmp_out():
    """Test calculation of isentropic pressure function, temperature output."""
    lev = [100000.0, 95000.0, 90000.0, 85000.0] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.0
    tmp[1, :] = 292.0
    tmp[2, :] = 290.0
    tmp[3, :] = 288.0
    tmpk = tmp * units.kelvin
    isentlev = [296.0] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, temperature_out=True)
    truetmp = 296.0 * units.kelvin
    assert_almost_equal(isentprs[1], truetmp, 3)


def test_isentropic_pressure_p_increase_rh_out():
    """Test calculation of isentropic pressure function, p increasing order."""
    lev = [85000.0, 90000.0, 95000.0, 100000.0] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 288.0
    tmp[1, :] = 290.0
    tmp[2, :] = 292.0
    tmp[3, :] = 296.0
    tmpk = tmp * units.kelvin
    rh = np.ones((4, 5, 5))
    rh[0, :] = 20.0
    rh[1, :] = 40.0
    rh[2, :] = 80.0
    rh[3, :] = 100.0
    relh = rh * units.percent
    isentlev = 296.0 * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, relh)
    truerh = 100.0 * units.percent
    assert_almost_equal(isentprs[1], truerh, 3)


def test_isentropic_pressure_interp():
    """Test calculation of isentropic pressure function."""
    lev = [100000.0, 95000.0, 90000.0, 85000.0] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.0
    tmp[1, :] = 292.0
    tmp[2, :] = 290
    tmp[3, :] = 288.0
    tmpk = tmp * units.kelvin
    isentlev = [296.0, 297] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk)
    trueprs = 936.18057 * units.hPa
    assert_almost_equal(isentprs[0][1], trueprs, 3)


def test_isentropic_pressure_addition_args_interp():
    """Test calculation of isentropic pressure function, additional args."""
    lev = [100000.0, 95000.0, 90000.0, 85000.0] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.0
    tmp[1, :] = 292.0
    tmp[2, :] = 290.0
    tmp[3, :] = 288.0
    rh = np.ones((4, 5, 5))
    rh[0, :] = 100.0
    rh[1, :] = 80.0
    rh[2, :] = 40.0
    rh[3, :] = 20.0
    relh = rh * units.percent
    tmpk = tmp * units.kelvin
    isentlev = [296.0, 297.0] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, relh)
    truerh = 69.171 * units.percent
    assert_almost_equal(isentprs[1][1], truerh, 3)


def test_isentropic_pressure_tmp_out_interp():
    """Test calculation of isentropic pressure function, temperature output."""
    lev = [100000.0, 95000.0, 90000.0, 85000.0] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.0
    tmp[1, :] = 292.0
    tmp[2, :] = 290.0
    tmp[3, :] = 288.0
    tmpk = tmp * units.kelvin
    isentlev = [296.0, 297.0] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, temperature_out=True)
    truetmp = 291.4579 * units.kelvin
    assert_almost_equal(isentprs[1][1], truetmp, 3)


def test_isentropic_pressure_data_bounds_error():
    """Test calculation of isentropic pressure function, error for data out of bounds."""
    lev = [100000.0, 95000.0, 90000.0, 85000.0] * units.Pa
    tmp = np.ones((4, 5, 5))
    tmp[0, :] = 296.0
    tmp[1, :] = 292.0
    tmp[2, :] = 290.0
    tmp[3, :] = 288.0
    tmpk = tmp * units.kelvin
    isentlev = [296.0, 350.0] * units.kelvin
    with pytest.raises(ValueError):
        isentropic_interpolation(isentlev, lev, tmpk)


def test_isentropic_pressure_4d():
    """Test calculation of isentropic pressure function."""
    lev = [100000.0, 95000.0, 90000.0, 85000.0] * units.Pa
    tmp = np.ones((3, 4, 5, 5))
    tmp[:, 0, :] = 296.0
    tmp[:, 1, :] = 292.0
    tmp[:, 2, :] = 290
    tmp[:, 3, :] = 288.0
    tmpk = tmp * units.kelvin
    rh = np.ones((3, 4, 5, 5))
    rh[:, 0, :] = 100.0
    rh[:, 1, :] = 80.0
    rh[:, 2, :] = 40.0
    rh[:, 3, :] = 20.0
    relh = rh * units.percent
    isentlev = [296.0, 297.0, 300.0] * units.kelvin
    isentprs = isentropic_interpolation(isentlev, lev, tmpk, relh, vertical_dim=1)
    trueprs = 1000.0 * units.hPa
    trueprs2 = 936.18057 * units.hPa
    trueprs3 = 879.446 * units.hPa
    truerh = 69.171 * units.percent
    assert isentprs[0].shape == (3, 3, 5, 5)
    assert_almost_equal(isentprs[0][:, 0, :], trueprs, 3)
    assert_almost_equal(isentprs[0][:, 1, :], trueprs2, 3)
    assert_almost_equal(isentprs[0][:, 2, :], trueprs3, 3)
    assert_almost_equal(
        isentprs[1][
            :,
            1,
        ],
        truerh,
        3,
    )


def test_isentropic_interpolation_as_dataset():
    """Test calculation of isentropic interpolation with xarray."""
    data = xr.Dataset(
        {
            "temperature": (
                ("isobaric", "y", "x"),
                [[[296.0]], [[292.0]], [[290.0]], [[288.0]]] * units.K,
            ),
            "rh": (
                ("isobaric", "y", "x"),
                [[[100.0]], [[80.0]], [[40.0]], [[20.0]]] * units.percent,
            ),
        },
        coords={
            "isobaric": (("isobaric",), [1000.0, 950.0, 900.0, 850.0], {"units": "hPa"}),
            "time": "2020-01-01T00:00Z",
        },
    )
    isentlev = [296.0, 297.0] * units.kelvin
    result = isentropic_interpolation_as_dataset(isentlev, data["temperature"], data["rh"])
    expected = xr.Dataset(
        {
            "pressure": (
                ("isentropic_level", "y", "x"),
                [[[1000.0]], [[936.18057]]] * units.hPa,
                {"standard_name": "air_pressure"},
            ),
            "temperature": (
                ("isentropic_level", "y", "x"),
                [[[296.0]], [[291.4579]]] * units.K,
                {"standard_name": "air_temperature"},
            ),
            "rh": (("isentropic_level", "y", "x"), [[[100.0]], [[69.171]]] * units.percent),
        },
        coords={
            "isentropic_level": (
                ("isentropic_level",),
                [296.0, 297.0],
                {"units": "kelvin", "positive": "up"},
            ),
            "time": "2020-01-01T00:00Z",
        },
    )
    xr.testing.assert_allclose(result, expected)
    assert result["pressure"].attrs == expected["pressure"].attrs
    assert result["temperature"].attrs == expected["temperature"].attrs
    assert result["isentropic_level"].attrs == expected["isentropic_level"].attrs


@pytest.mark.parametrize("array_class", (units.Quantity, masked_array))
def test_surface_based_cape_cin(array_class):
    """Test the surface-based CAPE and CIN calculation."""
    p = array_class([959.0, 779.2, 751.3, 724.3, 700.0, 269.0], units.mbar)
    temperature = array_class([22.2, 14.6, 12.0, 9.4, 7.0, -38.0], units.celsius)
    dewpoint = array_class([19.0, -11.2, -10.8, -10.4, -10.0, -53.2], units.celsius)
    cape, cin = surface_based_cape_cin(p, temperature, dewpoint)
    assert_almost_equal(cape, 75.7340825 * units("joule / kilogram"), 2)
    assert_almost_equal(cin, -136.607809 * units("joule / kilogram"), 2)


def test_surface_based_cape_cin_with_xarray():
    """Test the surface-based CAPE and CIN calculation with xarray."""
    data = xr.Dataset(
        {
            "temperature": (("isobaric",), [22.2, 14.6, 12.0, 9.4, 7.0, -38.0] * units.degC),
            "dewpoint": (
                ("isobaric",),
                [19.0, -11.2, -10.8, -10.4, -10.0, -53.2] * units.degC,
            ),
        },
        coords={
            "isobaric": (
                ("isobaric",),
                [959.0, 779.2, 751.3, 724.3, 700.0, 269.0],
                {"units": "hPa"},
            )
        },
    )
    cape, cin = surface_based_cape_cin(data["isobaric"], data["temperature"], data["dewpoint"])
    assert_almost_equal(cape, 75.7340825 * units("joule / kilogram"), 2)
    assert_almost_equal(cin, -136.607809 * units("joule / kilogram"), 2)


def test_profile_with_nans():
    """Test a profile with nans to make sure it calculates functions appropriately (#1187)."""
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
    temperature = (
        np.array(
            [
                -22.5,
                -22.7,
                -23.1,
                np.nan,
                -24.5,
                -25.1,
                np.nan,
                -24.5,
                -23.9,
                np.nan,
                -24.7,
                np.nan,
                -21.3,
                -21.3,
                -22.7,
                np.nan,
                -20.7,
                -16.3,
                -15.5,
                np.nan,
                np.nan,
                -15.3,
                np.nan,
                -17.3,
                -20.9,
                -22.5,
                np.nan,
                -25.5,
                np.nan,
                -31.5,
                np.nan,
                -31.5,
                -34.1,
                -34.3,
                -37.3,
                -37.7,
                -39.5,
                -42.1,
                -43.1,
                -45.1,
                -45.7,
                -46.7,
            ]
        )
        * units.degC
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
                -21,
                -23.7,
                np.nan,
                -28.3,
                np.nan,
                -32.6,
                np.nan,
                -33.8,
                -35,
                -35.1,
                -38.1,
                -40,
                -43.3,
                -44.6,
                -46.4,
                -47,
                -49.2,
                -50.7,
            ]
        )
        * units.degC
    )
    lfc_p, _ = lfc(pressure, temperature, dewpoint)
    profile = parcel_profile(pressure, temperature[0], dewpoint[0])
    cape, cin = cape_cin(pressure, temperature, dewpoint, profile)
    sbcape, sbcin = surface_based_cape_cin(pressure, temperature, dewpoint)
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    assert_nan(lfc_p, units.hPa)
    assert_almost_equal(cape, 0 * units("J/kg"), 0)
    assert_almost_equal(cin, 0 * units("J/kg"), 0)
    assert_almost_equal(sbcape, 0 * units("J/kg"), 0)
    assert_almost_equal(sbcin, 0 * units("J/kg"), 0)
    assert_almost_equal(mucape, 0 * units("J/kg"), 0)
    assert_almost_equal(mucin, 0 * units("J/kg"), 0)


def test_most_unstable_cape_cin_surface():
    """Test the most unstable CAPE/CIN calculation when surface is most unstable."""
    pressure = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.mbar
    temperature = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * units.celsius
    dewpoint = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * units.celsius
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    assert_almost_equal(mucape, 75.7340825 * units("joule / kilogram"), 2)
    assert_almost_equal(mucin, -136.607809 * units("joule / kilogram"), 2)


def test_most_unstable_cape_cin():
    """Test the most unstable CAPE/CIN calculation."""
    pressure = np.array([1000.0, 959.0, 867.9, 850.0, 825.0, 800.0]) * units.mbar
    temperature = np.array([18.2, 22.2, 17.4, 10.0, 0.0, 15]) * units.celsius
    dewpoint = np.array([19.0, 19.0, 14.3, 0.0, -10.0, 0.0]) * units.celsius
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    assert_almost_equal(mucape, 157.1401 * units("joule / kilogram"), 4)
    assert_almost_equal(mucin, -31.82547 * units("joule / kilogram"), 4)


def test_mixed_parcel():
    """Test the mixed parcel calculation."""
    pressure = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.hPa
    temperature = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * units.degC
    dewpoint = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * units.degC
    parcel_pressure, parcel_temperature, parcel_dewpoint = mixed_parcel(
        pressure, temperature, dewpoint, depth=250 * units.hPa
    )
    assert_almost_equal(parcel_pressure, 959.0 * units.hPa, 6)
    assert_almost_equal(parcel_temperature, 28.7363771 * units.degC, 6)
    assert_almost_equal(parcel_dewpoint, 7.1534658 * units.degC, 6)


def test_mixed_layer_cape_cin(multiple_intersections):
    """Test the calculation of mixed layer cape/cin."""
    pressure, temperature, dewpoint = multiple_intersections
    mlcape, mlcin = mixed_layer_cape_cin(pressure, temperature, dewpoint)
    assert_almost_equal(mlcape, 991.4484 * units("joule / kilogram"), 2)
    assert_almost_equal(mlcin, -20.6552 * units("joule / kilogram"), 2)


def test_mixed_layer():
    """Test the mixed layer calculation."""
    pressure = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.hPa
    temperature = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * units.degC
    mixed_layer_temperature = mixed_layer(pressure, temperature, depth=250 * units.hPa)[0]
    assert_almost_equal(mixed_layer_temperature, 16.4024930 * units.degC, 6)


def test_dry_static_energy():
    """Test the dry static energy calculation."""
    dse = dry_static_energy(1000 * units.m, 25 * units.degC)
    assert_almost_equal(dse, 309.4474 * units("kJ/kg"), 6)


def test_moist_static_energy():
    """Test the moist static energy calculation."""
    mse = moist_static_energy(1000 * units.m, 25 * units.degC, 0.012 * units.dimensionless)
    assert_almost_equal(mse, 339.4594 * units("kJ/kg"), 6)


def test_thickness_hydrostatic():
    """Test the thickness calculation for a moist layer."""
    pressure = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.hPa
    temperature = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * units.degC
    mixing = np.array([0.01458, 0.00209, 0.00224, 0.00240, 0.00256, 0.00010])
    thickness = thickness_hydrostatic(pressure, temperature, mixing_ratio=mixing)
    assert_almost_equal(thickness, 9892.07 * units.m, 2)


def test_thickness_hydrostatic_subset():
    """Test the thickness calculation with a subset of the moist layer."""
    pressure = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.hPa
    temperature = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * units.degC
    mixing = np.array([0.01458, 0.00209, 0.00224, 0.00240, 0.00256, 0.00010])
    thickness = thickness_hydrostatic(
        pressure,
        temperature,
        mixing_ratio=mixing,
        bottom=850 * units.hPa,
        depth=150 * units.hPa,
    )
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
    thickness = thickness_hydrostatic(
        pressure, temperature, bottom=850 * units.hPa, depth=350 * units.hPa
    )
    assert_almost_equal(thickness, 4242.68 * units.m, 2)


def test_thickness_hydrostatic_from_relative_humidity():
    """Test the thickness calculation for a moist layer using RH data."""
    pressure = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.hPa
    temperature = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * units.degC
    relative_humidity = np.array([81.69, 15.43, 18.95, 23.32, 28.36, 18.55]) * units.percent
    thickness = thickness_hydrostatic_from_relative_humidity(
        pressure, temperature, relative_humidity
    )
    assert_almost_equal(thickness, 9892.07 * units.m, 2)


def test_mixing_ratio_dimensions():
    """Verify mixing ratio returns a dimensionless number."""
    p = 998.0 * units.mbar
    e = 73.75 * units.hPa
    assert str(mixing_ratio(e, p).units) == "dimensionless"


def test_saturation_mixing_ratio_dimensions():
    """Verify saturation mixing ratio returns a dimensionless number."""
    p = 998.0 * units.mbar
    temp = 20 * units.celsius
    assert str(saturation_mixing_ratio(p, temp).units) == "dimensionless"


def test_mixing_ratio_from_rh_dimensions():
    """Verify mixing ratio from RH returns a dimensionless number."""
    p = 1000.0 * units.mbar
    temperature = 0.0 * units.degC
    rh = 100.0 * units.percent
    assert (
        str(mixing_ratio_from_relative_humidity(p, temperature, rh).units) == "dimensionless"
    )


@pytest.fixture
def bv_data():
    """Return height and potential temperature data for testing Brunt-Vaisala functions."""
    heights = [1000.0, 1500.0, 2000.0, 2500.0] * units("m")
    potential_temperatures = [
        [290.0, 290.0, 290.0, 290.0],
        [292.0, 293.0, 293.0, 292.0],
        [294.0, 296.0, 293.0, 293.0],
        [296.0, 295.0, 293.0, 296.0],
    ] * units("K")
    return heights, potential_temperatures


def test_brunt_vaisala_frequency_squared(bv_data):
    """Test Brunt-Vaisala frequency squared function."""
    truth = [
        [1.35264138e-04, 2.02896207e-04, 3.04344310e-04, 1.69080172e-04],
        [1.34337671e-04, 2.00818771e-04, 1.00409386e-04, 1.00753253e-04],
        [1.33423810e-04, 6.62611486e-05, 0, 1.33879181e-04],
        [1.32522297e-04, -1.99457288e-04, 0.0, 2.65044595e-04],
    ] * units("s^-2")
    bv_freq_sqr = brunt_vaisala_frequency_squared(bv_data[0], bv_data[1])
    assert_almost_equal(bv_freq_sqr, truth, 6)


def test_brunt_vaisala_frequency(bv_data):
    """Test Brunt-Vaisala frequency function."""
    truth = [
        [0.01163031, 0.01424416, 0.01744547, 0.01300308],
        [0.01159041, 0.01417105, 0.01002045, 0.01003759],
        [0.01155092, 0.00814010, 0.0, 0.01157062],
        [0.01151183, np.nan, 0.0, 0.01628019],
    ] * units("s^-1")
    bv_freq = brunt_vaisala_frequency(bv_data[0], bv_data[1])
    assert_almost_equal(bv_freq, truth, 6)


def test_brunt_vaisala_period(bv_data):
    """Test Brunt-Vaisala period function."""
    truth = [
        [540.24223556, 441.10593821, 360.16149037, 483.20734521],
        [542.10193894, 443.38165033, 627.03634320, 625.96540075],
        [543.95528431, 771.88106656, np.nan, 543.02940230],
        [545.80233643, np.nan, np.nan, 385.94053328],
    ] * units("s")
    bv_period = brunt_vaisala_period(bv_data[0], bv_data[1])
    assert_almost_equal(bv_period, truth, 6)


def test_wet_bulb_temperature():
    """Test wet bulb calculation with scalars."""
    val = wet_bulb_temperature(1000 * units.hPa, 25 * units.degC, 15 * units.degC)
    truth = 18.34345936 * units.degC  # 18.59 from NWS calculator
    assert_almost_equal(val, truth, 5)


def test_wet_bulb_temperature_1d():
    """Test wet bulb calculation with 1d list."""
    pressures = [1013, 1000, 990] * units.hPa
    temperatures = [25, 20, 15] * units.degC
    dewpoints = [20, 15, 10] * units.degC
    val = wet_bulb_temperature(pressures, temperatures, dewpoints)
    truth = [21.4449794, 16.7368576, 12.0656909] * units.degC
    # 21.58, 16.86, 12.18 from NWS Calculator
    assert_array_almost_equal(val, truth, 5)


def test_wet_bulb_temperature_2d():
    """Test wet bulb calculation with 2d list."""
    pressures = [[1013, 1000, 990], [1012, 999, 989]] * units.hPa
    temperatures = [[25, 20, 15], [24, 19, 14]] * units.degC
    dewpoints = [[20, 15, 10], [19, 14, 9]] * units.degC
    val = wet_bulb_temperature(pressures, temperatures, dewpoints)
    truth = [
        [21.4449794, 16.7368576, 12.0656909],
        [20.5021631, 15.801218, 11.1361878],
    ] * units.degC
    # 21.58, 16.86, 12.18
    # 20.6, 15.9, 11.2 from NWS Calculator
    assert_array_almost_equal(val, truth, 5)


def test_static_stability_adiabatic():
    """Test static stability calculation with a dry adiabatic profile."""
    pressures = [1000.0, 900.0, 800.0, 700.0, 600.0, 500.0] * units.hPa
    temperature_start = 20 * units.degC
    temperatures = dry_lapse(pressures, temperature_start)
    sigma = static_stability(pressures, temperatures)
    truth = np.zeros_like(pressures) * units("J kg^-1 hPa^-2")
    # Should be zero with a dry adiabatic profile
    assert_almost_equal(sigma, truth, 6)


def test_static_stability_cross_section():
    """Test static stability calculation with a 2D cross-section."""
    pressures = [
        [850.0, 700.0, 500.0],
        [850.0, 700.0, 500.0],
        [850.0, 700.0, 500.0],
    ] * units.hPa
    temperatures = [[17.0, 11.0, -10.0], [16.0, 10.0, -11.0], [11.0, 6.0, -12.0]] * units.degC
    sigma = static_stability(pressures, temperatures, vertical_dim=1)
    truth = [
        [0.02819452, 0.02016804, 0.00305262],
        [0.02808841, 0.01999462, 0.00274956],
        [0.02840196, 0.02366708, 0.0131604],
    ] * units("J kg^-1 hPa^-2")
    assert_almost_equal(sigma, truth, 6)


def test_dewpoint_specific_humidity():
    """Test relative humidity from specific humidity."""
    p = 1013.25 * units.mbar
    temperature = 20.0 * units.degC
    q = 0.012 * units.dimensionless
    td = dewpoint_from_specific_humidity(p, temperature, q)
    assert_almost_equal(td, 16.973 * units.degC, 3)


def test_lfc_not_below_lcl():
    """Test sounding where LFC appears to be (but isn't) below LCL."""
    levels = (
        np.array(
            [
                1002.5,
                1001.7,
                1001.0,
                1000.3,
                999.7,
                999.0,
                998.2,
                977.9,
                966.2,
                952.3,
                940.6,
                930.5,
                919.8,
                909.1,
                898.9,
                888.4,
                878.3,
                868.1,
                858.0,
                848.0,
                837.2,
                827.0,
                816.7,
                805.4,
            ]
        )
        * units.hPa
    )
    temperatures = (
        np.array(
            [
                17.9,
                17.9,
                17.8,
                17.7,
                17.7,
                17.6,
                17.5,
                16.0,
                15.2,
                14.5,
                13.8,
                13.0,
                12.5,
                11.9,
                11.4,
                11.0,
                10.3,
                9.7,
                9.2,
                8.7,
                8.0,
                7.4,
                6.8,
                6.1,
            ]
        )
        * units.degC
    )
    dewpoints = (
        np.array(
            [
                13.6,
                13.6,
                13.5,
                13.5,
                13.5,
                13.5,
                13.4,
                12.5,
                12.1,
                11.8,
                11.4,
                11.3,
                11.0,
                9.3,
                10.0,
                8.7,
                8.9,
                8.6,
                8.1,
                7.6,
                7.0,
                6.5,
                6.0,
                5.4,
            ]
        )
        * units.degC
    )
    lfc_pressure, lfc_temp = lfc(levels, temperatures, dewpoints)
    # Before patch, LFC pressure would show 1000.5912165339967 hPa
    assert_almost_equal(lfc_pressure, 811.8263397 * units.mbar, 3)
    assert_almost_equal(lfc_temp, 6.4992871 * units.celsius, 3)


@pytest.fixture
def multiple_intersections():
    """Create profile with multiple LFCs and ELs for testing."""
    levels = (
        np.array(
            [
                966.0,
                937.2,
                925.0,
                904.6,
                872.6,
                853.0,
                850.0,
                836.0,
                821.0,
                811.6,
                782.3,
                754.2,
                726.9,
                700.0,
                648.9,
                624.6,
                601.1,
                595.0,
                587.0,
                576.0,
                555.7,
                534.2,
                524.0,
                500.0,
                473.3,
                400.0,
                384.5,
                358.0,
                343.0,
                308.3,
                300.0,
                276.0,
                273.0,
                268.5,
                250.0,
                244.2,
                233.0,
                200.0,
            ]
        )
        * units.mbar
    )
    temperatures = (
        np.array(
            [
                18.2,
                16.8,
                16.2,
                15.1,
                13.3,
                12.2,
                12.4,
                14.0,
                14.4,
                13.7,
                11.4,
                9.1,
                6.8,
                4.4,
                -1.4,
                -4.4,
                -7.3,
                -8.1,
                -7.9,
                -7.7,
                -8.7,
                -9.8,
                -10.3,
                -13.5,
                -17.1,
                -28.1,
                -30.7,
                -35.3,
                -37.1,
                -43.5,
                -45.1,
                -49.9,
                -50.4,
                -51.1,
                -54.1,
                -55.0,
                -56.7,
                -57.5,
            ]
        )
        * units.degC
    )
    dewpoints = (
        np.array(
            [
                16.9,
                15.9,
                15.5,
                14.2,
                12.1,
                10.8,
                8.6,
                0.0,
                -3.6,
                -4.4,
                -6.9,
                -9.5,
                -12.0,
                -14.6,
                -15.8,
                -16.4,
                -16.9,
                -17.1,
                -27.9,
                -42.7,
                -44.1,
                -45.6,
                -46.3,
                -45.5,
                -47.1,
                -52.1,
                -50.4,
                -47.3,
                -57.1,
                -57.9,
                -58.1,
                -60.9,
                -61.4,
                -62.1,
                -65.1,
                -65.6,
                -66.7,
                -70.5,
            ]
        )
        * units.degC
    )
    return levels, temperatures, dewpoints


def test_multiple_lfcs_simple(multiple_intersections):
    """Test sounding with multiple LFCs.

    If which='top', return lowest-pressure LFC.
    If which='bottom', return the highest-pressure LFC.
    If which='all', return all LFCs

    """
    levels, temperatures, dewpoints = multiple_intersections
    lfc_pressure_top, lfc_temp_top = lfc(levels, temperatures, dewpoints)
    lfc_pressure_bottom, lfc_temp_bottom = lfc(levels, temperatures, dewpoints, which="bottom")
    lfc_pressure_all, _ = lfc(levels, temperatures, dewpoints, which="all")
    assert_almost_equal(lfc_pressure_top, 705.4346277 * units.mbar, 3)
    assert_almost_equal(lfc_temp_top, 4.8922235 * units.degC, 3)
    assert_almost_equal(lfc_pressure_bottom, 884.1954356 * units.mbar, 3)
    assert_almost_equal(lfc_temp_bottom, 13.9597571 * units.degC, 3)
    assert_almost_equal(len(lfc_pressure_all), 2, 0)


def test_multiple_lfs_wide(multiple_intersections):
    """Test 'wide' LFC for sounding with multiple LFCs."""
    levels, temperatures, dewpoints = multiple_intersections
    lfc_pressure_wide, lfc_temp_wide = lfc(levels, temperatures, dewpoints, which="wide")
    assert_almost_equal(lfc_pressure_wide, 705.4346277 * units.hPa, 3)
    assert_almost_equal(lfc_temp_wide, 4.8922235 * units.degC, 3)


def test_invalid_which(multiple_intersections):
    """Test error message for invalid which option for LFC and EL."""
    levels, temperatures, dewpoints = multiple_intersections
    with pytest.raises(ValueError):
        lfc(levels, temperatures, dewpoints, which="test")
    with pytest.raises(ValueError):
        el(levels, temperatures, dewpoints, which="test")


def test_multiple_els_simple(multiple_intersections):
    """Test sounding with multiple ELs.

    If which='top', return lowest-pressure EL.
    If which='bottom', return the highest-pressure EL.
    If which='all', return all ELs

    """
    levels, temperatures, dewpoints = multiple_intersections
    el_pressure_top, el_temp_top = el(levels, temperatures, dewpoints)
    el_pressure_bottom, el_temp_bottom = el(levels, temperatures, dewpoints, which="bottom")
    el_pressure_all, _ = el(levels, temperatures, dewpoints, which="all")
    assert_almost_equal(el_pressure_top, 228.0575059 * units.mbar, 3)
    assert_almost_equal(el_temp_top, -56.8123126 * units.degC, 3)
    assert_almost_equal(el_pressure_bottom, 849.7942185 * units.mbar, 3)
    assert_almost_equal(el_temp_bottom, 12.4233265 * units.degC, 3)
    assert_almost_equal(len(el_pressure_all), 2, 0)


def test_multiple_el_wide(multiple_intersections):
    """Test 'wide' EL for sounding with multiple ELs."""
    levels, temperatures, dewpoints = multiple_intersections
    el_pressure_wide, el_temp_wide = el(levels, temperatures, dewpoints, which="wide")
    assert_almost_equal(el_pressure_wide, 228.0575059 * units.hPa, 3)
    assert_almost_equal(el_temp_wide, -56.8123126 * units.degC, 3)


def test_muliple_el_most_cape(multiple_intersections):
    """Test 'most_cape' EL for sounding with multiple ELs."""
    levels, temperatures, dewpoints = multiple_intersections
    el_pressure_wide, el_temp_wide = el(levels, temperatures, dewpoints, which="most_cape")
    assert_almost_equal(el_pressure_wide, 228.0575059 * units.hPa, 3)
    assert_almost_equal(el_temp_wide, -56.8123126 * units.degC, 3)


def test_muliple_lfc_most_cape(multiple_intersections):
    """Test 'most_cape' LFC for sounding with multiple LFCs."""
    levels, temperatures, dewpoints = multiple_intersections
    lfc_pressure_wide, lfc_temp_wide = lfc(levels, temperatures, dewpoints, which="most_cape")
    assert_almost_equal(lfc_pressure_wide, 705.4346277 * units.hPa, 3)
    assert_almost_equal(lfc_temp_wide, 4.8922235 * units.degC, 3)


def test_el_lfc_most_cape_bottom():
    """Test 'most_cape' LFC/EL when the bottom combination produces the most CAPE."""
    levels = (
        np.array(
            [
                966.0,
                937.2,
                904.6,
                872.6,
                853.0,
                850.0,
                836.0,
                821.0,
                811.6,
                782.3,
                754.2,
                726.9,
                700.0,
                648.9,
            ]
        )
        * units.mbar
    )
    temperatures = (
        np.array(
            [18.2, 16.5, 15.1, 11.5, 11.0, 12.4, 14.0, 14.4, 13.7, 11.4, 9.1, 6.8, 3.8, 1.5]
        )
        * units.degC
    )
    dewpoints = (
        np.array(
            [16.9, 15.9, 14.2, 11, 9.5, 8.6, 0.0, -3.6, -4.4, -6.9, -9.5, -12.0, -14.6, -15.8]
        )
        * units.degC
    )
    lfc_pres, lfc_temp = lfc(levels, temperatures, dewpoints, which="most_cape")
    el_pres, el_temp = el(levels, temperatures, dewpoints, which="most_cape")
    assert_almost_equal(lfc_pres, 900.7395292 * units.hPa, 3)
    assert_almost_equal(lfc_temp, 14.672512 * units.degC, 3)
    assert_almost_equal(el_pres, 849.7942184 * units.hPa, 3)
    assert_almost_equal(el_temp, 12.4233265 * units.degC, 3)


def test_cape_cin_top_el_lfc(multiple_intersections):
    """Test using LFC/EL options for CAPE/CIN."""
    levels, temperatures, dewpoints = multiple_intersections
    parcel_prof = parcel_profile(levels, temperatures[0], dewpoints[0]).to("degC")
    cape, cin = cape_cin(levels, temperatures, dewpoints, parcel_prof, which_lfc="top")
    assert_almost_equal(cape, 1262.8618 * units("joule / kilogram"), 3)
    assert_almost_equal(cin, -97.6499 * units("joule / kilogram"), 3)


def test_cape_cin_bottom_el_lfc(multiple_intersections):
    """Test using LFC/EL options for CAPE/CIN."""
    levels, temperatures, dewpoints = multiple_intersections
    parcel_prof = parcel_profile(levels, temperatures[0], dewpoints[0]).to("degC")
    cape, cin = cape_cin(levels, temperatures, dewpoints, parcel_prof, which_el="bottom")
    assert_almost_equal(cape, 2.1967 * units("joule / kilogram"), 3)
    assert_almost_equal(cin, -8.1545 * units("joule / kilogram"), 3)


def test_cape_cin_wide_el_lfc(multiple_intersections):
    """Test using LFC/EL options for CAPE/CIN."""
    levels, temperatures, dewpoints = multiple_intersections
    parcel_prof = parcel_profile(levels, temperatures[0], dewpoints[0]).to("degC")
    cape, cin = cape_cin(
        levels, temperatures, dewpoints, parcel_prof, which_lfc="wide", which_el="wide"
    )
    assert_almost_equal(cape, 1262.8618 * units("joule / kilogram"), 3)
    assert_almost_equal(cin, -97.6499 * units("joule / kilogram"), 3)


def test_cape_cin_custom_profile():
    """Test the CAPE and CIN calculation with a custom profile passed to LFC and EL."""
    p = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * units.mbar
    temperature = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * units.celsius
    dewpoint = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * units.celsius
    parcel_prof = parcel_profile(p, temperature[0], dewpoint[0]) + 5 * units.delta_degC
    cape, cin = cape_cin(p, temperature, dewpoint, parcel_prof)
    assert_almost_equal(cape, 1443.505086499895 * units("joule / kilogram"), 2)
    assert_almost_equal(cin, 0.0 * units("joule / kilogram"), 2)


def test_parcel_profile_below_lcl():
    """Test parcel profile calculation when pressures do not reach LCL (#827)."""
    pressure = (
        np.array(
            [
                981,
                949.2,
                925.0,
                913.9,
                903,
                879.4,
                878,
                864,
                855,
                850,
                846.3,
                838,
                820,
                814.5,
                799,
                794,
            ]
        )
        * units.hPa
    )
    truth = (
        np.array(
            [
                276.35,
                273.76110242,
                271.74910213,
                270.81364639,
                269.88711359,
                267.85332225,
                267.73145436,
                266.5050728,
                265.70916946,
                265.264412,
                264.93408677,
                264.18931638,
                262.55585912,
                262.0516423,
                260.61745662,
                260.15057861,
            ]
        )
        * units.kelvin
    )
    profile = parcel_profile(pressure, 3.2 * units.degC, -10.8 * units.degC)
    assert_almost_equal(profile, truth, 6)


def test_vertical_velocity_pressure_dry_air():
    """Test conversion of w to omega assuming dry air."""
    w = 1 * units("cm/s")
    omega_truth = -1.250690495 * units("microbar/second")
    omega_test = vertical_velocity_pressure(w, 1000.0 * units.mbar, 273.15 * units.K)
    assert_almost_equal(omega_test, omega_truth, 6)


def test_vertical_velocity_dry_air():
    """Test conversion of w to omega assuming dry air."""
    omega = 1 * units("microbar/second")
    w_truth = -0.799558327 * units("cm/s")
    w_test = vertical_velocity(omega, 1000.0 * units.mbar, 273.15 * units.K)
    assert_almost_equal(w_test, w_truth, 6)


def test_vertical_velocity_pressure_moist_air():
    """Test conversion of w to omega assuming moist air."""
    w = -1 * units("cm/s")
    omega_truth = 1.032100858 * units("microbar/second")
    omega_test = vertical_velocity_pressure(
        w, 850.0 * units.mbar, 280.0 * units.K, 8 * units("g/kg")
    )
    assert_almost_equal(omega_test, omega_truth, 6)


def test_vertical_velocity_moist_air():
    """Test conversion of w to omega assuming moist air."""
    omega = -1 * units("microbar/second")
    w_truth = 0.968897557 * units("cm/s")
    w_test = vertical_velocity(omega, 850.0 * units.mbar, 280.0 * units.K, 8 * units("g/kg"))
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
    pressure = (
        np.array(
            [
                1012.0,
                1009.0,
                1002.0,
                1000.0,
                925.0,
                896.0,
                855.0,
                850.0,
                849.0,
                830.0,
                775.0,
                769.0,
                758.0,
                747.0,
                741.0,
                731.0,
                712.0,
                700.0,
                691.0,
                671.0,
                636.0,
                620.0,
                610.0,
                601.0,
                594.0,
                587.0,
                583.0,
                580.0,
                571.0,
                569.0,
                554.0,
                530.0,
                514.0,
                506.0,
                502.0,
                500.0,
                492.0,
                484.0,
                475.0,
                456.0,
                449.0,
                442.0,
                433.0,
                427.0,
                400.0,
                395.0,
                390.0,
                351.0,
                300.0,
                298.0,
                294.0,
                274.0,
                250.0,
            ]
        )
        * units.hPa
    )
    temperature = (
        np.array(
            [
                27.8,
                25.8,
                24.2,
                24,
                18.8,
                16,
                13,
                12.6,
                12.6,
                11.6,
                9.2,
                8.6,
                8.4,
                9.2,
                10,
                9.4,
                7.4,
                6.2,
                5.2,
                3.2,
                -0.3,
                -2.3,
                -3.3,
                -4.5,
                -5.5,
                -6.1,
                -6.1,
                -6.1,
                -6.3,
                -6.3,
                -7.7,
                -9.5,
                -9.9,
                -10.3,
                -10.9,
                -11.1,
                -11.9,
                -12.7,
                -13.7,
                -16.1,
                -16.9,
                -17.9,
                -19.1,
                -19.9,
                -23.9,
                -24.7,
                -25.3,
                -29.5,
                -39.3,
                -39.7,
                -40.5,
                -44.3,
                -49.3,
            ]
        )
        * units.degC
    )
    dewpoint = (
        np.array(
            [
                19.8,
                16.8,
                16.2,
                16,
                13.8,
                12.8,
                10.1,
                9.7,
                9.7,
                8.6,
                4.2,
                3.9,
                0.4,
                -5.8,
                -32,
                -34.6,
                -35.6,
                -34.8,
                -32.8,
                -10.8,
                -9.3,
                -10.3,
                -9.3,
                -10.5,
                -10.5,
                -10,
                -16.1,
                -19.1,
                -23.3,
                -18.3,
                -17.7,
                -20.5,
                -27.9,
                -32.3,
                -33.9,
                -34.1,
                -35.9,
                -26.7,
                -37.7,
                -43.1,
                -33.9,
                -40.9,
                -46.1,
                -34.9,
                -33.9,
                -33.7,
                -33.3,
                -42.5,
                -50.3,
                -49.7,
                -49.5,
                -58.3,
                -61.3,
            ]
        )
        * units.degC
    )
    cape, cin = surface_based_cape_cin(pressure, temperature, dewpoint)
    expected_cape, expected_cin = [2010.4136 * units("joules/kg"), 0.0 * units("joules/kg")]
    assert_almost_equal(cape, expected_cape, 3)
    assert_almost_equal(cin, expected_cin, 3)


def test_lcl_grid_surface_LCLs():
    """Test surface grid where some values have LCLs at the surface."""
    pressure = np.array([1000, 990, 1010]) * units.hPa
    temperature = np.array([15, 14, 13]) * units.degC
    dewpoint = np.array([15, 10, 13]) * units.degC
    lcl_pressure, lcl_temperature = lcl(pressure, temperature, dewpoint)
    pres_truth = np.array([1000, 932.1515324, 1010]) * units.hPa
    temp_truth = np.array([15, 9.10391763, 13]) * units.degC
    assert_array_almost_equal(lcl_pressure, pres_truth, 4)
    assert_array_almost_equal(lcl_temperature, temp_truth, 4)


def test_lifted_index():
    """Test the Lifted Index calculation."""
    pressure = (
        np.array(
            [
                1014.0,
                1000.0,
                997.0,
                981.2,
                947.4,
                925.0,
                914.9,
                911.0,
                902.0,
                883.0,
                850.0,
                822.3,
                816.0,
                807.0,
                793.2,
                770.0,
                765.1,
                753.0,
                737.5,
                737.0,
                713.0,
                700.0,
                688.0,
                685.0,
                680.0,
                666.0,
                659.8,
                653.0,
                643.0,
                634.0,
                615.0,
                611.8,
                566.2,
                516.0,
                500.0,
                487.0,
                484.2,
                481.0,
                475.0,
                460.0,
                400.0,
            ]
        )
        * units.hPa
    )
    temperature = (
        np.array(
            [
                24.2,
                24.2,
                24.0,
                23.1,
                21.0,
                19.6,
                18.7,
                18.4,
                19.2,
                19.4,
                17.2,
                15.3,
                14.8,
                14.4,
                13.4,
                11.6,
                11.1,
                10.0,
                8.8,
                8.8,
                8.2,
                7.0,
                5.6,
                5.6,
                5.6,
                4.4,
                3.8,
                3.2,
                3.0,
                3.2,
                1.8,
                1.5,
                -3.4,
                -9.3,
                -11.3,
                -13.1,
                -13.1,
                -13.1,
                -13.7,
                -15.1,
                -23.5,
            ]
        )
        * units.degC
    )
    dewpoint = (
        np.array(
            [
                23.2,
                23.1,
                22.8,
                22.0,
                20.2,
                19.0,
                17.6,
                17.0,
                16.8,
                15.5,
                14.0,
                11.7,
                11.2,
                8.4,
                7.0,
                4.6,
                5.0,
                6.0,
                4.2,
                4.1,
                -1.8,
                -2.0,
                -1.4,
                -0.4,
                -3.4,
                -5.6,
                -4.3,
                -2.8,
                -7.0,
                -25.8,
                -31.2,
                -31.4,
                -34.1,
                -37.3,
                -32.3,
                -34.1,
                -37.3,
                -41.1,
                -37.7,
                -58.1,
                -57.5,
            ]
        )
        * units.degC
    )
    parcel_prof = parcel_profile(pressure, temperature[0], dewpoint[0])
    LI = lifted_index(pressure, temperature, parcel_prof)
    assert_almost_equal(LI, -7.9176350 * units.delta_degree_Celsius, 2)


def test_gradient_richardson_number():
    """Test gradient Richardson number calculation."""
    theta = units("K") * np.asarray([254.5, 258.3, 262.2])
    u_wnd = units("m/s") * np.asarray([-2.0, -1.1, 0.23])
    v_wnd = units("m/s") * np.asarray([3.3, 4.2, 5.2])
    height = units("km") * np.asarray([0.2, 0.4, 0.6])

    result = gradient_richardson_number(height, theta, u_wnd, v_wnd)
    expected = np.asarray([24.2503551, 13.6242603, 8.4673744])

    assert_array_almost_equal(result, expected, 4)


def test_gradient_richardson_number_with_xarray():
    """Test gradient Richardson number calculation using xarray."""
    data = xr.Dataset(
        {
            "theta": (("height",), [254.5, 258.3, 262.2] * units.K),
            "u_wind": (("height",), [-2.0, -1.1, 0.23] * units("m/s")),
            "v_wind": (("height",), [3.3, 4.2, 5.2] * units("m/s")),
            "Ri_g": (("height",), [24.2503551, 13.6242603, 8.4673744]),
        },
        coords={"height": (("height",), [0.2, 0.4, 0.6], {"units": "kilometer"})},
    )

    result = gradient_richardson_number(
        data["height"], data["theta"], data["u_wind"], data["v_wind"]
    )

    assert isinstance(result, xr.DataArray)
    xr.testing.assert_identical(result["height"], data["Ri_g"]["height"])
    assert_array_almost_equal(result.data.m_as(""), data["Ri_g"].data)
