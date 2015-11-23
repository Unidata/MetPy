# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from metpy.calc.thermo import *  # noqa
from metpy.units import units
from metpy.testing import assert_almost_equal, assert_array_almost_equal


class TestPotentialTemperature(object):
    def test_basic(self):
        temp = np.array([278., 283., 291., 298.]) * units.kelvin
        pres = np.array([900., 500., 300., 100.]) * units.mbar
        real_th = np.array([286.5, 345.0155, 410.5467, 575.5397]) * units.kelvin
        assert_array_almost_equal(potential_temperature(pres, temp), real_th, 3)

    def test_scalar(self):
        assert_almost_equal(potential_temperature(1000. * units.mbar, 293. * units.kelvin),
                            293. * units.kelvin, 4)
        assert_almost_equal(potential_temperature(800. * units.mbar, 293. * units.kelvin),
                            312.2987 * units.kelvin, 4)

    def test_farenheit(self):
        assert_almost_equal(potential_temperature(800. * units.mbar, 68. * units.degF),
                            (312.4586 * units.kelvin).to(units.degF), 4)


class TestDryLapse(object):
    def test_array(self):
        levels = np.array([1000, 900, 864.89]) * units.mbar
        temps = dry_lapse(levels, 303.15 * units.kelvin)
        assert_array_almost_equal(temps,
                                  np.array([303.15, 294.16, 290.83]) * units.kelvin, 2)

    def test_2_levels(self):
        temps = dry_lapse(np.array([1000., 500.]) * units.mbar, 293. * units.kelvin)
        assert_array_almost_equal(temps, [293., 240.3341] * units.kelvin, 4)


class TestMoistLapse(object):
    def test_array(self):
        temp = moist_lapse(np.array([1000., 800., 600., 500., 400.]) * units.mbar,
                           293. * units.kelvin)
        true_temp = np.array([293, 284.64, 272.8, 264.4, 252.87]) * units.kelvin
        assert_array_almost_equal(temp, true_temp, 2)

    def test_degc(self):
        'Test moist lapse with Celsius'
        temp = moist_lapse(np.array([1000., 800., 600., 500., 400.]) * units.mbar,
                           19.85 * units.degC)
        true_temp = np.array([293, 284.64, 272.8, 264.4, 252.87]) * units.kelvin
        assert_array_almost_equal(temp, true_temp, 2)


class TestParcelProfile(object):
    def test_basic(self):
        levels = np.array([1000., 900., 800., 700., 600., 500., 400.]) * units.mbar
        true_prof = np.array([303.15, 294.16, 288.02, 283.06, 277.04, 269.38,
                              258.93]) * units.kelvin

        prof = parcel_profile(levels, 30. * units.degC, 20. * units.degC)
        assert_array_almost_equal(prof, true_prof, 2)


class TestSatVaporPressure(object):
    def test_basic(self):
        temp = np.array([5., 10., 18., 25.]) * units.degC
        real_es = np.array([8.72, 12.27, 20.63, 31.67]) * units.mbar
        assert_array_almost_equal(saturation_vapor_pressure(temp), real_es, 2)

    def test_scalar(self):
        es = saturation_vapor_pressure(0 * units.degC)
        assert_almost_equal(es, 6.112 * units.mbar, 3)

    def test_farenheit(self):
        temp = np.array([50., 68.]) * units.degF
        real_es = np.array([12.2717, 23.3695]) * units.mbar
        assert_array_almost_equal(saturation_vapor_pressure(temp), real_es, 4)


class TestDewpointRH(object):
    def test_basic(self):
        temp = np.array([30., 25., 10., 20., 25.]) * units.degC
        rh = np.array([30., 45., 55., 80., 85.]) / 100.

        real_td = np.array([11, 12, 1, 16, 22]) * units.degC
        assert_array_almost_equal(real_td, dewpoint_rh(temp, rh), 0)

    def test_scalar(self):
        td = dewpoint_rh(10.6 * units.degC, 0.37)
        assert_almost_equal(td, 26. * units.degF, 0)


class TestDewpoint(object):
    def test_scalar(self):
        assert_almost_equal(dewpoint(6.112 * units.mbar), 0. * units.degC, 2)

    def test_weird_units(self):
        # This was revealed by a having odd dimensionless units and ending up using
        # numpy.ma math functions instead of numpy ones.
        assert_almost_equal(dewpoint(15825.6 * units('g * mbar / kg')),
                            13.8564 * units.degC, 4)


class TestMixingRatio(object):
    def test_scalar(self):
        p = 998. * units.mbar
        e = 73.75 * units.mbar
        assert_almost_equal(mixing_ratio(e, p), 0.04963, 2)


class TestVaporPressure(object):
    def test_scalar(self):
        assert_almost_equal(vapor_pressure(998. * units.mbar, 0.04963),
                            73.76 * units.mbar, 3)


class TestLCL(object):
    def test_basic(self):
        'Simple test of LCL calculation.'
        l = lcl(1000. * units.mbar, 30. * units.degC, 20. * units.degC)
        assert_almost_equal(l, 864.89 * units.mbar, 2)
