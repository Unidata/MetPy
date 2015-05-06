from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_almost_equal)
import numpy as np
from metpy.calc.thermo import *  # noqa
from metpy.constants import C2K


class TestPotentialTemperature(object):
    def test_basic(self):
        temp = np.array([278, 283, 291, 298])
        pres = np.array([900, 500, 300, 100])
        real_th = np.array([286.5, 345.0155, 410.5467, 575.5397])
        assert_array_almost_equal(potential_temperature(pres, temp), real_th, 3)

    def test_scalar(self):
        assert_almost_equal(potential_temperature(1000, 293), 293, 4)
        assert_almost_equal(potential_temperature(800, 293), 312.2987, 4)


class TestDryLapse(object):
    def test_array(self):
        levels = np.array([1000, 900, 864.89])
        temps = dry_lapse(levels, 303.15)
        assert_array_almost_equal(temps, np.array([303.15, 294.16, 290.83]), 2)

    def test_2_levels(self):
        temps = dry_lapse(np.array([1000., 500.]), 293)
        assert_array_almost_equal(temps, [293., 240.3341], 4)


class TestMoistLapse(object):
    def test_array(self):
        temp = moist_lapse(np.array([1000, 800, 600, 500, 400]), 293)
        true_temp = np.array([293, 284.64, 272.8, 264.4, 252.87])
        assert_array_almost_equal(temp, true_temp, 2)


class TestParcelProfile(object):
    def test_basic(self):
        levels = np.array([1000., 900., 800., 700., 600., 500., 400.])
        true_prof = np.array([303.15, 294.16, 288.02, 283.06, 277.04, 269.38, 258.93])

        prof = parcel_profile(levels, C2K(30.), C2K(20.))
        assert_array_almost_equal(prof, true_prof, 2)


class TestSatVaporPressure(object):
    def test_basic(self):
        temp = np.array([5, 10, 18, 25])
        real_es = np.array([8.72, 12.28, 20.64, 31.68])
        assert_array_almost_equal(saturation_vapor_pressure(temp), real_es, 2)

    def test_scalar(self):
        es = saturation_vapor_pressure(0)
        assert_almost_equal(es, 6.112, 3)


class TestDewpointRH(object):
    def test_basic(self):
        temp = np.array([30, 25, 10, 20, 25])
        rh = np.array([30, 45, 55, 80, 85]) / 100.

        real_td = np.array([11, 12, 1, 16, 22])
        assert_array_almost_equal(real_td, dewpoint_rh(temp, rh), 0)

    def test_scalar(self):
        td = dewpoint_rh(10.6, .37) * 1.8 + 32.
        assert_almost_equal(td, 26, 0)


class TestDewpoint(object):
    def test_scalar(self):
        assert_almost_equal(dewpoint(6.112), 0., 2)


class TestMixingRatio(object):
    def test_scalar(self):
        p = 998.
        e = 73.75
        assert_almost_equal(mixing_ratio(e, p), 0.04963, 2)


class TestVaporPressure(object):
    def test_scalar(self):
        assert_almost_equal(vapor_pressure(998, 0.04963), 73.76, 3)


class TestLCL(object):
    def test_basic(self):
        'Simple test of LCL calculation.'
        l = lcl(1000., C2K(30.), C2K(20.))
        assert_almost_equal(l, 864.89, 2)
