from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_almost_equal, assert_array_equal)
import numpy as np
from metpy.calc.basic import *  # noqa
from metpy.constants import F2C


class TestWindComps(TestCase):
    def test_basic(self):
        'Test the basic calculation.'
        speed = np.array([4, 4, 4, 4, 25, 25, 25, 25, 10.])
        dirs = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
        s2 = np.sqrt(2)

        u, v = get_wind_components(speed, dirs)

        true_u = np.array([0, -4 / s2, -4, -4 / s2, 0, 25 / s2, 25, 25 / s2, 0])
        true_v = np.array([-4, -4 / s2, 0, 4 / s2, 25, 25 / s2, 0, -25 / s2, -10])

        assert_array_almost_equal(true_u, u, 4)
        assert_array_almost_equal(true_v, v, 4)

    def test_scalar(self):
        comps = np.array(get_wind_components(8, 150))
        assert_array_almost_equal(comps, np.array([-4, 6.9282]), 3)


class TestWindChill(TestCase):
    def test_scalar(self):
        wc = windchill(-5, 35)
        assert_almost_equal(wc, -18.9357, 0)

    def test_basic(self):
        'Test the basic wind chill calculation.'
        temp = (np.array([40, -10, -45, 20]) - 32.) / 1.8
        speed = np.array([5, 55, 25, 15]) * .44704

        wc = windchill(temp, speed)
        values = (np.array([36, -46, -84, 6]) - 32.) / 1.8
        assert_array_almost_equal(wc, values, 0)

    def test_invalid(self):
        'Test for values that should be masked.'
        temp = np.array([10, 51, 49, 60, 80, 81])
        speed = np.array([4, 4, 3, 1, 10, 39])

        wc = windchill(temp, speed)
        mask = np.array([False, True, True, True, True, True])
        assert_array_equal(wc.mask, mask)

    def test_undefined_flag(self):
        'Tests whether masking values can be disabled.'
        temp = np.ma.array([49, 50, 49, 60, 80, 81])
        speed = np.ma.array([4, 4, 3, 1, 10, 39])

        wc = windchill(temp, speed, mask_undefined=False)
        mask = np.array([False] * 6)
        assert_array_equal(wc.mask, mask)


class TestHeatIndex(TestCase):
    def test_basic(self):
        'Test the basic heat index calculation.'
        temp = F2C(np.array([80, 88, 92, 110]))
        rh = np.array([40, 100, 70, 40])

        hi = heat_index(temp, rh)
        values = F2C(np.array([80, 121, 112, 136]))
        assert_array_almost_equal(hi, values, 0)

    def test_scalar(self):
        hi = heat_index(F2C(96), 65)
        assert_almost_equal(hi, F2C(121), 0)

    def test_invalid(self):
        'Test for values that should be masked.'
        temp = F2C(np.array([80, 88, 92, 79, 30, 81]))
        rh = np.array([40, 39, 2, 70, 50, 39])

        hi = F2C(heat_index(temp, rh))
        mask = np.array([False, True, True, True, True, True])
        assert_array_equal(hi.mask, mask)

    def test_undefined_flag(self):
        'Tests whether masking values can be disabled.'
        temp = np.ma.array([80, 88, 92, 79, 30, 81])
        rh = np.ma.array([40, 39, 2, 70, 50, 39])

        hi = heat_index(temp, rh, mask_undefined=False)
        mask = np.array([False] * 6)
        assert_array_equal(hi.mask, mask)


# class TestIrrad(TestCase):
#    def test_basic(self):
#        'Test the basic solar irradiance calculation.'
#        from datetime import date

#        d = date(2008, 9, 28)
#        lat = 35.25
#        hours = np.linspace(6,18,10)

#        s = solar_irradiance(lat, d, hours)
#        values = np.array([0., 344.1, 682.6, 933.9, 1067.6, 1067.6, 933.9,
#            682.6, 344.1, 0.])
#        assert_array_almost_equal(s, values, 1)

#    def test_scalar(self):
#        from datetime import date
#        d = date(2008, 9, 28)
#        lat = 35.25
#        hour = 9.5
#        s = solar_irradiance(lat, d, hour)
#        assert_almost_equal(s, 852.1, 1)

#    def test_invalid(self):
#        'Test for values that should be masked.'
#        from datetime import date
#        d = date(2008, 9, 28)
#        lat = 35.25
#        hours = np.linspace(0,22,12)
#        s = solar_irradiance(lat, d, hours)

#        mask = np.array([ True,  True,  True,  True, False, False, False,
#            False, False, True,  True,  True])
#        assert_array_equal(s.mask, mask)


if __name__ == '__main__':
    run_module_suite()
