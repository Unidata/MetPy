from numpy.testing import *
import numpy as np
from metpy.readers import Data, ArrayData

class TestData(TestCase):
    @classmethod
    def setup_class(self):
        a1 = np.arange(5)
        a2 = np.arange(5, 10)
        d = dict(x=a1, y=a2)
        self.data = Data(d)

    def test_basic(self):
        assert_array_equal(self.data['x'], np.arange(5))
        assert_array_equal(self.data['y'], np.arange(5, 10))

    def test_attrs(self):
        assert self.data.units == dict()
        assert self.data.metadata == dict()

    def test_descr(self):
        self.data.set_descrip('x', 'xaxis')

        assert self.data.get_descrip('x') == 'xaxis'
        assert_array_equal(self.data['xaxis'], np.arange(5))
        assert_array_equal(self.data['x'], np.arange(5))

class TestArrayData(TestCase):
    @classmethod
    def setup_class(self):
        a1 = np.arange(5)
        a2 = np.arange(5, 10)
        d = np.empty((5,),dtype=[('x', np.int),(('yaxis', 'y'), np.int)])
        d['x'] = a1
        d['y'] = a2
        self.data = ArrayData(d)

    def test_basic(self):
        assert_array_equal(self.data['x'], np.arange(5))
        assert_array_equal(self.data['y'], np.arange(5, 10))

    def test_attrs(self):
        assert self.data.units == dict()
        assert self.data.metadata == dict()

    def test_init_descr(self):
        print self.data.get_descrip('y')
        assert self.data.get_descrip('y') == 'yaxis'
        assert_array_equal(self.data['yaxis'], np.arange(5, 10))

    def test_descr(self):
        self.data.set_descrip('x', 'xaxis')

        assert self.data.get_descrip('x') == 'xaxis'
        assert_array_equal(self.data['xaxis'], np.arange(5))
        assert_array_equal(self.data['x'], np.arange(5))

if __name__ == '__main__':
    run_module_suite()
