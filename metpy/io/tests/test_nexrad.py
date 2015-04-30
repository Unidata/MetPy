import os.path
from numpy.testing import TestCase
from metpy.io.nexrad import Level2File, Level3File

curdir, f = os.path.split(__file__)
datadir = os.path.join(curdir, '../../../examples/testdata')


class TestLevel3(TestCase):
    def test_basic(self):
        Level3File(os.path.join(datadir, 'nids/Level3_FFC_N0Q_20140407_1805.nids'))


class TestLevel2(TestCase):
    def test_basic(self):
        Level2File(os.path.join(datadir, 'KTLX20130520_190411_V06.gz'))
