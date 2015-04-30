import glob
import os.path
from numpy.testing import TestCase
from metpy.io.nexrad import Level2File, Level3File

curdir, f = os.path.split(__file__)
datadir = os.path.join(curdir, '../../../examples/testdata')


def test_generator():
    for fname in glob.glob(os.path.join(datadir, 'nids', 'KOUN*')):
        yield read_level3_file, fname


def read_level3_file(fname):
    Level3File(fname)


class TestLevel3(TestCase):
    def test_basic(self):
        Level3File(os.path.join(datadir, 'nids/Level3_FFC_N0Q_20140407_1805.nids'))


class TestLevel2(TestCase):
    def test_basic(self):
        Level2File(os.path.join(datadir, 'KTLX20130520_201643_V06.gz'))
