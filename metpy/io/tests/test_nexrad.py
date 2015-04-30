import glob
import os.path
from numpy.testing import TestCase
from metpy.io.nexrad import Level2File, Level3File, is_precip_mode

curdir, f = os.path.split(__file__)
datadir = os.path.join(curdir, '../../../examples/testdata')


def test_level3_generator():
    for fname in glob.glob(os.path.join(datadir, 'nids', 'K???_*')):
        yield read_level3_file, fname


def read_level3_file(fname):
    Level3File(fname)

# 1999 file tests old message 1
# KFTG tests bzip compression and newer format for a part of message 31
level2_files = ['KTLX20130520_201643_V06.gz', 'KTLX19990503_235621.gz',
                'Level2_KFTG_20150430_1419.ar2v']


def test_level2_generator():
    for fname in level2_files:
        yield read_level2_file, os.path.join(datadir, fname)


def read_level2_file(fname):
    Level2File(fname)


class TestLevel3(TestCase):
    def test_basic(self):
        Level3File(os.path.join(datadir, 'nids/Level3_FFC_N0Q_20140407_1805.nids'))


class TestPrecipMode(TestCase):
    def test21(self):
        assert is_precip_mode(21), 'VCP 21 is precip'

    def test11(self):
        assert is_precip_mode(11), 'VCP 11 is precip'

    def test31(self):
        assert not is_precip_mode(31), 'VCP 31 is not precip'
