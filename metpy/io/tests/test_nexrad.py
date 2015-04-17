import os.path
from numpy.testing import *
from metpy.io.nexrad import *

datadir, f = os.path.split(__file__)
class TestLevel3(TestCase):
    def test_basic(self):
        f = Level3File(os.path.join(datadir, 'Level3_FFC_N0Q_20140407_1805.nids'))
