import gzip
from datetime import datetime

from nose.tools import eq_, nottest
from metpy.io.text import TextProductFile, WMOTextProduct
from metpy.cbook import get_test_data

get_test_data = nottest(get_test_data)


class TestWMOParsing(object):
    def setUp(self):
        gzfile = gzip.open(get_test_data('metar_20151105_0000.txt.gz', as_file_obj=False),
                           'rt', encoding='latin-1')
        self.prod_file = TextProductFile(gzfile)

    def test_wmo_file(self):
        eq_(len(list(self.prod_file)), 3051)

    def test_wmo_text_product(self):
        prod = WMOTextProduct(next(iter(self.prod_file)))
        eq_(prod.seq_num, 235)
        eq_(prod.data_designator, 'SALC31')
        eq_(prod.datetime, datetime(2015, 11, 5, 00, 00))
        eq_(prod.center, 'TLPL')
        eq_(prod.additional, '')
