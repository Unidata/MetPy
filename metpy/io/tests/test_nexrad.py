# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import logging
import os.path

import pytest

import numpy as np

from metpy.io.nexrad import Level2File, Level3File, is_precip_mode
from metpy.cbook import get_test_data

# Turn off the warnings for tests
logging.getLogger("metpy.io.nexrad").setLevel(logging.CRITICAL)

#
# NEXRAD Level 2 Tests
#

# 1999 file tests old message 1
# KFTG tests bzip compression and newer format for a part of message 31
# KTLX 2015 has missing segments for message 18, which was causing exception
level2_files = ['KTLX20130520_201643_V06.gz', 'KTLX19990503_235621.gz',
                'Level2_KFTG_20150430_1419.ar2v', 'KTLX20150530_000802_V06.bz2']


@pytest.mark.parametrize('fname', level2_files)
def test_level2(fname):
    'Test reading NEXRAD level 2 files from the filename'
    Level2File(get_test_data(fname, as_file_obj=False))


def test_level2_fobj():
    'Test reading NEXRAD level2 data from a file object'
    Level2File(get_test_data('Level2_KFTG_20150430_1419.ar2v'))

#
# NIDS/Level 3 Tests
#

nexrad_nids_files = glob.glob(os.path.join(get_test_data('nids', as_file_obj=False), 'K???_*'))


@pytest.mark.parametrize('fname', nexrad_nids_files)
def test_level3_files(fname):
    Level3File(fname)


class TestLevel3(object):
    def test_basic(self):
        Level3File(get_test_data('nids/Level3_FFC_N0Q_20140407_1805.nids', as_file_obj=False))

    def test_nwstg(self):
        Level3File(get_test_data('nids/sn.last', as_file_obj=False))

    def test_fobj(self):
        Level3File(get_test_data('nids/Level3_FFC_N0Q_20140407_1805.nids'))


class TestPrecipMode(object):
    def test21(self):
        assert is_precip_mode(21), 'VCP 21 is precip'

    def test11(self):
        assert is_precip_mode(11), 'VCP 11 is precip'

    def test31(self):
        assert not is_precip_mode(31), 'VCP 31 is not precip'


class TestZipCalls(object):
    def test_msg15(self):
        f = Level2File(get_test_data('KTLX20130520_201643_V06.gz', as_file_obj=False))
        data = f.clutter_filter_map['data']
        assert isinstance(data[0][0], list)

    def test_tracks(self):
        f = Level3File(get_test_data('nids/KOUN_SDUS34_NSTTLX_201305202016'))
        for data in f.sym_block[0]:
            if 'track' in data:
                x, y = np.array(data['track']).T
                assert len(x)
                assert len(y)

    def test_vector_packet(self):
        f = Level3File(get_test_data('nids/KOUN_SDUS64_NHITLX_201305202016'))
        for page in f.graph_pages:
            for item in page:
                if 'vectors' in item:
                    x1, x2, y1, y2 = np.array(item['vectors']).T
                    assert len(x1)
                    assert len(x2)
                    assert len(y1)
                    assert len(y2)
