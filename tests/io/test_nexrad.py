# Copyright (c) 2015,2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `nexrad` module."""

from datetime import datetime
from io import BytesIO
import logging

import numpy as np
import pytest

from metpy.cbook import get_test_data, POOCH
from metpy.io import is_precip_mode, Level2File, Level3File

# Turn off the warnings for tests
logging.getLogger('metpy.io.nexrad').setLevel(logging.CRITICAL)

#
# NEXRAD Level 2 Tests
#

# 1999 file tests old message 1
# KFTG tests bzip compression and newer format for a part of message 31
# KTLX 2015 has missing segments for message 18, which was causing exception
level2_files = [('KTLX20130520_201643_V06.gz', datetime(2013, 5, 20, 20, 16, 46), 17, 4, 6),
                ('KTLX19990503_235621.gz', datetime(1999, 5, 3, 23, 56, 21), 16, 1, 3),
                ('Level2_KFTG_20150430_1419.ar2v', datetime(2015, 4, 30, 14, 19, 11),
                 12, 4, 6),
                ('KTLX20150530_000802_V06.bz2', datetime(2015, 5, 30, 0, 8, 3), 14, 4, 6),
                ('KICX_20170712_1458', datetime(2017, 7, 12, 14, 58, 5), 14, 4, 6),
                ('TDAL20191021021543V08.raw.gz', datetime(2019, 10, 21, 2, 15, 43), 10, 1, 3),
                ('Level2_FOP1_20191223_003655.ar2v', datetime(2019, 12, 23, 0, 36, 55, 649000),
                 16, 5, 7)]


# ids here fixes how things are presented in pycharm
@pytest.mark.parametrize('fname, voltime, num_sweeps, mom_first, mom_last', level2_files,
                         ids=[i[0].replace('.', '_') for i in level2_files])
def test_level2(fname, voltime, num_sweeps, mom_first, mom_last):
    """Test reading NEXRAD level 2 files from the filename."""
    f = Level2File(get_test_data(fname, as_file_obj=False))
    assert f.dt == voltime
    assert len(f.sweeps) == num_sweeps
    assert len(f.sweeps[0][0][-1]) == mom_first
    assert len(f.sweeps[-1][0][-1]) == mom_last


def test_level2_fobj():
    """Test reading NEXRAD level2 data from a file object."""
    Level2File(get_test_data('Level2_KFTG_20150430_1419.ar2v'))


def test_doubled_file():
    """Test for #489 where doubled-up files didn't parse at all."""
    data = get_test_data('Level2_KFTG_20150430_1419.ar2v').read()
    fobj = BytesIO(data + data)
    f = Level2File(fobj)
    assert len(f.sweeps) == 12


@pytest.mark.parametrize('fname, has_v2', [('KTLX20130520_201643_V06.gz', False),
                                           ('Level2_KFTG_20150430_1419.ar2v', True),
                                           ('TDAL20191021021543V08.raw.gz', False)])
def test_conditional_radconst(fname, has_v2):
    """Test whether we're using the right volume constants."""
    f = Level2File(get_test_data(fname, as_file_obj=False))
    assert hasattr(f.sweeps[0][0][3], 'calib_dbz0_v') == has_v2


#
# NIDS/Level 3 Tests
#
nexrad_nids_files = [get_test_data(fname, as_file_obj=False)
                     for fname in POOCH.registry if fname.startswith('nids/K')]


@pytest.mark.parametrize('fname', nexrad_nids_files)
def test_level3_files(fname):
    """Test opening a NEXRAD NIDS file."""
    f = Level3File(fname)

    # If we have some raster data in the symbology block, feed it into the mapper to make
    # sure it's working properly (Checks for #253)
    if hasattr(f, 'sym_block'):
        block = f.sym_block[0][0]
        if 'data' in block:
            f.map_data(block['data'])

    assert f.filename == fname


tdwr_nids_files = [get_test_data(fname, as_file_obj=False)
                   for fname in POOCH.registry if fname.startswith('nids/Level3_MCI_')]


@pytest.mark.parametrize('fname', tdwr_nids_files)
def test_tdwr_nids(fname):
    """Test opening a TDWR NIDS file."""
    Level3File(fname)


def test_basic():
    """Test reading one specific NEXRAD NIDS file based on the filename."""
    f = Level3File(get_test_data('nids/Level3_FFC_N0Q_20140407_1805.nids', as_file_obj=False))
    assert f.metadata['prod_time'].replace(second=0) == datetime(2014, 4, 7, 18, 5)
    assert f.metadata['vol_time'].replace(second=0) == datetime(2014, 4, 7, 18, 5)
    assert f.metadata['msg_time'].replace(second=0) == datetime(2014, 4, 7, 18, 6)
    assert f.filename == get_test_data('nids/Level3_FFC_N0Q_20140407_1805.nids',
                                       as_file_obj=False)

    # At this point, really just want to make sure that __str__ is able to run and produce
    # something not empty, the format is still up for grabs.
    assert str(f)


def test_bad_length(caplog):
    """Test reading a product with too many bytes produces a log message."""
    fname = get_test_data('nids/KOUN_SDUS84_DAATLX_201305202016', as_file_obj=False)
    with open(fname, 'rb') as inf:
        data = inf.read()
        fobj = BytesIO(data + data)

    with caplog.at_level(logging.WARNING, 'metpy.io.nexrad'):
        Level3File(fobj)
        assert len(caplog.records) == 1
        assert 'This product may not parse correctly' in caplog.records[0].message


def test_tdwr():
    """Test reading a specific TDWR file."""
    f = Level3File(get_test_data('nids/Level3_SLC_TV0_20160516_2359.nids'))
    assert f.prod_desc.prod_code == 182


def test_dhr():
    """Test reading a time field for DHR product."""
    f = Level3File(get_test_data('nids/KOUN_SDUS54_DHRTLX_201305202016'))
    assert f.metadata['avg_time'] == datetime(2013, 5, 20, 20, 18)


def test_nwstg():
    """Test reading a nids file pulled from the NWSTG."""
    Level3File(get_test_data('nids/sn.last', as_file_obj=False))


def test_fobj():
    """Test reading a specific NEXRAD NIDS files from a file object."""
    Level3File(get_test_data('nids/Level3_FFC_N0Q_20140407_1805.nids'))


def test21_precip():
    """Test checking whether VCP 21 is precipitation mode."""
    assert is_precip_mode(21), 'VCP 21 is precip'


def test11_precip():
    """Test checking whether VCP 11 is precipitation mode."""
    assert is_precip_mode(11), 'VCP 11 is precip'


def test31_clear_air():
    """Test checking whether VCP 31 is clear air mode."""
    assert not is_precip_mode(31), 'VCP 31 is not precip'


def test_msg15():
    """Check proper decoding of message type 15."""
    f = Level2File(get_test_data('KTLX20130520_201643_V06.gz', as_file_obj=False))
    data = f.clutter_filter_map['data']
    assert isinstance(data[0][0], list)
    assert f.clutter_filter_map['datetime'] == datetime(2013, 5, 19, 0, 0, 0, 315000)


def test_tracks():
    """Check that tracks are properly decoded."""
    f = Level3File(get_test_data('nids/KOUN_SDUS34_NSTTLX_201305202016'))
    for data in f.sym_block[0]:
        if 'track' in data:
            x, y = np.array(data['track']).T
            assert len(x)
            assert len(y)


def test_vector_packet():
    """Check that vector packets are properly decoded."""
    f = Level3File(get_test_data('nids/KOUN_SDUS64_NHITLX_201305202016'))
    for page in f.graph_pages:
        for item in page:
            if 'vectors' in item:
                x1, x2, y1, y2 = np.array(item['vectors']).T
                assert len(x1)
                assert len(x2)
                assert len(y1)
                assert len(y2)
