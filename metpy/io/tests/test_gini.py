# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from datetime import datetime

from nose.tools import assert_almost_equal, eq_, nottest
from metpy.io.gini import GiniFile, GiniProjection
from metpy.cbook import get_test_data

log = logging.getLogger('metpy.io.gini')
log.setLevel(logging.ERROR)

get_test_data = nottest(get_test_data)


class TestGini(object):
    'Tests for GINI file reader'
    @staticmethod
    def basic_test():
        'Basic test of GINI reading'
        f = GiniFile(get_test_data('WEST-CONUS_4km_WV_20151208_2200.gini'))
        pdb = f.prod_desc
        eq_(pdb.source, 1)
        eq_(pdb.creating_entity, 'GOES-15')
        eq_(pdb.sector_id, 'West CONUS')
        eq_(pdb.channel, 'WV (6.5/6.7 micron)')
        eq_(pdb.num_records, 1280)
        eq_(pdb.record_len, 1100)
        eq_(pdb.datetime, datetime(2015, 12, 8, 22, 0, 19, 0))
        eq_(pdb.projection, GiniProjection.lambert_conformal)
        eq_(pdb.nx, 1100)
        eq_(pdb.ny, 1280)
        assert_almost_equal(pdb.la1, 12.19, 4)
        assert_almost_equal(pdb.lo1, -133.4588, 4)

        proj = f.proj_info
        eq_(proj.reserved, 0)
        assert_almost_equal(proj.lov, -95.0, 4)
        assert_almost_equal(proj.dx, 4.0635, 4)
        assert_almost_equal(proj.dy, 4.0635, 4)
        eq_(proj.proj_center, 0)

        pdb2 = f.prod_desc2
        eq_(pdb2.scanning_mode, [False, False, False])
        assert_almost_equal(pdb2.lat_in, 25.0, 4)
        eq_(pdb2.resolution, 4)
        eq_(pdb2.compression, 0)
        eq_(pdb2.version, 1)
        eq_(pdb2.pdb_size, 512)

        eq_(f.data.shape, (pdb.num_records, pdb.record_len))

    @staticmethod
    def test_bad_size():
        'Test reading a GINI file that reports a bad header size'
        f = GiniFile(get_test_data('NHEM-MULTICOMP_1km_IR_20151208_2100.gini'))
        pdb2 = f.prod_desc2
        eq_(pdb2.pdb_size, 512)  # Catching bad size

    @staticmethod
    def test_dataset():
        'Test the dataset interface for GINI'
        f = GiniFile(get_test_data('WEST-CONUS_4km_WV_20151208_2200.gini'))
        ds = f.to_dataset()
        assert 'x' in ds.variables
        assert 'y' in ds.variables
        assert 'WV' in ds.variables
        assert hasattr(ds.variables['WV'], 'grid_mapping')
        assert ds.variables['WV'].grid_mapping in ds.variables

    @staticmethod
    def test_str():
        'Test the str representation of GiniFile'
        f = GiniFile(get_test_data('WEST-CONUS_4km_WV_20151208_2200.gini'))
        truth = ('GiniFile: GOES-15 West CONUS WV (6.5/6.7 micron)\n'
                 '\tTime: 2015-12-08 22:00:19\n\tSize: 1280x1100\n'
                 '\tProjection: lambert_conformal\n'
                 '\tLower Left Corner (Lon, Lat): (-133.4588, 12.19)\n\tResolution: 4km')
        assert str(f) == truth
