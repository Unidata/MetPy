# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from datetime import datetime
from numpy.testing import assert_almost_equal

from metpy.io.gini import GiniFile, GiniProjection
from metpy.cbook import get_test_data

log = logging.getLogger('metpy.io.gini')
log.setLevel(logging.ERROR)


def test_gini_basic():
    'Basic test of GINI reading'
    f = GiniFile(get_test_data('WEST-CONUS_4km_WV_20151208_2200.gini'))
    pdb = f.prod_desc
    assert pdb.source == 1
    assert pdb.creating_entity == 'GOES-15'
    assert pdb.sector_id == 'West CONUS'
    assert pdb.channel == 'WV (6.5/6.7 micron)'
    assert pdb.num_records == 1280
    assert pdb.record_len == 1100
    assert pdb.datetime, datetime(2015, 12, 8, 22, 0, 19 == 0)
    assert pdb.projection == GiniProjection.lambert_conformal
    assert pdb.nx == 1100
    assert pdb.ny == 1280
    assert_almost_equal(pdb.la1, 12.19, 4)
    assert_almost_equal(pdb.lo1, -133.4588, 4)

    proj = f.proj_info
    assert proj.reserved == 0
    assert_almost_equal(proj.lov, -95.0, 4)
    assert_almost_equal(proj.dx, 4.0635, 4)
    assert_almost_equal(proj.dy, 4.0635, 4)
    assert proj.proj_center == 0

    pdb2 = f.prod_desc2
    assert pdb2.scanning_mode == [False, False, False]
    assert_almost_equal(pdb2.lat_in, 25.0, 4)
    assert pdb2.resolution == 4
    assert pdb2.compression == 0
    assert pdb2.version == 1
    assert pdb2.pdb_size == 512

    assert f.data.shape, (pdb.num_records == pdb.record_len)


def test_gini_ak_regional():
    'Test reading of AK Regional Gini file'
    f = GiniFile(get_test_data('AK-REGIONAL_8km_3.9_20160408_1445.gini'))
    pdb = f.prod_desc
    assert pdb.source == 1
    assert pdb.creating_entity == 'GOES-15'
    assert pdb.sector_id == 'Alaska Regional'
    assert pdb.channel == 'IR (3.9 micron)'
    assert pdb.num_records == 408
    assert pdb.record_len == 576
    assert pdb.datetime, datetime(2016, 4, 8, 14, 45, 20 == 0)
    assert pdb.projection == GiniProjection.polar_stereographic
    assert pdb.nx == 576
    assert pdb.ny == 408
    assert_almost_equal(pdb.la1, 42.0846, 4)
    assert_almost_equal(pdb.lo1, -175.641, 4)

    proj = f.proj_info
    assert proj.reserved == 0
    assert_almost_equal(proj.lov, 210.0, 1)
    assert_almost_equal(proj.dx, 7.9375, 4)
    assert_almost_equal(proj.dy, 7.9375, 4)
    assert proj.proj_center == 0

    pdb2 = f.prod_desc2
    assert pdb2.scanning_mode == [False, False, False]
    assert_almost_equal(pdb2.lat_in, 0.0, 4)
    assert pdb2.resolution == 8
    assert pdb2.compression == 0
    assert pdb2.version == 1
    assert pdb2.pdb_size == 512
    assert pdb2.nav_cal == 0

    assert f.data.shape, (pdb.num_records == pdb.record_len)


def test_gini_bad_size():
    'Test reading a GINI file that reports a bad header size'
    f = GiniFile(get_test_data('NHEM-MULTICOMP_1km_IR_20151208_2100.gini'))
    pdb2 = f.prod_desc2
    assert pdb2.pdb_size == 512  # Catching bad size


def test_gini_dataset():
    'Test the dataset interface for GINI'
    f = GiniFile(get_test_data('WEST-CONUS_4km_WV_20151208_2200.gini'))
    ds = f.to_dataset()
    assert 'x' in ds.variables
    assert 'y' in ds.variables
    assert 'WV' in ds.variables
    assert hasattr(ds.variables['WV'], 'grid_mapping')
    assert ds.variables['WV'].grid_mapping in ds.variables
    assert_almost_equal(ds.variables['lon'][0, 0], f.prod_desc.lo1, 4)
    assert_almost_equal(ds.variables['lat'][0, 0], f.prod_desc.la1, 4)


def test_gini_ak_regional_dataset():
    'Test the dataset interface for GINI of a AK REGIONAL file'
    f = GiniFile(get_test_data('AK-REGIONAL_8km_3.9_20160408_1445.gini'))
    ds = f.to_dataset()
    assert 'x' in ds.variables
    assert 'y' in ds.variables
    assert 'IR' in ds.variables
    assert hasattr(ds.variables['IR'], 'grid_mapping')
    assert ds.variables['IR'].grid_mapping in ds.variables
    assert_almost_equal(ds.variables['lon'][0, 0], f.prod_desc.lo1, 4)
    assert_almost_equal(ds.variables['lat'][0, 0], f.prod_desc.la1, 4)


def test_gini_str():
    'Test the str representation of GiniFile'
    f = GiniFile(get_test_data('WEST-CONUS_4km_WV_20151208_2200.gini'))
    truth = ('GiniFile: GOES-15 West CONUS WV (6.5/6.7 micron)\n'
             '\tTime: 2015-12-08 22:00:19\n\tSize: 1280x1100\n'
             '\tProjection: lambert_conformal\n'
             '\tLower Left Corner (Lon, Lat): (-133.4588, 12.19)\n\tResolution: 4km')
    assert str(f) == truth
