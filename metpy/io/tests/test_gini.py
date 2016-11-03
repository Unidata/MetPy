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
    assert pdb.datetime == datetime(2015, 12, 8, 22, 0, 19)
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
    assert pdb.datetime == datetime(2016, 4, 8, 14, 45, 20)
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


def test_gini_mercator():
    'Test reading of GINI file with Mercator projection (from HI)'
    f = GiniFile(get_test_data('HI-REGIONAL_4km_3.9_20160616_1715.gini'))
    pdb = f.prod_desc
    assert pdb.source == 1
    assert pdb.creating_entity == 'GOES-15'
    assert pdb.sector_id == 'Hawaii Regional'
    assert pdb.channel == 'IR (3.9 micron)'
    assert pdb.num_records == 520
    assert pdb.record_len == 560
    assert pdb.datetime == datetime(2016, 6, 16, 17, 15, 18)

    assert pdb.projection == GiniProjection.mercator
    assert pdb.nx == 560
    assert pdb.ny == 520
    assert_almost_equal(pdb.la1, 9.343, 4)
    assert_almost_equal(pdb.lo1, -167.315, 4)

    proj = f.proj_info
    assert proj.resolution == 0
    assert_almost_equal(proj.la2, 28.0922, 4)
    assert_almost_equal(proj.lo2, -145.878, 4)
    assert proj.di == 0
    assert proj.dj == 0

    pdb2 = f.prod_desc2
    assert pdb2.scanning_mode == [False, False, False]
    assert_almost_equal(pdb2.lat_in, 20.0, 4)
    assert pdb2.resolution == 4
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
    x = ds.variables['x']
    assert_almost_equal(x[0], -4226066.37649, 4)
    assert_almost_equal(x[-1], 239720.12351, 4)

    y = ds.variables['y']
    assert_almost_equal(y[0], -832700.70519, 4)
    assert_almost_equal(y[-1], 4364515.79481, 4)

    assert 'WV' in ds.variables
    assert hasattr(ds.variables['WV'], 'grid_mapping')
    assert ds.variables['WV'].grid_mapping in ds.variables
    assert_almost_equal(ds.variables['lon'][0, 0], f.prod_desc.lo1, 4)
    assert_almost_equal(ds.variables['lat'][0, 0], f.prod_desc.la1, 4)


def test_gini_ak_regional_dataset():
    'Test the dataset interface for GINI of a AK REGIONAL file'
    f = GiniFile(get_test_data('AK-REGIONAL_8km_3.9_20160408_1445.gini'))
    ds = f.to_dataset()
    x = ds.variables['x']
    assert_almost_equal(x[0], -2286001.13195, 4)
    assert_almost_equal(x[-1], 2278061.36805, 4)

    y = ds.variables['y']
    assert_almost_equal(y[0], -4762503.5992, 4)
    assert_almost_equal(y[-1], -1531941.0992, 4)

    assert 'IR' in ds.variables
    assert hasattr(ds.variables['IR'], 'grid_mapping')
    assert ds.variables['IR'].grid_mapping in ds.variables
    assert_almost_equal(ds.variables['lon'][0, 0], f.prod_desc.lo1, 4)
    assert_almost_equal(ds.variables['lat'][0, 0], f.prod_desc.la1, 4)

    proj_var = ds.variables['Polar_Stereographic']
    assert_almost_equal(proj_var.straight_vertical_longitude_from_pole, 210.0)
    assert_almost_equal(proj_var.latitude_of_projection_origin, 90.0)
    assert_almost_equal(proj_var.standard_parallel, 60.0)


def test_gini_mercator_dataset():
    'Test the dataset interface for a GINI file with Mercator projection'
    f = GiniFile(get_test_data('HI-REGIONAL_4km_3.9_20160616_1715.gini'))
    ds = f.to_dataset()
    x = ds.variables['x']
    assert_almost_equal(x[0], 0.0, 4)
    assert_almost_equal(x[-1], 2236000.0, 4)

    y = ds.variables['y']
    assert_almost_equal(y[0], 980627.44738, 4)
    assert_almost_equal(y[-1], 3056627.44738, 4)

    assert 'IR' in ds.variables
    assert hasattr(ds.variables['IR'], 'grid_mapping')
    assert ds.variables['IR'].grid_mapping in ds.variables
    lat = ds.variables['lat']
    lon = ds.variables['lon']
    assert_almost_equal(lon[0, 0], f.prod_desc.lo1, 4)
    assert_almost_equal(lat[0, 0], f.prod_desc.la1, 4)
    # 2nd corner lat/lon are at the "upper right" corner of the pixel, so need to add one
    # more grid point
    assert_almost_equal(lon[-1, -1] + (lon[-1, -1] - lon[-1, -2]), f.proj_info.lo2, 4)
    assert_almost_equal(lat[-1, -1] + (lat[-1, -1] - lat[-2, -1]), f.proj_info.la2, 4)
    assert_almost_equal(ds.variables['Mercator'].longitude_of_projection_origin, -167.315)
    assert_almost_equal(ds.variables['Mercator'].latitude_of_projection_origin, 9.343)


def test_gini_str():
    'Test the str representation of GiniFile'
    f = GiniFile(get_test_data('WEST-CONUS_4km_WV_20151208_2200.gini'))
    truth = ('GiniFile: GOES-15 West CONUS WV (6.5/6.7 micron)\n'
             '\tTime: 2015-12-08 22:00:19\n\tSize: 1280x1100\n'
             '\tProjection: lambert_conformal\n'
             '\tLower Left Corner (Lon, Lat): (-133.4588, 12.19)\n\tResolution: 4km')
    assert str(f) == truth
