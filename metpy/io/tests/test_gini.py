# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from datetime import datetime

from numpy.testing import assert_almost_equal
import pytest

from metpy.io.gini import GiniFile, GiniProjection
from metpy.cbook import get_test_data

log = logging.getLogger('metpy.io.gini')
log.setLevel(logging.ERROR)

# Reference contents of the named tuples from each file
make_pdb = GiniFile.prod_desc_fmt.make_tuple
make_pdb2 = GiniFile.prod_desc2_fmt.make_tuple
raw_gini_info = [('WEST-CONUS_4km_WV_20151208_2200.gini',
                  make_pdb(source=1, creating_entity='GOES-15', sector_id='West CONUS',
                           channel='WV (6.5/6.7 micron)', num_records=1280, record_len=1100,
                           datetime=datetime(2015, 12, 8, 22, 0, 19),
                           projection=GiniProjection.lambert_conformal, nx=1100, ny=1280,
                           la1=12.19, lo1=-133.4588),
                  make_pdb2(scanning_mode=[False, False, False], lat_in=25.0, resolution=4,
                            compression=0, version=1, pdb_size=512, nav_cal=0),
                  GiniFile.lc_ps_fmt.make_tuple(reserved=0, lov=-95.0, dx=4.0635, dy=4.0635,
                                                proj_center=0)),
                 ('AK-REGIONAL_8km_3.9_20160408_1445.gini',
                  make_pdb(source=1, creating_entity='GOES-15', sector_id='Alaska Regional',
                           channel='IR (3.9 micron)', num_records=408, record_len=576,
                           datetime=datetime(2016, 4, 8, 14, 45, 20),
                           projection=GiniProjection.polar_stereographic, nx=576, ny=408,
                           la1=42.0846, lo1=-175.641),
                  make_pdb2(scanning_mode=[False, False, False], lat_in=0.0, resolution=8,
                            compression=0, version=1, pdb_size=512, nav_cal=0),
                  GiniFile.lc_ps_fmt.make_tuple(reserved=0, lov=210.0, dx=7.9375, dy=7.9375,
                                                proj_center=0)),
                 ('HI-REGIONAL_4km_3.9_20160616_1715.gini',
                  make_pdb(source=1, creating_entity='GOES-15', sector_id='Hawaii Regional',
                           channel='IR (3.9 micron)', num_records=520, record_len=560,
                           datetime=datetime(2016, 6, 16, 17, 15, 18),
                           projection=GiniProjection.mercator, nx=560, ny=520,
                           la1=9.343, lo1=-167.315),
                  make_pdb2(scanning_mode=[False, False, False], lat_in=20.0, resolution=4,
                            compression=0, version=1, pdb_size=512, nav_cal=0),
                  GiniFile.mercator_fmt.make_tuple(resolution=0, la2=28.0922, lo2=-145.878,
                                                   di=0, dj=0))
                 ]


@pytest.mark.parametrize('filename,pdb,pdb2,proj_info', raw_gini_info,
                         ids=['LCC', 'Stereographic', 'Mercator'])
def test_raw_gini(filename, pdb, pdb2, proj_info):
    'Test of raw GINI parsing'
    f = GiniFile(get_test_data(filename))
    assert f.prod_desc == pdb
    assert f.prod_desc2 == pdb2
    assert f.proj_info == proj_info
    assert f.data.shape == (pdb.num_records, pdb.record_len)


def test_gini_bad_size():
    'Test reading a GINI file that reports a bad header size'
    f = GiniFile(get_test_data('NHEM-MULTICOMP_1km_IR_20151208_2100.gini'))
    pdb2 = f.prod_desc2
    assert pdb2.pdb_size == 512  # Catching bad size


# Reference information coming out of the dataset interface, coordinate calculations,
# inclusion of correct projection metadata, etc.
gini_dataset_info = [('WEST-CONUS_4km_WV_20151208_2200.gini',
                      (-4226066.37649, 239720.12351, -832700.70519, 4364515.79481), 'WV',
                      dict(grid_mapping_name='lambert_conformal_conic', standard_parallel=25.0,
                           earth_radius=6371200., latitude_of_projection_origin=25.0,
                           longitude_of_central_meridian=-95.0)),
                     ('AK-REGIONAL_8km_3.9_20160408_1445.gini',
                      (-2286001.13195, 2278061.36805, -4762503.5992, -1531941.0992), 'IR',
                      dict(grid_mapping_name='polar_stereographic', standard_parallel=60.0,
                           earth_radius=6371200., latitude_of_projection_origin=90.,
                           straight_vertical_longitude_from_pole=210.0)),
                     ('HI-REGIONAL_4km_3.9_20160616_1715.gini',
                      (0.0, 2236000.0, 980627.44738, 3056627.44738), 'IR',
                      dict(grid_mapping_name='mercator', standard_parallel=20.0,
                           earth_radius=6371200., latitude_of_projection_origin=9.343,
                           longitude_of_projection_origin=-167.315))
                     ]


@pytest.mark.parametrize('filename,bounds,data_var,proj_attrs', gini_dataset_info,
                         ids=['LCC', 'Stereographic', 'Mercator'])
def test_gini_dataset(filename, bounds, data_var, proj_attrs):
    'Test the dataset interface for GINI'
    f = GiniFile(get_test_data(filename))
    ds = f.to_dataset()

    # Check our calculated x and y arrays
    x0, x1, y0, y1 = bounds
    x = ds.variables['x']
    assert_almost_equal(x[0], x0, 4)
    assert_almost_equal(x[-1], x1, 4)

    y = ds.variables['y']
    assert_almost_equal(y[0], y0, 4)
    assert_almost_equal(y[-1], y1, 4)

    # Check the projection metadata
    proj_name = ds.variables[data_var].grid_mapping
    proj_var = ds.variables[proj_name]
    for attr, val in proj_attrs.items():
        assert getattr(proj_var, attr) == val, 'Values mismatch for ' + attr

    # Check the lon/lat corner
    assert_almost_equal(ds.variables['lon'][0, 0], f.prod_desc.lo1, 4)
    assert_almost_equal(ds.variables['lat'][0, 0], f.prod_desc.la1, 4)


def test_gini_mercator_upper_corner():
    'Test that the upper corner of the Mercator coordinates is correct'
    f = GiniFile(get_test_data('HI-REGIONAL_4km_3.9_20160616_1715.gini'))
    ds = f.to_dataset()
    lat = ds.variables['lat']
    lon = ds.variables['lon']

    # 2nd corner lat/lon are at the "upper right" corner of the pixel, so need to add one
    # more grid point
    assert_almost_equal(lon[-1, -1] + (lon[-1, -1] - lon[-1, -2]), f.proj_info.lo2, 4)
    assert_almost_equal(lat[-1, -1] + (lat[-1, -1] - lat[-2, -1]), f.proj_info.la2, 4)


def test_gini_str():
    'Test the str representation of GiniFile'
    f = GiniFile(get_test_data('WEST-CONUS_4km_WV_20151208_2200.gini'))
    truth = ('GiniFile: GOES-15 West CONUS WV (6.5/6.7 micron)\n'
             '\tTime: 2015-12-08 22:00:19\n\tSize: 1280x1100\n'
             '\tProjection: lambert_conformal\n'
             '\tLower Left Corner (Lon, Lat): (-133.4588, 12.19)\n\tResolution: 4km')
    assert str(f) == truth
