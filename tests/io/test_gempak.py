# Copyright (c) 2015,2016,2017,2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `gempak` module."""

from datetime import datetime
import logging

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest

from metpy.cbook import get_test_data
from metpy.io.gempak import GempakGrid, GempakSounding, GempakSurface

logging.getLogger('metpy.io.gempak').setLevel(logging.ERROR)


@pytest.mark.parametrize('grid_name', ['none', 'diff', 'dec', 'grib'])
def test_grid_loading(grid_name):
    """Test reading grids with different packing."""
    grid = GempakGrid(get_test_data(f'gem_packing_{grid_name}.grd')).gdxarray(
        parameter='TMPK',
        level=850
    )
    gio = grid[0].values.squeeze()

    gempak = np.load(get_test_data(f'gem_packing_{grid_name}.npz'))['values']

    assert_allclose(gio, gempak, rtol=1e-6, atol=0)


def test_merged_sounding():
    """Test loading a merged sounding.

    These are most often from models.
    """
    gso = GempakSounding(get_test_data('gem_model_mrg.snd')).snxarray(
        station_id='KMSN'
    )
    gpres = gso[0].pressure.values
    gtemp = gso[0].tmpc.values.squeeze()
    gdwpt = gso[0].dwpc.values.squeeze()
    gdrct = gso[0].drct.values.squeeze()
    gsped = gso[0].sped.values.squeeze()
    ghght = gso[0].hght.values.squeeze()
    gomeg = gso[0].omeg.values.squeeze()
    gcwtr = gso[0].cwtr.values.squeeze()
    gdtcp = gso[0].dtcp.values.squeeze()
    gdtgp = gso[0].dtgp.values.squeeze()
    gdtsw = gso[0].dtsw.values.squeeze()
    gdtlw = gso[0].dtlw.values.squeeze()
    gcfrl = gso[0].cfrl.values.squeeze()
    gtkel = gso[0].tkel.values.squeeze()
    gimxr = gso[0].imxr.values.squeeze()
    gdtar = gso[0].dtar.values.squeeze()

    gempak = pd.read_csv(get_test_data('gem_model_mrg.csv'), na_values=-9999)
    dpres = gempak.PRES.values
    dtemp = gempak.TMPC.values
    ddwpt = gempak.DWPC.values
    ddrct = gempak.DRCT.values
    dsped = gempak.SPED.values
    dhght = gempak.HGHT.values
    domeg = gempak.OMEG.values
    dcwtr = gempak.CWTR.values
    ddtcp = gempak.DTCP.values
    ddtgp = gempak.DTGP.values
    ddtsw = gempak.DTSW.values
    ddtlw = gempak.DTLW.values
    dcfrl = gempak.CFRL.values
    dtkel = gempak.TKEL.values
    dimxr = gempak.IMXR.values
    ddtar = gempak.DTAR.values

    np.testing.assert_allclose(gpres, dpres, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gtemp, dtemp, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdwpt, ddwpt, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdrct, ddrct, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gsped, dsped, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(ghght, dhght, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gomeg, domeg, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gcwtr, dcwtr, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtcp, ddtcp, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtgp, ddtgp, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtsw, ddtsw, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtlw, ddtlw, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gcfrl, dcfrl, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gtkel, dtkel, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gimxr, dimxr, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtar, ddtar, rtol=1e-10, atol=1e-2)


@pytest.mark.parametrize('gem,gio,station', [
    ('gem_sigw_hght_unmrg.csv', 'gem_sigw_hght_unmrg.snd', 'TOP'),
    ('gem_sigw_pres_unmrg.csv', 'gem_sigw_pres_unmrg.snd', 'WAML')
])
def test_unmerged_sounding(gem, gio, station):
    """Test loading an unmerged sounding.

    PPBB and PPDD groups will be in height coordinates.
    """
    gso = GempakSounding(get_test_data(f'{gio}')).snxarray(
        station_id=f'{station}'
    )
    gpres = gso[0].pressure.values
    gtemp = gso[0].temp.values.squeeze()
    gdwpt = gso[0].dwpt.values.squeeze()
    gdrct = gso[0].drct.values.squeeze()
    gsped = gso[0].sped.values.squeeze()
    ghght = gso[0].hght.values.squeeze()

    gempak = pd.read_csv(get_test_data(f'{gem}'), na_values=-9999)
    dpres = gempak.PRES.values
    dtemp = gempak.TEMP.values
    ddwpt = gempak.DWPT.values
    ddrct = gempak.DRCT.values
    dsped = gempak.SPED.values
    dhght = gempak.HGHT.values

    assert_allclose(gpres, dpres, rtol=1e-10, atol=1e-2)
    assert_allclose(gtemp, dtemp, rtol=1e-10, atol=1e-2)
    assert_allclose(gdwpt, ddwpt, rtol=1e-10, atol=1e-2)
    assert_allclose(gdrct, ddrct, rtol=1e-10, atol=1e-2)
    assert_allclose(gsped, dsped, rtol=1e-10, atol=1e-2)
    assert_allclose(ghght, dhght, rtol=1e-10, atol=1e-1)


def test_unmerged_sigw_pressure_sounding():
    """Test loading an unmerged sounding.

    PPBB and PPDD groups will be in pressure coordinates and there will
    be MAN level below the surface.
    """
    gso = GempakSounding(get_test_data('gem_sigw_pres_unmrg_man_bgl.snd')).snxarray()
    gpres = gso[0].pressure.values
    gtemp = gso[0].temp.values.squeeze()
    gdwpt = gso[0].dwpt.values.squeeze()
    gdrct = gso[0].drct.values.squeeze()
    gsped = gso[0].sped.values.squeeze()
    ghght = gso[0].hght.values.squeeze()

    gempak = pd.read_csv(get_test_data('gem_sigw_pres_unmrg_man_bgl.csv'), na_values=-9999)
    dpres = gempak.PRES.values
    dtemp = gempak.TEMP.values
    ddwpt = gempak.DWPT.values
    ddrct = gempak.DRCT.values
    dsped = gempak.SPED.values
    dhght = gempak.HGHT.values

    assert_allclose(gpres, dpres, rtol=1e-10, atol=1e-2)
    assert_allclose(gtemp, dtemp, rtol=1e-10, atol=1e-2)
    assert_allclose(gdwpt, ddwpt, rtol=1e-10, atol=1e-2)
    assert_allclose(gdrct, ddrct, rtol=1e-10, atol=1e-2)
    assert_allclose(gsped, dsped, rtol=1e-10, atol=1e-2)
    assert_allclose(ghght, dhght, rtol=1e-10, atol=1e-1)


def test_standard_surface():
    """Test to read a standard surface file."""
    def dtparse(string):
        return datetime.strptime(string, '%y%m%d/%H%M')

    skip = ['text', 'spcl']

    gsf = GempakSurface(get_test_data('gem_std.sfc'))
    gstns = gsf.sfjson()

    gempak = pd.read_csv(get_test_data('gem_std.csv'),
                         index_col=['STN', 'YYMMDD/HHMM'],
                         parse_dates=['YYMMDD/HHMM'],
                         date_parser=dtparse)

    for stn in gstns:
        idx_key = (stn['properties']['station_id'],
                   stn['properties']['date_time'])
        gemsfc = gempak.loc[idx_key, :]

        for param, val in stn['values'].items():
            if param not in skip:
                assert val == pytest.approx(gemsfc[param.upper()])


def test_ship_surface():
    """Test to read a ship surface file."""
    def dtparse(string):
        return datetime.strptime(string, '%y%m%d/%H%M')

    skip = ['text', 'spcl']

    gsf = GempakSurface(get_test_data('gem_ship.sfc'))

    gempak = pd.read_csv(get_test_data('gem_ship.csv'),
                         index_col=['STN', 'YYMMDD/HHMM'],
                         parse_dates=['YYMMDD/HHMM'],
                         date_parser=dtparse)
    gempak.sort_index(inplace=True)

    uidx = gempak.index.unique()

    for stn, dt in uidx:
        ugem = gempak.loc[(stn, dt), ]
        gstns = gsf.sfjson(station_id=stn, date_time=dt)

        assert len(ugem) == len(gstns)

        params = gempak.columns
        for param in params:
            if param not in skip:
                decoded_vals = [d['values'][param.lower()] for d in gstns]
                actual_vals = ugem.loc[:, param].values
                assert_allclose(decoded_vals, actual_vals)


@pytest.mark.parametrize('proj_type', ['conical', 'cylindrical', 'azimuthal'])
def test_coordinates_creation(proj_type):
    """Test projections and coordinates."""
    grid = GempakGrid(get_test_data(f'gem_{proj_type}.grd'))
    decode_lat = grid.lat
    decode_lon = grid.lon

    gempak = np.load(get_test_data(f'gem_{proj_type}.npz'))
    true_lat = gempak['lat']
    true_lon = gempak['lon']

    assert_allclose(decode_lat, true_lat, rtol=1e-6, atol=1e-2)
    assert_allclose(decode_lon, true_lon, rtol=1e-6, atol=1e-2)


@pytest.mark.parametrize('proj_type, proj_attrs', [
    ('conical', {
        'grid_mapping_name': 'lambert_conformal_conic',
        'standard_parallel': (25.0, 25.0),
        'latitude_of_projection_origin': 0.0,
        'longitude_of_central_meridian': -95.0,
        'semi_major_axis': 6371200.,
        'semi_minor_axis': 6371200.,
    }),
    ('azimuthal', {
        'grid_mapping_name': 'polar_stereographic',
        'latitude_of_projection_origin': 90.0,
        'straight_vertical_longitude_from_pole': -105.0,
        'scale_factor_at_projection_origin': 1.0,
        'semi_major_axis': 6371200.,
        'semi_minor_axis': 6371200.,
    })
])
def test_metpy_crs_creation(proj_type, proj_attrs):
    """Test grid mapping metadata."""
    grid = GempakGrid(get_test_data(f'gem_{proj_type}.grd'))
    arr = grid.gdxarray()[0]
    metpy_crs = arr.metpy.crs
    for k, v in proj_attrs.items():
        assert metpy_crs[k] == v
    x_unit = arr['x'].units
    y_unit = arr['y'].units
    assert x_unit == 'meters'
    assert y_unit == 'meters'


def test_date_parsing():
    """Test parsing of dates with leading zeroes."""
    sfc_data = GempakSurface(get_test_data('sfc_obs.gem'))
    dat = sfc_data.sfinfo()[0].DATTIM
    assert dat == datetime(2000, 1, 2)


@pytest.mark.parametrize('text_type,date_time', [
    ('text', '202109070000'), ('spcl', '202109071600')
])
def test_surface_text(text_type, date_time):
    """Test text decoding of surface hourly and special observations."""
    g = get_test_data('gem_surface_with_text.sfc')
    d = get_test_data('gem_surface_with_text.csv')

    gsf = GempakSurface(g)
    text = gsf.nearest_time(date_time, station_id='MSN')[0]['values'][text_type]

    gempak = pd.read_csv(d)
    gem_text = gempak.loc[:, text_type.upper()][0]

    assert text == gem_text


@pytest.mark.parametrize('text_type', ['txta', 'txtb', 'txtc', 'txpb'])
def test_sounding_text(text_type):
    """Test for proper decoding of coded message text."""
    g = get_test_data('gem_unmerged_with_text.snd')
    d = get_test_data('gem_unmerged_with_text.csv')

    gso = GempakSounding(g).snxarray(station_id='OUN')[0]
    gempak = pd.read_csv(d)

    text = gso.attrs['WMO_CODES'][text_type]
    gem_text = gempak.loc[:, text_type.upper()][0]

    assert text == gem_text


def test_special_surface_observation():
    """Test special surface observation conversion."""
    sfc = get_test_data('gem_surface_with_text.sfc')

    gsf = GempakSurface(sfc)
    stn = gsf.nearest_time('202109071601',
                           station_id='MSN',
                           include_special=True)[0]['values']

    assert_almost_equal(stn['pmsl'], 1003.81, 2)
    assert stn['alti'] == 29.66
    assert stn['tmpc'] == 22
    assert stn['dwpc'] == 18
    assert stn['sknt'] == 9
    assert stn['drct'] == 230
    assert stn['gust'] == 18
    assert stn['wnum'] == 77
    assert stn['chc1'] == 2703
    assert stn['chc2'] == 8004
    assert stn['chc3'] == -9999
    assert stn['vsby'] == 2
