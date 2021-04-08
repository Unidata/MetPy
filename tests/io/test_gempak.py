# Copyright (c) 2015,2016,2017,2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `gempak` module."""

from datetime import datetime
import logging

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from metpy.cbook import get_test_data
from metpy.io.gempak import GempakGrid, GempakSounding, GempakSurface

logging.getLogger('metpy.io.gempak').setLevel(logging.ERROR)


@pytest.mark.parametrize('grid_name', ['none', 'diff', 'dec', 'grib'])
def test_grid_loading(grid_name):
    """Test reading grids with different packing."""
    g = get_test_data(f'gem_packing_{grid_name}.grd')
    d = get_test_data(f'gem_packing_{grid_name}.npz')

    grid = GempakGrid(g).gdxarray(parameter='TMPK', level=850)[0]
    gio = grid.values.squeeze()

    gempak = np.load(d)['values']

    assert_allclose(gio, gempak, rtol=1e-6, atol=0)


def test_merged_sounding():
    """Test loading a merged sounding.

    These are most often from models.
    """
    g = get_test_data('gem_model_mrg.snd')
    d = get_test_data('gem_model_mrg.csv')

    gso = GempakSounding(g).snxarray(station_id='KMSN')[0]
    gpres = gso.pres.values
    gtemp = gso.tmpc.values.squeeze()
    gdwpt = gso.dwpc.values.squeeze()
    gdrct = gso.drct.values.squeeze()
    gsped = gso.sped.values.squeeze()
    ghght = gso.hght.values.squeeze()
    gomeg = gso.omeg.values.squeeze()
    gcwtr = gso.cwtr.values.squeeze()
    gdtcp = gso.dtcp.values.squeeze()
    gdtgp = gso.dtgp.values.squeeze()
    gdtsw = gso.dtsw.values.squeeze()
    gdtlw = gso.dtlw.values.squeeze()
    gcfrl = gso.cfrl.values.squeeze()
    gtkel = gso.tkel.values.squeeze()
    gimxr = gso.imxr.values.squeeze()
    gdtar = gso.dtar.values.squeeze()

    gempak = pd.read_csv(d, na_values=-9999)
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
    g = get_test_data(f'{gio}')
    d = get_test_data(f'{gem}')

    gso = GempakSounding(g).snxarray(station_id=f'{station}')[0]
    gpres = gso.pres.values
    gtemp = gso.temp.values.squeeze()
    gdwpt = gso.dwpt.values.squeeze()
    gdrct = gso.drct.values.squeeze()
    gsped = gso.sped.values.squeeze()
    ghght = gso.hght.values.squeeze()

    gempak = pd.read_csv(d, na_values=-9999)
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

    skip = ['text']

    g = get_test_data('gem_std.sfc')
    d = get_test_data('gem_std.csv')

    gsf = GempakSurface(g)
    gstns = gsf.sfjson()

    gempak = pd.read_csv(d, index_col=['STN', 'YYMMDD/HHMM'],
                         parse_dates=['YYMMDD/HHMM'],
                         date_parser=dtparse)
    if not gempak.index.is_lexsorted():
        gempak.sort_index(inplace=True)

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

    skip = ['text']

    g = get_test_data('gem_ship.sfc')
    d = get_test_data('gem_ship.csv')

    gsf = GempakSurface(g)

    gempak = pd.read_csv(d, index_col=['STN', 'YYMMDD/HHMM'],
                         parse_dates=['YYMMDD/HHMM'],
                         date_parser=dtparse)
    if not gempak.index.is_lexsorted():
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
    g = get_test_data(f'gem_{proj_type}.grd')
    d = get_test_data(f'gem_{proj_type}.npz')

    grid = GempakGrid(g)
    decode_lat = grid.lat
    decode_lon = grid.lon

    gempak = np.load(d)
    true_lat = gempak['lat']
    true_lon = gempak['lon']

    assert_allclose(decode_lat, true_lat, rtol=1e-6, atol=1e-2)
    assert_allclose(decode_lon, true_lon, rtol=1e-6, atol=1e-2)
