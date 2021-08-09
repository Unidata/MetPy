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


def test_standard_surface():
    """Test to read a standard surface file."""
    def dtparse(string):
        return datetime.strptime(string, '%y%m%d/%H%M')

    skip = ['text']

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

    skip = ['text']

    gsf = GempakSurface(get_test_data('gem_ship.sfc'))

    gempak = pd.read_csv(get_test_data('gem_ship.csv'),
                         index_col=['STN', 'YYMMDD/HHMM'],
                         parse_dates=['YYMMDD/HHMM'],
                         date_parser=dtparse)

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
