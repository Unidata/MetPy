# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools to process GEMPAK-formatted products."""

import bisect
from collections import namedtuple
from collections.abc import Iterable
import contextlib
from datetime import datetime, timedelta
from enum import Enum
from itertools import product, repeat
import logging
import math
import struct
import sys

import numpy as np
import pyproj
import xarray as xr

from ._tools import IOBuffer, NamedStruct, open_as_needed
from .. import constants
from ..calc import (altimeter_to_sea_level_pressure, scale_height,
                    specific_humidity_from_dewpoint, thickness_hydrostatic,
                    virtual_temperature)
from ..io.metar import parse_metar
from ..package_tools import Exporter
from ..plots.mapping import CFProjection
from ..units import units

exporter = Exporter(globals())
log = logging.getLogger(__name__)

BYTES_PER_WORD = 4
PARAM_ATTR = [('name', (4, 's')), ('scale', (1, 'i')),
              ('offset', (1, 'i')), ('bits', (1, 'i'))]
USED_FLAG = 9999
UNUSED_FLAG = -9999
GEMPAK_HEADER = 'GEMPAK DATA MANAGEMENT FILE '
GEMPROJ_TO_PROJ = {
    'MER': ('merc', 'cyl'),
    'NPS': ('stere', 'azm'),
    'SPS': ('stere', 'azm'),
    'LCC': ('lcc', 'con'),
    'SCC': ('lcc', 'con'),
    'CED': ('eqc', 'cyl'),
    'MCD': ('eqc', 'cyl'),
    'NOR': ('ortho', 'azm'),
    'SOR': ('ortho', 'azm'),
    'STR': ('stere', 'azm'),
    'AED': ('aeqd', 'azm'),
    'ORT': ('ortho', 'azm'),
    'LEA': ('laea', 'azm'),
    'GNO': ('gnom', 'azm'),
    'TVM': ('tmerc', 'obq'),
    'UTM': ('utm', 'obq'),
}
GVCORD_TO_VAR = {
    'PRES': 'p',
    'HGHT': 'z',
    'THTA': 'theta',
}


class FileTypes(Enum):
    """GEMPAK file type."""

    surface = 1
    sounding = 2
    grid = 3


class DataTypes(Enum):
    """Data management library data types."""

    real = 1
    integer = 2
    character = 3
    realpack = 4
    grid = 5


class VerticalCoordinates(Enum):
    """Vertical coordinates."""

    none = 0
    pres = 1
    thta = 2
    hght = 3
    sgma = 4
    dpth = 5
    hybd = 6
    pvab = 7
    pvbl = 8


class PackingType(Enum):
    """GRIB packing type."""

    none = 0
    grib = 1
    nmc = 2
    diff = 3
    dec = 4
    grib2 = 5


class ForecastType(Enum):
    """Forecast type."""

    analysis = 0
    forecast = 1
    guess = 2
    initial = 3


class DataSource(Enum):
    """Data source."""

    model = 0
    airway_surface = 1
    metar = 2
    ship = 3
    raob_buoy = 4
    synop_raob_vas = 5
    grid = 6
    watch_by_county = 7
    unknown = 99
    text = 100
    metar2 = 102
    ship2 = 103
    raob_buoy2 = 104
    synop_raob_vas2 = 105


Grid = namedtuple('Grid', [
    'GRIDNO',
    'TYPE',
    'DATTIM1',
    'DATTIM2',
    'PARM',
    'LEVEL1',
    'LEVEL2',
    'COORD',
])

Sounding = namedtuple('Sounding', [
    'DTNO',
    'SNDNO',
    'DATTIM',
    'ID',
    'NUMBER',
    'LAT',
    'LON',
    'ELEV',
    'STATE',
    'COUNTRY',
])

Surface = namedtuple('Surface', [
    'ROW',
    'COL',
    'DATTIM',
    'ID',
    'NUMBER',
    'LAT',
    'LON',
    'ELEV',
    'STATE',
    'COUNTRY',
])


def _check_nan(value, missing=-9999):
    """Check for nan values and replace with missing."""
    return missing if math.isnan(value) else value


def convert_degc_to_k(val, missing=-9999):
    """Convert scalar values from degC to K, handling missing values."""
    return val + constants.nounit.zero_degc if val != missing else val


def _data_source(source):
    """Get data source from stored integer."""
    try:
        return DataSource(source)
    except ValueError:
        log.warning('Could not interpret data source `%s`. '
                    'Setting to `Unknown`.', source)
        return DataSource(99)


def _word_to_position(word, bytes_per_word=BYTES_PER_WORD):
    """Return beginning position of a word in bytes."""
    return (word * bytes_per_word) - bytes_per_word


def _interp_logp_data(sounding, missing=-9999):
    """Interpolate missing sounding data.

    This function is similar to the MR_MISS subroutine in GEMPAK.
    """
    size = len(sounding['PRES'])
    recipe = [('TEMP', 'DWPT'), ('DRCT', 'SPED'), ('DWPT', None)]

    for var1, var2 in recipe:
        iabove = 0
        i = 1
        more = True
        while i < (size - 1) and more:
            if sounding[var1][i] == missing:
                if iabove <= i:
                    iabove = i + 1
                    found = False
                    while not found:
                        if sounding[var1][iabove] != missing:
                            found = True
                        else:
                            iabove += 1
                            if iabove >= size:
                                found = True
                                iabove = 0
                                more = False

                if (var2 is None and iabove != 0
                   and sounding['PRES'][i - 1] > 100
                   and sounding['PRES'][iabove] < 100):
                    iabove = 0

                if iabove != 0:
                    adata = {}
                    bdata = {}
                    for param, val in sounding.items():
                        if (param in ['PRES', 'TEMP', 'DWPT',
                                      'DRCT', 'SPED', 'HGHT']):
                            adata[param] = val[i - 1]
                            bdata[param] = val[iabove]
                    vlev = sounding['PRES'][i]
                    outdata = _interp_parameters(vlev, adata, bdata, missing)
                    sounding[var1][i] = outdata[var1]
                    if var2 is not None:
                        sounding[var2][i] = outdata[var2]
            i += 1


def _interp_logp_height(sounding, missing=-9999):
    """Interpolate height linearly with respect to log p.

    This function mimics the functionality of the MR_INTZ
    subroutine in GEMPAK.
    """
    size = len(sounding['HGHT'])

    idx = -1
    maxlev = -1
    while size + idx != 0:
        if sounding['HGHT'][idx] != missing:
            maxlev = size + idx
            break
        else:
            idx -= 1

    pbot = missing
    for i in range(maxlev):
        press = sounding['PRES'][i]
        hght = sounding['HGHT'][i]

        if press == missing:
            continue
        elif hght != missing:
            pbot = press
            zbot = hght
            ptop = 2000
        elif pbot == missing:
            continue
        else:
            ilev = i + 1
            while press <= ptop:
                if sounding['HGHT'][ilev] != missing:
                    ptop = sounding['PRES'][ilev]
                    ztop = sounding['HGHT'][ilev]
                else:
                    ilev += 1
            sounding['HGHT'][i] = (zbot + (ztop - zbot)
                                   * (np.log(press / pbot) / np.log(ptop / pbot)))

    if maxlev < size - 1:
        if maxlev > -1:
            pb = sounding['PRES'][maxlev] * 100  # hPa to Pa
            zb = sounding['HGHT'][maxlev]  # m
            tb = convert_degc_to_k(sounding['TEMP'][maxlev], missing)
            tdb = convert_degc_to_k(sounding['DWPT'][maxlev], missing)
        else:
            pb, zb, tb, tdb = repeat(missing, 4)

        for i in range(maxlev + 1, size):
            if sounding['HGHT'][i] == missing:
                tt = convert_degc_to_k(sounding['TEMP'][i], missing)
                tdt = convert_degc_to_k(sounding['DWPT'][i], missing)
                pt = sounding['PRES'][i] * 100  # hPa to Pa

                pl = np.array([pb, pt])
                tl = np.array([tb, tt])
                tdl = np.array([tdb, tdt])

                if missing in tdl:
                    tvl = tl
                else:
                    ql = specific_humidity_from_dewpoint._nounit(pl, tdl)
                    tvl = virtual_temperature._nounit(tl, ql)

                if missing not in [*tvl, zb]:
                    sounding['HGHT'][i] = (zb + thickness_hydrostatic._nounit(pl, tvl))
                else:
                    sounding['HGHT'][i] = missing


def _interp_logp_pressure(sounding, missing=-9999):
    """Interpolate pressure from heights.

    This function is similar to the MR_INTP subroutine from GEMPAK.
    """
    i = 0
    ilev = -1
    klev = -1
    size = len(sounding['PRES'])
    pt = missing
    zt = missing
    pb = missing
    zb = missing

    while i < size:
        p = sounding['PRES'][i]
        z = sounding['HGHT'][i]

        if p != missing and z != missing:
            klev = i
            pt = p
            zt = z

        if ilev != -1 and klev != -1:
            for j in range(ilev + 1, klev):
                z = sounding['HGHT'][j]
                if missing not in [z, zb, pb]:
                    sounding['PRES'][j] = (
                        pb * np.exp((z - zb) * np.log(pt / pb) / (zt - zb))
                    )
        ilev = klev
        pb = pt
        zb = zt
        i += 1


def _interp_moist_height(sounding, missing=-9999):
    """Interpolate moist hydrostatic height.

    This function mimics the functionality of the MR_SCMZ
    subroutine in GEMPAK. This the default behavior when
    merging observed sounding data.
    """
    hlist = np.ones(len(sounding['PRES'])) * -9999

    ilev = -1
    top = False

    found = False
    while not found and not top:
        ilev += 1
        if ilev >= len(sounding['PRES']):
            top = True
        elif missing not in [
            sounding['PRES'][ilev],
            sounding['TEMP'][ilev],
            sounding['HGHT'][ilev]
        ]:
            found = True

    while not top:
        plev = sounding['PRES'][ilev] * 100  # hPa to Pa
        pb = sounding['PRES'][ilev] * 100  # hPa to Pa
        tb = convert_degc_to_k(sounding['TEMP'][ilev], missing)
        tdb = convert_degc_to_k(sounding['DWPT'][ilev], missing)
        zb = sounding['HGHT'][ilev]  # m
        zlev = sounding['HGHT'][ilev]  # m
        jlev = ilev
        klev = 0
        mand = False

        while not mand:
            jlev += 1
            if jlev >= len(sounding['PRES']):
                mand = True
                top = True
            else:
                pt = sounding['PRES'][jlev] * 100  # hPa to Pa
                tt = convert_degc_to_k(sounding['TEMP'][jlev], missing)
                tdt = convert_degc_to_k(sounding['DWPT'][jlev], missing)
                zt = sounding['HGHT'][jlev]  # m
                if (zt != missing and tt != missing):
                    mand = True
                    klev = jlev

                pl = np.array([pb, pt])
                tl = np.array([tb, tt])
                tdl = np.array([tdb, tdt])

                if missing in tdl:
                    tvl = tl
                else:
                    ql = specific_humidity_from_dewpoint._nounit(pl, tdl)
                    tvl = virtual_temperature._nounit(tl, ql)

                if missing not in [*tl, zb]:
                    scale_z = scale_height._nounit(*tvl)
                    znew = zb + thickness_hydrostatic._nounit(pl, tvl)
                    tb = tt
                    tdb = tdt
                    pb = pt
                    zb = znew
                else:
                    scale_z, znew = repeat(missing, 2)
                hlist[jlev] = scale_z

        if klev != 0:
            s = (zt - zlev) / (znew - zlev)
            for h in range(ilev + 1, klev + 1):
                hlist[h] *= s

        hbb = zlev
        pbb = plev
        for ii in range(ilev + 1, jlev):
            p = sounding['PRES'][ii] * 100  # hPa to Pa
            scale_z = hlist[ii]
            if missing not in [scale_z, hbb, pbb, p]:
                th = (scale_z * constants.nounit.g) / constants.nounit.Rd
                tbar = np.array([th, th])
                pll = np.array([pbb, p])
                z = hbb + thickness_hydrostatic._nounit(pll, tbar)
            else:
                z = missing
            sounding['HGHT'][ii] = z
            hbb = z
            pbb = p

        ilev = klev


def _interp_parameters(vlev, adata, bdata, missing=-9999):
    """General interpolation with respect to log-p.

    See the PC_INTP subroutine in GEMPAK.
    """
    pres1 = adata['PRES']
    pres2 = bdata['PRES']
    between = (((pres1 < pres2) and (pres1 < vlev)
               and (vlev < pres2))
               or ((pres2 < pres1) and (pres2 < vlev)
               and (vlev < pres1)))

    if not between:
        raise ValueError('Current pressure does not fall between levels.')
    elif pres1 <= 0 or pres2 <= 0:
        raise ValueError('Pressure cannot be negative.')

    outdata = {}
    rmult = np.log(vlev / pres1) / np.log(pres2 / pres1)
    outdata['PRES'] = vlev
    for param, aval in adata.items():
        bval = bdata[param]
        if param == 'DRCT':
            ang1 = aval % 360
            ang2 = bval % 360
            if abs(ang1 - ang2) > 180:
                if ang1 < ang2:
                    ang1 += 360
                else:
                    ang2 += 360
            ang = ang1 + (ang2 - ang1) * rmult
            outdata[param] = ang % 360
        else:
            outdata[param] = aval + (bval - aval) * rmult

        if missing in [aval, bval]:
            outdata[param] = missing

    return outdata


def _wx_to_wnum(wx1, wx2, wx3, missing=-9999):
    """Convert METAR present weather code to GEMPAK weather number.

    Notes
    -----
    See GEMAPK function PT_WNMT.
    """
    metar_codes = [
        'BR', 'DS', 'DU', 'DZ', 'FC', 'FG', 'FU', 'GR', 'GS',
        'HZ', 'IC', 'PL', 'PO', 'RA', 'SA', 'SG', 'SN', 'SQ',
        'SS', 'TS', 'UP', 'VA', '+DS', '-DZ', '+DZ', '+FC',
        '-GS', '+GS', '-PL', '+PL', '-RA', '+RA', '-SG',
        '+SG', '-SN', '+SN', '+SS', 'BCFG', 'BLDU', 'BLPY',
        'BLSA', 'BLSN', 'DRDU', 'DRSA', 'DRSN', 'FZDZ', 'FZFG',
        'FZRA', 'MIFG', 'PRFG', 'SHGR', 'SHGS', 'SHPL', 'SHRA',
        'SHSN', 'TSRA', '+BLDU', '+BLSA', '+BLSN', '-FZDZ',
        '+FZDZ', '+FZFG', '-FZRA', '+FZRA', '-SHGS', '+SHGS',
        '-SHPL', '+SHPL', '-SHRA', '+SHRA', '-SHSN', '+SHSN',
        '-TSRA', '+TSRA'
    ]

    gempak_wnum = [
        9, 33, 8, 2, -2, 9, 7, 4, 25, 6, 36, 23, 40, 1, 35, 24, 3, 10,
        35, 5, 41, 11, 68, 17, 18, -1, 61, 62, 57, 58, 13, 14, 59, 60, 20,
        21, 69, 9, 33, 34, 35, 32, 33, 35, 32, 19, 30, 15, 31, 9, 27, 67,
        63, 16, 22, 66, 68, 69, 70, 53, 54, 30, 49, 50, 67, 67, 75, 76, 51,
        52, 55, 56, 77, 78
    ]

    if wx1 in metar_codes:
        wn1 = gempak_wnum[metar_codes.index(wx1)]
    else:
        wn1 = 0

    if wx2 in metar_codes:
        wn2 = gempak_wnum[metar_codes.index(wx2)]
    else:
        wn2 = 0

    if wx3 in metar_codes:
        wn3 = gempak_wnum[metar_codes.index(wx3)]
    else:
        wn3 = 0

    if all(w >= 0 for w in [wn1, wn2, wn3]):
        wnum = wn3 * 80 * 80 + wn2 * 80 + wn1
    else:
        wnum = min([wn1, wn2, wn3])
        if wnum == 0:
            wnum = missing

    return wnum


def _convert_clouds(cover, height, missing=-9999):
    """Convert METAR cloud cover to GEMPAK code.

    Notes
    -----
    See GEMPAK function BR_CMTN.
    """
    cover_text = ['CLR', 'SCT', 'BKN', 'OVC', 'VV', 'FEW', 'SKC']
    if not isinstance(cover, str):
        return missing

    code = 0
    if cover in cover_text:
        code = cover_text.index(cover) + 1

    if code == 7:
        code = 1

    if not math.isnan(height):
        code += height
        if height == 0:
            code *= -1

    return code


class GempakFile():
    """Base class for GEMPAK files.

    Reads ubiquitous GEMPAK file headers (i.e., the data management portion of
    each file).
    """

    prod_desc_fmt = [('version', 'i'), ('file_headers', 'i'),
                     ('file_keys_ptr', 'i'), ('rows', 'i'),
                     ('row_keys', 'i'), ('row_keys_ptr', 'i'),
                     ('row_headers_ptr', 'i'), ('columns', 'i'),
                     ('column_keys', 'i'), ('column_keys_ptr', 'i'),
                     ('column_headers_ptr', 'i'), ('parts', 'i'),
                     ('parts_ptr', 'i'), ('data_mgmt_ptr', 'i'),
                     ('data_mgmt_length', 'i'), ('data_block_ptr', 'i'),
                     ('file_type', 'i', FileTypes),
                     ('data_source', 'i', _data_source),
                     ('machine_type', 'i'), ('missing_int', 'i'),
                     (None, '12x'), ('missing_float', 'f')]

    grid_nav_fmt = [('grid_definition_type', 'f'),
                    ('projection', '3sx', bytes.decode),
                    ('left_grid_number', 'f'), ('bottom_grid_number', 'f'),
                    ('right_grid_number', 'f'), ('top_grid_number', 'f'),
                    ('lower_left_lat', 'f'), ('lower_left_lon', 'f'),
                    ('upper_right_lat', 'f'), ('upper_right_lon', 'f'),
                    ('proj_angle1', 'f'), ('proj_angle2', 'f'),
                    ('proj_angle3', 'f'), (None, '972x')]

    grid_anl_fmt1 = [('analysis_type', 'f'), ('delta_n', 'f'),
                     ('delta_x', 'f'), ('delta_y', 'f'),
                     (None, '4x'), ('garea_llcr_lat', 'f'),
                     ('garea_llcr_lon', 'f'), ('garea_urcr_lat', 'f'),
                     ('garea_urcr_lon', 'f'), ('extarea_llcr_lat', 'f'),
                     ('extarea_llcr_lon', 'f'), ('extarea_urcr_lat', 'f'),
                     ('extarea_urcr_lon', 'f'), ('datarea_llcr_lat', 'f'),
                     ('datarea_llcr_lon', 'f'), ('datarea_urcr_lat', 'f'),
                     ('datarea_urcrn_lon', 'f'), (None, '444x')]

    grid_anl_fmt2 = [('analysis_type', 'f'), ('delta_n', 'f'),
                     ('grid_ext_left', 'f'), ('grid_ext_down', 'f'),
                     ('grid_ext_right', 'f'), ('grid_ext_up', 'f'),
                     ('garea_llcr_lat', 'f'), ('garea_llcr_lon', 'f'),
                     ('garea_urcr_lat', 'f'), ('garea_urcr_lon', 'f'),
                     ('extarea_llcr_lat', 'f'), ('extarea_llcr_lon', 'f'),
                     ('extarea_urcr_lat', 'f'), ('extarea_urcr_lon', 'f'),
                     ('datarea_llcr_lat', 'f'), ('datarea_llcr_lon', 'f'),
                     ('datarea_urcr_lat', 'f'), ('datarea_urcrn_lon', 'f'),
                     (None, '440x')]

    data_management_fmt = ([('next_free_word', 'i'), ('max_free_pairs', 'i'),
                           ('actual_free_pairs', 'i'), ('last_word', 'i')]
                           + [('free_word{:d}'.format(n), 'i') for n in range(1, 29)])

    def __init__(self, file):
        """Instantiate GempakFile object from file."""
        fobj = open_as_needed(file)

        with contextlib.closing(fobj):
            self._buffer = IOBuffer.fromfile(fobj)

        # Save file start position as pointers use this as reference
        self._start = self._buffer.set_mark()

        # Process the main GEMPAK header to verify file format
        self._process_gempak_header()
        meta = self._buffer.set_mark()

        # # Check for byte swapping
        self._swap_bytes(bytes(self._buffer.read_binary(4)))
        self._buffer.jump_to(meta)

        # Process main metadata header
        self.prod_desc = self._buffer.read_struct(NamedStruct(self.prod_desc_fmt,
                                                              self.prefmt,
                                                              'ProductDescription'))

        # File Keys
        # Surface and upper-air files will not have the file headers, so we need to check.
        if self.prod_desc.file_headers > 0:
            # This would grab any file headers, but NAVB and ANLB are the only ones used.
            fkey_prod = product(['header_name', 'header_length', 'header_type'],
                                range(1, self.prod_desc.file_headers + 1))
            fkey_names = ['{}{}'.format(*x) for x in fkey_prod]
            fkey_info = list(zip(fkey_names, np.repeat(('4s', 'i', 'i'),
                                                       self.prod_desc.file_headers)))
            self.file_keys_format = NamedStruct(fkey_info, self.prefmt, 'FileKeys')

            self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.file_keys_ptr))
            self.file_keys = self._buffer.read_struct(self.file_keys_format)

            # file_key_blocks = self._buffer.set_mark()
            # Navigation Block
            navb_size = self._buffer.read_int(4, self.endian, False)

            nav_stuct = NamedStruct(self.grid_nav_fmt,
                                    self.prefmt,
                                    'NavigationBlock')

            if navb_size != nav_stuct.size // BYTES_PER_WORD:
                raise ValueError('Navigation block size does not match GEMPAK specification')
            else:
                self.navigation_block = (
                    self._buffer.read_struct(nav_stuct)
                )
            self.kx = int(self.navigation_block.right_grid_number)
            self.ky = int(self.navigation_block.top_grid_number)

            # Analysis Block
            anlb_size = self._buffer.read_int(4, self.endian, False)
            anlb_start = self._buffer.set_mark()
            anlb1_struct = NamedStruct(self.grid_anl_fmt1,
                                       self.prefmt,
                                       'AnalysisBlock')
            anlb2_struct = NamedStruct(self.grid_anl_fmt2,
                                       self.prefmt,
                                       'AnalysisBlock')

            if anlb_size not in [anlb1_struct.size // BYTES_PER_WORD,
                                 anlb2_struct.size // BYTES_PER_WORD]:
                raise ValueError('Analysis block size does not match GEMPAK specification')
            else:
                anlb_type = self._buffer.read_struct(struct.Struct(self.prefmt + 'f'))[0]
                self._buffer.jump_to(anlb_start)
                if anlb_type == 1:
                    self.analysis_block = (
                        self._buffer.read_struct(anlb1_struct)
                    )
                elif anlb_type == 2:
                    self.analysis_block = (
                        self._buffer.read_struct(anlb2_struct)
                    )
                else:
                    self.analysis_block = None
        else:
            self.analysis_block = None
            self.navigation_block = None

        # Data Management
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.data_mgmt_ptr))
        self.data_management = self._buffer.read_struct(NamedStruct(self.data_management_fmt,
                                                                    self.prefmt,
                                                                    'DataManagement'))

        # Row Keys
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.row_keys_ptr))
        row_key_info = [('row_key{:d}'.format(n), '4s', self._decode_strip)
                        for n in range(1, self.prod_desc.row_keys + 1)]
        row_key_info.extend([(None, None)])
        row_keys_fmt = NamedStruct(row_key_info, self.prefmt, 'RowKeys')
        self.row_keys = self._buffer.read_struct(row_keys_fmt)

        # Column Keys
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.column_keys_ptr))
        column_key_info = [('column_key{:d}'.format(n), '4s', self._decode_strip)
                           for n in range(1, self.prod_desc.column_keys + 1)]
        column_key_info.extend([(None, None)])
        column_keys_fmt = NamedStruct(column_key_info, self.prefmt, 'ColumnKeys')
        self.column_keys = self._buffer.read_struct(column_keys_fmt)

        # Parts
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.parts_ptr))
        # parts = self._buffer.set_mark()
        self.parts = []
        parts_info = [('name', '4s', self._decode_strip),
                      (None, '{:d}x'.format((self.prod_desc.parts - 1) * BYTES_PER_WORD)),
                      ('header_length', 'i'),
                      (None, '{:d}x'.format((self.prod_desc.parts - 1) * BYTES_PER_WORD)),
                      ('data_type', 'i', DataTypes),
                      (None, '{:d}x'.format((self.prod_desc.parts - 1) * BYTES_PER_WORD)),
                      ('parameter_count', 'i')]
        parts_info.extend([(None, None)])
        parts_fmt = NamedStruct(parts_info, self.prefmt, 'Parts')
        for n in range(1, self.prod_desc.parts + 1):
            self.parts.append(self._buffer.read_struct(parts_fmt))
            self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.parts_ptr + n))

        # Parameters
        # No need to jump to any position as this follows parts information
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.parts_ptr
                                                            + self.prod_desc.parts * 4))
        self.parameters = [{key: [] for key, _ in PARAM_ATTR}
                           for n in range(self.prod_desc.parts)]
        for attr, fmt in PARAM_ATTR:
            fmt = (fmt[0], self.prefmt + fmt[1] if fmt[1] != 's' else fmt[1])
            for n, part in enumerate(self.parts):
                for _ in range(part.parameter_count):
                    if 's' in fmt[1]:
                        self.parameters[n][attr] += [
                            self._decode_strip(self._buffer.read_binary(*fmt)[0])
                        ]
                    else:
                        self.parameters[n][attr] += self._buffer.read_binary(*fmt)

    def _swap_bytes(self, binary):
        """Swap between little and big endian."""
        self.swaped_bytes = (struct.pack('@i', 1) != binary)

        if self.swaped_bytes:
            if sys.byteorder == 'little':
                self.prefmt = '>'
                self.endian = 'big'
            elif sys.byteorder == 'big':
                self.prefmt = '<'
                self.endian = 'little'
        else:
            self.prefmt = ''
            self.endian = sys.byteorder

    def _process_gempak_header(self):
        """Read the GEMPAK header from the file."""
        fmt = [('text', '28s', bytes.decode), (None, None)]

        header = self._buffer.read_struct(NamedStruct(fmt, '', 'GempakHeader'))
        if header.text != GEMPAK_HEADER:
            raise TypeError('Unknown file format or invalid GEMPAK file')

    @staticmethod
    def _convert_dattim(dattim):
        """Convert GEMPAK DATTIM integer to datetime object."""
        if dattim:
            if dattim < 100000000:
                dt = datetime.strptime(f'{dattim:06d}', '%y%m%d')
            else:
                dt = datetime.strptime('{:010d}'.format(dattim), '%m%d%y%H%M')
        else:
            dt = None
        return dt

    @staticmethod
    def _convert_ftime(ftime):
        """Convert GEMPAK forecast time and type integer."""
        if ftime >= 0:
            iftype = ForecastType(ftime // 100000)
            iftime = ftime - iftype.value * 100000
            hours = iftime // 100
            minutes = iftime - hours * 100
            out = (iftype.name, timedelta(hours=hours, minutes=minutes))
        else:
            out = None
        return out

    @staticmethod
    def _convert_level(level):
        """Convert levels."""
        if isinstance(level, (int, float)) and level >= 0:
            return level
        else:
            return None

    @staticmethod
    def _convert_vertical_coord(coord):
        """Convert integer vertical coordinate to name."""
        if coord <= 8:
            return VerticalCoordinates(coord).name.upper()
        else:
            return struct.pack('i', coord).decode()

    @staticmethod
    def _fortran_ishift(i, shift):
        """Python-friendly bit shifting."""
        if shift >= 0:
            # Shift left and only keep low 32 bits
            ret = (i << shift) & 0xffffffff

            # If high bit, convert back to negative of two's complement
            if ret > 0x7fffffff:
                ret = -(~ret & 0x7fffffff) - 1
            return ret
        else:
            # Shift right the low 32 bits
            return (i & 0xffffffff) >> -shift

    @staticmethod
    def _decode_strip(b):
        """Decode bytes to string and strip whitespace."""
        return b.decode().strip()

    @staticmethod
    def _make_date(dattim):
        """Make a date object from GEMPAK DATTIM integer."""
        return GempakFile._convert_dattim(dattim).date()

    @staticmethod
    def _make_time(t):
        """Make a time object from GEMPAK FTIME integer."""
        string = '{:04d}'.format(t)
        return datetime.strptime(string, '%H%M').time()

    def _unpack_real(self, buffer, parameters, length):
        """Unpack floating point data packed in integers.

        Similar to DP_UNPK subroutine in GEMPAK.
        """
        nparms = len(parameters['name'])
        mskpat = 0xffffffff

        pwords = (sum(parameters['bits']) - 1) // 32 + 1
        npack = (length - 1) // pwords + 1
        unpacked = np.ones(npack * nparms, dtype=np.float32) * self.prod_desc.missing_float
        if npack * pwords != length:
            raise ValueError('Unpacking length mismatch.')

        ir = 0
        ii = 0
        for _i in range(npack):
            pdat = buffer[ii:(ii + pwords)]
            rdat = unpacked[ir:(ir + nparms)]
            itotal = 0
            for idata in range(nparms):
                scale = 10**parameters['scale'][idata]
                offset = parameters['offset'][idata]
                bits = parameters['bits'][idata]
                isbitc = (itotal % 32) + 1
                iswrdc = (itotal // 32)
                imissc = self._fortran_ishift(mskpat, bits - 32)

                jbit = bits
                jsbit = isbitc
                jshift = 1 - jsbit
                jsword = iswrdc
                jword = pdat[jsword]
                mask = self._fortran_ishift(mskpat, jbit - 32)
                ifield = self._fortran_ishift(jword, jshift)
                ifield &= mask

                if (jsbit + jbit - 1) > 32:
                    jword = pdat[jsword + 1]
                    jshift += 32
                    iword = self._fortran_ishift(jword, jshift)
                    iword &= mask
                    ifield |= iword

                if ifield == imissc:
                    rdat[idata] = self.prod_desc.missing_float
                else:
                    rdat[idata] = (ifield + offset) * scale
                itotal += bits
            unpacked[ir:(ir + nparms)] = rdat
            ir += nparms
            ii += pwords

        return unpacked.tolist()


@exporter.export
class GempakGrid(GempakFile):
    """Subclass of GempakFile specific to GEMPAK gridded data."""

    def __init__(self, file, *args, **kwargs):
        super().__init__(file)

        datetime_names = ['GDT1', 'GDT2']
        level_names = ['GLV1', 'GLV2']
        ftime_names = ['GTM1', 'GTM2']
        string_names = ['GPM1', 'GPM2', 'GPM3']

        # Row Headers
        # Based on GEMPAK source, row/col headers have a 0th element in their Fortran arrays.
        # This appears to be a flag value to say a header is used or not. 9999
        # means its in use, otherwise -9999. GEMPAK allows empty grids, etc., but
        # no real need to keep track of that in Python.
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.row_headers_ptr))
        self.row_headers = []
        row_headers_info = [(key, 'i') for key in self.row_keys]
        row_headers_info.extend([(None, None)])
        row_headers_fmt = NamedStruct(row_headers_info, self.prefmt, 'RowHeaders')
        for _ in range(1, self.prod_desc.rows + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.row_headers.append(self._buffer.read_struct(row_headers_fmt))

        # Column Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.column_headers_ptr))
        self.column_headers = []
        column_headers_info = [(key, 'i', self._convert_level) if key in level_names
                               else (key, 'i', self._convert_vertical_coord) if key == 'GVCD'
                               else (key, 'i', self._convert_dattim) if key in datetime_names
                               else (key, 'i', self._convert_ftime) if key in ftime_names
                               else (key, '4s', self._decode_strip) if key in string_names
                               else (key, 'i')
                               for key in self.column_keys]
        column_headers_info.extend([(None, None)])
        column_headers_fmt = NamedStruct(column_headers_info, self.prefmt, 'ColumnHeaders')
        for _ in range(1, self.prod_desc.columns + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.column_headers.append(self._buffer.read_struct(column_headers_fmt))

        self._gdinfo = []
        for n, head in enumerate(self.column_headers):
            self._gdinfo.append(
                Grid(
                    n,
                    head.GTM1[0],
                    head.GDT1 + head.GTM1[1],
                    head.GDT2 + head.GTM2[1] if head.GDT2 and head.GDTM2 else None,
                    head.GPM1 + head.GPM2 + head.GPM3,
                    head.GLV1,
                    head.GLV2,
                    head.GVCD,
                )
            )

        # Coordinates
        if self.navigation_block is not None:
            self._get_crs()
            self._set_coordinates()

    def gdinfo(self):
        """Return grid information."""
        return self._gdinfo

    def _get_crs(self):
        """Create CRS from GEMPAK navigation block."""
        gemproj = self.navigation_block.projection
        proj, ptype = GEMPROJ_TO_PROJ[gemproj]
        radius_sph = 6371200.0

        if ptype == 'azm':
            lat_0 = self.navigation_block.proj_angle1
            lon_0 = self.navigation_block.proj_angle2
            rot = self.navigation_block.proj_angle3
            if rot != 0:
                log.warning('Rotated projections currently '
                            'not supported. Angle3 (%7.2f) ignored.', rot)
            self.crs = pyproj.CRS.from_dict({'proj': proj,
                                             'lat_0': lat_0,
                                             'lon_0': lon_0,
                                             'R': radius_sph})
        elif ptype == 'cyl':
            if gemproj != 'mcd':
                lat_0 = self.navigation_block.proj_angle1
                lon_0 = self.navigation_block.proj_angle2
                rot = self.navigation_block.proj_angle3
                if rot != 0:
                    log.warning('Rotated projections currently '
                                'not supported. Angle3 (%7.2f) ignored.', rot)
                self.crs = pyproj.CRS.from_dict({'proj': proj,
                                                 'lat_0': lat_0,
                                                 'lon_0': lon_0,
                                                 'R': radius_sph})
            else:
                avglat = (self.navigation_block.upper_right_lat
                          + self.navigation_block.lower_left_lat) * 0.5
                k_0 = (1 / math.cos(avglat)
                       if self.navigation_block.proj_angle1 == 0
                       else self.navigation_block.proj_angle1
                       )
                lon_0 = self.navigation_block.proj_angle2
                self.crs = pyproj.CRS.from_dict({'proj': proj,
                                                 'lat_0': avglat,
                                                 'lon_0': lon_0,
                                                 'k_0': k_0,
                                                 'R': radius_sph})
        elif ptype == 'con':
            lat_1 = self.navigation_block.proj_angle1
            lon_0 = self.navigation_block.proj_angle2
            lat_2 = self.navigation_block.proj_angle3
            self.crs = pyproj.CRS.from_dict({'proj': proj,
                                             'lon_0': lon_0,
                                             'lat_1': lat_1,
                                             'lat_2': lat_2,
                                             'R': radius_sph})

    def _set_coordinates(self):
        """Use GEMPAK navigation block to define coordinates.

        Defines geographic and projection coordinates for the object.
        """
        transform = pyproj.Proj(self.crs)
        llx, lly = transform(self.navigation_block.lower_left_lon,
                             self.navigation_block.lower_left_lat)
        urx, ury = transform(self.navigation_block.upper_right_lon,
                             self.navigation_block.upper_right_lat)
        self.x = np.linspace(llx, urx, self.kx, dtype=np.float32)
        self.y = np.linspace(lly, ury, self.ky, dtype=np.float32)
        xx, yy = np.meshgrid(self.x, self.y, copy=False)
        self.lon, self.lat = transform(xx, yy, inverse=True)
        self.lon = self.lon.astype(np.float32)
        self.lat = self.lat.astype(np.float32)

    def _unpack_grid(self, packing_type, part):
        """Read raw GEMPAK grid integers and unpack into floats."""
        if packing_type == PackingType.none:
            lendat = self.data_header_length - part.header_length - 1

            if lendat > 1:
                buffer_fmt = '{}{}f'.format(self.prefmt, lendat)
                buffer = self._buffer.read_struct(struct.Struct(buffer_fmt))
                grid = np.zeros(self.ky * self.kx, dtype=np.float32)
                grid[...] = buffer
            else:
                grid = None

            return grid

        elif packing_type == PackingType.nmc:
            raise NotImplementedError('NMC unpacking not supported.')
            # integer_meta_fmt = [('bits', 'i'), ('missing_flag', 'i'), ('kxky', 'i')]
            # real_meta_fmt = [('reference', 'f'), ('scale', 'f')]
            # self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt,
            #                                                           self.prefmt,
            #                                                           'GridMetaInt'))
            # self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt,
            #                                                            self.prefmt,
            #                                                            'GridMetaReal'))
            # grid_start = self._buffer.set_mark()
        elif packing_type == PackingType.diff:
            integer_meta_fmt = [('bits', 'i'), ('missing_flag', 'i'),
                                ('kxky', 'i'), ('kx', 'i')]
            real_meta_fmt = [('reference', 'f'), ('scale', 'f'), ('diffmin', 'f')]
            self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt,
                                                                      self.prefmt,
                                                                      'GridMetaInt'))
            self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt,
                                                                       self.prefmt,
                                                                       'GridMetaReal'))
            # grid_start = self._buffer.set_mark()

            imiss = 2**self.grid_meta_int.bits - 1
            lendat = self.data_header_length - part.header_length - 8
            packed_buffer_fmt = '{}{}i'.format(self.prefmt, lendat)
            packed_buffer = self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
            grid = np.zeros((self.ky, self.kx), dtype=np.float32)

            if lendat > 1:
                iword = 0
                ibit = 1
                first = True
                for j in range(self.ky):
                    line = False
                    for i in range(self.kx):
                        jshft = self.grid_meta_int.bits + ibit - 33
                        idat = self._fortran_ishift(packed_buffer[iword], jshft)
                        idat &= imiss

                        if jshft > 0:
                            jshft -= 32
                            idat2 = self._fortran_ishift(packed_buffer[iword + 1], jshft)
                            idat |= idat2

                        ibit += self.grid_meta_int.bits
                        if ibit > 32:
                            ibit -= 32
                            iword += 1

                        if (self.grid_meta_int.missing_flag and idat == imiss):
                            grid[j, i] = self.prod_desc.missing_float
                        else:
                            if first:
                                grid[j, i] = self.grid_meta_real.reference
                                psav = self.grid_meta_real.reference
                                plin = self.grid_meta_real.reference
                                line = True
                                first = False
                            else:
                                if not line:
                                    grid[j, i] = plin + (self.grid_meta_real.diffmin
                                                         + idat * self.grid_meta_real.scale)
                                    line = True
                                    plin = grid[j, i]
                                else:
                                    grid[j, i] = psav + (self.grid_meta_real.diffmin
                                                         + idat * self.grid_meta_real.scale)
                                psav = grid[j, i]
            else:
                grid = None

            return grid

        elif packing_type in [PackingType.grib, PackingType.dec]:
            integer_meta_fmt = [('bits', 'i'), ('missing_flag', 'i'), ('kxky', 'i')]
            real_meta_fmt = [('reference', 'f'), ('scale', 'f')]
            self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt,
                                                                      self.prefmt,
                                                                      'GridMetaInt'))
            self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt,
                                                                       self.prefmt,
                                                                       'GridMetaReal'))
            # grid_start = self._buffer.set_mark()

            lendat = self.data_header_length - part.header_length - 6
            packed_buffer_fmt = '{}{}i'.format(self.prefmt, lendat)

            grid = np.zeros(self.grid_meta_int.kxky, dtype=np.float32)
            packed_buffer = self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
            if lendat > 1:
                imax = 2**self.grid_meta_int.bits - 1
                ibit = 1
                iword = 0
                for cell in range(self.grid_meta_int.kxky):
                    jshft = self.grid_meta_int.bits + ibit - 33
                    idat = self._fortran_ishift(packed_buffer[iword], jshft)
                    idat &= imax

                    if jshft > 0:
                        jshft -= 32
                        idat2 = self._fortran_ishift(packed_buffer[iword + 1], jshft)
                        idat |= idat2

                    if (idat == imax) and self.grid_meta_int.missing_flag:
                        grid[cell] = self.prod_desc.missing_float
                    else:
                        grid[cell] = (self.grid_meta_real.reference
                                      + (idat * self.grid_meta_real.scale))

                    ibit += self.grid_meta_int.bits
                    if ibit > 32:
                        ibit -= 32
                        iword += 1
            else:
                grid = None

            return grid
        elif packing_type == PackingType.grib2:
            raise NotImplementedError('GRIB2 unpacking not supported.')
            # integer_meta_fmt = [('iuscal', 'i'), ('kx', 'i'),
            #                     ('ky', 'i'), ('iscan_mode', 'i')]
            # real_meta_fmt = [('rmsval', 'f')]
            # self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt,
            #                                                           self.prefmt,
            #                                                           'GridMetaInt'))
            # self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt,
            #                                                            self.prefmt,
            #                                                            'GridMetaReal'))
            # grid_start = self._buffer.set_mark()
        else:
            raise NotImplementedError('No method for unknown grid packing {}'
                                      .format(packing_type.name))

    def gdxarray(self, parameter=None, date_time=None, coordinate=None,
                 level=None, date_time2=None, level2=None):
        """Select grids and output as list of xarray DataArrays.

        Subset the data by parameter values. The default is to not
        subset and return the entire dataset.

        Parameters
        ----------
        parameter : str or Sequence[str]
            Name of GEMPAK parameter.

        date_time : `~datetime.datetime` or Sequence[datetime]
            Valid datetime of the grid. Alternatively can be
            a string with the format YYYYmmddHHMM.

        coordinate : str or Sequence[str]
            Vertical coordinate.

        level : float or Sequence[float]
            Vertical level.

        date_time2 : `~datetime.datetime` or Sequence[datetime]
            Secondary valid datetime of the grid. Alternatively can be
            a string with the format YYYYmmddHHMM.

        level2: float or Sequence[float]
            Secondary vertical level. Typically used for layers.

        Returns
        -------
        list
            List of `xarray.DataArray` objects for each grid.
        """
        if parameter is not None:
            if (not isinstance(parameter, Iterable)
               or isinstance(parameter, str)):
                parameter = [parameter]
            parameter = [p.upper() for p in parameter]

        if date_time is not None:
            if (not isinstance(date_time, Iterable)
               or isinstance(date_time, str)):
                date_time = [date_time]
            for i, dt in enumerate(date_time):
                if isinstance(dt, str):
                    date_time[i] = datetime.strptime(dt, '%Y%m%d%H%M')

        if coordinate is not None:
            if (not isinstance(coordinate, Iterable)
               or isinstance(coordinate, str)):
                coordinate = [coordinate]
            coordinate = [c.upper() for c in coordinate]

        if level is not None and not isinstance(level, Iterable):
            level = [level]

        if date_time2 is not None:
            if (not isinstance(date_time2, Iterable)
               or isinstance(date_time2, str)):
                date_time2 = [date_time2]
            for i, dt in enumerate(date_time2):
                if isinstance(dt, str):
                    date_time2[i] = datetime.strptime(dt, '%Y%m%d%H%M')

        if level2 is not None and not isinstance(level2, Iterable):
            level2 = [level2]

        # Figure out which columns to extract from the file
        matched = self._gdinfo.copy()

        if parameter is not None:
            matched = filter(
                lambda grid: grid if grid.PARM in parameter else False,
                matched
            )

        if date_time is not None:
            matched = filter(
                lambda grid: grid if grid.DATTIM1 in date_time else False,
                matched
            )

        if coordinate is not None:
            matched = filter(
                lambda grid: grid if grid.COORD in coordinate else False,
                matched
            )

        if level is not None:
            matched = filter(
                lambda grid: grid if grid.LEVEL1 in level else False,
                matched
            )

        if date_time2 is not None:
            matched = filter(
                lambda grid: grid if grid.DATTIM2 in date_time2 else False,
                matched
            )

        if level2 is not None:
            matched = filter(
                lambda grid: grid if grid.LEVEL2 in level2 else False,
                matched
            )

        matched = list(matched)

        if len(matched) < 1:
            raise KeyError('No grids were matched with given parameters.')

        gridno = [g.GRIDNO for g in matched]

        grids = []
        irow = 0  # Only one row for grids
        for icol, col_head in enumerate(self.column_headers):
            if icol not in gridno:
                continue
            for iprt, part in enumerate(self.parts):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts + iprt))
                self._buffer.jump_to(self._start, _word_to_position(pointer))
                self.data_ptr = self._buffer.read_int(4, self.endian, False)
                self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                self.data_header_length = self._buffer.read_int(4, self.endian, False)
                data_header = self._buffer.set_mark()
                self._buffer.jump_to(data_header,
                                     _word_to_position(part.header_length + 1))
                packing_type = PackingType(self._buffer.read_int(4, self.endian, False))

                full_name = col_head.GPM1 + col_head.GPM2 + col_head.GPM3
                ftype, ftime = col_head.GTM1
                valid = col_head.GDT1 + ftime
                gvcord = col_head.GVCD.lower() if col_head.GVCD is not None else 'none'
                var = (GVCORD_TO_VAR[full_name]
                       if full_name in GVCORD_TO_VAR
                       else full_name.lower()
                       )
                data = self._unpack_grid(packing_type, part)
                if data is not None:
                    if data.ndim < 2:
                        data = np.ma.array(data.reshape((self.ky, self.kx)),
                                           mask=data == self.prod_desc.missing_float,
                                           dtype=np.float32)
                    else:
                        data = np.ma.array(data, mask=data == self.prod_desc.missing_float,
                                           dtype=np.float32)

                    xrda = xr.DataArray(
                        data=data[np.newaxis, np.newaxis, ...],
                        coords={
                            'time': [valid],
                            gvcord: [col_head.GLV1],
                            'x': self.x,
                            'y': self.y,
                            'metpy_crs': CFProjection(self.crs.to_cf())
                        },
                        dims=['time', gvcord, 'y', 'x'],
                        name=var,
                        attrs={
                            'gempak_grid_type': ftype,
                        }
                    )
                    xrda = xrda.metpy.assign_latitude_longitude()
                    xrda['x'].attrs['units'] = 'meters'
                    xrda['y'].attrs['units'] = 'meters'
                    grids.append(xrda)

                else:
                    log.warning('Unable to read grid for %s', col_head.GPM1)
        return grids


@exporter.export
class GempakSounding(GempakFile):
    """Subclass of GempakFile specific to GEMPAK sounding data."""

    def __init__(self, file, *args, **kwargs):
        super().__init__(file)

        # Row Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.row_headers_ptr))
        self.row_headers = []
        row_headers_info = [(key, 'i', self._make_date) if key == 'DATE'
                            else (key, 'i', self._make_time) if key == 'TIME'
                            else (key, 'i')
                            for key in self.row_keys]
        row_headers_info.extend([(None, None)])
        row_headers_fmt = NamedStruct(row_headers_info, self.prefmt, 'RowHeaders')
        for _ in range(1, self.prod_desc.rows + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.row_headers.append(self._buffer.read_struct(row_headers_fmt))

        # Column Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.column_headers_ptr))
        self.column_headers = []
        column_headers_info = [(key, '4s', self._decode_strip) if key == 'STID'
                               else (key, 'i') if key == 'STNM'
                               else (key, 'i', lambda x: x / 100) if key == 'SLAT'
                               else (key, 'i', lambda x: x / 100) if key == 'SLON'
                               else (key, 'i') if key == 'SELV'
                               else (key, '4s', self._decode_strip) if key == 'STAT'
                               else (key, '4s', self._decode_strip) if key == 'COUN'
                               else (key, '4s', self._decode_strip) if key == 'STD2'
                               else (key, 'i')
                               for key in self.column_keys]
        column_headers_info.extend([(None, None)])
        column_headers_fmt = NamedStruct(column_headers_info, self.prefmt, 'ColumnHeaders')
        for _ in range(1, self.prod_desc.columns + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.column_headers.append(self._buffer.read_struct(column_headers_fmt))

        self.merged = 'SNDT' in (part.name for part in self.parts)

        self._sninfo = []
        for irow, row_head in enumerate(self.row_headers):
            for icol, col_head in enumerate(self.column_headers):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts))

                self._buffer.jump_to(self._start, _word_to_position(pointer))
                data_ptr = self._buffer.read_int(4, self.endian, False)

                if data_ptr:
                    self._sninfo.append(
                        Sounding(
                            irow,
                            icol,
                            datetime.combine(row_head.DATE, row_head.TIME),
                            col_head.STID,
                            col_head.STNM,
                            col_head.SLAT,
                            col_head.SLON,
                            col_head.SELV,
                            col_head.STAT,
                            col_head.COUN,
                        )
                    )

    def sninfo(self):
        """Return sounding information."""
        return self._sninfo

    def _unpack_merged(self, sndno):
        """Unpack merged sounding data."""
        soundings = []
        for irow, row_head in enumerate(self.row_headers):
            for icol, col_head in enumerate(self.column_headers):
                if (irow, icol) not in sndno:
                    continue
                sounding = {'STID': col_head.STID,
                            'STNM': col_head.STNM,
                            'SLAT': col_head.SLAT,
                            'SLON': col_head.SLON,
                            'SELV': col_head.SELV,
                            'STAT': col_head.STAT,
                            'COUN': col_head.COUN,
                            'DATE': row_head.DATE,
                            'TIME': row_head.TIME,
                            }
                for iprt, part in enumerate(self.parts):
                    pointer = (self.prod_desc.data_block_ptr
                               + (irow * self.prod_desc.columns * self.prod_desc.parts)
                               + (icol * self.prod_desc.parts + iprt))
                    self._buffer.jump_to(self._start, _word_to_position(pointer))
                    self.data_ptr = self._buffer.read_int(4, self.endian, False)
                    if not self.data_ptr:
                        continue
                    self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                    self.data_header_length = self._buffer.read_int(4, self.endian, False)
                    data_header = self._buffer.set_mark()
                    self._buffer.jump_to(data_header,
                                         _word_to_position(part.header_length + 1))
                    lendat = self.data_header_length - part.header_length

                    fmt_code = {
                        DataTypes.real: 'f',
                        DataTypes.realpack: 'i',
                        DataTypes.character: 's',
                    }.get(part.data_type)

                    if fmt_code is None:
                        raise NotImplementedError('No methods for data type {}'
                                                  .format(part.data_type))

                    if fmt_code == 's':
                        lendat *= BYTES_PER_WORD

                    packed_buffer = (
                        self._buffer.read_struct(
                            struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                        )
                    )

                    parameters = self.parameters[iprt]
                    nparms = len(parameters['name'])

                    if part.data_type == DataTypes.realpack:
                        unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                        for iprm, param in enumerate(parameters['name']):
                            sounding[param] = unpacked[iprm::nparms]
                    else:
                        for iprm, param in enumerate(parameters['name']):
                            sounding[param] = np.array(
                                packed_buffer[iprm::nparms], dtype=np.float32
                            )

                soundings.append(sounding)
        return soundings

    def _unpack_unmerged(self, sndno):
        """Unpack unmerged sounding data."""
        soundings = []
        for irow, row_head in enumerate(self.row_headers):
            for icol, col_head in enumerate(self.column_headers):
                if (irow, icol) not in sndno:
                    continue
                sounding = {'STID': col_head.STID,
                            'STNM': col_head.STNM,
                            'SLAT': col_head.SLAT,
                            'SLON': col_head.SLON,
                            'SELV': col_head.SELV,
                            'STAT': col_head.STAT,
                            'COUN': col_head.COUN,
                            'DATE': row_head.DATE,
                            'TIME': row_head.TIME,
                            }
                for iprt, part in enumerate(self.parts):
                    pointer = (self.prod_desc.data_block_ptr
                               + (irow * self.prod_desc.columns * self.prod_desc.parts)
                               + (icol * self.prod_desc.parts + iprt))
                    self._buffer.jump_to(self._start, _word_to_position(pointer))
                    self.data_ptr = self._buffer.read_int(4, self.endian, False)
                    if not self.data_ptr:
                        continue
                    self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                    self.data_header_length = self._buffer.read_int(4, self.endian, False)
                    data_header = self._buffer.set_mark()
                    self._buffer.jump_to(data_header,
                                         _word_to_position(part.header_length + 1))
                    lendat = self.data_header_length - part.header_length

                    fmt_code = {
                        DataTypes.real: 'f',
                        DataTypes.realpack: 'i',
                        DataTypes.character: 's',
                    }.get(part.data_type)

                    if fmt_code is None:
                        raise NotImplementedError('No methods for data type {}'
                                                  .format(part.data_type))

                    if fmt_code == 's':
                        lendat *= BYTES_PER_WORD

                    packed_buffer = (
                        self._buffer.read_struct(
                            struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                        )
                    )

                    parameters = self.parameters[iprt]
                    nparms = len(parameters['name'])
                    sounding[part.name] = {}

                    if part.data_type == DataTypes.realpack:
                        unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                        for iprm, param in enumerate(parameters['name']):
                            sounding[part.name][param] = unpacked[iprm::nparms]
                    elif part.data_type == DataTypes.character:
                        for iprm, param in enumerate(parameters['name']):
                            sounding[part.name][param] = (
                                self._decode_strip(packed_buffer[iprm])
                            )
                    else:
                        for iprm, param in enumerate(parameters['name']):
                            sounding[part.name][param] = (
                                np.array(packed_buffer[iprm::nparms], dtype=np.float32)
                            )

                soundings.append(self._merge_sounding(sounding))
        return soundings

    def _merge_significant_temps(self, merged, parts, section, pbot):
        """Process and merge a significant temperature sections."""
        for isigt, press in enumerate(parts[section]['PRES']):
            press = abs(press)
            if self.prod_desc.missing_float not in [
                press,
                parts[section]['TEMP'][isigt]
            ] and press != 0:
                if press > pbot:
                    continue
                elif press in merged['PRES']:
                    ploc = merged['PRES'].index(press)
                    if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                        merged['TEMP'][ploc] = parts[section]['TEMP'][isigt]
                        merged['DWPT'][ploc] = parts[section]['DWPT'][isigt]
                else:
                    size = len(merged['PRES'])
                    loc = size - bisect.bisect_left(merged['PRES'][::-1], press)
                    merged['PRES'].insert(loc, press)
                    merged['TEMP'].insert(loc, parts[section]['TEMP'][isigt])
                    merged['DWPT'].insert(loc, parts[section]['DWPT'][isigt])
                    merged['DRCT'].insert(loc, self.prod_desc.missing_float)
                    merged['SPED'].insert(loc, self.prod_desc.missing_float)
                    merged['HGHT'].insert(loc, self.prod_desc.missing_float)
            pbot = press

        return pbot

    def _merge_tropopause_data(self, merged, parts, section, pbot):
        """Process and merge tropopause sections."""
        for itrp, press in enumerate(parts[section]['PRES']):
            press = abs(press)
            if self.prod_desc.missing_float not in [
                press,
                parts[section]['TEMP'][itrp]
            ] and press != 0:
                if press > pbot:
                    continue
                elif press in merged['PRES']:
                    ploc = merged['PRES'].index(press)
                    if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                        merged['TEMP'][ploc] = parts[section]['TEMP'][itrp]
                        merged['DWPT'][ploc] = parts[section]['DWPT'][itrp]
                    if merged['DRCT'][ploc] == self.prod_desc.missing_float:
                        merged['DRCT'][ploc] = parts[section]['DRCT'][itrp]
                        merged['SPED'][ploc] = parts[section]['SPED'][itrp]
                    merged['HGHT'][ploc] = self.prod_desc.missing_float
                else:
                    size = len(merged['PRES'])
                    loc = size - bisect.bisect_left(merged['PRES'][::-1], press)
                    merged['PRES'].insert(loc, press)
                    merged['TEMP'].insert(loc, parts[section]['TEMP'][itrp])
                    merged['DWPT'].insert(loc, parts[section]['DWPT'][itrp])
                    merged['DRCT'].insert(loc, parts[section]['DRCT'][itrp])
                    merged['SPED'].insert(loc, parts[section]['SPED'][itrp])
                    merged['HGHT'].insert(loc, self.prod_desc.missing_float)
            pbot = press

        return pbot

    def _merge_mandatory_temps(self, merged, parts, section, qcman, bgl, plast):
        """Process and merge mandatory temperature sections."""
        num_levels = len(parts[section]['PRES'])
        start_level = {
            'TTAA': 1,
            'TTCC': 0,
        }
        for i in range(start_level[section], num_levels):
            if (parts[section]['PRES'][i] < plast
                and self.prod_desc.missing_float not in [
                    parts[section]['PRES'][i],
                    parts[section]['TEMP'][i],
                    parts[section]['HGHT'][i]
            ]):
                for pname, pval in parts[section].items():
                    merged[pname].append(pval[i])
                plast = merged['PRES'][-1]
            else:
                if section == 'TTAA':
                    if parts[section]['PRES'][i] > merged['PRES'][0]:
                        bgl += 1
                    else:
                        # GEMPAK ignores MAN data with missing TEMP/HGHT and does not
                        # interpolate for them.
                        if parts[section]['PRES'][i] != self.prod_desc.missing_float:
                            qcman.append(parts[section]['PRES'][i])

        return bgl, plast

    def _merge_mandatory_winds(self, merged, parts, section, qcman):
        """Process and merge manadatory wind sections."""
        for iwind, press in enumerate(parts[section]['PRES']):
            if press in merged['PRES'][1:]:
                loc = merged['PRES'].index(press)
                if merged['DRCT'][loc] == self.prod_desc.missing_float:
                    merged['DRCT'][loc] = parts[section]['DRCT'][iwind]
                    merged['SPED'][loc] = parts[section]['SPED'][iwind]
            else:
                if press not in qcman:
                    size = len(merged['PRES'])
                    loc = size - bisect.bisect_left(merged['PRES'][1:][::-1], press)
                    if loc >= size + 1:
                        loc = -1
                    merged['PRES'].insert(loc, press)
                    merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                    merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                    merged['DRCT'].insert(loc, parts[section]['DRCT'][iwind])
                    merged['SPED'].insert(loc, parts[section]['SPED'][iwind])
                    merged['HGHT'].insert(loc, self.prod_desc.missing_float)

    def _merge_winds_pressure(self, merged, parts, section, pbot):
        """Process and merge wind sections on pressure surfaces."""
        for ilevel, press in enumerate(parts[section]['PRES']):
            press = abs(press)
            if self.prod_desc.missing_float not in [
                press,
                parts[section]['DRCT'][ilevel],
                parts[section]['SPED'][ilevel]
            ] and press != 0:
                if press > pbot:
                    continue
                elif press in merged['PRES']:
                    ploc = merged['PRES'].index(press)
                    if self.prod_desc.missing_float in [
                        merged['DRCT'][ploc],
                        merged['SPED'][ploc]
                    ]:
                        merged['DRCT'][ploc] = parts[section]['DRCT'][ilevel]
                        merged['SPED'][ploc] = parts[section]['SPED'][ilevel]
                else:
                    size = len(merged['PRES'])
                    loc = size - bisect.bisect_left(merged['PRES'][::-1], press)
                    merged['PRES'].insert(loc, press)
                    merged['DRCT'].insert(loc, parts[section]['DRCT'][ilevel])
                    merged['SPED'].insert(loc, parts[section]['SPED'][ilevel])
                    merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                    merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                    merged['HGHT'].insert(loc, self.prod_desc.missing_float)
            pbot = press

        return pbot

    def _merge_winds_height(self, merged, parts, nsgw, nasw, istart):
        """Merge wind sections on height surfaces."""
        size = len(merged['HGHT'])
        psfc = merged['PRES'][0]
        zsfc = merged['HGHT'][0]

        if self.prod_desc.missing_float not in [
            psfc,
            zsfc
        ] and size >= 2:
            more = True
            zold = merged['HGHT'][0]
            znxt = merged['HGHT'][1]
            ilev = 1
        elif size >= 3:
            more = True
            zold = merged['HGHT'][1]
            znxt = merged['HGHT'][2]
            ilev = 2
        else:
            zold = self.prod_desc.missing_float
            znxt = self.prod_desc.missing_float

        if self.prod_desc.missing_float in [
            zold,
            znxt
        ]:
            more = False

        if istart <= nsgw:
            above = False
            i = istart
            iend = nsgw
        else:
            above = True
            i = 0
            iend = nasw

        while more and i < iend:
            if not above:
                hght = parts['PPBB']['HGHT'][i]
                drct = parts['PPBB']['DRCT'][i]
                sped = parts['PPBB']['SPED'][i]
            else:
                hght = parts['PPDD']['HGHT'][i]
                drct = parts['PPDD']['DRCT'][i]
                sped = parts['PPDD']['SPED'][i]
            skip = False

            if self.prod_desc.missing_float in [
                hght,
                drct,
                sped
            ] or hght <= zold:
                skip = True
            elif abs(zold - hght) < 1:
                skip = True
                if self.prod_desc.missing_float in [
                    merged['DRCT'][ilev - 1],
                    merged['SPED'][ilev - 1]
                ]:
                    merged['DRCT'][ilev - 1] = drct
                    merged['SPED'][ilev - 1] = sped
            elif hght >= znxt:
                while more and hght > znxt:
                    zold = znxt
                    ilev += 1
                    if ilev >= size:
                        more = False
                    else:
                        znxt = merged['HGHT'][ilev]
                        if znxt == self.prod_desc.missing_float:
                            more = False

            if more and not skip:
                if abs(znxt - hght) < 1:
                    if self.prod_desc.missing_float in [
                        merged['DRCT'][ilev - 1],
                        merged['SPED'][ilev - 1]
                    ]:
                        merged['DRCT'][ilev] = drct
                        merged['SPED'][ilev] = sped
                else:
                    loc = bisect.bisect_left(merged['HGHT'], hght)
                    merged['HGHT'].insert(loc, hght)
                    merged['DRCT'].insert(loc, drct)
                    merged['SPED'].insert(loc, sped)
                    merged['PRES'].insert(loc, self.prod_desc.missing_float)
                    merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                    merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                    size += 1
                    ilev += 1
                    zold = hght

            if not above and i == nsgw - 1:
                above = True
                i = 0
                iend = nasw
            else:
                i += 1

    def _merge_sounding(self, parts):
        """Merge unmerged sounding data."""
        merged = {'STID': parts['STID'],
                  'STNM': parts['STNM'],
                  'SLAT': parts['SLAT'],
                  'SLON': parts['SLON'],
                  'SELV': parts['SELV'],
                  'STAT': parts['STAT'],
                  'COUN': parts['COUN'],
                  'DATE': parts['DATE'],
                  'TIME': parts['TIME'],
                  'PRES': [],
                  'HGHT': [],
                  'TEMP': [],
                  'DWPT': [],
                  'DRCT': [],
                  'SPED': [],
                  }

        # Number of parameter levels
        num_man_levels = len(parts['TTAA']['PRES']) if 'TTAA' in parts else 0
        num_man_wind_levels = len(parts['PPAA']['PRES']) if 'PPAA' in parts else 0
        num_trop_levels = len(parts['TRPA']['PRES']) if 'TRPA' in parts else 0
        num_max_wind_levels = len(parts['MXWA']['PRES']) if 'MXWA' in parts else 0
        num_sigt_levels = len(parts['TTBB']['PRES']) if 'TTBB' in parts else 0
        num_sigw_levels = len(parts['PPBB']['SPED']) if 'PPBB' in parts else 0
        num_above_man_levels = len(parts['TTCC']['PRES']) if 'TTCC' in parts else 0
        num_above_trop_levels = len(parts['TRPC']['PRES']) if 'TRPC' in parts else 0
        num_above_max_wind_levels = len(parts['MXWC']['SPED']) if 'MXWC' in parts else 0
        num_above_sigt_levels = len(parts['TTDD']['PRES']) if 'TTDD' in parts else 0
        num_above_sigw_levels = len(parts['PPDD']['SPED']) if 'PPDD' in parts else 0
        num_above_man_wind_levels = len(parts['PPCC']['SPED']) if 'PPCC' in parts else 0

        total_data = (num_man_levels
                      + num_man_wind_levels
                      + num_trop_levels
                      + num_max_wind_levels
                      + num_sigt_levels
                      + num_sigw_levels
                      + num_above_man_levels
                      + num_above_trop_levels
                      + num_above_max_wind_levels
                      + num_above_sigt_levels
                      + num_above_sigw_levels
                      + num_above_man_wind_levels
                      )
        if total_data == 0:
            return None

        # Check SIG wind vertical coordinate
        # For some reason, the pressure data can get put into the
        # height array. Perhaps this is just a artifact of Python,
        # as GEMPAK itself just uses array indices without any
        # names involved. Since the first valid pressure of the
        # array will be negative in the case of pressure coordinates,
        # we can check for it and place data in the appropriate array.
        ppbb_is_z = True
        if num_sigw_levels:
            if 'PRES' in parts['PPBB']:
                ppbb_is_z = False
            else:
                for z in parts['PPBB']['HGHT']:
                    if z != self.prod_desc.missing_float and z < 0:
                        ppbb_is_z = False
                        parts['PPBB']['PRES'] = parts['PPBB']['HGHT']
                        break

        ppdd_is_z = True
        if num_above_sigw_levels:
            if 'PRES' in parts['PPDD']:
                ppdd_is_z = False
            else:
                for z in parts['PPDD']['HGHT']:
                    if z != self.prod_desc.missing_float and z < 0:
                        ppdd_is_z = False
                        parts['PPDD']['PRES'] = parts['PPDD']['HGHT']
                        break

        # Process surface data
        if num_man_levels < 1:
            merged['PRES'].append(self.prod_desc.missing_float)
            merged['HGHT'].append(self.prod_desc.missing_float)
            merged['TEMP'].append(self.prod_desc.missing_float)
            merged['DWPT'].append(self.prod_desc.missing_float)
            merged['DRCT'].append(self.prod_desc.missing_float)
            merged['SPED'].append(self.prod_desc.missing_float)
        else:
            merged['PRES'].append(parts['TTAA']['PRES'][0])
            merged['HGHT'].append(parts['TTAA']['HGHT'][0])
            merged['TEMP'].append(parts['TTAA']['TEMP'][0])
            merged['DWPT'].append(parts['TTAA']['DWPT'][0])
            merged['DRCT'].append(parts['TTAA']['DRCT'][0])
            merged['SPED'].append(parts['TTAA']['SPED'][0])

        merged['HGHT'][0] = merged['SELV']

        first_man_p = self.prod_desc.missing_float
        if num_man_levels >= 1:
            for mp, mt, mz in zip(parts['TTAA']['PRES'],
                                  parts['TTAA']['TEMP'],
                                  parts['TTAA']['HGHT']):
                if self.prod_desc.missing_float not in [
                    mp,
                    mt,
                    mz
                ]:
                    first_man_p = mp
                    break

        surface_p = merged['PRES'][0]
        if surface_p > 1060:
            surface_p = self.prod_desc.missing_float

        if (surface_p == self.prod_desc.missing_float
           or (surface_p < first_man_p
               and surface_p != self.prod_desc.missing_float)):
            merged['PRES'][0] = self.prod_desc.missing_float
            merged['HGHT'][0] = self.prod_desc.missing_float
            merged['TEMP'][0] = self.prod_desc.missing_float
            merged['DWPT'][0] = self.prod_desc.missing_float
            merged['DRCT'][0] = self.prod_desc.missing_float
            merged['SPED'][0] = self.prod_desc.missing_float

        if (num_sigt_levels >= 1
           and self.prod_desc.missing_float not in [
               parts['TTBB']['PRES'][0],
               parts['TTBB']['TEMP'][0]
           ]):
            first_man_p = merged['PRES'][0]
            first_sig_p = parts['TTBB']['PRES'][0]
            if (first_man_p == self.prod_desc.missing_float
               or np.isclose(first_man_p, first_sig_p)):
                merged['PRES'][0] = parts['TTBB']['PRES'][0]
                merged['DWPT'][0] = parts['TTBB']['DWPT'][0]
                merged['TEMP'][0] = parts['TTBB']['TEMP'][0]

        if num_sigw_levels >= 1:
            if ppbb_is_z:
                if (parts['PPBB']['HGHT'][0] == 0
                   and parts['PPBB']['DRCT'][0] != self.prod_desc.missing_float):
                    merged['DRCT'][0] = parts['PPBB']['DRCT'][0]
                    merged['SPED'][0] = parts['PPBB']['SPED'][0]
            else:
                if self.prod_desc.missing_float not in [
                    parts['PPBB']['PRES'][0],
                    parts['PPBB']['DRCT'][0]
                ]:
                    first_man_p = merged['PRES'][0]
                    first_sig_p = abs(parts['PPBB']['PRES'][0])
                    if (first_man_p == self.prod_desc.missing_float
                       or np.isclose(first_man_p, first_sig_p)):
                        merged['PRES'][0] = abs(parts['PPBB']['PRES'][0])
                        merged['DRCT'][0] = parts['PPBB']['DRCT'][0]
                        merged['SPED'][0] = parts['PPBB']['SPED'][0]

        # Merge MAN temperature
        bgl = 0
        qcman = []
        if num_man_levels >= 2 or num_above_man_levels >= 1:
            if merged['PRES'][0] == self.prod_desc.missing_float:
                plast = 2000
            else:
                plast = merged['PRES'][0]

        if num_man_levels >= 2:
            bgl, plast = self._merge_mandatory_temps(merged, parts, 'TTAA',
                                                     qcman, bgl, plast)

        if num_above_man_levels >= 1:
            bgl, plast = self._merge_mandatory_temps(merged, parts, 'TTCC',
                                                     qcman, bgl, plast)

        # Merge MAN wind
        if num_man_wind_levels >= 1 and num_man_levels >= 1 and len(merged['PRES']) >= 2:
            self._merge_mandatory_winds(merged, parts, 'PPAA', qcman)

        if num_above_man_wind_levels >= 1 and num_man_levels >= 1 and len(merged['PRES']) >= 2:
            self._merge_mandatory_winds(merged, parts, 'PPCC', qcman)

        # Merge TROP
        if num_trop_levels >= 1 or num_above_trop_levels >= 1:
            if merged['PRES'][0] != self.prod_desc.missing_float:
                pbot = merged['PRES'][0]
            elif len(merged['PRES']) > 1:
                pbot = merged['PRES'][1]
                if pbot < parts['TRPA']['PRES'][1]:
                    pbot = 1050
            else:
                pbot = 1050

        if num_trop_levels >= 1:
            pbot = self._merge_tropopause_data(merged, parts, 'TRPA', pbot)

        if num_above_trop_levels >= 1:
            pbot = self._merge_tropopause_data(merged, parts, 'TRPC', pbot)

        # Merge SIG temperature
        if num_sigt_levels >= 1 or num_above_sigt_levels >= 1:
            if merged['PRES'][0] != self.prod_desc.missing_float:
                pbot = merged['PRES'][0]
            elif len(merged['PRES']) > 1:
                pbot = merged['PRES'][1]
                if pbot < parts['TTBB']['PRES'][1]:
                    pbot = 1050
            else:
                pbot = 1050

        if num_sigt_levels >= 1:
            pbot = self._merge_significant_temps(merged, parts, 'TTBB', pbot)

        if num_above_sigt_levels >= 1:
            pbot = self._merge_significant_temps(merged, parts, 'TTDD', pbot)

        # Interpolate heights
        _interp_moist_height(merged, self.prod_desc.missing_float)

        # Merge SIG winds on pressure surfaces
        if not ppbb_is_z or not ppdd_is_z:
            if num_sigw_levels >= 1 or num_above_sigw_levels >= 1:
                if merged['PRES'][0] != self.prod_desc.missing_float:
                    pbot = merged['PRES'][0]
                elif len(merged['PRES']) > 1:
                    pbot = merged['PRES'][1]
                else:
                    pbot = 0

            if num_sigw_levels >= 1 and not ppbb_is_z:
                pbot = self._merge_winds_pressure(merged, parts, 'PPBB', pbot)

            if num_above_sigw_levels >= 1 and not ppdd_is_z:
                pbot = self._merge_winds_pressure(merged, parts, 'PPDD', pbot)

        # Merge max winds on pressure surfaces
        if num_max_wind_levels >= 1 or num_above_max_wind_levels >= 1:
            if merged['PRES'][0] != self.prod_desc.missing_float:
                pbot = merged['PRES'][0]
            elif len(merged['PRES']) > 1:
                pbot = merged['PRES'][1]
            else:
                pbot = 0

        if num_max_wind_levels >= 1:
            pbot = self._merge_winds_pressure(merged, parts, 'MXWA', pbot)

        if num_above_max_wind_levels >= 1:
            _ = self._merge_winds_pressure(merged, parts, 'MXWC', pbot)

        # Interpolate height for SIG/MAX winds
        _interp_logp_height(merged, self.prod_desc.missing_float)

        # Merge SIG winds on height surfaces
        if ppbb_is_z or ppdd_is_z:
            nsgw = num_sigw_levels if ppbb_is_z else 0
            nasw = num_above_sigw_levels if ppdd_is_z else 0
            if (nsgw >= 1 and (parts['PPBB']['HGHT'][0] == 0
               or parts['PPBB']['HGHT'][0] == merged['HGHT'][0])):
                istart = 1
            else:
                istart = 0

            self._merge_winds_height(merged, parts, nsgw, nasw, istart)

            # Interpolate missing pressure with height
            _interp_logp_pressure(merged, self.prod_desc.missing_float)

        # Interpolate missing data
        _interp_logp_data(merged, self.prod_desc.missing_float)

        # Add below ground MAN data
        if merged['PRES'][0] != self.prod_desc.missing_float and bgl > 0:
            size = len(merged['PRES'])
            for ibgl in range(1, num_man_levels):
                press = parts['TTAA']['PRES'][ibgl]
                if press > merged['PRES'][0]:
                    loc = size - bisect.bisect_left(merged['PRES'][1:][::-1], press)
                    merged['PRES'].insert(loc, press)
                    merged['TEMP'].insert(loc, parts['TTAA']['TEMP'][ibgl])
                    merged['DWPT'].insert(loc, parts['TTAA']['DWPT'][ibgl])
                    merged['DRCT'].insert(loc, parts['TTAA']['DRCT'][ibgl])
                    merged['SPED'].insert(loc, parts['TTAA']['SPED'][ibgl])
                    merged['HGHT'].insert(loc, parts['TTAA']['HGHT'][ibgl])
                    size += 1

        # Add text data, if it is included
        if 'TXTA' in parts:
            merged['TXTA'] = parts['TXTA']['TEXT']
        if 'TXTB' in parts:
            merged['TXTB'] = parts['TXTB']['TEXT']
        if 'TXTC' in parts:
            merged['TXTC'] = parts['TXTC']['TEXT']
        if 'TXPB' in parts:
            merged['TXPB'] = parts['TXPB']['TEXT']

        return merged

    def snxarray(self, station_id=None, station_number=None,
                 date_time=None, state=None, country=None):
        """Select soundings and output as list of xarray Datasets.

        Subset the data by parameter values. The default is to not
        subset and return the entire dataset.

        Parameters
        ----------
        station_id : str or Sequence[str]
            Station ID of sounding site.

        station_number : int or Sequence[int]
            Station number of sounding site.

        date_time : `~datetime.datetime` or Sequence[datetime]
            Valid datetime of the grid. Alternatively can be
            a string with the format YYYYmmddHHMM.

        state : str or Sequence[str]
            State where sounding site is located.

        country : str or Sequence[str]
            Country where sounding site is located.

        Returns
        -------
        list[xarray.Dataset]
            List of `xarray.Dataset` objects for each sounding.
        """
        if station_id is not None:
            if (not isinstance(station_id, Iterable)
               or isinstance(station_id, str)):
                station_id = [station_id]
            station_id = [c.upper() for c in station_id]

        if station_number is not None:
            if not isinstance(station_number, Iterable):
                station_number = [station_number]
            station_number = [int(sn) for sn in station_number]

        if date_time is not None:
            if (not isinstance(date_time, Iterable)
               or isinstance(date_time, str)):
                date_time = [date_time]
            for i, dt in enumerate(date_time):
                if isinstance(dt, str):
                    date_time[i] = datetime.strptime(dt, '%Y%m%d%H%M')

        if (state is not None
           and (not isinstance(state, Iterable)
                or isinstance(state, str))):
            state = [state]
            state = [s.upper() for s in state]

        if (country is not None
           and (not isinstance(country, Iterable)
                or isinstance(country, str))):
            country = [country]
            country = [c.upper() for c in country]

        # Figure out which columns to extract from the file
        matched = self._sninfo.copy()

        if station_id is not None:
            matched = filter(
                lambda snd: snd if snd.ID in station_id else False,
                matched
            )

        if station_number is not None:
            matched = filter(
                lambda snd: snd if snd.NUMBER in station_number else False,
                matched
            )

        if date_time is not None:
            matched = filter(
                lambda snd: snd if snd.DATTIM in date_time else False,
                matched
            )

        if state is not None:
            matched = filter(
                lambda snd: snd if snd.STATE in state else False,
                matched
            )

        if country is not None:
            matched = filter(
                lambda snd: snd if snd.COUNTRY in country else False,
                matched
            )

        matched = list(matched)

        if len(matched) < 1:
            raise KeyError('No stations were matched with given parameters.')

        sndno = [(s.DTNO, s.SNDNO) for s in matched]

        if self.merged:
            data = self._unpack_merged(sndno)
        else:
            data = self._unpack_unmerged(sndno)

        soundings = []
        for snd in data:
            if snd is None or 'PRES' not in snd:
                continue
            station_pressure = snd['PRES'][0]
            wmo_text = {}
            attrs = {
                'station_id': snd.pop('STID'),
                'station_number': snd.pop('STNM'),
                'lat': snd.pop('SLAT'),
                'lon': snd.pop('SLON'),
                'elevation': snd.pop('SELV'),
                'station_pressure': station_pressure,
                'state': snd.pop('STAT'),
                'country': snd.pop('COUN'),
            }

            if 'TXTA' in snd:
                wmo_text['txta'] = snd.pop('TXTA')
            if 'TXTB' in snd:
                wmo_text['txtb'] = snd.pop('TXTB')
            if 'TXTC' in snd:
                wmo_text['txtc'] = snd.pop('TXTC')
            if 'TXPB' in snd:
                wmo_text['txpb'] = snd.pop('TXPB')
            if wmo_text:
                attrs['WMO_CODES'] = wmo_text

            dt = datetime.combine(snd.pop('DATE'), snd.pop('TIME'))
            press = np.array(snd.pop('PRES'))

            var = {}
            for param, values in snd.items():
                values = np.array(values)[np.newaxis, ...]
                maskval = np.ma.array(values, mask=values == self.prod_desc.missing_float,
                                      dtype=np.float32)
                var[param.lower()] = (['time', 'pressure'], maskval)

            xrds = xr.Dataset(var,
                              coords={'time': np.atleast_1d(dt), 'pressure': press},
                              attrs=attrs)

            # Sort to fix GEMPAK surface data at first level
            xrds = xrds.sortby('pressure', ascending=False)

            soundings.append(xrds)
        return soundings


@exporter.export
class GempakSurface(GempakFile):
    """Subclass of GempakFile specific to GEMPAK surface data."""

    def __init__(self, file, *args, **kwargs):
        super().__init__(file)

        # Row Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.row_headers_ptr))
        self.row_headers = []
        row_headers_info = self._key_types(self.row_keys)
        row_headers_info.extend([(None, None)])
        row_headers_fmt = NamedStruct(row_headers_info, self.prefmt, 'RowHeaders')
        for _ in range(1, self.prod_desc.rows + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.row_headers.append(self._buffer.read_struct(row_headers_fmt))

        # Column Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.column_headers_ptr))
        self.column_headers = []
        column_headers_info = self._key_types(self.column_keys)
        column_headers_info.extend([(None, None)])
        column_headers_fmt = NamedStruct(column_headers_info, self.prefmt, 'ColumnHeaders')
        for _ in range(1, self.prod_desc.columns + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.column_headers.append(self._buffer.read_struct(column_headers_fmt))

        self._get_surface_type()

        self._sfinfo = []
        if self.surface_type == 'standard':
            for irow, row_head in enumerate(self.row_headers):
                for icol, col_head in enumerate(self.column_headers):
                    for iprt in range(len(self.parts)):
                        pointer = (self.prod_desc.data_block_ptr
                                   + (irow * self.prod_desc.columns * self.prod_desc.parts)
                                   + (icol * self.prod_desc.parts + iprt))

                        self._buffer.jump_to(self._start, _word_to_position(pointer))
                        data_ptr = self._buffer.read_int(4, self.endian, False)

                        if data_ptr:
                            self._sfinfo.append(
                                Surface(
                                    irow,
                                    icol,
                                    datetime.combine(row_head.DATE, row_head.TIME),
                                    col_head.STID + col_head.STD2,
                                    col_head.STNM,
                                    col_head.SLAT,
                                    col_head.SLON,
                                    col_head.SELV,
                                    col_head.STAT,
                                    col_head.COUN,
                                )
                            )
        elif self.surface_type == 'ship':
            irow = 0
            for icol, col_head in enumerate(self.column_headers):
                for iprt in range(len(self.parts)):
                    pointer = (self.prod_desc.data_block_ptr
                               + (irow * self.prod_desc.columns * self.prod_desc.parts)
                               + (icol * self.prod_desc.parts + iprt))

                    self._buffer.jump_to(self._start, _word_to_position(pointer))
                    data_ptr = self._buffer.read_int(4, self.endian, False)

                    if data_ptr:
                        self._sfinfo.append(
                            Surface(
                                irow,
                                icol,
                                datetime.combine(col_head.DATE, col_head.TIME),
                                col_head.STID + col_head.STD2,
                                col_head.STNM,
                                col_head.SLAT,
                                col_head.SLON,
                                col_head.SELV,
                                col_head.STAT,
                                col_head.COUN,
                            )
                        )
        elif self.surface_type == 'climate':
            for icol, col_head in enumerate(self.column_headers):
                for irow, row_head in enumerate(self.row_headers):
                    for iprt in range(len(self.parts)):
                        pointer = (self.prod_desc.data_block_ptr
                                   + (irow * self.prod_desc.columns * self.prod_desc.parts)
                                   + (icol * self.prod_desc.parts + iprt))

                        self._buffer.jump_to(self._start, _word_to_position(pointer))
                        data_ptr = self._buffer.read_int(4, self.endian, False)

                        if data_ptr:
                            self._sfinfo.append(
                                Surface(
                                    irow,
                                    icol,
                                    datetime.combine(col_head.DATE, col_head.TIME),
                                    row_head.STID + row_head.STD2,
                                    row_head.STNM,
                                    row_head.SLAT,
                                    row_head.SLON,
                                    row_head.SELV,
                                    row_head.STAT,
                                    row_head.COUN,
                                )
                            )
        else:
            raise TypeError('Unknown surface type {}'.format(self.surface_type))

    def sfinfo(self):
        """Return station information."""
        return self._sfinfo

    def _get_surface_type(self):
        """Determine type of surface file."""
        if len(self.row_headers) == 1:
            self.surface_type = 'ship'
        elif 'DATE' in self.row_keys:
            self.surface_type = 'standard'
        elif 'DATE' in self.column_keys:
            self.surface_type = 'climate'
        else:
            raise TypeError('Unknown surface data type')

    def _key_types(self, keys):
        """Determine header information from a set of keys."""
        return [(key, '4s', self._decode_strip) if key == 'STID'
                else (key, 'i') if key == 'STNM'
                else (key, 'i', lambda x: x / 100) if key == 'SLAT'
                else (key, 'i', lambda x: x / 100) if key == 'SLON'
                else (key, 'i') if key == 'SELV'
                else (key, '4s', self._decode_strip) if key == 'STAT'
                else (key, '4s', self._decode_strip) if key == 'COUN'
                else (key, '4s', self._decode_strip) if key == 'STD2'
                else (key, 'i', self._make_date) if key == 'DATE'
                else (key, 'i', self._make_time) if key == 'TIME'
                else (key, 'i')
                for key in keys]

    def _unpack_climate(self, sfcno):
        """Unpack a climate surface data file."""
        stations = []
        for icol, col_head in enumerate(self.column_headers):
            for irow, row_head in enumerate(self.row_headers):
                if (irow, icol) not in sfcno:
                    continue
                station = {'STID': row_head.STID,
                           'STNM': row_head.STNM,
                           'SLAT': row_head.SLAT,
                           'SLON': row_head.SLON,
                           'SELV': row_head.SELV,
                           'STAT': row_head.STAT,
                           'COUN': row_head.COUN,
                           'STD2': row_head.STD2,
                           'SPRI': row_head.SPRI,
                           'DATE': col_head.DATE,
                           'TIME': col_head.TIME,
                           }
                for iprt, part in enumerate(self.parts):
                    pointer = (self.prod_desc.data_block_ptr
                               + (irow * self.prod_desc.columns * self.prod_desc.parts)
                               + (icol * self.prod_desc.parts + iprt))
                    self._buffer.jump_to(self._start, _word_to_position(pointer))
                    self.data_ptr = self._buffer.read_int(4, self.endian, False)
                    if not self.data_ptr:
                        continue
                    self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                    self.data_header_length = self._buffer.read_int(4, self.endian, False)
                    data_header = self._buffer.set_mark()
                    self._buffer.jump_to(data_header,
                                         _word_to_position(part.header_length + 1))
                    lendat = self.data_header_length - part.header_length

                    fmt_code = {
                        DataTypes.real: 'f',
                        DataTypes.realpack: 'i',
                        DataTypes.character: 's',
                    }.get(part.data_type)

                    if fmt_code is None:
                        raise NotImplementedError('No methods for data type {}'
                                                  .format(part.data_type))

                    if fmt_code == 's':
                        lendat *= BYTES_PER_WORD

                    packed_buffer = (
                        self._buffer.read_struct(
                            struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                        )
                    )

                    parameters = self.parameters[iprt]

                    if part.data_type == DataTypes.realpack:
                        unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = unpacked[iprm]
                    elif part.data_type == DataTypes.character:
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = self._decode_strip(packed_buffer[iprm])
                    else:
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = np.array(
                                packed_buffer[iprm], dtype=np.float32
                            )

                stations.append(station)
        return stations

    def _unpack_ship(self, sfcno):
        """Unpack ship (moving observation) surface data file."""
        stations = []
        irow = 0
        for icol, col_head in enumerate(self.column_headers):
            if (irow, icol) not in sfcno:
                continue
            station = {'STID': col_head.STID,
                       'STNM': col_head.STNM,
                       'SLAT': col_head.SLAT,
                       'SLON': col_head.SLON,
                       'SELV': col_head.SELV,
                       'STAT': col_head.STAT,
                       'COUN': col_head.COUN,
                       'STD2': col_head.STD2,
                       'SPRI': col_head.SPRI,
                       'DATE': col_head.DATE,
                       'TIME': col_head.TIME,
                       }
            for iprt, part in enumerate(self.parts):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts + iprt))
                self._buffer.jump_to(self._start, _word_to_position(pointer))
                self.data_ptr = self._buffer.read_int(4, self.endian, False)
                if not self.data_ptr:
                    continue
                self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                self.data_header_length = self._buffer.read_int(4, self.endian, False)
                data_header = self._buffer.set_mark()
                self._buffer.jump_to(data_header,
                                     _word_to_position(part.header_length + 1))
                lendat = self.data_header_length - part.header_length

                fmt_code = {
                    DataTypes.real: 'f',
                    DataTypes.realpack: 'i',
                    DataTypes.character: 's',
                }.get(part.data_type)

                if fmt_code is None:
                    raise NotImplementedError('No methods for data type {}'
                                              .format(part.data_type))

                if fmt_code == 's':
                    lendat *= BYTES_PER_WORD

                packed_buffer = (
                    self._buffer.read_struct(
                        struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                    )
                )

                parameters = self.parameters[iprt]

                if part.data_type == DataTypes.realpack:
                    unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = unpacked[iprm]
                elif part.data_type == DataTypes.character:
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = self._decode_strip(packed_buffer[iprm])
                else:
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = np.array(
                            packed_buffer[iprm], dtype=np.float32
                        )

            stations.append(station)
        return stations

    def _unpack_standard(self, sfcno):
        """Unpack a standard surface data file."""
        stations = []
        for irow, row_head in enumerate(self.row_headers):
            for icol, col_head in enumerate(self.column_headers):
                if (irow, icol) not in sfcno:
                    continue
                station = {'STID': col_head.STID,
                           'STNM': col_head.STNM,
                           'SLAT': col_head.SLAT,
                           'SLON': col_head.SLON,
                           'SELV': col_head.SELV,
                           'STAT': col_head.STAT,
                           'COUN': col_head.COUN,
                           'STD2': col_head.STD2,
                           'SPRI': col_head.SPRI,
                           'DATE': row_head.DATE,
                           'TIME': row_head.TIME,
                           }
                for iprt, part in enumerate(self.parts):
                    pointer = (self.prod_desc.data_block_ptr
                               + (irow * self.prod_desc.columns * self.prod_desc.parts)
                               + (icol * self.prod_desc.parts + iprt))
                    self._buffer.jump_to(self._start, _word_to_position(pointer))
                    self.data_ptr = self._buffer.read_int(4, self.endian, False)
                    if not self.data_ptr:
                        continue
                    self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                    self.data_header_length = self._buffer.read_int(4, self.endian, False)
                    data_header = self._buffer.set_mark()
                    self._buffer.jump_to(data_header,
                                         _word_to_position(part.header_length + 1))
                    lendat = self.data_header_length - part.header_length

                    fmt_code = {
                        DataTypes.real: 'f',
                        DataTypes.realpack: 'i',
                        DataTypes.character: 's',
                    }.get(part.data_type)

                    if fmt_code is None:
                        raise NotImplementedError('No methods for data type {}'
                                                  .format(part.data_type))

                    if fmt_code == 's':
                        lendat *= BYTES_PER_WORD

                    packed_buffer = (
                        self._buffer.read_struct(
                            struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                        )
                    )

                    parameters = self.parameters[iprt]

                    if part.data_type == DataTypes.realpack:
                        unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = unpacked[iprm]
                    elif part.data_type == DataTypes.character:
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = self._decode_strip(packed_buffer[iprm])
                    else:
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = packed_buffer[iprm]

                stations.append(station)
        return stations

    @staticmethod
    def _decode_special_observation(station, missing=-9999):
        """Decode raw special obsrvation text."""
        text = station['SPCL']
        dt = datetime.combine(station['DATE'], station['TIME'])
        parsed = parse_metar(text, dt.year, dt.month)

        station['TIME'] = parsed.date_time.time()
        if math.nan in [parsed.altimeter, parsed.elevation, parsed.temperature]:
            station['PMSL'] = missing
        else:
            station['PMSL'] = altimeter_to_sea_level_pressure(
                units.Quantity(parsed.altimeter, 'inHg'),
                units.Quantity(parsed.elevation, 'm'),
                units.Quantity(parsed.temperature, 'degC')
            ).to('hPa').m
        station['ALTI'] = _check_nan(parsed.altimeter, missing)
        station['TMPC'] = _check_nan(parsed.temperature, missing)
        station['DWPC'] = _check_nan(parsed.dewpoint, missing)
        station['SKNT'] = _check_nan(parsed.wind_speed, missing)
        station['DRCT'] = _check_nan(float(parsed.wind_direction), missing)
        station['GUST'] = _check_nan(parsed.wind_gust, missing)
        station['WNUM'] = float(_wx_to_wnum(parsed.current_wx1, parsed.current_wx2,
                                            parsed.current_wx3, missing))
        station['CHC1'] = _convert_clouds(parsed.skyc1, parsed.skylev1, missing)
        station['CHC2'] = _convert_clouds(parsed.skyc2, parsed.skylev2, missing)
        station['CHC3'] = _convert_clouds(parsed.skyc3, parsed.skylev3, missing)
        if math.isnan(parsed.visibility):
            station['VSBY'] = missing
        else:
            station['VSBY'] = float(round(parsed.visibility / 1609.344))

        return station

    def nearest_time(self, date_time, station_id=None, station_number=None,
                     include_special=False):
        """Get nearest observation to given time for selected stations.

        Parameters
        ----------
        date_time : `~datetime.datetime` or Sequence[datetime]
            Valid/observed datetime of the surface station. Alternatively
            object or a string with the format YYYYmmddHHMM.

        station_id : str or Sequence[str]
            Station ID of the surface station(s).

        station_number : int or Sequence[int]
            Station number of the surface station.

        include_special : bool
            If True, parse special observations that are stored
            as raw METAR text. Default is False.

        Returns
        -------
        list
            List of dicts/JSONs for each surface station.

        Notes
        -----
        One of either station_id or station_number must be used. If both
        are present, station_id will take precedence.
        """
        if isinstance(date_time, str):
            date_time = datetime.strptime(date_time, '%Y%m%d%H%M')

        if station_id is None and station_number is None:
            raise ValueError('Must have either station_id or station_number')

        if station_id is not None and station_number is not None:
            station_number = None

        if (station_id is not None
           and (not isinstance(station_id, Iterable)
                or isinstance(station_id, str))):
            station_id = [station_id]
            station_id = [c.upper() for c in station_id]

        if station_number is not None and not isinstance(station_number, Iterable):
            station_number = [station_number]
            station_number = [int(sn) for sn in station_number]

        time_matched = []
        if station_id:
            for stn in station_id:
                matched = self.sfjson(station_id=stn, include_special=include_special)

                nearest = min(
                    matched,
                    key=lambda d: abs(d['properties']['date_time'] - date_time)
                )

                time_matched.append(nearest)

        if station_number:
            for stn in station_id:
                matched = self.sfjson(station_number=stn, include_special=include_special)

                nearest = min(
                    matched,
                    key=lambda d: abs(d['properties']['date_time'] - date_time)
                )

                time_matched.append(nearest)

        return time_matched

    def sfjson(self, station_id=None, station_number=None,
               date_time=None, state=None, country=None,
               include_special=False):
        """Select surface stations and output as list of JSON objects.

        Subset the data by parameter values. The default is to not
        subset and return the entire dataset.

        Parameters
        ----------
        station_id : str or Sequence[str]
            Station ID of the surface station.

        station_number : int or Sequence[int]
            Station number of the surface station.

        date_time : `~datetime.datetime` or Sequence[datetime]
            Valid datetime of the grid. Alternatively can be
            a string with the format YYYYmmddHHMM.

        state : str or Sequence[str]
            State where surface station is located.

        country : str or Sequence[str]
            Country where surface station is located.

        include_special : bool
            If True, parse special observations that are stored
            as raw METAR text. Default is False.

        Returns
        -------
        list
            List of dicts/JSONs for each surface station.
        """
        if (station_id is not None
           and (not isinstance(station_id, Iterable)
                or isinstance(station_id, str))):
            station_id = [station_id]
            station_id = [c.upper() for c in station_id]

        if station_number is not None and not isinstance(station_number, Iterable):
            station_number = [station_number]
            station_number = [int(sn) for sn in station_number]

        if date_time is not None:
            if (not isinstance(date_time, Iterable)
               or isinstance(date_time, str)):
                date_time = [date_time]
            for i, dt in enumerate(date_time):
                if isinstance(dt, str):
                    date_time[i] = datetime.strptime(dt, '%Y%m%d%H%M')

        if (state is not None
           and (not isinstance(state, Iterable)
                or isinstance(state, str))):
            state = [state]
            state = [s.upper() for s in state]

        if (country is not None
           and (not isinstance(country, Iterable)
                or isinstance(country, str))):
            country = [country]
            country = [c.upper() for c in country]

        # Figure out which columns to extract from the file
        matched = self._sfinfo.copy()

        if station_id is not None:
            matched = filter(
                lambda sfc: sfc if sfc.ID in station_id else False,
                matched
            )

        if station_number is not None:
            matched = filter(
                lambda sfc: sfc if sfc.NUMBER in station_number else False,
                matched
            )

        if date_time is not None:
            matched = filter(
                lambda sfc: sfc if sfc.DATTIM in date_time else False,
                matched
            )

        if state is not None:
            matched = filter(
                lambda sfc: sfc if sfc.STATE in state else False,
                matched
            )

        if country is not None:
            matched = filter(
                lambda sfc: sfc if sfc.COUNTRY in country else False,
                matched
            )

        matched = list(matched)

        if len(matched) < 1:
            raise KeyError('No stations were matched with given parameters.')

        sfcno = [(s.ROW, s.COL) for s in matched]

        if self.surface_type == 'standard':
            data = self._unpack_standard(sfcno)
        elif self.surface_type == 'ship':
            data = self._unpack_ship(sfcno)
        elif self.surface_type == 'climate':
            data = self._unpack_climate(sfcno)
        else:
            raise ValueError(f'Unknown surface data type: {self.surface_type}')

        stnarr = []
        for stn in data:
            if 'SPCL' in stn and include_special:
                stn = self._decode_special_observation(stn, self.prod_desc.missing_float)
            props = {'date_time': datetime.combine(stn.pop('DATE'), stn.pop('TIME')),
                     'station_id': stn.pop('STID') + stn.pop('STD2'),
                     'station_number': stn.pop('STNM'),
                     'longitude': stn.pop('SLON'),
                     'latitude': stn.pop('SLAT'),
                     'elevation': stn.pop('SELV'),
                     'state': stn.pop('STAT'),
                     'country': stn.pop('COUN'),
                     'priority': stn.pop('SPRI')}
            if stn:
                vals = {name.lower(): ob for name, ob in stn.items()}
                stnarr.append({'properties': props, 'values': vals})

        return stnarr
