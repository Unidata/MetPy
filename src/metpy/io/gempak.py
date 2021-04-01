# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools to process GEMPAK-formatted products."""

import bisect
from collections import namedtuple
from collections.abc import Iterable
import contextlib
import ctypes
from datetime import datetime, timedelta
from enum import Enum
from itertools import product
import logging
import math
import struct
import sys

import numpy as np
import pyproj
import xarray as xr

from ._tools import IOBuffer, NamedStruct, open_as_needed
from ..calc import (hydrostatic_height, mixing_ratio_from_specific_humidity,
                    scale_height, specific_humidity_from_dewpoint,
                    virtual_temperature)
from ..package_tools import Exporter
from ..units import units

exporter = Exporter(globals())
log = logging.getLogger(__name__)

ANLB_SIZE = 128
BYTES_PER_WORD = 4
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
}
GVCORD_TO_VAR = {
    'PRES': 'p',
    'HGHT': 'z',
    'THTA': 'theta',
}
NAVB_SIZE = 256
PARAM_ATTR = [('name', (4, 's')), ('scale', (1, 'i')),
              ('offset', (1, 'i')), ('bits', (1, 'i'))]
USED_FLAG = 9999
UNUSED_FLAG = -9999


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
    """Veritical coordinates."""

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

Sounding = namedtuple('Soungind', [
    'DTNO',
    'SNDNO',
    'DATTIM',
    'ID',
    'NUMBER',
    'LAT',
    'LON',
    'ELEV',
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


def _word_to_position(word, bytes_per_word=BYTES_PER_WORD):
    """Return beginning position of a word in bytes."""
    return (word * bytes_per_word) - bytes_per_word


def _interp_logp_data(sounding, missing=-9999):
    """Interpolate missing data with respect to log p.

    This function is similar to the MR_MISS subroutine
    in GEMPAK and also incorporates PC_INTP functionality.
    """
    size = len(sounding['PRES'])
    recipe = [('TEMP', 'DWPT'), ('DRCT', 'SPED'), ('DWPT', None)]

    for var1, var2 in recipe:
        iabove = 0
        i = 1
        more = True
        while (i < size - 1) and more:
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

                if var2 is None and iabove != 0:
                    if sounding['PRES'][i] - sounding['PRES'][iabove] > 100:
                        for j in range(iabove, size):
                            sounding['DWPT'][j] = missing
                        iabove = 0
                        more = False

                if (var1 == 'DRCT' and iabove != 0
                   and sounding['PRES'][i - 1] > 100
                   and sounding['PRES'][iabove] < 100):
                    iabove = 0

                if iabove != 0:
                    output = {}
                    pres1 = sounding['PRES'][i - 1]
                    pres2 = sounding['PRES'][iabove]
                    vlev = sounding['PRES'][i]
                    between = (((pres1 < pres2) and (pres1 < vlev) and (vlev < pres2))
                               or ((pres2 < pres1) and (pres2 < vlev) and (vlev < pres1)))
                    if not between:
                        raise RuntimeError('Cannot interpolate sounding data.')
                    elif pres1 <= 0 or pres2 <= 0:
                        raise ValueError('Pressure cannot be negative.')

                    rmult = np.log(vlev / pres1) / np.log(pres2 / pres1)
                    output['PRES'] = vlev
                    for parm in [var1, var2]:
                        if parm is None:
                            continue
                        output[parm] = missing
                        if parm == 'DRCT':
                            angle1 = sounding[parm][i - 1] % 360
                            angle2 = sounding[parm][iabove] % 360
                            if abs(angle1 - angle2) > 180:
                                if angle1 < angle2:
                                    angle1 -= 360
                                else:
                                    angle2 -= 360
                            iangle = angle1 + (angle2 - angle1) * rmult
                            output[parm] = iangle % 360
                        else:
                            output[parm] = (
                                sounding[parm][i - 1]
                                + (sounding[parm][iabove] - sounding[parm][i - 1]) * rmult
                            )
                    sounding[var1][i] = output[var1]
                    if var2 is not None:
                        sounding[var2][i] = output[var2]
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
        pres = sounding['PRES'][i]
        hght = sounding['HGHT'][i]

        if pres == missing:
            continue
        elif hght != missing:
            pbot = pres
            zbot = hght
            ptop = 2000
        elif pbot == missing:
            continue
        else:
            ilev = i + 1
            while pres <= ptop:
                if sounding['HGHT'][ilev] != missing:
                    ptop = sounding['PRES'][ilev]
                    ztop = sounding['HGHT'][ilev]
                else:
                    ilev += 1
            sounding['HGHT'][i] = (zbot + (ztop - zbot)
                                   * (np.log(pres / pbot) / np.log(ptop / pbot)))

    if maxlev < size - 1:
        if maxlev > -1:
            pb = sounding['PRES'][maxlev] * units.hPa
            zb = sounding['HGHT'][maxlev] * units.m
            tb = sounding['TEMP'][maxlev] * units.degC
            tdb = sounding['DWPT'][maxlev] * units.degC
        else:
            pb = missing * units.hPa
            zb = missing * units.m
            tb = missing * units.degC
            tdb = missing * units.degC

        for i in range(maxlev + 1, size):
            if sounding['HGHT'][i] == missing:
                tt = sounding['TEMP'][i] * units.degC
                tdt = sounding['DWPT'][i] * units.degC
                pt = sounding['PRES'][i] * units.hPa
                if tdb.magnitude == missing:
                    tvb = tb
                else:
                    qb = specific_humidity_from_dewpoint(pb, tdb)
                    rb = mixing_ratio_from_specific_humidity(qb)
                    tvb = virtual_temperature(tb, rb)
                if tdt.magnitude == missing:
                    tvt = tt
                else:
                    qt = specific_humidity_from_dewpoint(pt, tdt)
                    rt = mixing_ratio_from_specific_humidity(qt)
                    tvt = virtual_temperature(tt, rt)
                H = scale_height(tvb, tvt)
                sounding['HGHT'][i] = hydrostatic_height(zb, pb, pt, H).magnitude


def _interp_moist_height(sounding, missing=-9999):
    """Interpolate moist hydrostatic height.

    This function mimics the functionality of the MR_SCMZ
    subroutine in GEMPAK. This the default behavior when
    merging observed sounding data.
    """
    hlist = (np.ones(len(sounding['PRES'])) * -9999) * units.m

    ilev = -1
    top = False

    found = False
    while not found and not top:
        ilev += 1
        if ilev >= len(sounding['PRES']):
            top = True
        elif (sounding['PRES'][ilev] != missing
              and sounding['TEMP'][ilev] != missing
              and sounding['HGHT'][ilev] != missing):
            found = True

    while not top:
        pb = sounding['PRES'][ilev] * units.hPa
        plev = sounding['PRES'][ilev] * units.hPa
        tb = sounding['TEMP'][ilev] * units.degC
        tdb = sounding['DWPT'][ilev] * units.degC
        zb = sounding['HGHT'][ilev] * units.m
        zlev = sounding['HGHT'][ilev] * units.m
        jlev = ilev
        klev = 0
        mand = False

        while not mand:
            jlev += 1
            if jlev >= len(sounding['PRES']):
                mand = True
                top = True
            else:
                pt = sounding['PRES'][jlev] * units.hPa
                tt = sounding['TEMP'][jlev] * units.degC
                tdt = sounding['DWPT'][jlev] * units.degC
                zt = sounding['HGHT'][jlev] * units.m
                if (zt.magnitude != missing
                   and tt.magnitude != missing):
                    mand = True
                    klev = jlev
                if (sounding['PRES'][ilev] != missing
                   and sounding['TEMP'][ilev] != missing
                   and sounding['PRES'][jlev] != missing
                   and sounding['TEMP'][jlev] != missing):
                    if tdt.magnitude == missing:
                        tvb = tb
                    else:
                        qb = specific_humidity_from_dewpoint(pb, tdb)
                        rb = mixing_ratio_from_specific_humidity(qb)
                        tvb = virtual_temperature(tb, rb)
                    if tdt.magnitude == missing:
                        tvt = tt
                    else:
                        qt = specific_humidity_from_dewpoint(pt, tdt)
                        rt = mixing_ratio_from_specific_humidity(qt)
                        tvt = virtual_temperature(tt, rt)
                    H = scale_height(tvb, tvt)
                    znew = hydrostatic_height(zb, pb, pt, H)
                    tb = tt
                    tdb = tdt
                    pb = pt
                    zb = znew
                else:
                    H = missing * units.m
                    znew = missing * units.m
                hlist[jlev] = H

        if klev != 0:
            s = (zt - zlev) / (znew - zlev)
            for h in range(ilev + 1, klev + 1):
                hlist[h] *= s

        hbb = zlev
        pbb = plev
        for ii in range(ilev + 1, jlev):
            p = sounding['PRES'][ii] * units.hPa
            H = hlist[ii]
            z = hydrostatic_height(hbb, pbb, p, H)
            sounding['HGHT'][ii] = z.magnitude
            hbb = z
            pbb = p

        ilev = klev


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
                if z != missing and zb != missing and pb != missing:
                    sounding['PRES'][j] = (
                        pb * np.exp((z - zb) * np.log(pt / pb) / (zt - zb))
                    )
        ilev = klev
        pb = pt
        zb = zt
        i += 1


class GempakFile():
    """Base class for GEMPAK files.

    Reads ubiquitous GEMPAK file headers (i.e., the data managment portion of
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
                     ('data_source', 'i', DataSource),
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
            if navb_size != NAVB_SIZE:
                raise ValueError('Navigation block size does not match GEMPAK specification')
            else:
                self.navigation_block = (
                    self._buffer.read_struct(NamedStruct(self.grid_nav_fmt,
                                                         self.prefmt,
                                                         'NavigationBlock'))
                )
            self.kx = int(self.navigation_block.right_grid_number)
            self.ky = int(self.navigation_block.top_grid_number)

            # Analysis Block
            anlb_size = self._buffer.read_int(4, self.endian, False)
            anlb_start = self._buffer.set_mark()
            if anlb_size != ANLB_SIZE:
                raise ValueError('Analysis block size does not match GEMPAK specification')
            else:
                anlb_type = self._buffer.read_struct(struct.Struct(self.prefmt + 'f'))[0]
                self._buffer.jump_to(anlb_start)
                if anlb_type == 1:
                    self.analysis_block = (
                        self._buffer.read_struct(NamedStruct(self.grid_anl_fmt1,
                                                             self.prefmt,
                                                             'AnalysisBlock'))
                    )
                elif anlb_type == 2:
                    self.analysis_block = (
                        self._buffer.read_struct(NamedStruct(self.grid_anl_fmt2,
                                                             self.prefmt,
                                                             'AnalysisBlock'))
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
            fmt = (fmt[0], self.prefmt + fmt[1])
            for n, part in enumerate(self.parts):
                for _ in range(part.parameter_count):
                    if fmt[1] == 's':
                        self.parameters[n][attr] += [self._buffer.read_binary(*fmt)[0].decode()]  # noqa: E501
                    else:
                        self.parameters[n][attr] += self._buffer.read_binary(*fmt)

    def _swap_bytes(self, binary):
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
        """Read the GEMPAK header from the file, if necessary."""
        fmt = [('text', '28s', bytes.decode), (None, None)]

        header = self._buffer.read_struct(NamedStruct(fmt, '', 'GempakHeader'))
        if header.text != GEMPAK_HEADER:
            raise TypeError('Unknown file format or invalid GEMPAK file')

    @staticmethod
    def _convert_dattim(dattim):
        if dattim:
            if dattim < 100000000:
                dt = datetime.strptime(str(dattim), '%y%m%d')
            else:
                dt = datetime.strptime('{:010d}'.format(dattim), '%m%d%y%H%M')
        else:
            dt = None
        return dt

    @staticmethod
    def _convert_ftime(ftime):
        if ftime:
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
        if (isinstance(level, int)
           or isinstance(level, float)):
            return level
        else:
            return None

    @staticmethod
    def _convert_vertical_coord(coord):
        if coord <= 8:
            return VerticalCoordinates(coord).name.upper()
        else:
            return struct.pack('i', coord).decode()

    @staticmethod
    def _convert_parms(parm):
        dparm = parm.decode()
        return dparm.strip() if dparm.strip() else ''

    @staticmethod
    def _fortran_ishift(i, shift):
        mask = 0xffffffff
        if shift > 0:
            shifted = ctypes.c_int32(i << shift).value
        elif shift < 0:
            if i < 0:
                shifted = (i & mask) >> abs(shift)
            else:
                shifted = i >> abs(shift)
        elif shift == 0:
            shifted = i
        else:
            raise ValueError('Bad shift value {}.'.format(shift))
        return shifted

    @staticmethod
    def _decode_strip(b):
        return b.decode().strip()

    @staticmethod
    def _make_date(dattim):
        return GempakFile._convert_dattim(dattim).date()

    @staticmethod
    def _make_time(t):
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
        unpacked = np.ones(npack * nparms) * self.prod_desc.missing_float
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
                               else (key, '4s', self._convert_parms) if key in string_names
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
            self._get_coordinates()

    def _get_crs(self):
        gemproj = self.navigation_block.projection
        proj, ptype = GEMPROJ_TO_PROJ[gemproj]

        if ptype == 'azm':
            lat_0 = self.navigation_block.proj_angle1
            lon_0 = self.navigation_block.proj_angle2
            lat_ts = self.navigation_block.proj_angle3
            self.crs = pyproj.CRS.from_dict({'proj': proj,
                                             'lat_0': lat_0,
                                             'lon_0': lon_0,
                                             'lat_ts': lat_ts})
        elif ptype == 'cyl':
            if gemproj != 'mcd':
                lat_0 = self.navigation_block.proj_angle1
                lon_0 = self.navigation_block.proj_angle2
                lat_ts = self.navigation_block.proj_angle3
                self.crs = pyproj.CRS.from_dict({'proj': proj,
                                                 'lat_0': lat_0,
                                                 'lon_0': lon_0,
                                                 'lat_ts': lat_ts})
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
                                                 'k_0': k_0})
        elif ptype == 'con':
            lat_1 = self.navigation_block.proj_angle1
            lon_0 = self.navigation_block.proj_angle2
            lat_2 = self.navigation_block.proj_angle3
            self.crs = pyproj.CRS.from_dict({'proj': proj,
                                             'lon_0': lon_0,
                                             'lat_1': lat_1,
                                             'lat_2': lat_2})

    def _get_coordinates(self):
        transform = pyproj.Proj(self.crs)
        llx, lly = transform(self.navigation_block.lower_left_lon,
                             self.navigation_block.lower_left_lat)
        urx, ury = transform(self.navigation_block.upper_right_lon,
                             self.navigation_block.upper_right_lat)
        self.x = np.linspace(llx, urx, self.kx)
        self.y = np.linspace(lly, ury, self.ky)
        xx, yy = np.meshgrid(self.x, self.y)
        self.lon, self.lat = transform(xx, yy, inverse=True)

    def _unpack_grid(self, packing_type, part):
        if packing_type == PackingType.none:
            lendat = self.data_header_length - part.header_length - 1

            if lendat > 1:
                buffer_fmt = '{}{}f'.format(self.prefmt, lendat)
                buffer = self._buffer.read_struct(struct.Struct(buffer_fmt))
                grid = np.zeros(self.ky * self.kx)
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
            grid = np.zeros((self.ky, self.kx))

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

            grid = np.zeros(self.grid_meta_int.kxky)
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

    def gdinfo(self):
        """Return grid information."""
        return self._gdinfo

    def gdxarray(self, parameter=None, date_time=None, coordinate=None,
                 level=None, date_time2=None, level2=None):
        """Select grids and output as list of xarray DataArrays."""
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

        if level is not None:
            if not isinstance(level, Iterable):
                level = [level]

        if date_time2 is not None:
            if (not isinstance(date_time2, Iterable)
               or isinstance(date_time2, str)):
                date_time2 = [date_time2]
            for i, dt in enumerate(date_time2):
                if isinstance(dt, str):
                    date_time2[i] = datetime.strptime(dt, '%Y%m%d%H%M')

        if level2 is not None:
            if not isinstance(level2, Iterable):
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
            matched = filter(lambda grid: grid if grid.LEVEL2 in level2 else False, matched)

        matched = list(matched)

        if len(matched) < 1:
            raise RuntimeError('No grids were matched.')

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
                                           mask=data == self.prod_desc.missing_float)
                    else:
                        data = np.ma.array(data, mask=data == self.prod_desc.missing_float)

                    xrda = xr.DataArray(
                        data=data[np.newaxis, np.newaxis, ...],
                        coords={
                            'time': [valid],
                            gvcord: [col_head.GLV1],
                            'x': self.x,
                            'y': self.y,
                        },
                        dims=['time', gvcord, 'y', 'x'],
                        name=var,
                        attrs={
                            **self.crs.to_cf(),
                            'grid_type': ftype,
                        }
                    )
                    grids.append(xrda)

                else:
                    log.warning('Bad grid for %s', col_head.GPM1)
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

                if pointer:
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
                        )
                    )

    def _unpack_merged(self, sndno):
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

                    if part.data_type == DataTypes.real:
                        packed_buffer_fmt = '{}{}f'.format(self.prefmt, lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    elif part.data_type == DataTypes.realpack:
                        packed_buffer_fmt = '{}{}i'.format(self.prefmt, lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    else:
                        raise NotImplementedError('No methods for data type {}'
                                                  .format(part.data_type))

                    parameters = self.parameters[iprt]
                    nparms = len(parameters['name'])

                    if part.data_type == DataTypes.realpack:
                        unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                        for iprm, param in enumerate(parameters['name']):
                            sounding[param] = unpacked[iprm::nparms]
                    else:
                        for iprm, param in enumerate(parameters['name']):
                            sounding[param] = packed_buffer[iprm::nparms]

                soundings.append(sounding)
        return soundings

    def _unpack_unmerged(self, sndno):
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

                    if part.data_type == DataTypes.real:
                        packed_buffer_fmt = '{}{}f'.format(self.prefmt, lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    elif part.data_type == DataTypes.realpack:
                        packed_buffer_fmt = '{}{}i'.format(self.prefmt, lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    elif part.data_type == DataTypes.character:
                        packed_buffer_fmt = '{}s'.format(lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    else:
                        raise NotImplementedError('No methods for data type {}'
                                                  .format(part.data_type))

                    parameters = self.parameters[iprt]
                    nparms = len(parameters['name'])
                    sounding[part.name] = {}
                    for iprm, param in enumerate(parameters['name']):
                        if part.data_type == DataTypes.realpack:
                            unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                            for iprm, param in enumerate(parameters['name']):
                                sounding[part.name][param] = unpacked[iprm::nparms]
                        elif part.data_type == DataTypes.character:
                            for iprm, param in enumerate(parameters['name']):
                                sounding[part.name][param] = (
                                    packed_buffer[iprm].decode().strip()
                                )
                        else:
                            sounding[part.name][param] = packed_buffer[iprm::nparms]

                soundings.append(self._merge_sounding(sounding))
        return soundings

    def _merge_sounding(self, parts):
        merged = {'STID': parts['STID'],
                  'STNM': parts['STNM'],
                  'SLAT': parts['SLAT'],
                  'SLON': parts['SLON'],
                  'SELV': parts['SELV'],
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
                    if z != self.prod_desc.missing_float:
                        if z < 0:
                            ppbb_is_z = False
                            parts['PPBB']['PRES'] = parts['PPBB']['HGHT']
                            break

        ppdd_is_z = True
        if num_above_sigw_levels:
            if 'PRES' in parts['PPDD']:
                ppdd_is_z = False
            else:
                for z in parts['PPDD']['HGHT']:
                    if z != self.prod_desc.missing_float:
                        if z < 0:
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
                if (mp != self.prod_desc.missing_float
                   and mt != self.prod_desc.missing_float
                   and mz != self.prod_desc.missing_float):
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
           and parts['TTBB']['PRES'][0] != self.prod_desc.missing_float
           and parts['TTBB']['TEMP'][0] != self.prod_desc.missing_float):
            first_man_p = merged['PRES'][0]
            first_sig_p = parts['TTBB']['PRES'][0]
            if (first_man_p == self.prod_desc.missing_float
               or np.isclose(first_man_p, first_sig_p)):
                merged['PRES'][0] = parts['TTBB']['PRES'][0]
                merged['DWPT'][0] = parts['TTBB']['DWPT'][0]
                merged['TEMP'][0] = parts['TTBB']['TEMP'][0]

        if ppbb_is_z:
            if (num_sigw_levels >= 1
               and parts['PPBB']['HGHT'][0] == 0
               and parts['PPBB']['DRCT'][0] != self.prod_desc.missing_float):
                merged['DRCT'][0] = parts['PPBB']['DRCT'][0]
                merged['SPED'][0] = parts['PPBB']['SPED'][0]
        else:
            if (num_sigw_levels >= 1
               and parts['PPBB']['PRES'][0] != self.prod_desc.missing_float
               and parts['PPBB']['DRCT'][0] != self.prod_desc.missing_float):
                first_man_p = merged['PRES'][0]
                first_sig_p = abs(parts['PPBB']['PRES'][0])
                if (first_man_p == self.prod_desc.missing_float
                   or np.isclose(first_man_p, first_sig_p)):
                    merged['DRCT'][0] = abs(parts['PPBB']['PRES'][0])
                    merged['DRCT'][0] = parts['PPBB']['DRCT'][0]
                    merged['SPED'][0] = parts['PPBB']['SPED'][0]

        # Merge MAN temperature
        bgl = 0
        if num_man_levels >= 2 or num_above_man_levels >= 1:
            if merged['PRES'][0] == self.prod_desc.missing_float:
                plast = 2000
            else:
                plast = merged['PRES'][0]

            for i in range(1, num_man_levels):
                if (parts['TTAA']['PRES'][i] < plast
                   and parts['TTAA']['PRES'][i] != self.prod_desc.missing_float
                   and parts['TTAA']['TEMP'][i] != self.prod_desc.missing_float
                   and parts['TTAA']['HGHT'][i] != self.prod_desc.missing_float):
                    for pname, pval in parts['TTAA'].items():
                        merged[pname].append(pval[i])
                    plast = merged['PRES'][-1]
                else:
                    bgl += 1

            for i in range(num_above_man_levels):
                if (parts['TTCC']['PRES'][i] < plast
                   and parts['TTCC']['PRES'][i] != self.prod_desc.missing_float
                   and parts['TTCC']['TEMP'][i] != self.prod_desc.missing_float
                   and parts['TTCC']['HGHT'][i] != self.prod_desc.missing_float):
                    for pname, pval in parts['TTCC'].items():
                        merged[pname].append(pval[i])
                    plast = merged['PRES'][-1]

        # Merge MAN wind
        if num_man_wind_levels >= 1 and num_man_levels >= 1:
            for iwind, pres in enumerate(parts['PPAA']['PRES']):
                if pres in merged['PRES'][1:]:
                    loc = merged['PRES'].index(pres)
                    if merged['DRCT'][loc] == self.prod_desc.missing_float:
                        merged['DRCT'][loc] = parts['PPAA']['DRCT'][iwind]
                        merged['SPED'][loc] = parts['PPAA']['SPED'][iwind]
                else:
                    size = len(merged['PRES'])
                    loc = size - bisect.bisect_left(merged['PRES'][1:][::-1], pres)
                    if loc >= size + 1:
                        loc = -1
                    merged['PRES'].insert(loc, pres)
                    merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                    merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                    merged['DRCT'].insert(loc, parts['PPAA']['DRCT'][iwind])
                    merged['SPED'].insert(loc, parts['PPAA']['SPED'][iwind])
                    merged['HGHT'].insert(loc, self.prod_desc.missing_float)

        if num_above_man_wind_levels >= 1 and num_man_levels >= 1:
            for iwind, pres in enumerate(parts['PPCC']['PRES']):
                if pres in merged['PRES'][1:]:
                    loc = merged['PRES'].index(pres)
                    if merged['DRCT'][loc] == self.prod_desc.missing_float:
                        merged['DRCT'][loc] = parts['PPCC']['DRCT'][iwind]
                        merged['SPED'][loc] = parts['PPCC']['SPED'][iwind]
                else:
                    size = len(merged['PRES'])
                    loc = size - bisect.bisect_left(merged['PRES'][1:][::-1], pres)
                    if loc >= size + 1:
                        loc = -1
                    merged['PRES'].insert(loc, pres)
                    merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                    merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                    merged['DRCT'].insert(loc, parts['PPCC']['DRCT'][iwind])
                    merged['SPED'].insert(loc, parts['PPCC']['SPED'][iwind])
                    merged['HGHT'].insert(loc, self.prod_desc.missing_float)

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
            for itrp, pres in enumerate(parts['TRPA']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['TRPA']['TEMP'][itrp] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                            merged['TEMP'][ploc] = parts['TRPA']['TEMP'][itrp]
                            merged['DWPT'][ploc] = parts['TRPA']['DWPT'][itrp]
                        if merged['DRCT'][ploc] == self.prod_desc.missing_float:
                            merged['DRCT'][ploc] = parts['TRPA']['DRCT'][itrp]
                            merged['SPED'][ploc] = parts['TRPA']['SPED'][itrp]
                        merged['HGHT'][ploc] = self.prod_desc.missing_float
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, parts['TRPA']['TEMP'][itrp])
                        merged['DWPT'].insert(loc, parts['TRPA']['DWPT'][itrp])
                        merged['DRCT'].insert(loc, parts['TRPA']['DRCT'][itrp])
                        merged['SPED'].insert(loc, parts['TRPA']['SPED'][itrp])
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

        if num_above_trop_levels >= 1:
            for itrp, pres in enumerate(parts['TRPC']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['TRPC']['TEMP'][itrp] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                            merged['TEMP'][ploc] = parts['TRPC']['TEMP'][itrp]
                            merged['DWPT'][ploc] = parts['TRPC']['DWPT'][itrp]
                        if merged['DRCT'][ploc] == self.prod_desc.missing_float:
                            merged['DRCT'][ploc] = parts['TRPC']['DRCT'][itrp]
                            merged['SPED'][ploc] = parts['TRPC']['SPED'][itrp]
                        merged['HGHT'][ploc] = self.prod_desc.missing_float
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, parts['TRPC']['TEMP'][itrp])
                        merged['DWPT'].insert(loc, parts['TRPC']['DWPT'][itrp])
                        merged['DRCT'].insert(loc, parts['TRPC']['DRCT'][itrp])
                        merged['SPED'].insert(loc, parts['TRPC']['SPED'][itrp])
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

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
            for isigt, pres in enumerate(parts['TTBB']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['TTBB']['TEMP'][isigt] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                            merged['TEMP'][ploc] = parts['TTBB']['TEMP'][isigt]
                            merged['DWPT'][ploc] = parts['TTBB']['DWPT'][isigt]
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, parts['TTBB']['TEMP'][isigt])
                        merged['DWPT'].insert(loc, parts['TTBB']['DWPT'][isigt])
                        merged['DRCT'].insert(loc, self.prod_desc.missing_float)
                        merged['SPED'].insert(loc, self.prod_desc.missing_float)
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

        if num_above_sigt_levels >= 1:
            for isigt, pres in enumerate(parts['TTDD']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['TTDD']['TEMP'][isigt] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                            merged['TEMP'][ploc] = parts['TTDD']['TEMP'][isigt]
                            merged['DWPT'][ploc] = parts['TTDD']['DWPT'][isigt]
                        merged['DRCT'][ploc] = self.prod_desc.missing_float
                        merged['SPED'][ploc] = self.prod_desc.missing_float
                        merged['HGHT'][ploc] = self.prod_desc.missing_float
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, parts['TTDD']['TEMP'][isigt])
                        merged['DWPT'].insert(loc, parts['TTDD']['DWPT'][isigt])
                        merged['DRCT'].insert(loc, self.prod_desc.missing_float)
                        merged['SPED'].insert(loc, self.prod_desc.missing_float)
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

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
                for isigw, pres in enumerate(parts['PPBB']['PRES']):
                    pres = abs(pres)
                    if (pres != self.prod_desc.missing_float
                       and parts['PPBB']['DRCT'][isigw] != self.prod_desc.missing_float
                       and parts['PPBB']['SPED'][isigw] != self.prod_desc.missing_float
                       and pres != 0):
                        if pres > pbot:
                            continue
                        elif pres in merged['PRES']:
                            ploc = merged['PRES'].index(pres)
                            if (merged['DRCT'][ploc] == self.prod_desc.missing_float
                               or merged['SPED'][ploc] == self.prod_desc.missing_float):
                                merged['DRCT'][ploc] = parts['PPBB']['DRCT'][isigw]
                                merged['SPED'][ploc] = parts['PPBB']['SPED'][isigw]
                        else:
                            size = len(merged['PRES'])
                            loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                            merged['PRES'].insert(loc, pres)
                            merged['DRCT'].insert(loc, parts['PPBB']['DRCT'][isigw])
                            merged['SPED'].insert(loc, parts['PPBB']['SPED'][isigw])
                            merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                            merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                            merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                    pbot = pres

            if num_above_sigw_levels >= 1 and not ppdd_is_z:
                for isigw, pres in enumerate(parts['PPDD']['PRES']):
                    pres = abs(pres)
                    if (pres != self.prod_desc.missing_float
                       and parts['PPDD']['DRCT'][isigw] != self.prod_desc.missing_float
                       and parts['PPDD']['SPED'][isigw] != self.prod_desc.missing_float
                       and pres != 0):
                        if pres > pbot:
                            continue
                        elif pres in merged['PRES']:
                            ploc = merged['PRES'].index(pres)
                            if (merged['DRCT'][ploc] == self.prod_desc.missing_float
                               or merged['SPED'][ploc] == self.prod_desc.missing_float):
                                merged['DRCT'][ploc] = parts['PPDD']['DRCT'][isigw]
                                merged['SPED'][ploc] = parts['PPDD']['SPED'][isigw]
                        else:
                            size = len(merged['PRES'])
                            loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                            merged['PRES'].insert(loc, pres)
                            merged['DRCT'].insert(loc, parts['PPDD']['DRCT'][isigw])
                            merged['SPED'].insert(loc, parts['PPDD']['SPED'][isigw])
                            merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                            merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                            merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                    pbot = pres

        # Merge max winds on pressure surfaces
        if num_max_wind_levels >= 1 or num_above_max_wind_levels >= 1:
            if merged['PRES'][0] != self.prod_desc.missing_float:
                pbot = merged['PRES'][0]
            elif len(merged['PRES']) > 1:
                pbot = merged['PRES'][1]
            else:
                pbot = 0

        if num_max_wind_levels >= 1:
            for imxw, pres in enumerate(parts['MXWA']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['MXWA']['DRCT'][imxw] != self.prod_desc.missing_float
                   and parts['MXWA']['SPED'][imxw] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if (merged['DRCT'][ploc] == self.prod_desc.missing_float
                           or merged['SPED'][ploc] == self.prod_desc.missing_float):
                            merged['DRCT'][ploc] = parts['MXWA']['DRCT'][imxw]
                            merged['SPED'][ploc] = parts['MXWA']['SPED'][imxw]
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['DRCT'].insert(loc, parts['MXWA']['DRCT'][imxw])
                        merged['SPED'].insert(loc, parts['MXWA']['SPED'][imxw])
                        merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                        merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

        if num_above_max_wind_levels >= 1:
            for imxw, pres in enumerate(parts['MXWC']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['MXWC']['DRCT'][imxw] != self.prod_desc.missing_float
                   and parts['MXWC']['SPED'][imxw] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if (merged['DRCT'][ploc] == self.prod_desc.missing_float
                           or merged['SPED'][ploc] == self.prod_desc.missing_float):
                            merged['DRCT'][ploc] = parts['MXWC']['DRCT'][imxw]
                            merged['SPED'][ploc] = parts['MXWC']['SPED'][imxw]
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['DRCT'].insert(loc, parts['MXWC']['DRCT'][imxw])
                        merged['SPED'].insert(loc, parts['MXWC']['SPED'][imxw])
                        merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                        merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

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

            size = len(merged['HGHT'])
            psfc = merged['PRES'][0]
            zsfc = merged['HGHT'][0]

            if (size >= 2 and psfc != self.prod_desc.missing_float
               and zsfc != self.prod_desc.missing_float):
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

            if (zold == self.prod_desc.missing_float
               or znxt == self.prod_desc.missing_float):
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

                if (hght == self.prod_desc.missing_float
                   and drct == self.prod_desc.missing_float
                   and sped == self.prod_desc.missing_float):
                    skip = True
                elif abs(zold - hght) < 1:
                    skip = True
                    if (merged['DRCT'][ilev - 1] == self.prod_desc.missing_float
                       or merged['SPED'][ilev - 1] == self.prod_desc.missing_float):
                        merged['DRCT'][ilev - 1] = drct
                        merged['SPED'][ilev - 1] = sped
                elif hght <= zold:
                    skip = True
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
                        if (merged['DRCT'][ilev - 1] == self.prod_desc.missing_float
                           or merged['SPED'][ilev - 1] == self.prod_desc.missing_float):
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

            # Interpolate misssing pressure with height
            _interp_logp_pressure(merged, self.prod_desc.missing_float)

        # Interpolate missing data
        _interp_logp_data(merged, self.prod_desc.missing_float)

        # Add below ground MAN data
        if merged['PRES'][0] != self.prod_desc.missing_float:
            if bgl > 0:
                for ibgl in range(1, num_man_levels):
                    pres = parts['TTAA']['PRES'][ibgl]
                    if pres > merged['PRES'][0]:
                        loc = size - bisect.bisect_left(merged['PRES'][1:][::-1], pres)
                        merged['PRES'].insert(loc, pres)
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
                 date_time=None):
        """Select soundings and output as list of xarray Datasets."""
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

        # Figure out which columns to extract from the file
        matched = self._sninfo.copy()

        if station_id is not None:
            matched = filter(
                lambda grid: grid if grid.ID in station_id else False,
                matched
            )

        if station_number is not None:
            matched = filter(
                lambda grid: grid if grid.NUMBER in station_number else False,
                matched
            )

        if date_time is not None:
            matched = filter(
                lambda grid: grid if grid.DATTIM in date_time else False,
                matched
            )

        matched = list(matched)

        if len(matched) < 1:
            raise RuntimeError('No grids were matched.')

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
            radat_text = {}
            attrs = {
                'station_id': snd.pop('STID'),
                'station_number': snd.pop('STNM'),
                'lat': snd.pop('SLAT'),
                'lon': snd.pop('SLON'),
                'elevation': snd.pop('SELV'),
                'station_pressure': station_pressure,
            }

            if 'TXTA' in snd:
                radat_text['txta'] = snd.pop('TXTA')
            if 'TXTB' in snd:
                radat_text['txtb'] = snd.pop('TXTB')
            if 'TXTC' in snd:
                radat_text['txtc'] = snd.pop('TXTC')
            if 'TXPB' in snd:
                radat_text['txpb'] = snd.pop('TXPB')
            if radat_text:
                attrs['RADAT'] = radat_text

            dt = datetime.combine(snd.pop('DATE'), snd.pop('TIME'))
            pres = np.array(snd.pop('PRES'))

            var = {}
            for param, values in snd.items():
                values = np.array(values)[np.newaxis, ...]
                maskval = np.ma.array(values, mask=values == self.prod_desc.missing_float)
                var[param.lower()] = (['time', 'pres'], maskval)

            xrds = xr.Dataset(var,
                              coords={'time': np.atleast_1d(dt), 'pres': pres},
                              attrs=attrs)

            # Sort to fix GEMPAK surface data at first level
            xrds = xrds.sortby('pres', ascending=False)

            soundings.append(xrds)
        return soundings


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
                    pointer = (self.prod_desc.data_block_ptr
                               + (irow * self.prod_desc.columns * self.prod_desc.parts)
                               + (icol * self.prod_desc.parts))

                    if pointer:
                        self._sfinfo.append(
                            Surface(
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
        elif self.surface_type == 'ship':
            irow = 0
            for icol, col_head in enumerate(self.column_headers):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts))

                if pointer:
                    self._sfinfo.append(
                        Surface(
                            irow,
                            icol,
                            datetime.combine(col_head.DATE, col_head.TIME),
                            col_head.STID,
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
                    pointer = (self.prod_desc.data_block_ptr
                               + (irow * self.prod_desc.columns * self.prod_desc.parts)
                               + (icol * self.prod_desc.parts))

                    if pointer:
                        self._sfinfo.append(
                            Surface(
                                irow,
                                icol,
                                datetime.combine(col_head.DATE, col_head.TIME),
                                row_head.STID,
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

    def _get_surface_type(self):
        if len(self.row_headers) == 1:
            self.surface_type = 'ship'
        elif 'DATE' in self.row_keys:
            self.surface_type = 'standard'
        elif 'DATE' in self.column_keys:
            self.surface_type = 'climate'
        else:
            raise RuntimeError('Unknown surface data type')

    def _key_types(self, keys):
        header_info = [(key, '4s', self._decode_strip) if key == 'STID'
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

        return header_info

    def _unpack_climate(self, sfcno):
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

                    if part.data_type == DataTypes.real:
                        packed_buffer_fmt = '{}{}f'.format(self.prefmt, lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    elif part.data_type == DataTypes.realpack:
                        packed_buffer_fmt = '{}{}i'.format(self.prefmt, lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    elif part.data_type == DataTypes.character:
                        packed_buffer_fmt = '{}s'.format(lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    else:
                        raise NotImplementedError('No methods for data type {}'
                                                  .format(part.data_type))

                    parameters = self.parameters[iprt]

                    if part.data_type == DataTypes.realpack:
                        unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = unpacked[iprm]
                    elif part.data_type == DataTypes.character:
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = packed_buffer[iprm].decode().strip()
                    else:
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = packed_buffer[iprm]

                stations.append(station)
        return stations

    def _unpack_ship(self, sfcno):
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

                if part.data_type == DataTypes.real:
                    packed_buffer_fmt = '{}{}f'.format(self.prefmt, lendat)
                    packed_buffer = (
                        self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                    )
                elif part.data_type == DataTypes.realpack:
                    packed_buffer_fmt = '{}{}i'.format(self.prefmt, lendat)
                    packed_buffer = (
                        self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                    )
                elif part.data_type == DataTypes.character:
                    packed_buffer_fmt = '{}s'.format(lendat)
                    packed_buffer = (
                        self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                    )
                else:
                    raise NotImplementedError('No methods for data type {}'
                                              .format(part.data_type))

                parameters = self.parameters[iprt]

                if part.data_type == DataTypes.realpack:
                    unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = unpacked[iprm]
                elif part.data_type == DataTypes.character:
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = packed_buffer[iprm].decode().strip()
                else:
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = packed_buffer[iprm]

            stations.append(station)
        return stations

    def _unpack_standard(self, sfcno):
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

                    if part.data_type == DataTypes.real:
                        packed_buffer_fmt = '{}{}f'.format(self.prefmt, lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    elif part.data_type == DataTypes.realpack:
                        packed_buffer_fmt = '{}{}i'.format(self.prefmt, lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    elif part.data_type == DataTypes.character:
                        packed_buffer_fmt = '{}s'.format(lendat)
                        packed_buffer = (
                            self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
                        )
                    else:
                        raise NotImplementedError('No methods for data type {}'
                                                  .format(part.data_type))

                    parameters = self.parameters[iprt]

                    if part.data_type == DataTypes.realpack:
                        unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = unpacked[iprm]
                    elif part.data_type == DataTypes.character:
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = packed_buffer[iprm].decode().strip()
                    else:
                        for iprm, param in enumerate(parameters['name']):
                            station[param] = packed_buffer[iprm]

                stations.append(station)
        return stations

    def sfjson(self, station_id=None, station_number=None,
               date_time=None, state=None, country=None):
        """Select surface stations and output as JSON."""
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

        if state is not None:
            if (not isinstance(state, Iterable)
               or isinstance(state, str)):
                state = [state]
                state = [s.upper() for s in state]

        if country is not None:
            if (not isinstance(country, Iterable)
               or isinstance(country, str)):
                country = [country]
                country = [c.upper() for c in country]

        # Figure out which columns to extract from the file
        matched = self._sfinfo.copy()

        if station_id is not None:
            matched = filter(
                lambda grid: grid if grid.ID in station_id else False,
                matched
            )

        if station_number is not None:
            matched = filter(
                lambda grid: grid if grid.NUMBER in station_number else False,
                matched
            )

        if date_time is not None:
            matched = filter(
                lambda grid: grid if grid.DATTIM in date_time else False,
                matched
            )

        if state is not None:
            matched = filter(
                lambda grid: grid if grid.STATE in state else False,
                matched
            )

        if country is not None:
            matched = filter(
                lambda grid: grid if grid.COUNTRY in country else False,
                matched
            )

        matched = list(matched)

        if len(matched) < 1:
            raise RuntimeError('No grids were matched.')

        sfcno = [(s.ROW, s.COL) for s in matched]

        if self.surface_type == 'standard':
            data = self._unpack_standard(sfcno)
        elif self.surface_type == 'ship':
            data = self._unpack_ship(sfcno)
        elif self.surface_type == 'climate':
            data = self._unpack_climate(sfcno)

        stnarr = []
        for stn in data:
            stnobj = {}
            stnobj['properties'] = {}
            stnobj['values'] = {}
            stnobj['properties']['date_time'] = datetime.combine(stn.pop('DATE'),
                                                                 stn.pop('TIME'))
            stnobj['properties']['station_id'] = stn.pop('STID')
            stnobj['properties']['station_number'] = stn.pop('STNM')
            stnobj['properties']['longitude'] = stn.pop('SLON')
            stnobj['properties']['latitude'] = stn.pop('SLAT')
            stnobj['properties']['elevation'] = stn.pop('SELV')
            stnobj['properties']['state'] = stn.pop('STAT')
            stnobj['properties']['station_id_alt'] = stn.pop('STD2')
            stnobj['properties']['priority'] = stn.pop('SPRI')
            if stn:
                for name, ob in stn.items():
                    stnobj['values'][name.lower()] = ob
                stnarr.append(stnobj)

        return stnarr
