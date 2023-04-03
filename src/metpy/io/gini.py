# Copyright (c) 2015,2016,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools to process GINI-formatted products."""

import contextlib
from datetime import datetime
from enum import Enum
from io import BytesIO
from itertools import repeat  # noqa: I202
import logging
from pathlib import Path
import re

import numpy as np
from xarray import Dataset, Variable
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractDataStore
from xarray.coding.times import CFDatetimeCoder
from xarray.coding.variables import CFMaskCoder
from xarray.core.utils import FrozenDict

from ._tools import Bits, IOBuffer, NamedStruct, open_as_needed, zlib_decompress_all_frames
from ..package_tools import Exporter

exporter = Exporter(globals())
log = logging.getLogger(__name__)


def _make_datetime(s):
    r"""Convert 7 bytes from a GINI file to a `datetime` instance."""
    year, month, day, hour, minute, second, cs = s
    if year < 70:
        year += 100
    return datetime(1900 + year, month, day, hour, minute, second, 10000 * cs)


def _scaled_int(s):
    r"""Convert a 3 byte string to a signed integer value."""
    # Get leftmost bit (sign) as 1 (if 0) or -1 (if 1)
    sign = 1 - ((s[0] & 0x80) >> 6)

    # Combine remaining bits
    int_val = (((s[0] & 0x7f) << 16) | (s[1] << 8) | s[2])
    log.debug('Source: %s Int: %x Sign: %d', ' '.join(hex(c) for c in s), int_val, sign)

    # Return scaled and with proper sign
    return (sign * int_val) / 10000.


def _name_lookup(names):
    r"""Create an io helper to convert an integer to a named value."""
    mapper = dict(zip(range(len(names)), names))

    def lookup(val):
        return mapper.get(val, 'UnknownValue')
    return lookup


class GiniProjection(Enum):
    r"""Represents projection values in GINI files."""

    mercator = 1
    lambert_conformal = 3
    polar_stereographic = 5


@exporter.export
class GiniFile(AbstractDataStore):
    """A class that handles reading the GINI format satellite images from the NWS.

    This class attempts to decode every byte that is in a given GINI file.

    Notes
    -----
    The internal data structures that things are decoded into are subject to change.

    """

    missing = 255
    wmo_finder = re.compile('(T\\w{3}\\d{2})[\\s\\w\\d]+\\w*(\\w{3})\r\r\n')

    crafts = ['Unknown', 'Unknown', 'Miscellaneous', 'JERS', 'ERS/QuikSCAT', 'POES/NPOESS',
              'Composite', 'DMSP', 'GMS', 'METEOSAT', 'GOES-7', 'GOES-8', 'GOES-9',
              'GOES-10', 'GOES-11', 'GOES-12', 'GOES-13', 'GOES-14', 'GOES-15', 'GOES-16']

    sectors = ['NH Composite', 'East CONUS', 'West CONUS', 'Alaska Regional',
               'Alaska National', 'Hawaii Regional', 'Hawaii National', 'Puerto Rico Regional',
               'Puerto Rico National', 'Supernational', 'NH Composite', 'Central CONUS',
               'East Floater', 'West Floater', 'Central Floater', 'Polar Floater']

    channels = ['Unknown', 'Visible', 'IR (3.9 micron)', 'WV (6.5/6.7 micron)',
                'IR (11 micron)', 'IR (12 micron)', 'IR (13 micron)', 'IR (1.3 micron)',
                'Reserved', 'Reserved', 'Reserved', 'Reserved', 'Reserved', 'LI (Imager)',
                'PW (Imager)', 'Surface Skin Temp (Imager)', 'LI (Sounder)', 'PW (Sounder)',
                'Surface Skin Temp (Sounder)', 'CAPE', 'Land-sea Temp', 'WINDEX',
                'Dry Microburst Potential Index', 'Microburst Day Potential Index',
                'Convective Inhibition', 'Volcano Imagery', 'Scatterometer', 'Cloud Top',
                'Cloud Amount', 'Rainfall Rate', 'Surface Wind Speed', 'Surface Wetness',
                'Ice Concentration', 'Ice Type', 'Ice Edge', 'Cloud Water Content',
                'Surface Type', 'Snow Indicator', 'Snow/Water Content', 'Volcano Imagery',
                'Reserved', 'Sounder (14.71 micron)', 'Sounder (14.37 micron)',
                'Sounder (14.06 micron)', 'Sounder (13.64 micron)', 'Sounder (13.37 micron)',
                'Sounder (12.66 micron)', 'Sounder (12.02 micron)', 'Sounder (11.03 micron)',
                'Sounder (9.71 micron)', 'Sounder (7.43 micron)', 'Sounder (7.02 micron)',
                'Sounder (6.51 micron)', 'Sounder (4.57 micron)', 'Sounder (4.52 micron)',
                'Sounder (4.45 micron)', 'Sounder (4.13 micron)', 'Sounder (3.98 micron)',
                # Percent Normal TPW found empirically from Service Change Notice 20-03
                'Sounder (3.74 micron)', 'Sounder (Visible)', 'Percent Normal TPW']

    prod_desc_fmt = NamedStruct([('source', 'b'),
                                 ('creating_entity', 'b', _name_lookup(crafts)),
                                 ('sector_id', 'b', _name_lookup(sectors)),
                                 ('channel', 'b', _name_lookup(channels)),
                                 ('num_records', 'H'), ('record_len', 'H'),
                                 ('datetime', '7s', _make_datetime),
                                 ('projection', 'b', GiniProjection), ('nx', 'H'), ('ny', 'H'),
                                 ('la1', '3s', _scaled_int), ('lo1', '3s', _scaled_int)
                                 ], '>', 'ProdDescStart')

    lc_ps_fmt = NamedStruct([('reserved', 'b'), ('lov', '3s', _scaled_int),
                             ('dx', '3s', _scaled_int), ('dy', '3s', _scaled_int),
                             ('proj_center', 'b')], '>', 'LambertOrPolarProjection')

    mercator_fmt = NamedStruct([('resolution', 'b'), ('la2', '3s', _scaled_int),
                                ('lo2', '3s', _scaled_int), ('di', 'H'), ('dj', 'H')
                                ], '>', 'MercatorProjection')

    prod_desc2_fmt = NamedStruct([('scanning_mode', 'b', Bits(3)),
                                  ('lat_in', '3s', _scaled_int), ('resolution', 'b'),
                                  ('compression', 'b'), ('version', 'b'), ('pdb_size', 'H'),
                                  ('nav_cal', 'b')], '>', 'ProdDescEnd')

    nav_fmt = NamedStruct([('sat_lat', '3s', _scaled_int), ('sat_lon', '3s', _scaled_int),
                           ('sat_height', 'H'), ('ur_lat', '3s', _scaled_int),
                           ('ur_lon', '3s', _scaled_int)], '>', 'Navigation')

    def __init__(self, filename):
        r"""Create an instance of `GiniFile`.

        Parameters
        ----------
        filename : str or file-like object
            If str, the name of the file to be opened. Gzip-ed files are
            recognized with the extension ``'.gz'``, as are bzip2-ed files with
            the extension ``'.bz2'`` If `filename` is a file-like object,
            this will be read from directly.

        """
        fobj = open_as_needed(filename)

        # Just read in the entire set of data at once
        with contextlib.closing(fobj):
            self._buffer = IOBuffer.fromfile(fobj)

        # Pop off the WMO header if we find it
        self.wmo_code = ''
        self._process_wmo_header()
        log.debug('First wmo code: %s', self.wmo_code)

        # Decompress the data if necessary, and if so, pop off new header
        log.debug('Length before decompression: %s', len(self._buffer))
        self._buffer = IOBuffer(self._buffer.read_func(zlib_decompress_all_frames))
        log.debug('Length after decompression: %s', len(self._buffer))

        # Process WMO header inside compressed data if necessary
        self._process_wmo_header()
        log.debug('2nd wmo code: %s', self.wmo_code)

        # Read product description start
        start = self._buffer.set_mark()

        #: :desc: Decoded first section of product description block
        #: :type: namedtuple
        self.prod_desc = self._buffer.read_struct(self.prod_desc_fmt)
        log.debug(self.prod_desc)

        #: :desc: Decoded geographic projection information
        #: :type: namedtuple
        self.proj_info = None

        # Handle projection-dependent parts
        if self.prod_desc.projection in (GiniProjection.lambert_conformal,
                                         GiniProjection.polar_stereographic):
            self.proj_info = self._buffer.read_struct(self.lc_ps_fmt)
        elif self.prod_desc.projection == GiniProjection.mercator:
            self.proj_info = self._buffer.read_struct(self.mercator_fmt)
        else:
            log.warning('Unknown projection: %d', self.prod_desc.projection)
        log.debug(self.proj_info)

        # Read the rest of the guaranteed product description block (PDB)
        #: :desc: Decoded second section of product description block
        #: :type: namedtuple
        self.prod_desc2 = self._buffer.read_struct(self.prod_desc2_fmt)
        log.debug(self.prod_desc2)

        if self.prod_desc2.nav_cal not in (0, -128):  # TODO: See how GEMPAK/MCIDAS parses
            # Only warn if there actually seems to be useful navigation data
            if self._buffer.get_next(self.nav_fmt.size) != b'\x00' * self.nav_fmt.size:
                log.warning('Navigation/Calibration unhandled: %d', self.prod_desc2.nav_cal)
            if self.prod_desc2.nav_cal in (1, 2):
                self.navigation = self._buffer.read_struct(self.nav_fmt)
                log.debug(self.navigation)

        # Catch bad PDB with size set to 0
        if self.prod_desc2.pdb_size == 0:
            log.warning('Adjusting bad PDB size from 0 to 512.')
            self.prod_desc2 = self.prod_desc2._replace(pdb_size=512)

        # Jump past the remaining empty bytes in the product description block
        self._buffer.jump_to(start, self.prod_desc2.pdb_size)

        # Read the actual raster--unless it's PNG compressed, in which case that happens later
        blob = self._buffer.read(self.prod_desc.num_records * self.prod_desc.record_len)

        # Check for end marker
        end = self._buffer.read(self.prod_desc.record_len)
        if end != b''.join(repeat(b'\xff\x00', self.prod_desc.record_len // 2)):
            log.warning('End marker not as expected: %s', end)

        # Check to ensure that we processed all of the data
        if not self._buffer.at_end():
            if not blob:
                log.debug('No data read yet, trying to decompress remaining data as an image.')
                from matplotlib.image import imread
                blob = (imread(BytesIO(self._buffer.read())) * 255).astype('uint8')
            else:
                log.warning('Leftover unprocessed data beyond EOF marker: %s',
                            self._buffer.get_next(10))

        self.data = np.array(blob).reshape((self.prod_desc.ny,
                                            self.prod_desc.nx))

    def _process_wmo_header(self):
        """Read off the WMO header from the file, if necessary."""
        data = self._buffer.get_next(64).decode('utf-8', 'ignore')
        match = self.wmo_finder.search(data)
        if match:
            self.wmo_code = match.groups()[0]
            self.siteID = match.groups()[-1]
            self._buffer.skip(match.end())

    def __str__(self):
        """Return a string representation of the product."""
        parts = [self.__class__.__name__ + ': {0.creating_entity} {0.sector_id} {0.channel}',
                 'Time: {0.datetime}', 'Size: {0.ny}x{0.nx}',
                 'Projection: {0.projection.name}',
                 'Lower Left Corner (Lon, Lat): ({0.lo1}, {0.la1})',
                 'Resolution: {1.resolution}km']
        return '\n\t'.join(parts).format(self.prod_desc, self.prod_desc2)

    def _make_proj_var(self):
        proj_info = self.proj_info
        prod_desc2 = self.prod_desc2
        attrs = {'earth_radius': 6371200.0}
        if self.prod_desc.projection == GiniProjection.lambert_conformal:
            attrs['grid_mapping_name'] = 'lambert_conformal_conic'
            attrs['standard_parallel'] = prod_desc2.lat_in
            attrs['longitude_of_central_meridian'] = proj_info.lov
            attrs['latitude_of_projection_origin'] = prod_desc2.lat_in
        elif self.prod_desc.projection == GiniProjection.polar_stereographic:
            attrs['grid_mapping_name'] = 'polar_stereographic'
            attrs['straight_vertical_longitude_from_pole'] = proj_info.lov
            attrs['latitude_of_projection_origin'] = -90 if proj_info.proj_center else 90
            attrs['standard_parallel'] = 60.0  # See Note 2 for Table 4.4A in ICD
        elif self.prod_desc.projection == GiniProjection.mercator:
            attrs['grid_mapping_name'] = 'mercator'
            attrs['longitude_of_projection_origin'] = self.prod_desc.lo1
            attrs['latitude_of_projection_origin'] = self.prod_desc.la1
            attrs['standard_parallel'] = prod_desc2.lat_in
        else:
            raise NotImplementedError(
                f'Unhandled GINI Projection: {self.prod_desc.projection}')

        return 'projection', Variable((), 0, attrs)

    def _make_time_var(self):
        base_time = self.prod_desc.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        offset = self.prod_desc.datetime - base_time
        time_var = Variable((), data=offset.seconds + offset.microseconds / 1e6,
                            attrs={'units': 'seconds since ' + base_time.isoformat()})

        return 'time', time_var

    def _get_proj_and_res(self):
        import pyproj

        proj_info = self.proj_info
        prod_desc2 = self.prod_desc2

        kwargs = {'a': 6371200.0, 'b': 6371200.0}
        if self.prod_desc.projection == GiniProjection.lambert_conformal:
            kwargs['proj'] = 'lcc'
            kwargs['lat_0'] = prod_desc2.lat_in
            kwargs['lon_0'] = proj_info.lov
            kwargs['lat_1'] = prod_desc2.lat_in
            kwargs['lat_2'] = prod_desc2.lat_in
            dx, dy = proj_info.dx, proj_info.dy
        elif self.prod_desc.projection == GiniProjection.polar_stereographic:
            kwargs['proj'] = 'stere'
            kwargs['lon_0'] = proj_info.lov
            kwargs['lat_0'] = -90 if proj_info.proj_center else 90
            kwargs['lat_ts'] = 60.0  # See Note 2 for Table 4.4A in ICD
            kwargs['x_0'] = False  # Easting
            kwargs['y_0'] = False  # Northing
            dx, dy = proj_info.dx, proj_info.dy
        elif self.prod_desc.projection == GiniProjection.mercator:
            kwargs['proj'] = 'merc'
            kwargs['lat_0'] = self.prod_desc.la1
            kwargs['lon_0'] = self.prod_desc.lo1
            kwargs['lat_ts'] = prod_desc2.lat_in
            kwargs['x_0'] = False  # Easting
            kwargs['y_0'] = False  # Northing
            dx, dy = prod_desc2.resolution, prod_desc2.resolution

        return pyproj.Proj(**kwargs), dx, dy

    def _make_coord_vars(self):
        proj, dx, dy = self._get_proj_and_res()

        # Get projected location of lower left point
        x0, y0 = proj(self.prod_desc.lo1, self.prod_desc.la1)

        # Coordinate variable for x
        xlocs = x0 + np.arange(self.prod_desc.nx) * (1000. * dx)
        attrs = {'units': 'm', 'long_name': 'x coordinate of projection',
                 'standard_name': 'projection_x_coordinate'}
        x_var = Variable(('x',), xlocs, attrs)

        # Now y--Need to flip y because we calculated from the lower left corner,
        # but the raster data is stored with top row first.
        ylocs = (y0 + np.arange(self.prod_desc.ny) * (1000. * dy))[::-1]
        attrs = {'units': 'm', 'long_name': 'y coordinate of projection',
                 'standard_name': 'projection_y_coordinate'}
        y_var = Variable(('y',), ylocs, attrs)

        # Get the two-D lon,lat grid as well
        x, y = np.meshgrid(xlocs, ylocs)
        lon, lat = proj(x, y, inverse=True)

        lon_var = Variable(('y', 'x'), data=lon,
                           attrs={'long_name': 'longitude', 'units': 'degrees_east'})
        lat_var = Variable(('y', 'x'), data=lat,
                           attrs={'long_name': 'latitude', 'units': 'degrees_north'})

        return [('x', x_var), ('y', y_var), ('lon', lon_var), ('lat', lat_var)]

    def _make_data_vars(self):
        proj_var_name, proj_var = self._make_proj_var()
        name = self.prod_desc.channel
        if '(' in name:
            name = name.split('(')[0].rstrip()

        missing_val = self.missing
        attrs = {'long_name': self.prod_desc.channel, 'missing_value': missing_val,
                 'coordinates': 'lon lat time', 'grid_mapping': proj_var_name}
        data_var = Variable(('y', 'x'), data=self.data, attrs=attrs)
        return [(proj_var_name, proj_var), (name, data_var)]

    def get_variables(self):
        """Get all variables in the file.

        This is used by `xarray.open_dataset`.

        """
        variables = [self._make_time_var()]
        variables.extend(self._make_coord_vars())
        variables.extend(self._make_data_vars())

        return FrozenDict(variables)

    def get_attrs(self):
        """Get the global attributes.

        This is used by `xarray.open_dataset`.

        """
        return FrozenDict(satellite=self.prod_desc.creating_entity,
                          sector=self.prod_desc.sector_id)


class GiniXarrayBackend(BackendEntrypoint):
    """Entry point for direct reading of GINI data into Xarray."""

    def open_dataset(self, filename_or_obj, *, drop_variables=None):
        """Open the GINI datafile as a Xarray dataset.

        This is the main entrypoint for plugging into Xarray read support.

        """
        # TODO: This can be structured much better when we're not still supporting both the
        # old Xarray API as well as direct use of GiniFile itself. In MetPy 2.0 the only
        # access should be as an xarray backend entrypoint.
        gini = GiniFile(filename_or_obj)
        gini_attrs = gini.get_attrs()
        coords = dict(gini._make_coord_vars() + [gini._make_time_var()])
        coords['time'] = CFDatetimeCoder().decode(coords['time'])
        (proj_name, proj_var), (data_name, data_var) = gini._make_data_vars()
        data_var.attrs.pop('coordinates')
        decoded_data_var = CFMaskCoder().decode(data_var, data_name)
        return Dataset({proj_name: proj_var, data_name: decoded_data_var}, coords, gini_attrs)

    def guess_can_open(self, filename_or_obj):
        """Try to guess whether we can read this file.

        This allows files ending in '.gini' to be automatically opened by xarray.

        """
        with contextlib.suppress(TypeError):
            return Path(filename_or_obj).suffix == '.gini'
        return False
