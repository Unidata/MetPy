# Copyright (c) 2015,2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools to process GINI-formatted products."""
from __future__ import absolute_import

import contextlib
from datetime import datetime
try:
    from enum import Enum
except ImportError:
    from enum34 import Enum
from io import BytesIO
from itertools import repeat  # noqa: I202
import logging
import re

import numpy as np
try:
    from xarray import Variable
    from xarray.backends.common import AbstractDataStore
    from xarray.core.utils import FrozenOrderedDict
except ImportError:
    # This way GiniFile is still usable without xarray
    AbstractDataStore = object

from ._tools import Bits, IOBuffer, NamedStruct, open_as_needed, zlib_decompress_all_frames
from .cdm import cf_to_proj, Dataset
from ..deprecation import deprecated
from ..package_tools import Exporter

exporter = Exporter(globals())
log = logging.getLogger(__name__)


def _make_datetime(s):
    r"""Convert 7 bytes from a GINI file to a `datetime` instance."""
    s = bytearray(s)  # For Python 2
    year, month, day, hour, minute, second, cs = s
    if year < 70:
        year += 100
    return datetime(1900 + year, month, day, hour, minute, second, 10000 * cs)


def _scaled_int(s):
    r"""Convert a 3 byte string to a signed integer value."""
    s = bytearray(s)  # For Python 2

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
        return mapper.get(val, 'Unknown')
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
    The internal data structures that things are decoded into are subject to change. For
    a more stable interface, use the :meth:`to_dataset` method.

    See Also
    --------
    GiniFile.to_dataset

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
                'Sounder (3.74 micron)', 'Sounder (Visible)']

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

    @deprecated(0.8, alternative='xarray.open_dataset(GiniFile)')
    def to_dataset(self):
        """Convert to a CDM dataset.

        Gives a representation of the data in a much more user-friendly manner, providing
        easy access to Variables and relevant attributes.

        Returns
        -------
        Dataset

        .. deprecated:: 0.8.0

        """
        ds = Dataset()

        # Put in time
        ds.createDimension('time', 1)
        time_var = ds.createVariable('time', np.int32, dimensions=('time',))
        base_time = self.prod_desc.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        time_var.units = 'milliseconds since ' + base_time.isoformat()
        offset = (self.prod_desc.datetime - base_time)
        time_var[:] = offset.seconds * 1000 + offset.microseconds / 1000.

        # Set up projection
        if self.prod_desc.projection == GiniProjection.lambert_conformal:
            proj_var = ds.createVariable('Lambert_Conformal', np.int32)
            proj_var.grid_mapping_name = 'lambert_conformal_conic'
            proj_var.standard_parallel = self.prod_desc2.lat_in
            proj_var.longitude_of_central_meridian = self.proj_info.lov
            proj_var.latitude_of_projection_origin = self.prod_desc2.lat_in
            proj_var.earth_radius = 6371200.0
            _add_projection_coords(ds, self.prod_desc, proj_var, self.proj_info.dx,
                                   self.proj_info.dy)
        elif self.prod_desc.projection == GiniProjection.polar_stereographic:
            proj_var = ds.createVariable('Polar_Stereographic', np.int32)
            proj_var.grid_mapping_name = 'polar_stereographic'
            proj_var.straight_vertical_longitude_from_pole = self.proj_info.lov
            proj_var.latitude_of_projection_origin = -90 if self.proj_info.proj_center else 90
            proj_var.earth_radius = 6371200.0
            proj_var.standard_parallel = 60.0  # See Note 2 for Table 4.4A in ICD
            _add_projection_coords(ds, self.prod_desc, proj_var, self.proj_info.dx,
                                   self.proj_info.dy)
        elif self.prod_desc.projection == GiniProjection.mercator:
            proj_var = ds.createVariable('Mercator', np.int32)
            proj_var.grid_mapping_name = 'mercator'
            proj_var.longitude_of_projection_origin = self.prod_desc.lo1
            proj_var.latitude_of_projection_origin = self.prod_desc.la1
            proj_var.standard_parallel = self.prod_desc2.lat_in
            proj_var.earth_radius = 6371200.0
            _add_projection_coords(ds, self.prod_desc, proj_var, self.prod_desc2.resolution,
                                   self.prod_desc2.resolution)
        else:
            raise NotImplementedError('Need to add more projections to dataset!')

        # Now the data
        name = self.prod_desc.channel
        if '(' in name:
            name = name.split('(')[0].rstrip()
        data_var = ds.createVariable(name, self.data.dtype, ('y', 'x'),
                                     wrap_array=np.ma.array(self.data,
                                                            mask=self.data == self.missing))
        data_var.long_name = self.prod_desc.channel
        data_var.missing_value = self.missing
        data_var.coordinates = 'y x'
        data_var.grid_mapping = proj_var.name

        # Add a bit more metadata
        ds.satellite = self.prod_desc.creating_entity
        ds.sector = self.prod_desc.sector_id
        return ds

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
                'Unhandled GINI Projection: {}'.format(self.prod_desc.projection))

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

    # FIXME: Work around xarray <=0.10.3 docstring for load angering sphinx
    # That's the only reason this exists.
    def load(self):
        """
        Load the variables and attributes simultaneously.

        A centralized loading function makes it easier to create
        data stores that do automatic encoding/decoding.

        For example::

            class SuffixAppendingDataStore(AbstractDataStore):

                def load(self):
                    variables, attributes = AbstractDataStore.load(self)
                    variables = {'%s_suffix' % k: v
                                 for k, v in iteritems(variables)}
                    attributes = {'%s_suffix' % k: v
                                  for k, v in iteritems(attributes)}
                    return variables, attributes

        This function will be called anytime variables or attributes
        are requested, so care should be taken to make sure its fast.
        """
        return super(GiniFile, self).load()

    def get_variables(self):
        """Get all variables in the file.

        This is used by `xarray.open_dataset`.

        """
        variables = [self._make_time_var()]
        proj_var_name, proj_var = self._make_proj_var()
        variables.append((proj_var_name, proj_var))
        variables.extend(self._make_coord_vars())

        # Now the data
        name = self.prod_desc.channel
        if '(' in name:
            name = name.split('(')[0].rstrip()

        missing_val = self.missing
        attrs = {'long_name': self.prod_desc.channel, 'missing_value': missing_val,
                 'coordinates': 'y x time', 'grid_mapping': proj_var_name}
        data_var = Variable(('y', 'x'),
                            data=np.ma.array(self.data,
                                             mask=self.data == missing_val),
                            attrs=attrs)
        variables.append((name, data_var))

        return FrozenOrderedDict(variables)

    def get_attrs(self):
        """Get the global attributes.

        This is used by `xarray.open_dataset`.

        """
        return FrozenOrderedDict(satellite=self.prod_desc.creating_entity,
                                 sector=self.prod_desc.sector_id)

    def get_dimensions(self):
        """Get the file's dimensions.

        This is used by `xarray.open_dataset`.

        """
        return FrozenOrderedDict(x=self.prod_desc.nx, y=self.prod_desc.ny)


def _add_projection_coords(ds, prod_desc, proj_var, dx, dy):
    """Add coordinate variables (projection and lon/lat) to a dataset."""
    proj = cf_to_proj(proj_var)

    # Get projected location of lower left point
    x0, y0 = proj(prod_desc.lo1, prod_desc.la1)

    # Coordinate variable for x
    ds.createDimension('x', prod_desc.nx)
    x_var = ds.createVariable('x', np.float64, dimensions=('x',))
    x_var.units = 'm'
    x_var.long_name = 'x coordinate of projection'
    x_var.standard_name = 'projection_x_coordinate'
    x_var[:] = x0 + np.arange(prod_desc.nx) * (1000. * dx)

    # Now y
    ds.createDimension('y', prod_desc.ny)
    y_var = ds.createVariable('y', np.float64, dimensions=('y',))
    y_var.units = 'm'
    y_var.long_name = 'y coordinate of projection'
    y_var.standard_name = 'projection_y_coordinate'

    # Need to flip y because we calculated from the lower left corner, but the raster data
    # is stored with top row first.
    y_var[::-1] = y0 + np.arange(prod_desc.ny) * (1000. * dy)

    # Get the two-D lon,lat grid as well
    x, y = np.meshgrid(x_var[:], y_var[:])
    lon, lat = proj(x, y, inverse=True)
    lon_var = ds.createVariable('lon', np.float64, dimensions=('y', 'x'), wrap_array=lon)
    lon_var.long_name = 'longitude'
    lon_var.units = 'degrees_east'

    lat_var = ds.createVariable('lat', np.float64, dimensions=('y', 'x'), wrap_array=lat)
    lat_var.long_name = 'latitude'
    lat_var.units = 'degrees_north'

    ds.img_extent = (x_var[:].min(), x_var[:].max(), y_var[:].min(), y_var[:].max())
