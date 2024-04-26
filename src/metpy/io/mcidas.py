# Copyright (c) 2024 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for decoding MCIDAS AREA files."""

import contextlib
from datetime import datetime
import logging
import struct
import sys

import numpy as np
import pyproj
import rasterio
import rasterio.mask
import rasterio.transform
import rasterio.warp
import shapely.geometry as sgeom
import shapely.ops as sops

from metpy.io._tools import IOBuffer, NamedStruct

logger = logging.getLogger(__name__)

SENSOR_SOURCE = {
    0: 'Derived data',
    1: 'Test patterns',
    2: 'Graphics',
    3: 'Miscellaneous',
    4: 'PDUS METEOSAT Visible',
    5: 'PDUS METEOSAT Infrared',
    6: 'PDUS METEOSAT Water Vapor',
    7: 'Radar',
    8: 'Miscellaneous aircraft data',
    9: 'Raw METEOSAT',
    10: 'Composite image',
    12: 'GMS Visible',
    13: 'GMS Infrared',
    14: 'ATS 6 Visible',
    15: 'ATS 6 Infrared',
    16: 'SMS-1 Visible',
    17: 'SMS-1 Infrared',
    18: 'SMS-2 Visible',
    19: 'SMS-2 Infrared',
    20: 'GOES-1 Visible',
    21: 'GOES-1 Infrared',
    22: 'GOES-2 Visible',
    23: 'GOES-2 Infrared',
    24: 'GOES-3 Visible',
    25: 'GOES-3 Infrared',
    26: 'GOES-4 Visible (VAS)',
    27: 'GOES-4 Infrared and Water Vapor (VAS)',
    28: 'GOES-5 Visible (VAS)',
    29: 'GOES-5 Infrared and Water Vapor (VAS)',
    30: 'GOES-6 Visible',
    31: 'GOES-6 Infrared',
    32: 'GOES-7 Visible, Block 1 Auxiliary Data',
    33: 'GOES-7 Infrared',
    34: 'FY-2B',
    35: 'FY-2C',
    36: 'FY-2D',
    37: 'FY-2E',
    38: 'FY-2F',
    39: 'FY-2G',
    40: 'FY-2H',
    41: 'TIROS-N (POES)',
    42: 'NOAA-6',
    43: 'NOAA-7',
    44: 'NOAA-8',
    45: 'NOAA-9',
    46: 'Venus',
    47: 'Voyager 1',
    48: 'Voyager 2',
    49: 'Galileo',
    50: 'Hubble Space Telescope',
    51: 'Meteosat-8 (MSG-1)',
    52: 'Meteosat-9 (MSG-2)',
    53: 'Meteosat-10 (MSG-3)',
    54: 'Meteosat-3',
    55: 'Meteosat-4',
    56: 'Meteosat-5',
    57: 'Meteosat-6',
    58: 'Meteosat-7',
    60: 'NOAA-10',
    61: 'NOAA-11',
    62: 'NOAA-12',
    63: 'NOAA-13',
    64: 'NOAA-14',
    65: 'NOAA-15',
    66: 'NOAA-16',
    67: 'NOAA-17',
    68: 'NOAA-18',
    69: 'NOAA-19',
    70: 'GOES-8',
    71: 'GOES-8 Sounder',
    72: 'GOES-9',
    73: 'GOES-9 Sounder',
    74: 'GOES-10',
    75: 'GOES-10 Sounder',
    76: 'GOES-11',
    77: 'GOES-11 Sounder',
    78: 'GOES-12',
    79: 'GOES-12 Sounder',
    80: 'ERBE',
    82: 'GMS-4',
    83: 'GMS-5',
    84: 'MTSAT-1R',
    85: 'MTSAT-2',
    86: 'Himawari-8',
    87: 'DMSP F-8',
    88: 'DMSP F-9',
    89: 'DMSP F-10',
    90: 'DMSP F-11',
    91: 'DMSP F-12',
    92: 'DMSP F-13',
    93: 'DMSP F-14',
    94: 'DMSP F-15',
    95: 'FY-1B',
    96: 'FY-1C',
    97: 'FY-1D',
    101: 'TERRA-L1B',
    102: 'TERRA-CLD',
    103: 'TERRA-GEO',
    104: 'TERRA-AER',
    106: 'TERRA-TOP',
    107: 'TERRA-ATM',
    108: 'TERRA-GUS',
    109: 'TERRA-RET',
    111: 'AQUA-L1B',
    112: 'AQUA-CLD',
    113: 'AQUA-GEO',
    114: 'AQUA-AER',
    116: 'AQUA-TOP',
    117: 'AQUA-ATM',
    118: 'AQUA-GUS',
    119: 'AQUA-RET',
    128: 'TERRA-SST',
    129: 'TERRA-LST',
    138: 'AQUA-SST',
    139: 'AQUA-LST',
    160: 'TERRA-NDVI',
    161: 'TERRA-CREF',
    170: 'AQUA-NDVI',
    171: 'AQUA-CREF',
    174: 'EWS-G1',
    175: 'EWS-G1 Sounder',
    176: 'EWS-G2',
    177: 'EWS-G2 Sounder',
    178: 'EWS-G3',
    179: 'EWS-G3 Sounder',
    180: 'GOES-13',
    181: 'GOES-13 Sounder',
    182: 'GOES-14',
    183: 'GOES-14 Sounder',
    184: 'GOES-15',
    185: 'GOES-15 Sounder',
    186: 'GOES-16',
    187: 'GOES-16 Level 2 Products',
    188: 'GOES-17',
    189: 'GOES-17 Level 2 Products',
    190: 'GOES-18',
    191: 'GOES-18 Level 2 Products',
    192: 'GOES-19',
    193: 'GOES-19 Level 2 Products',
    195: 'DMSP F-16',
    196: 'DMSP F-17',
    200: 'AIRS-L1B',
    210: 'AMSR-E L1B',
    211: 'AMSR-E RAIN',
    216: 'AMSU-A LWP',
    220: 'TRMM',
    221: 'GMS-1',
    222: 'GMS-2',
    223: 'GMS-3',
    230: 'Kalpana-1',
    231: 'INSAT-3D Imager',
    232: 'INSAT-3D Sounder',
    240: 'Metop-A',
    241: 'Metop-B',
    242: 'Metop-C',
    250: 'COMS-1',
    261: 'Landsat 1',
    262: 'Landsat 2',
    263: 'Landsat 3',
    264: 'Landsat 4',
    265: 'Landsat 5',
    266: 'Landsat 6',
    267: 'Landsat 7',
    268: 'Landsat 8',
    275: 'FY-3A',
    276: 'FY-3B',
    277: 'FY-3C',
    286: 'HimawariCast-8',
    287: 'Himawari-9',
    288: 'HimawariCast-9',
    289: 'HimawariCast-8/9',
    300: 'NPP-VIIRS',
    301: 'NOAA-20 (JPSS-1)',
    302: 'NOAA-21 (JPSS-2)',
    303: 'NOAA-22 (JPSS-3)',
    304: 'NOAA-23 (JPSS-4)',
    320: 'SNPP SDR',
    321: 'NOAA-20 SDR',
    322: 'NOAA-21 SDR',
    323: 'NOAA-22 SDR',
    324: 'NOAA-23 SDR',
    325: 'SNPP EDR',
    326: 'NOAA-20 EDR',
    327: 'NOAA-21 EDR',
    328: 'NOAA-22 EDR',
    329: 'NOAA-23 EDR',
    354: 'Meteosat-11 (MSG-4)',
    400: 'South Pole Composite',
    401: 'North Pole Composite',
    410: 'Megha-Tropic',
}


def _decode_strip(bytestring):
    """Decode and strip bytes."""
    return bytestring.decode().replace('\x00', '').strip()


def dms_to_decimal(dms):
    """Convert DMS coordinates to decimal.

    Parameters
    ----------
    dms : int
        Coordinate value in DDDMMSS format.

    Returns
    -------
    float
        Coordinate value as decimal.
    """
    dms_str = f'{dms:07d}'
    d = float(dms_str[:3])
    m = float(dms_str[3:5])
    s = float(dms_str[5:])

    return d + (m / 60) + (s / 3600)


def range_longitude(lon):
    """Convert 0-360 longitude to -180-180.

    Parameters
    ----------
    lon : float
        Longitude in range of 0-360.

    Returns
    -------
    float
        Longitude in range -180-180.
    """
    return (lon + 180) % 360 - 180


class AreaFile:
    """McIDAS AREA decoder class."""

    directory_format = [
        ('adde_position', 'i'), ('image_type', 'i'),
        ('sensor_source', 'i'), ('date', 'i'), ('time', 'i'),
        ('upper_left_line_coordinate', 'i'),
        ('upper_left_image_element', 'i'), (None, '4x'),
        ('image_lines', 'i'), ('data_per_line', 'i'),
        ('bytes_per_point', 'i'), ('line_resolution', 'i'),
        ('element_resolution', 'i'), ('spectral_bands', 'i'),
        ('line_prefix_length', 'i'), ('ssec_project', 'i'),
        ('creation_date', 'i'), ('creation_time', 'i'),
        ('spectral_band_map_32', 'i'), ('spectral_band_map_64', 'i'),
        ('word21', 'i'), ('word22', 'i'), ('word23', 'i'),
        ('word24', 'i'), ('memo', '32s', _decode_strip),
        ('area_number', 'i'), ('data_offset', 'i'), ('navigation_offset', 'i'),
        ('validity_code', 'i'), ('pdl1', 'i'), ('pdl2', 'i'),
        ('pdl3', 'i'), ('pdl4', 'i'), ('pdl5', 'i'), ('pdl6', 'i'),
        ('pdl7', 'i'), ('pdl8', 'i'), ('band8_source', 'i'),
        ('image_start_date', 'i'), ('image_start_time', 'i'),
        ('image_start_scan', 'i'), ('prefix_doc_length', 'i'),
        ('prefix_calibration_length', 'i'), ('prefix_band_length', 'i'),
        ('source_type', '4s', bytes.decode), ('calibration_type', '4s', bytes.decode),
        ('word54', 'i'), ('word55', 'i'), ('word56', 'i'),
        ('original_source_type', 'i'), ('units', 'i'), ('scaling', 'i'),
        ('supplemental_offset', 'i'), ('supplemental_length', 'i'),
        ('word62', 'i'), ('calibration_offset', 'i'), ('comment_length', 'i')
    ]

    def __init__(self, file, projection=None, bbox=None, num_threads=2):
        """Instantiate AREA object from file.

        Parameters
        ----------
        file : str or `pathlib.Path`
            AREA file.

        projection : `pyproj.Proj`
            Optional projection for reprojection of the image.

        bbox : tuple of floats
            Tuple corresponding to (minlon, minlat, maxlon, maxlat).
            Optional bounding box to crop raster.

        num_threads : int
            Number of threads to use for reprojection using `rasterio.warp`.
        """
        with contextlib.closing(open(file, 'rb')) as fobj:  # noqa: SIM115
            self._buffer = IOBuffer.fromfile(fobj)

        self._start = self._buffer.set_mark()

        check = self._buffer.read_binary(8)
        self._swap_bytes(check[4:])
        self._buffer.jump_to(self._start)

        self._num_threads = num_threads

        self.directory_block = self._buffer.read_struct(
            NamedStruct(self.directory_format, self.prefmt, 'Directory')
        )

        if self.directory_block.navigation_offset:
            self._buffer.jump_to(self._start, self.directory_block.navigation_offset)
            self.navigation_type = self._buffer.read_ascii(4)

            if self.navigation_type == 'GVAR':
                self.navigation_block = self._buffer.read_binary(639)
            else:
                self.navigation_block = self._buffer.read_struct(
                    NamedStruct(self._get_navigation_format(self.navigation_type),
                                self.prefmt, 'Navigation')
                )
        else:
            self.navigation_type = None

        self._set_georeference()

        self._get_data()

        self._set_extent()

        if projection is not None:
            self._reproject(projection)

        if bbox is not None:
            self._crop_raster(bbox)

        self._set_timestamp()

    @property
    def image(self):
        """Get image."""
        return self._image

    @property
    def shape(self):
        """Get shape."""
        return self._image.shape

    @property
    def crs(self):
        """Get CRS."""
        return self._crs

    @property
    def transform(self):
        """Get affine transform."""
        return self._transform

    @property
    def timestamp(self):
        """Get timestamp."""
        return self._timestamp

    def get_extent(self, projection=None):
        """Get extent of image.

        Parameters
        ----------
        projection : `pyproj.Proj`
            Optional projection for reprojection of the extent.
        """
        if projection is None:
            return self._extent
        else:
            tform = pyproj.Transformer.from_crs(self.crs, projection.crs)
            left, right, bot, top = self._extent
            llx, lly = tform.transform(left, bot)
            urx, ury = tform.transform(right, top)
            return llx, urx, lly, ury

    def _crop_raster(self, bbox):
        """Crop raster to a given bounding box.

        bbox : tuple of floats
            Tuple corresponding to (minlon, minlat, maxlon, maxlat).
        """
        rect = sgeom.box(*bbox)

        if self.is_projected:
            reproject = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'),
                                                    self._crs,
                                                    always_xy=True).transform
            rect = sops.transform(reproject, rect)

        height, width = self._image.shape
        src_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'crs': self._crs,
            'transform': self._transform,
            'dtype': self._image.dtype,
            'nodata': 0,
            'compress': 'lzw',
        }

        with rasterio.MemoryFile() as mem, mem.open(**src_profile) as dataset:
            dataset.write(self._image[np.newaxis, ...])  # [count, y, x]
            crop_image, crop_affine = rasterio.mask.mask(
                dataset, [rect], all_touched=True, crop=True, pad=False
            )

        self._image = crop_image[0, ...]
        self._transform = crop_affine
        self._set_extent()

    def _reproject(self, projection):
        """Reproject raster image.

        Parameters
        ----------
        projection : `pyproj.Proj`
            Projection for reprojection of the image.
        """
        height, width = self._image.shape
        src_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'crs': self._crs,
            'transform': self._transform,
            'dtype': self._image.dtype,
            'nodata': 0,
            'compress': 'lzw',
        }

        with rasterio.MemoryFile() as mem:
            with mem.open(**src_profile) as dataset:
                dataset.write(self._image[np.newaxis, ...])  # [count, y, x]
            with mem.open() as dataset:
                proj_img, aff = rasterio.warp.reproject(
                    rasterio.band(dataset, 1),
                    dst_crs=projection.crs,
                    num_threads=self._num_threads,
                    dst_nodata=0
                )
                self._image = proj_img[0, ...]

        self._transform = aff
        self._crs = projection.crs
        if projection.crs.coordinate_operation:
            self.is_projected = True
        else:
            self.is_projected = False
        self._set_extent()

    def _get_data(self):
        """Extract data from file."""
        data_ptr = self.directory_block.data_offset
        self._buffer.jump_to(self._start, data_ptr)
        self._data_start = self._buffer.set_mark()

        ny = self.directory_block.image_lines
        nx = self.directory_block.data_per_line

        len_prefix = self.directory_block.line_prefix_length

        data_dtype = np.dtype(f'u{self.directory_block.bytes_per_point}')
        if self.prefmt:
            data_dtype.newbyteorder(self.prefmt)
        data = np.zeros((ny, nx), data_dtype)

        for j in range(ny):
            if len_prefix:
                val_code = self._buffer.read_int(4, self.endian, False)
                if val_code != self.directory_block.validity_code:
                    continue
                docs = self._buffer.read_ascii(  # noqa: F841
                    self.directory_block.prefix_doc_length
                )
                dtype = np.dtype('int32')
                if self.prefmt:
                    dtype = dtype.newbyteorder(self.prefmt)
                cal = self._buffer.read_array(  # noqa: F841
                    self.directory_block.prefix_calibration_length,
                    dtype
                )
                bands = self._buffer.read_array(  # noqa: F841
                    self.directory_block.prefix_band_length,
                    dtype
                )

            buffer = self._buffer.read_array(nx, data_dtype)
            data[j, :] = buffer

        self._image = data

    @staticmethod
    def _get_navigation_format(navigation_type):
        """Construct struct format for a given navigation type.

        Parameters
        ----------
        navigation_type : str
            Navigation type from AREA file.
        """
        navigation_format = {
            'DMSP': [
                ('source_date', 'i'), ('image_time', 'i'), ('orbit_type', 'i'),
                ('epoch_date', '4s'), ('epoch_time', '4s'), ('mean_motion1', 'i'),
                ('mean_motion2', 'i'), ('mean_motion3', 'i'), ('bstart1', 'i'),
                ('bstart2', 'i'), ('inclination', 'i'), ('right_ascension', 'i'),
                ('eccentricity', 'i'), ('perigee', 'i'), ('mean_anomaly', 'i'),
                ('mean_motion3', 'i'), (None, '108x'), ('data_type', '4s', bytes.decode),
                ('ascending_lt', 'i'), ('first_scan', 'i'), ('first_scan_time', 'i'),
                ('scan_flipped', 'i'), ('element_flipped', 'i'), (None, '4x'),
                ('elem_per_scan', 'i'), (None, '272x'), ('ascii_record', '32s', _decode_strip)
            ],
            'GOES': [
                ('satellite_date', 'i'), ('image_time', 'i'), ('orbit_type', 'i'),
                ('epoch_date', 'i'), ('epoch_time', 'i'), ('semimajor_axis', 'i'),
                ('eccentricity', 'i'), ('inclination', 'i'), ('mean_anomaly', 'i'),
                ('perigee', 'i'), ('right_ascension', 'i'), ('declination', 'i'),
                ('right_ascension_sat', 'i'), ('center_line', 'i'), ('spin_period', 'i'),
                ('sweep_anlge_line', 'i'), ('scan_lines', 'i'), ('sensors', 'i'),
                ('lines', 'i'), ('sweep_angle_element', 'i'), ('elements', 'i'),
                ('pitch', 'i'), ('yaw', 'i'), ('roll', 'i'), (None, '4x'),
                ('east_west_adjust', 'i'), ('asjust_time', 'i'), (None, '4x'),
                ('sensor_angle_delta', 'i'), ('skew', 'i'), (None, '4x'),
                ('first_beta_line', 'i'), ('first_beta_time1', 'i'),
                ('first_beta_time2', 'i'), ('beta_count1', 'i'), ('second_beta_line', 'i'),
                ('second_beta_time1', 'i'), ('second_beta_time2', 'i'), ('beta_count2', 'i'),
                ('gamma_offset', 'i'), ('time_zero', 'i'), ('gamma_dot', 'i'), (None, '4x'),
                ('memo', '32s', _decode_strip)
            ],
            'LALO': [
                ('navigation_source', '4s', bytes.decode), (None, '252x'),
                ('number_rows', 'i'), ('number_elements', 'i'), ('min_lat', 'i'),
                ('min_lon', 'i'), ('max_lat', 'i'), ('max_lon', 'i'),
                ('line_resolution', 'i'), ('element_resolution', 'i'),
                ('navigation_size', 'i'), ('nav_data_size', 'i'),
                ('top_line_numbner', 'i'), ('left_element', 'i'),
                ('block_size', 'i'), ('lat_header_size', 'i'),
                ('aux_lon_offset', 'i'), ('dir_lat_offset', 'i'),
                ('dir_lon_offset', 'i'), (None, '184x')
            ],
            'LAMB': [
                ('image_pole_line', 'i'), ('image_pole_element', 'i'),
                ('standard_lat1', 'i'), ('standard_lat2', 'i'),
                ('standard_lat_resolution', 'i'), ('central_lon', 'i'),
                ('sphere_radius', 'i'), ('sphere_eccentricity', 'i'),
                ('coordinate_type', 'i'), ('longitude_convention', 'i'),
                (None, '436x'), ('memo', '32s', _decode_strip)
            ],
            'MERC': [
                ('equator_line', 'i'), ('central_lon_element', 'i'),
                ('standard_lat', 'i', dms_to_decimal), ('standard_lat_resolution', 'i'),
                ('central_lon', 'i', dms_to_decimal), ('sphere_radius', 'i'),
                ('sphere_eccentricity', 'i'), ('coordinate_type', 'i'),
                ('longitude_convention', 'i'), (None, '440x'),
                ('memo', '32s', _decode_strip)
            ],
            'MSAT': [
                ('navigation_date', 'i'), ('navigation_time', 'i'),
                ('reference_position', 'i'), ('ref_line', 'i'),
                ('center_line', 'i'), ('center_lon', 'i'), (None, '8x'),
                ('navigation_date2', 'i'), (None, '984x')
            ],
            'MSG ': [
                (None, '508x')
            ],
            'PS  ': [
                ('image_pole_line', 'i'), ('image_pole_element', 'i'),
                ('standard_lat', 'i'), ('standard_lat_resolution', 'i'),
                ('central_lon', 'i'), ('sphere_radius', 'i'),
                ('sphere_eccentricity', 'i'), ('coordinate_type', 'i'),
                ('longitude_convention', 'i'), (None, '440x'),
                ('memo', '32s', _decode_strip)
            ],
            'RADR': [
                ('rda_row', 'i'), ('rda_column', 'i'), ('rda_lat', 'i'),
                ('rda_lon', 'i'), ('resolution', 'i'), ('zenith_angle', 'i'),
                ('longitude_resolution', 'i')
            ],
            'RECT': [
                ('image_row_number', 'i'), ('image_row_latitude', 'i'),
                ('image_column_number', 'i'), ('image_column_longitude', 'i'),
                ('dy', 'i'), ('dx', 'i'), ('sphere_radius', 'i'),
                ('sphere_eccentricity', 'i'), ('coordinate_type', 'i'),
                ('longitude_convention', 'i')
            ],
            'TANC': [
                ('image_pole_line', 'i'), ('image_pole_element', 'i'),
                ('km_per_pixel', 'i'), ('standard_lat', 'i'), ('standard_lon', 'i'),
                (None, '456x'), ('memo', '32s', _decode_strip)
            ],
            'TIRO': [
                ('source_date', 'i'), ('navigation_time', 'i'), ('orbit_type', 'i'),
                ('epoch_date', 'i'), ('epoch_time', 'i'), ('semimajor_axis', 'i'),
                ('eccentricity', 'i'), ('inclination', 'i'), ('mean_anomaly', 'i'),
                ('perigee', 'i'), ('right_ascension', 'i'), ('samples_per_line', 'i'),
                ('angular_increment', 'i'), ('fraction_seconds', 'i'), (None, '120x'),
                ('pass_type', 'i'), ('first_line_coord', 'i'), ('first_line_time', 'i'),
                ('line_interval1', 'i'), ('inverted', 'i'), ('inverted_lines', 'i'),
                ('inverted_elements', 'i'), ('line_interval2', 'i'), ('data_interval', 'i'),
                (None, '264x'), ('comments', '32s', _decode_strip)
            ]
        }.get(navigation_type)

        if navigation_format is None:
            raise NotImplementedError(f'No format for navigation type {navigation_type}.')
        else:
            return navigation_format

    def _set_georeference(self):
        """Get geographic transform for image data."""
        nav = self.navigation_type

        yres = self.directory_block.line_resolution
        xres = self.directory_block.element_resolution
        origin_line = self.directory_block.upper_left_line_coordinate
        origin_elem = self.directory_block.upper_left_image_element

        if nav == 'RECT':
            # FIXME: NASA RGBs have odd dx/dy values that seem to
            # not be scaled correctly. We account for that here
            # until a better way is found.
            if self.navigation_block.dx > 1e4:
                dx = self.navigation_block.dx / 1e6
            else:
                dx = self.navigation_block.dx / 1e4

            if self.navigation_block.dy > 1e4:
                dy = self.navigation_block.dy / 1e6
            else:
                dy = self.navigation_block.dy / 1e4

            # ecc = self.navigation_block.sphere_eccentricity / 1e6
            # semimajor_r = self.navigation_block.sphere_radius
            # semiminor_r = (1 - ecc**2) * semimajor_r**2

            # Account for area resolution and map to area coordinates
            diff_y = (self.navigation_block.image_row_number - origin_line) / yres
            diff_x = (self.navigation_block.image_column_number - origin_elem) / xres

            base_lat = self.navigation_block.image_row_latitude / 1e4
            base_lon = self.navigation_block.image_column_longitude / 1e4

            if self.navigation_block.longitude_convention >= 0:
                base_lon *= -1

            # Find upper-left coordinates
            origin_lat = base_lat + (diff_y * dy)
            origin_lon = base_lon - (diff_x * dx)

            self._transform = rasterio.transform.from_origin(
                origin_lon, origin_lat, dx, dy
            )

            self._crs = pyproj.CRS(
                proj='longlat',
                R=self.navigation_block.sphere_radius,
                e=self.navigation_block.sphere_eccentricity / 1e6,
            )

            self.is_projected = False
        elif nav == 'MERC':
            lat_ts = self.navigation_block.standard_lat
            lat_ts_res = self.navigation_block.standard_lat_resolution
            lon_0 = self.navigation_block.central_lon

            # Account for area resolution and map to area coordinates
            diff_y = (self.navigation_block.equator_line - origin_line) / yres
            diff_x = (self.navigation_block.central_lon_element - origin_elem) / xres

            # Account for area resolution in dx, dy too
            dx = lat_ts_res * xres
            dy = lat_ts_res * yres

            if self.navigation_block.longitude_convention >= 0:
                lon_0 *= -1

            self._crs = pyproj.CRS(
                proj='merc',
                R=self.navigation_block.sphere_radius,
                e=self.navigation_block.sphere_eccentricity / 1e6,
                lat_ts=lat_ts,
                lon_0=lon_0
            )

            # Center (x, y) always (0, 0), so skip some steps
            # to get the origin
            origin_x = -diff_x * dx
            origin_y = diff_y * dy

            self._transform = rasterio.transform.from_origin(
                origin_x, origin_y, dx, dy
            )

            self.is_projected = True
        elif nav == 'TANC':
            lon_0 = -self.navigation_block.standard_lon / 1e4
            lat_1 = self.navigation_block.standard_lat / 1e4
            r_sphere = 6371100.0  # m
            res = (self.navigation_block.km_per_pixel / 1e4) * 1000  # m

            # Account for area resolution
            pole_line = self.navigation_block.image_pole_line / 1e4
            pole_element = self.navigation_block.image_pole_element / 1e4
            diff_y = (pole_line - origin_line) / yres
            diff_x = (pole_element - origin_elem) / xres

            px = diff_x * res
            py = diff_y * res

            self._crs = pyproj.CRS(
                proj='lcc',
                R=r_sphere,
                lat_0=lat_1,
                lat_1=lat_1,
                lat_2=lat_1,
                lon_0=lon_0
            )

            proj = pyproj.Proj(self._crs)
            _pole_x, pole_y = proj(lon_0, 90)

            uly = pole_y + py
            ulx = -px

            self._transform = rasterio.transform.from_origin(
                ulx, uly, res, res
            )

            self.is_projected = True
        else:
            raise NotImplementedError(f'{nav} navigation not currently supported.')

    def _set_extent(self):
        """Set image extent."""
        left, bottom, right, top = rasterio.transform.array_bounds(
            *self._image.shape, self._transform
        )
        self._extent = left, right, bottom, top

    def _set_timestamp(self):
        """Set timestamp."""
        dstr = f'{self.directory_block.date:06d}'
        tstr = f'{self.directory_block.time:06d}'
        year = 1900 + int(dstr[:3])
        doy = dstr[3:]
        self._timestamp = datetime.strptime(
            f'{year}{doy}{tstr}', '%Y%j%H%M%S'
        )

    def _swap_bytes(self, binary):
        """Swap between little and big endian.

        Parameters
        ----------
        binary : bytes
        """
        self.swapped_bytes = (struct.pack('@i', 4) != binary)

        if self.swapped_bytes:
            if sys.byteorder == 'little':
                self.prefmt = '>'
                self.endian = 'big'
            elif sys.byteorder == 'big':
                self.prefmt = '<'
                self.endian = 'little'
        else:
            self.prefmt = ''
            self.endian = sys.byteorder
