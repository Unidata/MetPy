# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Support reading information from various text file formats."""

import contextlib
import re

import geopandas
import numpy as np
from shapely.geometry import LineString, Point

from ._tools import open_as_needed


def _decode_coords(coordinates):
    """Turn a string of coordinates from WPC coded surface bulletin into a lon/lat tuple.

    Parameters
    ----------
    coordinates : str
        A string of numbers that can be converted into a lon/lat tuple

    Returns
    -------
    (lon, lat) : tuple
        Longitude and latitude parsed from `coordinates`

    Notes
    -----
    In the WPC coded surface bulletin, latitude and longitude are given in degrees north and
    degrees west, respectivley. Therefore, this function always returns latitude as a positive
    number and longitude as a negative number.

    Examples
    --------
    >>> _decode_coords('4731193')
    (-119.3, 47.3)

    """
    # Based on the number of digits, find the correct place to split between lat and lon
    # Hires bulletins provide 7 digits for coordinates; regular bulletins provide 4 or 5 digits
    split_pos = int(len(coordinates) / 2)
    lat, lon = coordinates[:split_pos], coordinates[split_pos:]

    # Insert decimal point at the correct place and convert to float
    lat = float(f'{lat[:2]}.{lat[2:]}')
    lon = -float(f'{lon[:3]}.{lon[3:]}')

    return lon, lat


def parse_wpc_sfc_bulletin(bulletin):
    """Parse a coded surface bulletin from NWS WPC into a GeoPandas GeoDataFrame.

    Parameters
    ----------
    bulletin : str or file-like object
        If str, the name of the file to be opened. If `bulletin` is a file-like object,
        this will be read from directly.

    Returns
    -------
    parsed_text : geopandas.GeoDataFrame
        A GeoDataFrame where each row represents a pressure center or front. The GeoDataFrame
        has four columns: 'valid', 'feature', 'strength', and 'geometry'

    """
    parsed_text = geopandas.GeoDataFrame()

    # Create list with lines of text from file
    with contextlib.closing(open_as_needed(bulletin)) as file:
        text = file.read().decode('utf-8')
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # Remove empty strings

    # Lines that begin with numbers are simply continuations of the previous line. When we find
    # a line that begins with a number, we append it to the end of the last line.
    for i in range(len(lines)):
        line = lines[i]
        if re.match('^[0-9]', line):
            lines[i - 1] += f' {line}'
            i -= 1

    for line in lines:
        # A single file may have multiple sets of data that are valid at different times. Set
        # the valid_time string that will correspond to all the following lines parsed, until
        # the next valid_time is found.
        if line.startswith(('VALID', 'SURFACE PROG VALID')):
            valid_time = line.split(' ')[-1]
            continue

        feature, *info = line.split(' ')

        if feature in ['HIGHS', 'LOWS']:
            # Create list of tuples, each one representing a pressure center
            pres_centers = zip(info[::2], info[1::2])  # [a, b, c, d] -> [(a, b), (c, d)]

            # For each pressure center, add its data to a new row in the geodataframe
            for pres_center in pres_centers:
                strength, position = pres_center
                parsed_text = parsed_text.append({'valid': valid_time,
                                                  'feature': feature.rstrip('S'),
                                                  'strength': strength,
                                                  'geometry': Point(_decode_coords(position))},
                                                 ignore_index=True)

        elif feature in ['WARM', 'COLD', 'STNRY', 'OCFNT', 'TROF']:
            # Some bulletins include 'WK', 'MDT', or 'STG' to indicate the front's strength.
            # If present, separate it from the rest of the info, which gives the position of
            # the front.
            if re.match('^[A-Za-z]', info[0]):
                strength, *boundary = info
            else:
                strength, boundary = np.nan, info

            # Create a list of Point objects, and create Line object from points, if possible
            boundary = [Point(_decode_coords(point)) for point in boundary]
            boundary = LineString(boundary) if len(boundary) > 1 else boundary[0]

            # Add new row in the geodataframe for each front
            parsed_text = parsed_text.append({'valid': valid_time,
                                              'feature': feature.rstrip('S'),
                                              'strength': strength,
                                              'geometry': boundary},
                                             ignore_index=True)

    return parsed_text
