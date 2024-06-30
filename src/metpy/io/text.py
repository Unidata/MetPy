# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Support reading information from various text file formats."""

import contextlib
from datetime import datetime, timezone
import re
import string

import numpy as np
import pandas as pd

from ._tools import open_as_needed
from ..package_tools import Exporter

exporter = Exporter(globals())


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
    degrees west, respectively. Therefore, this function always returns latitude as a positive
    number and longitude as a negative number.

    Examples
    --------
    >>> _decode_coords('4731193')
    (-119.3, 47.3)

    """
    # Define latitude orientation
    flip = 1

    if coordinates[0] == '-':
        coordinates = coordinates[1:]
        # Flip latitude to Southern Hemisphere
        flip = -1

    # Based on the number of digits, find the correct place to split between lat and lon
    # Hires bulletins provide 7 digits for coordinates; regular bulletins provide 4 or 5 digits
    split_pos = int(len(coordinates) / 2)
    lat, lon = coordinates[:split_pos], coordinates[split_pos:]

    # Insert decimal point at the correct place and convert to float
    lat = float(f'{lat[:2]}.{lat[2:]}') * flip
    lon = -float(f'{lon[:3]}.{lon[3:]}')
    return lon, lat


def _regroup_lines(iterable):
    starting_num = re.compile('^[0-9]')
    lines = list(iterable)[::-1]
    while lines:
        line = lines.pop()
        if not line.strip():
            continue
        parts = line.split()
        while lines and starting_num.match(lines[-1]):
            parts.extend(lines.pop().split())
        yield parts


@exporter.export
def parse_wpc_surface_bulletin(bulletin, year=None):
    """Parse a coded surface bulletin from NWS WPC into a Pandas DataFrame.

    Parameters
    ----------
    bulletin : str or file-like object
        If str, the name of the file to be opened. If `bulletin` is a file-like object,
        this will be read from directly.

    Returns
    -------
    dataframe : pandas.DataFrame
        A `DataFrame` where each row represents a pressure center or front. The `DataFrame`
        has four columns: 'valid', 'feature', 'strength', and 'geometry'.
    year : int
        Year to assume when parsing the timestamp from the bulletin. Defaults to `None`,
        which results in the parser trying to find a year in the product header; if this
        search fails, the current year is assumed.

    """
    from shapely.geometry import LineString, Point

    # Create list with lines of text from file
    with contextlib.closing(open_as_needed(bulletin)) as file:
        text = file.read().decode('utf-8')

    parsed_text = []
    valid_time = datetime.now(timezone.utc).replace(tzinfo=None)
    for parts in _regroup_lines(text.splitlines()):
        # A single file may have multiple sets of data that are valid at different times. Set
        # the valid_time string that will correspond to all the following lines parsed, until
        # the next valid_time is found.
        if parts[0] in ('VALID', 'SURFACE PROG VALID'):
            dtstr = parts[-1]
            valid_time = valid_time.replace(year=year or valid_time.year, month=int(dtstr[:2]),
                                            day=int(dtstr[2:4]), hour=int(dtstr[4:6]),
                                            minute=0, second=0, microsecond=0)
        else:
            feature, *info = parts
            if feature in {'HIGHS', 'LOWS'}:
                # For each pressure center, add its data as a new row
                # While ideally these occur in pairs, some bulletins have had multiple
                # locations for a single center strength value. So instead walk one at a time
                # and keep track of the most recent strength.
                strength = np.nan
                for item in info:
                    if len(item) <= 4 and item[0] in {'8', '9', '1'}:
                        strength = int(item)
                    else:
                        parsed_text.append((valid_time, feature.rstrip('S'), strength,
                                            Point(_decode_coords(item))))
            elif feature in {'WARM', 'COLD', 'STNRY', 'OCFNT', 'TROF'}:
                # Some bulletins include 'WK', 'MDT', or 'STG' to indicate the front's
                # strength. If present, separate it from the rest of the info, which gives the
                # position of the front.
                if info[0][0] in string.ascii_letters:
                    strength, *boundary = info
                else:
                    strength, boundary = np.nan, info

                # Create a list of Points and create Line from points, if possible
                boundary = [Point(_decode_coords(point)) for point in boundary]
                boundary = LineString(boundary) if len(boundary) > 1 else boundary[0]

                # Add new row in the data for each front
                parsed_text.append((valid_time, feature, strength, boundary))
            # Look for a year at the end of the line (from the product header)
            elif (year is None and len(info) >= 2 and re.match(r'\d{4}', info[-1])
                  and re.match(r'\d{2}', info[-2])):
                with contextlib.suppress(ValueError):
                    year = int(info[-1])

    return pd.DataFrame(parsed_text, columns=['valid', 'feature', 'strength', 'geometry'])
