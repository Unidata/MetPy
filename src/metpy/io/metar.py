# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Parse METAR-formatted data."""
# Import the necessary libraries
from collections import namedtuple
import contextlib
from datetime import datetime, timezone
import warnings

import numpy as np
import pandas as pd

from ._metar_parser.metar_parser import parse, ParseError, TreeNode
from ._tools import open_as_needed
from .station_data import station_info
from ..package_tools import Exporter
from ..units import units

exporter = Exporter(globals())

# Configure the named tuple used for storing METAR data
Metar = namedtuple('metar', ['station_id', 'latitude', 'longitude', 'elevation', 'date_time',
                             'wind_direction', 'wind_speed', 'wind_gust', 'visibility',
                             'current_wx1', 'current_wx2', 'current_wx3', 'skyc1', 'skylev1',
                             'skyc2', 'skylev2', 'skyc3', 'skylev3', 'skyc4', 'skylev4',
                             'cloudcover', 'temperature', 'dewpoint', 'altimeter',
                             'current_wx1_symbol', 'current_wx2_symbol', 'current_wx3_symbol',
                             'remarks'])

# Create a dictionary for attaching units to the different variables
col_units = {'station_id': None,
             'latitude': 'degrees',
             'longitude': 'degrees',
             'elevation': 'meters',
             'date_time': None,
             'wind_direction': 'degrees',
             'wind_speed': 'kts',
             'wind_gust': 'kts',
             'visibility': 'meters',
             'eastward_wind': 'kts',
             'northward_wind': 'kts',
             'current_wx1': None,
             'current_wx2': None,
             'current_wx3': None,
             'low_cloud_type': None,
             'low_cloud_level': 'feet',
             'medium_cloud_type': None,
             'medium_cloud_level': 'feet',
             'high_cloud_type': None,
             'high_cloud_level': 'feet',
             'highest_cloud_type': None,
             'highest_cloud_level:': None,
             'cloud_coverage': None,
             'air_temperature': 'degC',
             'dew_point_temperature': 'degC',
             'altimeter': 'inHg',
             'air_pressure_at_sea_level': 'hPa',
             'current_wx1_symbol': None,
             'current_wx2_symbol': None,
             'current_wx3_symbol': None}


def _tree_repr_(self):
    """Produce string representation of a TreeNodex object."""
    rep = self.__class__.__name__ + '('
    args = []
    for key, value in self.__dict__.items():
        if key == 'elements':
            continue
        args.append(key + '=' + repr(value))
        dict_str = ', '.join(args) + ')'
    rep += dict_str
    return rep


# Monkey patch to improve debugging
TreeNode.__repr__ = _tree_repr_


@exporter.export
def parse_metar_to_dataframe(metar_text, *, year=None, month=None):
    """Parse a single METAR report into a Pandas DataFrame.

    Takes a METAR string in a text form, and creates a `pandas.DataFrame` including the
    essential information (not including the remarks)

    The parser follows the WMO format, allowing for missing data and assigning
    nan values where necessary. The WMO code is also provided for current weather,
    which can be utilized when plotting.

    Parameters
    ----------
    metar_text : str
        The METAR report
    year : int, optional
        Year in which observation was taken, defaults to current year. Keyword-only argument.
    month : int, optional
        Month in which observation was taken, defaults to current month. Keyword-only argument.

    Returns
    -------
    `pandas.DataFrame`

    """
    return _metars_to_dataframe([metar_text], year=year, month=month)


def parse_metar(metar_text, year, month, station_metadata=station_info):
    """Parse a METAR report in text form into a list of named tuples.

    Parameters
    ----------
    metar_text : str
        The METAR report
    station_metadata : dict
        Mapping of station identifiers to station metadata
    year : int
        Reported year of observation for constructing 'date_time'
    month : int
        Reported month of observation for constructing 'date_time'

    Returns
    -------
    metar : namedtuple
        Named tuple of parsed METAR fields

    Notes
    -----
    Returned data has named tuples with the following attributes:

    * 'station_id': Station Identifier (ex. KLOT)
    * 'latitude': Latitude of the observation, measured in degrees
    * 'longitude': Longitude of the observation, measured in degrees
    * 'elevation': Elevation of the observation above sea level, measured in meters
    * 'date_time': Date and time of the observation, datetime object
    * 'wind_direction': Direction the wind is coming from, measured in degrees
    * 'wind_speed': Wind speed, measured in knots
    * 'wind_gust': Wind gust, measured in knots
    * 'current_wx1': Current weather (1 of 3)
    * 'current_wx2': Current weather (2 of 3)
    * 'current_wx3': Current weather (3 of 3)
    * 'skyc1': Sky cover (ex. FEW)
    * 'skylev1': Height of sky cover 1, measured in feet
    * 'skyc2': Sky cover (ex. OVC)
    * 'skylev2': Height of sky cover 2, measured in feet
    * 'skyc3': Sky cover (ex. FEW)
    * 'skylev3': Height of sky cover 3, measured in feet
    * 'skyc4': Sky cover (ex. CLR)
    * 'skylev4:': Height of sky cover 4, measured in feet
    * 'cloudcover': Cloud coverage measured in oktas, taken from maximum of sky cover values
    * 'temperature': Temperature, measured in degrees Celsius
    * 'dewpoint': Dewpoint, measured in degrees Celsius
    * 'altimeter': Altimeter value, measured in inches of mercury
    * 'current_wx1_symbol': Current weather symbol (1 of 3), WMO integer code from [WMO306]_
      Attachment IV
    * 'current_wx2_symbol': Current weather symbol (2 of 3), WMO integer code from [WMO306]_
      Attachment IV
    * 'current_wx3_symbol': Current weather symbol (3 of 3), WMO integer code from [WMO306]_
      Attachment IV
    * 'visibility': Visibility distance, measured in meters
    * 'remarks': Remarks (unparsed) in the report

    """
    from ..plots.wx_symbols import wx_code_to_numeric

    # Decode the data using the parser (built using Canopy) the parser utilizes a grammar
    # file which follows the format structure dictated by the WMO Handbook, but has the
    # flexibility to decode the METAR text when there are missing or incorrectly
    # encoded values
    tree = parse(metar_text)

    # Station ID which is used to find the latitude, longitude, and elevation
    station_id = tree.siteid.text.strip()

    # Extract the latitude and longitude values from 'master' dictionary
    try:
        info = station_metadata[station_id]
        lat = info.latitude
        lon = info.longitude
        elev = info.altitude
    except KeyError:
        lat = np.nan
        lon = np.nan
        elev = np.nan

    # Set the datetime, day, and time_utc
    try:
        day_time_utc = tree.datetime.text.strip()
        day = int(day_time_utc[0:2])
        hour = int(day_time_utc[2:4])
        minute = int(day_time_utc[4:6])
        date_time = datetime(year, month, day, hour, minute)
    except ValueError:
        date_time = np.nan

    # Set the wind values
    wind_units = 'kts'
    try:
        # If there are missing wind values, set wind speed and wind direction to nan
        if ('/' in tree.wind.text) or (tree.wind.text == 'KT') or (tree.wind.text == ''):
            wind_dir = np.nan
            wind_spd = np.nan
        # If the wind direction is variable, set wind direction to nan but keep the wind speed
        else:
            wind_spd = float(tree.wind.wind_spd.text)
            if 'MPS' in tree.wind.text:
                wind_units = 'm/s'
                wind_spd = units.Quantity(wind_spd, wind_units).m_as('knots')
            if (tree.wind.wind_dir.text == 'VRB') or (tree.wind.wind_dir.text == 'VAR'):
                wind_dir = np.nan
            else:
                wind_dir = int(tree.wind.wind_dir.text)
    # If there are any errors, return nan
    except ValueError:
        wind_dir = np.nan
        wind_spd = np.nan

    # Parse out the wind gust field
    if 'G' in tree.wind.text:
        wind_gust = units.Quantity(float(tree.wind.gust.text.strip()[1:]),
                                   wind_units).m_as('knots')
    else:
        wind_gust = np.nan

    # Handle visibility
    try:
        if tree.vis.text.endswith('SM'):
            visibility = 0
            # Strip off the SM and any whitespace around the value and any leading 'M'
            vis_str = tree.vis.text[:-2].strip().lstrip('M')

            # Case of e.g. 1 1/4SM
            if ' ' in vis_str:
                whole, vis_str = vis_str.split(maxsplit=1)
                visibility += int(whole)

            # Handle fraction regardless
            if '/' in vis_str:
                num, denom = vis_str.split('/', maxsplit=1)
                if int(denom) == 0:
                    raise ValueError('Visibility denominator is 0.')
                visibility += int(num) / int(denom)
            else:  # Should be getting all cases of whole number without fraction
                visibility += int(vis_str)
            visibility = units.Quantity(visibility, 'miles').m_as('meter')
        # CAVOK means vis is "at least 10km" and no significant clouds or weather
        elif 'CAVOK' in tree.vis.text:
            visibility = 10000
        elif not tree.vis.text or tree.vis.text.strip() == '////':
            visibility = np.nan
        else:
            # Only worry about the first 4 characters (digits) and ignore possible 'NDV'
            visibility = int(tree.vis.text.strip()[:4])
    # If there are any errors, return nan
    except ValueError:
        visibility = np.nan

    # Set the weather symbols
    # If the weather symbol is missing, set values to nan
    current_wx = []
    current_wx_symbol = []
    if tree.curwx.text.strip() not in ('', '//', 'NSW'):
        current_wx = tree.curwx.text.strip().split()

        # Handle having e.g. '+' and 'TSRA' parsed into separate items
        if current_wx[0] in ('-', '+') and current_wx[1]:
            current_wx[0] += current_wx[1]
            current_wx.pop(1)

        current_wx_symbol = wx_code_to_numeric(current_wx).tolist()
    while len(current_wx) < 3:
        current_wx.append(np.nan)
    while len(current_wx_symbol) < 3:
        current_wx_symbol.append(0)

    # Set the sky conditions
    skyc = [np.nan] * 4
    skylev = [np.nan] * 4
    if tree.skyc.text[1:3] == 'VV':
        skyc[0] = 'VV'
        level = tree.skyc.text.strip()[2:5]
        skylev[0] = np.nan if not level or '/' in level else 100 * int(level)
    else:
        for ind, part in enumerate(tree.skyc.text.strip().split(maxsplit=3)):
            cover = part[:3]
            level = part[3:6]  # Strips off any ending text like in FEW017CB
            if '/' not in cover:
                skyc[ind] = cover
            if level and '/' not in level:
                with contextlib.suppress(ValueError):
                    skylev[ind] = float(level) * 100

    # Set the cloud cover variable (measured in oktas)
    if 'OVC' in tree.skyc.text or 'VV' in tree.skyc.text:
        cloudcover = 8
    elif 'BKN' in tree.skyc.text:
        cloudcover = 6
    elif 'SCT' in tree.skyc.text:
        cloudcover = 4
    elif 'FEW' in tree.skyc.text:
        cloudcover = 2
    elif ('SKC' in tree.skyc.text or 'NCD' in tree.skyc.text or 'NSC' in tree.skyc.text
          or 'CLR' in tree.skyc.text or 'CAVOK' in tree.vis.text):
        cloudcover = 0
    else:
        cloudcover = 10

    # Set the temperature and dewpoint
    temp = np.nan
    dewp = np.nan
    if tree.temp_dewp.text and tree.temp_dewp.text != ' MM/MM':
        with contextlib.suppress(ValueError):
            temp = float(tree.temp_dewp.temp.text[-2:])
            if 'M' in tree.temp_dewp.temp.text:
                temp *= -1

        with contextlib.suppress(ValueError):
            dewp = float(tree.temp_dewp.dewp.text[-2:])
            if 'M' in tree.temp_dewp.dewp.text:
                dewp *= -1

    # Set the altimeter value and sea level pressure
    if tree.altim.text:
        val = float(tree.altim.text.strip()[1:5])
        altim = val / 100 if val > 1100 else units.Quantity(val, 'hPa').m_as('inHg')
    else:
        altim = np.nan

    # Strip off extraneous stuff off the remarks section
    remarks = tree.remarks.text.lstrip().rstrip('= ')
    if remarks.startswith('RMK'):
        remarks = remarks[3:].strip()

    # Returns a named tuple with all the relevant variables
    return Metar(station_id, lat, lon, elev, date_time, wind_dir, wind_spd, wind_gust,
                 visibility, current_wx[0], current_wx[1], current_wx[2], skyc[0], skylev[0],
                 skyc[1], skylev[1], skyc[2], skylev[2], skyc[3], skylev[3], cloudcover, temp,
                 dewp, altim, current_wx_symbol[0], current_wx_symbol[1], current_wx_symbol[2],
                 remarks)


@exporter.export
def parse_metar_file(filename, *, year=None, month=None):
    """Parse a text file containing multiple METAR reports and/or text products.

    Parameters
    ----------
    filename : str or file-like object
        If str, the name of the file to be opened. If `filename` is a file-like object,
        this will be read from directly.
    year : int, optional
        Year in which observation was taken, defaults to current year. Keyword-only argument.
    month : int, optional
        Month in which observation was taken, defaults to current month. Keyword-only argument.

    Returns
    -------
    `pandas.DataFrame`

    """
    # Function to merge METARs
    def full_metars(x, prefix='     '):
        tmp = []
        for i in x:
            # Skip any blank lines
            if not i.strip():
                continue
            # No prefix signals a new report, so yield
            if not i.startswith(prefix) and tmp:
                yield ' '.join(tmp)
                tmp = []
            tmp.append(i.strip())

        # Handle any leftovers
        if tmp:
            yield ' '.join(tmp)

    # Open the file
    with contextlib.closing(open_as_needed(filename, 'rt')) as myfile:
        # Merge multi-line METARs into a single report--drop reports that are too short to
        # be a METAR with a robust amount of data.
        return _metars_to_dataframe(filter(lambda m: len(m) > 25, full_metars(myfile)),
                                    year=year, month=month)


def _metars_to_dataframe(metar_iter, *, year=None, month=None):
    """Turn an iterable of METAR reports into a DataFrame.

    Notes
    -----
    The output has the following columns:

    * 'station_id': Station Identifier (ex. KLOT)
    * 'latitude': Latitude of the observation, measured in degrees
    * 'longitude': Longitude of the observation, measured in degrees
    * 'elevation': Elevation of the observation above sea level, measured in meters
    * 'date_time': Date and time of the observation, datetime object
    * 'wind_direction': Direction the wind is coming from, measured in degrees
    * 'wind_speed': Wind speed, measured in knots
    * 'wind_gust': Wind gust, measured in knots
    * 'visibility': Visibility distance, measured in meters
    * 'current_wx1': Current weather (1 of 3)
    * 'current_wx2': Current weather (2 of 3)
    * 'current_wx3': Current weather (3 of 3)
    * 'low_cloud_type': Low-level sky cover (ex. FEW)
    * 'low_cloud_level': Height of low-level sky cover, measured in feet
    * 'medium_cloud_type': Medium-level sky cover (ex. OVC)
    * 'medium_cloud_level': Height of medium-level sky cover, measured in feet
    * 'high_cloud_type': High-level sky cover (ex. FEW)
    * 'high_cloud_level': Height of high-level sky cover, measured in feet
    * 'highest_cloud_type': Highest-level Sky cover (ex. CLR)
    * 'highest_cloud_level:': Height of highest-level sky cover, measured in feet
    * 'cloud_coverage': Cloud cover measured in oktas, taken from maximum of sky cover values
    * 'air_temperature': Temperature, measured in degrees Celsius
    * 'dew_point_temperature': Dew point, measured in degrees Celsius
    * 'altimeter': Altimeter value, measured in inches of mercury
    * 'remarks': Any remarks section in the report
    * 'current_wx1_symbol': Current weather symbol (1 of 3), WMO integer code from [WMO306]_
      Attachment IV
    * 'current_wx2_symbol': Current weather symbol (2 of 3), WMO integer code from [WMO306]_
      Attachment IV
    * 'current_wx3_symbol': Current weather symbol (3 of 3), WMO integer code from [WMO306]_
      Attachment IV
    * 'air_pressure_at_sea_level': Sea level pressure, derived from temperature, elevation
      and altimeter value
    * 'eastward_wind': Eastward component (u-component) of the wind vector, measured in knots
    * 'northward_wind': Northward component (v-component) of the wind vector, measured in knots

    """
    from ..calc import altimeter_to_sea_level_pressure, wind_components

    # Defaults year and/or month to present reported date if not provided
    if year is None or month is None:
        now = datetime.now(timezone.utc)
        year = now.year if year is None else year
        month = now.month if month is None else month

    # Try to parse each METAR that is given
    metars = []
    for metar in metar_iter:
        with contextlib.suppress(ParseError):
            # Parse the string of text and assign to values within the named tuple
            metars.append(parse_metar(metar, year=year, month=month))

    # Take the list of Metar objects and turn it into a DataFrame with appropriate columns
    df = pd.DataFrame(metars)
    df.set_index('station_id', inplace=True, drop=False)
    df.rename(columns={'skyc1': 'low_cloud_type', 'skylev1': 'low_cloud_level',
                       'skyc2': 'medium_cloud_type', 'skylev2': 'medium_cloud_level',
                       'skyc3': 'high_cloud_type', 'skylev3': 'high_cloud_level',
                       'skyc4': 'highest_cloud_type', 'skylev4': 'highest_cloud_level',
                       'cloudcover': 'cloud_coverage', 'temperature': 'air_temperature',
                       'dewpoint': 'dew_point_temperature'}, inplace=True)

    # Drop duplicate values
    df.drop_duplicates(subset=['date_time', 'latitude', 'longitude'], keep='last',
                       inplace=True)

    # Calculate sea-level pressure from function in metpy.calc
    df['air_pressure_at_sea_level'] = altimeter_to_sea_level_pressure(
        units.Quantity(df.altimeter.values, col_units['altimeter']),
        units.Quantity(df.elevation.values, col_units['elevation']),
        units.Quantity(df.air_temperature.values, col_units['air_temperature'])).m_as('hPa')

    # Use get wind components and assign them to eastward and northward winds
    u, v = wind_components(
        units.Quantity(df.wind_speed.values, col_units['wind_speed']),
        units.Quantity(df.wind_direction.values, col_units['wind_direction']))
    df['eastward_wind'] = u.m
    df['northward_wind'] = v.m

    # Round altimeter and sea-level pressure values
    df['altimeter'] = df.altimeter.round(2)
    df['air_pressure_at_sea_level'] = df.air_pressure_at_sea_level.round(2)

    # Set the units for the dataframe--filter out warning from Pandas
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        df.units = col_units

    return df


# Patch in the notes section into the two main parse functions
if _metars_to_dataframe.__doc__:
    # Finds the Notes snippet in the docstring
    snippet = _metars_to_dataframe.__doc__[_metars_to_dataframe.__doc__.find('Notes'):]
    parse_metar_file.__doc__ = parse_metar_file.__doc__ + snippet
    parse_metar_to_dataframe.__doc__ = parse_metar_to_dataframe.__doc__ + snippet
