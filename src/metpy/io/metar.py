# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Parse METAR-formatted data."""
# Import the necessary libraries
from collections import namedtuple
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

from ._tools import open_as_needed
from .metar_parser import parse, ParseError
from .station_data import station_info
from ..calc import altimeter_to_sea_level_pressure, wind_components
from ..package_tools import Exporter
from ..plots.wx_symbols import wx_code_map
from ..units import pandas_dataframe_to_unit_arrays, units

exporter = Exporter(globals())

# Ignore the pandas warning
warnings.filterwarnings('ignore', "Pandas doesn't allow columns to be created", UserWarning)

# Configure the named tuple used for storing METAR data
Metar = namedtuple('metar', ['station_id', 'latitude', 'longitude', 'elevation',
                             'date_time', 'wind_direction', 'wind_speed', 'current_wx1',
                             'current_wx2', 'current_wx3', 'skyc1', 'skylev1', 'skyc2',
                             'skylev2', 'skyc3', 'skylev3', 'skyc4', 'skylev4',
                             'cloudcover', 'temperature', 'dewpoint', 'altimeter',
                             'current_wx1_symbol', 'current_wx2_symbol',
                             'current_wx3_symbol'])

# Create a dictionary for attaching units to the different variables
col_units = {'station_id': None,
             'latitude': 'degrees',
             'longitude': 'degrees',
             'elevation': 'meters',
             'date_time': None,
             'wind_direction': 'degrees',
             'wind_speed': 'kts',
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
             'present_weather': None,
             'past_weather': None,
             'past_weather2': None}


@exporter.export
def parse_metar_to_dataframe(metar_text, year=datetime.now().year, month=datetime.now().month):
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
        Year in which observation was taken, defaults to the current year
    month : int, optional
        Month in which observation was taken, defaults to the current month

    Returns
    -------
    `pandas.DataFrame`

    Notes
    -----
    The output has the following columns:
    'station_id': Station Identifier (ex. KLOT)
    'latitude': Latitude of the observation, measured in degrees
    'longitude': Longitude of the observation, measured in degrees
    'elevation': Elevation of the observation above sea level, measured in meters
    'date_time': Date and time of the observation, datetime object
    'wind_direction': Direction the wind is coming from, measured in degrees
    'wind_spd': Wind speed, measured in knots
    'current_wx1': Current weather (1 of 3)
    'current_wx2': Current weather (2 of 3)
    'current_wx3': Current weather (3 of 3)
    'skyc1': Sky cover (ex. FEW)
    'skylev1': Height of sky cover 1, measured in feet
    'skyc2': Sky cover (ex. OVC)
    'skylev2': Height of sky cover 2, measured in feet
    'skyc3': Sky cover (ex. FEW)
    'skylev3': Height of sky cover 3, measured in feet
    'skyc4': Sky cover (ex. CLR)
    'skylev4:': Height of sky cover 4, measured in feet
    'cloudcover': Cloud coverage measured in oktas, taken from maximum of sky cover values
    'temperature': Temperature, measured in degrees Celsius
    'dewpoint': Dew point, measured in degrees Celsius
    'altimeter': Altimeter value, measured in inches of mercury, float
    'current_wx1_symbol': Current weather symbol (1 of 3), integer
    'current_wx2_symbol': Current weather symbol (2 of 3), integer
    'current_wx3_symbol': Current weather symbol (3 of 3), integer
    'sea_level_pressure': Sea level pressure, derived from temperature, elevation
    and altimeter value, float

    """
    # Use the named tuple parsing function to separate metar
    # Utilizes the station dictionary which contains elevation, latitude, and longitude
    metar_vars = parse_metar_to_named_tuple(metar_text, station_info, year, month)

    # Use a pandas dataframe to store the data
    df = pd.DataFrame({'station_id': metar_vars.station_id,
                       'latitude': metar_vars.latitude,
                       'longitude': metar_vars.longitude,
                       'elevation': metar_vars.elevation,
                       'date_time': metar_vars.date_time,
                       'wind_direction': metar_vars.wind_direction,
                       'wind_speed': metar_vars.wind_speed,
                       'current_wx1': metar_vars.current_wx1,
                       'current_wx2': metar_vars.current_wx2,
                       'current_wx3': metar_vars.current_wx3,
                       'low_cloud_type': metar_vars.skyc1,
                       'low_cloud_level': metar_vars.skylev1,
                       'medium_cloud_type': metar_vars.skyc2,
                       'medium_cloud_level': metar_vars.skylev2,
                       'high_cloud_type': metar_vars.skyc3,
                       'high_cloud_level': metar_vars.skylev3,
                       'highest_cloud_type': metar_vars.skyc4,
                       'highest_cloud_level': metar_vars.skylev4,
                       'cloud_coverage': metar_vars.cloudcover,
                       'air_temperature': metar_vars.temperature,
                       'dew_point_temperature': metar_vars.dewpoint,
                       'altimeter': metar_vars.altimeter,
                       'present_weather': metar_vars.current_wx1_symbol,
                       'past_weather': metar_vars.current_wx2_symbol,
                       'past_weather2': metar_vars.current_wx3_symbol},
                      index=[metar_vars.station_id])

    # Convert to sea level pressure using calculation in metpy.calc
    try:
        # Create a field for sea-level pressure and make sure it is a float
        df['air_pressure_at_sea_level'] = float(altimeter_to_sea_level_pressure(
            df.altimeter.values * units('inHg'),
            df.elevation.values * units('meters'),
            df.temperature.values * units('degC')).to('hPa').magnitude)
    except AttributeError:
        df['air_pressure_at_sea_level'] = [np.nan]

    # Use get wind components and assign them to u and v variables
    df['eastward_wind'], df['northward_wind'] = wind_components((df.wind_speed.values
                                                                 * units.kts),
                                                                df.wind_direction.values
                                                                * units.degree)

    # Round the altimeter and sea-level pressure values
    df['altimeter'] = df.altimeter.round(2)
    df['air_pressure_at_sea_level'] = df.air_pressure_at_sea_level.round(2)

    # Set the units for the dataframe
    df.units = col_units

    # Add the array for units to the dataframe
    pandas_dataframe_to_unit_arrays(df)

    # Return the dataframe
    return df


def parse_metar_to_named_tuple(metar_text, station_metadata, year=datetime.now().year,
                               month=datetime.now().month):
    """Parse a METAR report in text form into a list of named tuples.

    Parameters
    ----------
    metar_text : str
        The METAR report
    station_metadata : dict
        Mapping of station identifiers to station metadata

    Returns
    -------
    `pandas.DataFrame`

    Notes
    -----
    Returned data has named tuples with the following attributes:
    'station_id': Station Identifier (ex. KLOT)
    'latitude': Latitude of the observation, measured in degrees
    'longitude': Longitude of the observation, measured in degrees
    'elevation': Elevation of the observation above sea level, measured in meters
    'date_time': Date and time of the observation, datetime object
    'wind_direction': Direction the wind is coming from, measured in degrees
    'wind_spd': Wind speed, measured in knots
    'current_wx1': Current weather (1 of 3)
    'current_wx2': Current weather (2 of 3)
    'current_wx3': Current weather (3 of 3)
    'skyc1': Sky cover (ex. FEW)
    'skylev1': Height of sky cover 1, measured in feet
    'skyc2': Sky cover (ex. OVC)
    'skylev2': Height of sky cover 2, measured in feet
    'skyc3': Sky cover (ex. FEW)
    'skylev3': Height of sky cover 3, measured in feet
    'skyc4': Sky cover (ex. CLR)
    'skylev4:': Height of sky cover 4, measured in feet
    'cloudcover': Cloud coverage measured in oktas, taken from maximum of sky cover values
    'temperature': Temperature, measured in degrees Celsius
    'dewpoint': Dewpoint, measured in degrees Celsius
    'altimeter': Altimeter value, measured in inches of mercury, float
    'current_wx1_symbol': Current weather symbol (1 of 3), integer
    'current_wx2_symbol': Current weather symbol (2 of 3), integer
    'current_wx3_symbol': Current weather symbol (3 of 3), integer
    'sea_level_pressure': Sea level pressure, derived from temperature, elevation
    and altimeter value, float

    """
    # Decode the data using the parser (built using Canopy) the parser utilizes a grammar
    # file which follows the format structure dictated by the WMO Handbook, but has the
    # flexibility to decode the METAR text when there are missing or incorrectly
    # encoded values
    tree = parse(metar_text)

    # Station ID which is used to find the latitude, longitude, and elevation
    station_id = tree.siteid.text.strip()

    # Extract the latitude and longitude values from 'master' dictionary
    try:
        lat = station_metadata[tree.siteid.text.strip()].latitude
        lon = station_metadata[tree.siteid.text.strip()].longitude
        elev = station_metadata[tree.siteid.text.strip()].altitude
    except KeyError:
        lat = np.nan
        lon = np.nan
        elev = np.nan

    # Set the datetime, day, and time_utc
    try:
        day_time_utc = tree.datetime.text[:-1].strip()
        day = int(day_time_utc[0:2])
        hour = int(day_time_utc[2:4])
        minute = int(day_time_utc[4:7])
        date_time = datetime(year, month, day, hour, minute)
    except (AttributeError, ValueError):
        date_time = np.nan

    # Set the wind values
    try:
        # If there are missing wind values, set wind speed and wind direction to nan
        if ('/' in tree.wind.text) or (tree.wind.text == 'KT') or (tree.wind.text == ''):
            wind_dir = np.nan
            wind_spd = np.nan
        # If the wind direction is variable, set wind direction to nan but keep the wind speed
        else:
            if (tree.wind.wind_dir.text == 'VRB') or (tree.wind.wind_dir.text == 'VAR'):
                wind_dir = np.nan
                wind_spd = float(tree.wind.wind_spd.text)
            else:
                # If the wind speed and direction is given, keep the values
                wind_dir = int(tree.wind.wind_dir.text)
                wind_spd = int(tree.wind.wind_spd.text)
    # If there are any errors, return nan
    except ValueError:
        wind_dir = np.nan
        wind_spd = np.nan

    # Set the weather symbols
    # If the weather symbol is missing, set values to nan
    if tree.curwx.text == '':
        current_wx1 = np.nan
        current_wx2 = np.nan
        current_wx3 = np.nan
        current_wx1_symbol = 0
        current_wx2_symbol = 0
        current_wx3_symbol = 0
    else:
        wx = [np.nan, np.nan, np.nan]
        # Loop through symbols and assign according WMO codes
        wx[0:len((tree.curwx.text.strip()).split())] = tree.curwx.text.strip().split()
        current_wx1 = wx[0]
        current_wx2 = wx[1]
        current_wx3 = wx[2]
        try:
            current_wx1_symbol = int(wx_code_map[wx[0]])
        except (IndexError, KeyError):
            current_wx1_symbol = 0
        try:
            current_wx2_symbol = int(wx_code_map[wx[1]])
        except (IndexError, KeyError):
            current_wx2_symbol = 0
        try:
            current_wx3_symbol = int(wx_code_map[wx[3]])
        except (IndexError, KeyError):
            current_wx3_symbol = 0

    # Set the sky conditions
    if tree.skyc.text[1:3] == 'VV':
        skyc1 = 'VV'
        skylev1 = tree.skyc.text.strip()[2:]
        skyc2 = np.nan
        skylev2 = np.nan
        skyc3 = np.nan
        skylev3 = np.nan
        skyc4 = np.nan
        skylev4 = np.nan

    else:
        skyc = []
        skyc[0:len((tree.skyc.text.strip()).split())] = tree.skyc.text.strip().split()
        try:
            skyc1 = skyc[0][0:3]
            if '/' in skyc1:
                skyc1 = np.nan
        except (IndexError, ValueError, TypeError):
            skyc1 = np.nan
        try:
            skylev1 = skyc[0][3:]
            if '/' in skylev1:
                skylev1 = np.nan
            else:
                skylev1 = float(skylev1) * 100
        except (IndexError, ValueError, TypeError):
            skylev1 = np.nan
        try:
            skyc2 = skyc[1][0:3]
            if '/' in skyc2:
                skyc2 = np.nan
        except (IndexError, ValueError, TypeError):
            skyc2 = np.nan
        try:
            skylev2 = skyc[1][3:]
            if '/' in skylev2:
                skylev2 = np.nan
            else:
                skylev2 = float(skylev2) * 100
        except (IndexError, ValueError, TypeError):
            skylev2 = np.nan
        try:
            skyc3 = skyc[2][0:3]
            if '/' in skyc3:
                skyc3 = np.nan
        except (IndexError, ValueError):
            skyc3 = np.nan
        try:
            skylev3 = skyc[2][3:]
            if '/' in skylev3:
                skylev3 = np.nan
            else:
                skylev3 = float(skylev3) * 100
        except (IndexError, ValueError, TypeError):
            skylev3 = np.nan
        try:
            skyc4 = skyc[3][0:3]
            if '/' in skyc4:
                skyc4 = np.nan
        except (IndexError, ValueError, TypeError):
            skyc4 = np.nan
        try:
            skylev4 = skyc[3][3:]
            if '/' in skylev4:
                skylev4 = np.nan
            else:
                skylev4 = float(skylev4) * 100
        except (IndexError, ValueError, TypeError):
            skylev4 = np.nan

    # Set the cloud cover variable (measured in oktas)
    if ('OVC' or 'VV') in tree.skyc.text:
        cloudcover = 8
    elif 'BKN' in tree.skyc.text:
        cloudcover = 6
    elif 'SCT' in tree.skyc.text:
        cloudcover = 4
    elif 'FEW' in tree.skyc.text:
        cloudcover = 2
    elif ('SKC' in tree.skyc.text) or ('NCD' in tree.skyc.text) \
            or ('NSC' in tree.skyc.text) or 'CLR' in tree.skyc.text:
        cloudcover = 0
    else:
        cloudcover = 10

    # Set the temperature and dewpoint
    if (tree.temp_dewp.text == '') or (tree.temp_dewp.text == ' MM/MM'):
        temp = np.nan
        dewp = np.nan
    else:
        try:
            if 'M' in tree.temp_dewp.temp.text:
                temp = (-1 * float(tree.temp_dewp.temp.text[-2:]))
            else:
                temp = float(tree.temp_dewp.temp.text[-2:])
        except ValueError:
            temp = np.nan
        try:
            if 'M' in tree.temp_dewp.dewp.text:
                dewp = (-1 * float(tree.temp_dewp.dewp.text[-2:]))
            else:
                dewp = float(tree.temp_dewp.dewp.text[-2:])
        except ValueError:
            dewp = np.nan

    # Set the altimeter value and sea level pressure
    if tree.altim.text == '':
        altim = np.nan
    else:
        if (float(tree.altim.text.strip()[1:5])) > 1100:
            altim = float(tree.altim.text.strip()[1:5]) / 100
        else:
            altim = (int(tree.altim.text.strip()[1:5]) * units.hPa).to('inHg').magnitude

    # Returns a named tuple with all the relevant variables
    return Metar(station_id, lat, lon, elev, date_time, wind_dir, wind_spd,
                 current_wx1, current_wx2, current_wx3, skyc1, skylev1, skyc2,
                 skylev2, skyc3, skylev3, skyc4, skylev4, cloudcover, temp, dewp,
                 altim, current_wx1_symbol, current_wx2_symbol, current_wx3_symbol)


@exporter.export
def parse_metar_file(filename, year=datetime.now().year, month=datetime.now().month):
    """Parse a text file containing multiple METAR reports and/or text products.

    Parameters
    ----------
    filename : str or file-like object
        If str, the name of the file to be opened. If `filename` is a file-like object,
        this will be read from directly.
    year : int, optional
        Year in which observation was taken, defaults to the current year
    month : int, optional
        Month in which observation was taken, defaults to the current month

    Returns
    -------
    `pandas.DataFrame`

    Notes
    -----
    The returned `pandas.DataFrame` has the following columns:
    'station_id': Station Identifier (ex. KLOT)
    'latitude': Latitude of the observation, measured in degrees
    'longitude': Longitude of the observation, measured in degrees
    'elevation': Elevation of the observation above sea level, measured in meters
    'date_time': Date and time of the observation, datetime object
    'wind_direction': Direction the wind is coming from, measured in degrees
    'wind_spd': Wind speed, measured in knots
    'current_wx1': Current weather (1 of 3)
    'current_wx2': Current weather (2 of 3)
    'current_wx3': Current weather (3 of 3)
    'skyc1': Sky cover (ex. FEW)
    'skylev1': Height of sky cover 1, measured in feet
    'skyc2': Sky cover (ex. OVC)
    'skylev2': Height of sky cover 2, measured in feet
    'skyc3': Sky cover (ex. FEW)
    'skylev3': Height of sky cover 3, measured in feet
    'skyc4': Sky cover (ex. CLR)
    'skylev4:': Height of sky cover 4, measured in feet
    'cloudcover': Cloud coverage measured in oktas, taken from maximum of sky cover values
    'temperature': Temperature, measured in degrees Celsius
    'dewpoint': Dew point, measured in degrees Celsius
    'altimeter': Altimeter value, measured in inches of mercury, float
    'current_wx1_symbol': Current weather symbol (1 of 3), integer
    'current_wx2_symbol': Current weather symbol (2 of 3), integer
    'current_wx3_symbol': Current weather symbol (3 of 3), integer
    'sea_level_pressure': Sea level pressure, derived from temperature, elevation
    and altimeter value, float

    """
    # Function to merge METARs
    def merge(x, key='     '):
        tmp = []
        for i in x:
            if (i[0:len(key)] != key) and len(tmp):
                yield ' '.join(tmp)
                tmp = []
            if i.startswith(key):
                i = i[5:]
            tmp.append(i)
        if len(tmp):
            yield ' '.join(tmp)

    # Open the file
    myfile = open_as_needed(filename, 'rt')

    # Clean up the file and take out the next line (\n)
    value = myfile.read().rstrip()
    list_values = value.split('\n')
    list_values = list(filter(None, list_values))

    # Call the merge function and assign the result to the list of metars
    list_values = list(merge(list_values))

    # Remove the short lines that do not contain METAR observations or contain
    # METAR observations that lack a robust amount of data
    metars = []
    for metar in list_values:
        if len(metar) > 25:
            metars.append(metar)
        else:
            continue

    # Create a dictionary with all the station name, locations, and elevations
    master = station_info

    # Setup lists to append the data to
    station_id = []
    lat = []
    lon = []
    elev = []
    date_time = []
    wind_dir = []
    wind_spd = []
    current_wx1 = []
    current_wx2 = []
    current_wx3 = []
    skyc1 = []
    skylev1 = []
    skyc2 = []
    skylev2 = []
    skyc3 = []
    skylev3 = []
    skyc4 = []
    skylev4 = []
    cloudcover = []
    temp = []
    dewp = []
    altim = []
    current_wx1_symbol = []
    current_wx2_symbol = []
    current_wx3_symbol = []

    # Loop through the different metars within the text file
    for metar in metars:
        try:
            # Parse the string of text and assign to values within the named tuple
            metar = parse_metar_to_named_tuple(metar, master, year=year, month=month)

            # Append the different variables to their respective lists
            station_id.append(metar.station_id)
            lat.append(metar.latitude)
            lon.append(metar.longitude)
            elev.append(metar.elevation)
            date_time.append(metar.date_time)
            wind_dir.append(metar.wind_direction)
            wind_spd.append(metar.wind_speed)
            current_wx1.append(metar.current_wx1)
            current_wx2.append(metar.current_wx2)
            current_wx3.append(metar.current_wx3)
            skyc1.append(metar.skyc1)
            skylev1.append(metar.skylev1)
            skyc2.append(metar.skyc2)
            skylev2.append(metar.skylev2)
            skyc3.append(metar.skyc3)
            skylev3.append(metar.skylev3)
            skyc4.append(metar.skyc4)
            skylev4.append(metar.skylev4)
            cloudcover.append(metar.cloudcover)
            temp.append(metar.temperature)
            dewp.append(metar.dewpoint)
            altim.append(metar.altimeter)
            current_wx1_symbol.append(metar.current_wx1_symbol)
            current_wx2_symbol.append(metar.current_wx2_symbol)
            current_wx3_symbol.append(metar.current_wx3_symbol)

        except ParseError:
            continue

    df = pd.DataFrame({'station_id': station_id,
                       'latitude': lat,
                       'longitude': lon,
                       'elevation': elev,
                       'date_time': date_time,
                       'wind_direction': wind_dir,
                       'wind_speed': wind_spd,
                       'current_wx1': current_wx1,
                       'current_wx2': current_wx2,
                       'current_wx3': current_wx3,
                       'low_cloud_type': skyc1,
                       'low_cloud_level': skylev1,
                       'medium_cloud_type': skyc2,
                       'medium_cloud_level': skylev2,
                       'high_cloud_type': skyc3,
                       'high_cloud_level': skylev3,
                       'highest_cloud_type': skyc4,
                       'highest_cloud_level': skylev4,
                       'cloud_coverage': cloudcover,
                       'air_temperature': temp,
                       'dew_point_temperature': dewp,
                       'altimeter': altim,
                       'present_weather': current_wx1_symbol,
                       'past_weather': current_wx2_symbol,
                       'past_weather2': current_wx3_symbol},
                      index=station_id)

    # Calculate sea-level pressure from function in metpy.calc
    df['air_pressure_at_sea_level'] = altimeter_to_sea_level_pressure(
        altim * units('inHg'),
        elev * units('meters'),
        temp * units('degC')).to('hPa').magnitude

    # Use get wind components and assign them to eastward and northward winds
    df['eastward_wind'], df['northward_wind'] = wind_components((df.wind_speed.values
                                                                 * units.kts),
                                                                df.wind_direction.values
                                                                * units.degree)

    # Drop duplicate values
    df = df.drop_duplicates(subset=['date_time', 'latitude', 'longitude'], keep='last')

    # Round altimeter and sea-level pressure values
    df['altimeter'] = df.altimeter.round(2)
    df['air_pressure_at_sea_level'] = df.air_pressure_at_sea_level.round(2)

    # Set the units for the dataframe
    df.units = col_units
    pandas_dataframe_to_unit_arrays(df)

    return df
