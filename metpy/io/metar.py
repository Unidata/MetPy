# Import the neccessary libraries
import pandas as pd
import numpy as np
from metpy.io import surface_station_data, metar_parse
from metpy.plots.wx_symbols import wx_code_map
from metpy.units import units, pandas_dataframe_to_unit_arrays
from metpy.calc import altimeter_to_sea_level_pressure
import warnings
from collections import namedtuple
from datetime import datetime
from metpy.io.metar_parse import ParseError

warnings.filterwarnings('ignore', 'Pandas doesn\'t allow columns to be created', UserWarning)
Metar = namedtuple('metar', ['station_id', 'latitude', 'longitude', 'elevation',
'date_time', 'wind_direction', 'wind_speed', 'current_wx1',
'current_wx2', 'current_wx3', 'skyc1', 'skylev1', 'skyc2', 'skylev2', 'skyc3',
'skylev3', 'skyc4', 'skylev4', 'cloudcover', 'temperature', 'dewpoint', 'altimeter',
'current_wx1_symbol', 'current_wx2_symbol', 'current_wx3_symbol'])

def parse_metar_to_pandas(metar_text, year = datetime.now().year, month = datetime.now().month):
    """Takes in a metar file, in a text form, and creates a pandas
    dataframe that can be easily subset

    Input:
    metar_text = string with the METAR data
    create_df = True or False
        True creates a Pandas dataframe as the Output
        False creates a list of lists containing the values in the following order:

        [station_id, latitude, longitude, elevation, date_time, day, time_utc,
        wind_direction, wind_speed, wxsymbol1, wxsymbol2, skycover1, skylevel1,
        skycover2, skylevel2, skycover3, skylevel3, skycover4, skylevel4,
        cloudcover, temperature, dewpoint, altimeter_value, sea_level_pressure]

    Output:
    Pandas Dataframe that can be subset easily
    """

    #Create a dictionary with all the station metadata
    station_metadata = surface_station_data.station_dict()

    # Decode the data using the parser (built using Canopy)
    tree = metar_parse.parse(metar_text)

    #Station ID, Latitude, Longitude, and Elevation
    if tree.siteid.text == '':
        station_id = [np.nan]
    else:
        station_id = [tree.siteid.text.strip()]
        #Extract the latitude and longitude values from 'master' dictionary
        try:
            lat = station_metadata[tree.siteid.text.strip()].latitude
            lon = station_metadata[tree.siteid.text.strip()].longitude
            elev = station_metadata[tree.siteid.text.strip()].altitude
        except:
            lat = np.nan
            lon = np.nan
            elev = np.nan

    # Set the datetime, day, and time_utc
    if tree.datetime.text == '':
        datetime = np.nan
        day = np.nan
        time_utc = np.nan
    else:
        day_time_utc = tree.datetime.text[:-1].strip()
        day = int(day_time_utc[0:2])
        hour = int(day_time_utc[2:4])
        minute = int(day_time_utc[4:7])
        date_time = datetime(year, month, day, hour, minute)

    # Set the wind variables
    if tree.wind.text == '':
        wind_dir = np.nan
        wind_spd = np.nan
    elif (tree.wind.text == '/////KT') or (tree.wind.text ==' /////KT') or (tree.wind.text == 'KT'):
        wind_dir = np.nan
        wind_spd = np.nan
    else:
        if (tree.wind.wind_dir.text == 'VRB') or (tree.wind.wind_dir.text == 'VAR'):
            wind_dir = np.nan
            wind_spd = float(tree.wind.wind_spd.text)
        else:
            wind_dir = int(tree.wind.wind_dir.text)
            wind_spd = int(tree.wind.wind_spd.text)

    # Set the weather symbols
    if tree.curwx.text == '':
        current_wx1 = np.nan
        current_wx2 = np.nan
        current_wx3 = np.nan
        current_wx1_symbol = np.nan
        current_wx2_symbol = np.nan
        current_wx3_symbol = np.nan
    else:
        wx = [np.nan, np.nan, np.nan]
        wx[0:len((tree.curwx.text.strip()).split())] = tree.curwx.text.strip().split()
        current_wx1 = wx[0]
        current_wx2 = wx[1]
        current_wx3 = wx[2]
        try:
            current_wx1_symbol = int(wx_code_map[wx[0]])
        except:
            current_wx1_symbol = np.nan
        try:
            current_wx2_symbol = int(wx_code_map[wx[1]])
        except:
            current_wx2_symbol = np.nan
        try:
            current_wx3_symbol = int(wx_code_map[wx[3]])
        except:
            current_wx3_symbol = np.nan

    # Set the sky conditions
    if tree.skyc.text == '':
        skyc1 = np.nan
        skylev1 = np.nan
        skyc2 = np.nan
        skylev2 = np.nan
        skyc3 = np.nan
        skylev3 = np.nan
        skyc4 = np.nan
        skylev4 = np.nan

    elif tree.skyc.text[1:3] == 'VV':
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
            skylev1 = float(skyc[0][3:])*100
        except:
            skyc1 = np.nan
            skylev1 = np.nan
        try:
            skyc2 = skyc[1][0:3]
            skylev2 = float(skyc[1][3:])*100
        except:
            skyc2 = np.nan
            skylev2 = np.nan
        try:
            skyc3 = skyc[2][0:3]
            skylev3 = float(skyc[2][3:])*100
        except:
            skyc3 = np.nan
            skylev3 = np.nan
        try:
            skyc4 = skyc[3][0:3]
            skylev4 = float(skyc[3][3:])*100
        except:
            skyc4 = np.nan
            skylev4 = np.nan


    if ('OVC' or 'VV') in tree.skyc.text:
        cloudcover = 8
    elif 'BKN' in tree.skyc.text:
        cloudcover = 6
    elif 'SCT' in tree.skyc.text:
        cloudcover = 4
    elif 'FEW' in tree.skyc.text:
        cloudcover = 2
    elif ('SKC' in tree.skyc.text) or ('NCD' in tree.skyc.text) \
    or ('NSC' in tree.skyc.text) or ('CLR') in tree.skyc.text:
        cloudcover = 2
    else:
        cloudcover = np.nan

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
        except:
            temp = np.nan
        try:
            if 'M' in tree.temp_dewp.dewp.text:
                dewp = (-1 * float(tree.temp_dewp.dewp.text[-2:]))
            else:
                dewp = float(tree.temp_dewp.dewp.text[-2:])
        except:
            dewp = np.nan

    # Set the altimeter value and sea level pressure
    if tree.altim.text == '':
        altim = np.nan
    else:
        if (float(tree.altim.text.strip()[1:5])) > 1100:
            altim = (float(tree.altim.text.strip()[1:5]) / 100)
        else:
            altim = ((int(tree.altim.text.strip()[1:5])*units.hPa).to('inHg').magnitude)

    col_units = {
    'station_id': None,
    'lat': 'degrees',
    'lon': 'degrees',
    'elev': 'meters',
    'date_time': None,
    'day': None,
    'time_utc': None,
    'wind_dir': 'degrees',
    'wind_spd': 'kts',
    'current_wx1': None,
    'current_wx2': None,
    'current_wx3': None,
    'skyc1': None,
    'skylev1': 'feet',
    'skyc2': None,
    'skylev2': 'feet',
    'skyc3': None,
    'skylev3': 'feet',
    'skyc4': None,
    'skylev4:': None,
    'cloudcover': None,
    'temp': 'degC',
    'dewp': 'degC',
    'altim': 'inHg',
    'current_wx1_symbol': None,
    'current_wx2_symbol': None,
    'current_wx3_symbol': None,
    'slp': 'hectopascals'}

    df = pd.DataFrame({'station_id':station_id, 'latitude':lat,
    'longitude':lon, 'elevation':elev, 'date_time':date_time,
    'wind_direction':wind_dir, 'wind_speed':wind_spd,'current_wx1':current_wx1,
    'current_wx2':current_wx2, 'current_wx3':current_wx3, 'skyc1':skyc1,
    'skylev1':skylev1, 'skyc2':skyc2, 'skylev2':skylev2, 'skyc3':skyc3,
    'skylev3': skylev3, 'skyc4':skyc4, 'skylev4':skylev4,
    'cloudcover':cloudcover, 'temperature':temp, 'dewpoint':dewp,
    'altimeter':altim, 'current_wx1_symbol':current_wx2_symbol,
    'current_wx2_symbol':current_wx2_symbol, 'current_wx3_symbol':current_wx3_symbol},
    index = station_id)

    try:
        df['sea_level_pressure'] = float(format(altimeter_to_slp(
        altim * units('inHg'),
        elev * units('meters'),
        temp * units('degC')).magnitude, '.1f'))
    except:
        df['sea_level_pressure'] = [np.nan]

    df['altimeter'] = df.altimeter.round(2)
    df['sea_level_pressure'] = df.sea_level_pressure.round(2)

    df.index = df.station_id

    #Set the units for the dataframe
    df.units = col_units
    pandas_dataframe_to_unit_arrays(df)

    return df

def parse_metar_to_named_tuple(metar_text, station_dict, year = datetime.now().year, month = datetime.now().month):
    """Takes in a metar file, in a text form, and creates a pandas
    dataframe that can be easily subset

    Input:
    metar_text = string with the METAR data
    create_df = True or False
        True creates a Pandas dataframe as the Output
        False creates a list of lists containing the values in the following order:

        [station_id, latitude, longitude, elevation, date_time, day, time_utc,
        wind_direction, wind_speed, wxsymbol1, wxsymbol2, skycover1, skylevel1,
        skycover2, skylevel2, skycover3, skylevel3, skycover4, skylevel4,
        cloudcover, temperature, dewpoint, altimeter_value, sea_level_pressure]

    Output:
    Pandas Dataframe that can be subset easily
    """
    from datetime import datetime
    station_metadata = station_dict

    # Decode the data using the parser (built using Canopy)
    tree = metar_parse.parse(metar_text)

    #Station ID, Latitude, Longitude, and Elevation
    if tree.siteid.text == '':
        station_id = np.nan
    else:
        station_id = tree.siteid.text.strip()
        #Extract the latitude and longitude values from 'master' dictionary
        try:
            lat = station_metadata[station_id].latitude
            lon = station_metadata[station_id].longitude
            elev = station_metadata[station_id].altitude
        except:
            lat = np.nan
            lon = np.nan
            elev = np.nan

    # Set the datetime, day, and time_utc
    if tree.datetime.text == '':
        datetime = np.nan
        day = np.nan
        time_utc = np.nan
    else:
        day_time_utc = tree.datetime.text[:-1].strip()
        day = int(day_time_utc[0:2])
        hour = int(day_time_utc[2:4])
        minute = int(day_time_utc[4:7])
        date_time = datetime(year, month, day, hour, minute)

    # Set the wind variables
    if ('/' in tree.wind.text) or (tree.wind.text == 'KT'):
        wind_dir = np.nan
        wind_spd = np.nan
    else:
        try:
            if (tree.wind.wind_dir.text == 'VRB') or (tree.wind.wind_dir.text == 'VAR') \
            or (tree.wind.wind_dir.text == '///'):
                wind_dir = np.nan
                wind_spd = float(tree.wind.wind_spd.text)
            else:
                wind_dir = int(tree.wind.wind_dir.text)
                wind_spd = int(tree.wind.wind_spd.text)
        except:
            wind_dir = np.nan
            wind_spd = np.nan
    # Set the weather symbols
    if tree.curwx.text == '':
        current_wx1 = np.nan
        current_wx2 = np.nan
        current_wx3 = np.nan
        current_wx1_symbol = np.nan
        current_wx2_symbol = np.nan
        current_wx3_symbol = np.nan
    else:
        wx = [np.nan, np.nan, np.nan]
        wx[0:len((tree.curwx.text.strip()).split())] = tree.curwx.text.strip().split()
        current_wx1 = wx[0]
        current_wx2 = wx[1]
        current_wx3 = wx[2]
        try:
            current_wx1_symbol = int(wx_code_map[wx[0]])
        except:
            current_wx1_symbol = np.nan
        try:
            current_wx2_symbol = int(wx_code_map[wx[1]])
        except:
            current_wx2_symbol = np.nan
        try:
            current_wx3_symbol = int(wx_code_map[wx[2]])
        except:
            current_wx3_symbol = np.nan

    # Set the sky conditions
    if tree.skyc.text == '':
        skyc1 = np.nan
        skylev1 = np.nan
        skyc2 = np.nan
        skylev2 = np.nan
        skyc3 = np.nan
        skylev3 = np.nan
        skyc4 = np.nan
        skylev4 = np.nan

    elif tree.skyc.text[1:3] == 'VV':
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
            skylev1 = float(skyc[0][3:])*100
        except:
            skyc1 = np.nan
            skylev1 = np.nan
        try:
            skyc2 = skyc[1][0:3]
            skylev2 = float(skyc[1][3:])*100
        except:
            skyc2 = np.nan
            skylev2 = np.nan
        try:
            skyc3 = skyc[2][0:3]
            skylev3 = float(skyc[2][3:])*100
        except:
            skyc3 = np.nan
            skylev3 = np.nan
        try:
            skyc4 = skyc[3][0:3]
            skylev4 = float(skyc[3][3:])*100
        except:
            skyc4 = np.nan
            skylev4 = np.nan


    if ('OVC' or 'VV') in tree.skyc.text:
        cloudcover = 8
    elif 'BKN' in tree.skyc.text:
        cloudcover = 6
    elif 'SCT' in tree.skyc.text:
        cloudcover = 4
    elif 'FEW' in tree.skyc.text:
        cloudcover = 2
    elif ('SKC' in tree.skyc.text) or ('NCD' in tree.skyc.text) \
    or ('NSC' in tree.skyc.text) or ('CLR') in tree.skyc.text:
        cloudcover = 2
    else:
        cloudcover = np.nan

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
        except:
            temp = np.nan
        try:
            if 'M' in tree.temp_dewp.dewp.text:
                dewp = (-1 * float(tree.temp_dewp.dewp.text[-2:]))
            else:
                dewp = float(tree.temp_dewp.dewp.text[-2:])
        except:
            dewp = np.nan

    # Set the altimeter value and sea level pressure
    if tree.altim.text == '':
        altim = np.nan
    else:
        if (float(tree.altim.text.strip()[1:5])) > 1100:
            altim = (float(tree.altim.text.strip()[1:5]) / 100)
        else:
            altim = ((int(tree.altim.text.strip()[1:5])*units.hPa).to('inHg').magnitude)

    return Metar(station_id, lat, lon, elev, date_time, wind_dir, wind_spd,
    current_wx1, current_wx2, current_wx3, skyc1, skylev1, skyc2, skylev2, skyc3, skylev3,
    skyc4, skylev4, cloudcover, temp, dewp, altim, current_wx1_symbol, current_wx2_symbol,
    current_wx3_symbol)


def text_file_parse(file, year = datetime.now().year, month = datetime.now().month):
    """ Takes a text file taken from the NOAA PORT system containing
    METAR data and creates a dataframe with all the observations

    parameters
    ----------
    file: string
          The path to the file containing the data. It should be extracted
          from NOAA PORT and NOT be in binary format

    return
    ---------
    df : pandas dataframe wtih the station id as the index

    """


    #Function to merge METARs
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

    #Open the file
    myfile = open(file)

    #Clean up the file and take out the next line (\n)
    value = myfile.read().rstrip()
    list_values = value.split(sep = '\n')
    list_values = list(filter(None, list_values))

    #Call the merge function and assign the result to the list of metars
    list_values = list(merge(list_values))

    #Remove the short lines that do not contain METAR observations or contain
    #METAR observations that lack a robust amount of data
    metars = []
    for metar in list_values:
        if len(metar) > 25:
            metars.append(metar)
    else:
        None

    #Create a dictionary with all the station name, locations, and elevations
    master = surface_station_data.station_dict()

    #Setup lists to append the data to
    station_id = []
    lat = []
    lon = []
    elev = []
    date_time = []
    wind_dir = []
    wind_spd = []
    current_wx1= []
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

    for metar in metars:
        try:
            metar = parse_metar_to_named_tuple(metar, master, year = year, month = month)
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
            None

    col_units = {
    'station_id': None,
    'latitude': 'degrees',
    'longitude': 'degrees',
    'elevation': 'meters',
    'date_time': None,
    'wind_direction': 'degrees',
    'wind_speed': 'kts',
    'current_wx1': None,
    'current_wx2': None,
    'current_wx3': None,
    'skyc1': None,
    'skylev1': 'feet',
    'skyc2': None,
    'skylev2': 'feet',
    'skyc3': None,
    'skylev3': 'feet',
    'skyc4': None,
    'skylev4:': None,
    'cloudcover': None,
    'temperature': 'degC',
    'dewpoint': 'degC',
    'altimeter': 'inHg',
    'sea_level_pressure': 'hPa',
    'current_wx1_symbol': None,
    'current_wx2_symbol': None,
    'current_wx3_symbol': None,}

    df = pd.DataFrame({'station_id':station_id, 'latitude':lat, 'longitude':lon,
    'elevation':elev, 'date_time':date_time, 'wind_direction':wind_dir,
    'wind_speed':wind_spd, 'current_wx1':current_wx1, 'current_wx2':current_wx2,
    'current_wx3':current_wx3, 'skyc1':skyc1, 'skylev1':skylev1, 'skyc2':skyc2,
    'skylev2':skylev2, 'skyc3':skyc3, 'skylev3': skylev3, 'skyc4':skyc4,
    'skylev4':skylev4, 'cloudcover':cloudcover, 'temperature':temp, 'dewpoint':dewp,
    'altimeter':altim, 'current_wx1_symbol':current_wx2_symbol,
    'current_wx2_symbol':current_wx2_symbol, 'current_wx3_symbol':current_wx3_symbol})

    try:
        df['sea_level_pressure'] = altimeter_to_sea_level_pressure(
        altim * units('inHg'),
        elev * units('meters'),
        temp * units('degC')).magnitude
    except:
        df['sea_level_pressure'] = [np.nan]
    #Drop duplicates
    df = df.drop_duplicates(subset = ['date_time','latitude', 'longitude'], keep = 'last')

    df['altimeter'] = df.altimeter.round(2)
    df['sea_level_pressure'] = df.sea_level_pressure.round(2)

    df.index = df.station_id

    #Set the units for the dataframe
    df.units = col_units
    pandas_dataframe_to_unit_arrays(df)

    return df
