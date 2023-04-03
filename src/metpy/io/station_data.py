# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Pull out station metadata."""
from collections import ChainMap, namedtuple
from collections.abc import Mapping
from functools import cached_property

import numpy as np
import pandas as pd

from ..cbook import get_test_data
from ..package_tools import Exporter
from ..units import units

exporter = Exporter(globals())
Station = namedtuple('Station', ['id', 'synop_id', 'name', 'state', 'country',
                                 'longitude', 'latitude', 'altitude', 'source'])


def to_dec_deg(dms):
    """Convert to decimal degrees."""
    if not dms:
        return 0.
    deg, minutes = dms.split()
    side = minutes[-1]
    minutes = minutes[:2]
    float_deg = int(deg) + int(minutes) / 60.
    return float_deg if side in ('N', 'E') else -float_deg


def _read_station_table(input_file=None):
    """Read in the GEMPAK station table.

    Yields tuple of station ID and `Station` for each entry.
    """
    if input_file is None:
        input_file = get_test_data('sfstns.tbl', as_file_obj=False)
    with open(input_file) as station_file:
        for line in station_file:
            stid = line[:9].strip()
            synop_id = int(line[9:16].strip())
            name = line[16:49].strip()
            state = line[49:52].strip()
            country = line[52:55].strip()
            lat = int(line[55:61].strip()) / 100.
            lon = int(line[61:68].strip()) / 100.
            alt = int(line[68:74].strip())
            yield stid, Station(stid, synop_id=synop_id, name=name.title(), latitude=lat,
                                longitude=lon, altitude=alt, country=country, state=state,
                                source=input_file)


def _read_master_text_file(input_file=None):
    """Read in the master text file.

    Yields tuple of station ID and `Station` for each entry.
    """
    if input_file is None:
        input_file = get_test_data('master.txt', as_file_obj=False)
    with open(input_file) as station_file:
        station_file.readline()
        for line in station_file:
            state = line[:3].strip()
            name = line[3:20].strip().replace('_', ' ')
            stid = line[20:25].strip()
            synop_id = line[32:38].strip()
            lat = to_dec_deg(line[39:46].strip())
            lon = to_dec_deg(line[47:55].strip())
            alt_part = line[55:60].strip()
            alt = int(alt_part or 0.)
            if stid:
                if stid[0] in ('P', 'K'):
                    country = 'US'
                else:
                    country = state
                    state = '--'
            yield stid, Station(stid, synop_id=synop_id, name=name.title(), latitude=lat,
                                longitude=lon, altitude=alt, country=country, state=state,
                                source=input_file)


def _read_station_text_file(input_file=None):
    """Read the station text file.

    Yields tuple of station ID and `Station` for each entry.
    """
    if input_file is None:
        input_file = get_test_data('stations.txt', as_file_obj=False)
    with open(input_file) as station_file:
        for line in station_file:
            if line[0] == '!':
                continue
            lat = line[39:45].strip()
            if not lat or lat == 'LAT':
                continue
            lat = to_dec_deg(lat)
            state = line[:3].strip()
            name = line[3:20].strip().replace('_', ' ')
            stid = line[20:25].strip()
            synop_id = line[32:38].strip()
            lon = to_dec_deg(line[47:55].strip())
            alt = int(line[55:60].strip())
            country = line[81:83].strip()
            yield stid, Station(stid, synop_id=synop_id, name=name.title(), latitude=lat,
                                longitude=lon, altitude=alt, country=country, state=state,
                                source=input_file)


def _read_airports_file(input_file=None):
    """Read the airports file."""
    if input_file is None:
        input_file = get_test_data('airport-codes.csv', as_file_obj=False)
    df = pd.read_csv(input_file)
    return pd.DataFrame({'id': df.ident.values, 'synop_id': 99999,
                         'latitude': df.latitude_deg.values,
                         'longitude': df.longitude_deg.values,
                         'altitude': units.Quantity(df.elevation_ft.values, 'ft').to('m').m,
                         'country': df.iso_region.str.split('-', n=1, expand=True)[1].values,
                         'source': input_file
                         }).to_dict()


@exporter.export
class StationLookup(Mapping):
    """Look up station information from multiple sources.

    This class follows the `Mapping` protocol with station ID as the key. This makes it
    possible to e.g. iterate over all locations and get all of a certain criteria:

    >>> import metpy.io
    >>> conus_stations = [s for s in metpy.io.station_info if s.startswith('K')]
    >>> conus_stations[:3]
    ['KEET', 'K8A0', 'KALX']
    """

    @cached_property
    def tables(self):
        """Return an iterable mapping combining all the tables."""
        return ChainMap(dict(_read_station_table()),
                        dict(_read_master_text_file()),
                        dict(_read_station_text_file()),
                        dict(_read_airports_file()))

    def __len__(self):
        """Get the number of stations."""
        return len(self.tables)

    def __iter__(self):
        """Allow iteration over the stations."""
        return iter(self.tables)

    def __getitem__(self, stid):
        """Lookup station information from the ID."""
        try:
            return self.tables[stid]
        except KeyError:
            raise KeyError(f'No station information for {stid}') from None


with exporter:
    station_info = StationLookup()


@exporter.export
def add_station_lat_lon(df, stn_var=None):
    """Lookup station information to add the station latitude and longitude to the DataFrame.

    This function will add two columns to the DataFrame ('latitude' and 'longitude') after
    looking up all unique station identifiers available in the DataFrame.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The DataFrame that contains the station observations
    stn_var : str, optional
        The string of the variable name that represents the station in the DataFrame. If not
        provided, 'station', 'stid', and 'station_id' are tried in that order.

    Returns
    -------
    `pandas.DataFrame` that contains original Dataframe now with the latitude and longitude
    values for each location found in :data:`!station_info`.
    """

    def key_finder(df):
        names_to_try = ('station', 'stid', 'station_id')
        for id_name in names_to_try:
            if id_name in df:
                return id_name
        raise KeyError('Second argument not provided to add_station_lat_lon, but none of '
                       f'{names_to_try} were found.')

    df['latitude'] = None
    df['longitude'] = None
    if stn_var is None:
        stn_var = key_finder(df)
    for stn in df[stn_var].unique():
        try:
            info = station_info[stn]
            df.loc[df[stn_var] == stn, 'latitude'] = info.latitude
            df.loc[df[stn_var] == stn, 'longitude'] = info.longitude
        except KeyError:
            df.loc[df[stn_var] == stn, 'latitude'] = np.nan
            df.loc[df[stn_var] == stn, 'longitude'] = np.nan
    return df
