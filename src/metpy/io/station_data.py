# Copyright (c) 2019,2026 MetPy Developers.
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
    """Read the airports file.

    Yields tuple of station ID and `Station` for each entry.
    """
    if input_file is None:
        input_file = get_test_data('airport-codes.csv', as_file_obj=False)
    airports = pd.read_csv(
        input_file,
        usecols=['ident', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft',
                 'iso_country', 'iso_region'])
    altitude_m = units.Quantity(airports.elevation_ft.to_numpy(), 'ft').to('m').m
    region = airports.iso_region.str.split('-', n=1, expand=True)
    countries = airports.iso_country.fillna(region[0]).fillna('')
    states = region[1].fillna('')
    for ident, name, lat, lon, alt, state, country in zip(
            airports.ident, airports.name, airports.latitude_deg, airports.longitude_deg,
            altitude_m, states, countries, strict=True):
        yield ident, Station(ident, synop_id=99999, name=name, latitude=float(lat),
                             longitude=float(lon), altitude=float(alt), country=country,
                             state=state, source=input_file)


# Values stored when a source has no SYNOP/WMO identifier.
_PLACEHOLDER_SYNOP_IDS = frozenset({'', '99999', '999999'})


def _synop_id_key(synop_id):
    """Normalize a SYNOP identifier for comparison.

    Readers store this field as either ``int`` or ``str``. Whole-number floats, such as
    those pandas may produce, compare as their integer text.
    """
    if isinstance(synop_id, str):
        return synop_id.strip()
    if isinstance(synop_id, (float, np.floating)):
        if np.isnan(synop_id):
            return ''
        if synop_id.is_integer():
            return str(int(synop_id))
    return str(synop_id).strip()


@exporter.export
class StationLookup(Mapping):
    """Look up station information from multiple sources.

    This class follows the `Mapping` protocol with station ID as the key. Indexing
    returns the definition from the highest-priority source that contains that ID:

    >>> import metpy.io
    >>> conus_stations = [s for s in metpy.io.station_info if s.startswith('K')]
    >>> conus_stations[:3]
    ['K00', 'K00C', 'K00F']
    >>> metpy.io.station_info['FYOW'].synop_id
    999999

    :meth:`lookup_all` returns every definition of a station ID. :meth:`find` searches
    every source table by metadata such as SYNOP id:

    >>> metpy.io.station_info.find(synop_id=68004)[0].id
    'FYOW'

    Notes
    -----
    Indexing uses the first matching station ID from these sources, in order:
    ``sfstns.tbl``, ``master.txt``, ``stations.txt``, then ``airport-codes.csv``.
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

    def lookup_all(self, stid):
        """Return every stored definition of a station ID.

        Definitions are in source priority order. The first item, when present, is the
        station returned by indexing this lookup with ``stid``.

        Parameters
        ----------
        stid : str
            Station identifier, such as ``'FYOW'``.

        Returns
        -------
        list of `Station`
            Matching definitions. The list is empty when ``stid`` is absent from every
            source.

        """
        return [table[stid] for table in self.tables.maps if stid in table]

    def find(self, *, synop_id=None):
        """Find station definitions matching metadata.

        This searches every source table. A station ID that also exists in a
        higher-priority source is still included when the requested metadata is on the
        lower-priority definition.

        Parameters
        ----------
        synop_id : str or int
            WMO/SYNOP station identifier. Integers and strings are compared as text, so
            ``68004`` and ``'68004'`` match the same records. Leading zeros are preserved,
            so ``'06450'`` is distinct from ``6450``. Placeholder values (``''``,
            ``99999``, and ``999999``) match only that same placeholder.

        Returns
        -------
        list of `Station`
            Matching definitions in source priority order.

        Raises
        ------
        ValueError
            If no search criterion is provided.

        """
        if synop_id is None:
            raise ValueError('At least one search criterion must be provided.')

        target = _synop_id_key(synop_id)
        return [
            station
            for table in self.tables.maps
            for station in table.values()
            if _synop_id_key(station.synop_id) == target
        ]


with exporter:
    station_info = StationLookup()


def _lookup_station(identifier, lookup):
    """Return the station definition used to locate one identifier."""
    if lookup == 'id':
        try:
            return station_info[identifier]
        except KeyError:
            return None
    if _synop_id_key(identifier) in _PLACEHOLDER_SYNOP_IDS:
        return None
    matches = station_info.find(synop_id=identifier)
    return matches[0] if matches else None


@exporter.export
def add_station_lat_lon(df, stn_var=None, *, lookup='id'):
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
    lookup : {'id', 'synop_id'}, optional
        How values in ``stn_var`` are matched. ``'id'`` looks up the station identifier,
        such as ``'KDEN'``. ``'synop_id'`` looks up the WMO/SYNOP identifier, such as
        ``68004``. When several definitions share that SYNOP id, coordinates come from the
        highest-priority source. Placeholder SYNOP values (``99999``, ``999999``, and
        blanks) are left unresolved.

    Returns
    -------
    `pandas.DataFrame` that contains original Dataframe now with the latitude and longitude
    values for each location found in :data:`!station_info`.

    Raises
    ------
    ValueError
        If ``lookup`` is not ``'id'`` or ``'synop_id'``.

    """
    if lookup not in {'id', 'synop_id'}:
        raise ValueError(f"lookup must be 'id' or 'synop_id', got {lookup!r}.")

    def key_finder(df):
        names_to_try = ('station', 'stid', 'station_id')
        for id_name in names_to_try:
            if id_name in df:
                return id_name
        raise KeyError('Second argument not provided to add_station_lat_lon, but none of '
                       f'{names_to_try} were found.')

    df.insert(len(df.columns), 'latitude', np.nan)
    df.insert(len(df.columns), 'longitude', np.nan)

    if stn_var is None:
        stn_var = key_finder(df)
    for stn in df[stn_var].unique():
        info = _lookup_station(stn, lookup)
        if info is not None:
            rows = df[stn_var] == stn
            df.loc[rows, 'latitude'] = info.latitude
            df.loc[rows, 'longitude'] = info.longitude
    return df
