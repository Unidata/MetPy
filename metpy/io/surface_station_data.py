# Copyright (c) 2008,2015,2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Pull out station metadata for metars."""
from collections import defaultdict, namedtuple
import csv
import logging


from metpy.cbook import get_test_data

log = logging.getLogger('stations')
log.addHandler(logging.StreamHandler())  # Python 2.7 needs a handler set
log.setLevel(logging.WARNING)


Station = namedtuple('Station', ['id', 'synop_id', 'name', 'state', 'country',
                                 'longitude', 'latitude', 'altitude'])

station_map = {}


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
    """Read in the station table."""
    if input_file is None:
        input_file = get_test_data('sfstns.tbl', as_file_obj=False)
    with open(input_file, 'rt') as station_file:
        for line in station_file:
            stid = line[:9].strip()
            synop_id = int(line[9:16].strip())
            name = line[16:49].strip()
            state = line[49:52].strip()
            country = line[52:55].strip()
            lat = int(line[55:61].strip()) / 100.
            lon = int(line[61:68].strip()) / 100.
            alt = int(line[68:74].strip())
            station_map[stid] = Station(stid, synop_id=synop_id, name=name.title(),
                                        latitude=lat,
                                        longitude=lon, altitude=alt,
                                        country=country, state=state)
    return station_map


def _read_master_text_file(input_file=None):
    """Read in the master text file."""
    if input_file is None:
        input_file = get_test_data('master.txt', as_file_obj=False)
    with open(input_file, 'rt') as station_file:
        station_file.readline()
        for line in station_file:
            state = line[:3].strip()
            name = line[3:20].strip().replace('_', ' ')
            stid = line[20:25].strip()
            synop_id = line[32:38].strip()
            lat = to_dec_deg(line[39:46].strip())
            lon = to_dec_deg(line[47:55].strip())
            alt_part = line[55:60].strip()
            alt = int(alt_part if alt_part else 0.)
            if stid:
                if stid[0] in ('P', 'K'):
                    country = 'US'
                else:
                    country = state
                    state = '--'
            station_map[stid] = Station(stid, synop_id=synop_id, name=name.title(),
                                        latitude=lat,
                                        longitude=lon, altitude=alt,
                                        country=country, state=state)
    return station_map


def _read_station_text_file(input_file=None):
    """Read the station text file."""
    if input_file is None:
        input_file = get_test_data('stations.txt', as_file_obj=False)
    with open(input_file, 'rt') as station_file:
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
            station_map[stid] = Station(stid, synop_id=synop_id, name=name.title(),
                                        latitude=lat,
                                        longitude=lon, altitude=alt,
                                        country=country, state=state)
    return station_map


def _read_airports_file(input_file=None):
    """Read the airports file."""
    if input_file is None:
        input_file = get_test_data('airport-codes.csv', as_file_obj=False)
    with open(input_file, 'rt') as station_file:
        station_file.readline()  # Skip header
        csvreader = csv.reader(station_file)
        for info in csvreader:
            stid = info[0]
            new_stid = info[10]  # Use GPS code rather than ID at start of line
            if not stid.endswith(new_stid):
                stid = new_stid
            station_map[stid] = Station(stid, synop_id=99999,
                                        latitude=float(info[3]), longitude=float(info[4]),
                                        altitude=float(info[5] if info[5]
                                                       else 0) * (25.4 * 12 / 1000.),
                                        country=info[7],
                                        state=info[8].split('-')[-1], name=info[9])
    return station_map


class StationLookup:
    """Utilize all the different tables puts dictionaries together."""

    def __init__(self):
        """Initialize different files."""
        self.sources = []
        self.sources.append(('gempak', _read_station_table()))
        self.sources.append(('master', _read_master_text_file()))
        self.sources.append(('stations', _read_station_text_file()))
        self.sources.append(('airport', _read_airports_file()))
        self.sites = defaultdict(set)

    def call(self, stid):
        """Call different tables and files."""
        for name, table in self.sources:
            if stid in table:
                self.sites[name].add(stid)
                return table[stid]
        self.sites['missing'].add(stid)
        log.warning('Missing station: %s', stid)
        raise KeyError('')

    def string(self):
        """Join strings."""
        return '\n'.join('{0}: ({1}) {2}'.format(s, len(v), ' '.join(str(i) for i in v))
                         for s, v in self.sites.items())


def station_dict():
    """Assemble a master dictionary from StationLookup Function."""
    master = StationLookup().sources[0][1]
    for station in StationLookup().sources:
        master.update(**station[1])
    return master
