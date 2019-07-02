import csv
import logging
from collections import defaultdict, namedtuple

log = logging.getLogger("stations")
log.addHandler(logging.StreamHandler())  # Python 2.7 needs a handler set
log.setLevel(logging.WARNING)


Station = namedtuple('Station', ['id', 'synop_id', 'name', 'state', 'country',
                                 'longitude', 'latitude', 'altitude'])


def to_dec_deg(dms):
    if not dms:
        return 0.
    deg, minutes = dms.split()
    side = minutes[-1]
    minutes = minutes[:2]
    float_deg = int(deg) + int(minutes) / 60.
    return float_deg if side in ('N', 'E') else -float_deg


def read_station_table(filename='sfstns.tbl'):
    station_map = dict()
    with open(filename, 'rt') as station_file:
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


def read_world_table(filename='sfworld.tbl'):
    station_map = dict()
    with open(filename, 'rt') as station_file:
        for line in station_file:
            stid = line[:10].strip()
            synop_id = int(line[10:16].strip())
            name = line[16:49].strip().replace('_', ' ')
            state = line[50:52].strip()
            country = line[53:55].strip()
            lat = int(line[56:61].strip()) / 100.
            lon = int(line[62:68].strip()) / 100.
            alt = int(line[69:74].strip())
            station_map[stid] = Station(stid, synop_id, name, state, country, lon, lat, alt)
    return station_map


def read_master_text_file(filename='master.txt'):
    station_map = dict()
    with open(filename, 'rt') as station_file:
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


def read_station_text_file(filename='stations.txt'):
    station_map = dict()
    with open(filename, 'rt') as station_file:
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


def read_airports_file(filename='airport-codes.csv'):
    station_map = dict()
    with open(filename, 'rt') as station_file:
        station_file.readline()  # Skip header
        csvreader = csv.reader(station_file)
        for stid, *info in csvreader:
            new_stid = info[9]  # Use GPS code rather than ID at start of line
            if not stid.endswith(new_stid):
                stid = new_stid
            station_map[stid] = Station(stid, synop_id=99999,
                                        latitude=float(info[2]), longitude=float(info[3]),
                                        altitude=float(info[4] if info[4]
                                                       else 0) * (25.4 * 12 / 1000.),
                                        country=info[6],
                                        state=info[7].split('-')[-1], name=info[8])
    return station_map


class StationLookup:
    def __init__(self):
        self.sources = []
        self.sources.append(('gempak', read_station_table()))
        self.sources.append(('master', read_master_text_file()))
        self.sources.append(('stations', read_station_text_file()))
        self.sources.append(('airport', read_airports_file()))
        self.sites = defaultdict(set)

    def __call__(self, stid):
        for name, table in self.sources:
            if stid in table:
                self.sites[name].add(stid)
                return table[stid]
        self.sites['missing'].add(stid)
        log.warning('Missing station: %s', stid)
        raise KeyError('')

    def __str__(self):
        return '\n'.join('{0}: ({1}) {2}'.format(s, len(v), ' '.join(str(i) for i in v))
                         for s, v in self.sites.items())


def station_dict():
    master = StationLookup().sources[0][1]
    for station in StationLookup().sources:
        master = {**master, **station[1]}
    return master


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Read in station tables and dump out a few'
                                                 'in GEMPAK format.')
    parser.add_argument('stations', nargs='+', type=str, help='Stations to write out')
    args = parser.parse_args()

    lookup = StationLookup()
    stations = [lookup(stid) for stid in args.stations]
    for station in sorted(stations, key=lambda s: (s.country, s.state, s.name)):
        print('{0.id:9}{3:<7}{0.name:33}{0.state:3}{0.country:2}{1:>6.0f}{2:>7.0f}'
              '{0.altitude:>6}  0'.format(station, station.latitude * 100,
                                          station.longitude * 100,
                                          station.synop_id if station.synop_id else '999999')
              .upper())
