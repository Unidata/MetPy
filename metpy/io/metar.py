from __future__ import division
import logging
import re
from collections import namedtuple
from datetime import datetime

from ..package_tools import Exporter
from .text import ParseError, RegexParser, WMOTextProduct, parse_wmo_time
from ..units import units

exporter = Exporter(globals())

log = logging.getLogger('metpy.io.metar')
log.addHandler(logging.StreamHandler())  # Python 2.7 needs a handler set
log.setLevel(logging.WARNING)


def raise_parse_error(msg, *args):
    'Format message and raise as an error'
    raise ParseError(msg % args)


class MetarProduct(WMOTextProduct):
    def _parse(self, it):
        # Handle NWS style where it's just specified once at the top rather than per METAR
        if it.peek() in ('METAR', 'SPECI'):
            def_kind = next(it)
        else:
            def_kind = 'METAR'

        it.linesep = '=[\n]{0,2}'
        self.reports = []

        parser = MetarParser(default_kind=def_kind, ref_time=self.datetime)
        for l in it:
            # Skip SAOs
            if l[3:7] != ' SA ':
                report = parser.parse(l)
                # Only add the report if it's not empty
                if report:
                    self.reports.append(report)

    def __str__(self):
        return (super(MetarProduct, self).__str__() + '\n\tReports:' +
                '\n\t\t'.join(map(str, self.reports)))


def as_value(val, units):
    'Parse a value from a METAR report, attaching units'
    try:
        if val is None:
            return None
        elif val[0] in 'MP':
            log.warning('Got unhandled M/P value: %s', val)
            val = val[1:]
        elif val == '/' * len(val):
            val = 'NaN'
        return float(val) * units
    except (AttributeError, TypeError, ValueError) as e:
        log.debug('Failed converting to value (%s): %s', e, val)
        return val


# Helper for parsing. Generates a function to grab a given group from the matches, optionally
# applying a converter
def grab_group(group, conv=None):
    if conv:
        def process(matches, *args):
            return conv(matches[group])
    else:
        def process(matches, *args):
            return matches[group]
    return process


class MetarParser(object):
    'Class that parses a single METAR report'
    def __init__(self, default_kind='METAR', ref_time=None):
        # Reports should start with METAR/SPECI, but of course NWS doesn't actually
        # do this...
        self.default_kind = default_kind

        # Can specify the appropriate date for year/month. Defaults to using current
        self.ref_time = ref_time if ref_time else datetime.utcnow()

        # Main expected groups in the report
        self.main_groups = [('kind', kind(default_kind)), ('stid', stid),
                            ('datetime', dt(ref_time)), ('null', null), ('auto', auto),
                            ('corrected', corrected),
                            ('wind', wind), ('visibility', vis), ('runway_range', rvr),
                            ('present_wx', wx), ('sky_coverage', sky_cover),
                            ('temperature', basic_temp), ('altimeter', altimeter),
                            ('runway_state', runway_state)]

        # Complete set of possible groups in the remarks section
        self.remarks = [('volcano', volcano), ('automated', automated_type),
                        ('peak_wind', peak_wind), ('wind_shift', wind_shift),
                        ('sfc_vis', sfc_vis),
                        ('variable_vis', var_vis), ('sector_vis', sector_vis),
                        ('lightning', lightning),
                        ('precip_times', precip_times), ('thunderstorm', thunderstorm),
                        ('virga', virga),
                        ('variable_ceiling', var_ceiling), ('variable_sky_cover', var_sky),
                        ('significant_clouds', sig_cloud), ('mountains', mountains),
                        ('pressure_change', pressure_change),
                        ('sea_level_pressure', slp), ('no_speci', nospeci),
                        ('report_sequence', report_sequence),
                        ('hourly_precip', hourly_precip), ('period_precip', period_precip),
                        ('snow_6hr', snow_6hr), ('snow_depth', snow_depth),
                        ('snow_liquid_equivalent', snow_liquid_equivalent),
                        ('hourly_ice', hourly_ice), ('ice_3hr', ice_3hr), ('ice_6hr', ice_6hr),
                        ('daily_precip', daily_precip), ('cloud_types', cloud_types),
                        ('hourly_temperature', hourly_temp), ('max_temp_6hr', max_temp_6hr),
                        ('min_temp_6hr', min_temp_6hr),
                        ('daily_temperature', daily_temp_range),
                        ('pressure_tendency_3hr', press_tend),
                        ('non-operational sensors', non_op_sensors),
                        ('pilot', pilot_remark), ('needs_maintenance', maint), ('null', null)]

        self.clean_whitespace = re.compile('\s+')

    def parse(self, report):
        'Parses the report and returns a dictionary of parsed results'
        report = self.clean_whitespace.sub(' ', report)
        ob = dict(report=report, null=False)

        # Split into main and remark sections so we can treat slightly differently
        if 'RMK' in report:
            main, remark = report.split('RMK', 1)
        else:
            main = report
            remark = ''

        # Need to split out the trend forecast, otherwise will break parsing
        split = trend_forecast_regex.split(main, 1)
        if len(split) > 1:
            main, match, trend = trend_forecast_regex.split(main, 1)
            trend = trend.strip()
            if trend:
                trend_store = dict()
                trend = self._look_for_groups(trend, self.main_groups, trend_store)
                trend_store['unparsed'] = trend
                ob['trend_forecast'] = (match, trend_store)
            else:
                ob['trend_forecast'] = match

        # Start with the main groups. Get back what remains of the report
        main = self._look_for_groups(main, self.main_groups, ob)

        # Try a second pass for some badly ordered groups; skip first few groups
        if main:
            main = self._look_for_groups(main, self.main_groups[6:], ob)

        # If we have anything left now, it's un-parsed data and we should flag it. We check
        # to make sure it's actually useful leftovers
        if main and set(main) - set(' /'):
            ob['unparsed'] = main

        # If we have a remarks section, try to parse it
        if remark:
            # The groups in the remarks rely upon information from earlier in the report,
            # like the current time or units
            speed_units = ob['wind']['speed'].units if 'wind' in ob else units.knot
            context = dict(datetime=ob.get('datetime', self.ref_time),
                           speed_units=units.Quantity(1.0, speed_units))

            remark = self._look_for_groups_reduce(remark, self.remarks, ob, context)
            if remark:
                ob['remarks'] = remark

        # Handle parsing garbage by checking for either datetime or null report
        if ob['null'] or 'datetime' in ob:
            return ob
        else:
            return dict()

    def _look_for_groups(self, string, groups, store, *context):
        # Walk through the list of (name, group) and try parsing the report with the group.
        # This will return the string that was parsed, so that we can keep track of where
        # we are in the string. We use a while loop so that we can repeat a group if
        # appropriate.
        string = string.strip()
        cursor = 0
        leftover = []
        groups = iter(groups)
        name, group = next(groups)
        while True:
            # Skip spaces and newlines, won't exceed end because no trailing whitespace
            while string[cursor] == ' ':
                cursor += 1

            # Try to parse using the group.
            rng, data = group.parse(string, cursor, *context)

            # If we got back a range, that means the group succeeded in parsing
            if rng:
                start, end = rng
                log.debug('%s parsed %s', name, string[start:end])

                # If the match didn't start at the cursor, that means we skipped some
                # data and should flag as necessary
                if start > cursor:
                    leftover.append(string[cursor:start].strip())

                # Update the cursor in the string to where the group finished parsing
                cursor = end

            # If we got back some data, we should store. Possible to get back a default
            # value even if no parsing done.
            if data is not None:
                log.debug('%s returned %s', name, data)

                # If it's a repeated group, we store in a list regardless
                if group.repeat and group.keepall:
                    store.setdefault(name, []).append(data)
                else:
                    store[name] = data

            # If we've finished the string, get out
            if cursor >= len(string):
                break

            # If we shouldn't repeat the group, get the next one
            if not group.repeat or data is None:
                try:
                    name, group = next(groups)
                except StopIteration:
                    break

        # Return what remains of the string (removing whitespace)
        leftover.append(string[cursor:].strip())
        return ' '.join(leftover)

    def _look_for_groups_reduce(self, string, groups, store, *context):
        # Walk through the list of (name, group) and try parsing the report with the group.
        # This will return the string that was parsed, so that we can keep track of where
        # we are in the string. We use a while loop so that we can repeat a group if
        # appropriate.
        string = string.strip()
        groups = iter(groups)
        name, group = next(groups)
        while True:
            # Try to parse using the group.
            rng, data = group.parse(string, 0, *context)

            # If we got back a range, that means the group succeeded in parsing
            if rng:
                start, end = rng
                log.debug('%s parsed %s', name, string[start:end])

                string = string[:start].strip() + ' ' + string[end:].strip()

            # If we got back some data, we should store. Possible to get back a default
            # value even if no parsing done.
            if data is not None:
                log.debug('%s returned %s', name, data)

                # If it's a repeated group, we store in a list regardless
                if group.repeat and group.keepall:
                    store.setdefault(name, []).append(data)
                else:
                    store[name] = data

            # If we shouldn't repeat the group, get the next one
            if not group.repeat or data is None:
                try:
                    name, group = next(groups)
                except StopIteration:
                    break

        # Return what remains of the string (removing whitespace)
        return string.strip()

#
# Parsers for METAR groups -- main report
#


# Parse out METAR/SPECI
def kind(default):
    return RegexParser(r'\b(?P<kind>METAR|SPECI)\b', grab_group('kind'), default=default)

# Grab STID (CCCC)
stid = RegexParser(r'\b(?P<stid>[0-9A-Z]{4})\b', grab_group('stid'))


# Process the datetime in METAR to a full datetime (YYGGggZ)
def dt(ref_time):
    return RegexParser(r'\b(?P<datetime>[0-3]\d[0-5]\d[0-5]\dZ)',
                       lambda matches: parse_wmo_time(matches['datetime'], ref_time))

# Look for AUTO
auto = RegexParser(r'\b(?P<auto>AUTO)', grab_group('auto', bool), default=False)

# Look for COR
corrected = RegexParser(r'\b(?P<cor>COR)\b', grab_group('cor', bool), default=False)

# Look for NIL reports
null = RegexParser(r'\b(?P<null>NIL)', grab_group('null', bool), default=False)


# Process the full wind group (dddfffGfffKT dddVddd)
def process_wind(matches):
    speed_unit = units('m/s') if matches.pop('units') == 'MPS' else units.knot
    matches['direction'] = as_value(matches['direction'], units.deg)
    matches['speed'] = as_value(matches['speed'], speed_unit)
    matches['gust'] = as_value(matches['gust'], speed_unit)
    matches['dir1'] = as_value(matches['dir1'], units.deg)
    matches['dir2'] = as_value(matches['dir2'], units.deg)
    return matches

wind = RegexParser(r'''(?P<direction>VRB|///|[0-3]\d{2})
                       (?P<speed>P?[\d]{2,3}|//)
                       (G(?P<gust>P?\d{2,3}))?
                       ((?P<units>KT|MPS)|\b|\ )
                       (\ (?P<dir1>\d{3})V(?P<dir2>\d{3}))?''', process_wind)


# The visibilty group (VVVVV)
frac_conv = {'1/4': 1 / 4, '1/2': 1 / 2, '3/4': 3 / 4,
             '1/8': 1 / 8, '3/8': 3 / 8, '5/8': 5 / 8, '7/8': 7 / 8,
             '1/16': 1 / 16, '3/16': 3 / 16, '5/16': 5 / 16, '7/16': 7 / 16,
             '9/16': 9 / 16, '11/16': 11 / 16, '13/16': 13 / 16, '15/16': 15 / 16}


def vis_to_float(dist, units):
    'Convert visibility, including fraction, to a value with units'
    if dist[0] == 'M':
        dist = dist[1:]
    dist = dist.strip()

    if '/' in dist:
        # Handle the case where the entire group is all '////'
        if dist[0] == '/' and all(c == '/' for c in dist):
            return float('nan') * units
        parts = dist.split(maxsplit=1)
        if len(parts) > 1:
            return as_value(parts[0], units) + frac_conv.get(parts[1], float('nan')) * units
        else:
            return frac_conv.get(dist, float('nan')) * units
    else:
        return as_value(dist, units)


def process_vis(matches):
    if matches['cavok']:
        return 'CAVOK'
    elif matches['vismiles']:
        return vis_to_float(matches['vismiles'], units.mile)
    elif matches['vismetric']:
        return as_value(matches['vismetric'], units.meter)

vis = RegexParser(r'''(?P<cavok>CAVOK)|
                      ((?P<vismiles>M?[0-9 /]{1,5})SM\b)|
                      (?P<vismetric>\b\d{4}\b)''', process_vis)


# Runway visual range (RDD/VVVV(VVVVV)FT)
def to_rvr_value(dist, units):
    if dist[0] in ('M', 'P'):
        dist = dist[1:]
    return as_value(dist, units)


def process_rvr(matches):
    dist_units = units(matches.pop('units').lower())
    ret = dict()
    ret[matches['runway']] = to_rvr_value(matches['distance'], dist_units)
    if matches['max_dist']:
        ret[matches['runway']] = (ret[matches['runway']],
                                  to_rvr_value(matches['max_dist'], dist_units))
    if matches['change']:
        change_map = dict(D='down', U='up', N='no change')
        ret[matches['runway']] = (ret[matches['runway']], change_map[matches['change']])

    return ret

rvr = RegexParser(r'''R(?P<runway>\d{2}[RLC]?)
                      /(?P<distance>[MP]?\d{4})
                       (V(?P<max_dist>[MP]?\d{4}))?
                       (?P<units>FT)/?(?P<change>[UDN])?''', process_rvr)


# Present weather (w'w')
precip_abbr = {'DZ': 'Drizzle', 'RA': 'Rain', 'SN': 'Snow', 'SG': 'Snow Grains',
               'IC': 'Ice Crystals', 'PL': 'Ice Pellets', 'GR': 'Hail',
               'GS': 'Small Hail or Snow Pellets', 'UP': 'Unknown Precipitation',
               'RASN': 'Rain and Snow'}


class Weather(namedtuple('WxBase', 'mod desc precip obscur other')):
    lookups = [{'-': 'Light', '+': 'Heavy', 'VC': 'In the vicinity'},
               {'MI': 'Shallow', 'PR': 'Partial', 'BC': 'Patches', 'DR': 'Low Drifting',
                'BL': 'Blowing', 'SH': 'Showers', 'TS': 'Thunderstorm', 'FZ': 'Freezing'},
               precip_abbr,
               {'BR': 'Mist', 'FG': 'Fog', 'FU': 'Smoke', 'VA': 'Volcanic Ash',
                'DU': 'Widespread Dust', 'SA': 'Sand', 'HZ': 'Haze', 'PY': 'Spray'},
               {'PO': 'Well-developed Dust/Sand Whirls', 'SQ': 'Squalls', 'FC': 'Funnel Cloud',
                'SS': 'Sandstorm', 'DS': 'Duststorm'}]

    @classmethod
    def fillin(cls, **kwargs):
        args = [None] * 5
        base = cls(*args)
        return base._replace(**kwargs)

    def __str__(self):
        if self.mod == '+' and self.other == 'FC':
            return 'Tornado'

        return ' '.join(lookup[val] for val, lookup in zip(self, self.lookups) if val)


def process_wx(matches):
    if matches['vdesc']:
        matches['mod'] = matches.pop('vicinity')
        matches['desc'] = matches.pop('vdesc')
        if matches['desc'] == 'ST':
            matches['desc'] = 'TS'
    else:
        matches.pop('vdesc')
        matches.pop('vicinity')

    return Weather(**matches)

wx = RegexParser(r'''(((?P<mod>[-+])|\b)  # Begin with one of these mods or nothing
                      (?P<desc>MI|PR|BC|DR|BL|SH|TS|FZ)?
                      ((?P<precip>(DZ|RA|SN|SG|IC|PL|GR|GS|UP){1,3})
                      |(?P<obscur>BR|FG|FU|VA|DU|SA|HZ|PY)
                      |(?P<other>PO|SQ|FC|SS|DS)))
                     |((?P<vicinity>VC)?(?P<vdesc>SH|TS|ST))''', process_wx, repeat=True)


# Sky condition (NNNhhh or VVhhh or SKC/CLR)
def process_sky(matches):
    coverage_to_value = dict(VV=8, FEW=2, SCT=4, BKN=6, BKM=6, OVC=8)
    if matches.pop('clear'):
        return 'clear'
    hgt = as_value(matches['height'], 100 * units.feet)
    return hgt, coverage_to_value[matches['coverage']], matches['cumulus']

sky_cover = RegexParser(r'''\b(?P<clear>SKC|CLR|NSC|NCD)\b|
                              ((?P<coverage>VV|FEW|SCT|BK[MN]|OVC)
                               \ ?(?P<height>\d{3})
                               (?P<cumulus>CB|TCU)?)''', process_sky, repeat=True)


# Temperature/Dewpoint group -- whole values (TT/TdTd)
def parse_whole_temp(temp):
    if temp in ('//', 'MM'):
        return float('NaN') * units.degC
    elif temp.startswith('M'):
        return -as_value(temp[1:], units.degC)
    else:
        return as_value(temp, units.degC)


def process_temp(matches):
    temp = parse_whole_temp(matches['temperature'])
    if matches['dewpoint']:
        dewpt = parse_whole_temp(matches['dewpoint'])
    else:
        dewpt = float('NaN') * units.degC

    return temp, dewpt

basic_temp = RegexParser(r'''(?P<temperature>(M?\d{2})|MM)/
                             (?P<dewpoint>(M?[\d]{1,2})|//|MM)?''', process_temp)


# Altimeter setting (APPPP)
def process_altimeter(matches):
    if matches['unit'] == 'A':
        alt_unit = 0.01 * units.inHg
    else:
        alt_unit = units('mbar')
    return as_value(matches['altimeter'], alt_unit)

altimeter = RegexParser(r'\b(?P<unit>[AQ])(?P<altimeter>\d{4})', process_altimeter,
                        repeat=True, keepall=False)

#
# Extended International groups
#

# Runway conditions
runway_extent = {'1': 0.1, '2': 0.25, '5': 0.5, '9': 1.0, '/': float('NaN')}
runway_contaminant = {'0': 'Clear and dry', '1': 'Damp', '2': 'Wet and water patches',
                      '3': 'Rime and frost covered', '4': 'Dry snow', '5': 'Wet snow',
                      '6': 'Slush', '7': 'Ice', '8': 'Compacted or rolled snow',
                      '9': 'Frozen ruts or ridges', '/': 'No Report'}


def runway_code_to_depth(code):
    if code == '//':
        return float('NaN') * units.mm
    code = int(code)
    if code < 91:
        return code * units.mm
    elif code < 99:
        return (code - 90) * 5 * units.cm
    else:
        return 'Inoperable'


def runway_code_to_braking(code):
    if code == '//':
        return float('NaN')
    code = int(code)
    if code < 91:
        return float(code) / 100
    else:
        return {91: 'poor', 92: 'medium/poor', 93: 'medium', 94: 'medium/good',
                95: 'good'}.get(code, 'unknown')


def process_runway_state(matches):
    if matches['deposit']:
        matches['deposit'] = runway_contaminant.get(matches['deposit'], 'Unknown')
    if matches['extent']:
        matches['extent'] = runway_extent.get(matches['extent'], 'Unknown')
    if matches['depth']:
        matches['depth'] = runway_code_to_depth(matches['depth'])

    matches['cleared'] = bool(matches['cleared'])
    matches['braking'] = runway_code_to_braking(matches['braking'])

    return matches


runway_state = RegexParser(r'''\bR(?P<runway>\d{2})
                                 /((?P<deposit>[\d/])(?P<extent>[\d/])(?P<depth>\d{2}|//)|(?P<cleared>CLRD))?
                                  (?P<braking>\d{2}|//)''', process_runway_state)

# Trend forecast (mostly international)
trend_forecast_regex = re.compile(r'\b(?P<trend>NOSIG|BECMG|TEMPO)')

#
# Parsers for METAR groups -- remarks
#


# Combine time in the remark with the report datetime to make a proper datetime object
def process_time(matches, context):
    repl = dict(minute=int(matches['minute']))
    if matches['hour']:
        repl['hour'] = int(matches['hour'])

    return context['datetime'].replace(**repl)

# Volcanic eruption, first in NWS reports
volcano = RegexParser(r'[A-Z0-9 .]*VOLCANO[A-Z0-9 .]*')

# Type of automatic station
automated_type = RegexParser(r'\bA[O0][12]A?')


# Peak wind remark (PK WND dddfff/hhmm)
def process_peak_wind(matches, context):
    peak_time = process_time(matches, context)
    return dict(time=peak_time, speed=as_value(matches['speed'], context['speed_units']),
                direction=as_value(matches['direction'], units.deg))

peak_wind = RegexParser(r'''\bPK\ WND\ ?(?P<direction>\d{3})
                              (?P<speed>\d{2,3})/
                              (?P<hour>\d{2})?
                              (?P<minute>\d{2})''', process_peak_wind)


# Wind shift (WSHFT hhmm)
def process_shift(matches, context):
    time = process_time(matches, context)
    front = bool(matches['frontal'])
    return dict(time=time, frontal=front)

wind_shift = RegexParser(r'''\bWSHFT\ (?P<hour>\d{2})?
                               (?P<minute>\d{2})
                               \ (?P<frontal>FROPA)?''', process_shift)


# Tower/surface visibility (TWR(SFC) VIS vvvvv)
def process_twrsfc_vis(matches, *args):
    abbr_to_kind = dict(TWR='tower', SFC='surface')
    return {abbr_to_kind[matches['kind']]: vis_to_float(matches['vis'], units.mile)}

sfc_vis = RegexParser(r'''(?P<kind>TWR|SFC)\ VIS
                          \ (?P<vis>[0-9 /]{1,5})''', process_twrsfc_vis)


# Variable prevailing visibility (VIS vvvvvVvvvvv)
def process_var_vis(matches, *args):
    vis1 = vis_to_float(matches['vis1'], units.mile)
    vis2 = vis_to_float(matches['vis2'], units.mile)
    return vis1, vis2

var_vis = RegexParser(r'''VIS\ (?P<vis1>M?[0-9 /]{1,5})V
                          (?P<vis2>[0-9 /]{1,5})''', process_var_vis)


# Sector visibility (VIS DIR vvvvv)
def process_sector_vis(matches, *args):
    # compass_to_float = dict(N=0, NE=45, E=90, SE=135, S=180, SW=225, W=270, NW=315)
    vis = vis_to_float(matches['vis'], units.mile)
    return {matches['direc']: vis}

sector_vis = RegexParser(r'''VIS\ (?P<direc>[NSEW]{1,2})
                             \ (?P<vis>[0-9 /]{1,5})''', process_sector_vis)


# Lightning
def process_lightning(matches, *args):
    if not matches['dist']:
        matches.pop('dist')

    if not matches['loc']:
        matches.pop('loc')

    if not matches['type']:
        matches.pop('type')
    else:
        type_str = matches['type']
        matches['type'] = []
        while type_str:
            matches['type'].append(type_str[:2])
            type_str = type_str[2:]

    if not matches['frequency']:
        matches.pop('frequency')

    return matches

lightning = RegexParser(r'''((?P<frequency>OCNL|FRQ|CONS)\ )?
                            \bLTG(?P<type>(IC|CG|CC|CA)+)?
                            \ ((?P<dist>OHD|VC|DSNT)\ )?
                              (?P<loc>([NSEW\-]|ALQD?S|\ AND\ |\ THRU\ )+)?\b''',
                        process_lightning)

# Precipitation/Thunderstorm begin and end
precip_times_regex = re.compile(r'([BE])(\d{2,4})')


def process_precip_times(matches, context):
    ref_time = context['datetime']
    kind = matches['precip']
    times = []
    start = None
    for be, time in precip_times_regex.findall(matches['times']):
        if len(time) == 2:
            time = ref_time.replace(minute=int(time))
        else:
            time = ref_time.replace(hour=int(time[:2]), minute=int(time[2:4]))

        if be == 'B':
            start = time
        else:
            if start:
                times.append((start, time))
                start = None
            else:
                times.append((None, time))

    if start:
        times.append((start, None))

    return kind, times

precip_times = RegexParser(r'''(SH)?(?P<precip>TS|DZ|FZRA|RA|SN|SG|IC|PL|GR|GS|UP)
                                    (?P<times>([BE]([0-2]\d)?[0-5]\d)+)''',
                           process_precip_times, repeat=True)


# Thunderstorm (TS LOC MOV DIR)
def process_thunderstorm(matches, *args):
    return matches

thunderstorm = RegexParser(r'''\bTS\ (?P<loc>[NSEW\-]+)(\ MOV\ (?P<mov>[NSEW\-]+))?''',
                           process_thunderstorm)

# Virga
virga = RegexParser(r'''\bVIRGA\ (?P<direction>[NSEW\-])''', grab_group('direction'))


# Variable Ceiling
def process_var_ceiling(matches, *args):
    return (as_value(matches['ceil1'], 100 * units.feet),
            as_value(matches['ceil2'], 100 * units.feet))

var_ceiling = RegexParser(r'\bCIG\ (?P<ceil1>\d{3})V(?P<ceil2>\d{3})\b', process_var_ceiling)


# Variable sky cover
def process_var_sky(matches, *args):
    matches['height'] = as_value(matches['height'], 100 * units.feet)
    matches['cover'] = (matches.pop('cover1'), matches.pop('cover2'))
    return matches

var_sky = RegexParser(r'''\b(?P<cover1>CLR|FEW|SCT|BKN|OVC)
                            (?P<height>\d{3})?\ V
                            \ (?P<cover2>CLR|FEW|SCT|BKN|OVC)''', process_var_sky)

# Mountains obscured
mountains = RegexParser(r'''\bMTNS?(\ PTLY)?(\ OBSCD?)?(\ DSNT)?(\ [NSEW\-]+)?''')

# Significant cloud types (CLD DIR (MOV DIR))
sig_cloud = RegexParser(r'''(?P<cloudtype>CB(MAM)?|TCU|ACC|[ACS]CSL|(APRNT\ ROTOR\ CLD))
                            \ (?P<dir>VC\ ALQD?S|[NSEW-]+)(\ MOV\ (?P<movdir>[NSEW]{1,2}))?''')


# Cloud Types (8/ClCmCh)
def process_cloud_types(matches, *args):
    ret = dict()
    for k, v in matches.items():
        if v == '/':
            ret[k] = None
        else:
            ret[k] = int(v)
    return ret

cloud_types = RegexParser(r'''\b8/(?P<low>[\d/])(?P<middle>[\d/])(?P<high>[\d/])''',
                          process_cloud_types)


# Pressure changes (PRESRR/PRESFR)
def process_pressure_change(matches, *args):
    if matches['tend'] == 'R':
        return 'rising rapidly'
    else:
        return 'falling rapidly'

pressure_change = RegexParser(r'\bPRES(?P<tend>[FR])R\b', process_pressure_change)


# Sea-level pressure (SLPppp)
def process_slp(matches, *args):
    if matches['slp'] == 'NO':
        matches['slp'] = 'NaN'

    slp = as_value(matches['slp'], 0.1 * units('mbar'))
    if slp < 50 * units('mbar'):
        slp += 1000 * units('mbar')
    else:
        slp += 900 * units('mbar')
    return slp

slp = RegexParser(r'SLP(?P<slp>\d{3}|NO)', process_slp)


# No SPECI
nospeci = RegexParser(r'\bNO(\ )?SPECI')

# First/last report
report_sequence = RegexParser(r'''\b(FIRST|LAST)''')


# Parse precip report
def parse_rmk_precip(precip):
    return as_value(precip, 0.01 * units.inch)


# Hourly Precip (Prrrr)
hourly_precip = RegexParser(r'\bP(?P<precip>\d{4})\b', grab_group('precip', parse_rmk_precip))

# 3/6-hour precip (6RRRR)
period_precip = RegexParser(r'\b6(?P<precip>\d{4}|////)',
                            grab_group('precip', parse_rmk_precip))


# Parse snow report
def parse_rmk_snow(snow):
    return as_value(snow, 0.1 * units.inch)

# 6-hour snow (931RRR)
snow_6hr = RegexParser(r'\b931(?P<snow>\d{3})\b', grab_group('snow', parse_rmk_snow))


def parse_rmk_snow_depth(snow):
    return as_value(snow, units.inch)

# Snow depth
snow_depth = RegexParser(r'\b4/(?P<snow>\d{3})\b', grab_group('snow', parse_rmk_snow_depth))

# Snow liquid equivalent (933RRR)
snow_liquid_equivalent = RegexParser(r'\b933(?P<snow>\d{3})\b',
                                     grab_group('snow', parse_rmk_snow))

# 24-hour precip (7RRRR)
daily_precip = RegexParser(r'\b7(?P<precip>\d{4}|////)',
                           grab_group('precip', parse_rmk_precip))

# Hourly ice accretion (I1RRR)
hourly_ice = RegexParser(r'\bI1(?P<ice>\d{3})', grab_group('ice', parse_rmk_precip))

# 3-hour ice accretion (I3RRR)
ice_3hr = RegexParser(r'\bI3(?P<ice>\d{3})', grab_group('ice', parse_rmk_precip))

# 6-hour ice accretion (I6RRR)
ice_6hr = RegexParser(r'\bI6(?P<ice>\d{3})', grab_group('ice', parse_rmk_precip))


# Handles parsing temperature format from remarks
def parse_rmk_temp(temp):
    if temp.startswith('1'):
        return -as_value(temp[1:], 0.1 * units.degC)
    else:
        return as_value(temp, 0.1 * units.degC)


# Hourly temperature (TsTTTsTdTdTd)
def process_hourly_temp(matches, *args):
    temp = parse_rmk_temp(matches['temperature'])
    if matches['dewpoint']:
        dewpt = parse_rmk_temp(matches['dewpoint'])
    else:
        dewpt = float('NaN') * units.degC
    return temp, dewpt

hourly_temp = RegexParser(r'''\bT(?P<temperature>[01]\d{3})
                                 (?P<dewpoint>[01]\d{3})?''', process_hourly_temp)


# 6-hour max temp (1sTTT)
max_temp_6hr = RegexParser(r'\b1(?P<temperature>[01]\d{3})\b',
                           grab_group('temperature', parse_rmk_temp))

# 6-hour max temp (1sTTT)
min_temp_6hr = RegexParser(r'\b2(?P<temperature>[01]\d{3})\b',
                           grab_group('temperature', parse_rmk_temp))


# 24-hour temp (4sTTTsTTT)
def process_daily_temp(matches, *args):
    return parse_rmk_temp(matches['min']), parse_rmk_temp(matches['max'])

daily_temp_range = RegexParser(r'\b4(?P<max>[01]\d{3})\ ?(?P<min>[01]\d{3})\b',
                               process_daily_temp)


# 3-hour pressure tendency (5appp)
def process_press_tend(matches, *args):
    return int(matches['character']), as_value(matches['amount'], 0.1 * units.mbar)

press_tend = RegexParser(r'5(?P<character>[0-8])(?P<amount>\d{3})\b', process_press_tend)


# Parse non-operational sensors
def process_nonop_sensors(matches, *args):
    sensors = dict(RVRNO='Runway Visual Range', PWINO='Present Weather Identifier',
                   PNO='Precipitation', FZRANO='Freezing Rain Sensor',
                   TSNO='Lightning Detection System', VISNO='Secondary Visibility Sensor',
                   CHINO='Secondary Ceiling Height Indicator')
    if matches['nonop']:
        return sensors.get(matches['nonop'], matches['nonop'])
    if matches['nonop2']:
        return sensors.get(matches['nonop2'], matches['nonop2']), matches['loc']

non_op_sensors = RegexParser(r'''\b(?P<nonop>RVRNO|PWINO|PNO|FZRANO|TSNO)
                                  |((?P<nonop2>VISNO|CHINO)\ (?P<loc>\w+))''',
                             process_nonop_sensors, repeat=True)

# Some free-text remarks
pilot_remark = RegexParser(r'([\w\ ;\.]*ATIS\ \w[\w\ ;\.]*)|(QFE[\d\.\ ]+)')

# Parse maintenance flag
maint = RegexParser(r'(?P<maint>\$)', grab_group('maint', bool), default=False)
