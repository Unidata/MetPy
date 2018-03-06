# Copyright (c) 2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Read upper air data from remote archives."""

import contextlib
from io import BytesIO
import json
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import numpy as np
import numpy.ma as ma

from ._tools import UnitLinker
from .cdm import Dataset
from ..calc import get_wind_components
from ..deprecation import deprecated
from ..package_tools import Exporter

exporter = Exporter(globals())


@deprecated('0.6', addendum=' This function is being moved to the Siphon package.',
            pending=False)
@exporter.export
def get_upper_air_data(time, site_id, source='wyoming', **kwargs):
    r"""Download and parse upper air observations from an online archive.

    Parameters
    ----------
    time : datetime
        The date and time of the desired observation.

    site_id : str
        The three letter ICAO identifier of the station for which data should be
        downloaded.

    source : str
        The archive to use as a source of data. Current can be one of 'wyoming' or 'iastate'.
        Defaults to 'wyoming'.

    kwargs
        Arbitrary keyword arguments to use to initialize source

    Returns
    -------
        :class:`metpy.io.cdm.Dataset` containing the data

    .. deprecated:: 0.6.0
      Function has been moved to the Siphon library and will be removed from MetPy in 0.8.0.

    """
    sources = {'wyoming': WyomingUpperAir, 'iastate': IAStateUpperAir}
    src = sources.get(source)
    if src is None:
        raise ValueError('Unknown source for data: {0}'.format(str(source)))

    fobj = src.get_data(time, site_id, **kwargs)
    info = src.parse(fobj)

    ds = Dataset()
    ds.createDimension('pressure', len(info['p'][0]))

    # Simplify the process of creating variables that wrap the parsed arrays and can
    # return appropriate units attached to data
    def add_unit_var(name, std_name, arr, unit):
        var = ds.createVariable(name, arr.dtype, ('pressure',), wrap_array=arr)
        var.standard_name = std_name
        var.units = unit
        ds.variables[name] = UnitLinker(var)
        return var

    # Add variables for all the data columns
    for key, name, std_name in [('p', 'pressure', 'air_pressure'),
                                ('z', 'height', 'geopotential_height'),
                                ('t', 'temperature', 'air_temperature'),
                                ('td', 'dewpoint', 'dew_point_temperature')]:
        data, units = info[key]
        add_unit_var(name, std_name, data, units)

    direc, spd, spd_units = info['wind']
    u, v = get_wind_components(spd, np.deg2rad(direc))
    add_unit_var('u_wind', 'eastward_wind', u, spd_units)
    add_unit_var('v_wind', 'northward_wind', v, spd_units)
    add_unit_var('speed', 'wind_speed', spd, spd_units)
    add_unit_var('direction', 'wind_from_direction', direc, 'deg')

    return ds


class UseSampleData(object):
    r"""Class to temporarily point to local sample data instead of downloading."""

    url_map = {r'http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST'
               r'&YEAR=1999&MONTH=05&FROM=0400&TO=0400&STNM=OUN': 'may4_sounding.txt',
               r'http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST'
               r'&YEAR=2013&MONTH=01&FROM=2012&TO=2012&STNM=OUN': 'sounding_data.txt',
               r'http://mesonet.agron.iastate.edu/json/raob.py?ts=201607301200'
               r'&station=KDEN': 'sounding_iastate.txt',
               r'http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST'
               r'&YEAR=2010&MONTH=12&FROM=0912&TO=0912&STNM=BOI': 'sounding_wyoming_upper.txt',
               r'http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST'
               r'&YEAR=2016&MONTH=05&FROM=2200&TO=2200&STNM=DDC': 'may22_sounding.txt',
               r'http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST'
               r'&YEAR=2002&MONTH=11&FROM=1100&TO=1100&STNM=BNA': 'nov11_sounding.txt'}

    def __init__(self):
        r"""Initialize the wrapper."""
        self._urlopen = urlopen

    def _wrapped_urlopen(self, url):
        r"""Wrap urlopen and look to see if the request should be redirected."""
        from metpy.cbook import get_test_data

        filename = self.url_map.get(url)

        if filename is None:
            return self._urlopen(url)
        else:
            return open(get_test_data(filename, False), 'rb')

    def __enter__(self):
        """Set up our custom `urlopen` wrapper."""
        global urlopen
        urlopen = self._wrapped_urlopen

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the normal `urlopen` functions."""
        global urlopen
        urlopen = self._urlopen


class WyomingUpperAir(object):
    r"""Download and parse data from the University of Wyoming's upper air archive."""

    @staticmethod
    def get_data(time, site_id, region='naconf'):
        r"""Download data from the University of Wyoming's upper air archive.

        Parameters
        ----------
        time : datetime
            Date and time for which data should be downloaded
        site_id : str
            Site id for which data should be downloaded
        region : str
            The region in which the station resides. Defaults to `naconf`.

        Returns
        -------
        a file-like object from which to read the data

        """
        url = ('http://weather.uwyo.edu/cgi-bin/sounding?region={region}&TYPE=TEXT%3ALIST'
               '&YEAR={time:%Y}&MONTH={time:%m}&FROM={time:%d%H}&TO={time:%d%H}'
               '&STNM={stid}').format(region=region, time=time, stid=site_id)
        with contextlib.closing(urlopen(url)) as fobj:
            data = fobj.read()

        # See if the return is valid, but has no data
        if data.find(b'Can\'t') != -1:
            raise ValueError('No data available for {time:%Y-%m-%d %HZ} from region {region} '
                             'for station {stid}.'.format(time=time, region=region,
                                                          stid=site_id))

        # Since the archive text format is embedded in HTML, look for the <PRE> tags
        data_start = data.find(b'<PRE>')
        data_end = data.find(b'</PRE>', data_start)

        # Grab the stuff *between* the <PRE> tags -- 6 below is len('<PRE>\n')
        buf = data[data_start + 6:data_end]
        return BytesIO(buf.strip())

    @staticmethod
    def parse(fobj):
        r"""Parse Wyoming Upper Air Data.

        This parses the particular tabular layout of upper air data used by the University of
        Wyoming upper air archive.

        Parameters
        ----------
        fobj : file-like object
            The file-like object from which the data should be read. This needs to be set up
            to return bytes when read, not strings.

        Returns
        -------
        dict of information used by :func:`get_upper_air_data`

        """
        def to_float(s):
            # Remove all whitespace and replace empty values with NaN
            if not s.strip():
                s = 'nan'
            return float(s)

        # Skip the row of dashes and column names
        for _ in range(2):
            fobj.readline()

        # Parse the actual data, only grabbing the columns for pressure, T/Td, and wind
        unit_strs = ['degC' if u == 'C' else u
                     for u in fobj.readline().decode('ascii').split()]

        # Skip last header row of dashes
        fobj.readline()

        # Initiate lists for variables
        arr_data = []

        # Read all lines of data and append to lists only if there is some data
        for row in fobj:
            level = to_float(row[0:7])
            values = (to_float(row[7:14]), to_float(row[14:21]), to_float(row[21:28]),
                      to_float(row[42:49]), to_float(row[49:56]))

            if any(np.invert(np.isnan(values[1:]))):
                arr_data.append((level,) + values)

        p, z, t, td, direc, spd = np.array(arr_data).T

        return {'p': (p, unit_strs[0]), 'z': (z, unit_strs[1]), 't': (t, unit_strs[2]),
                'td': (td, unit_strs[3]), 'wind': (direc, spd, unit_strs[7])}


class IAStateUpperAir(object):
    r"""Download and parse data from the Iowa State's upper air archive."""

    @staticmethod
    def get_data(time, site_id):
        r"""Download data from the Iowa State's upper air archive.

        Parameters
        ----------
        time : datetime
            Date and time for which data should be downloaded
        site_id : str
            Site id for which data should be downloaded

        Returns
        -------
        list of json data

        """
        url = ('http://mesonet.agron.iastate.edu/json/raob.py?ts={time:%Y%m%d%H}00'
               '&station={stid}').format(time=time, stid=site_id)

        with contextlib.closing(urlopen(url)) as fobj:
            json_data = json.loads(fobj.read().decode('utf-8'))['profiles'][0]['profile']

        # See if the return is valid, but has no data
        if not json_data:
            raise ValueError('No data available for {time:%Y-%m-%d %HZ} '
                             'for station {stid}.'.format(time=time, stid=site_id))
        return json_data

    @staticmethod
    def parse(json_data):
        r"""Parse Iowa State Upper Air Data.

        This parses the JSON formatted data returned by the Iowa State upper air data archive.

        Parameters
        ----------
        json_data : list
            list containing json data

        Returns
        -------
        dict of information used by :func:`get_upper_air_data`

        """
        data = {}
        for pt in json_data:
            for field in ('drct', 'dwpc', 'hght', 'pres', 'sknt', 'tmpc'):
                data.setdefault(field, []).append(np.nan if pt[field] is None else pt[field])

        # Make sure that the first entry has a valid temperature and dewpoint
        idx = np.argmax(~(np.isnan(data['tmpc']) | np.isnan(data['tmpc'])))

        ret = {'p': (ma.masked_invalid(data['pres'][idx:]), 'mbar'),
               'z': (ma.masked_invalid(data['hght'][idx:]), 'meter'),
               't': (ma.masked_invalid(data['tmpc'][idx:]), 'degC'),
               'td': (ma.masked_invalid(data['dwpc'][idx:]), 'degC'),
               'wind': (ma.masked_invalid(data['drct'][idx:]),
                        ma.masked_invalid(data['sknt'][idx:]), 'knot')}

        return ret
