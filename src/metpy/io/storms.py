# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Utilities for reading storms databases.

This module contains functionality for reading storms databases.  Currently
implemented are routines to read the NOAA NCEI Storm Events Database
located at https://www.ncdc.noaa.gov/stormevents/, which includes a database
dump in CSV format.  The primary function to obtain data from this database is
:func:`get_noaa_storms_for_period`.

Functionality in this module is experimental and subject to change.

Example use::

    >>> import metpy.io.storms
    >>> import datetime
    >>> db = metpy.io.storms.get_noaa_storms_for_period(
    ...         datetime.datetime(2019, 12, 1),
    ...         datetime.datetime(2020, 2, 1))
    >>> print(db.head())
                    start_datetime END_YEARMONTH  ...  wind_speed hail_size
    874  2019-12-01 15:00:00+00:00        201912  ...         nan       nan
    1220 2019-12-22 19:10:00+00:00        201912  ...        70.0       nan
    1221 2019-12-23 00:33:00+00:00        201912  ...         nan       nan
    1286 2019-12-22 15:00:00+00:00        201912  ...         nan       nan
    1287 2019-12-25 15:00:00+00:00        201912  ...         nan       nan

    [5 rows x 50 columns]
"""

import fsspec.implementations.ftp
import numpy
import pandas
import pint_pandas

ncei_storm_server = 'ftp.ncdc.noaa.gov'
ncei_storm_path = '/pub/data/swdi/stormevents/csvfiles'

# dtypes inferred manually from documentation at
# ftp://ftp.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Bulk-csv-Format.pdf

ncei_storm_dtypes = {
    'BEGIN_YEARMONTH': pandas.StringDtype(),
    'BEGIN_DAY': pandas.StringDtype(),
    'BEGIN_TIME': pandas.StringDtype(),
    'END_YEARMONTH': pandas.StringDtype(),
    'END_DAY': numpy.uint8,
    'END_TIME': pandas.StringDtype(),
    'EPISODE_ID': numpy.float32,  # to allow for missing data
    'EVENT_ID': numpy.uint32,
    'STATE': pandas.StringDtype(),
    'STATE_FIPS': numpy.uint8,
    'YEAR': numpy.uint16,
    'MONTH_NAME': pandas.StringDtype(),
    'EVENT_TYPE': pandas.StringDtype(),
    'CZ_TYPE': pandas.StringDtype(),
    'CZ_FIPS': numpy.uint16,
    'CZ_NAME': pandas.StringDtype(),
    'WFO': pandas.StringDtype(),
    'BEGIN_DATE_TIME': pandas.StringDtype(),
    'CZ_TIMEZONE': pandas.StringDtype(),
    'END_DATE_TIME': pandas.StringDtype(),
    'INJURIES_DIRECT': numpy.uint16,
    'INJURIES_INDIRECT': numpy.uint16,
    'DEATHS_DIRECT': numpy.uint16,
    'DEATHS_INDIRECT': numpy.uint16,
    'DAMAGE_PROPERTY': pandas.StringDtype(),
    'DAMAGE_CROPS': pandas.StringDtype(),
    'SOURCE': pandas.StringDtype(),
    'MAGNITUDE': numpy.float32,
    'MAGNITUDE_TYPE': pandas.StringDtype(),
    'FLOOD_CAUSE': pandas.StringDtype(),
    'CATEGORY': pandas.StringDtype(),
    'TOR_F_SCALE': pandas.StringDtype(),
    'TOR_LENGTH': numpy.float32,
    'TOR_WIDTH': numpy.float32,
    'TOR_OTHER_WFO': pandas.StringDtype(),
    'TOR_OTHER_CZ_STATE': pandas.StringDtype(),
    'TOR_OTHER_CZ_FLIPS': pandas.StringDtype(),
    'TOR_OTHER_CZ_NAME': pandas.StringDtype(),
    'BEGIN_RANGE': numpy.float32,
    'BEGIN_AZIMUTH': pandas.StringDtype(),
    'BEGIN_LOCATION': pandas.StringDtype(),
    'END_RANGE': numpy.float32,
    'END_AZIMUTH': pandas.StringDtype(),
    'END_LOCATION': pandas.StringDtype(),
    'BEGIN_LAT': numpy.float32,
    'BEGIN_LON': numpy.float32,
    'END_LAT': numpy.float32,
    'END_LON': numpy.float32,
    'EPISODE_NARRATIVE': pandas.StringDtype(),
    'EVENT_NARRATIVE': pandas.StringDtype()}


def get_noaa_storm_uri(year, server=ncei_storm_server, path=ncei_storm_path):
    """Get URI for year for NOAA NCEI storms database.

    Given a year, return the FTP-URI for the file containing
    detailed storm information in CSV format for that year.

    Parameters
    ----------
    year (int): Year for which to obtain the URI
    server (Optional[str]): FTP server where the storm database is located.
        Defaults to the NCEI storm database server as of July 2021.
    path (Optional[str]): Path on FTP server where the storm database is
        located.  Defailts to the location on the NCEI storm database server as
        of July 2021.

    Returns
    -------
    String containing the FTP URI pointing to the compressed CSV file
    containing the storm database for the year requested.
    """
    ftpfs = fsspec.implementations.ftp.FTPFileSystem(server)
    paths = ftpfs.glob(path + f'/*details*d{year:d}*.csv.gz')
    if len(paths) > 1:
        raise ValueError(f'Found {len(paths):d} matching files for {year:d}, expected 1')
    elif len(paths) == 0:
        raise FileNotFoundError(f'No storm database found for {year:d}')
    return f'ftp://{server:s}/{paths[0]:s}'


# Timezone abbreviations as found in NCEI storm database.
# Although newer entries contain the offset in a non-standard format (single
# unpadded integer for hours compared to UTC), older entries contain only the
# timezone code.
# Standardised libraries are tricky here because those timezone abbreviations
# are unique in the USA but not globally.
_ncei_tz = {
    'SST': '-11:00',
    'HST': '-10:00',
    'AKST': '-09:00',
    'PST': '-08:00',
    'MST': '-07:00',
    'CST': '-06:00',
    'EST': '-05:00',
    'AST': '-04:00',
    'GST': '+10:00',  # never called ChST even in 2021
    # found in older entries
    'PDT': '-07:00',
    'MDT': '-06:00',
    'CDT': '-05:00',
    'EDT': '-04:00',
    # also found
    'Cst': '-06:00',
    'SCT': '-06:00',
    'CSt': '-06:00',
    'CSC': '-06:00',
    'Est': '-05:00',
    'ESt': '-05:00',
    'UNK': None,  # will become NaT, found in very old entries
    'GMT': '+00:00'}


def _parse_ncei_storm_date(begin_yearmonth, begin_day, begin_time,
                           cz_timezone):
    """Parse combination of date-time str and timezone.

    The NCEI storm events database presents dates in local time and
    timezone information in two different fields.  This function
    takes our pandas ``StringArray``s corresponding to the fields
    ``BEGIN_YEARMONTH``, ``BEGIN_DAY``, ``BEGIN_TIME``, and
    ``CZ_TIMEZONE`` in the storm database.  It's not intended to be
    called directly, but rather to be passed to :func:`pandas.read_csv`
    to the ``date_parser`` argument.

    The ``BEGIN_DATE_TIME`` field has a format such as "24-JUN-20 16:20:00".
    The ``CZ_TIMEZONE`` has a non-standard format such as "EST-5" for newer
    entries or just "EST" for older entries.  Note that times appear to be
    recorded in standard time even when daylight savings time is in effect,
    except for older entries where "EDT" etc. do occur.

    Parameters
    ----------
    begin_yearmonth (StringArray): Array of strings representing year and
        month.
    begin_day (StringArray): Array of strings representing day of month.
    begin_time (StringArray): Array of strings representing local time.
    cz_timezone (StringArray): Array of strings representing timezone.

    Returns
    -------
    pandas Series with dtype ``datetime64[ns, UTC]``
    """
    tz_label = pandas.Series(cz_timezone).str.replace(r'-?\d*', '')
    tz_offset = pandas.Series(
        [_ncei_tz[label] for label in tz_label],
        dtype=pandas.StringDtype())

    times = pandas.to_datetime(
        pandas.Series(begin_yearmonth) + '-' + pandas.Series(begin_day)
        + ' ' + pandas.Series(begin_time) + ' ' + tz_offset,
        format='%Y%m-%d %H%M %z',
        errors='coerce',
        utc=True)

    return times


def _ncei_db_get_magnitude_with_unit(db, event_types, units):
    """Get magnitude with unit from NCEI storm db.

    From the NCEI storm db as a pandas array, interpret the MAGNITUDE field as
    a pint-pandas array with associated units.

    Parameters
    ----------
    db (pandas.DataFrame): NCEI storm database
    fields (List[str]): List of event types for which the MAGNITUDE field
        should be interpreted as describing a quantity with associated unit.
    units (str): Unit contained by MAGNITUDE field.  Should be interpretable
        by the pint-pandas unit registry.
    """
    quantity_with_unit = pint_pandas.PintArray(
        numpy.full(db.shape[0], numpy.nan),
        dtype=f'pint[{units:s}]')
    for event_type in event_types:
        idx = db.EVENT_TYPE == event_type
        # workaround for pint-pandas bug with empty slices, see
        # https://github.com/hgrecco/pint-pandas/issues/68
        # remove when https://github.com/hgrecco/pint-pandas/pull/69 merged and
        # released:
        if idx.any():
            quantity_with_unit[idx] = pint_pandas.PintArray(
                db.MAGNITUDE[idx],
                dtype=f'pint[{units:s}]')
    return quantity_with_unit


def _ncei_db_add_windspeeds(db, name='wind_speed'):
    """Add windspeeds to NCEI storm db.

    For any of the event types associated with wind, add the wind speed to the
    NCEI db as a new field.  The field will have units as written in the NCEI
    database, "knot".

    Parameters
    ----------
    db (pandas.DataFrame): NCEI storm database
    name (Optional[str]): Field in which to add this.  Defaults to
        "wind_speed".
    """
    wind_speed = _ncei_db_get_magnitude_with_unit(
        db,
        ['Thunderstorm Wind', 'Strong Wind', 'High Wind',
         'Marine Thunderstorm Wind', 'Marine High Wind',
         'Marine Strong Wind'],
        'knot')
    db[name] = wind_speed
    return db


def _ncei_db_add_hailsize(db, name='hail_size'):
    """Add hail sizes to NCEI storm db.

    For any of the event types associated with hail, add the hail size to the
    NCEI db as a new field.  The field will have units as written in the NCEI
    database, 1/100th of an inch (here centiinch, because units in pint cannot
    have scaling factors)

    Parameters
    ----------
    db (pandas.DataFrame): NCEI storm database
    name (Optional[str]): Field in which to add this.  Defaults to
        "hail_size".
    """
    hail_size = _ncei_db_get_magnitude_with_unit(
        db,
        ['Hail', 'Marine Hail'],
        'centiinch')
    db[name] = hail_size
    return db


def get_noaa_storms_from_uri(
        uri,
        parse_dates=True,
        parse_windspeed=True,
        parse_hailsize=True):
    """Read NCEI storms db from URI.

    Read the NCEI storms database located at the indicated Uniform Resource
    Identifier (URI).  Optionally parse dates, wind speeds, and hail size.
    Date parsing will add a ``start_datetime`` column, which replaces the
    columns ``BEGIN_YEARMONTH``, ``BEGIN_DAY``, ``BEGIN_TIME``, and
    ``CZ_TIMEZONE``.  The ``start_datetime`` column will have the dtype
    ``datetime64[ns, UTC]``.  Wind speeds and hail sizes are (optionally) added
    as extra units-aware fields using pint-pandas, to extend the units-unaware
    ``MAGNITUDE`` field, which mixes information for both and is therefore
    harder to use.

    The URI containing data certain year can be obtained from
    :func:`get_noaa_storm_uri`, or to read all storm entries in an indicated
    period, use :func:`get_noaa_storms_for_period`.

    Parameters
    ----------
    uri (str): Source to read from.
    parse_dates (bool): Parse dates from source and add fields.
    parse_windspeed (bool): Add windspeed as a unit-aware field.
    parse_hailsize (bool): Add hailsize as a unit-aware field.

    Returns
    -------
    Pandas DataFrame containing columns from the database plus those
    interpreted by this function (see above).
    """
    db = pandas.read_csv(
        uri,
        dtype=ncei_storm_dtypes,
        parse_dates=({
            'start_datetime': [
                'BEGIN_YEARMONTH', 'BEGIN_DAY', 'BEGIN_TIME', 'CZ_TIMEZONE']}
            if parse_dates else None),
        date_parser=_parse_ncei_storm_date)
    # Add units in a separate step.  The units of the magnitude field depend
    # on the quantity in the event_type field.
    if parse_windspeed:
        db = _ncei_db_add_windspeeds(db)
    if parse_hailsize:
        db = _ncei_db_add_hailsize(db)
    return db


def get_noaa_storms_for_period(
        start_time, end_time, server=ncei_storm_server, path=ncei_storm_path,
        parse_dates=True, parse_windspeed=True, parse_hailsize=True):
    """Get storms from NOAA NCEI storms database.

    Obtain storms from the NOAA NCEI storms database between the indicated
    times.  Underlying data reading and interpretation is performed using
    :func:`get_noaa_storms_from_uri`, see documentation there on what fields
    are added and how to control this using ``parse_dates``,
    ``parse_windspeed``, and ``parse_hailsize``.  Data are obtained from the
    bulk download at https://www.ncdc.noaa.gov/stormevents/ftp.jsp.  The NOAA
    storms database contains data from January 1950 until recently.

    Parameters
    ----------
    start_time (datetime.datetime): Start time from which to report storms.
        Interpreted as UTC.
    end_time (datetime.datetime): End time to which to report storms.
        Interpreted as UTC.
    server (Optional[str]): Server to read from.  Defaults to NCEI server.
    path (Optional[str]): Path (directory) to read from.  Defaults to NCEI
        location as of July 2021.
    parse_dates (Optional[bool]): Parse dates to datetime64 objects.  Defaults
        to True.
    parse_windspeed (Optional[bool]): Parse wind speeds to units-aware objects.
        Defaults to true.
    parse_hailsize (Optional[bool]): Parse hail sizes to units-aware objects.
        Defaults to true.

    Returns
    -------
    pandas.DataFrame containing the storms in the indicated period.  WIll
    include fields from the NCEI database plus those interpreted by
    :func:`get_noaa_storms_from_uri`.

    Raises
    ------
    ValueError: no storms can be found
    """
    dbs = []
    for year in range(start_time.year, end_time.year + 1):
        try:
            uri = get_noaa_storm_uri(year, server=server, path=path)
        except FileNotFoundError:
            continue  # no file for this year
        dbs.append(
            get_noaa_storms_from_uri(
                uri, parse_dates=parse_dates,
                parse_windspeed=parse_windspeed, parse_hailsize=parse_hailsize))
    db = pandas.concat(dbs)
    db = filter_noaa_storms(db, start_time, end_time)
    return db


def filter_noaa_storms(db, start_time, end_time):
    """Filter NOAA storms db based on criteria.

    Select NOAA storms occurring between ``start_time`` and ``end_time``.

    Parameters
    ----------
    db (pandas.DataFrame): DataFrame describing the NCEI storm database,
        including a ``start_datetime`` field, which will be present if the
        database was obtained with :func:`get_noaa_storms_from_uri` or
        :func:`get_noaa_storms_for_period` and ``parse_dates=True`` (the
        default).
    start_time (datetime.datetime): Start time from which to report storms.
    end_time (datetime.datetime): End time to which to report storms.

    Returns
    -------
    pandas.DataFrame selected based on criteria given.
    """
    idx = ((db.start_datetime >= pandas.Timestamp(start_time, tz='UTC'))
           & (db.start_datetime <= pandas.Timestamp(end_time, tz='UTC')))
    return db[idx]
