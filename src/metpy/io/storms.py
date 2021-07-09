"""Utilities for reading storms databases.

Functionality in this module is experimental and subject to change.
"""

import numpy
import pandas
import fsspec.implementations.ftp
import pint_pandas

ncei_storm_server = "ftp.ncdc.noaa.gov"
ncei_storm_path = "/pub/data/swdi/stormevents/csvfiles"

# dtypes inferred manually from documentation at
# ftp://ftp.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Bulk-csv-Format.pdf

ncei_storm_dtypes = {
    "BEGIN_YEARMONTH": pandas.StringDtype(),
    "BEGIN_DAY": numpy.uint8,
    "BEGIN_TIME": pandas.StringDtype(),
    "END_YEARMONTH": pandas.StringDtype(),
    "END_DAY": numpy.uint8,
    "END_TIME": pandas.StringDtype(),
    "EPISODE_ID": numpy.uint32,
    "EVENT_ID": numpy.uint32,
    "STATE": pandas.StringDtype(),
    "STATE_FIPS": numpy.uint8,
    "YEAR": numpy.uint16,
    "MONTH_NAME": pandas.StringDtype(),
    "EVENT_TYPE": pandas.StringDtype(),
    "CZ_TYPE": pandas.StringDtype(),
    "CZ_FIPS": numpy.uint16,
    "CZ_NAME": pandas.StringDtype(),
    "WFO": pandas.StringDtype(),
    "BEGIN_DATE_TIME": pandas.StringDtype(),
    "CZ_TIMEZONE": pandas.StringDtype(),
    "END_DATE_TIME": pandas.StringDtype(),
    "INJURIES_DIRECT": numpy.uint16,
    "INJURIES_INDIRECT": numpy.uint16,
    "DEATHS_DIRECT": numpy.uint16,
    "DEATHS_INDIRECT": numpy.uint16,
    "DAMAGE_PROPERTY": pandas.StringDtype(),
    "DAMAGE_CROPS": pandas.StringDtype(),
    "SOURCE": pandas.StringDtype(),
    "MAGNITUDE": numpy.float32,
    "MAGNITUDE_TYPE": pandas.StringDtype(),
    "FLOOD_CAUSE": pandas.StringDtype(),
    "CATEGORY": pandas.StringDtype(),
    "TOR_F_SCALE": pandas.StringDtype(),
    "TOR_LENGTH": numpy.float32,
    "TOR_WIDTH": numpy.float32,
    "TOR_OTHER_WFO": pandas.StringDtype(),
    "TOR_OTHER_CZ_STATE": pandas.StringDtype(),
    "TOR_OTHER_CZ_FLIPS": pandas.StringDtype(),
    "TOR_OTHER_CZ_NAME": pandas.StringDtype(),
    "BEGIN_RANGE": numpy.float32,
    "BEGIN_AZIMUTH": pandas.StringDtype(),
    "BEGIN_LOCATION": pandas.StringDtype(),
    "END_RANGE": numpy.float32,
    "END_AZIMUTH": pandas.StringDtype(),
    "END_LOCATION": pandas.StringDtype(),
    "BEGIN_LAT": numpy.float32,
    "BEGIN_LON": numpy.float32,
    "END_LAT": numpy.float32,
    "END_LON": numpy.float32,
    "EPISODE_NARRATIVE": pandas.StringDtype(),
    "EVENT_NARRATIVE": pandas.StringDtype()}


def get_noaa_storm_uri(year, server=ncei_storm_server, path=ncei_storm_path):
    """Get URI for year for NOAA NCEI storms database.

    Given a datetime object, return the FTP-URI for the file containing
    detailed storm information in CSV format for that year.
    """
    ftpfs = fsspec.implementations.ftp.FTPFileSystem(server)
    paths = ftpfs.glob(path + f"/*details*d{year:d}*.csv.gz")
    if len(paths) > 1:
        raise ValueError(f"Found {len(paths):d} matching files for {year:d}, expected 1")
    elif len(paths) == 0:
        raise FileNotFoundError(f"No storm database found for {year:d}")
    return f"ftp://{server:s}/{paths[0]:s}"


def _parse_ncei_storm_date(begin_date_time, cz_timezone):
    """Parse combination of date-time str and timezone.

    The NCEI storm events database presents dates in local time and timezone
    information in two different fields.  This function takes two pandas
    ``StringArray``s corresponding to the fields ``BEGIN_DATE_TIME`` and
    ``CZ_TIMEZONE`` in the storm database.  It's not intended to be called
    directly, but rather to be passed to :func:`pandas.read_csv`.

    The ``BEGIN_DATE_TIME`` field has a format such as "24-JUN-20 16:20:00".
    The ``CZ_TIMEZONE`` has a non-standard format such as "EST-5".  Note that
    times appear to be recorded in standard time even when daylight savings
    time is in effect.

    Parameters
    ----------
    begin_date_time (StringArray): Array of strings representing date and time.
    cz_timezone (StringArray): Array of strings representing timezone.

    Returns
    -------
    pandas Series with dtype ``datetime64[ns, UTC]``
    """
    return pandas.to_datetime(
        begin_date_time + pandas.Series(cz_timezone).str.replace(
            ".ST-", "-0") + ":00",
        utc=True)


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
        dtype=f"pint[{units:s}]")
    for event_type in event_types:
        idx = db.EVENT_TYPE == event_type
        # workaround for pint-pandas bug with empty slices, see
        # https://github.com/hgrecco/pint-pandas/issues/68
        # remove when https://github.com/hgrecco/pint-pandas/pull/69 merged and
        # released:
        if idx.any():
            quantity_with_unit[idx] = pint_pandas.PintArray(
                db.MAGNITUDE[idx],
                dtype=f"pint[{units:s}]")
    return quantity_with_unit


def _ncei_db_add_windspeeds(db, name="wind_speed"):
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
        "knot")
    db[name] = wind_speed
    return db


def _ncei_db_add_hailsize(db, name="hail_size"):
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
        "centiinch")
    db[name] = hail_size
    return db


def get_noaa_storms_from_uri(
        uri,
        parse_dates=True,
        parse_windspeed=True,
        parse_hailsize=True):
    """Read NOAA storms db from URI.

    URI can be obtained from :func:`get_noaa_storm_uri`.

    For a higher level function, see :func:`get_noaa_storms_for_period`.

    Parameters
    ----------
    uri (str): Source to read from.
    parse_dates (bool): Parse dates from source and add fields.
    parse_windspeed (bool): Add windspeed as a unit-aware field.
    parse_hailsize (bool): Add hailsize as a unit-aware field.

    Returns
    -------
    Pandas DataFrame
    """
    db = pandas.read_csv(
        uri,
        dtype=ncei_storm_dtypes,
        parse_dates={"start_datetime": ["BEGIN_DATE_TIME", "CZ_TIMEZONE"]},
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
    """Get storms from NOAA storms database.

    Read the NOAA NCEI storms database from its bulk download as documented at
    https://www.ncdc.noaa.gov/stormevents/ftp.jsp.  The NOAA storms database
    contains data from January 1950 until recently.

    Parameters
    ----------
    start_time (datetime.datetime): Start time from which to report storms.
        Interpreted as UTC.
    end_time (datetime.datetime): End time to which to report storms.
        Interpreted as UTC.
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
    """
    idx = ((db.start_datetime >= pandas.Timestamp(start_time, tz="UTC"))
           & (db.start_datetime <= pandas.Timestamp(end_time, tz="UTC")))
    return db[idx]
