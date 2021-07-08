"""Utilities for reading storms databases.

Functionality in this module is experimental and subject to change.
"""

import numpy
import pandas
import fsspec.implementations.ftp

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
    if len(paths) != 1:
        raise ValueError(f"Found {len(paths):d} matching files, expected 1")
    return f"ftp://{server:s}/{paths[0]:s}"


def _bla(*args):
    breakpoint()
    pass


def get_noaa_storms_from_uri(uri):
    """Read NOAA storms db from URI.

    URI can be obtained from :func:`get_noaa_storm_uri`.

    Parameters
    ----------
    uri (str): Source to read from.

    Returns
    -------
    Pandas DataFrame
    """
    db = pandas.read_csv(
        uri,
        dtype=ncei_storm_dtypes,
        parse_dates={"start_datetime": ["BEGIN_DATE_TIME"]},
        infer_datetime_format=True,
        date_parser=_bla,
        )
    return db


def get_noaa_storms(start_time, end_time):
    """Get storms from NOAA storms database.

    Read the NOAA NCEI storms database from its bulk download as documented at
    https://www.ncdc.noaa.gov/stormevents/ftp.jsp.  The NOAA storms database
    contains data from January 1950 until recently.

    Parameters
    ----------
    start_time (datetime.datetime): Start time from which to report storms.
    end_time (datetime.datetime): End time to which to report storms.
    """
    dbs = []
    for year in pandas.date_range(start_time, end_time).year:
        src = get_noaa_storm_uri(year)
        dbs.append(get_noaa_storms_from_uri(uri))
    db = pandas.concat(dbs)
    db = filter_noaa_storms(db, start_time, end_time)
    return db


def filter_noaa_storms(db, start_time, end_time):
    """Filter NOAA storms db based on criteria.

    Parameters
    ----------
    start_time (datetime.datetime): Start time from which to report storms.
    end_time (datetime.datetime): End time to which to report storms.
    """
    raise NotImplementedError()
