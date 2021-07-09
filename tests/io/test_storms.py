"""Test functionality to read storm events database."""

import os
import pytest
import gzip

import pandas
import datetime
import unittest.mock
import fsspec.implementations.local

# selected and edited from <URL:https://www1.ncdc.noaa.gov/pub/data/swdi/
# stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2020_c20210604.csv.gz>
# on 2021-07-08

csv_fake_content = """BEGIN_YEARMONTH,BEGIN_DAY,BEGIN_TIME,END_YEARMONTH,END_DAY,END_TIME,EPISODE_ID,EVENT_ID,STATE,STATE_FIPS,YEAR,MONTH_NAME,EVENT_TYPE,CZ_TYPE,CZ_FIPS,CZ_NAME,WFO,BEGIN_DATE_TIME,CZ_TIMEZONE,END_DATE_TIME,INJURIES_DIRECT,INJURIES_INDIRECT,DEATHS_DIRECT,DEATHS_INDIRECT,DAMAGE_PROPERTY,DAMAGE_CROPS,SOURCE,MAGNITUDE,MAGNITUDE_TYPE,FLOOD_CAUSE,CATEGORY,TOR_F_SCALE,TOR_LENGTH,TOR_WIDTH,TOR_OTHER_WFO,TOR_OTHER_CZ_STATE,TOR_OTHER_CZ_FIPS,TOR_OTHER_CZ_NAME,BEGIN_RANGE,BEGIN_AZIMUTH,BEGIN_LOCATION,END_RANGE,END_AZIMUTH,END_LOCATION,BEGIN_LAT,BEGIN_LON,END_LAT,END_LON,EPISODE_NARRATIVE,EVENT_NARRATIVE,DATA_SOURCE
202006,24,1620,202006,24,1620,149684,902190,"GEORGIA",13,2020,"June","Thunderstorm Wind","C",321,"WORTH","TAE","24-JUN-20 16:20:00","EST-5","24-JUN-20 16:20:00","0","0","0","0","0.00K","0.00K","911 Call Center","50.00","EG",,,,,,,,,,"1","W","DOLES","1","W","DOLES","31.7","-83.89","31.7","-83.89","Bla.","Bla.","CSV"
202005,25,1700,202005,25,2000,147310,885808,"WEST VIRGINIA",54,2020,"May","Flash Flood","C",101,"WEBSTER","RLX","25-MAY-20 17:00:00","EST-5","25-MAY-20 20:00:00","0","0","0","0","3.00K","0.00K","Department of Highways",,,"Heavy Rain",,,,,,,,,"1","NNW","ERBACON","2","E","ERBACON","38.5367","-80.5887","38.5186","-80.5378","Bla","Bla","CSV"
202002,6,1600,202002,7,2100,146077,877747,"NEW YORK",36,2020,"February","Winter Storm","Z",41,"NORTHERN SARATOGA","ALY","06-FEB-20 16:00:00","EST-5","07-FEB-20 21:00:00","0","0","0","0",,,"Trained Spotter",,,,,,,,,,,,,,,,,,,,,,"Bla","Bla","CSV"
202005,22,1931,202005,22,1931,147059,883975,"ALABAMA",1,2020,"May","Hail","C",95,"MARSHALL","HUN","22-MAY-20 19:31:00","CST-6","22-MAY-20 19:31:00","0","0","0","0","0.00K","0.00K","Emergency Manager","1.75",,,,,,,,,,,"0","N","HORTON","0","N","HORTON","34.2","-86.3","34.2","-86.3","Bla","Bla","CSV"
202005,22,1932,202005,22,1932,147059,883976,"ALABAMA",1,2020,"May","Hail","C",95,"MARSHALL","HUN","22-MAY-20 19:32:00","CST-6","22-MAY-20 19:32:00","0","0","0","0","0.00K","0.00K","Emergency Manager","1.75",,,,,,,,,,,"0","N","HORTON","0","N","HORTON","34.2","-86.3","34.2","-86.3","Bla","Bla","CSV"
202004,10,0,202004,10,1500,145924,881761,"WEST VIRGINIA",54,2020,"April","Strong Wind","Z",31,"HARRISON","RLX","10-APR-20 00:00:00","EST-5","10-APR-20 15:00:00","0","0","0","0","5.00K","0.00K","ASOS","31.00","MG",,,,,,,,,,,,,,,,,,,,"Bla","Bla","CSV"
202004,13,30,202004,13,330,146523,880416,"VIRGINIA",51,2020,"April","Flood","C",51,"DICKENSON","RLX","13-APR-20 00:30:00","EST-5","13-APR-20 03:30:00","0","0","0","0","5.00K","0.00K","911 Call Center",,,"Heavy Rain",,,,,,,,,"1","E","OSBORNS GAP","2","NE","BARTLICK","37.1988","-82.5269","37.2685","-82.3034","Bla","Bla","CSV"
202004,12,1900,202004,13,600,146523,881979,"VIRGINIA",51,2020,"April","Strong Wind","Z",3,"DICKENSON","RLX","12-APR-20 19:00:00","EST-5","13-APR-20 06:00:00","0","0","0","0","8.00K","0.00K","Public","42.00","MG",,,,,,,,,,,,,,,,,,,,"Bla","Bla","CSV"
202001,29,1555,202001,29,1855,146334,879265,"CALIFORNIA",6,2020,"January","High Wind","Z",54,"LOS ANGELES COUNTY MOUNTAINS EXCLUDING THE SANTA MONICA RANGE","LOX","29-JAN-20 15:55:00","PST-8","29-JAN-20 18:55:00","0","0","0","0","0.00K","0.00K","RAWS","56.00","MG",,,,,,,,,,,,,,,,,,,,"Bla","Bla","CSV"
202007,30,1732,202007,30,1732,149478,900958,"ATLANTIC NORTH",88,2020,"July","Marine Thunderstorm Wind","Z",535,"TIDAL POTOMAC KEY BRIDGE TO INDIAN HD MD","LWX","30-JUL-20 17:32:00","EST-5","30-JUL-20 17:32:00","0","0","0","0","0.00K","0.00K","ASOS","34.00","EG",,,,,,,,,,"0","N","(DCA)REAGAN NATIONAL AIRPORT","0","N","(DCA)REAGAN NATIONAL AIRPORT","38.86","-77.03","38.86","-77.03","Bla","Bla","CSV"
202003,4,1033,202003,4,1300,145274,872239,"TEXAS",48,2020,"March","Flash Flood","C",317,"MARTIN","MAF","04-MAR-20 10:33:00","CST-6","04-MAR-20 13:00:00","0","0","0","0","0.50K","0.00K","Department of Highways",,,"Heavy Rain",,,,,,,,,"12","WSW","TARZAN","11","WSW","TARZAN","32.2156","-102.1478","32.2207","-102.1252","Bla","Bla","CSV"
202006,28,1644,202006,28,1645,150225,906375,"ALABAMA",1,2020,"June","Thunderstorm Wind","C",11,"BULLOCK","BMX","28-JUN-20 16:44:00","CST-6","28-JUN-20 16:45:00","0","0","0","0","0.00K","0.00K","911 Call Center","50.00","EG",,,,,,,,,,"0","N","UNION SPGS","0","N","UNION SPGS","32.15","-85.72","32.15","-85.72","Bla","Bla","CSV"
202007,25,300,202007,26,500,151469,912764,"GULF OF MEXICO",85,2020,"July","Marine Tropical Storm","Z",255,"MATAGORDA SHIP CHNL TO PT ARANSAS OUT 20NM","CRP","25-JUL-20 03:00:00","CST-6","26-JUL-20 05:00:00","0","0","0","0","0.00K","0.00K","Mesonet",,,,,,,,,,,,,,,,,,,,,,"Bla","Bla","CSV"
202005,16,200,202005,16,230,148038,891439,"TEXAS",48,2020,"May","Flash Flood","C",355,"NUECES","CRP","16-MAY-20 02:00:00","CST-6","16-MAY-20 02:30:00","0","0","0","0","0.00K","0.00K","Public",,,"Heavy Rain",,,,,,,,,"2","SE","CORPUS CHRISTI","2","SE","CORPUS CHRISTI","27.76","-97.4","27.7563","-97.4014","Bla","Bla","CSV"
202006,2,2320,202006,2,2320,149944,903982,"NEW YORK",36,2020,"June","Thunderstorm Wind","C",9,"CATTARAUGUS","BUF","02-JUN-20 23:20:00","EST-5","02-JUN-20 23:20:00","0","0","0","0","2.00K","0.00K","Law Enforcement","51.00","EG",,,,,,,,,,"1","W","MACHIAS","1","W","MACHIAS","42.42","-78.49","42.42","-78.49","Bla","Bla","CSV"
202007,7,1030,202007,7,1515,150118,905130,"TEXAS",48,2020,"July","Flood","C",293,"LIMESTONE","FWD","07-JUL-20 10:30:00","CST-6","07-JUL-20 15:15:00","0","0","0","0","0.00K","0.00K","Emergency Manager",,,"Heavy Rain",,,,,,,,,"2","NNE","FROSA","2","SW","ECHOLS","31.6618","-96.6891","31.6645","-96.6756","Bla","Bla","CSV"
202001,11,318,202001,11,323,145633,874565,"ARKANSAS",5,2020,"January","Thunderstorm Wind","C",93,"MISSISSIPPI","MEG","11-JAN-20 03:18:00","CST-6","11-JAN-20 03:23:00","0","0","0","0","2.00K","0.00K","Social Media","50.00","EG",,,,,,,,,,"0","N","BURDETTE","0","N","BURDETTE","35.82","-89.93","35.82","-89.93","Bla","Bla","CSV"
202011,15,1000,202011,15,1700,154156,929709,"OHIO",39,2020,"November","Strong Wind","Z",74,"HOCKING","ILN","15-NOV-20 10:00:00","EST-5","15-NOV-20 17:00:00","0","0","0","0","5.00K",,"911 Call Center","43.00","EG",,,,,,,,,,,,,,,,,,,,"Bla","Bla","CSV"
202011,12,1000,202011,12,1615,154105,930975,"NORTH CAROLINA",37,2020,"November","Flash Flood","C",195,"WILSON","RAH","12-NOV-20 10:00:00","EST-5","12-NOV-20 16:15:00","0","0","0","0","0.00K","0.00K","Unknown",,,"Heavy Rain",,,,,,,,,"4","WSW","WILSON","1","W","CONTENTNEA","35.7137","-77.9848","35.6813","-77.9426","Bla","Bla","CSV"
"""  # noqa: E501


@pytest.fixture
def fake_csv_files(tmp_path):
    """Create fake CSV files and return their paths."""
    d = tmp_path / "csvfiles"
    fns = []
    for year in (1980, 2000, 2020):
        fn = d / f"StormEvents_details-ftp_v1.0_d{year:d}_c20210604.csv.gz"
        fn.parent.mkdir(exist_ok=True, parents=True)
        with gzip.GzipFile(fn, "w") as fp:
            fp.write(csv_fake_content.replace("2020", f"{year:d}", 1).encode("ascii"))
        fns.append(fn)
    return fns


def test_infer_uri(fake_csv_files):
    """Test inferring the URI for a certain year/timestamp."""
    from metpy.io.storms import get_noaa_storm_uri
    fakedir = os.fspath(fake_csv_files[0].parent)
    with unittest.mock.patch(
            "fsspec.implementations.ftp.FTPFileSystem",
            new=fsspec.implementations.local.LocalFileSystem):
        for year in (1980, 2000, 2020):
            storm_uri = get_noaa_storm_uri(year, path=fakedir)
            assert f"_d{year:>d}_" in storm_uri
            assert "details" in storm_uri
            assert storm_uri.startswith("ftp://")
        with pytest.raises(FileNotFoundError):
            get_noaa_storm_uri(1854, path=fakedir)
        fake_csv_files[-1].with_name(
            "StormEvents_details-ftp_v1.0_d2020_c20210709.csv.gz").touch()
        with pytest.raises(ValueError):
            get_noaa_storm_uri(2020, path=fakedir)


def test_get_noaa_storms_from_uri(fake_csv_files):
    """Test getting NOAA storms db from URI."""
    import pint_pandas
    from metpy.io.storms import get_noaa_storms_from_uri
    db = get_noaa_storms_from_uri(fake_csv_files[-1])
    assert db.shape == (19, 52)
    assert db.dtypes.start_datetime == pandas.DatetimeTZDtype("ns", "UTC")
    assert all(db.start_datetime.dt.year == 2020)
    assert db.dtypes.wind_speed.units == pint_pandas.PintType.ureg.knot
    assert db.dtypes.hail_size.units == pint_pandas.PintType.ureg.centiinch


def test_noaa_storms(fake_csv_files):
    """Test reading noaa storms."""
    from metpy.io.storms import get_noaa_storms_for_period

    def fake_storm_uri(year, server=None, path=None):
        if year == 1980:
            return fake_csv_files[0]
        elif year == 2000:
            return fake_csv_files[1]
        elif year == 2020:
            return fake_csv_files[2]
        else:
            raise FileNotFoundError(f"No storm database found for {year:d}")
    with unittest.mock.patch("metpy.io.storms.get_noaa_storm_uri",
                             new=fake_storm_uri):
        db = get_noaa_storms_for_period(
            datetime.datetime(2020, 6, 1),
            datetime.datetime(2020, 7, 1))
        assert db.shape == (3, 52)
        db = get_noaa_storms_for_period(
            datetime.datetime(1970, 1, 1),
            datetime.datetime(2020, 12, 31))
        assert db.shape == (57, 52)
