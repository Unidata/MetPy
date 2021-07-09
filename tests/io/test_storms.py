# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test functionality to read storm events database."""

import os
import pytest
import gzip

import numpy.testing
import pandas
import datetime
import unittest.mock
import fsspec.implementations.local

# selected and edited from files in <URL:https://www1.ncdc.noaa.gov/pub/data/swdi/
# stormevents/csvfiles/> on 2021-07-09

csv_fake_content = """BEGIN_YEARMONTH,BEGIN_DAY,BEGIN_TIME,END_YEARMONTH,END_DAY,END_TIME,EPISODE_ID,EVENT_ID,STATE,STATE_FIPS,YEAR,MONTH_NAME,EVENT_TYPE,CZ_TYPE,CZ_FIPS,CZ_NAME,WFO,BEGIN_DATE_TIME,CZ_TIMEZONE,END_DATE_TIME,INJURIES_DIRECT,INJURIES_INDIRECT,DEATHS_DIRECT,DEATHS_INDIRECT,DAMAGE_PROPERTY,DAMAGE_CROPS,SOURCE,MAGNITUDE,MAGNITUDE_TYPE,FLOOD_CAUSE,CATEGORY,TOR_F_SCALE,TOR_LENGTH,TOR_WIDTH,TOR_OTHER_WFO,TOR_OTHER_CZ_STATE,TOR_OTHER_CZ_FIPS,TOR_OTHER_CZ_NAME,BEGIN_RANGE,BEGIN_AZIMUTH,BEGIN_LOCATION,END_RANGE,END_AZIMUTH,END_LOCATION,BEGIN_LAT,BEGIN_LON,END_LAT,END_LON,EPISODE_NARRATIVE,EVENT_NARRATIVE,DATA_SOURCE
195004,28,1445,195004,28,1445,,10096222,"OKLAHOMA",40,1950,"April","Tornado","C",149,"WASHITA",,"28-APR-50 14:45:00","CST","28-APR-50 14:45:00","0","0","0","0","250K","0",,"0",,,,"F3","3.4","400",,,,,"0",,,"0",,,"35.12","-99.20","35.17","-99.20",,,"PUB"
202103,16,2235,202103,16,2235,157216,950338,"KANSAS",20,2021,"March","Hail","C",111,"LYON","TOP","16-MAR-21 22:35:00","CST-6","16-MAR-21 22:35:00","0","0","0","0","0.00K","0.00K","Trained Spotter","1.00",,,,,,,,,,,"6","NNE","PLYMOUTH","6","NNE","PLYMOUTH","38.5","-96.26","38.5","-96.26","A couple of reports of quarter size hail occurred late on March 16th with a small number of t-storms.","A spotter reported quarter size hail.","CSV"
199706,11,600,199706,11,1800,2071988,5623668,"GUAM",98,1997,"June","Hurricane (Typhoon)","Z",5,"NORTHERN MARIANAS","GUA","11-JUN-97 06:00:00","SST","11-JUN-97 18:00:00","0","0","0","0",,,"OTHER FEDERAL AGENCY",,,,,,,,,,,,,,,,,,,,,,"Super Typhoon Nestor (07W) moved through the Guam Area of Responsibility from the 1st to the 12th.It passed through the northern Marianas Islands, passing 45 miles away from the island of Agrihan on the 11th. No damage was reported.Typhoon Opal (08W) moved through the Guam Area of Responsibility from the 13th to the 18th. It became a tropical storm when it was out over the ocean far to the west of Guam. No damage was reported.Tropical Disturbance formed near 5.5N 162.4E on the 16th. It moved west-northwest and passed south of Guam and north of Yap the 19th and 20th. No damage was reported. The tropical depression continued in Guam Area of Responsibility until the 23rd. It eventually became Typhoon Peter while it was west of the Guam Area.A waterspout was observed from the National Weather Service Office at Tiyan northwest of the station. It formed over the water just to the west of the island.",,"PDC"
202102,22,1600,202102,22,1800,157341,951100,"AMERICAN SAMOA",97,2021,"February","Heavy Rain","C",2,"TUTUILA","ASO","22-FEB-21 16:00:00","SST-11","22-FEB-21 18:00:00","0","0","0","0","10.00K","0.00K","Public",,,,,,,,,,,,"8","W","PAGO PAGO","7","W","PAGO PAGO","-14.2861","-170.7165","-14.2844","-170.7108","An active South Pacific Convergence Zone (SPCZ) over the islands developed strong winds and heavy rainfall for the territory.","Scattered landslides reported around the Island. Motorists driving up the Fagasa Pass shared videos and pics of fallen trees and debris from mudslides blocking the road close to 4 pm. Villagers in Alofau also reported a landslide that blocked the west bound lane. Rainfall from the past 24 hours reported at WSO Pago Pago was 5.13 inches. ||Crews from Public Works and ASPA were at the scene clearing the road while cars in both directions waited.","CSV"
197511,26,2230,197511,26,2230,,10010634,"HAWAII",15,1975,"November","Tornado","C",0,,,"26-NOV-75 22:30:00","HST","26-NOV-75 22:30:00","0","0","0","0","0K","0",,"0",,,,"F0","1.5","60",,,,,"0",,,"0",,,"21.25","-158.03",,,,,"PUB"
202103,13,525,202103,13,738,155613,941899,"HAWAII",15,2021,"March","Flash Flood","C",9,"MAUI","HFO","13-MAR-21 05:25:00","HST-10","13-MAR-21 07:38:00","0","0","0","0","0.00K","0.00K","Emergency Manager",,,"Heavy Rain",,,,,,,,,"3","ENE","KAWELA","4","NW","KAMALO","21.0967","-156.9244","21.0956","-156.9153","Conditions remained unstable across most of the Aloha State with features aloft and at the surface west of the islands.  Low level moisture continued to pool over the area, and the result was more heavy precipitation and periods of flash flooding.  The costs of damages were hard to separate during the extended heavy rain event and were included in the previous episode.  No significant injuries were reported.","Peak of flood wave occurred at Kawela Gulch on the island of Molokai.","CSV"
200611,26,735,200611,26,1035,969,4134,"ALASKA",2,2006,"November","High Wind","Z",213,"ST LAWRENCE IS. BERING STRAIT","AFG","26-NOV-06 07:35:00","AKST-9","26-NOV-06 10:35:00","0","0","0","0","0.00K","0.00K","AWOS","62.00","MG",,,,,,,,,,,,,,,,,,,,"On the night of the 25th, a 970 mb low moved norht of Adak on the Aleutian chain, then curved norhtwest across Saint Matthew Island on the maorning of the 26th reacing the southern Gulf of Anadyr on the afternoon of the 26th, then moving inland over Russia Far East and weakening. With persistant strong high pressure of 1052 mb over interior Alaska, the combination of these features produced a variety of winter weather over Western Alaska.|Blizzard Conditions were reported at:|Zone 211 - Nome|Zone 213 - Gambell|Zone 214 - Saint Marys visibility reached 1/2 mile, with lower visibilities likely in other parts of the zone.|Heavy Snow reported at: |Zone 211 - Nome, by the NWS WSO. Storm Total of 7.5 inches. Snow began 9am AST on the 26th, with the Warning Criteria of 6 inches reached 9am AST on the 27th.|Zone 216 - Kaltag Co-op Observer reported 12 inches for a Storm Total. snow began around midnight AST on the 26th; and Warning Criteria (8 inches) likely reached by midnight AST on the 27th.  |High Winds Reported at:|Zone 207 - Point Hope AWOS, highest gust 53 knots (61 mph).|Zone 213 - Gambell AWOS, highest gust 62 knots (71 mph).","","CSV"
195404,30,1200,195404,30,1200,,9984138,"ARKANSAS",5,1954,"April","Tornado","C",57,"HEMPSTEAD",,"30-APR-54 12:00:00","CST","30-APR-54 12:00:00","0","0","0","0","250K","0",,"0",,,,"F3","51.6","1760",,,,,"0",,,"0",,,"33.93","-93.82","33.48","-93.10",,,"PUB"
202103,10,400,202103,10,1200,156919,948561,"CALIFORNIA",6,2021,"March","Debris Flow","C",59,"ORANGE","SGX","10-MAR-21 04:00:00","PST-8","10-MAR-21 12:00:00","0","0","0","0","50.00K","0.00K","County Official",,,"Heavy Rain / Burn Area",,,,,,,,,"1","W","SILVERADO","0","SE","SILVERADO","33.749","-117.6428","33.7469","-117.6275","A West Coast trough of low pressure impacted Southern California March 9th through 13th. This storm brought periods of windy conditions across the mountains and deserts, along with widespread rain and snowfall, and areas of flooding and debris flows being observed as well.","Heavy rain of near 0.20 inches in 15 minutes occurred between 6 and 7 am over Bond fire scar near the Silverado Canyon and this prompted evacuation orders of residents. The debris flow brought areas of mud and debris over roads and into homes. This caused damage to 6 homes and 8 vehicles in the town of Silverado. This also closed Silverado Canyon Road between Olive Drive and Ladd Canyon Drive.","CSV"
195106,27,2204,195106,27,2204,,10104935,"PENNSYLVANIA",42,1951,"June","Tornado","C",5,"ARMSTRONG",,"27-JUN-51 22:04:00","CST","27-JUN-51 22:04:00","0","0","0","0","2.5K","0",,"0",,,,"F2","19.7","33",,,,,"0",,,"0",,,"40.58","-79.25","40.87","-79.18",,,"PUB"
202103,19,2100,202103,20,2000,157357,951222,"IDAHO",16,2021,"March","Heavy Snow","Z",53,"UPPER SNAKE RIVER PLAIN","PIH","19-MAR-21 21:00:00","MST-7","20-MAR-21 20:00:00","0","0","0","0","5.00K","0.00K","COOP Observer",,,,,,,,,,,,,,,,,,,,,,"A Pacific system brought about 6 to 10 inches of snow to parts of the Upper Snake River Plain and Upper Snake River Highlands.  Fall River Electric reported thousands of power outages on March 20th.","Six inches of snow fell in Sugar City and 6 inches fell in Saint Anthony.  2 inches fell in Idaho Falls.   The very heavy and wet snow caused power outages to nearly 2,000 customers in the Idaho Falls area on March 20th.","CSV"
195004,28,1445,195004,28,1445,,10096222,"OKLAHOMA",40,1950,"April","Tornado","C",149,"WASHITA",,"28-APR-50 14:45:00","CST","28-APR-50 14:45:00","0","0","0","0","250K","0",,"0",,,,"F3","3.4","400",,,,,"0",,,"0",,,"35.12","-99.20","35.17","-99.20",,,"PUB"
202103,17,545,202103,17,545,155786,939776,"TEXAS",48,2021,"March","Thunderstorm Wind","C",159,"FRANKLIN","SHV","17-MAR-21 05:45:00","CST-6","17-MAR-21 05:45:00","0","0","0","0","0.00K","0.00K","Law Enforcement","61.00","EG",,,,,,,,,,"2","SE","PURLEY","2","SE","PURLEY","33.0765","-95.2241","33.0765","-95.2241","A strong closed upper low pressure system ejected northeast through the Oklahoma/Texas Panhandle region during the afternoon and evening hours of March 16th, which helped mix a dry line east across Oklahoma and Texas. Ahead of this dry line, a broad warm and moist sector spread north across the Ark-La-Tex, which allowed for greater instability to overspread into the area ahead of this surface feature. A complex of showers and thunderstorms developed over West Texas during the late afternoon through the evening hours on the 16th, before advancing east across the state during the overnight and early morning hours of the 17th. Some of these storms remained severe as they moved through East Texas around and shortly after daybreak, where damaging winds downed trees and power lines before gradually weakening as they moved into Western Louisiana.","Trees were blown down just east of Highway 37 along FM 900 near Lake Cypress Springs.","CSV"
195002,12,610,195002,12,610,,10120408,"TEXAS",48,1950,"February","Tornado","C",293,"LIMESTONE",,"12-FEB-50 06:10:00","CST","12-FEB-50 06:10:00","0","0","0","0","25K","0",,"0",,,,"F2","3.4","100",,,,,"0",,,"0",,,"31.52","-96.55","31.57","-96.55",,,"PUB"
202103,31,1142,202103,31,1600,155999,951413,"GEORGIA",13,2021,"March","Flood","C",223,"PAULDING","FFC","31-MAR-21 11:42:00","EST-5","31-MAR-21 16:00:00","0","0","0","0","0.00K","0.00K","Emergency Manager",,,"Heavy Rain",,,,,,,,,"2","N","HIRAM","1","N","HIRAM","33.8929","-84.7477","33.8874","-84.7499","Strong thunderstorms ahead of a cold front produced numerous reports of flooding across portions of north Georgia and scattered reports of damaging winds across parts of north and central Georgia. Rainfall was highest over far northwest Georgia where totals ranged from 2 to 3.5 inches with locally higher amounts.","The Emergency Manager reported standing water near the intersection of Jimmy Lee Smith Parkway and Highway 92. Radar estimates indicate that 3 to 5 inches of rain occurred over the area, causing the flooding.","CSV"
199309,20,2200,199309,20,2200,,10316535,"ALASKA",2,1993,"September","Thunderstorm Wind","C",15,"COOK INLET AND COPPER RIVER BASIN",,"20-SEP-93 22:00:00","AST","20-SEP-93 22:00:00","0","0","0","0","500K","0",,"0",,,,,"0","0",,,,,"0",,"Delta","0",,"Southcentral",,,,,,"An active frontal system associated with a very strong upper level trough moved across south-central Alaska during the late evening.  Thunderstorms, hail and strong winds were reported over Cook Inlet, the Susitna Valley, the Kenai Peninsula and Prince William Sound.  In the Anchorage/Eagle River area hail stones from 0.13 to 0.50 inches were reported.  Winds to 55 knots accompanied the hail and thunderstorms.  The storm blew down shallow rooted trees, did damage to fences, signs and roofs as well as causing power outages to more than 6,000 homes.  Two miles west of Cordova At 2230 AST 60 knot winds grounded the state ferry ""Bartlett"" carrying 33 passengers and 23 crew members. No one on board was injured.  Strong winds blew an airplane that was tied up at Eyak Lake spit into the lake and was lost.  As the front moved into the Copper River Basin winds gusted to 77 knots in exposed areas.  Gulkana recorded a peak gust of 46 knots.","CSV"
202103,20,1630,202103,20,1800,157132,950891,"PUERTO RICO",99,2021,"March","Flash Flood","C",33,"CATANO","SJU","20-MAR-21 16:30:00","AST-4","20-MAR-21 18:00:00","0","0","0","0","0.00K","0.00K","Emergency Manager",,,"Heavy Rain",,,,,,,,,"1","WNW","CATANO","1","WNW","CATANO","18.4388","-66.1394","18.4388","-66.1387","A surge in low-level moisture combined with daytime heating and local effects to generate afternoon convection across the northwestern quadrant of PR. In addition, a persistent streamer developed from Trujillo Alto to Toa Baja, generating significant rainfall activity, especially across Toa Baja and Catao, where estimates were in excess of 3-5 inches.","Emergency manager called to report a rescue of three people due to heavy rainfall and the Rio Cucharillas out of its banks in sector Reparto Paraiso.","CSV"
200610,28,1315,200610,28,1345,1842,9089,"GUAM",98,2006,"October","Rip Current","Z",6,"GUAM","GUM","28-OCT-06 13:15:00","GST10","28-OCT-06 13:45:00","4","0","0","0","0.00K","0.00K","Newspaper",,,,,,,,,,,,,,,,,,,,,,"Four Japanese tourists, two men and two women, were swept over the reef by strong rip currents at Ritidian Point on Guam's northern tip. They made it back to shore, and were then taken by ambulance to Guam Memorial Hospital for treatment of numerous cuts and bruises.","Four Japanese tourists, two men and two women, were swept over the reef at Ritidian Point on Guam's northern tip. The made it back to shore, and were then taken by ambulance to Guam Memorial Hospital for treatment of numerous cuts and bruises.","CSV"
198607,21,1345,198607,21,1345,,9989076,"CALIFORNIA",6,1986,"July","Tornado","C",71,"SAN BERNARDINO",,"21-JUL-86 13:45:00","PDT","21-JUL-86 13:45:00","0","0","0","0","0K","0",,"0",,,,"F0",".2","10",,,,,"0",,,"0",,,"34.72","-117.03",,,,,"PUB"
199407,10,1900,199407,10,1900,,10346643,"SOUTH DAKOTA",46,1994,"July","Hail","C",63,"HARDING",,"10-JUL-94 19:00:00","MDT","10-JUL-94 19:00:00","0","0","0","0","50K","0",,"1.50",,,,,"0","0",,,,,"0",,,"0",,,,,,,,"Widespread hail accumulated on roads, broke windows in several homes, and damaged some vehicles.","CSV"
199505,16,1926,199505,16,1926,,10326592,"KANSAS",20,1995,"May","Hail","C",83,"HODGEMAN",,"16-MAY-95 19:26:00","CDT","16-MAY-95 19:26:00","0","0","0","0","0","0",,"1.00",,,,,"0","0",,,,,"6","E","Kalvesta","0",,,,,,,,"","CSV"
199506,10,1525,199506,10,1525,,10324635,"KENTUCKY",21,1995,"June","Hail","C",67,"FAYETTE",,"10-JUN-95 15:25:00","EDT","10-JUN-95 15:25:00","0","0","0","0","0","0",,"1.00",,,,,"0","0",,,,,"0",,"Lexington","0",,,,,,,,"Quarter-size hail reported by observer at the National Weather Service.","CSV"
195606,1,1133,195606,1,1133,,10039215,"MASSACHUSETTS",25,1956,"June","Tornado","C",13,"HAMPDEN",,"01-JUN-56 11:33:00","UNK","01-JUN-56 11:33:00","0","0","0","0","250K","0",,"0",,,,"F1","1","67",,,,,"0",,,"0",,,"42.10","-72.70",,,,,"PUB"
199302,21,1310,199302,21,1310,,10325335,"KENTUCKY",21,1993,"February","Hail","C",33,"CALDWELL",,"21-FEB-93 13:10:00","CSt","21-FEB-93 13:10:00","0","0","0","0","0","0",,".75",,,,,"0","0",,,,,"1","N","Princeton","0",,,,,,,,"","CSV"
199407,19,1920,199407,19,1920,,10321512,"IOWA",19,1994,"July","Tornado","C",37,"CHICKASAW",,"19-JUL-94 19:20:00","CSC","19-JUL-94 19:20:00","0","0","0","0",".05K",".05K",,"0",,,,"F0",".2","30",,,,,"3","NE","Alta Vista","0",,,"43.22","-92.40",,,,"","CSV"
199506,30,1950,199506,30,1950,,10320967,"GEORGIA",13,1995,"June","Thunderstorm Wind","C",223,"PAULDING",,"30-JUN-95 19:50:00","ESt","30-JUN-95 19:50:00","0","0","0","0","1K","0",,"0",,,,,"0","0",,,,,"0",,"Dallas","0",,,,,,,,"Thunderstorm winds blew trees down on ground wire.","CSV"
199507,31,1548,199507,31,1548,,10358055,"WISCONSIN",55,1995,"July","Thunderstorm Wind","C",115,"SHAWANO",,"31-JUL-95 15:48:00","SCT","31-JUL-95 15:48:00","0","0","0","0","0","0",,"0",,,,,"0","0",,,,,"3","N","Embarrass","0",,,,,,,,"","CSV"
199507,25,1810,199507,25,1810,,10325479,"KANSAS",20,1995,"July","Hail","C",167,"RUSSELL",,"25-JUL-95 18:10:00","CSt","25-JUL-95 18:10:00","0","0","0","0","0","0",,"1.75",,,,,"0","0",,,,,"10","W","Wilson","0",,,,,,,,"","CSV"
"""  # noqa: E501


@pytest.fixture
def fake_csv_files(tmp_path):
    """Create fake CSV files and return their paths."""
    d = tmp_path / 'csvfiles'
    fns = []
    for year in (1980, 2000, 2020):
        fn = d / f'StormEvents_details-ftp_v1.0_d{year:d}_c20210604.csv.gz'
        fn.parent.mkdir(exist_ok=True, parents=True)
        with gzip.GzipFile(fn, 'w') as fp:
            fp.write(csv_fake_content.replace('2020', f'{year:d}', 1).encode('ascii'))
        fns.append(fn)
    return fns


def test_infer_uri(fake_csv_files):
    """Test inferring the URI for a certain year/timestamp."""
    from metpy.io.storms import get_noaa_storm_uri
    fakedir = os.fspath(fake_csv_files[0].parent)
    with unittest.mock.patch(
            'fsspec.implementations.ftp.FTPFileSystem',
            new=fsspec.implementations.local.LocalFileSystem):
        for year in (1980, 2000, 2020):
            storm_uri = get_noaa_storm_uri(year, path=fakedir)
            assert f'_d{year:>d}_' in storm_uri
            assert 'details' in storm_uri
            assert storm_uri.startswith('ftp://')
        with pytest.raises(FileNotFoundError):
            get_noaa_storm_uri(1854, path=fakedir)
        fake_csv_files[-1].with_name(
            'StormEvents_details-ftp_v1.0_d2020_c20210709.csv.gz').touch()
        with pytest.raises(ValueError):
            get_noaa_storm_uri(2020, path=fakedir)


def test_get_noaa_storms_from_uri(fake_csv_files):
    """Test getting NOAA storms db from URI."""
    import pint_pandas
    from metpy.io.storms import get_noaa_storms_from_uri
    db = get_noaa_storms_from_uri(
        fake_csv_files[-1],
        parse_dates=True,
        parse_windspeed=True,
        parse_hailsize=True)
    assert db.shape == (28, 50)
    assert db.dtypes.start_datetime == pandas.DatetimeTZDtype('ns', 'UTC')
    numpy.testing.assert_array_equal(
        db.start_datetime.dt.year,
        [1950, 2021, 1997, 2021, 1975, 2021, 2006, 1954, 2021, 1951, 2021,
            1950, 2021, 1950, 2021, 1993, 2021, 2006, 1986, 1994, 1995,
            1995, numpy.nan, 1993, 1994, 1995, 1995, 1995])
    numpy.testing.assert_array_equal(
        db.start_datetime.dt.hour,
        [20, 4, 17, 3, 8, 15, 16, 18, 12, 4, 4, 20, 11, 12, 16, 2,
            20, 3, 20, 1, 0, 19, numpy.nan, 19, 1, 0, 21, 0])
    assert db.dtypes.wind_speed.units == pint_pandas.PintType.ureg.knot
    assert db.dtypes.hail_size.units == pint_pandas.PintType.ureg.centiinch
    db = get_noaa_storms_from_uri(
        fake_csv_files[-1],
        parse_dates=False,
        parse_windspeed=False,
        parse_hailsize=False)
    assert 'start_datetime' not in db.columns
    assert 'wind_speed' not in db.columns
    assert 'hail_size' not in db.columns


def test_noaa_get_storms_for_period(fake_csv_files):
    """Test reading noaa storms."""
    from metpy.io.storms import get_noaa_storms_for_period

    def fake_storm_uri(year, server=None, path=None):
        if year == 1980:
            return fake_csv_files[0]
        elif year == 2000:
            return fake_csv_files[1]
        elif year == 2021:
            return fake_csv_files[2]
        else:
            raise FileNotFoundError(f'No storm database found for {year:d}')
    with unittest.mock.patch('metpy.io.storms.get_noaa_storm_uri',
                             new=fake_storm_uri):
        db = get_noaa_storms_for_period(
            datetime.datetime(2021, 3, 1),
            datetime.datetime(2021, 4, 1))
        assert db.shape == (7, 50)
        db = get_noaa_storms_for_period(
            datetime.datetime(1970, 1, 1),
            datetime.datetime(2020, 12, 31))
        assert db.shape == (28, 50)
        with pytest.raises(ValueError):
            db = get_noaa_storms_for_period(
                datetime.datetime(1900, 1, 1),
                datetime.datetime(1900, 1, 1))
