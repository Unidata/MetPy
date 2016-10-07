import gzip
import logging
import math

from datetime import datetime
from metpy.cbook import get_test_data
from metpy.io.text import TextProductFile
from metpy.io.metar import MetarParser, MetarProduct, Weather, log
from metpy.units import units

import pytest

log.setLevel(logging.ERROR)

metar_tests = [('K0CO 042354Z AUTO 01007KT M1/4SM -SN OVC001 M07/M13 A2992 RMK\n\n     AO2',
                [('stid', 'K0CO'),
                 ('present_wx', [Weather.fillin(precip='SN', mod='-')])]),
               ('METAR EETN 042350Z 30005KT CAVOK 07/05 Q1019 R08/95',
                [('visibility', 'CAVOK'),
                 ('runway_state', dict(braking='good', runway='08', depth=None, extent=None,
                                       deposit=None, cleared=False))]),
               ('PAKU 042345Z 16012KT 10SM R06/P6000FT BR SCT065 BKN180 M06/M09\n\n'
                '     A2966 RMK', [('runway_range', {'06': 6000 * units.feet}),
                                   ('temperature', (-6 * units.degC, -9 * units.degC)),
                                   ('present_wx', [Weather.fillin(obscur='BR')])]),
               ('KLZU 042345Z 08006KT 3/4SM BR OVC002 17/13 A3025 RMK '
                'ATIS D; RWY\n\n   25 ILS; BK',
                [('visibility', 0.75 * units.mile), ('pilot', 'ATIS D; RWY 25 ILS; BK')]),
               ('METAR KQES 042350Z AUTO 1600 HZ CLR 12/9 A3037 RMK A02 TSNO',
                [('visibility', 1600 * units.meter), ('automated', 'A02'),
                 ('sky_coverage', ['clear']),
                 ('temperature', (12 * units.degC, 9 * units.degC)),
                 ('present_wx', [Weather.fillin(obscur='HZ')])]),
               ('KSSC 042258Z 02005KT 4SM BR OVC004 18/17 A3021 RMK AO2A PK WND 27018/2242 '
                'TWR VIS 3 VIS N 2 1/2 SLP231 T01840173 $',
                [('stid', 'KSSC'), ('datetime', datetime(2015, 11, 4, 22, 58)),
                 ('auto', False), ('wind', dict(direction=20 * units.deg, speed=5 * units.kt,
                                                gust=None, dir1=None, dir2=None)),
                 ('visibility', 4 * units.mile),
                 ('present_wx', [Weather.fillin(obscur='BR')]),
                 ('sky_coverage', [(400 * units.feet, 8, None)]),
                 ('temperature', (18 * units.degC, 17 * units.degC)),
                 ('altimeter', 30.21 * units.inHg), ('automated', 'AO2A'),
                 ('peak_wind', dict(direction=270 * units.deg, speed=18 * units.knot,
                                    time=datetime(2015, 11, 4, 22, 42))),
                 ('sfc_vis', dict(tower=3 * units.mile)), ('needs_maintenance', True),
                 ('sector_vis', dict(N=2.5 * units.mile)),
                 ('sea_level_pressure', 1023.1 * units.mbar),
                 ('hourly_temperature', (184 * (0.1 * units.degC), 17.3 * units.degC))]),
               ('KSEM 042345Z AUTO 07004KT 10SM FEW049 24/20 A3012 RMK AO2 PRESFR TSNO',
                [('auto', True), ('non-operational sensors', ['Lightning Detection System']),
                 ('pressure_change', 'falling rapidly')]),
               ('METAR SVGI 021500Z /////KT 9999 FEW010 SCT070 30/22',
                [('visibility', 9999 * units.meter),
                 ('sky_coverage', [(1000 * units.feet, 2, None),
                                   (7000 * units.feet, 4, None)]),
                 ('wind', lambda w: math.isnan(w['direction'].magnitude)),
                 ('wind', lambda w: math.isnan(w['speed'].magnitude))]),
               ('KEHA 042349Z AUTO 14016KT 18/09 RMK AO1 T01780089',
                [('temperature', (18 * units.degC, 9 * units.degC))]),
               ('KSEE 042347Z 28010 10SM BKN045 17/07 A2999 RMK CIG 007V012 BKN V OVC',
                [('wind', dict(direction=280 * units.deg, speed=10 * units.kt,
                               gust=None, dir1=None, dir2=None)),
                 ('variable_ceiling', (700 * units.feet, 1200 * units.feet)),
                 ('variable_sky_cover', dict(height=None, cover=('BKN', 'OVC')))]),
               ('KOMN 042350Z 00000KT 10SM BKN110 MM/MM A3013 RMK LAST',
                [('temperature', lambda t: math.isnan(t[0].magnitude)),
                 ('temperature', lambda t: math.isnan(t[1].magnitude)),
                 ('report_sequence', 'LAST')]),
               ('KCPR 042348Z 35007KT 2SM -SN BR FEW007 OVC012 01/M01 A2985 RMK AO2 '
                'RAB10E26UPB37E44SNB08E10B26E37B44 SLP113 P0001 60001 T00061006 '
                '10050 20006 53019',
                [('present_wx', [Weather.fillin(mod='-', precip='SN'),
                                 Weather.fillin(obscur='BR')]),
                 ('precip_times',
                  [('RA', [(datetime(2015, 11, 4, 23, 10), datetime(2015, 11, 4, 23, 26))]),
                   ('UP', [(datetime(2015, 11, 4, 23, 37), datetime(2015, 11, 4, 23, 44))]),
                   ('SN', [(datetime(2015, 11, 4, 23, 8), datetime(2015, 11, 4, 23, 10)),
                           (datetime(2015, 11, 4, 23, 26), datetime(2015, 11, 4, 23, 37)),
                           (datetime(2015, 11, 4, 23, 44), None)])]),
                 ('max_temp_6hr', 5 * units.degC), ('min_temp_6hr', 6 * (0.1 * units.degC)),
                 ('hourly_precip', 0.01 * units.inch), ('period_precip', 0.01 * units.inch),
                 ('pressure_tendency_3hr', (3, 19 * (0.1 * units.mbar)))]),
               ('PTRO 042350Z 11005KT 14SM VCTS SCT016TCU BKN120 BKN300 30/26 A2985 RMK '
                'TCU VC ALQDS LTG DSNT SE THRU SW',
                [('visibility', 14 * units.mile),
                 ('sky_coverage', [(1600 * units.feet, 4, 'TCU'),
                                   (12000 * units.feet, 6, None),
                                   (30000 * units.feet, 6, None)]),
                 ('significant_clouds', 'TCU VC ALQDS'),
                 ('lightning', dict(dist='DSNT', loc='SE THRU SW')),
                 ('present_wx', [Weather.fillin(mod='VC', desc='TS')])]),
               ('METAR OJAM 050000Z 04004KT 4000 DU NSC 13/08 Q1013 A2991 R36/190095 NOSIG',
                [('sky_coverage', ['clear']), ('present_wx', [Weather.fillin(obscur='DU')]),
                 ('altimeter', 29.91 * units.inHg), ('trend_forecast', 'NOSIG'),
                 ('visibility', 4000 * units.meter),
                 ('runway_state', dict(deposit='Damp', extent=1.0, braking='good',
                                       depth=0 * units.mm, runway='36', cleared=False))]),
               ('METAR UBBL 050000Z 00000KT 7000 NSC 06/05 Q1023 R33/CLRD70 RMK',
                [('temperature', (6 * units.degC, 5 * units.degC)),
                 ('sky_coverage', ['clear']),
                 ('runway_state', dict(deposit=None, extent=None, braking=0.7, depth=None,
                                       runway='33', cleared=True))]),
               ('METAR LLBG 042350Z 04004KT 5000 DU VV013 22/11 Q1010 TEMPO SHRA FEW050TCU',
                [('temperature', (22 * units.degC, 11 * units.degC)),
                 ('visibility', 5000 * units.meter),
                 ('sky_coverage', [(1300 * units.feet, 8, None)]),
                 ('present_wx', [Weather.fillin(obscur='DU')])]),
               ('KABQ 042352Z 29013KT 10SM FEW035 BKN065 BKN090 10/02 A2991 RMK AO2 '
                'PK WND 20029/2253 WSHFT 2318 RAE08 SLPNO VIRGA S MTNS OBSC NE-SE '
                'P0001 60014 T01000022 10144 20089 53014',
                [('peak_wind', dict(direction=200 * units.deg, speed=29 * units.knot,
                                    time=datetime(2015, 11, 4, 22, 53))),
                 ('sea_level_pressure', lambda s: math.isnan(s.magnitude)),
                 ('pressure_tendency_3hr', (3, 14 * (0.1 * units.mbar))),
                 ('hourly_precip', 0.01 * units.inch), ('period_precip', 0.14 * units.inch),
                 ('virga', 'S'), ('mountains', 'MTNS OBSC NE-SE'),
                 ('wind_shift', dict(time=datetime(2015, 11, 4, 23, 18), frontal=False))]),
               ('METAR PTKK 052350Z 02007KT 15SM FEW012 SCT120 BKN280 30/27 A2986 RMK SLP111 '
                '60008 8/171 T03000270 10300 20256 50002',
                [('cloud_types', dict(low=1, middle=7, high=1)),
                 ('visibility', 15 * units.mile)]),
               ('CPBT RMK NIL', [('null', True)]),
               ('PAAP 042350Z VRB02KT 10SM BKN030 BKN050 07/05 A2991 RMK 70109 400940056 '
                'NO SPECI',
                [('daily_precip', 1.09 * units.inch), ('no_speci', 'NO SPECI'),
                 ('daily_temperature', (56 * (0.1 * units.degC), 9.4 * units.degC))]),
               ('KPRB 042353Z AUTO 29004KT 10SM CLR 17/01 A3001 RMK AO2 SLP160 6//// '
                'T01670006 10178 20106 53004 TSNO',
                [('period_precip', lambda p: math.isnan(p.magnitude))]),
               ('KSMO 042351Z COR 24008KT 10SM SCT085 18/06 A2996 RMK AO2 SLP143 T01830056 '
                '10194 20172 53005', [('corrected', True)]),
               ('PABT 050028Z 14007KT 3/4SM -FZRASN FZFG OVC015 M02/M03 A2971 RMK '
                'FZRAB08PLB02E08',
                [('present_wx', [Weather.fillin(mod='-', desc='FZ', precip='RASN'),
                                 Weather.fillin(desc='FZ', obscur='FG')]),
                 ('precip_times',
                  [('FZRA', [(datetime(2015, 11, 5, 0, 8), None)]),
                   ('PL', [(datetime(2015, 11, 5, 0, 2), datetime(2015, 11, 5, 0, 8))])])]),
               ('SPECI KSSC 050003Z 04007KT1 1/4SM BR R04/5500FT SCT002 OVC003 18/17 A3023 ',
                [('kind', 'SPECI'), ('unparsed', 'BR')]),
               ('KTDR 042356Z AUTO 25002KT 9SM CLR 25/24 A3011 RMK AO2 LTG DSNT NE SLP200 '
                'T02490243 10295 20249 53007 TSNO $',
                [('sea_level_pressure', 1020 * units.mbar),
                 ('lightning', dict(dist='DSNT', loc='NE'))]),
               ('KLAS 050005Z 08008KT 10SM -TSRA FEW025 BKN055CB BKN075 12/05 A2988 RMK AO2 '
                'TSB04 OCNL LTGICCG N TS N MOV SE P0000 T01170050 $',
                [('lightning', dict(frequency='OCNL', type=['IC', 'CG'], loc='N')),
                 ('thunderstorm', dict(loc='N', mov='SE'))]),
               ('PABT 042353Z 14008KT 1SM -SN BR FEW010 OVC021 M01/M03 A2971 RMK AO2 SNB21 '
                'SLP076 60009 931013 4/012 T10111028 11011 21033 55004',
                [('snow_depth', 12 * units.inch), ('snow_6hr', 1.3 * units.inch)]),
               ('PAEN 042353Z 04010KT 10SM BKN120 M02/M03 A2990 RMK AO2 SLP129 I1000 I6005 '
                'T10171028 11017 21072 58010',
                [('hourly_ice', 0. * units.inch), ('ice_6hr', 0.05 * units.inch)]),
               ('KEHY 050115Z AUTO 28011KT 10SM TS BKN055 OVC070 02/M01 A2979 RMK AO2',
                [('present_wx', [Weather.fillin(desc='TS')]),
                 ('visibility', 10 * units.mile)]),
               ('KRPD 050110Z AUTO 00000KT 1/4SM FG VV002 10/09 A2986 RMK AO2 '
                'VIS M1/4V2 1/2 T01000094',
                [('variable_vis', (0.25 * units.mile, 2.5 * units.mile)),
                 ('visibility', 0.25 * units.mile)]),
               ('SPECI KCBM 050128Z AUTO 00000KT 1/4SM R13C/0800FT FG CLR 19/18 A3013 '
                'RMK AO2 VIS 1/4V3/4 PNO $',
                [('variable_vis', (0.25 * units.mile, 0.75 * units.mile))]),
               ('SPECI CYAM 050150Z 14004KT 120V180 3/8SM R12/6000FT/D BR BCFG BKN001 '
                '10/09 A3002 RMK SLP170',
                [('visibility', 0.375 * units.mile),
                 ('runway_range', {'12': (6000 * units.feet, 'down')})]),
               ('PATA 051752Z 08009KT 10SM FEW050 SCT090 BKN200 M03/M06 A2944 '
                'RMK AO2 SLP974 4/008 933012 T10331061 11011 21039 56024',
                [('snow_liquid_equivalent', 12 * (0.1 * units.inch))]),
               ('METAR CWUW 292300Z AUTO 14011KT ////SM //// OVC002 M01/M01 A2956',
                [('visibility', lambda v: math.isnan(v.magnitude)),
                 ('temperature', (-1 * units.degC, -1 * units.degC)),
                 ('wind', dict(direction=140 * units.deg, speed=11 * units.kt,
                               gust=None, dir1=None, dir2=None))]),
               ('SPECI PAOM 292321Z 16010KT 2 1/2SM R28/5000VP6000FT BR OVC002 07/07 '
                'A3040 RMK AO2 T00720072',
                [('visibility', 2.5 * units.mile), ('altimeter', 3040 * 0.01 * units.inHg),
                 ('runway_range', {'28': (5000 * units.feet, 6000 * units.feet)})]),
               ('PAKU 292250Z 22007KT 10SM R24/P6000FT SCT170 BKN200 05/M01 A3037',
                [('runway_range', {'24': (6000 * units.feet)})]),
               ('KEKM 122250Z 31049G25KT 21/2SM -SN SCT019 BKN028 OVC039 M07/M14',
                [('visibility', lambda v:math.isnan(v.magnitude)),
                 ('wind', dict(direction=310 * units.deg, speed=49 * units.kt,
                               gust=25 * units.kt, dir1=None, dir2=None)),
                 ('temperature', (-7 * units.degC, -14 * units.degC))]),
               ('SPECI PADU 131511Z 030/05 9SM -RA SCT014 OVC020 02/01 A2898',
                [('visibility', 9 * units.mile), ('unparsed', '030/05'),
                 ('temperature', (2 * units.degC, 1 * units.degC))])]


@pytest.mark.parametrize('metar, params', metar_tests)
def test_metars(metar, params):
    parser = MetarParser(default_kind='METAR', ref_time=datetime(2015, 11, 1))
    ob = parser.parse(metar)
    fields = [p[0] for p in params]

    if 'unparsed' not in fields:
        assert 'unparsed' not in ob

    assert 'remarks' not in ob

    for name, val in params:
        assert name in ob
        if callable(val):
            assert val(ob[name])
        else:
            assert val == ob[name]


@pytest.fixture()
def metar_product_file():
    gzfile = gzip.open(get_test_data('metar_20151105_0000.txt.gz', as_file_obj=False),
                       'rt', encoding='latin-1')
    return TextProductFile(gzfile)


def test_ob_count(metar_product_file):
    prod = MetarProduct(next(iter(metar_product_file)))
    assert len(prod.reports) == 1


def test_odd_lines(metar_product_file):
    it = iter(metar_product_file)
    # Need to skip some first
    for i in range(6):
        next(it)
    prod = MetarProduct(next(it))
    assert prod.seq_num == 349
    assert len(prod.reports) == 4
