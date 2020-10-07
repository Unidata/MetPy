#  Copyright (c) 2018,2019 MetPy Developers.
#  Distributed under the terms of the BSD 3-Clause License.
#  SPDX-License-Identifier: BSD-3-Clause
"""Declarative plotting tools."""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from traitlets import (
    Any,
    Bool,
    Float,
    HasTraits,
    Instance,
    Int,
    List,
    Tuple,
    Unicode,
    Union,
    observe,
)

from . import ctables, wx_symbols
from ..calc import reduce_point_density
from ..package_tools import Exporter
from ..units import units
from .cartopy_utils import import_cartopy
from .station_plot import StationPlot

ccrs = import_cartopy()
exporter = Exporter(globals())

_areas = {
    "105": (-129.3, -22.37, 17.52, 53.78),
    "local": (-92.0, -64.0, 28.5, 48.5),
    "wvaac": (120.86, -15.07, -53.6, 89.74),
    "tropsfc": (-100.0, -55.0, 8.0, 33.0),
    "epacsfc": (-155.0, -75.0, -20.0, 33.0),
    "ofagx": (-100.0, -80.0, 20.0, 35.0),
    "ahsf": (-105.0, -30.0, -5.0, 35.0),
    "ehsf": (-145.0, -75.0, -5.0, 35.0),
    "shsf": (-125.0, -75.0, -20.0, 5.0),
    "tropful": (-160.0, 0.0, -20.0, 50.0),
    "tropatl": (-115.0, 10.0, 0.0, 40.0),
    "subtrop": (-90.0, -20.0, 20.0, 60.0),
    "troppac": (-165.0, -80.0, -25.0, 45.0),
    "gulf": (-105.0, -70.0, 10.0, 40.0),
    "carib": (-100.0, -50.0, 0.0, 40.0),
    "sthepac": (-170.0, -70.0, -60.0, 0.0),
    "opcahsf": (-102.0, -20.0, 0.0, 45.0),
    "opcphsf": (175.0, -70.0, -28.0, 45.0),
    "wwe": (-106.0, -50.0, 18.0, 54.0),
    "world": (-24.0, -24.0, -90.0, 90.0),
    "nwwrd1": (-180.0, 180.0, -90.0, 90.0),
    "nwwrd2": (0.0, 0.0, -90.0, 90.0),
    "afna": (-135.02, -23.04, 10.43, 40.31),
    "awna": (-141.03, -18.58, 7.84, 35.62),
    "medr": (-178.0, -25.0, -15.0, 5.0),
    "pacsfc": (129.0, -95.0, -5.0, 18.0),
    "saudi": (4.6, 92.5, -13.2, 60.3),
    "natlmed": (-30.0, 70.0, 0.0, 65.0),
    "ncna": (-135.5, -19.25, 8.0, 37.7),
    "ncna2": (-133.5, -20.5, 10.0, 42.0),
    "hpcsfc": (-124.0, -26.0, 15.0, 53.0),
    "atlhur": (-96.0, -6.0, 4.0, 3.0),
    "nam": (-134.0, 3.0, -4.0, 39.0),
    "sam": (-120.0, -20.0, -60.0, 20.0),
    "samps": (-148.0, -36.0, -28.0, 12.0),
    "eur": (-16.0, 80.0, 24.0, 52.0),
    "afnh": (-155.19, 18.76, -6.8, -3.58),
    "awnh": (-158.94, 15.35, -11.55, -8.98),
    "wwwus": (-127.7, -59.0, 19.8, 56.6),
    "ccfp": (-130.0, -65.0, 22.0, 52.0),
    "llvl": (-119.6, -59.5, 19.9, 44.5),
    "llvl2": (-125.0, -32.5, 5.0, 46.0),
    "llvl_e": (-89.0, -59.5, 23.5, 44.5),
    "llvl_c": (-102.4, -81.25, 23.8, 51.6),
    "llvl_w": (-119.8, -106.5, 19.75, 52.8),
    "ak_artc": (163.7, -65.3, 17.5, 52.6),
    "fxpswna": (-80.5, 135.0, -1.0, 79.0),
    "fxpsnna": (-80.5, 54.0, -1.0, 25.5),
    "fxpsna": (-72.6, 31.4, -3.6, 31.0),
    "natl_ps": (-80.5, 54.0, -1.0, 25.5),
    "fxpsena": (-45.0, 54.0, 11.0, 25.5),
    "fxpsnp": (155.5, -106.5, 22.5, 47.0),
    "npac_ps": (155.5, -106.5, 22.5, 47.0),
    "fxpsus": (-120.0, -59.0, 20.0, 44.5),
    "fxmrwrd": (58.0, 58.0, -70.0, 70.0),
    "fxmrwr2": (-131.0, -131.0, -70.0, 70.0),
    "nwmrwrd": (70.0, 70.0, -70.0, 70.0),
    "wrld_mr": (58.0, 58.0, -70.0, 70.0),
    "fxmr110": (-180.0, -110.0, -20.0, 50.5),
    "fxmr180": (110.0, -180.0, -20.0, 50.5),
    "fxmrswp": (97.5, -147.5, -36.0, 45.5),
    "fxmrus": (-162.5, -37.5, -28.0, 51.2),
    "fxmrea": (-40.0, 20.0, -20.0, 54.2),
    "fxmrjp": (100.0, -160.0, 0.0, 45.0),
    "icao_a": (-137.4, -12.6, -54.0, 67.0),
    "icao_b": (-52.5, -16.0, -62.5, 77.5),
    "icao_b1": (-125.0, 40.0, -45.5, 62.7),
    "icao_c": (-35.0, 70.0, -45.0, 75.0),
    "icao_d": (-15.0, 132.0, -27.0, 63.0),
    "icao_e": (25.0, 180.0, -54.0, 40.0),
    "icao_f": (100.0, -110.0, -52.7, 50.0),
    "icao_g": (34.8, 157.2, -0.8, 13.7),
    "icao_h": (-79.1, 56.7, 1.6, 25.2),
    "icao_i": (166.24, -60.62, -6.74, 33.32),
    "icao_j": (106.8, -101.1, -27.6, 0.8),
    "icao_k": (3.3, 129.1, -11.1, 6.7),
    "icao_m": (100.0, -110.0, -10.0, 70.0),
    "icao_eu": (-21.6, 68.4, 21.4, 58.7),
    "icao_me": (17.0, 70.0, 10.0, 44.0),
    "icao_as": (53.0, 108.0, 00.0, 36.0),
    "icao_na": (-54.1, 60.3, 17.2, 50.7),
    "nhem": (-135.0, 45.0, -15.0, -15.0),
    "nhem_ps": (-135.0, 45.0, -15.0, -15.0),
    "nhem180": (135.0, -45.0, -15.0, -15.0),
    "nhem155": (160.0, -20.0, -15.0, -15.0),
    "nhem165": (150.0, -30.0, -15.0, -15.0),
    "nh45_ps": (-90.0, 90.0, -15.0, -15.0),
    "nhem0": (-45.0, 135.0, -15.0, -15.0),
    "shem_ps": (88.0, -92.0, 30.0, 30.0),
    "hfo_gu": (160.0, -130.0, -30.0, 40.0),
    "natl": (-110.0, 20.1, 15.0, 70.0),
    "watl": (-84.0, -38.0, 25.0, 46.0),
    "tatl": (-90.0, -15.0, -10.0, 35.0),
    "npac": (102.0, -110.0, -12.0, 60.0),
    "spac": (102.0, -70.0, -60.0, 20.0),
    "tpac": (-165.0, -75.0, -10.0, 40.0),
    "epac": (-134.0, -110.0, 12.0, 75.0),
    "wpac": (130.0, -120.0, 0.0, 63.0),
    "mpac": (128.0, -108.0, 15.0, 71.95),
    "opcsfp": (128.89, -105.3, 3.37, 16.77),
    "opcsfa": (-55.5, 75.0, -8.5, 52.6),
    "opchur": (-99.0, -15.0, 1.0, 50.05),
    "us": (-119.0, -56.0, 19.0, 47.0),
    "spcus": (-116.4, -63.9, 22.1, 47.2),
    "afus": (-119.04, -63.44, 23.1, 44.63),
    "ncus": (-124.2, -40.98, 17.89, 47.39),
    "nwus": (-118.0, -55.5, 17.0, 46.5),
    "awips": (-127.0, -59.0, 20.0, 50.0),
    "bwus": (-124.6, -46.7, 13.1, 43.1),
    "usa": (-118.0, -62.0, 22.8, 45.0),
    "usnps": (-118.0, -62.0, 18.0, 51.0),
    "uslcc": (-118.0, -62.0, 20.0, 51.0),
    "uswn": (-129.0, -45.0, 17.0, 53.0),
    "ussf": (-123.5, -44.5, 13.0, 32.1),
    "ussp": (-126.0, -49.0, 13.0, 54.0),
    "whlf": (-123.8, -85.9, 22.9, 50.2),
    "chlf": (-111.0, -79.0, 27.5, 50.5),
    "centus": (-105.4, -77.0, 24.7, 47.6),
    "ehlf": (-96.2, -62.7, 22.0, 49.0),
    "mehlf": (-89.9, -66.6, 23.8, 49.1),
    "bosfa": (-87.5, -63.5, 34.5, 50.5),
    "miafa": (-88.0, -72.0, 23.0, 39.0),
    "chifa": (-108.0, -75.0, 34.0, 50.0),
    "dfwfa": (-106.5, -80.5, 22.0, 40.0),
    "slcfa": (-126.0, -98.0, 29.5, 50.5),
    "sfofa": (-129.0, -111.0, 30.0, 50.0),
    "g8us": (-116.0, -58.0, 19.0, 56.0),
    "wsig": (155.0, -115.0, 18.0, 58.0),
    "esig": (-80.0, -30.0, 25.0, 51.0),
    "eg8": (-79.0, -13.0, 24.0, 52.0),
    "west": (-125.0, -90.0, 25.0, 55.0),
    "cent": (-107.4, -75.3, 24.3, 49.7),
    "east": (-100.55, -65.42, 24.57, 47.2),
    "nwse": (-126.0, -102.0, 38.25, 50.25),
    "swse": (-126.0, -100.0, 28.25, 40.25),
    "ncse": (-108.0, -84.0, 38.25, 50.25),
    "scse": (-108.9, -84.0, 24.0, 40.25),
    "nese": (-89.0, -64.0, 37.25, 47.25),
    "sese": (-90.0, -66.0, 28.25, 40.25),
    "afwh": (170.7, 15.4, -48.6, 69.4),
    "afeh": (-9.3, -164.6, -48.6, 69.4),
    "afpc": (80.7, -74.6, -48.6, 69.4),
    "ak": (-179.0, -116.4, 49.0, 69.0),
    "ak2": (-180.0, -106.0, 42.0, 73.0),
    "nwak": (-180.0, -110.0, 50.0, 60.0),
    "al": (-95.0, -79.0, 27.0, 38.0),
    "ar": (-100.75, -84.75, 29.5, 40.5),
    "ca": (-127.75, -111.75, 31.5, 42.5),
    "co": (-114.0, -98.0, 33.5, 44.5),
    "ct": (-81.25, -65.25, 36.0, 47.0),
    "dc": (-85.0, -69.0, 33.35, 44.35),
    "de": (-83.75, -67.75, 33.25, 44.25),
    "fl": (-90.0, -74.0, 23.0, 34.0),
    "ga": (-92.0, -76.0, 27.5, 38.5),
    "hi": (-161.5, -152.5, 17.0, 23.0),
    "nwxhi": (-166.0, -148.0, 14.0, 26.0),
    "ia": (-102.0, -86.0, 36.5, 47.5),
    "id": (-123.0, -107.0, 39.25, 50.25),
    "il": (-97.75, -81.75, 34.5, 45.5),
    "in": (-94.5, -78.5, 34.5, 45.5),
    "ks": (-106.5, -90.5, 33.25, 44.25),
    "ky": (-93.0, -77.0, 31.75, 42.75),
    "la": (-100.75, -84.75, 25.75, 36.75),
    "ma": (-80.25, -64.25, 36.75, 47.75),
    "md": (-85.25, -69.25, 33.75, 44.75),
    "me": (-77.75, -61.75, 39.5, 50.5),
    "mi": (-93.0, -77.0, 37.75, 48.75),
    "mn": (-102.0, -86.0, 40.5, 51.5),
    "mo": (-101.0, -85.0, 33.0, 44.0),
    "ms": (-98.0, -82.0, 27.0, 38.0),
    "mt": (-117.0, -101.0, 41.5, 52.5),
    "nc": (-87.25, -71.25, 30.0, 41.0),
    "nd": (-107.5, -91.5, 42.25, 53.25),
    "ne": (-107.5, -91.5, 36.25, 47.25),
    "nh": (-79.5, -63.5, 38.25, 49.25),
    "nj": (-82.5, -66.5, 34.75, 45.75),
    "nm": (-114.25, -98.25, 29.0, 40.0),
    "nv": (-125.0, -109.0, 34.0, 45.0),
    "ny": (-84.0, -68.0, 37.25, 48.25),
    "oh": (-91.0, -75.0, 34.5, 45.5),
    "ok": (-105.25, -89.25, 30.25, 41.25),
    "or": (-128.0, -112.0, 38.75, 49.75),
    "pa": (-86.0, -70.0, 35.5, 46.5),
    "ri": (-79.75, -63.75, 36.0, 47.0),
    "sc": (-89.0, -73.0, 28.5, 39.5),
    "sd": (-107.5, -91.5, 39.0, 50.0),
    "tn": (-95.0, -79.0, 30.0, 41.0),
    "tx": (-107.0, -91.0, 25.4, 36.5),
    "ut": (-119.0, -103.0, 34.0, 45.0),
    "va": (-86.5, -70.5, 32.25, 43.25),
    "vt": (-80.75, -64.75, 38.25, 49.25),
    "wi": (-98.0, -82.0, 38.5, 49.5),
    "wv": (-89.0, -73.0, 33.0, 44.0),
    "wy": (-116.0, -100.0, 37.75, 48.75),
    "az": (-119.0, -103.0, 29.0, 40.0),
    "wa": (-128.0, -112.0, 41.75, 52.75),
    "abrfc": (-108.0, -88.0, 30.0, 42.0),
    "ab10": (-106.53, -90.28, 31.69, 40.01),
    "cbrfc": (-117.0, -103.0, 28.0, 46.0),
    "cb10": (-115.69, -104.41, 29.47, 44.71),
    "lmrfc": (-100.0, -77.0, 26.0, 40.0),
    "lm10": (-97.17, -80.07, 28.09, 38.02),
    "marfc": (-83.5, -70.0, 35.5, 44.0),
    "ma10": (-81.27, -72.73, 36.68, 43.1),
    "mbrfc": (-116.0, -86.0, 33.0, 53.0),
    "mb10": (-112.8, -89.33, 35.49, 50.72),
    "ncrfc": (-108.0, -76.0, 34.0, 53.0),
    "nc10": (-104.75, -80.05, 35.88, 50.6),
    "nerfc": (-84.0, -61.0, 39.0, 49.0),
    "ne10": (-80.11, -64.02, 40.95, 47.62),
    "nwrfc": (-128.0, -105.0, 35.0, 55.0),
    "nw10": (-125.85, -109.99, 38.41, 54.46),
    "ohrfc": (-92.0, -75.0, 34.0, 44.0),
    "oh10": (-90.05, -77.32, 35.2, 42.9),
    "serfc": (-94.0, -70.0, 22.0, 40.0),
    "se10": (-90.6, -73.94, 24.12, 37.91),
    "wgrfc": (-112.0, -88.0, 21.0, 42.0),
    "wg10": (-108.82, -92.38, 23.99, 39.18),
    "nwcn": (-133.5, -10.5, 32.0, 56.0),
    "cn": (-120.4, -14.0, 37.9, 58.6),
    "ab": (-119.6, -108.2, 48.6, 60.4),
    "bc": (-134.5, -109.0, 47.2, 60.7),
    "mb": (-102.4, -86.1, 48.3, 60.2),
    "nb": (-75.7, -57.6, 42.7, 49.6),
    "nf": (-68.0, -47.0, 45.0, 62.0),
    "ns": (-67.0, -59.0, 43.0, 47.5),
    "nt": (-131.8, -33.3, 57.3, 67.8),
    "on": (-94.5, -68.2, 41.9, 55.0),
    "pe": (-64.6, -61.7, 45.8, 47.1),
    "qb": (-80.0, -49.2, 44.1, 60.9),
    "sa": (-111.2, -97.8, 48.5, 60.3),
    "yt": (-142.0, -117.0, 59.0, 70.5),
    "ag": (-80.0, -53.0, -56.0, -20.0),
    "ah": (60.0, 77.0, 27.0, 40.0),
    "afrca": (-25.0, 59.4, -36.0, 41.0),
    "ai": (-14.3, -14.1, -8.0, -7.8),
    "alba": (18.0, 23.0, 39.0, 43.0),
    "alge": (-9.0, 12.0, 15.0, 38.0),
    "an": (10.0, 25.0, -20.0, -5.0),
    "antl": (-70.0, -58.0, 11.0, 19.0),
    "antg": (-86.0, -65.0, 17.0, 25.0),
    "atg": (-62.0, -61.6, 16.9, 17.75),
    "au": (101.0, 148.0, -45.0, -6.5),
    "azor": (-27.6, -23.0, 36.0, 41.0),
    "ba": (-80.5, -72.5, 22.5, 28.5),
    "be": (-64.9, -64.5, 32.2, 32.6),
    "bel": (2.5, 6.5, 49.4, 51.6),
    "bf": (113.0, 116.0, 4.0, 5.5),
    "bfa": (-6.0, 3.0, 9.0, 15.1),
    "bh": (-89.3, -88.1, 15.7, 18.5),
    "bi": (29.0, 30.9, -4.6, -2.2),
    "bj": (0.0, 5.0, 6.0, 12.6),
    "bn": (50.0, 51.0, 25.5, 27.1),
    "bo": (-72.0, -50.0, -24.0, -8.0),
    "bots": (19.0, 29.6, -27.0, -17.0),
    "br": (-62.5, -56.5, 12.45, 13.85),
    "bt": (71.25, 72.6, -7.5, -5.0),
    "bu": (22.0, 30.0, 40.0, 45.0),
    "bv": (3.0, 4.0, -55.0, -54.0),
    "bw": (87.0, 93.0, 20.8, 27.0),
    "by": (19.0, 33.0, 51.0, 60.0),
    "bz": (-75.0, -30.0, -35.0, 5.0),
    "cais": (-172.0, -171.0, -3.0, -2.0),
    "nwcar": (-120.0, -50.0, -15.0, 35.0),
    "cari": (-103.0, -53.0, 3.0, 36.0),
    "cb": (13.0, 25.0, 7.0, 24.0),
    "ce": (14.0, 29.0, 2.0, 11.5),
    "cg": (10.0, 20.0, -6.0, 5.0),
    "ch": (-80.0, -66.0, -56.0, -15.0),
    "ci": (85.0, 145.0, 14.0, 48.5),
    "cm": (7.5, 17.1, 1.0, 14.0),
    "colm": (-81.0, -65.0, -5.0, 14.0),
    "cr": (-19.0, -13.0, 27.0, 30.0),
    "cs": (-86.5, -81.5, 8.2, 11.6),
    "cu": (-85.0, -74.0, 19.0, 24.0),
    "cv": (-26.0, -22.0, 14.0, 18.0),
    "cy": (32.0, 35.0, 34.0, 36.0),
    "cz": (8.9, 22.9, 47.4, 52.4),
    "dj": (41.5, 44.1, 10.5, 13.1),
    "dl": (4.8, 16.8, 47.0, 55.0),
    "dn": (8.0, 11.0, 54.0, 58.6),
    "do": (-61.6, -61.2, 15.2, 15.8),
    "dr": (-72.2, -68.0, 17.5, 20.2),
    "eg": (24.0, 37.0, 21.0, 33.0),
    "eq": (-85.0, -74.0, -7.0, 3.0),
    "er": (50.0, 57.0, 22.0, 26.6),
    "es": (-90.3, -87.5, 13.0, 14.6),
    "et": (33.0, 49.0, 2.0, 19.0),
    "fa": (-8.0, -6.0, 61.0, 63.0),
    "fg": (-55.0, -49.0, 1.0, 7.0),
    "fi": (20.9, 35.1, 59.0, 70.6),
    "fj": (176.0, -179.0, 16.0, 19.0),
    "fk": (-61.3, -57.5, -53.0, -51.0),
    "fn": (0.0, 17.0, 11.0, 24.0),
    "fr": (-5.0, 11.0, 41.0, 51.5),
    "gb": (-17.1, -13.5, 13.0, 14.6),
    "gc": (-82.8, -77.6, 17.9, 21.1),
    "gh": (-4.5, 1.5, 4.0, 12.0),
    "gi": (-8.0, -4.0, 35.0, 38.0),
    "gl": (-56.7, 14.0, 58.3, 79.7),
    "glp": (-64.2, -59.8, 14.8, 19.2),
    "gm": (144.5, 145.1, 13.0, 14.0),
    "gn": (2.0, 16.0, 3.5, 15.5),
    "go": (8.0, 14.5, -4.6, 3.0),
    "gr": (20.0, 27.6, 34.0, 42.0),
    "gu": (-95.6, -85.0, 10.5, 21.1),
    "gw": (-17.5, -13.5, 10.8, 12.8),
    "gy": (-62.0, -55.0, 0.0, 10.0),
    "ha": (-75.0, -71.0, 18.0, 20.0),
    "he": (-6.1, -5.5, -16.3, -15.5),
    "hk": (113.5, 114.7, 22.0, 23.0),
    "ho": (-90.0, -83.0, 13.0, 16.6),
    "hu": (16.0, 23.0, 45.5, 49.1),
    "ic": (43.0, 45.0, -13.2, -11.0),
    "icel": (-24.1, -11.5, 63.0, 67.5),
    "ie": (-11.1, -4.5, 50.0, 55.6),
    "inda": (67.0, 92.0, 4.2, 36.0),
    "indo": (95.0, 141.0, -8.0, 6.0),
    "iq": (38.0, 50.0, 29.0, 38.0),
    "ir": (44.0, 65.0, 25.0, 40.0),
    "is": (34.0, 37.0, 29.0, 34.0),
    "iv": (-9.0, -2.0, 4.0, 11.0),
    "iw": (34.8, 35.6, 31.2, 32.6),
    "iy": (6.6, 20.6, 35.6, 47.2),
    "jd": (34.0, 39.6, 29.0, 33.6),
    "jm": (-80.0, -76.0, 16.0, 19.0),
    "jp": (123.0, 155.0, 24.0, 47.0),
    "ka": (131.0, 155.0, 1.0, 9.6),
    "kash": (74.0, 78.0, 32.0, 35.0),
    "kb": (172.0, 177.0, -3.0, 3.2),
    "khm": (102.0, 108.0, 10.0, 15.0),
    "ki": (105.2, 106.2, -11.0, -10.0),
    "kn": (32.5, 42.1, -6.0, 6.0),
    "kna": (-62.9, -62.4, 17.0, 17.5),
    "ko": (124.0, 131.5, 33.0, 43.5),
    "ku": (-168.0, -155.0, -24.1, -6.1),
    "kw": (46.5, 48.5, 28.5, 30.5),
    "laos": (100.0, 108.0, 13.5, 23.1),
    "lb": (34.5, 37.1, 33.0, 35.0),
    "lc": (60.9, 61.3, 13.25, 14.45),
    "li": (-12.0, -7.0, 4.0, 9.0),
    "ln": (-162.1, -154.9, -4.2, 6.0),
    "ls": (27.0, 29.6, -30.6, -28.0),
    "lt": (9.3, 9.9, 47.0, 47.6),
    "lux": (5.6, 6.6, 49.35, 50.25),
    "ly": (8.0, 26.0, 19.0, 35.0),
    "maar": (-63.9, -62.3, 17.0, 18.6),
    "made": (-17.3, -16.5, 32.6, 33.0),
    "mala": (100.0, 119.6, 1.0, 8.0),
    "mali": (-12.5, 6.0, 8.5, 25.5),
    "maur": (57.2, 57.8, -20.7, -19.9),
    "maut": (-17.1, -4.5, 14.5, 28.1),
    "mc": (-13.0, -1.0, 25.0, 36.0),
    "mg": (43.0, 50.6, -25.6, -12.0),
    "mh": (160.0, 172.0, 4.5, 12.1),
    "ml": (14.3, 14.7, 35.8, 36.0),
    "mmr": (92.0, 102.0, 7.5, 28.5),
    "mong": (87.5, 123.1, 38.5, 52.6),
    "mr": (-61.2, -60.8, 14.3, 15.1),
    "mu": (113.0, 114.0, 22.0, 23.0),
    "mv": (70.1, 76.1, -6.0, 10.0),
    "mw": (32.5, 36.1, -17.0, -9.0),
    "mx": (-119.0, -83.0, 13.0, 34.0),
    "my": (142.5, 148.5, 9.0, 25.0),
    "mz": (29.0, 41.0, -26.5, -9.5),
    "nama": (11.0, 25.0, -29.5, -16.5),
    "ncal": (158.0, 172.0, -23.0, -18.0),
    "ng": (130.0, 152.0, -11.0, 0.0),
    "ni": (2.0, 14.6, 3.0, 14.0),
    "nk": (-88.0, -83.0, 10.5, 15.1),
    "nl": (3.5, 7.5, 50.5, 54.1),
    "no": (3.0, 35.0, 57.0, 71.5),
    "np": (80.0, 89.0, 25.0, 31.0),
    "nw": (166.4, 167.4, -1.0, 0.0),
    "nz": (165.0, 179.0, -48.0, -33.0),
    "om": (52.0, 60.0, 16.0, 25.6),
    "os": (9.0, 18.0, 46.0, 50.0),
    "pf": (-154.0, -134.0, -28.0, -8.0),
    "ph": (116.0, 127.0, 4.0, 21.0),
    "pi": (-177.5, -167.5, -9.0, 1.0),
    "pk": (60.0, 78.0, 23.0, 37.0),
    "pl": (14.0, 25.0, 48.5, 55.0),
    "pm": (-83.0, -77.0, 7.0, 10.0),
    "po": (-10.0, -4.0, 36.5, 42.5),
    "pr": (-82.0, -68.0, -20.0, 5.0),
    "pt": (-130.6, -129.6, -25.56, -24.56),
    "pu": (-67.5, -65.5, 17.5, 18.5),
    "py": (-65.0, -54.0, -32.0, -17.0),
    "qg": (7.0, 12.0, -2.0, 3.0),
    "qt": (50.0, 52.0, 24.0, 27.0),
    "ra": (60.0, -165.0, 25.0, 55.0),
    "re": (55.0, 56.0, -21.5, -20.5),
    "riro": (-18.0, -12.0, 17.5, 27.5),
    "ro": (19.0, 31.0, 42.5, 48.5),
    "rw": (29.0, 31.0, -3.0, -1.0),
    "saud": (34.5, 56.1, 15.0, 32.6),
    "sb": (79.0, 83.0, 5.0, 10.0),
    "seyc": (55.0, 56.0, -5.0, -4.0),
    "sg": (-18.0, -10.0, 12.0, 17.0),
    "si": (39.5, 52.1, -4.5, 13.5),
    "sk": (109.5, 119.3, 1.0, 7.0),
    "sl": (-13.6, -10.2, 6.9, 10.1),
    "sm": (-59.0, -53.0, 1.0, 6.0),
    "sn": (10.0, 25.0, 55.0, 69.6),
    "so": (156.0, 167.0, -12.0, -6.0),
    "sp": (-10.0, 6.0, 35.0, 44.0),
    "sr": (103.0, 105.0, 1.0, 2.0),
    "su": (21.5, 38.5, 3.5, 23.5),
    "sv": (30.5, 33.1, -27.5, -25.3),
    "sw": (5.9, 10.5, 45.8, 48.0),
    "sy": (35.0, 42.6, 32.0, 37.6),
    "tanz": (29.0, 40.6, -13.0, 0.0),
    "td": (-62.1, -60.5, 10.0, 11.6),
    "tg": (-0.5, 2.5, 5.0, 12.0),
    "th": (97.0, 106.0, 5.0, 21.0),
    "ti": (-71.6, -70.6, 21.0, 22.0),
    "tk": (-173.0, -171.0, -11.5, -7.5),
    "to": (-178.5, -170.5, -22.0, -15.0),
    "tp": (6.0, 7.6, 0.0, 2.0),
    "ts": (7.0, 13.0, 30.0, 38.0),
    "tu": (25.0, 48.0, 34.1, 42.1),
    "tv": (176.0, 180.0, -11.0, -5.0),
    "tw": (120.0, 122.0, 21.9, 25.3),
    "ug": (29.0, 35.0, -3.5, 5.5),
    "uk": (-11.0, 5.0, 49.0, 60.0),
    "ur": (24.0, 41.0, 44.0, 55.0),
    "uy": (-60.0, -52.0, -35.5, -29.5),
    "vanu": (167.0, 170.0, -21.0, -13.0),
    "vi": (-65.5, -64.0, 16.6, 19.6),
    "vk": (13.8, 25.8, 46.75, 50.75),
    "vn": (-75.0, -60.0, -2.0, 14.0),
    "vs": (102.0, 110.0, 8.0, 24.0),
    "wk": (166.1, 167.1, 18.8, 19.8),
    "ye": (42.5, 54.1, 12.5, 19.1),
    "yg": (13.5, 24.6, 40.0, 47.0),
    "za": (16.0, 34.0, -36.0, -22.0),
    "zb": (21.0, 35.0, -20.0, -7.0),
    "zm": (170.5, 173.5, -15.0, -13.0),
    "zr": (12.0, 31.6, -14.0, 6.0),
    "zw": (25.0, 34.0, -22.9, -15.5),
}


def lookup_projection(projection_code):
    """Get a Cartopy projection based on a short abbreviation."""
    import cartopy.crs as ccrs

    projections = {
        "lcc": ccrs.LambertConformal(
            central_latitude=40, central_longitude=-100, standard_parallels=[30, 60]
        ),
        "ps": ccrs.NorthPolarStereo(central_longitude=-100),
        "mer": ccrs.Mercator(),
    }
    return projections[projection_code]


def lookup_map_feature(feature_name):
    """Get a Cartopy map feature based on a name."""
    import cartopy.feature as cfeature

    from . import cartopy_utils

    name = feature_name.upper()
    try:
        feat = getattr(cfeature, name)
        scaler = cfeature.AdaptiveScaler("110m", (("50m", 50), ("10m", 15)))
    except AttributeError:
        feat = getattr(cartopy_utils, name)
        scaler = cfeature.AdaptiveScaler("20m", (("5m", 5), ("500k", 1)))
    return feat.with_scale(scaler)


class Panel(HasTraits):
    """Draw one or more plots."""


@exporter.export
class PanelContainer(HasTraits):
    """Collects panels and set complete figure related settings (e.g., size)."""

    size = Union([Tuple(Int(), Int()), Instance(type(None))], default_value=None)
    size.__doc__ = """This trait takes a tuple of (width, height) to set the size of the
    figure.

    This trait defaults to None and will assume the default `matplotlib.pyplot.figure` size.
    """

    panels = List(Instance(Panel))
    panels.__doc__ = """A list of panels to plot on the figure.

    This trait must contain at least one panel to plot on the figure."""

    @property
    def panel(self):
        """Provide simple access for a single panel."""
        return self.panels[0]

    @panel.setter
    def panel(self, val):
        self.panels = [val]

    @observe("panels")
    def _panels_changed(self, change):
        for panel in change.new:
            panel.parent = self
            panel.observe(self.refresh, names=("_need_redraw"))

    @property
    def figure(self):
        """Provide access to the underlying figure object."""
        if not hasattr(self, "_fig"):
            self._fig = plt.figure(figsize=self.size)
        return self._fig

    def refresh(self, _):
        """Refresh the rendering of all panels."""
        # First make sure everything is properly constructed
        self.draw()

        # Trigger the graphics refresh
        self.figure.canvas.draw()

        # Flush out interactive events--only ok on Agg for newer matplotlib
        try:
            self.figure.canvas.flush_events()
        except NotImplementedError:
            pass

    def draw(self):
        """Draw the collection of panels."""
        for panel in self.panels:
            with panel.hold_trait_notifications():
                panel.draw()

    def save(self, *args, **kwargs):
        """Save the constructed graphic as an image file.

        This method takes a string for saved file name. Additionally, the same arguments and
        keyword arguments that `matplotlib.pyplot.savefig` does.
        """
        self.draw()
        self.figure.savefig(*args, **kwargs)

    def show(self):
        """Show the constructed graphic on the screen."""
        self.draw()
        plt.show()


@exporter.export
class MapPanel(Panel):
    """Set figure related elements for an individual panel.

    Parameters that need to be set include collecting all plotting types
    (e.g., contours, wind barbs, etc.) that are desired to be in a given panel.
    Additionally, traits can be set to plot map related features (e.g., coastlines, borders),
    projection, graphics area, and title.
    """

    parent = Instance(PanelContainer)

    layout = Tuple(Int(), Int(), Int(), default_value=(1, 1, 1))
    layout.__doc__ = """A tuple that contains the description (nrows, ncols, index) of the
    panel position; default value is (1, 1, 1).

    This trait is set to describe the panel position and the default is for a single panel. For
    example, a four-panel plot will have two rows and two columns with the tuple setting for
    the upper-left panel as (2, 2, 1), upper-right as (2, 2, 2), lower-left as (2, 2, 3), and
    lower-right as (2, 2, 4). For more details see the documentation for
    `matplotlib.figure.Figure.add_subplot`.
    """

    plots = List(Any())
    plots.__doc__ = """A list of handles that represent the plots (e.g., `ContourPlot`,
    `FilledContourPlot`, `ImagePlot`) to put on a given panel.

    This trait collects the different plots, including contours and images, that are intended
    for a given panel.
    """

    _need_redraw = Bool(default_value=True)

    area = Union(
        [Unicode(), Tuple(Float(), Float(), Float(), Float())],
        allow_none=True,
        default_value=None,
    )
    area.__doc__ = """A tuple or string value that indicates the graphical area of the plot.

    The tuple value coresponds to longitude/latitude box based on the projection of the map
    with the format (west-most longitude, east-most longitude, south-most latitude,
    north-most latitude). This tuple defines a box from the lower-left to the upper-right
    corner.

    This trait can also be set with a string value associated with the named geographic regions
    within MetPy. The tuples associated with the names are based on a PlatteCarree projection.
    For a CONUS region, the following strings can be used: 'us', 'spcus', 'ncus', and 'afus'.
    For regional plots, US postal state abbreviations can be used.
    """

    projection = Union([Unicode(), Instance("cartopy.crs.Projection")], default_value="data")
    projection.__doc__ = """A string for a pre-defined projection or a Cartopy projection
    object.

    There are three pre-defined projections that can be called with a short name:
    Lambert conformal conic ('lcc'), Mercator ('mer'), or polar-stereographic ('ps').
    Additionally, this trait can be set to a Cartopy projection object.
    """

    layers = List(
        Union([Unicode(), Instance("cartopy.feature.Feature")]), default_value=["coastline"]
    )
    layers.__doc__ = """A list of strings for a pre-defined feature layer or a Cartopy Feature object.

    Like the projection, there are a couple of pre-defined feature layers that can be called
    using a short name. The pre-defined layers are: 'coastline', 'states', 'borders', 'lakes',
    'land', 'ocean', 'rivers', 'usstates', and 'uscounties'. Additionally, this can accept
    Cartopy Feature objects.
    """

    title = Unicode()
    title.__doc__ = """A string to set a title for the figure.

    This trait sets a user-defined title that will plot at the top center of the figure.
    """

    @observe("plots")
    def _plots_changed(self, change):
        """Handle when our collection of plots changes."""
        for plot in change.new:
            plot.parent = self
            plot.observe(self.refresh, names=("_need_redraw"))
        self._need_redraw = True

    @observe("parent")
    def _parent_changed(self, _):
        """Handle when the parent is changed."""
        self.ax = None

    @property
    def _proj_obj(self):
        """Return the projection as a Cartopy object.

        Handles looking up a string for the projection, or if the projection
        is set to ``'data'`` looks at the data for the projection.

        """
        if isinstance(self.projection, str):
            if self.projection == "data":
                if isinstance(self.plots[0].griddata, tuple):
                    return self.plots[0].griddata[0].metpy.cartopy_crs
                else:
                    return self.plots[0].griddata.metpy.cartopy_crs
            else:
                return lookup_projection(self.projection)
        else:
            return self.projection

    @property
    def _layer_features(self):
        """Iterate over all map features and return as Cartopy objects.

        Handle converting names of maps to auto-scaling map features.

        """
        for item in self.layers:
            if isinstance(item, str):
                feat = lookup_map_feature(item)
            else:
                feat = item

            yield feat

    @observe("area")
    def _set_need_redraw(self, _):
        """Watch traits and set the need redraw flag as necessary."""
        self._need_redraw = True

    @property
    def ax(self):
        """Get the :class:`matplotlib.axes.Axes` to draw on.

        Creates a new instance if necessary.

        """
        # If we haven't actually made an instance yet, make one with the right size and
        # map projection.
        if getattr(self, "_ax", None) is None:
            self._ax = self.parent.figure.add_subplot(*self.layout, projection=self._proj_obj)

        return self._ax

    @ax.setter
    def ax(self, val):
        """Set the :class:`matplotlib.axes.Axes` to draw on.

        Clears existing state as necessary.

        """
        if getattr(self, "_ax", None) is not None:
            self._ax.cla()
        self._ax = val

    def refresh(self, changed):
        """Refresh the drawing if necessary."""
        self._need_redraw = changed.new

    def draw(self):
        """Draw the panel."""
        # Only need to run if we've actually changed.
        if self._need_redraw:

            # Set the extent as appropriate based on the area. One special case for 'global'
            if self.area == "global":
                self.ax.set_global()
            elif self.area is not None:
                # Try to look up if we have a string
                if isinstance(self.area, str):
                    area = _areas[self.area.lower()]
                # Otherwise, assume we have a tuple to use as the extent
                else:
                    area = self.area
                self.ax.set_extent(area, ccrs.PlateCarree())

            # Draw all of the plots.
            for p in self.plots:
                with p.hold_trait_notifications():
                    p.draw()

            # Add all of the maps
            for feat in self._layer_features:
                self.ax.add_feature(feat)

            # Use the set title or generate one.
            title = self.title or ",\n".join(plot.name for plot in self.plots)
            self.ax.set_title(title)
            self._need_redraw = False


@exporter.export
class Plots2D(HasTraits):
    """The highest level class related to plotting 2D data.

    This class collects all common methods no matter whether plotting a scalar variable or
    vector. Primary settings common to all types of 2D plots are time and level.
    """

    parent = Instance(Panel)
    _need_redraw = Bool(default_value=True)

    level = Union([Int(allow_none=True, default_value=None), Instance(units.Quantity)])
    level.__doc__ = """The level of the field to be plotted.

    This is a value with units to choose the desired plot level. For example, selecting the
    850-hPa level, set this parameter to ``850 * units.hPa``
    """

    time = Instance(datetime, allow_none=True)
    time.__doc__ = """Set the valid time to be plotted as a datetime object.

    If a forecast hour is to be plotted the time should be set to the valid future time, which
    can be done using the `~datetime.datetime` and `~datetime.timedelta` objects
    from the Python standard library.
    """

    plot_units = Unicode(allow_none=True, default_value=None)
    plot_units.__doc__ = """The desired units to plot the field in.

    Setting this attribute will convert the units of the field variable to the given units for
    plotting using the MetPy Units module.
    """

    scale = Float(default_value=1e0)
    scale.__doc__ = """Scale the field to be plotted by the value given.

    This attribute will scale the field by multiplying by the scale. For example, to
    scale vorticity to be whole values for contouring you could set the scale to 1e5, such that
    the data values will be scaled by 10^5.
    """

    @property
    def _cmap_obj(self):
        """Return the colormap object.

        Handle convert the name of the colormap to an object from matplotlib or metpy.

        """
        try:
            return ctables.registry.get_colortable(self.colormap)
        except KeyError:
            return plt.get_cmap(self.colormap)

    @property
    def _norm_obj(self):
        """Return the normalization object.

        Converts the tuple image range to a matplotlib normalization instance.

        """
        return plt.Normalize(*self.image_range)

    def clear(self):
        """Clear the plot.

        Resets all internal state and sets need for redraw.

        """
        if getattr(self, "handle", None) is not None:
            if getattr(self.handle, "collections", None) is not None:
                self.clear_collections()
            else:
                self.clear_handle()
            self._need_redraw = True

    def clear_handle(self):
        """Clear the handle to the plot instance."""
        self.handle.remove()
        self.handle = None

    def clear_collections(self):
        """Clear the handle collections to the plot instance."""
        for col in self.handle.collections:
            col.remove()
        self.handle = None

    @observe("parent")
    def _parent_changed(self, _):
        """Handle setting the parent object for the plot."""
        self.clear()

    @observe("level", "time")
    def _update_data(self, _=None):
        """Handle updating the internal cache of data.

        Responds to changes in various subsetting parameters.

        """
        self._griddata = None
        self.clear()

    # Can't be a Traitlet because notifications don't work with arrays for traits
    # notification never happens
    @property
    def data(self):
        """Xarray dataset that contains the field to be plotted."""
        return self._data

    @data.setter
    def data(self, val):
        self._data = val
        self._update_data()

    @property
    def name(self):
        """Generate a name for the plot."""
        if isinstance(self.field, tuple):
            ret = ""
            ret += " and ".join(f for f in self.field)
        else:
            ret = self.field
        if self.level is not None:
            ret += f"@{self.level:d}"
        return ret


@exporter.export
class PlotScalar(Plots2D):
    """Defines the common elements of 2D scalar plots for single scalar value fields.

    Most of the other traits here are for one or more of the specific plots. Currently this
    allows too many options for `ContourPlot` since it does not user image_range, for
    example. Similar issues for `ImagePlot` and `FilledContourPlot`.
    """

    field = Unicode()
    field.__doc__ = """Name of the field to be plotted.

    This is the name of the variable from the dataset that is to be plotted. An example,
    from a model grid file that uses the THREDDS convention for naming would be
    `Geopotential_height_isobaric` or `Temperature_isobaric`. For GOES-16/17 satellite data it
    might be `Sectorized_CMI`. To check for the variables available within a dataset, list the
    variables with the following command assuming the dataset was read using xarray as `ds`,
    `list(ds)`
    """

    @observe("field")
    def _update_data(self, _=None):
        """Handle updating the internal cache of data.

        Responds to changes in various subsetting parameters.

        """
        self._griddata = None
        self.clear()

    @property
    def griddata(self):
        """Return the internal cached data."""
        if getattr(self, "_griddata", None) is None:

            if self.field:
                data = self.data.metpy.parse_cf(self.field)

            elif not hasattr(self.data.metpy, "x"):
                # Handles the case where we have a dataset but no specified field
                raise ValueError("field attribute has not been set.")
            else:
                data = self.data

            subset = {"method": "nearest"}
            if self.level is not None:
                subset[data.metpy.vertical.name] = self.level

            if self.time is not None:
                subset[data.metpy.time.name] = self.time
            data_subset = data.metpy.sel(**subset).squeeze()

            if self.plot_units is not None:
                data_subset = data_subset.metpy.convert_units(self.plot_units)
            self._griddata = data_subset * self.scale

        return self._griddata

    @property
    def plotdata(self):
        """Return the data for plotting.

        The data array, x coordinates, and y coordinates.

        """
        x = self.griddata.metpy.x
        y = self.griddata.metpy.y

        if "degree" in x.units:
            x, y, _ = self.griddata.metpy.cartopy_crs.transform_points(
                ccrs.PlateCarree(), *np.meshgrid(x, y)
            ).T
            x = x[:, 0] % 360
            y = y[0, :]

        return x, y, self.griddata

    def draw(self):
        """Draw the plot."""
        if self._need_redraw:
            if getattr(self, "handle", None) is None:
                self._build()
            if getattr(self, "colorbar", None) is not None:
                self.parent.ax.figure.colorbar(
                    self.handle, orientation=self.colorbar, pad=0, aspect=50
                )
            self._need_redraw = False


class ContourTraits(HasTraits):
    """Represents common contour traits."""

    contours = Union([List(Float()), Int()], default_value=25)
    contours.__doc__ = """A list of values to contour or an integer number of contour levels.

    This parameter sets contour or colorfill values for a plot. Values can be entered either
    as a list of values or as an integer with the number of contours to be plotted (as per
    matplotlib documentation). A list can be generated by using square brackets or creating
    a numpy 1D array and converting it to a list with the `~numpy.ndarray.tolist` method.
    """

    clabels = Bool(default_value=False)
    clabels.__doc__ = """A boolean (True/False) on whether to plot contour labels.

    To plot contour labels set this trait to ``True``, the default value is ``False``.
    """


class ColorfillTraits(HasTraits):
    """Represent common colorfill traits."""

    colormap = Unicode(allow_none=True, default_value=None)
    colormap.__doc__ = """The string name for a Matplolib or MetPy colormap.

    For example, the Blue-Purple colormap from Matplotlib can be accessed using 'BuPu'.
    """

    image_range = Union(
        [Tuple(Int(allow_none=True), Int(allow_none=True)), Instance(plt.Normalize)],
        default_value=(None, None),
    )
    image_range.__doc__ = """A tuple of min and max values that represent the range of values
    to color the rasterized image.

    The min and max values entered as a tuple will be converted to a
    `matplotlib.colors.Normalize` instance for plotting.
    """

    colorbar = Unicode(default_value=None, allow_none=True)
    colorbar.__doc__ = """A string (horizontal/vertical) on whether to add a colorbar to the plot.

    To add a colorbar associated with the plot, set the trait to ``horizontal`` or
    ``vertical``,specifying the orientation of the produced colorbar. The default value is
    ``None``.
    """


@exporter.export
class ImagePlot(PlotScalar, ColorfillTraits):
    """Make raster image using `~matplotlib.pyplot.imshow` for satellite or colored image."""

    @observe("colormap", "image_range")
    def _set_need_redraw(self, _):
        """Handle changes to attributes that just need a simple redraw."""
        if hasattr(self, "handle"):
            self.handle.set_cmap(self._cmap_obj)
            self.handle.set_norm(self._norm_obj)
            self._need_redraw = True

    @observe("colorbar")
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    @property
    def plotdata(self):
        """Return the data for plotting.

        The data array, x coordinates, and y coordinates.

        """
        x = self.griddata.metpy.x
        y = self.griddata.metpy.y

        # At least currently imshow with cartopy does not like this
        if "degree" in x.units:
            x = x.data
            x[x > 180] -= 360

        return x, y, self.griddata

    def _build(self):
        """Build the plot by calling any plotting methods as necessary."""
        x, y, imdata = self.plotdata

        # We use min/max for y and manually figure out origin to try to avoid upside down
        # images created by images where y[0] > y[-1]
        extents = (x[0], x[-1], y.min(), y.max())
        origin = "upper" if y[0] > y[-1] else "lower"
        self.handle = self.parent.ax.imshow(
            imdata,
            extent=extents,
            origin=origin,
            cmap=self._cmap_obj,
            norm=self._norm_obj,
            transform=imdata.metpy.cartopy_crs,
        )


@exporter.export
class ContourPlot(PlotScalar, ContourTraits):
    """Make contour plots by defining specific traits."""

    linecolor = Unicode("black")
    linecolor.__doc__ = """A string value to set the color of plotted contours; default is
    black.

    This trait can be set to any Matplotlib color
    (https://matplotlib.org/3.1.0/gallery/color/named_colors.html)
    """

    linewidth = Int(2)
    linewidth.__doc__ = """An integer value to set the width of plotted contours; default value
    is 2.

    This trait changes the thickness of contour lines with a higher value plotting a thicker
    line.
    """

    linestyle = Unicode("solid", allow_none=True)
    linestyle.__doc__ = """A string value to set the linestyle (e.g., dashed); default is
    solid.

    The valid string values are those of Matplotlib which are solid, dashed, dotted, and
    dashdot.
    """

    @observe("contours", "linecolor", "linewidth", "linestyle", "clabels")
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    def _build(self):
        """Build the plot by calling any plotting methods as necessary."""
        x, y, imdata = self.plotdata
        self.handle = self.parent.ax.contour(
            x,
            y,
            imdata,
            self.contours,
            colors=self.linecolor,
            linewidths=self.linewidth,
            linestyles=self.linestyle,
            transform=imdata.metpy.cartopy_crs,
        )
        if self.clabels:
            self.handle.clabel(inline=1, fmt="%.0f", inline_spacing=8, use_clabeltext=True)


@exporter.export
class FilledContourPlot(PlotScalar, ColorfillTraits, ContourTraits):
    """Make color-filled contours plots by defining appropriate traits."""

    @observe("contours", "colorbar", "colormap")
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    def _build(self):
        """Build the plot by calling any plotting methods as necessary."""
        x, y, imdata = self.plotdata
        self.handle = self.parent.ax.contourf(
            x,
            y,
            imdata,
            self.contours,
            cmap=self._cmap_obj,
            norm=self._norm_obj,
            transform=imdata.metpy.cartopy_crs,
        )


@exporter.export
class PlotVector(Plots2D):
    """Defines common elements for 2D vector plots.

    This class collects common elements including the field trait, which is a tuple argument
    accepting two strings, for plotting 2D vector fields.
    """

    field = Tuple(Unicode(), Unicode())
    field.__doc__ = """A tuple containing the two components of the vector field from the
    dataset in the form (east-west component, north-south component).

    For a wind barb plot each component of the wind must be specified and should be of the form
    (u-wind, v-wind).
    """

    pivot = Unicode("middle")
    pivot.__doc__ = """A string setting the pivot point of the vector. Default value is
    'middle'.

    This trait takes the values of the keyword argument from `matplotlin.pyplot.barbs`:
    'tip' or 'middle'.
    """

    skip = Tuple(Int(), Int(), default_value=(1, 1))
    skip.__doc__ = """A tuple of integers to indicate the number of grid points to skip between
    plotting vectors. Default is (1, 1).

    This trait is to be used to reduce the number of vectors plotted in the (east-west,
    north-south) components. The two values can be set to the same or different integer values
    depending on what is desired.
    """

    earth_relative = Bool(default_value=True)
    earth_relative.__doc__ = """A boolean value to indicate whether the vector to be plotted
    is earth- or grid-relative. Default value is `True`, indicating that vectors are
    earth-relative.

    Common gridded meteorological datasets including GFS and NARR output contain wind
    components that are earth-relative. The primary expection is NAM output with wind
    components that are grid-relative. For any grid-relative vectors set this trait to `False`.
    """

    color = Unicode(default_value="black")
    color.__doc__ = """A string value that controls the color of the vectors. Default value is
    black.

    This trait can be set to any named color from
    `Matplotlibs Colors <https://matplotlib.org/3.1.0/gallery/color/named_colors.html>`
    """

    @observe("field")
    def _update_data(self, _=None):
        """Handle updating the internal cache of data.

        Responds to changes in various subsetting parameters.

        """
        self._griddata_u = None
        self._griddata_v = None
        self.clear()

    @property
    def griddata(self):
        """Return the internal cached data."""
        if getattr(self, "_griddata_u", None) is None:

            if self.field[0]:
                u = self.data.metpy.parse_cf(self.field[0])
                v = self.data.metpy.parse_cf(self.field[1])
            else:
                raise ValueError("field attribute not set correctly")

            subset = {"method": "nearest"}
            if self.level is not None:
                subset[u.metpy.vertical.name] = self.level

            if self.time is not None:
                subset[u.metpy.time.name] = self.time
            data_subset_u = u.metpy.sel(**subset).squeeze()
            data_subset_v = v.metpy.sel(**subset).squeeze()

            if self.plot_units is not None:
                data_subset_u = data_subset_u.metpy.convert_units(self.plot_units)
                data_subset_v = data_subset_v.metpy.convert_units(self.plot_units)
            self._griddata_u = data_subset_u
            self._griddata_v = data_subset_v

        return (self._griddata_u, self._griddata_v)

    @property
    def plotdata(self):
        """Return the data for plotting.

        The data array, x coordinates, and y coordinates.

        """
        x = self.griddata[0].metpy.x
        y = self.griddata[0].metpy.y

        if self.earth_relative:
            x, y, _ = (
                ccrs.PlateCarree()
                .transform_points(self.griddata[0].metpy.cartopy_crs, *np.meshgrid(x, y))
                .T
            )
            x = x.T
            y = y.T
        else:
            if "degree" in x.units:
                x, y, _ = (
                    self.griddata[0]
                    .metpy.cartopy_crs.transform_points(ccrs.PlateCarree(), *np.meshgrid(x, y))
                    .T
                )
                x = x.T % 360
                y = y.T

        if x.ndim == 1:
            xx, yy = np.meshgrid(x, y)
        else:
            xx, yy = x, y

        return xx, yy, self.griddata[0], self.griddata[1]

    def draw(self):
        """Draw the plot."""
        if self._need_redraw:
            if getattr(self, "handle", None) is None:
                self._build()
            self._need_redraw = False


@exporter.export
class BarbPlot(PlotVector):
    """Make plots of wind barbs on a map with traits to refine the look of plotted elements."""

    barblength = Float(default_value=7)
    barblength.__doc__ = """A float value that changes the length of the wind barbs. Default
    value is 7.

    This trait corresponds to the keyword length in `matplotlib.pyplot.barbs`.
    """

    @observe("barblength", "pivot", "skip", "earth_relative", "color")
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    def _build(self):
        """Build the plot by calling needed plotting methods as necessary."""
        x, y, u, v = self.plotdata
        if self.earth_relative:
            transform = ccrs.PlateCarree()
        else:
            transform = u.metpy.cartopy_crs

        wind_slice = (slice(None, None, self.skip[0]), slice(None, None, self.skip[1]))

        self.handle = self.parent.ax.barbs(
            x[wind_slice],
            y[wind_slice],
            u.values[wind_slice],
            v.values[wind_slice],
            color=self.color,
            pivot=self.pivot,
            length=self.barblength,
            transform=transform,
        )


@exporter.export
class PlotObs(HasTraits):
    """The highest level class related to plotting observed surface and upperair data.

    This class collects all common methods no matter whether plotting a upper-level or
    surface data using station plots.

    List of Traits:
      * level
      * time
      * fields
      * locations (optional)
      * time_window (optional)
      * formats (optional)
      * colors (optional)
      * plot_units (optional)
      * vector_field (optional)
      * vector_field_color (optional)
      * vector_field_length (optional)
      * vector_plot_units (optional)
      * reduce_points (optional)
    """

    parent = Instance(Panel)
    _need_redraw = Bool(default_value=True)

    level = Union([Int(allow_none=True), Instance(units.Quantity)], default_value=None)
    level.__doc__ = """The level of the field to be plotted.

    This is a value with units to choose the desired plot level. For example, selecting the
    850-hPa level, set this parameter to ``850 * units.hPa``. For surface data, parameter
    must be set to `None`.
    """

    time = Instance(datetime, allow_none=True)
    time.__doc__ = """Set the valid time to be plotted as a datetime object.

    If a forecast hour is to be plotted the time should be set to the valid future time, which
    can be done using the `~datetime.datetime` and `~datetime.timedelta` objects
    from the Python standard library.
    """

    time_window = Instance(timedelta, default_value=timedelta(minutes=0), allow_none=True)
    time_window.__doc__ = """Set a range to look for data to plot as a timedelta object.

    If this parameter is set, it will subset the data provided to be within the time and plus
    or minus the range value given. If there is more than one observation from a given station
    then it will keep only the most recent one for plotting purposes. Default value is to have
    no range. (optional)
    """

    fields = List(Unicode())
    fields.__doc__ = """Name of the scalar or symbol fields to be plotted.

    List of parameters to be plotted around station plot (e.g., temperature, dewpoint, skyc).
    """

    locations = List(default_value=["C"])
    locations.__doc__ = """List of strings for scalar or symbol field plotting locations.

    List of parameters locations for plotting parameters around the station plot (e.g.,
    NW, NE, SW, SE, W, C). (optional)
    """

    formats = List(default_value=[None])
    formats.__doc__ = """List of the scalar, symbol, and text field data formats. (optional)

    List of scalar parameters formmaters or mapping values (if symbol) for plotting text and/or
    symbols around the station plot (e.g., for pressure variable
    ```lambda v: format(10 * v, '.0f')[-3:]```).

    For symbol mapping the following options are available to be put in as a string:
    current_weather, sky_cover, low_clouds, mid_clouds, high_clouds, and pressure_tendency.

    For plotting text, use the format setting of 'text'.
    """

    colors = List(Unicode(), default_value=["black"])
    colors.__doc__ = """List of the scalar and symbol field colors.

    List of strings that represent the colors to be used for the variable being plotted.
    (optional)
    """

    vector_field = List(default_value=[None], allow_none=True)
    vector_field.__doc__ = """List of the vector field to be plotted.

    List of vector components to combined and plotted from the center of the station plot
    (e.g., wind components). (optional)
    """

    vector_field_color = Unicode("black", allow_none=True)
    vector_field_color.__doc__ = """String color name to plot the vector. (optional)"""

    vector_field_length = Int(default_value=None, allow_none=True)
    vector_field_length.__doc__ = """Integer value to set the length of the plotted vector.
    (optional)
    """

    reduce_points = Float(default_value=0)
    reduce_points.__doc__ = """Float to reduce number of points plotted. (optional)"""

    plot_units = List(default_value=[None], allow_none=True)
    plot_units.__doc__ = """A list of the desired units to plot the fields in.

    Setting this attribute will convert the units of the field variable to the given units for
    plotting using the MetPy Units module, provided that units are attached to the DataFrame.
    """

    vector_plot_units = Unicode(default_value=None, allow_none=True)
    vector_plot_units.__doc__ = """The desired units to plot the vector field in.

    Setting this attribute will convert the units of the field variable to the given units for
    plotting using the MetPy Units module, provided that units are attached to the DataFrame.
    """

    def clear(self):
        """Clear the plot.

        Resets all internal state and sets need for redraw.

        """
        if getattr(self, "handle", None) is not None:
            self.handle.ax.cla()
            self.handle = None
            self._need_redraw = True

    @observe("parent")
    def _parent_changed(self, _):
        """Handle setting the parent object for the plot."""
        self.clear()

    @observe("fields", "level", "time", "vector_field", "time_window")
    def _update_data(self, _=None):
        """Handle updating the internal cache of data.

        Responds to changes in various subsetting parameters.

        """
        self._obsdata = None
        self.clear()

    # Can't be a Traitlet because notifications don't work with arrays for traits
    # notification never happens
    @property
    def data(self):
        """Pandas dataframe that contains the fields to be plotted."""
        return self._data

    @data.setter
    def data(self, val):
        self._data = val
        self._update_data()

    @property
    def name(self):
        """Generate a name for the plot."""
        ret = ""
        ret += " and ".join(f for f in self.fields)
        if self.level is not None:
            ret += f"@{self.level:d}"
        return ret

    @property
    def obsdata(self):
        """Return the internal cached data."""
        if getattr(self, "_obsdata", None) is None:
            # Use a copy of data so we retain all of the original data passed in unmodified
            data = self.data

            # Subset for a particular level if given
            if self.level is not None:
                mag = getattr(self.level, "magnitude", self.level)
                data = data[data.pressure == mag]

            # Subset for our particular time
            if self.time is not None:
                # If data are not currently indexed by time, we need to do so choosing one of
                # the columns we're looking for
                if not isinstance(data.index, pd.DatetimeIndex):
                    time_vars = ["valid", "time", "valid_time", "date_time", "date"]
                    dim_times = [
                        time_var for time_var in time_vars if time_var in list(self.data)
                    ]
                    if not dim_times:
                        raise AttributeError(
                            "Time variable not found. Valid variable names are:" f"{time_vars}"
                        )

                    data = data.set_index(dim_times[0])
                    if not isinstance(data.index, pd.DatetimeIndex):
                        # Convert our column of interest to a datetime
                        data = data.reset_index()
                        time_index = pd.to_datetime(data[dim_times[0]])
                        data = data.set_index(time_index)

                # Works around the fact that traitlets 4.3 insists on sending us None by
                # default because timedelta(0) is Falsey.
                window = timedelta(minutes=0) if self.time_window is None else self.time_window

                # Indexes need to be properly sorted for the slicing below to work; the
                # error you get if that's not the case really convoluted, which is why
                # we don't rely on users doing it.
                data = data.sort_index()
                data = data[self.time - window : self.time + window]

            # Look for the station column
            stn_vars = ["station", "stn", "station_id", "stid"]
            dim_stns = [stn_var for stn_var in stn_vars if stn_var in list(self.data)]
            if not dim_stns:
                raise AttributeError(
                    "Station variable not found. Valid variable names are: " f"{stn_vars}"
                )
            else:
                dim_stn = dim_stns[0]

            # Make sure we only use one observation per station
            self._obsdata = data.groupby(dim_stn).tail(1)

        return self._obsdata

    @property
    def plotdata(self):
        """Return the data for plotting.

        The data arrays, x coordinates, and y coordinates.

        """
        plot_data = {}
        for dim_name in list(self.obsdata):
            if dim_name.find("lat") != -1:
                lat = self.obsdata[dim_name]
            elif dim_name.find("lon") != -1:
                lon = self.obsdata[dim_name]
            else:
                plot_data[dim_name] = self.obsdata[dim_name]
        return lon.values, lat.values, plot_data

    def draw(self):
        """Draw the plot."""
        if self._need_redraw:
            if getattr(self, "handle", None) is None:
                self._build()
            self._need_redraw = False

    @observe("colors", "formats", "locations", "reduce_points", "vector_field_color")
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    def _build(self):
        """Build the plot by calling needed plotting methods as necessary."""
        lon, lat, data = self.plotdata

        # Use the cartopy map projection to transform station locations to the map and
        # then refine the number of stations plotted by setting a radius
        if self.parent._proj_obj == ccrs.PlateCarree():
            scale = 1.0
        else:
            scale = 100000.0
        point_locs = self.parent._proj_obj.transform_points(ccrs.PlateCarree(), lon, lat)
        subset = reduce_point_density(point_locs, self.reduce_points * scale)

        self.handle = StationPlot(
            self.parent.ax,
            lon[subset],
            lat[subset],
            clip_on=True,
            transform=ccrs.PlateCarree(),
            fontsize=10,
        )

        for i, ob_type in enumerate(self.fields):
            field_kwargs = {}
            if len(self.locations) > 1:
                location = self.locations[i]
            else:
                location = self.locations[0]
            if len(self.colors) > 1:
                field_kwargs["color"] = self.colors[i]
            else:
                field_kwargs["color"] = self.colors[0]
            if len(self.formats) > 1:
                field_kwargs["formatter"] = self.formats[i]
            else:
                field_kwargs["formatter"] = self.formats[0]
            if len(self.plot_units) > 1:
                field_kwargs["plot_units"] = self.plot_units[i]
            else:
                field_kwargs["plot_units"] = self.plot_units[0]
            if hasattr(self.data, "units") and (field_kwargs["plot_units"] is not None):
                parameter = data[ob_type][subset].values * units(self.data.units[ob_type])
            else:
                parameter = data[ob_type][subset]
            if field_kwargs["formatter"] is not None:
                mapper = getattr(wx_symbols, str(field_kwargs["formatter"]), None)
                if mapper is not None:
                    field_kwargs.pop("formatter")
                    self.handle.plot_symbol(location, parameter, mapper, **field_kwargs)
                else:
                    if self.formats[i] == "text":
                        self.handle.plot_text(
                            location, data[ob_type][subset], color=field_kwargs["color"]
                        )
                    else:
                        self.handle.plot_parameter(
                            location, data[ob_type][subset], **field_kwargs
                        )
            else:
                field_kwargs.pop("formatter")
                self.handle.plot_parameter(location, parameter, **field_kwargs)

        if self.vector_field[0] is not None:
            vector_kwargs = {}
            vector_kwargs["color"] = self.vector_field_color
            vector_kwargs["plot_units"] = self.vector_plot_units
            if hasattr(self.data, "units") and (vector_kwargs["plot_units"] is not None):
                u = data[self.vector_field[0]][subset].values * units(
                    self.data.units[self.vector_field[0]]
                )
                v = data[self.vector_field[1]][subset].values * units(
                    self.data.units[self.vector_field[1]]
                )
            else:
                vector_kwargs.pop("plot_units")
                u = data[self.vector_field[0]][subset]
                v = data[self.vector_field[1]][subset]
            if self.vector_field_length is not None:
                vector_kwargs["length"] = self.vector_field_length
            self.handle.plot_barb(u, v, **vector_kwargs)
