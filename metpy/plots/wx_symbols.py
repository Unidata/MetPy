# Copyright (c) 2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Simplify using the weather symbol font.

See WMO manual 485 Vol 1 for more info on the symbols.
"""

import matplotlib.font_manager as fm
from pkg_resources import resource_filename

from ..package_tools import Exporter

exporter = Exporter(globals())

# Create a matplotlib font object pointing to our weather symbol font
wx_symbol_font = fm.FontProperties(fname=resource_filename('metpy.plots',
                                                           'fonts/wx_symbols.ttf'))

# Deal with the fact that Python 2 chr() can't handle unicode, but unichr is gone
# on python 3
try:
    code_point = unichr
except NameError:
    code_point = chr


class CodePointMapping(object):
    """Map integer values to font code points."""

    def __init__(self, num, font_start, font_jumps=None, char_jumps=None):
        """Initialize the instance.

        Parameters
        ----------
        num : int
            The number of values that will be mapped
        font_start : int
            The first code point in the font to use in the mapping
        font_jumps : list[int, int], optional
            Sequence of code point jumps in the font. These are places where the next
            font code point does not correspond to a new input code. This is usually caused
            by there being multiple symbols for a single code. Defaults to :data:`None`, which
            indicates no jumps.
        char_jumps : list[int, int], optional
            Sequence of code jumps. These are places where the next code value does not
            have a valid code point in the font. This usually comes from place in the WMO
            table where codes have no symbol. Defaults to :data:`None`, which indicates no
            jumps.

        """
        next_font_jump = self._safe_pop(font_jumps)
        next_char_jump = self._safe_pop(char_jumps)
        font_point = font_start
        self.chrs = []
        code = 0
        while code < num:
            if next_char_jump and code >= next_char_jump[0]:
                jump_len = next_char_jump[1]
                code += jump_len
                self.chrs.extend([''] * jump_len)
                next_char_jump = self._safe_pop(char_jumps)
            else:
                self.chrs.append(code_point(font_point))
                if next_font_jump and code >= next_font_jump[0]:
                    font_point += next_font_jump[1]
                    next_font_jump = self._safe_pop(font_jumps)
                code += 1
                font_point += 1

    @staticmethod
    def _safe_pop(l):
        """Safely pop from a list.

        Returns None if list empty.

        """
        return l.pop(0) if l else None

    def __call__(self, code):
        """Return the Unicode code point corresponding to `code`."""
        return self.chrs[code]

    def __len__(self):
        """Return the number of codes supported by this mapping."""
        return len(self.chrs)

    def alt_char(self, code, alt):
        """Get one of the alternate code points for a given value.

        In the WMO tables, some code have multiple symbols. This allows getting that
        symbol rather than main one.

        Parameters
        ----------
        code : int
            The code for looking up the font code point
        alt : int
            The number of the alternate symbol

        Returns
        -------
        int
            The appropriate code point in the font

        """
        return code_point(ord(self(code)) + alt)


#
# Set up mapping objects for various groups of symbols. The integer values follow from
# the WMO.
#

with exporter:
    #: Current weather
    current_weather = CodePointMapping(100, 0xE9A2, [(7, 2), (93, 2), (94, 2), (95, 2),
                                                     (97, 2)], [(0, 4)])

    #: Current weather from an automated station
    current_weather_auto = CodePointMapping(100, 0xE94B, [(92, 2), (95, 2)],
                                            [(6, 4), (13, 5), (19, 1), (36, 4), (49, 1),
                                             (59, 1), (69, 1), (79, 1), (88, 1), (97, 2)])

    #: Low clouds
    low_clouds = CodePointMapping(10, 0xE933, [(7, 1)], [(0, 1)])

    #: Mid-altitude clouds
    mid_clouds = CodePointMapping(10, 0xE93D, char_jumps=[(0, 1)])

    #: High clouds
    high_clouds = CodePointMapping(10, 0xE946, char_jumps=[(0, 1)])

    #: Sky cover symbols
    sky_cover = CodePointMapping(12, 0xE90A)

    #: Pressure tendency
    pressure_tendency = CodePointMapping(10, 0xE900)

    #####################################################################
    # This dictionary is for mapping METAR present weather text codes
    # to WMO codes for plotting wx symbols along with the station plots.
    # Pages II-4-3 and II-4-4 of this document describes the difference between
    # manned and automated stations:
    # https://github.com/Unidata/MetPy/files/1151142/485_Vol_I_en.pdf
    # It may become necessary to add automated station wx_codes in the future,
    # but that will also require knowing the status of all stations.

    wx_code_map = {'': 0, 'M': 0, 'TSNO': 0, 'TS': 0, 'VA': 4, 'FU': 4, 'HZ': 5,
                   'DU': 6, 'BLDU': 7, 'PO': 8, 'VCSS': 9, 'BR': 10,
                   'MIFG': 11, 'VCTS': 13, 'VIRGA': 14, 'VCSH': 16,
                   '-VCTSRA': 17, 'VCTSRA': 17, '+VCTSRA': 17,
                   'THDR': 17, 'SQ': 18, 'FC': 19, 'DS': 31, 'SS': 31,
                   '+DS': 34, '+SS': 34, 'DRSN': 36, '+DRSN': 37, 'BLSN': 38,
                   '+BLSN': 39, 'VCFG': 40, 'BCFG': 41, 'PRFG': 44, 'FG': 45,
                   'FZFG': 49, '-DZ': 51, 'DZ': 53, '+DZ': 55, '-FZDZ': 56,
                   'FZDZ': 57, '+FZDZ': 57, '-DZRA': 58, 'DZRA': 59, '-RA': 61,
                   'RA': 63, '+RA': 65, '-FZRA': 66, 'FZRA': 67, '+FZRA': 67,
                   '-RASN': 68, 'RASN': 69, '+RASN': 69, '-SN': 71, 'SN': 73,
                   '+SN': 75, 'IN': 76, '-UP': 76, 'UP': 76, '+UP': 76, 'SG': 77,
                   'IC': 78, '-PL': 79, 'PL': 79, '-SH': 80, '-SHRA': 80,
                   'SH': 81, 'SHRA': 81, '+SH': 81, '+SHRA': 81, '-SHRASN': 83,
                   '-SHSNRA': 83, 'SHRASN': 84, '+SHRASN': 84, 'SHSNRA': 84,
                   '+SHSNRA': 84, '-SHSN': 85, 'SHSN': 86, '+SHSN': 86, '-GS': 87,
                   '-SHGS': 87, 'GS': 88, 'SHGS': 88, '+GS': 88, '+SHGS': 88,
                   '-GR': 89, '-SHGR': 89, 'GR': 90, 'SHGR': 90, '+GR': 90,
                   '+SHGR': 90, '-TSRA': 95, 'TSRA': 95, 'TSSN': 95, 'TSPL': 95,
                   'TSGS': 96, 'TSGR': 96, '+TSRA': 97, '+TSSN': 97, '+TSPL': 97,
                   'TSSA': 98, 'TSDS': 98, '+TSGS': 99, '+TSGR': 99}
