# Copyright (c) 2016,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Simplify using the weather symbol font.

See WMO manual 485 Vol 1 for more info on the symbols.
"""

import matplotlib.font_manager as fm
import numpy as np
from pkg_resources import resource_filename

from ..package_tools import Exporter

exporter = Exporter(globals())

# Create a matplotlib font object pointing to our weather symbol font
wx_symbol_font = fm.FontProperties(fname=resource_filename('metpy.plots',
                                                           'fonts/wx_symbols.ttf'))


@exporter.export
def wx_code_to_numeric(codes):
    """Determine the numeric weather symbol value from METAR code text.

    A robust method to identifies the numeric value for plotting the correct symbol from a
    decoded METAR current weather group. The METAR codes should be strings with no missing
    values or NaN strings (empty strings are okay).

    For example, if from a Pandas Dataframe sfc_df.wxcodes.fillna('')

    Parameters
    ----------
    codes : Array like containing string values of METAR weather codes

    Returns
    -------
    array of numeric codes of current weather symbols from the wx_code_map for use in
    plotting.
    """
    wx_sym_list = []
    for s in codes:
        wxcode = s.split()[0] if ' ' in s else s
        try:
            wx_sym_list.append(wx_code_map[wxcode])
        except KeyError:
            if wxcode[0].startswith(('-', '+')):
                options = [slice(None, 7), slice(None, 5), slice(1, 5), slice(None, 3),
                           slice(1, 3)]
            else:
                options = [slice(None, 6), slice(None, 4), slice(None, 2)]

            for opt in options:
                try:
                    wx_sym_list.append(wx_code_map[wxcode[opt]])
                    break
                except KeyError:
                    pass
            else:
                wx_sym_list.append(0)

    return np.array(wx_sym_list)


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
                self.chrs.append(chr(font_point))
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
        return chr(ord(self(code)) + alt)


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

    wx_code_map = {'': 0, 'M': 0, 'TSNO': 0, 'VA': 4, 'FU': 4,
                   'HZ': 5, 'DU': 6, 'BLDU': 7, 'SA': 7,
                   'BLSA': 7, 'VCBLSA': 7, 'VCBLDU': 7, 'BLPY': 7,
                   'PO': 8, 'VCPO': 8, 'VCDS': 9, 'VCSS': 9,
                   'BR': 10, 'BCBR': 10, 'BC': 11, 'MIFG': 12,
                   'VCTS': 13, 'VIRGA': 14, 'VCSH': 16, 'TS': 17,
                   'THDR': 17, 'VCTSHZ': 17, 'TSFZFG': 17, 'TSBR': 17,
                   'TSDZ': 17, 'SQ': 18, 'FC': 19, '+FC': 19,
                   'DS': 31, 'SS': 31, 'DRSA': 31, 'DRDU': 31,
                   'TSUP': 32, '+DS': 34, '+SS': 34, '-BLSN': 36,
                   'BLSN': 36, '+BLSN': 36, 'VCBLSN': 36, 'DRSN': 38,
                   '+DRSN': 38, 'VCFG': 40, 'BCFG': 41, 'PRFG': 44,
                   'FG': 45, 'FZFG': 49, '-VCTSDZ': 51, '-DZ': 51,
                   '-DZBR': 51, 'VCTSDZ': 53, 'DZ': 53, '+VCTSDZ': 55,
                   '+DZ': 55, '-FZDZ': 56, '-FZDZSN': 56, 'FZDZ': 57,
                   '+FZDZ': 57, 'FZDZSN': 57, '-DZRA': 58, 'DZRA': 59,
                   '+DZRA': 59, '-VCTSRA': 61, '-RA': 61, '-RABR': 61,
                   'VCTSRA': 63, 'RA': 63, 'RABR': 63, 'RAFG': 63,
                   '+VCTSRA': 65, '+RA': 65, '-FZRA': 66, '-FZRASN': 66,
                   '-FZRABR': 66, '-FZRAPL': 66, '-FZRASNPL': 66, 'TSFZRAPL': 67,
                   '-TSFZRA': 67, 'FZRA': 67, '+FZRA': 67, 'FZRASN': 67,
                   'TSFZRA': 67, '-DZSN': 68, '-RASN': 68, '-SNRA': 68,
                   '-SNDZ': 68, 'RASN': 69, '+RASN': 69, 'SNRA': 69,
                   'DZSN': 69, 'SNDZ': 69, '+DZSN': 69, '+SNDZ': 69,
                   '-VCTSSN': 71, '-SN': 71, '-SNBR': 71, 'VCTSSN': 73,
                   'SN': 73, '+VCTSSN': 75, '+SN': 75, 'VCTSUP': 76,
                   'IN': 76, '-UP': 76, 'UP': 76, '+UP': 76,
                   '-SNSG': 77, 'SG': 77, '-SG': 77, 'IC': 78,
                   '-FZDZPL': 79, '-FZDZPLSN': 79, 'FZDZPL': 79, '-FZRAPLSN': 79,
                   'FZRAPL': 79, '+FZRAPL': 79, '-RAPL': 79, '-RASNPL': 79,
                   '-RAPLSN': 79, '+RAPL': 79, 'RAPL': 79, '-SNPL': 79,
                   'SNPL': 79, '-PL': 79, 'PL': 79, '-PLSN': 79,
                   '-PLRA': 79, 'PLRA': 79, '-PLDZ': 79, '+PL': 79,
                   'PLSN': 79, 'PLUP': 79, '+PLSN': 79, '-SH': 80,
                   '-SHRA': 80, 'SH': 81, 'SHRA': 81, '+SH': 81,
                   '+SHRA': 81, '-SHRASN': 83, '-SHSNRA': 83, '+SHRABR': 84,
                   'SHRASN': 84, '+SHRASN': 84, 'SHSNRA': 84, '+SHSNRA': 84,
                   '-SHSN': 85, 'SHSN': 86, '+SHSN': 86, '-GS': 87,
                   '-SHGS': 87, 'FZRAPLGS': 88, '-SNGS': 88, 'GSPLSN': 88,
                   'GSPL': 88, 'PLGSSN': 88, 'GS': 88, 'SHGS': 88,
                   '+GS': 88, '+SHGS': 88, '-GR': 89, '-SHGR': 89,
                   '-SNGR': 90, 'GR': 90, 'SHGR': 90, '+GR': 90,
                   '+SHGR': 90, '-TSRA': 95, 'TSRA': 95, 'TSSN': 95,
                   'TSPL': 95, '-TSDZ': 95, '-TSSN': 95, '-TSPL': 95,
                   'TSPLSN': 95, 'TSSNPL': 95, '-TSSNPL': 95, 'TSRAGS': 96,
                   'TSGS': 96, 'TSGR': 96, '+TSRA': 97, '+TSSN': 97,
                   '+TSPL': 97, '+TSPLSN': 97, 'TSSA': 98, 'TSDS': 98,
                   'TSDU': 98, '+TSGS': 99, '+TSGR': 99}
