# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""A module with utilities to simplify using the weather symbol font.

See WMO manual 485 Vol 1 for more info on the symbols.
"""

from pkg_resources import resource_filename
import matplotlib.font_manager as fm

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
    r'Maps integer values to font code points.'
    def __init__(self, num, font_start, font_jumps=None, char_jumps=None):
        """

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
        "Safely pop from a list; returns None if list empty."
        return l.pop(0) if l else None

    def __call__(self, code):
        return self.chrs[code]

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

#: Current weather
current_weather = CodePointMapping(100, 0xE9A2, [(7, 2), (93, 2), (94, 2), (95, 2), (97, 2)],
                                   [(0, 4)])

#: Current weather from an automated station
current_weather_auto = CodePointMapping(100, 0xE94B, [(92, 2), (95, 2)],
                                        [(6, 4), (13, 5), (19, 1), (36, 4), (49, 1), (59, 1),
                                         (69, 1), (79, 1), (88, 1), (97, 2)])

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
