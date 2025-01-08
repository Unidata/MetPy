# Copyright (c) 2016,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Functionality for doing text-based plotting."""

import itertools

from matplotlib import rcParams
from matplotlib.artist import allow_rasterization
import matplotlib.cbook as cbook
from matplotlib.text import Text
import matplotlib.transforms as mtransforms
import numpy as np

from ..package_tools import Exporter

exporter = Exporter(globals())


@exporter.export
def scattertext(ax, x, y, texts, loc=(0, 0), formatter=None, **kw):
    """Add text to the axes.

    Add text in string `s` to axis at location `x`, `y`, data
    coordinates.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Matplotlib Axes to draw on

    x, y : array-like, (N, )
        Input positions

    texts : array-like, (N, )
        Collection of text that will be plotted at each (x,y) location

    loc : tuple[int, int], optional
        Offset (in screen coordinates) from x,y position. Allows positioning text relative
        to original point. Default is (0, 0), which is no offset.

    formatter : str or Callable, optional
        How to format the each entry in `texts` for plotting. If a string, it should be
        compatible with the :func:`format` builtin. If a callable, this should take a
        value and return a string. Default is ``None``, which performs no formatting and
        requires every object in `texts` to be a `str`.

    Other Parameters
    ----------------
    kwargs : `~matplotlib.text.TextCollection` properties.
        Other miscellaneous text parameters.

    Examples
    --------
    Individual keyword arguments can be used to override any given
    parameter::

        >>> ax = plt.gca()
        >>> scattertext(ax, [0.25, 0.75], [0.25, 0.75], ['aa', 'bb'],
        ... fontsize=12)  #doctest: +ELLIPSIS
        TextCollection

    The default setting to to center the text at the specified x, y
    locations in data coordinates. The example below places the text
    above and to the right by 10 pixels::

        >>> ax = plt.gca()
        >>> scattertext(ax, [0.25, 0.75], [0.25, 0.75], ['aa', 'bb'],
        ... loc=(10, 10))  #doctest: +ELLIPSIS
        TextCollection

    """
    # Start with default args and update from kw
    new_kw = {
        'verticalalignment': 'center',
        'horizontalalignment': 'center',
        'transform': ax.transData,
        'clip_on': False}
    new_kw.update(kw)

    # Handle masked arrays
    x, y, texts = cbook.delete_masked_points(x, y, texts)

    if formatter is not None:
        if not callable(formatter):
            fmt = formatter

            def formatter(s):
                """Turn a format string into a callable."""
                return format(s, fmt)
        texts = [formatter(v) for v in texts]

    # If there is nothing left after deleting the masked points, return None
    if x.size == 0:
        return None

    # Make the TextCollection object
    text_obj = TextCollection(x, y, texts, offset=loc, **new_kw)

    # The margin adjustment is a hack to deal with the fact that we don't
    # want to transform all the symbols whose scales are in points
    # to data coords to get the exact bounding box for efficiency
    # reasons.  It can be done right if this is deemed important.
    # Also, only bother with this padding if there is anything to draw.
    xmargin, ymargin = ax.margins()
    if xmargin < 0.05:
        ax.set_xmargin(0.05)

    if ymargin < 0.05:
        ax.set_ymargin(0.05)

    # Add it to the axes and update range
    ax.add_artist(text_obj)

    # Matplotlib at least up to 3.2.2 does not properly clip text with paths, so
    # work-around by setting to the bounding box of the Axes
    # TODO: Remove when fixed in our minimum supported version of matplotlib
    text_obj.clipbox = ax.bbox

    ax.update_datalim(text_obj.get_datalim(ax.transData))
    ax.autoscale_view()
    return text_obj


class TextCollection(Text):
    """Handle plotting a collection of text.

    Text Collection plots text with a collection of similar properties: font, color,
    and an offset relative to the x,y data location.
    """

    def __init__(self, x, y, text, offset=(0, 0), **kwargs):
        """Initialize an instance of `TextCollection`.

        This class encompasses drawing a collection of text values at a variety
        of locations.

        Parameters
        ----------
        x : array-like
            The x locations, in data coordinates, for the text

        y : array-like
            The y locations, in data coordinates, for the text

        text : array-like of str
            The string values to draw

        offset : (int, int), optional
            The offset x and y, in normalized coordinates, to draw the text relative
            to the data locations. Defaults to (0, 0), which has no offset.

        kwargs : arbitrary keywords arguments

        """
        Text.__init__(self, **kwargs)
        self.x = x
        self.y = y
        self.text = text
        self.offset = offset

    def __str__(self):
        """Make a string representation of `TextCollection`."""
        return 'TextCollection'

    __repr__ = __str__

    def get_datalim(self, transData):  # noqa: N803
        """Return the limits of the data.

        Parameters
        ----------
        transData : matplotlib.transforms.Transform

        Returns
        -------
        matplotlib.transforms.Bbox
            The bounding box of the data

        """
        full_transform = self.get_transform() - transData
        posx = self.convert_xunits(self.x)
        posy = self.convert_yunits(self.y)
        XY = full_transform.transform(np.vstack((posx, posy)).T)  # noqa: N806
        bbox = mtransforms.Bbox.null()
        bbox.update_from_data_xy(XY, ignore=True)
        return bbox

    @allow_rasterization
    def draw(self, renderer):
        """Draw the :class:`TextCollection` object to the given *renderer*."""
        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible():
            return
        if not any(self.text):
            return

        renderer.open_group('text', self.get_gid())

        trans = self.get_transform()
        if self.offset != (0, 0):
            scale = self.axes.figure.dpi / 72
            xoff, yoff = self.offset
            trans += mtransforms.Affine2D().translate(scale * xoff,
                                                      scale * yoff)

        posx = self.convert_xunits(self.x)
        posy = self.convert_yunits(self.y)
        pts = np.vstack((posx, posy)).T
        pts = trans.transform(pts)
        _, canvash = renderer.get_canvas_width_height()

        gc = renderer.new_gc()
        gc.set_foreground(self.get_color())
        gc.set_alpha(self.get_alpha())
        gc.set_url(self._url)
        self._set_gc_clip(gc)

        angle = self.get_rotation()

        # cycle() makes text repeat to match the points given
        for (posx, posy), t in zip(pts, itertools.cycle(self.text)):
            # Skip empty strings--not only is this a performance gain, but it fixes
            # rendering with path effects below.
            if not t:
                continue

            self._text = t  # hack to allow self._get_layout to work
            _, info, _ = self._get_layout(renderer)
            self._text = ''

            for line, _, x, y in info:

                mtext = self if len(info) == 1 else None
                x = x + posx
                y = y + posy
                if renderer.flipy():
                    y = canvash - y

                clean_line, ismath = self._preprocess_math(line)

                if self.get_path_effects():
                    from matplotlib.patheffects import PathEffectRenderer
                    textrenderer = PathEffectRenderer(
                                        self.get_path_effects(), renderer)  # noqa: E126
                else:
                    textrenderer = renderer

                if self.get_usetex():
                    textrenderer.draw_tex(gc, x, y, clean_line,
                                          self._fontproperties, angle,
                                          mtext=mtext)
                else:
                    textrenderer.draw_text(gc, x, y, clean_line,
                                           self._fontproperties, angle,
                                           ismath=ismath, mtext=mtext)

        gc.restore()
        renderer.close_group('text')

    def set_usetex(self, usetex):
        """
        Set this `Text` object to render using TeX (or not).

        If `None` is given, the option will be reset to use the value of
        `rcParams['text.usetex']`
        """
        self._usetex = None if usetex is None else bool(usetex)
        self.stale = True

    def get_usetex(self):
        """
        Return whether this `Text` object will render using TeX.

        If the user has not manually set this value, it will default to
        the value of `rcParams['text.usetex']`
        """
        if self._usetex is None:
            return rcParams['text.usetex']
        else:
            return self._usetex
