'This module fills in for functionality that we have (or will) upstreamed into matplotlib'
# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# See if we should monkey-patch Barbs for better pivot
import matplotlib
if float(matplotlib.__version__[:3]) < 2.1:
    import numpy as np
    from numpy import ma
    import matplotlib.transforms as transforms
    from matplotlib.patches import CirclePolygon
    from matplotlib.quiver import Barbs

    def _make_barbs(self, u, v, nflags, nbarbs, half_barb, empty_flag, length,
                    pivot, sizes, fill_empty, flip):
        'Monkey-patched version of _make_barbs. Allows pivot to be a float value.'

        # These control the spacing and size of barb elements relative to the
        # length of the shaft
        spacing = length * sizes.get('spacing', 0.125)
        full_height = length * sizes.get('height', 0.4)
        full_width = length * sizes.get('width', 0.25)
        empty_rad = length * sizes.get('emptybarb', 0.15)

        # Controls y point where to pivot the barb.
        pivot_points = dict(tip=0.0, middle=-length / 2.)

        # Check for flip
        if flip:
            full_height = -full_height

        endx = 0.0
        try:
            endy = float(pivot)
        except ValueError:
            endy = pivot_points[pivot.lower()]

        # Get the appropriate angle for the vector components.  The offset is
        # due to the way the barb is initially drawn, going down the y-axis.
        # This makes sense in a meteorological mode of thinking since there 0
        # degrees corresponds to north (the y-axis traditionally)
        angles = -(ma.arctan2(v, u) + np.pi / 2)

        # Used for low magnitude.  We just get the vertices, so if we make it
        # out here, it can be reused.  The center set here should put the
        # center of the circle at the location(offset), rather than at the
        # same point as the barb pivot; this seems more sensible.
        circ = CirclePolygon((0, 0), radius=empty_rad).get_verts()
        if fill_empty:
            empty_barb = circ
        else:
            # If we don't want the empty one filled, we make a degenerate
            # polygon that wraps back over itself
            empty_barb = np.concatenate((circ, circ[::-1]))

        barb_list = []
        for index, angle in np.ndenumerate(angles):
            # If the vector magnitude is too weak to draw anything, plot an
            # empty circle instead
            if empty_flag[index]:
                # We can skip the transform since the circle has no preferred
                # orientation
                barb_list.append(empty_barb)
                continue

            poly_verts = [(endx, endy)]
            offset = length

            # Add vertices for each flag
            for i in range(nflags[index]):
                # The spacing that works for the barbs is a little to much for
                # the flags, but this only occurs when we have more than 1
                # flag.
                if offset != length:
                    offset += spacing / 2.
                poly_verts.extend(
                    [[endx, endy + offset],
                     [endx + full_height, endy - full_width / 2 + offset],
                     [endx, endy - full_width + offset]])

                offset -= full_width + spacing

            # Add vertices for each barb.  These really are lines, but works
            # great adding 3 vertices that basically pull the polygon out and
            # back down the line
            for i in range(nbarbs[index]):
                poly_verts.extend(
                    [(endx, endy + offset),
                     (endx + full_height, endy + offset + full_width / 2),
                     (endx, endy + offset)])

                offset -= spacing

            # Add the vertices for half a barb, if needed
            if half_barb[index]:
                # If the half barb is the first on the staff, traditionally it
                # is offset from the end to make it easy to distinguish from a
                # barb with a full one
                if offset == length:
                    poly_verts.append((endx, endy + offset))
                    offset -= 1.5 * spacing
                poly_verts.extend(
                    [(endx, endy + offset),
                     (endx + full_height / 2, endy + offset + full_width / 4),
                     (endx, endy + offset)])

            # Rotate the barb according the angle. Making the barb first and
            # then rotating it made the math for drawing the barb really easy.
            # Also, the transform framework makes doing the rotation simple.
            poly_verts = transforms.Affine2D().rotate(-angle).transform(
                poly_verts)
            barb_list.append(poly_verts)

        return barb_list

    # Replace existing method
    Barbs._make_barbs = _make_barbs


# See if we need to patch in our own scattertext implementation
from matplotlib.axes import Axes  # noqa
if not hasattr(Axes, 'scattertext'):
    import matplotlib.cbook as cbook
    import matplotlib.transforms as mtransforms
    from matplotlib import rcParams
    from matplotlib.artist import allow_rasterization
    from matplotlib.text import Text

    def scattertext(self, x, y, texts, loc=(0, 0), **kw):
        """
        Add text to the axes.

        Add text in string `s` to axis at location `x`, `y`, data
        coordinates.

        Parameters
        ----------
        x, y : array_like, shape (n, )
            Input positions

        texts : array_like, shape (n, )
            Collection of text that will be plotted at each (x,y) location

        loc : length-2 tuple
            Offset (in screen coordinates) from x,y position. Allows
            positioning text relative to original point.

        Other parameters
        ----------------
        kwargs : `~matplotlib.text.TextCollection` properties.
            Other miscellaneous text parameters.

        Examples
        --------
        Individual keyword arguments can be used to override any given
        parameter::

            >>> scattertext(x, y, texts, fontsize=12)

        The default setting to to center the text at the specified x,y
        locations in data coordinates, and to take the data and format as
        float without any decimal places. The example below places the text
        above and to the right by 10 pixels, with 2 decimal places::

            >>> scattertext([0.25, 0.75], [0.25, 0.75], [0.5, 1.0],
            ...             loc=(10, 10))
        """
        # Start with default args and update from kw
        new_kw = {
            'verticalalignment': 'center',
            'horizontalalignment': 'center',
            'transform': self.transData,
            'clip_on': False}
        new_kw.update(kw)

        # Default to centered on point--special case it to keep transform
        # simpler.
        # t = new_kw['transform']
        # if loc == (0, 0):
        #     trans = t
        # else:
        #     x0, y0 = loc
        #     trans = t + mtransforms.Affine2D().translate(x0, y0)
        # new_kw['transform'] = trans

        # Handle masked arrays
        x, y, texts = cbook.delete_masked_points(x, y, texts)

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
        if self._xmargin < 0.05 and x.size > 0:
            self.set_xmargin(0.05)

        if self._ymargin < 0.05 and x.size > 0:
            self.set_ymargin(0.05)

        # Add it to the axes and update range
        self.add_artist(text_obj)
        self.update_datalim(text_obj.get_datalim(self.transData))
        self.autoscale_view()
        return text_obj

    class TextCollection(Text):
        """Handles plotting a collection of text.

        Text Collection plots text with a collection of similar properties: font, color,
        and an offset relative to the x,y data location.
        """
        def __init__(self, x, y, text, offset=(0, 0), **kwargs):
            Text.__init__(self, **kwargs)
            self.x = x
            self.y = y
            self.text = text
            self.offset = offset
            if not hasattr(self, '_usetex'):  # Only needed for matplotlib 1.4 compatibility
                self._usetex = None

        def __str__(self):
            return "TextCollection"

        def get_datalim(self, transData):  # noqa
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
            XY = full_transform.transform(np.vstack((self.x, self.y)).T)  # noqa
            bbox = transforms.Bbox.null()
            bbox.update_from_data_xy(XY, ignore=True)
            return bbox

        @allow_rasterization
        def draw(self, renderer):
            """
            Draws the :class:`TextCollection` object to the given *renderer*.
            """
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
            canvasw, canvash = renderer.get_canvas_width_height()

            gc = renderer.new_gc()
            gc.set_foreground(self.get_color())
            gc.set_alpha(self.get_alpha())
            gc.set_url(self._url)
            self._set_gc_clip(gc)

            angle = self.get_rotation()

            for (posx, posy), t in zip(pts, self.text):
                self._text = t  # hack to allow self._get_layout to work
                bbox, info, descent = self._get_layout(renderer)
                self._text = ''

                for line, wh, x, y in info:

                    mtext = self if len(info) == 1 else None
                    x = x + posx
                    y = y + posy
                    if renderer.flipy():
                        y = canvash - y
                    clean_line, ismath = self.is_math_text(line)

                    if self.get_path_effects():
                        from matplotlib.patheffects import PathEffectRenderer
                        textrenderer = PathEffectRenderer(
                                            self.get_path_effects(), renderer)  # noqa
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
            if usetex is None:
                self._usetex = None
            else:
                self._usetex = bool(usetex)
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

    # Monkey-patch scattertext onto Axes
    Axes.scattertext = scattertext
