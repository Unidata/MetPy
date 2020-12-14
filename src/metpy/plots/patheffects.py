# Copyright (c) 2016,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for matplotlib path effects."""

from matplotlib.path import Path
from matplotlib.patheffects import _subclass_with_normal, AbstractPathEffect
import numpy as np

from ..package_tools import Exporter

exporter = Exporter(globals())


@exporter.export
class ScallopedStroke(AbstractPathEffect):
    """A line-based PathEffect which draws a path with a scalloped style.

    The spacing, length, and side of the scallops can be controlled. This implementation is
    based off of :class:`matplotlib.patheffects.TickedStroke`.
    """

    def __init__(self, offset=(0, 0), spacing=10.0, side='left', length=1.15, **kwargs):
        """Create a scalloped path effect.

        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, in points.
        spacing : float, default: 10.0
            The spacing between ticks in points.
        side : str, default: left
            Side of the path scallops appear on from the reference of
            walking along the curve. Options are left and right.
        length : float, default: 1.414
            The length of the tick relative to spacing.
        kwargs :
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.
        """
        super().__init__(offset)

        self._spacing = spacing
        if side == 'left':
            self._angle = 90
        elif side == 'right':
            self._angle = -90
        else:
            raise ValueError('Side must be left or right.')
        self._length = length
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """Draw the path with updated gc."""
        # Do not modify the input! Use copy instead.
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        gc0 = self._update_gc(gc0, self._gc)
        try:
            # For matplotlib >= 3.2
            trans = affine + self._offset_transform(renderer)
        except TypeError:
            # For matplotlib < 3.2
            trans = self._offset_transform(renderer, affine)

        theta = -np.radians(self._angle)
        trans_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])

        # Convert spacing parameter to pixels.
        spacing_px = renderer.points_to_pixels(self._spacing)

        # Transform before evaluation because to_polygons works at resolution
        # of one -- assuming it is working in pixel space.
        transpath = affine.transform_path(tpath)

        # Evaluate path to straight line segments that can be used to
        # construct line scallops.
        polys = transpath.to_polygons(closed_only=False)

        for p in polys:
            x = p[:, 0]
            y = p[:, 1]

            # Can not interpolate points or draw line if only one point in
            # polyline.
            if x.size < 2:
                continue

            # Find distance between points on the line
            ds = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])

            # Build parametric coordinate along curve
            s = np.concatenate(([0.0], np.cumsum(ds)))
            s_total = s[-1]

            num = int(np.ceil(s_total / spacing_px)) - 1
            # Pick parameter values for scallops.
            s_tick = np.linspace(0, s_total - 1e-5, num)

            # Find points along the parameterized curve
            x_tick = np.interp(s_tick, s, x)
            y_tick = np.interp(s_tick, s, y)

            # Find unit vectors in local direction of curve
            delta_s = self._spacing * .001
            u = (np.interp(s_tick + delta_s, s, x) - x_tick) / delta_s
            v = (np.interp(s_tick + delta_s, s, y) - y_tick) / delta_s

            # Normalize slope into unit slope vector.
            n = np.hypot(u, v)
            mask = n == 0
            n[mask] = 1.0

            uv = np.array([u / n, v / n]).T
            uv[mask] = np.array([0, 0]).T

            # Rotate and scale unit vector
            dxy = np.dot(uv, trans_matrix) * self._length * spacing_px

            # Build endpoints
            x_end = x_tick + dxy[:, 0]
            y_end = y_tick + dxy[:, 1]

            # Interleave ticks to form Path vertices
            xyt = np.empty((2 * num, 2), dtype=x_tick.dtype)
            xyt[0::2, 0] = x_tick
            xyt[1::2, 0] = x_end
            xyt[0::2, 1] = y_tick
            xyt[1::2, 1] = y_end

            # Build path verticies that will define control points
            # the bezier curves
            verts = []
            i = 0
            nverts = 0
            while i < len(xyt) - 2:
                verts.append(xyt[i, :])
                verts.append(xyt[i + 1, :])
                verts.append(xyt[i + 3, :])
                verts.append(xyt[i + 2, :])
                nverts += 1
                i += 2

            # Build up vector of Path codes
            codes = np.tile([Path.LINETO, Path.CURVE4,
                             Path.CURVE4, Path.CURVE4], nverts)
            codes[0] = Path.MOVETO

            # Construct and draw resulting path
            h = Path(verts, codes)

            # Transform back to data space during render
            renderer.draw_path(gc0, h, affine.inverted() + trans, rgbFace)

        gc0.restore()


withScallopedStroke = _subclass_with_normal(effect_class=ScallopedStroke)
