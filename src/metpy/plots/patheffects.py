#  Copyright (c) 2020,2022,2023 MetPy Developers.
#  Distributed under the terms of the BSD 3-Clause License.
#  SPDX-License-Identifier: BSD-3-Clause
"""Add effects to matplotlib paths."""
from functools import cached_property
import itertools

import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.patheffects as mpatheffects
import matplotlib.transforms as mtransforms
import numpy as np

from ..package_tools import Exporter

exporter = Exporter(globals())


class Front(mpatheffects.AbstractPathEffect):
    """Provide base functionality for plotting fronts as a patheffect.

    These are plotted as symbol markers tangent to the path.

    """

    _symbol = mpath.Path([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]],
                         [mpath.Path.MOVETO, mpath.Path.LINETO,
                          mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY])

    def __init__(self, color, size=10, spacing=1, flip=False, filled=True):
        """Initialize the front path effect.

        Parameters
        ----------
        color : str or tuple[float]
            Color to use for the effect.
        size : int or float
            The size of the markers to plot in points. Defaults to 10.
        spacing : int or float
            The spacing between markers in normalized coordinates. Defaults to 1.
        flip : bool
            Whether the symbol should be flipped to the other side of the path. Defaults
            to `False`.
        filled : bool
            Whether the symbol should be filled with the color. Defaults to `True`.

        """
        super().__init__()
        self.size = size
        self.spacing = spacing
        self.color = mcolors.to_rgba(color)
        self.flip = flip
        self.filled = filled
        self._symbol_width = None

    @cached_property
    def symbol_width(self):
        """Return the width of the symbol being plotted."""
        return self._symbol.get_extents().width

    def _step_size(self, renderer):
        """Return the length of the step between markers in pixels."""
        return (self.symbol_width + self.spacing) * self._size_pixels(renderer)

    def _size_pixels(self, renderer):
        """Return the size of the marker in pixels."""
        return renderer.points_to_pixels(self.size)

    @staticmethod
    def _process_path(path, path_trans):
        """Transform the main path into pixel coordinates; calculate the needed components."""
        path_points = path.transformed(path_trans).interpolated(500).vertices
        deltas = (path_points[1:] - path_points[:-1]).T
        pt_offsets = np.concatenate(([0], np.hypot(*deltas).cumsum()))
        angles = np.arctan2(deltas[-1], deltas[0])
        return path_points, pt_offsets, angles

    def _get_marker_locations(self, segment_offsets, renderer):
        # Calculate increment of path length occupied by each marker drawn
        inc = self._step_size(renderer)

        # Find out how many markers that will accommodate, as well as remainder space
        num, leftover = divmod(segment_offsets[-1], inc)

        # Find the offset for each marker along the path length. We center along
        # the path by adding half of the remainder. The offset is also centered within
        # the marker by adding half of the marker increment
        marker_offsets = np.arange(num) * inc + (leftover + inc) / 2.

        # Find the location of these offsets within the total offset within each
        # path segment; subtracting 1 gives us the left point of the path rather
        # than the last. We then need to adjust for any offsets that are <= the first
        # point of the path (just set them to index 0).
        inds = np.searchsorted(segment_offsets, marker_offsets) - 1
        inds[inds < 0] = 0

        # Return the indices to the proper segment and the offset within that segment
        return inds, marker_offsets - segment_offsets[inds]

    def _override_gc(self, renderer, gc, **kwargs):
        ret = renderer.new_gc()
        ret.copy_properties(gc)
        ret.set_joinstyle('miter')
        ret.set_capstyle('butt')
        return self._update_gc(ret, kwargs)

    def _get_symbol_transform(self, renderer, offset, line_shift, angle, start):
        scalex = self._size_pixels(renderer)
        scaley, line_shift = (-scalex, -line_shift) if self.flip else (scalex, line_shift)
        return mtransforms.Affine2D().scale(scalex, scaley).translate(
            offset - self.symbol_width * self._size_pixels(renderer) / 2,
            line_shift).rotate(angle).translate(*start)

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the given path."""
        # Set up a new graphics context for rendering the front effect; override the color
        gc0 = self._override_gc(renderer, gc, foreground=self.color)

        # Get the information we need for drawing along the path
        starts, offsets, angles = self._process_path(path, affine)

        # Figure out what segments the markers should be drawn upon and how
        # far within that segment the markers will appear.
        segment_indices, marker_offsets = self._get_marker_locations(offsets, renderer)

        # Draw the original path
        renderer.draw_path(gc0, path, affine, rgbFace)  # noqa: N803

        # Need to account for the line width in order to properly draw symbols at line edge
        line_shift = renderer.points_to_pixels(gc.get_linewidth()) / 2

        # Loop over all the markers to draw
        for ind, marker_offset in zip(segment_indices, marker_offsets):
            sym_trans = self._get_symbol_transform(renderer, marker_offset, line_shift,
                                                   angles[ind], starts[ind])
            renderer.draw_path(gc0, self._symbol, sym_trans,
                               self.color if self.filled else None)

        gc0.restore()


class Frontogenesis(Front):
    """Provide base functionality for plotting strengthening fronts as a patheffect.

    These are plotted as symbol markers tangent to the path.

    """

    def __init__(self, color, size=10, spacing=1, flip=False):
        """Initialize the frontogenesis path effect.

        Parameters
        ----------
        color : str or tuple[float]
            Color to use for the effect.
        size : int or float
            The size of the markers to plot in points. Defaults to 10.
        spacing : int or float
            The spacing between markers in normalized coordinates. Defaults to 1.
        flip : bool
            Whether the symbol should be flipped to the other side of the path. Defaults
            to `False`.

        """
        super().__init__(color, size, spacing, flip)
        self._padding = 4

    def _step_size(self, renderer):
        """Return the length of the step between markers in pixels."""
        return (self.symbol_width + self.spacing + self._padding) * self._size_pixels(renderer)

    def _get_path_locations(self, segment_offsets, renderer):
        # Calculate increment of path length occupied by each marker drawn
        inc = self._step_size(renderer)

        # Find out how many markers that will accommodate, as well as remainder space
        num, leftover = divmod(segment_offsets[-1], inc)

        # Find the offset for each marker along the path length. We center along
        # the path by adding half of the remainder. The offset is also centered within
        # the marker by adding half of the marker increment
        marker_offsets = np.arange(num) * inc + (leftover + inc) / 2.

        # Do the same for path segments
        start_offsets = marker_offsets - 0.33 * inc
        end_offsets = marker_offsets + 0.33 * inc

        # Find the location of these offsets within the total offset within each
        # path segment; subtracting 1 gives us the left point of the path rather
        # than the last. We then need to adjust for any offsets that are <= the first
        # point of the path (just set them to index 0).
        inds = np.searchsorted(segment_offsets, marker_offsets) - 1
        inds[inds < 0] = 0

        start_inds = np.searchsorted(segment_offsets, start_offsets) - 1
        start_inds[start_inds < 0] = 0

        end_inds = np.searchsorted(segment_offsets, end_offsets) - 1
        end_inds[start_inds < 0] = 0

        # Return the indices to the proper segment and the offset within that segment
        return start_inds, end_inds, inds, marker_offsets - segment_offsets[inds]

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the given path."""
        # Set up a new graphics context for rendering the front effect; override the color
        gc0 = self._override_gc(renderer, gc, foreground=self.color)

        # Get the information we need for drawing along the path
        starts, offsets, angles = self._process_path(path, affine)

        # Figure out what segments the markers should be drawn upon, how
        # far within that segment the markers will appear, and the segment bounds.
        (segment_starts, segment_ends,
         segment_indices, marker_offsets) = self._get_path_locations(offsets, renderer)

        # Need to account for the line width in order to properly draw symbols at line edge
        line_shift = renderer.points_to_pixels(gc.get_linewidth()) / 2

        # Loop over all the segments to draw
        for start_path, end_path in zip(segment_starts, segment_ends):
            renderer.draw_path(gc0, mpath.Path(starts[start_path:end_path]),
                               mtransforms.Affine2D(), None)

        # Loop over all the markers to draw
        for ind, marker_offset in zip(segment_indices, marker_offsets):
            sym_trans = self._get_symbol_transform(renderer, marker_offset, line_shift,
                                                   angles[ind], starts[ind])

            renderer.draw_path(gc0, self._symbol, sym_trans, self.color)

        gc0.restore()


class Frontolysis(Front):
    """Provide base functionality for plotting weakening fronts as a patheffect.

    These are plotted as symbol markers tangent to the path.

    """

    def __init__(self, color, size=10, spacing=1, flip=False):
        """Initialize the frontolysis path effect.

        Parameters
        ----------
        color : str or tuple[float]
            Color to use for the effect.
        size : int or float
            The size of the markers to plot in points. Defaults to 10.
        spacing : int or float
            The spacing between markers in normalized coordinates. Defaults to 1.
        flip : bool
            Whether the symbol should be flipped to the other side of the path. Defaults
            to `False`.

        """
        super().__init__(color, size, spacing, flip)
        self._padding = 4

    def _step_size(self, renderer):
        """Return the length of the step between markers in pixels."""
        return (self.symbol_width + self.spacing + self._padding) * self._size_pixels(renderer)

    def _get_path_locations(self, segment_offsets, renderer):
        # Calculate increment of path length occupied by each marker drawn
        inc = self._step_size(renderer)

        # Find out how many markers that will accommodate, as well as remainder space
        num, leftover = divmod(segment_offsets[-1], inc)

        # Find the offset for each marker along the path length. We center along
        # the path by adding half of the remainder. The offset is also centered within
        # the marker by adding half of the marker increment
        marker_offsets = np.arange(num) * inc + (leftover + inc) / 2.

        # Do the same for path segments
        start_offsets = marker_offsets - 0.33 * inc
        end_offsets = marker_offsets + 0.33 * inc

        # Find the location of these offsets within the total offset within each
        # path segment; subtracting 1 gives us the left point of the path rather
        # than the last. We then need to adjust for any offsets that are <= the first
        # point of the path (just set them to index 0).
        inds = np.searchsorted(segment_offsets, marker_offsets) - 1
        inds[inds < 0] = 0

        start_inds = np.searchsorted(segment_offsets, start_offsets) - 1
        start_inds[start_inds < 0] = 0

        end_inds = np.searchsorted(segment_offsets, end_offsets) - 1
        end_inds[start_inds < 0] = 0

        # Return the indices to the proper segment and the offset within that segment
        return start_inds, end_inds, inds, marker_offsets - segment_offsets[inds]

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the given path."""
        # Set up a new graphics context for rendering the front effect; override the color
        gc0 = self._override_gc(renderer, gc, foreground=self.color)

        # Get the information we need for drawing along the path
        starts, offsets, angles = self._process_path(path, affine)

        # Figure out what segments the markers should be drawn upon, how
        # far within that segment the markers will appear, and the segment bounds.
        (segment_starts, segment_ends,
         segment_indices, marker_offsets) = self._get_path_locations(offsets, renderer)

        # Need to account for the line width in order to properly draw symbols at line edge
        line_shift = renderer.points_to_pixels(gc.get_linewidth()) / 2

        # Loop over all the segments to draw
        for start_path, end_path in zip(segment_starts, segment_ends):
            renderer.draw_path(gc0, mpath.Path(starts[start_path:end_path]),
                               mtransforms.Affine2D(), None)

        # Loop over all the markers to draw
        for ind, marker_offset in zip(segment_indices[::2], marker_offsets[::2]):
            sym_trans = self._get_symbol_transform(renderer, marker_offset, line_shift,
                                                   angles[ind], starts[ind])

            renderer.draw_path(gc0, self._symbol, sym_trans, self.color)

        gc0.restore()


@exporter.export
class ScallopedStroke(mpatheffects.AbstractPathEffect):
    """A line-based PathEffect which draws a path with a scalloped style.

    The spacing, length, and side of the scallops can be controlled. This implementation is
    based off of :class:`matplotlib.patheffects.TickedStroke`.
    """

    def __init__(self, offset=(0, 0), spacing=10.0, side='left', length=1.15, **kwargs):
        """Create a scalloped path effect.

        Parameters
        ----------
        offset : (float, float)
            The (x, y) offset to apply to the path, in points. Defaults to no offset.
        spacing : float
            The spacing between ticks in points. Defaults to 10.0.
        side : str
            Side of the path scallops appear on from the reference of
            walking along the curve. Options are left and right. Defaults to ``'left'``.
        length : float
            The length of the tick relative to spacing. Defaults to 1.414.
        kwargs :
            Extra keywords are stored and passed through to
            `~matplotlib.renderer.GraphicsContextBase`.
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

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the path with updated gc."""
        # Do not modify the input! Use copy instead.
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        gc0 = self._update_gc(gc0, self._gc)
        trans = affine + self._offset_transform(renderer)

        theta = -np.radians(self._angle)
        trans_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])

        # Convert spacing parameter to pixels.
        spacing_px = renderer.points_to_pixels(self._spacing)

        # Transform before evaluation because to_polygons works at resolution
        # of one -- assuming it is working in pixel space.
        transpath = affine.transform_path(path)

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
            s_tick = np.linspace(0, s_total, num)

            # Find points along the parameterized curve
            x_tick = np.interp(s_tick, s, x)
            y_tick = np.interp(s_tick, s, y)

            # Find unit vectors in local direction of curve
            delta_s = self._spacing * .001
            u = (np.interp(s_tick + delta_s, s, x) - x_tick) / delta_s
            v = (np.interp(s_tick + delta_s, s, y) - y_tick) / delta_s

            # Handle slope of end point
            if (x_tick[-1], y_tick[-1]) == (x_tick[0], y_tick[0]):  # periodic
                u[-1] = u[0]
                v[-1] = v[0]
            else:
                u[-1] = u[-2]
                v[-1] = v[-2]

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

            # Build path vertices that will define control points of the bezier curves
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
            codes = np.tile([mpath.Path.LINETO, mpath.Path.CURVE4,
                             mpath.Path.CURVE4, mpath.Path.CURVE4], nverts)
            codes[0] = mpath.Path.MOVETO

            # Construct and draw resulting path
            h = mpath.Path(verts, codes)

            # Transform back to data space during render
            renderer.draw_path(gc0, h, affine.inverted() + trans, rgbFace)

        gc0.restore()


@exporter.export
class ColdFront(Front):
    """Draw a path as a cold front, with (default blue) pips/triangles along the path."""

    _symbol = mpath.Path([[0, 0], [1, 1], [2, 0], [0, 0]],
                         [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
                          mpath.Path.CLOSEPOLY])

    def __init__(self, color='blue', **kwargs):
        super().__init__(color, **kwargs)


@exporter.export
class ColdFrontogenesis(Frontogenesis):
    """Draw a path as a strengthening cold."""

    _symbol = mpath.Path([[0, 0], [1, 1], [2, 0], [0, 0]],
                         [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
                          mpath.Path.CLOSEPOLY])

    def __init__(self, color='blue', **kwargs):
        super().__init__(color, **kwargs)


@exporter.export
class ColdFrontolysis(Frontolysis):
    """Draw a path as a weakening cold front."""

    _symbol = mpath.Path([[0, 0], [1, 1], [2, 0], [0, 0]],
                         [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
                          mpath.Path.CLOSEPOLY])

    def __init__(self, color='blue', **kwargs):
        super().__init__(color, **kwargs)


@exporter.export
class Dryline(Front):
    """Draw a path as a dryline with (default brown) scallops along the path."""

    _symbol = mpath.Path.wedge(0, 180).transformed(mtransforms.Affine2D().translate(1, 0))

    def __init__(self, color='brown', spacing=0.144, filled=False, **kwargs):
        super().__init__(color, spacing=spacing, filled=filled, **kwargs)


@exporter.export
class WarmFront(Front):
    """Draw a path as a warm front with (default red) scallops along the path."""

    _symbol = mpath.Path.wedge(0, 180).transformed(mtransforms.Affine2D().translate(1, 0))

    def __init__(self, color='red', **kwargs):
        super().__init__(color, **kwargs)


@exporter.export
class WarmFrontogenesis(Frontogenesis):
    """Draw a path as a strengthening warm front."""

    _symbol = mpath.Path.wedge(0, 180).transformed(mtransforms.Affine2D().translate(1, 0))

    def __init__(self, color='red', **kwargs):
        super().__init__(color, **kwargs)


@exporter.export
class WarmFrontolysis(Frontolysis):
    """Draw a path as a weakening warm front."""

    _symbol = mpath.Path.wedge(0, 180).transformed(mtransforms.Affine2D().translate(1, 0))

    def __init__(self, color='red', **kwargs):
        super().__init__(color, **kwargs)


@exporter.export
class OccludedFront(Front):
    """Draw an occluded front with (default purple) pips and scallops along the path."""

    def __init__(self, color='purple', **kwargs):
        self._symbol_cycle = None
        super().__init__(color, **kwargs)

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the given path."""
        self._symbol_cycle = None
        return super().draw_path(renderer, gc, path, affine, rgbFace)  # noqa: N803

    @property
    def _symbol(self):
        """Return the proper symbol to draw; alternatives between scallop and pip/triangle."""
        if self._symbol_cycle is None:
            self._symbol_cycle = itertools.cycle([WarmFront._symbol, ColdFront._symbol])
        return next(self._symbol_cycle)


@exporter.export
class OccludedFrontogenesis(Frontogenesis):
    """Draw a strengthening occluded front."""

    def __init__(self, color='purple', **kwargs):
        self._symbol_cycle = None
        super().__init__(color, **kwargs)

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the given path."""
        self._symbol_cycle = None
        return super().draw_path(renderer, gc, path, affine, rgbFace)  # noqa: N803

    @property
    def _symbol(self):
        """Return the proper symbol to draw; alternatives between scallop and pip/triangle."""
        if self._symbol_cycle is None:
            self._symbol_cycle = itertools.cycle([WarmFrontogenesis._symbol,
                                                  ColdFrontogenesis._symbol])
        return next(self._symbol_cycle)


@exporter.export
class OccludedFrontolysis(Frontolysis):
    """Draw a weakening occluded front."""

    def __init__(self, color='purple', **kwargs):
        self._symbol_cycle = None
        super().__init__(color, **kwargs)

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the given path."""
        self._symbol_cycle = None
        return super().draw_path(renderer, gc, path, affine, rgbFace)  # noqa: N803

    @property
    def _symbol(self):
        """Return the proper symbol to draw; alternatives between scallop and pip/triangle."""
        if self._symbol_cycle is None:
            self._symbol_cycle = itertools.cycle([WarmFrontolysis._symbol,
                                                  ColdFrontolysis._symbol])
        return next(self._symbol_cycle)


@exporter.export
class RidgeAxis(mpatheffects.AbstractPathEffect):
    """A line-based PathEffect which draws a path with a sawtooth-wave style.

    This line style is frequently used to represent a ridge axis.
    """

    def __init__(self, color='black', spacing=12.0, length=0.5):
        """Create ridge axis path effect.

        Parameters
        ----------
        color : str
            Color to use for the effect.
        spacing : float
            The spacing between ticks in points. Default is 12.
        length : float
            The length of the tick relative to spacing. Default is 0.5.

        """
        self._spacing = spacing
        self._angle = 90.0
        self._length = length
        self._color = color

    def _override_gc(self, renderer, gc, **kwargs):
        ret = renderer.new_gc()
        ret.copy_properties(gc)
        ret.set_joinstyle('miter')
        ret.set_capstyle('butt')
        return self._update_gc(ret, kwargs)

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):  # noqa: N803
        """Draw the path with updated gc."""
        # Do not modify the input! Use copy instead.
        gc0 = self._override_gc(renderer, gc, foreground=self._color)

        theta = -np.radians(self._angle)
        trans_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])

        # Convert spacing parameter to pixels.
        spacing_px = renderer.points_to_pixels(self._spacing)

        # Transform before evaluation because to_polygons works at resolution
        # of one -- assuming it is working in pixel space.
        transpath = affine.transform_path(tpath)

        # Evaluate path to straight line segments that can be used to
        # construct line ticks.
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
            # Pick parameter values for ticks.
            s_tick = np.linspace(spacing_px / 2, s_total - spacing_px / 2, num)

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

            # Rotate and scale unit vector into tick vector
            dxy1 = np.dot(uv[0::2], trans_matrix) * self._length * spacing_px
            dxy2 = np.dot(uv[1::2], trans_matrix.T) * self._length * spacing_px

            # Build tick endpoints
            x_end = np.zeros(num)
            y_end = np.zeros(num)
            x_end[0::2] = x_tick[0::2] + dxy1[:, 0]
            x_end[1::2] = x_tick[1::2] + dxy2[:, 0]
            y_end[0::2] = y_tick[0::2] + dxy1[:, 1]
            y_end[1::2] = y_tick[1::2] + dxy2[:, 1]

            # Interleave ticks to form Path vertices
            xyt = np.empty((num, 2), dtype=x_tick.dtype)
            xyt[:, 0] = x_end
            xyt[:, 1] = y_end

            # Build up vector of Path codes
            codes = np.concatenate([[mpath.Path.MOVETO], [mpath.Path.LINETO] * (len(xyt) - 1)])

            # Construct and draw resulting path
            h = mpath.Path(xyt, codes)

            # Transform back to data space during render
            renderer.draw_path(gc0, h, affine.inverted() + affine, rgbFace)  # noqa: N803

        gc0.restore()


@exporter.export
class Squall(mpatheffects.AbstractPathEffect):
    """Squall line path effect."""

    symbol = mpath.Path.circle((0, 0), radius=4)

    def __init__(self, color='black', spacing=75):
        """Initialize the squall line path effect.

        Parameters
        ----------
        color : str
            Color to use for the effect.
        spacing : float
            Spacing between symbols along path (in points).

        """
        self.marker_margin = 10
        self.spacing = spacing
        self.color = mcolors.to_rgba(color)
        self._symbol_width = None

    @staticmethod
    def _process_path(path, path_trans):
        """Transform the main path into pixel coordinates; calculate the needed components."""
        path_points = path.transformed(path_trans).interpolated(500).vertices
        deltas = (path_points[1:] - path_points[:-1]).T
        pt_offsets = np.concatenate(([0], np.hypot(*deltas).cumsum()))
        return path_points, pt_offsets

    def _override_gc(self, renderer, gc, **kwargs):
        ret = renderer.new_gc()
        ret.copy_properties(gc)
        ret.set_joinstyle('miter')
        ret.set_capstyle('butt')
        return self._update_gc(ret, kwargs)

    def _get_object_locations(self, segment_offsets, renderer):
        # Calculate increment of path length
        inc = renderer.points_to_pixels(self.spacing)
        margin = renderer.points_to_pixels(self.marker_margin)

        # Find out how many markers that will accommodate, as well as remainder space
        num, leftover = divmod(segment_offsets[-1], inc)

        # Find the offset for each marker along the path length. We center along
        # the path by adding half of the remainder. The offset is also centered within
        # the marker by adding half of the marker increment
        first_marker = np.arange(num) * inc - 0.5 * margin + (leftover + inc) / 2.
        second_marker = np.arange(num) * inc + 0.5 * margin + (leftover + inc) / 2.
        marker_offsets = np.sort(np.concatenate([first_marker, second_marker]))

        # Do the same for path segments
        first = segment_offsets[0]
        last = segment_offsets[-1]
        path_offset_1 = np.arange(num) * inc - 1.5 * margin + (leftover + inc) / 2
        path_offset_2 = np.arange(num) * inc + 1.5 * margin + (leftover + inc) / 2
        path_offsets = np.sort(np.concatenate(
            [[first], path_offset_1, path_offset_2, [last]]
        ))

        # Find the location of these offsets within the total offset within each
        # path segment; subtracting 1 gives us the left point of the path rather
        # than the last. We then need to adjust for any offsets that are <= the first
        # point of the path (just set them to index 0).
        marker_inds = np.searchsorted(segment_offsets, marker_offsets) - 1
        marker_inds[marker_inds < 0] = 0

        # Do the same for path segments
        path_inds = np.searchsorted(segment_offsets, path_offsets) - 1
        path_inds[path_inds < 0] = 0

        # Return the indices to the proper segment and the offset within that segment
        return marker_inds, path_inds

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw path."""
        # Set up a new graphics context for rendering the front effect; override the color
        gc0 = self._override_gc(renderer, gc, foreground=self.color)

        # Get the information we need for drawing along the path
        starts, offsets = self._process_path(path, affine)

        # Figure out what segments the markers should be drawn upon and how
        # far within that segment the markers will appear.
        marker_indices, path_indices = self._get_object_locations(offsets, renderer)

        base_trans = mtransforms.Affine2D()

        # Loop over the segmented path
        ipath = path.interpolated(500).vertices
        for i in range(0, len(path_indices) - 1, 2):
            start = path_indices[i]
            stop = path_indices[i + 1]
            n = stop - start
            spath = mpath.Path(
                ipath[start:stop],
                [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (n - 1)
            )
            renderer.draw_path(gc0, spath, affine, None)

        # Loop over all the markers to draw
        for ind in marker_indices:
            sym_trans = base_trans.frozen().translate(*starts[ind])
            renderer.draw_path(gc0, self.symbol, sym_trans, self.color)

        gc0.restore()


@exporter.export
class StationaryFront(Front):
    """Draw a stationary front as alternating cold and warm front segments."""

    _symbol = WarmFront._symbol.transformed(mtransforms.Affine2D().scale(1, -1))
    _symbol2 = ColdFront._symbol

    def __init__(self, colors=('red', 'blue'), **kwargs):
        """Initialize a stationary front path effect.

        This effect alternates between a warm front and cold front symbol.

        Parameters
        ----------
        colors : Sequence[str] or Sequence[tuple[float]]
            Matplotlib color identifiers to cycle between on the two different front styles.
            Defaults to alternating red and blue.
        size : int or float
            The size of the markers to plot in points. Defaults to 10.
        spacing : int or float
            The spacing between markers in normalized coordinates. Defaults to 1.
        flip : bool
            Whether the symbols should be flipped to the other side of the path. Defaults
            to `False`.

        """
        self._colors = list(map(mcolors.to_rgba, colors))
        super().__init__(color=self._colors[0], **kwargs)

    def _get_path_segment_ends(self, segment_offsets, renderer):
        # Calculate increment of path length occupied by each marker drawn
        inc = self._step_size(renderer)

        # Find out how many markers that will accommodate, as well as remainder space
        num, leftover = divmod(segment_offsets[-1], inc)

        # Find the offset for each path segment end. We center along
        # the entire path by adding half of the remainder.
        path_offsets = np.arange(1, num + 1) * inc + leftover / 2.

        # Find the location of these offsets within the total offset within each
        # path segment; subtracting 1 gives us the left point of the path rather
        # than the last.
        return np.searchsorted(segment_offsets, path_offsets)

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the given path."""
        gcs = [self._override_gc(renderer, gc, foreground=color) for color in self._colors]
        self._gc_cycle = itertools.cycle(gcs)
        self._symbol_cycle = itertools.cycle([self._symbol, self._symbol2])
        self._color_cycle = itertools.cycle(self._colors)

        # Get the information we need for drawing along the path
        starts, offsets, angles = self._process_path(path, affine)

        # Figure out what segments the markers should be drawn upon and how
        # far within that segment the markers will appear.
        segment_indices, marker_offsets = self._get_marker_locations(offsets, renderer)
        end_path_inds = self._get_path_segment_ends(offsets, renderer)
        start_path_inds = np.concatenate([[0], end_path_inds[:-1]])

        # Need to account for the line width in order to properly draw symbols at line edge
        line_shift = -renderer.points_to_pixels(gc.get_linewidth()) / 2

        # Loop over all the markers to draw
        for ind, start_path, end_path, marker_offset in zip(segment_indices, start_path_inds,
                                                            end_path_inds, marker_offsets):
            sym_trans = self._get_symbol_transform(renderer, marker_offset, line_shift,
                                                   angles[ind], starts[ind])
            gc = next(self._gc_cycle)
            color = next(self._color_cycle)
            symbol = next(self._symbol_cycle)

            renderer.draw_path(gc, symbol, sym_trans, color)
            renderer.draw_path(gc, mpath.Path(starts[start_path:end_path]),
                               mtransforms.Affine2D(), None)
            line_shift *= -1

        gcs[0].restore()


@exporter.export
class StationaryFrontogenesis(Frontogenesis):
    """Draw a strengthening stationary front."""

    _symbol = WarmFront._symbol
    _symbol2 = ColdFront._symbol.transformed(mtransforms.Affine2D().scale(1, -1))

    def __init__(self, colors=('red', 'blue'), **kwargs):
        """Initialize a strengthening stationary front path effect.

        This effect alternates between a warm front and cold front symbol.

        Parameters
        ----------
        colors : Sequence[str] or Sequence[tuple[float]]
            Matplotlib color identifiers to cycle between on the two different front styles.
            Defaults to alternating red and blue.
        size : int or float
            The size of the markers to plot in points. Defaults to 10.
        spacing : int or float
            The spacing between markers in normalized coordinates. Defaults to 1.
        flip : bool
            Whether the symbols should be flipped to the other side of the path. Defaults
            to `False`.

        """
        self._colors = list(map(mcolors.to_rgba, colors))
        super().__init__(color=self._colors[0], **kwargs)

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the given path."""
        gcs = [self._override_gc(renderer, gc, foreground=color) for color in self._colors]
        self._gc_cycle = itertools.cycle(gcs)
        self._symbol_cycle = itertools.cycle([self._symbol, self._symbol2])
        self._color_cycle = itertools.cycle(self._colors)

        # Get the information we need for drawing along the path
        starts, offsets, angles = self._process_path(path, affine)

        # Figure out what segments the markers should be drawn upon, how
        # far within that segment the markers will appear, and the segment bounds.
        (segment_starts, segment_ends,
         segment_indices, marker_offsets) = self._get_path_locations(offsets, renderer)

        # Need to account for the line width in order to properly draw symbols at line edge
        line_shift = renderer.points_to_pixels(gc.get_linewidth()) / 2

        # Loop over all the markers to draw
        for ind, start_path, end_path, marker_offset in zip(segment_indices, segment_starts,
                                                            segment_ends, marker_offsets):
            sym_trans = self._get_symbol_transform(renderer, marker_offset, line_shift,
                                                   angles[ind], starts[ind])
            gc = next(self._gc_cycle)
            color = next(self._color_cycle)
            symbol = next(self._symbol_cycle)

            renderer.draw_path(gc, symbol, sym_trans, color)
            renderer.draw_path(gc, mpath.Path(starts[start_path:end_path]),
                               mtransforms.Affine2D(), None)
            line_shift *= -1

        gcs[0].restore()


@exporter.export
class StationaryFrontolysis(Frontolysis):
    """Draw a weakening stationary front.."""

    _symbol = WarmFront._symbol
    _symbol2 = ColdFront._symbol.transformed(mtransforms.Affine2D().scale(1, -1))

    def __init__(self, colors=('red', 'blue'), **kwargs):
        """Initialize a weakening stationary front path effect.

        This effect alternates between a warm front and cold front symbol.

        Parameters
        ----------
        colors : Sequence[str] or Sequence[tuple[float]]
            Matplotlib color identifiers to cycle between on the two different front styles.
            Defaults to alternating red and blue.
        size : int or float
            The size of the markers to plot in points. Defaults to 10.
        spacing : int or float
            The spacing between markers in normalized coordinates. Defaults to 1.
        flip : bool
            Whether the symbols should be flipped to the other side of the path. Defaults
            to `False`.

        """
        self._colors = list(map(mcolors.to_rgba, colors))
        self._segment_colors = [
            (self._colors[0], self._colors[0]),
            (self._colors[0], self._colors[1]),
            (self._colors[1], self._colors[1]),
            (self._colors[1], self._colors[0])
        ]
        super().__init__(color=self._colors[0], **kwargs)

    def draw_path(self, renderer, gc, path, affine, rgbFace=None):  # noqa: N803
        """Draw the given path."""
        gcs = [self._override_gc(renderer, gc, foreground=color) for color in self._colors]
        self._gc_cycle = itertools.cycle(gcs)
        self._symbol_cycle = itertools.cycle([self._symbol, self._symbol2])
        self._color_cycle = itertools.cycle(self._colors)
        self._segment_cycle = itertools.cycle(self._segment_colors)

        # Get the information we need for drawing along the path
        starts, offsets, angles = self._process_path(path, affine)

        # Figure out what segments the markers should be drawn upon, how
        # far within that segment the markers will appear, and the segment bounds.
        (segment_starts, segment_ends,
         segment_indices, marker_offsets) = self._get_path_locations(offsets, renderer)

        # Need to account for the line width in order to properly draw symbols at line edge
        line_shift = renderer.points_to_pixels(gc.get_linewidth()) / 2

        # Loop over all the markers to draw
        for ind, marker_offset in zip(segment_indices[::2], marker_offsets[::2]):
            sym_trans = self._get_symbol_transform(renderer, marker_offset, line_shift,
                                                   angles[ind], starts[ind])
            gc = next(self._gc_cycle)
            color = next(self._color_cycle)
            symbol = next(self._symbol_cycle)

            renderer.draw_path(gc, symbol, sym_trans, color)

            line_shift *= -1

        for start_path, mid_path, end_path in zip(segment_starts,
                                                  segment_indices,
                                                  segment_ends):
            color1, color2 = next(self._segment_cycle)

            gcx = self._override_gc(renderer, gc, foreground=mcolors.to_rgb(color1))
            renderer.draw_path(gcx, mpath.Path(starts[start_path:mid_path]),
                               mtransforms.Affine2D(), None)

            gcx = self._override_gc(renderer, gc, foreground=mcolors.to_rgb(color2))
            renderer.draw_path(gcx, mpath.Path(starts[mid_path:end_path]),
                               mtransforms.Affine2D(), None)

        gcs[0].restore()
