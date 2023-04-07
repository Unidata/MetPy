#  Copyright (c) 2020,2022 MetPy Developers.
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
                         [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
                          mpath.Path.LINETO, mpath.Path.CLOSEPOLY])

    def __init__(self, color, size=10, spacing=1, flip=False):
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

        """
        super().__init__()
        self.size = size
        self.spacing = spacing
        self.color = mcolors.to_rgba(color)
        self.flip = flip
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
        renderer.draw_path(gc0, path, affine, rgbFace)

        # Need to account for the line width in order to properly draw symbols at line edge
        line_shift = renderer.points_to_pixels(gc.get_linewidth()) / 2

        # Loop over all the markers to draw
        for ind, marker_offset in zip(segment_indices, marker_offsets):
            sym_trans = self._get_symbol_transform(renderer, marker_offset, line_shift,
                                                   angles[ind], starts[ind])
            renderer.draw_path(gc0, self._symbol, sym_trans, self.color)

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
class WarmFront(Front):
    """Draw a path as a warm front with (default red) scallops along the path."""

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
        return super().draw_path(renderer, gc, path, affine, rgbFace)

    @property
    def _symbol(self):
        """Return the proper symbol to draw; alternatives between scallop and pip/triangle."""
        if self._symbol_cycle is None:
            self._symbol_cycle = itertools.cycle([WarmFront._symbol, ColdFront._symbol])
        return next(self._symbol_cycle)


@exporter.export
class StationaryFront(Front):
    """Draw a stationary front as alternating cold and warm front segments."""

    _symbol = WarmFront._symbol
    _symbol2 = ColdFront._symbol.transformed(mtransforms.Affine2D().scale(1, -1))

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
        line_shift = renderer.points_to_pixels(gc.get_linewidth()) / 2

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
