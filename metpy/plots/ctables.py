# Copyright (c) 2014,2015,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Work with custom color tables.

Contains a tools for reading color tables from files, and creating instances based on a
specific set of constraints (e.g. step size) for mapping.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import metpy.plots.ctables as ctables

   def plot_color_gradients(cmap_category, cmap_list, nrows):
       fig, axes = plt.subplots(figsize=(7, 6), nrows=nrows)
       fig.subplots_adjust(top=.93, bottom=0.01, left=0.32, right=0.99)
       axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

       for ax, name in zip(axes, cmap_list):
               ax.imshow(gradient, aspect='auto', cmap=ctables.registry.get_colortable(name))
               pos = list(ax.get_position().bounds)
               x_text = pos[0] - 0.01
               y_text = pos[1] + pos[3]/2.
               fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

       # Turn off *all* ticks & spines, not just the ones with colormaps.
       for ax in axes:
           ax.set_axis_off()

   cmaps = list(ctables.registry)
   cmaps = [name for name in cmaps if name[-2:]!='_r']
   nrows = len(cmaps)
   gradient = np.linspace(0, 1, 256)
   gradient = np.vstack((gradient, gradient))

   plot_color_gradients('MetPy', cmaps, nrows)
   plt.show()
"""

from __future__ import division

import ast
import contextlib
import glob
import logging
import os.path
import posixpath

import matplotlib.colors as mcolors
from pkg_resources import resource_listdir, resource_stream

from ..package_tools import Exporter

exporter = Exporter(globals())

TABLE_EXT = '.tbl'

log = logging.getLogger(__name__)


def _parse(s):
    if hasattr(s, 'decode'):
        s = s.decode('ascii')

    if not s.startswith('#'):
        return ast.literal_eval(s)

    return None


@exporter.export
def read_colortable(fobj):
    r"""Read colortable information from a file.

    Reads a colortable, which consists of one color per line of the file, where
    a color can be one of: a tuple of 3 floats, a string with a HTML color name,
    or a string with a HTML hex color.

    Parameters
    ----------
    fobj : a file-like object
        A file-like object to read the colors from

    Returns
    -------
    List of tuples
        A list of the RGB color values, where each RGB color is a tuple of 3 floats in the
        range of [0, 1].

    """
    ret = []
    try:
        for line in fobj:
            literal = _parse(line)
            if literal:
                ret.append(mcolors.colorConverter.to_rgb(literal))
        return ret
    except (SyntaxError, ValueError):
        raise RuntimeError('Malformed colortable.')


def convert_gempak_table(infile, outfile):
    r"""Convert a GEMPAK color table to one MetPy can read.

    Reads lines from a GEMPAK-style color table file, and writes them to another file in
    a format that MetPy can parse.

    Parameters
    ----------
    infile : file-like object
        The file-like object to read from
    outfile : file-like object
        The file-like object to write to

    """
    for line in infile:
        if not line.startswith('!') and line.strip():
            r, g, b = map(int, line.split())
            outfile.write('({0:f}, {1:f}, {2:f})\n'.format(r / 255, g / 255, b / 255))


class ColortableRegistry(dict):
    r"""Manages the collection of color tables.

    Provides access to color tables, read collections of files, and generates
    matplotlib's Normalize instances to go with the colortable.
    """

    def scan_resource(self, pkg, path):
        r"""Scan a resource directory for colortable files and add them to the registry.

        Parameters
        ----------
        pkg : str
            The package containing the resource directory
        path : str
            The path to the directory with the color tables

        """
        for fname in resource_listdir(pkg, path):
            if fname.endswith(TABLE_EXT):
                table_path = posixpath.join(path, fname)
                with contextlib.closing(resource_stream(pkg, table_path)) as stream:
                    self.add_colortable(stream,
                                        posixpath.splitext(posixpath.basename(fname))[0])

    def scan_dir(self, path):
        r"""Scan a directory on disk for color table files and add them to the registry.

        Parameters
        ----------
        path : str
            The path to the directory with the color tables

        """
        for fname in glob.glob(os.path.join(path, '*' + TABLE_EXT)):
            if os.path.isfile(fname):
                with open(fname, 'r') as fobj:
                    try:
                        self.add_colortable(fobj, os.path.splitext(os.path.basename(fname))[0])
                        log.debug('Added colortable from file: %s', fname)
                    except RuntimeError:
                        # If we get a file we can't handle, assume we weren't meant to.
                        log.info('Skipping unparsable file: %s', fname)

    def add_colortable(self, fobj, name):
        r"""Add a color table from a file to the registry.

        Parameters
        ----------
        fobj : file-like object
            The file to read the color table from
        name : str
            The name under which the color table will be stored

        """
        self[name] = read_colortable(fobj)
        self[name + '_r'] = self[name][::-1]

    def get_with_steps(self, name, start, step):
        r"""Get a color table from the registry with a corresponding norm.

        Builds a `matplotlib.colors.BoundaryNorm` using `start`, `step`, and
        the number of colors, based on the color table obtained from `name`.

        Parameters
        ----------
        name : str
            The name under which the color table will be stored
        start : float
            The starting boundary
        step : float
            The step between boundaries

        Returns
        -------
        `matplotlib.colors.BoundaryNorm`, `matplotlib.colors.ListedColormap`
            The boundary norm based on `start` and `step` with the number of colors
            from the number of entries matching the color table, and the color table itself.

        """
        from numpy import arange

        # Need one more boundary than color
        num_steps = len(self[name]) + 1
        boundaries = arange(start, start + step * num_steps, step)
        return self.get_with_boundaries(name, boundaries)

    def get_with_range(self, name, start, end):
        r"""Get a color table from the registry with a corresponding norm.

        Builds a `matplotlib.colors.BoundaryNorm` using `start`, `end`, and
        the number of colors, based on the color table obtained from `name`.

        Parameters
        ----------
        name : str
            The name under which the color table will be stored
        start : float
            The starting boundary
        end : float
            The ending boundary

        Returns
        -------
        `matplotlib.colors.BoundaryNorm`, `matplotlib.colors.ListedColormap`
            The boundary norm based on `start` and `end` with the number of colors
            from the number of entries matching the color table, and the color table itself.

        """
        from numpy import linspace

        # Need one more boundary than color
        num_steps = len(self[name]) + 1
        boundaries = linspace(start, end, num_steps)
        return self.get_with_boundaries(name, boundaries)

    def get_with_boundaries(self, name, boundaries):
        r"""Get a color table from the registry with a corresponding norm.

        Builds a `matplotlib.colors.BoundaryNorm` using `boundaries`.

        Parameters
        ----------
        name : str
            The name under which the color table will be stored
        boundaries : array_like
            The list of boundaries for the norm

        Returns
        -------
        `matplotlib.colors.BoundaryNorm`, `matplotlib.colors.ListedColormap`
            The boundary norm based on `boundaries`, and the color table itself.

        """
        cmap = self.get_colortable(name)
        return mcolors.BoundaryNorm(boundaries, cmap.N), cmap

    def get_colortable(self, name):
        r"""Get a color table from the registry.

        Parameters
        ----------
        name : str
            The name under which the color table will be stored

        Returns
        -------
        `matplotlib.colors.ListedColormap`
            The color table corresponding to `name`

        """
        return mcolors.ListedColormap(self[name], name=name)


registry = ColortableRegistry()
registry.scan_resource('metpy.plots', 'colortable_files')
registry.scan_dir(os.path.curdir)

with exporter:
    colortables = registry
