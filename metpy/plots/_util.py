# Copyright (c) 2015,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Utilities for use in making plots."""

from datetime import datetime
import posixpath

from matplotlib.collections import LineCollection
from matplotlib.pyplot import imread
import numpy as np
import pkg_resources

from ..units import concatenate


def add_timestamp(ax, time=None, x=0.99, y=-0.04, ha='right', **kwargs):
    """Add a timestamp at plot creation time.

    Adds an ISO format timestamp with the time of plot creation to the plot.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The `Axes` instance used for plotting
    time : `datetime.datetime`
        Specific time to be plotted - datetime.utcnow will be use if not specified
    x : float
        Relative x position on the axes of the timestamp
    y : float
        Relative y position on the axes of the timestamp
    ha : str
        Horizontal alignment of the time stamp string

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        The `Axes` instance used for plotting

    """
    if not time:
        time = datetime.utcnow()
    timestr = datetime.strftime(time, 'Created: %Y-%m-%dT%H:%M:%SZ')
    ax.text(x, y, timestr, ha=ha, transform=ax.transAxes, **kwargs)
    return ax


def _add_logo(fig, x=10, y=25, zorder=100, which='metpy', size='small', **kwargs):
    """Add the MetPy or Unidata logo to a figure.

    Adds an image to the figure.

    Parameters
    ----------
    fig : `matplotlib.figure`
       The `figure` instance used for plotting
    x : int
       x position padding in pixels
    y : float
       y position padding in pixels
    zorder : int
       The zorder of the logo
    which : str
       Which logo to plot 'metpy' or 'unidata'
    size : str
       Size of logo to be used. Can be 'small' for 75 px square or 'large' for
       150 px square.

    Returns
    -------
    fig : `matplotlib.figure`
       The `figure` instance used for plotting

    """
    fname_suffix = {'small': '_75x75.png',
                    'large': '_150x150.png'}
    fname_prefix = {'unidata': 'unidata',
                    'metpy': 'metpy'}
    try:
        fname = fname_prefix[which] + fname_suffix[size]
        fpath = posixpath.join('_static', fname)
    except KeyError:
        raise ValueError('Unknown logo size or selection')

    logo = imread(pkg_resources.resource_stream('metpy.plots', fpath))
    fig.figimage(logo, x, y, zorder=zorder, **kwargs)
    return fig


def add_metpy_logo(fig, x=10, y=25, zorder=100, size='small', **kwargs):
    """Add the MetPy logo to a figure.

    Adds an image of the MetPy logo to the figure.

    Parameters
    ----------
    fig : `matplotlib.figure`
       The `figure` instance used for plotting
    x : int
       x position padding in pixels
    y : float
       y position padding in pixels
    zorder : int
       The zorder of the logo
    size : str
       Size of logo to be used. Can be 'small' for 75 px square or 'large' for
       150 px square.

    Returns
    -------
    fig : `matplotlib.figure`
       The `figure` instance used for plotting

    """
    return _add_logo(fig, x=x, y=y, zorder=zorder, which='metpy', size=size, **kwargs)


def add_unidata_logo(fig, x=10, y=25, zorder=100, size='small', **kwargs):
    """Add the Unidata logo to a figure.

    Adds an image of the MetPy logo to the figure.

    Parameters
    ----------
    fig : `matplotlib.figure`
       The `figure` instance used for plotting
    x : int
       x position padding in pixels
    y : float
       y position padding in pixels
    zorder : int
       The zorder of the logo
    size : str
       Size of logo to be used. Can be 'small' for 75 px square or 'large' for
       150 px square.

    Returns
    -------
    fig : `matplotlib.figure`
       The `figure` instance used for plotting

    """
    return _add_logo(fig, x=x, y=y, zorder=zorder, which='unidata', size=size, **kwargs)


# Not part of public API
def colored_line(x, y, c, **kwargs):
    """Create a multi-colored line.

    Takes a set of points and turns them into a collection of lines colored by another array.

    Parameters
    ----------
    x : array-like
        x-axis coordinates
    y : array-like
        y-axis coordinates
    c : array-like
        values used for color-mapping
    kwargs : dict
        Other keyword arguments passed to :class:`matplotlib.collections.LineCollection`

    Returns
    -------
        The created :class:`matplotlib.collections.LineCollection` instance.

    """
    # Mask out any NaN values
    nan_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(c))
    x = x[nan_mask]
    y = y[nan_mask]
    c = c[nan_mask]

    # Paste values end to end
    points = concatenate([x, y])

    # Exploit numpy's strides to present a view of these points without copying.
    # Dimensions are (segment, start/end, x/y). Since x and y are concatenated back to back,
    # moving between segments only moves one item; moving start to end is only an item;
    # The move between x any moves from one half of the array to the other
    num_pts = points.size // 2
    final_shape = (num_pts - 1, 2, 2)
    final_strides = (points.itemsize, points.itemsize, num_pts * points.itemsize)
    segments = np.lib.stride_tricks.as_strided(points, shape=final_shape,
                                               strides=final_strides)

    # Create a LineCollection from the segments and set it to colormap based on c
    lc = LineCollection(segments, **kwargs)
    lc.set_array(c)
    return lc
