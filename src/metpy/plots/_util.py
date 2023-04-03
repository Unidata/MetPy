# Copyright (c) 2015,2017,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Utilities for use in making plots."""

from datetime import datetime

from matplotlib.collections import LineCollection
import matplotlib.patheffects as mpatheffects
from matplotlib.pyplot import imread
import numpy as np

from ..units import concatenate


def add_timestamp(ax, time=None, x=0.99, y=-0.04, ha='right', high_contrast=False,
                  pretext='Created: ', time_format='%Y-%m-%dT%H:%M:%SZ', **kwargs):
    """Add a timestamp to a plot.

    Adds a timestamp to a plot, defaulting to the time of plot creation in ISO format.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The `Axes` instance used for plotting
    time : `datetime.datetime` (or any object with a compatible ``strftime`` method)
        Specific time to be plotted - datetime.utcnow will be use if not specified
    x : float
        Relative x position on the axes of the timestamp
    y : float
        Relative y position on the axes of the timestamp
    ha : str
        Horizontal alignment of the time stamp string
    high_contrast : bool
        Outline text for increased contrast
    pretext : str
        Text to appear before the timestamp, optional. Defaults to 'Created: '
    time_format : str
        Display format of time, optional. Defaults to ISO format.

    Returns
    -------
    `matplotlib.text.Text`
        The `matplotlib.text.Text` instance created

    """
    if high_contrast:
        text_args = {'color': 'white',
                     'path_effects':
                         [mpatheffects.withStroke(linewidth=2, foreground='black')]}
    else:
        text_args = {}
    text_args.update(**kwargs)
    if not time:
        time = datetime.utcnow()
    timestr = time.strftime(time_format)
    # If we don't have a time string after that, assume xarray/numpy and see if item
    if not isinstance(timestr, str):
        timestr = timestr.item()
    return ax.text(x, y, pretext + timestr, ha=ha, transform=ax.transAxes, **text_args)


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
    `matplotlib.image.FigureImage`
       The `matplotlib.image.FigureImage` instance created

    """
    try:
        from importlib.resources import files as importlib_resources_files
    except ImportError:  # Can remove when we require Python > 3.8
        from importlib_resources import files as importlib_resources_files

    fname_suffix = {'small': '_75x75.png',
                    'large': '_150x150.png'}
    fname_prefix = {'unidata': 'unidata',
                    'metpy': 'metpy'}
    try:
        fname = fname_prefix[which] + fname_suffix[size]
    except KeyError:
        raise ValueError('Unknown logo size or selection') from None

    with (importlib_resources_files('metpy.plots') / '_static' / fname).open('rb') as fobj:
        logo = imread(fobj)
    return fig.figimage(logo, x, y, zorder=zorder, **kwargs)


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
    `matplotlib.image.FigureImage`
       The `matplotlib.image.FigureImage` instance created

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
    `matplotlib.image.FigureImage`
       The `matplotlib.image.FigureImage` instance created

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
    segments = np.lib.stride_tricks.as_strided(points.m, shape=final_shape,
                                               strides=final_strides)

    # Create a LineCollection from the segments and set it to colormap based on c
    lc = LineCollection(segments, **kwargs)
    lc.set_array(getattr(c, 'magnitude', c))
    return lc


def convert_gempak_color(c, style='psc'):
    """Convert GEMPAK color numbers into corresponding Matplotlib colors.

    Takes a sequence of GEMPAK color numbers and turns them into
    equivalent Matplotlib colors. Various GEMPAK quirks are respected,
    such as treating negative values as equivalent to 0.

    Parameters
    ----------
    c : int or Sequence[int]
        GEMPAK color number(s)
    style : str, optional
        The GEMPAK 'device' to use to interpret color numbers. May be 'psc'
        (the default; best for a white background) or 'xw' (best for a black background).

    Returns
    -------
        List of strings of Matplotlib colors, or a single string if only one color requested.

    """
    def normalize(x):
        """Transform input x to an int in range 0 to 31 consistent with GEMPAK color quirks."""
        x = int(x)
        if x < 0 or x == 101:
            x = 0
        else:
            x %= 32
        return x

    # Define GEMPAK colors (Matplotlib doesn't appear to like numbered variants)
    cols = ['white',       # 0/32
            'black',       # 1
            'red',         # 2
            'green',       # 3
            'blue',        # 4
            'yellow',      # 5
            'cyan',        # 6
            'magenta',     # 7
            '#CD6839',     # 8 (sienna3)
            '#FF8247',     # 9 (sienna1)
            '#FFA54F',     # 10 (tan1)
            '#FFAEB9',     # 11 (LightPink1)
            '#FF6A6A',     # 12 (IndianRed1)
            '#EE2C2C',     # 13 (firebrick2)
            '#8B0000',     # 14 (red4)
            '#CD0000',     # 15 (red3)
            '#EE4000',     # 16 (OrangeRed2)
            '#FF7F00',     # 17 (DarkOrange1)
            '#CD8500',     # 18 (orange3)
            'gold',        # 19
            '#EEEE00',     # 20 (yellow2)
            'chartreuse',  # 21
            '#00CD00',     # 22 (green3)
            '#008B00',     # 23 (green4)
            '#104E8B',     # 24 (DodgerBlue4)
            'DodgerBlue',  # 25
            '#00B2EE',     # 26 (DeepSkyBlue2)
            '#00EEEE',     # 27 (cyan2)
            '#8968CD',     # 28 (MediumPurple3)
            '#912CEE',     # 29 (purple2)
            '#8B008B',     # 30 (magenta4)
            'bisque']      # 31

    if style != 'psc':
        if style == 'xw':
            cols[0] = 'black'
            cols[1] = 'bisque'
            cols[31] = 'white'
        else:
            raise ValueError('Unknown style parameter')

    try:
        c_list = list(c)
        res = [cols[normalize(x)] for x in c_list]
    except TypeError:
        res = cols[normalize(c)]
    return res
