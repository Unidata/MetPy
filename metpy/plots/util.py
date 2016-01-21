# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from matplotlib.collections import LineCollection

from ..units import concatenate


def resample_nn_1d(a, centers):
    """Helper function that returns one-dimensional nearest-neighbor
    indexes based on user-specified centers.
    ----------
    a : array-like
        1-dimensional array of numeric values from which to
        extract indexes of nearest-neighbors
    centers : array-like
        1-dimensional array of numeric values representing a subset of values to approximate
    Returns
    -------
        An array of indexes representing values closest to given array values

    """
    ix = []
    for center in centers:
        index = (np.abs(a-center)).argmin()
        if index not in ix:
            ix.append(index)
    return ix


# Not part of public API
def colored_line(x, y, c, **kwargs):
    """Helper function to take a set of points and turn them into a collection of
    lines colored by another array

    Parameters
    ----------
    x : array-like
        x-axis coordinates
    y : array-like
        y-axis coordinates
    c : array-like
        values used for color-mapping
    kwargs : dict
        Other keyword arguments passed to ``matplotlib.collections.LineCollection``

    Returns
    -------
        The created ``matplotlib.collections.LineCollection`` instance.

    """
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
