# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from matplotlib.gridspec import GridSpec
import pytest

from metpy.plots.skewt import *  # noqa
from metpy.testing import hide_tick_labels, make_figure
from metpy.units import units


@pytest.mark.mpl_image_compare
def test_skewt_api():
    'Test the SkewT api'
    fig = make_figure(figsize=(9, 9))
    skew = SkewT(fig)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    p = np.linspace(1000, 100, 10)
    t = np.linspace(20, -20, 10)
    u = np.linspace(-10, 10, 10)
    skew.plot(p, t, 'r')
    skew.plot_barbs(p, u, u)

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    hide_tick_labels(skew.ax)

    return fig


@pytest.mark.mpl_image_compare
def test_skewt_subplot():
    'Test using SkewT on a sub-plot'
    fig = make_figure(figsize=(9, 9))
    hide_tick_labels(SkewT(fig, subplot=(2, 2, 1)).ax)
    return fig


@pytest.mark.mpl_image_compare
def test_skewt_gridspec():
    'Test using SkewT on a sub-plot'
    fig = make_figure(figsize=(9, 9))
    gs = GridSpec(1, 2)
    hide_tick_labels(SkewT(fig, subplot=gs[0, 1]).ax)
    return fig


@pytest.mark.mpl_image_compare
def test_hodograph_api():
    'Basic test of Hodograph API'
    fig = make_figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    hodo = Hodograph(ax, component_range=60)
    hodo.add_grid(increment=5, color='k')
    hodo.plot([1, 10], [1, 10], color='red')
    hodo.plot_colormapped(np.array([1, 3, 5, 10]), np.array([2, 4, 6, 11]),
                          np.array([0.1, 0.3, 0.5, 0.9]), cmap='Greys')
    hide_tick_labels(ax)
    return fig


@pytest.mark.mpl_image_compare
def test_hodograph_units():
    'Test passing unit-ed quantities to Hodograph'
    fig = make_figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    hodo = Hodograph(ax)
    u = np.arange(10) * units.kt
    v = np.arange(10) * units.kt
    hodo.plot(u, v)
    hodo.plot_colormapped(u, v, np.sqrt(u * u + v * v), cmap='Greys')
    hide_tick_labels(ax)
    return fig
