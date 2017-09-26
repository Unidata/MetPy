# Copyright (c) 2015,2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `skewt` module."""

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pytest

from metpy.plots import Hodograph, SkewT
# Fixtures to make sure we have the right backend and consistent round
from metpy.testing import patch_round, set_agg_backend  # noqa: F401
from metpy.units import units

MPL_VERSION = matplotlib.__version__[:3]


@pytest.mark.mpl_image_compare(tolerance=0.021, remove_text=True)
def test_skewt_api():
    """Test the SkewT API."""
    fig = plt.figure(figsize=(9, 9))
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

    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_skewt_subplot():
    """Test using SkewT on a sub-plot."""
    fig = plt.figure(figsize=(9, 9))
    SkewT(fig, subplot=(2, 2, 1))
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_skewt_gridspec():
    """Test using SkewT on a sub-plot."""
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(1, 2)
    SkewT(fig, subplot=gs[0, 1])
    return fig


def test_skewt_with_grid_enabled():
    """Test using SkewT when gridlines are already enabled (#271)."""
    with plt.rc_context(rc={'axes.grid': True}):
        # Also tests when we don't pass in Figure
        SkewT()


@pytest.fixture()
def test_profile():
    """Return data for a test profile."""
    return np.linspace(1000, 100, 10), np.linspace(20, -20, 10), np.linspace(25, -30, 10)


@pytest.mark.mpl_image_compare(tolerance={'1.4': 1.71}.get(MPL_VERSION, 0.), remove_text=True)
def test_skewt_shade_cape_cin(test_profile):
    """Test shading CAPE and CIN on a SkewT plot."""
    p, t, tp = test_profile
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig)
    skew.plot(p, t, 'r')
    skew.plot(p, tp, 'k')
    skew.shade_cape(p, t, tp)
    skew.shade_cin(p, t, tp)
    skew.ax.set_xlim(-50, 50)
    return fig


@pytest.mark.mpl_image_compare(tolerance={'1.4': 1.70}.get(MPL_VERSION, 0.), remove_text=True)
def test_skewt_shade_area(test_profile):
    """Test shading areas on a SkewT plot."""
    p, t, tp = test_profile
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig)
    skew.plot(p, t, 'r')
    skew.plot(p, tp, 'k')
    skew.shade_area(p, t, tp)
    skew.ax.set_xlim(-50, 50)
    return fig


def test_skewt_shade_area_invalid(test_profile):
    """Test shading areas on a SkewT plot."""
    p, t, tp = test_profile
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig)
    skew.plot(p, t, 'r')
    skew.plot(p, tp, 'k')
    with pytest.raises(ValueError):
        skew.shade_area(p, t, tp, which='positve')


@pytest.mark.mpl_image_compare(tolerance={'1.4': 1.75}.get(MPL_VERSION, 0.), remove_text=True)
def test_skewt_shade_area_kwargs(test_profile):
    """Test shading areas on a SkewT plot with kwargs."""
    p, t, tp = test_profile
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig)
    skew.plot(p, t, 'r')
    skew.plot(p, tp, 'k')
    skew.shade_area(p, t, tp, facecolor='m')
    skew.ax.set_xlim(-50, 50)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_api():
    """Basic test of Hodograph API."""
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    hodo = Hodograph(ax, component_range=60)
    hodo.add_grid(increment=5, color='k')
    hodo.plot([1, 10], [1, 10], color='red')
    hodo.plot_colormapped(np.array([1, 3, 5, 10]), np.array([2, 4, 6, 11]),
                          np.array([0.1, 0.3, 0.5, 0.9]), cmap='Greys')
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_units():
    """Test passing unit-ed quantities to Hodograph."""
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    hodo = Hodograph(ax)
    u = np.arange(10) * units.kt
    v = np.arange(10) * units.kt
    hodo.plot(u, v)
    hodo.plot_colormapped(u, v, np.sqrt(u * u + v * v), cmap='Greys')
    ax.set_xlabel('')
    ax.set_ylabel('')
    return fig


def test_hodograph_alone():
    """Test to create Hodograph without specifying axes."""
    Hodograph()


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_plot_colormapped():
    """Test hodograph colored line with NaN values."""
    u = np.arange(5., 65., 5)
    v = np.arange(-5., -65., -5)
    u[3] = np.nan
    v[6] = np.nan
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    hodo = Hodograph(ax, component_range=80)
    hodo.add_grid(increment=20, color='k')
    hodo.plot_colormapped(u, v, np.hypot(u, v), cmap='Greys')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.021, remove_text=True)
def test_skewt_barb_color():
    """Test plotting colored wind barbs on the Skew-T."""
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig)

    p = np.linspace(1000, 100, 10)
    u = np.linspace(-10, 10, 10)
    skew.plot_barbs(p, u, u, c=u)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_plot_layers():
    """Test hodograph colored height layers with interpolation."""
    u = np.zeros((6)) * units.knots
    v = np.array([0, 10, 20, 30, 40, 50]) * units.knots
    heights = np.array([0, 1000, 2000, 3000, 4000, 5000]) * units.m
    bounds = np.array([500, 1500, 2500, 3500, 4500]) * units.m
    colors = ['r', 'g', 'b', 'r']
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax1)
    h.add_grid(increment=10)
    h.plot_colormapped(u, v, heights, colors=colors, bounds=bounds)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-5, 50)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_plot_layers_different_units():
    """Test hodograph colored height layers with interpolation and different units."""
    u = np.zeros((6)) * units.knots
    v = np.array([0, 10, 20, 30, 40, 50]) * units.knots
    heights = np.array([0, 1, 2, 3, 4, 5]) * units.km
    bounds = np.array([500, 1500, 2500, 3500, 4500]) * units.m
    colors = ['r', 'g', 'b', 'r']
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax1)
    h.add_grid(increment=10)
    h.plot_colormapped(u, v, heights, colors=colors, bounds=bounds)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-5, 50)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_plot_arbitrary_layer():
    """Test hodograph colored layers for arbitrary variables without interpolation."""
    u = np.arange(5, 65, 5) * units('knot')
    v = np.arange(-5, -65, -5) * units('knot')
    speed = np.sqrt(u ** 2 + v ** 2)
    colors = ['red', 'green', 'blue']
    levels = [0, 10, 20, 30] * units('knot')
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    hodo = Hodograph(ax, component_range=80)
    hodo.add_grid(increment=20, color='k')
    hodo.plot_colormapped(u, v, speed, bounds=levels, colors=colors)

    return fig
