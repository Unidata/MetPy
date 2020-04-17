# Copyright (c) 2015,2016,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `skewt` module."""

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pytest

from metpy.plots import Hodograph, SkewT
from metpy.testing import check_and_silence_deprecation
# Fixtures to make sure we have the right backend and consistent round
from metpy.testing import patch_round, set_agg_backend  # noqa: F401, I202
from metpy.units import units


@pytest.mark.mpl_image_compare(tolerance=.0202, remove_text=True, style='default')
def test_skewt_api():
    """Test the SkewT API."""
    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, aspect='auto')

        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        p = np.linspace(1000, 100, 10)
        t = np.linspace(20, -20, 10)
        u = np.linspace(-10, 10, 10)
        skew.plot(p, t, 'r')
        skew.plot_barbs(p, u, u)

        skew.ax.set_xlim(-20, 30)
        skew.ax.set_ylim(1000, 100)

        # Add the relevant special lines
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()

        # Call again to hit removal statements
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()

    return fig


@pytest.mark.mpl_image_compare(tolerance=.0272 if matplotlib.__version__ < '3.2' else 34.4,
                               remove_text=True, style='default')
def test_skewt_api_units():
    """#Test the SkewT API when units are provided."""
    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig)
        p = (np.linspace(950, 100, 10) * units.hPa).to(units.Pa)
        t = (np.linspace(18, -20, 10) * units.degC).to(units.kelvin)
        u = np.linspace(-20, 20, 10) * units.knots

        skew.plot(p, t, 'r')
        skew.plot_barbs(p, u, u)

        # Add the relevant special lines
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()

        # This works around the fact that newer pint versions default to degrees_Celsius
        skew.ax.set_xlabel('degC')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0. if matplotlib.__version__ >= '3.2' else 30.,
                               remove_text=True, style='default')
def test_skewt_default_aspect_empty():
    """Test SkewT with default aspect and no plots, only special lines."""
    # With this rotation and the default aspect, this matches exactly the NWS SkewT PDF
    fig = plt.figure(figsize=(12, 9))
    skew = SkewT(fig, rotation=43)
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    return fig


@pytest.mark.skipif(matplotlib.__version__ < '3.2',
                    reason='Matplotlib versions generate different image sizes.')
@pytest.mark.mpl_image_compare(tolerance=0., remove_text=False, style='default',
                               savefig_kwargs={'bbox_inches': 'tight'})
def test_skewt_tight_bbox():
    """Test SkewT when saved with `savefig(..., bbox_inches='tight')`."""
    fig = plt.figure(figsize=(12, 9))
    SkewT(fig)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0.811, remove_text=True, style='default')
def test_skewt_subplot():
    """Test using SkewT on a sub-plot."""
    fig = plt.figure(figsize=(9, 9))
    SkewT(fig, subplot=(2, 2, 1), aspect='auto')
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True, style='default')
def test_skewt_gridspec():
    """Test using SkewT on a GridSpec sub-plot."""
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(1, 2)
    SkewT(fig, subplot=gs[0, 1], aspect='auto')
    return fig


def test_skewt_with_grid_enabled():
    """Test using SkewT when gridlines are already enabled (#271)."""
    with plt.rc_context(rc={'axes.grid': True}):
        # Also tests when we don't pass in Figure
        SkewT(aspect='auto')


@pytest.mark.mpl_image_compare(tolerance=0., remove_text=True, style='default')
def test_skewt_arbitrary_rect():
    """Test placing the SkewT in an arbitrary rectangle."""
    fig = plt.figure(figsize=(9, 9))
    SkewT(fig, rect=(0.15, 0.35, 0.8, 0.3), aspect='auto')
    return fig


def test_skewt_subplot_rect_conflict():
    """Test the subplot/rect conflict failure."""
    with pytest.raises(ValueError):
        SkewT(rect=(0.15, 0.35, 0.8, 0.3), subplot=(1, 1, 1))


@pytest.mark.mpl_image_compare(tolerance=0., remove_text=True, style='default')
def test_skewt_units():
    """Test that plotting with SkewT works with units properly."""
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, aspect='auto')

    skew.ax.axvline(np.array([273]) * units.kelvin, color='purple')
    skew.ax.axhline(np.array([50000]) * units.Pa, color='red')
    skew.ax.axvline(np.array([-20]) * units.degC, color='darkred')
    skew.ax.axvline(-10, color='orange')
    return fig


@pytest.fixture()
def test_profile():
    """Return data for a test profile."""
    return np.linspace(1000, 100, 10), np.linspace(20, -20, 10), np.linspace(25, -30, 10)


@pytest.mark.mpl_image_compare(tolerance=.02, remove_text=True, style='default')
def test_skewt_shade_cape_cin(test_profile):
    """Test shading CAPE and CIN on a SkewT plot."""
    p, t, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, aspect='auto')
        skew.plot(p, t, 'r')
        skew.plot(p, tp, 'k')
        skew.shade_cape(p, t, tp)
        skew.shade_cin(p, t, tp)
        skew.ax.set_xlim(-50, 50)
        skew.ax.set_ylim(1000, 100)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.02, remove_text=True, style='default')
def test_skewt_shade_area(test_profile):
    """Test shading areas on a SkewT plot."""
    p, t, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, aspect='auto')
        skew.plot(p, t, 'r')
        skew.plot(p, tp, 'k')
        skew.shade_area(p, t, tp)
        skew.ax.set_xlim(-50, 50)
        skew.ax.set_ylim(1000, 100)

    return fig


def test_skewt_shade_area_invalid(test_profile):
    """Test shading areas on a SkewT plot."""
    p, t, tp = test_profile
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, aspect='auto')
    skew.plot(p, t, 'r')
    skew.plot(p, tp, 'k')
    with pytest.raises(ValueError):
        skew.shade_area(p, t, tp, which='positve')


@pytest.mark.mpl_image_compare(tolerance=0.02, remove_text=True, style='default')
def test_skewt_shade_area_kwargs(test_profile):
    """Test shading areas on a SkewT plot with kwargs."""
    p, t, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, aspect='auto')
        skew.plot(p, t, 'r')
        skew.plot(p, tp, 'k')
        skew.shade_area(p, t, tp, facecolor='m')
        skew.ax.set_xlim(-50, 50)
        skew.ax.set_ylim(1000, 100)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True, style='default')
def test_skewt_wide_aspect_ratio(test_profile):
    """Test plotting a skewT with a wide aspect ratio."""
    p, t, tp = test_profile

    fig = plt.figure(figsize=(12.5, 3))
    skew = SkewT(fig, aspect='auto')
    skew.plot(p, t, 'r')
    skew.plot(p, tp, 'k')
    skew.ax.set_xlim(-30, 50)
    skew.ax.set_ylim(1050, 700)
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


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True, style='default')
def test_skewt_barb_color():
    """Test plotting colored wind barbs on the Skew-T."""
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, aspect='auto')

    p = np.linspace(1000, 100, 10)
    u = np.linspace(-10, 10, 10)
    skew.plot_barbs(p, u, u, c=u)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.02, remove_text=True, style='default')
def test_skewt_barb_unit_conversion():
    """Test that barbs units can be converted at plot time (#737)."""
    u_wind = np.array([3.63767155210412]) * units('m/s')
    v_wind = np.array([3.63767155210412]) * units('m/s')
    p_wind = np.array([500]) * units.hPa

    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, aspect='auto')
    skew.ax.set_ylabel('')  # remove_text doesn't do this as of pytest 0.9
    skew.plot_barbs(p_wind, u_wind, v_wind, plot_units='knots')
    skew.ax.set_ylim(1000, 500)
    skew.ax.set_yticks([1000, 750, 500])
    skew.ax.set_xlim(-20, 20)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.02, remove_text=True, style='default')
def test_skewt_barb_no_default_unit_conversion():
    """Test that barbs units are left alone by default (#737)."""
    u_wind = np.array([3.63767155210412]) * units('m/s')
    v_wind = np.array([3.63767155210412]) * units('m/s')
    p_wind = np.array([500]) * units.hPa

    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, aspect='auto')
    skew.ax.set_ylabel('')  # remove_text doesn't do this as of pytest 0.9
    skew.plot_barbs(p_wind, u_wind, v_wind)
    skew.ax.set_ylim(1000, 500)
    skew.ax.set_yticks([1000, 750, 500])
    skew.ax.set_xlim(-20, 20)

    return fig


@pytest.mark.parametrize('u,v', [(np.array([3]) * units('m/s'), np.array([3])),
                                 (np.array([3]), np.array([3]) * units('m/s'))])
def test_skewt_barb_unit_conversion_exception(u, v):
    """Test that errors are raise if unit conversion is requested on un-united data."""
    p_wind = np.array([500]) * units.hPa

    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, aspect='auto')
    with pytest.raises(ValueError):
        skew.plot_barbs(p_wind, u, v, plot_units='knots')


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_plot_layers():
    """Test hodograph colored height layers with interpolation."""
    u = np.zeros(6) * units.knots
    v = np.array([0, 10, 20, 30, 40, 50]) * units.knots
    heights = np.array([0, 1000, 2000, 3000, 4000, 5000]) * units.m
    intervals = np.array([500, 1500, 2500, 3500, 4500]) * units.m
    colors = ['r', 'g', 'b', 'r']
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax1)
    h.add_grid(increment=10)
    h.plot_colormapped(u, v, heights, colors=colors, intervals=intervals)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-5, 50)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_plot_layers_different_units():
    """Test hodograph colored height layers with interpolation and different units."""
    u = np.zeros(6) * units.knots
    v = np.array([0, 10, 20, 30, 40, 50]) * units.knots
    heights = np.array([0, 1, 2, 3, 4, 5]) * units.km
    intervals = np.array([500, 1500, 2500, 3500, 4500]) * units.m
    colors = ['r', 'g', 'b', 'r']
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax1)
    h.add_grid(increment=10)
    h.plot_colormapped(u, v, heights, colors=colors, intervals=intervals)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-5, 50)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_plot_layers_bound_units():
    """Test hodograph colored height layers with interpolation and different units."""
    u = np.zeros(6) * units.knots
    v = np.array([0, 10, 20, 30, 40, 50]) * units.knots
    heights = np.array([0, 1000, 2000, 3000, 4000, 5000]) * units.m
    intervals = np.array([0.5, 1.5, 2.5, 3.5, 4.5]) * units.km
    colors = ['r', 'g', 'b', 'r']
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax1)
    h.add_grid(increment=10)
    h.plot_colormapped(u, v, heights, colors=colors, intervals=intervals)
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
    hodo.plot_colormapped(u, v, speed, intervals=levels, colors=colors)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_wind_vectors():
    """Test plotting wind vectors onto a hodograph."""
    u_wind = np.array([-10, -7, 0, 7, 10, 7, 0, -7])
    v_wind = np.array([0, 7, 10, 7, 0, -7, -10, -7])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax, component_range=20)
    h.plot(u_wind, v_wind, linewidth=3)
    h.wind_vectors(u_wind, v_wind)
    return fig


@pytest.mark.xfail
def test_united_hodograph_range():
    """Tests making a hodograph with a united ranged."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    Hodograph(ax, component_range=60. * units.knots)


@check_and_silence_deprecation
def test_plot_colormapped_bounds_deprecation():
    """Test deprecation of bounds kwarg in `plot_colormapped`."""
    u = np.zeros(6) * units.knots
    v = np.array([0, 10, 20, 30, 40, 50]) * units.knots
    heights = np.array([0, 1000, 2000, 3000, 4000, 5000]) * units.m
    intervals = np.array([500, 1500, 2500, 3500, 4500]) * units.m
    colors = ['r', 'g', 'b', 'r']
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax1)
    h.plot_colormapped(u, v, heights, colors=colors, bounds=intervals)
