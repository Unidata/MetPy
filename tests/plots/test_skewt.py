# Copyright (c) 2015,2016,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `skewt` module."""
import warnings

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pytest

from metpy.plots import Hodograph, SkewT
from metpy.units import units

MPL_VERSION = matplotlib.__version__[:3]


@pytest.mark.mpl_image_compare(remove_text=True, style='default', tolerance=0.069)
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


@pytest.mark.mpl_image_compare(remove_text=True, style='default', tolerance=0.32)
def test_skewt_api_units():
    """Test the SkewT API when units are provided."""
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


@pytest.mark.mpl_image_compare(tolerance=0., remove_text=True, style='default')
def test_skewt_default_aspect_empty():
    """Test SkewT with default aspect and no plots, only special lines."""
    # With this rotation and the default aspect, this matches exactly the NWS SkewT PDF
    fig = plt.figure(figsize=(12, 9))
    skew = SkewT(fig, rotation=43)
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    return fig


@pytest.mark.mpl_image_compare(tolerance=0., remove_text=True, style='default')
def test_skewt_mixing_line_args():
    """Test plot_mixing_lines accepting kwargs for mixing ratio and pressure levels."""
    # Explicitly pass default values as kwargs the, should recreate NWS SkewT PDF as above
    fig = plt.figure(figsize=(12, 9))
    skew = SkewT(fig, rotation=43)
    mlines = np.array([0.0004, 0.001, 0.002, 0.004, 0.007, 0.01, 0.016, 0.024, 0.032])
    press = units.Quantity(np.linspace(600, max(skew.ax.get_ylim())), 'mbar')
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines(mixing_ratio=mlines, pressure=press)
    return fig


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


@pytest.mark.mpl_image_compare(tolerance=0.0198, remove_text=True, style='default')
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
    pressure = np.array([966., 937.2, 925., 904.6, 872.6, 853., 850., 836., 821., 811.6, 782.3,
                         754.2, 726.9, 700., 648.9, 624.6, 601.1, 595., 587., 576., 555.7,
                         534.2, 524., 500., 473.3, 400., 384.5, 358., 343., 308.3, 300., 276.,
                         273., 268.5, 250., 244.2, 233., 200.]) * units.mbar
    temperature = np.array([18.2, 16.8, 16.2, 15.1, 13.3, 12.2, 12.4, 14., 14.4,
                            13.7, 11.4, 9.1, 6.8, 4.4, -1.4, -4.4, -7.3, -8.1,
                            -7.9, -7.7, -8.7, -9.8, -10.3, -13.5, -17.1, -28.1, -30.7,
                            -35.3, -37.1, -43.5, -45.1, -49.9, -50.4, -51.1, -54.1, -55.,
                            -56.7, -57.5]) * units.degC
    dewpoint = np.array([16.9, 15.9, 15.5, 14.2, 12.1, 10.8, 8.6, 0., -3.6, -4.4,
                        -6.9, -9.5, -12., -14.6, -15.8, -16.4, -16.9, -17.1, -27.9, -42.7,
                        -44.1, -45.6, -46.3, -45.5, -47.1, -52.1, -50.4, -47.3, -57.1,
                        -57.9, -58.1, -60.9, -61.4, -62.1, -65.1, -65.6,
                        -66.7, -70.5]) * units.degC
    profile = np. array([18.2, 16.18287437, 15.68644745, 14.8369451,
                        13.45220646, 12.57020365, 12.43280242, 11.78283506,
                        11.0698586, 10.61393901, 9.14490966, 7.66233636,
                        6.1454231, 4.56888673, 1.31644072, -0.36678427,
                        -2.09120703, -2.55566745, -3.17594616, -4.05032505,
                        -5.73356001, -7.62361933, -8.56236581, -10.88846868,
                        -13.69095789, -22.82604468, -25.08463516, -29.26014016,
                        -31.81335912, -38.29612829, -39.97374452, -45.11966793,
                        -45.79482793, -46.82129892, -51.21936594, -52.65924319,
                        -55.52598916, -64.68843697]) * units.degC
    return pressure, temperature, dewpoint, profile


@pytest.mark.mpl_image_compare(tolerance=.033, remove_text=True, style='default')
def test_skewt_shade_cape_cin(test_profile):
    """Test shading CAPE and CIN on a SkewT plot."""
    p, t, td, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, aspect='auto')
        skew.plot(p, t, 'r')
        skew.plot(p, tp, 'k')
        skew.shade_cape(p, t, tp)
        skew.shade_cin(p, t, tp, td)
        skew.ax.set_xlim(-50, 50)
        skew.ax.set_ylim(1000, 100)

        # This works around the fact that newer pint versions default to degrees_Celsius
        skew.ax.set_xlabel('degC')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.033, remove_text=True, style='default')
def test_skewt_shade_cape_cin_no_limit(test_profile):
    """Test shading CIN without limits."""
    p, t, _, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, aspect='auto')
        skew.plot(p, t, 'r')
        skew.plot(p, tp, 'k')
        skew.shade_cape(p, t, tp)
        skew.shade_cin(p, t, tp)
        skew.ax.set_xlim(-50, 50)
        skew.ax.set_ylim(1000, 100)

        # This works around the fact that newer pint versions default to degrees_Celsius
        skew.ax.set_xlabel('degC')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.033, remove_text=True, style='default')
def test_skewt_shade_area(test_profile):
    """Test shading areas on a SkewT plot."""
    p, t, _, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, aspect='auto')
        skew.plot(p, t, 'r')
        skew.plot(p, tp, 'k')
        skew.shade_area(p, t, tp)
        skew.ax.set_xlim(-50, 50)
        skew.ax.set_ylim(1000, 100)

        # This works around the fact that newer pint versions default to degrees_Celsius
        skew.ax.set_xlabel('degC')

    return fig


def test_skewt_shade_area_invalid(test_profile):
    """Test shading areas on a SkewT plot."""
    p, t, _, tp = test_profile
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, aspect='auto')
    skew.plot(p, t, 'r')
    skew.plot(p, tp, 'k')
    with pytest.raises(ValueError):
        skew.shade_area(p, t, tp, which='positve')


@pytest.mark.mpl_image_compare(tolerance=0.033, remove_text=True, style='default')
def test_skewt_shade_area_kwargs(test_profile):
    """Test shading areas on a SkewT plot with kwargs."""
    p, t, _, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, aspect='auto')
        skew.plot(p, t, 'r')
        skew.plot(p, tp, 'k')
        skew.shade_area(p, t, tp, facecolor='m')
        skew.ax.set_xlim(-50, 50)
        skew.ax.set_ylim(1000, 100)

        # This works around the fact that newer pint versions default to degrees_Celsius
        skew.ax.set_xlabel('degC')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.039, remove_text=True, style='default')
def test_skewt_wide_aspect_ratio(test_profile):
    """Test plotting a skewT with a wide aspect ratio."""
    p, t, _, tp = test_profile

    fig = plt.figure(figsize=(12.5, 3))
    skew = SkewT(fig, aspect='auto')
    skew.plot(p, t, 'r')
    skew.plot(p, tp, 'k')
    skew.ax.set_xlim(-30, 50)
    skew.ax.set_ylim(1050, 700)

    # This works around the fact that newer pint versions default to degrees_Celsius
    skew.ax.set_xlabel('degC')
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_api():
    """Basic test of Hodograph API."""
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    hodo = Hodograph(ax, component_range=60)
    hodo.add_grid(increment=5, color='k')
    hodo.plot([1, 10], [1, 10], color='red')
    hodo.plot_colormapped(np.array([1, 3, 5, 10]), -np.array([2, 4, 6, 11]),
                          np.array([0.1, 0.3, 0.5, 0.9]), cmap='Greys')
    return fig


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.6 if MPL_VERSION == '3.3' else 0.)
def test_hodograph_units():
    """Test passing quantities to Hodograph."""
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    hodo = Hodograph(ax)
    u = np.arange(10) * units.kt
    v = np.arange(10) * units.kt
    hodo.plot(u, v)
    hodo.plot_colormapped(u, -v, np.hypot(u, v), cmap='Greys')
    ax.set_xlabel('')
    ax.set_ylabel('')
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_hodograph_masked_array():
    """Basic test of Hodograph API."""
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    hodo = Hodograph(ax, component_range=20)
    u = np.ma.array([1, 3, 5, 10])
    v = np.ma.array([2, 4, 6, 11])
    h = units.Quantity(np.array([0.1, 3.5, 5.5, 10.9]), 'km')
    intervals = units.Quantity(np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0]), 'km')
    colors = ['red', 'green', 'yellow', 'blue', 'purple']
    # Check that we're not triggering interpolation warnings
    with warnings.catch_warnings(record=True) as record:
        hodo.plot_colormapped(u, v, h, intervals=intervals, colors=colors)
        assert len(record) == 0
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


@pytest.mark.mpl_image_compare(tolerance=0.141, remove_text=True, style='default')
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
    """Test that an error is raised if unit conversion is requested on plain arrays."""
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


def test_hodograph_range_with_units():
    """Tests making a hodograph with a range with units."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    Hodograph(ax, component_range=60. * units.knots)
