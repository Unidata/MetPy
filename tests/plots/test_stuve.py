# Copyright (c) 2015,2016,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `Stuve` class."""
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pytest

from metpy.plots import Stuve
from metpy.testing import autoclose_figure, version_check
from metpy.units import units


@pytest.mark.mpl_image_compare(remove_text=True, style='default', tolerance=0.069)
def test_stuve_api():
    """Test the Stuve API."""
    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        stuve = Stuve(fig, aspect='auto')

        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        p = np.linspace(1000, 100, 10)
        t = np.linspace(20, -20, 10)
        u = np.linspace(-10, 10, 10)
        stuve.plot(p, t, 'r')
        stuve.plot_barbs(p, u, u)

        stuve.ax.set_xlim(-20, 30)
        stuve.ax.set_ylim(1000, 100)

        # Add the relevant special lines
        stuve.plot_dry_adiabats()
        stuve.plot_moist_adiabats()
        stuve.plot_mixing_lines()

        # Call again to hit removal statements
        stuve.plot_dry_adiabats()
        stuve.plot_moist_adiabats()
        stuve.plot_mixing_lines()

    return fig


@pytest.mark.mpl_image_compare(remove_text=True, style='default', tolerance=0.35)
def test_stuve_api_units():
    """Test the Stuve API when units are provided."""
    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        stuve = Stuve(fig)
        p = (np.linspace(950, 100, 10) * units.hPa).to(units.Pa)
        t = (np.linspace(18, -20, 10) * units.degC).to(units.kelvin)
        u = np.linspace(-20, 20, 10) * units.knots

        stuve.plot(p, t, 'r')
        stuve.plot_barbs(p, u, u)

        # Add the relevant special lines
        stuve.plot_dry_adiabats()
        stuve.plot_moist_adiabats()
        stuve.plot_mixing_lines()

        # This works around the fact that newer pint versions default to degrees_Celsius
        stuve.ax.set_xlabel('degC')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0., remove_text=True, style='default')
def test_stuve_default_aspect_empty():
    """Test Stuve with default aspect and no plots, only special lines."""
    # With this rotation and the default aspect, this matches exactly the NWS Stuve PDF
    fig = plt.figure(figsize=(12, 9))
    stuve = Stuve(fig, rotation=43)
    stuve.plot_dry_adiabats()
    stuve.plot_moist_adiabats()
    stuve.plot_mixing_lines()
    return fig


@pytest.mark.mpl_image_compare(tolerance=0., remove_text=True, style='default')
def test_stuve_mixing_line_args():
    """Test plot_mixing_lines accepting kwargs for mixing ratio and pressure levels."""
    # Explicitly pass default values as kwargs, should recreate NWS Stuve PDF as above
    fig = plt.figure(figsize=(12, 9))
    stuve = Stuve(fig, rotation=43)
    mlines = np.array([0.0004, 0.001, 0.002, 0.004, 0.007, 0.01, 0.016, 0.024, 0.032])
    press = units.Quantity(np.linspace(600, max(stuve.ax.get_ylim())), 'mbar')
    stuve.plot_dry_adiabats()
    stuve.plot_moist_adiabats()
    stuve.plot_mixing_lines(mixing_ratio=mlines, pressure=press)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0., remove_text=False, style='default',
                               savefig_kwargs={'bbox_inches': 'tight'})
def test_stuve_tight_bbox():
    """Test Stuve when saved with `savefig(..., bbox_inches='tight')`."""
    fig = plt.figure(figsize=(12, 9))
    Stuve(fig)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0.811, remove_text=True, style='default')
def test_stuve_subplot():
    """Test using Stuve on a sub-plot."""
    fig = plt.figure(figsize=(9, 9))
    Stuve(fig, subplot=(2, 2, 1), aspect='auto')
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True, style='default')
def test_stuve_gridspec():
    """Test using Stuve on a GridSpec sub-plot."""
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(1, 2)
    Stuve(fig, subplot=gs[0, 1], aspect='auto')
    return fig


def test_stuve_with_grid_enabled():
    """Test using Stuve when gridlines are already enabled (#271)."""
    with plt.rc_context(rc={'axes.grid': True}):
        # Also tests when we don't pass in Figure
        s = Stuve(aspect='auto')
        plt.close(s.ax.figure)


@pytest.mark.mpl_image_compare(tolerance=0., remove_text=True, style='default')
def test_stuve_arbitrary_rect():
    """Test placing the Stuve in an arbitrary rectangle."""
    fig = plt.figure(figsize=(9, 9))
    Stuve(fig, rect=(0.15, 0.35, 0.8, 0.3), aspect='auto')
    return fig


def test_stuve_subplot_rect_conflict():
    """Test the subplot/rect conflict failure."""
    with pytest.raises(ValueError), autoclose_figure(figsize=(7, 7)) as fig:
        Stuve(fig, rect=(0.15, 0.35, 0.8, 0.3), subplot=(1, 1, 1))


@pytest.mark.mpl_image_compare(tolerance=0.0198, remove_text=True, style='default')
def test_stuve_units():
    """Test that plotting with Stuve works with units properly."""
    fig = plt.figure(figsize=(9, 9))
    stuve = Stuve(fig, aspect='auto')

    stuve.ax.axvline(np.array([273]) * units.kelvin, color='purple')
    stuve.ax.axhline(np.array([50000]) * units.Pa, color='red')
    stuve.ax.axvline(np.array([-20]) * units.degC, color='darkred')
    stuve.ax.axvline(-10, color='orange')

    # On Matplotlib <= 3.6, ax[hv]line() doesn't trigger unit labels
    assert stuve.ax.get_xlabel() == (
        'degree_Celsius' if version_check('matplotlib==3.7.0') else '')
    assert stuve.ax.get_ylabel() == (
        'hectopascal' if version_check('matplotlib==3.7.0') else '')

    # Clear them for the image test
    stuve.ax.set_xlabel('')
    stuve.ax.set_ylabel('')

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
    profile = np.array([18.2, 16.18287437, 15.68644745, 14.8369451,
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
def test_stuve_shade_cape_cin(test_profile):
    """Test shading CAPE and CIN on a Stuve plot."""
    p, t, td, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        stuve = Stuve(fig, aspect='auto')
        stuve.plot(p, t, 'r')
        stuve.plot(p, tp, 'k')
        stuve.shade_cape(p, t, tp)
        stuve.shade_cin(p, t, tp, td)
        stuve.ax.set_xlim(-50, 50)
        stuve.ax.set_ylim(1000, 100)

        # This works around the fact that newer pint versions default to degrees_Celsius
        stuve.ax.set_xlabel('degC')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.033, remove_text=True, style='default')
def test_stuve_shade_cape_cin_no_limit(test_profile):
    """Test shading CIN without limits."""
    p, t, _, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        stuve = Stuve(fig, aspect='auto')
        stuve.plot(p, t, 'r')
        stuve.plot(p, tp, 'k')
        stuve.shade_cape(p, t, tp)
        stuve.shade_cin(p, t, tp)
        stuve.ax.set_xlim(-50, 50)
        stuve.ax.set_ylim(1000, 100)

        # This works around the fact that newer pint versions default to degrees_Celsius
        stuve.ax.set_xlabel('degC')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.033, remove_text=True, style='default')
def test_stuve_shade_area(test_profile):
    """Test shading areas on a Stuve plot."""
    p, t, _, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        stuve = Stuve(fig, aspect='auto')
        stuve.plot(p, t, 'r')
        stuve.plot(p, tp, 'k')
        stuve.shade_area(p, t, tp)
        stuve.ax.set_xlim(-50, 50)
        stuve.ax.set_ylim(1000, 100)

        # This works around the fact that newer pint versions default to degrees_Celsius
        stuve.ax.set_xlabel('degC')

    return fig


def test_stuve_shade_area_invalid(test_profile):
    """Test shading areas on a Stuve plot."""
    p, t, _, tp = test_profile
    with autoclose_figure(figsize=(9, 9)) as fig:
        stuve = Stuve(fig, aspect='auto')
        stuve.plot(p, t, 'r')
        stuve.plot(p, tp, 'k')
        with pytest.raises(ValueError):
            stuve.shade_area(p, t, tp, which='positve')


@pytest.mark.mpl_image_compare(tolerance=0.033, remove_text=True, style='default')
def test_stuve_shade_area_kwargs(test_profile):
    """Test shading areas on a Stuve plot with kwargs."""
    p, t, _, tp = test_profile

    with matplotlib.rc_context({'axes.autolimit_mode': 'data'}):
        fig = plt.figure(figsize=(9, 9))
        stuve = Stuve(fig, aspect='auto')
        stuve.plot(p, t, 'r')
        stuve.plot(p, tp, 'k')
        stuve.shade_area(p, t, tp, facecolor='m')
        stuve.ax.set_xlim(-50, 50)
        stuve.ax.set_ylim(1000, 100)

        # This works around the fact that newer pint versions default to degrees_Celsius
        stuve.ax.set_xlabel('degC')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.039, remove_text=True, style='default')
def test_stuve_wide_aspect_ratio(test_profile):
    """Test plotting a stuveT with a wide aspect ratio."""
    p, t, _, tp = test_profile

    fig = plt.figure(figsize=(12.5, 3))
    stuve = Stuve(fig, aspect='auto')
    stuve.plot(p, t, 'r')
    stuve.plot(p, tp, 'k')
    stuve.ax.set_xlim(-30, 50)
    stuve.ax.set_ylim(1050, 700)

    # This works around the fact that newer pint versions default to degrees_Celsius
    stuve.ax.set_xlabel('degC')
    return fig


@pytest.mark.mpl_image_compare(tolerance=0.141, remove_text=True, style='default')
def test_stuve_barb_color():
    """Test plotting colored wind barbs on the Stuve."""
    fig = plt.figure(figsize=(9, 9))
    stuve = Stuve(fig, aspect='auto')

    p = np.linspace(1000, 100, 10)
    u = np.linspace(-10, 10, 10)
    stuve.plot_barbs(p, u, u, c=u)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.02, remove_text=True, style='default')
def test_stuve_barb_unit_conversion():
    """Test that barbs units can be converted at plot time (#737)."""
    u_wind = np.array([3.63767155210412]) * units('m/s')
    v_wind = np.array([3.63767155210412]) * units('m/s')
    p_wind = np.array([500]) * units.hPa

    fig = plt.figure(figsize=(9, 9))
    stuve = Stuve(fig, aspect='auto')
    stuve.ax.set_ylabel('')  # remove_text doesn't do this as of pytest 0.9
    stuve.plot_barbs(p_wind, u_wind, v_wind, plot_units='knots')
    stuve.ax.set_ylim(1000, 500)
    stuve.ax.set_yticks([1000, 750, 500])
    stuve.ax.set_xlim(-20, 20)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.02, remove_text=True, style='default')
def test_stuve_barb_no_default_unit_conversion():
    """Test that barbs units are left alone by default (#737)."""
    u_wind = np.array([3.63767155210412]) * units('m/s')
    v_wind = np.array([3.63767155210412]) * units('m/s')
    p_wind = np.array([500]) * units.hPa

    fig = plt.figure(figsize=(9, 9))
    stuve = Stuve(fig, aspect='auto')
    stuve.ax.set_ylabel('')  # remove_text doesn't do this as of pytest 0.9
    stuve.plot_barbs(p_wind, u_wind, v_wind)
    stuve.ax.set_ylim(1000, 500)
    stuve.ax.set_yticks([1000, 750, 500])
    stuve.ax.set_xlim(-20, 20)

    return fig


@pytest.mark.parametrize('u,v', [(np.array([3]) * units('m/s'), np.array([3])),
                                 (np.array([3]), np.array([3]) * units('m/s'))])
def test_stuve_barb_unit_conversion_exception(u, v):
    """Test that an error is raised if unit conversion is requested on plain arrays."""
    p_wind = np.array([500]) * units.hPa

    with autoclose_figure(figsize=(9, 9)) as fig:
        stuve = Stuve(fig, aspect='auto')
        with pytest.raises(ValueError):
            stuve.plot_barbs(p_wind, u, v, plot_units='knots')
