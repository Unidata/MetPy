# Copyright (c) 2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `station_plot` module."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from metpy.plots import (current_weather, high_clouds, nws_layout, simple_layout, sky_cover,
                         StationPlot, StationPlotLayout)
from metpy.units import units


@pytest.mark.mpl_image_compare(tolerance=2.444, savefig_kwargs={'dpi': 300}, remove_text=True)
def test_stationplot_api():
    """Test the StationPlot API."""
    fig = plt.figure(figsize=(9, 9))

    # testing data
    x = np.array([1, 5])
    y = np.array([2, 4])

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=16)
    sp.plot_barb([20, 0], [0, -50])
    sp.plot_text('E', ['KOKC', 'ICT'], color='blue')
    sp.plot_parameter('NW', [10.5, 15] * units.degC, color='red')
    sp.plot_symbol('S', [5, 7], high_clouds, color='green')

    sp.ax.set_xlim(0, 6)
    sp.ax.set_ylim(0, 6)

    return fig


@pytest.mark.mpl_image_compare(tolerance=1.976, savefig_kwargs={'dpi': 300}, remove_text=True)
def test_stationplot_clipping():
    """Test the that clipping can be enabled as a default parameter."""
    fig = plt.figure(figsize=(9, 9))

    # testing data
    x = np.array([1, 5])
    y = np.array([2, 4])

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=16, clip_on=True)
    sp.plot_barb([20, 0], [0, -50])
    sp.plot_text('E', ['KOKC', 'ICT'], color='blue')
    sp.plot_parameter('NW', [10.5, 15] * units.degC, color='red')
    sp.plot_symbol('S', [5, 7], high_clouds, color='green')

    sp.ax.set_xlim(1, 5)
    sp.ax.set_ylim(1.75, 4.25)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.25, savefig_kwargs={'dpi': 300}, remove_text=True)
def test_station_plot_replace():
    """Test that locations are properly replaced."""
    fig = plt.figure(figsize=(3, 3))

    # testing data
    x = np.array([1])
    y = np.array([1])

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=16)
    sp.plot_barb([20], [0])
    sp.plot_barb([5], [0])
    sp.plot_parameter('NW', [10.5], color='red')
    sp.plot_parameter('NW', [20], color='blue')

    sp.ax.set_xlim(-3, 3)
    sp.ax.set_ylim(-3, 3)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.25, savefig_kwargs={'dpi': 300}, remove_text=True)
def test_station_plot_locations():
    """Test that locations are properly replaced."""
    fig = plt.figure(figsize=(3, 3))

    locations = ['C', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N2', 'NNE', 'ENE', 'E2',
                 'ESE', 'SSE', 'S2', 'SSW', 'WSW', 'W2', 'WNW', 'NNW']
    x_pos = np.array([0])
    y_pos = np.array([0])

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), x_pos, y_pos, fontsize=8, spacing=24)
    for loc in locations:
        sp.plot_text(loc, [loc])

    sp.ax.set_xlim(-2, 2)
    sp.ax.set_ylim(-2, 2)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.00413, savefig_kwargs={'dpi': 300},
                               remove_text=True)
def test_stationlayout_api():
    """Test the StationPlot API."""
    fig = plt.figure(figsize=(9, 9))

    # testing data
    x = np.array([1, 5])
    y = np.array([2, 4])
    data = {'temp': np.array([33., 212.]) * units.degF, 'u': np.array([2, 0]) * units.knots,
            'v': np.array([0, 5]) * units.knots, 'stid': ['KDEN', 'KSHV'], 'cover': [3, 8]}

    # Set up the layout
    layout = StationPlotLayout()
    layout.add_barb('u', 'v', units='knots')
    layout.add_value('NW', 'temp', fmt='0.1f', units=units.degC, color='darkred')
    layout.add_symbol('C', 'cover', sky_cover, color='magenta')
    layout.add_text((0, 2), 'stid', color='darkgrey')
    layout.add_value('NE', 'dewpt', color='green')  # This should be ignored

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=12)
    layout.plot(sp, data)

    sp.ax.set_xlim(0, 6)
    sp.ax.set_ylim(0, 6)

    return fig


def test_station_layout_odd_data():
    """Test more corner cases with data passed in."""
    fig = plt.figure(figsize=(9, 9))

    # Set up test layout
    layout = StationPlotLayout()
    layout.add_barb('u', 'v')
    layout.add_value('W', 'temperature', units='degF')

    # Now only use data without wind and no units
    data = {'temperature': [25.]}

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), [1], [2], fontsize=12)
    layout.plot(sp, data)
    assert True


def test_station_layout_replace():
    """Test that layout locations are replaced."""
    layout = StationPlotLayout()
    layout.add_text('E', 'temperature')
    layout.add_value('E', 'dewpoint')
    assert 'E' in layout
    assert layout['E'][0] is StationPlotLayout.PlotTypes.value
    assert layout['E'][1] == 'dewpoint'


def test_station_layout_names():
    """Test getting station layout names."""
    layout = StationPlotLayout()
    layout.add_barb('u', 'v')
    layout.add_text('E', 'stid')
    layout.add_value('W', 'temp')
    layout.add_symbol('C', 'cover', lambda x: x)
    assert sorted(layout.names()) == ['cover', 'stid', 'temp', 'u', 'v']


@pytest.mark.mpl_image_compare(tolerance=0, savefig_kwargs={'dpi': 300}, remove_text=True)
def test_simple_layout():
    """Test metpy's simple layout for station plots."""
    fig = plt.figure(figsize=(9, 9))

    # testing data
    x = np.array([1, 5])
    y = np.array([2, 4])
    data = {'air_temperature': np.array([33., 212.]) * units.degF,
            'dew_point_temperature': np.array([28., 80.]) * units.degF,
            'air_pressure_at_sea_level': np.array([29.92, 28.00]) * units.inHg,
            'eastward_wind': np.array([2, 0]) * units.knots,
            'northward_wind': np.array([0, 5]) * units.knots, 'cloud_coverage': [3, 8],
            'current_wx1_symbol': [65, 75], 'unused': [1, 2]}

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=12)
    simple_layout.plot(sp, data)

    sp.ax.set_xlim(0, 6)
    sp.ax.set_ylim(0, 6)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.688, savefig_kwargs={'dpi': 300}, remove_text=True)
def test_nws_layout():
    """Test metpy's NWS layout for station plots."""
    fig = plt.figure(figsize=(3, 3))

    # testing data
    x = np.array([1])
    y = np.array([2])
    data = {'air_temperature': np.array([77]) * units.degF,
            'dew_point_temperature': np.array([71]) * units.degF,
            'air_pressure_at_sea_level': np.array([999.8]) * units('mbar'),
            'eastward_wind': np.array([15.]) * units.knots,
            'northward_wind': np.array([15.]) * units.knots, 'cloud_coverage': [7],
            'current_wx1_symbol': [80], 'high_cloud_type': [1], 'medium_cloud_type': [3],
            'low_cloud_type': [2], 'visibility_in_air': np.array([5.]) * units.mile,
            'tendency_of_air_pressure': np.array([-0.3]) * units('mbar'),
            'tendency_of_air_pressure_symbol': [8]}

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=12, spacing=16)
    nws_layout.plot(sp, data)

    sp.ax.set_xlim(0, 3)
    sp.ax.set_ylim(0, 3)

    return fig


@pytest.mark.mpl_image_compare(tolerance=1.05, remove_text=True)
def test_plot_text_fontsize():
    """Test changing fontsize in plot_text."""
    fig = plt.figure(figsize=(3, 3))
    ax = plt.subplot(1, 1, 1)

    # testing data
    x = np.array([1])
    y = np.array([2])

    # Make the plot
    sp = StationPlot(ax, x, y, fontsize=36)
    sp.plot_text('NW', ['72'], fontsize=24)
    sp.plot_text('SW', ['60'], fontsize=4)

    sp.ax.set_xlim(0, 3)
    sp.ax.set_ylim(0, 3)

    return fig


@pytest.mark.mpl_image_compare(tolerance=1.05, remove_text=True)
def test_plot_symbol_fontsize():
    """Test changing fontsize in plotting of symbols."""
    fig = plt.figure(figsize=(3, 3))
    ax = plt.subplot(1, 1, 1)

    sp = StationPlot(ax, [0], [0], fontsize=8, spacing=32)
    sp.plot_symbol('E', [92], current_weather)
    sp.plot_symbol('W', [96], current_weather, fontsize=100)

    return fig


def test_layout_str():
    """Test layout string representation."""
    layout = StationPlotLayout()
    layout.add_barb('u', 'v')
    layout.add_text('E', 'stid')
    layout.add_value('W', 'temp')
    layout.add_symbol('C', 'cover', lambda x: x)
    assert str(layout) == ('{C: (symbol, cover, ...), E: (text, stid, ...), '
                           "W: (value, temp, ...), barb: (barb, ('u', 'v'), ...)}")


@pytest.fixture
def wind_plot():
    """Create southerly wind test data."""
    v = np.full((5, 5), 10, dtype=np.float64)
    u = np.zeros_like(v)
    x, y = np.meshgrid(np.linspace(-120, -60, 5), np.linspace(25, 50, 5))
    return u, v, x, y


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.499)
def test_barb_projection(wind_plot, ccrs):
    """Test that barbs are properly projected (#598)."""
    u, v, x, y = wind_plot

    # Plot and check barbs (they should align with grid lines)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
    ax.gridlines(xlocs=[-120, -105, -90, -75, -60], ylocs=np.arange(24, 55, 6))
    sp = StationPlot(ax, x, y, transform=ccrs.PlateCarree())
    sp.plot_barb(u, v)

    return fig


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.01)
def test_arrow_projection(wind_plot, ccrs):
    """Test that arrows are properly projected."""
    u, v, x, y = wind_plot

    # Plot and check barbs (they should align with grid lines)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
    ax.gridlines(xlocs=[-120, -105, -90, -75, -60], ylocs=np.arange(24, 55, 6))
    sp = StationPlot(ax, x, y, transform=ccrs.PlateCarree())
    sp.plot_arrow(u, v)
    sp.plot_arrow(u, v)  # plot_arrow used twice to hit removal if statement

    return fig


@pytest.fixture
def wind_projection_list():
    """Create wind lists for testing."""
    lat = [38.22, 38.18, 38.25]
    lon = [-85.76, -85.86, -85.77]
    u = [1.89778964, -3.83776523, 3.64147732] * units('m/s')
    v = [1.93480072, 1.31000184, 1.36075552] * units('m/s')
    return lat, lon, u, v


def test_barb_projection_list(wind_projection_list):
    """Test that barbs will be projected when lat/lon lists are provided."""
    lat, lon, u, v = wind_projection_list

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    stnplot = StationPlot(ax, lon, lat)
    stnplot.plot_barb(u, v)
    assert stnplot.barbs


def test_arrow_projection_list(wind_projection_list):
    """Test that arrows will be projected when lat/lon lists are provided."""
    lat, lon, u, v = wind_projection_list

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    stnplot = StationPlot(ax, lon, lat)
    stnplot.plot_arrow(u, v)
    assert stnplot.arrows


@pytest.fixture
def barbs_units():
    """Create barbs with units for testing."""
    x_pos = np.array([0])
    y_pos = np.array([0])
    u_wind = np.array([3.63767155210412]) * units('m/s')
    v_wind = np.array([3.63767155210412]) * units('m/s')
    return x_pos, y_pos, u_wind, v_wind


@pytest.mark.mpl_image_compare(tolerance=0.0048, remove_text=True)
def test_barb_unit_conversion(barbs_units):
    """Test that barbs units can be converted at plot time (#737)."""
    x_pos, y_pos, u_wind, v_wind = barbs_units

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    stnplot = StationPlot(ax, x_pos, y_pos)
    stnplot.plot_barb(u_wind, v_wind, plot_units='knots')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.0048, remove_text=True)
def test_arrow_unit_conversion(barbs_units):
    """Test that arrow units can be converted at plot time (#737)."""
    x_pos, y_pos, u_wind, v_wind = barbs_units

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    stnplot = StationPlot(ax, x_pos, y_pos)
    stnplot.plot_arrow(u_wind, v_wind, plot_units='knots')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.0878, remove_text=True)
def test_barb_no_default_unit_conversion():
    """Test that barbs units are left alone by default (#737)."""
    x_pos = np.array([0])
    y_pos = np.array([0])
    u_wind = np.array([3.63767155210412]) * units('m/s')
    v_wind = np.array([3.63767155210412]) * units('m/s')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    stnplot = StationPlot(ax, x_pos, y_pos)
    stnplot.plot_barb(u_wind, v_wind)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    return fig


@pytest.mark.parametrize('u,v', [(np.array([3]) * units('m/s'), np.array([3])),
                                 (np.array([3]), np.array([3]) * units('m/s'))])
def test_barb_unit_conversion_exception(u, v):
    """Test that an error is raised if unit conversion is requested on plain arrays."""
    x_pos = np.array([0])
    y_pos = np.array([0])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    stnplot = StationPlot(ax, x_pos, y_pos)
    with pytest.raises(ValueError):
        stnplot.plot_barb(u, v, plot_units='knots')


@pytest.mark.mpl_image_compare(tolerance=0.021, savefig_kwargs={'dpi': 300}, remove_text=True)
def test_symbol_pandas_timeseries():
    """Test the usage of Pandas DatetimeIndex as a valid `x` input into StationPlot."""
    pd.plotting.register_matplotlib_converters()
    rng = pd.date_range('12/1/2017', periods=5, freq='D')
    sc = [1, 2, 3, 4, 5]
    ts = pd.Series(sc, index=rng)
    fig, ax = plt.subplots()
    y = np.ones(len(ts.index))
    stationplot = StationPlot(ax, ts.index, y, fontsize=12)
    stationplot.plot_symbol('C', ts, sky_cover)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%-d'))

    return fig


@pytest.mark.mpl_image_compare(tolerance=2.444, savefig_kwargs={'dpi': 300}, remove_text=True)
def test_stationplot_unit_conversion():
    """Test the StationPlot API."""
    fig = plt.figure(figsize=(9, 9))

    # testing data
    x = np.array([1, 5])
    y = np.array([2, 4])

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=16)
    sp.plot_barb([20, 0], [0, -50])
    sp.plot_text('E', ['KOKC', 'ICT'], color='blue')
    sp.plot_parameter('NW', [10.5, 15] * units.degC, plot_units='degF', color='red')
    sp.plot_symbol('S', [5, 7], high_clouds, color='green')

    sp.ax.set_xlim(0, 6)
    sp.ax.set_ylim(0, 6)

    return fig


def test_scalar_unit_conversion_exception():
    """Test that an error is raised if unit conversion is requested on plain arrays."""
    x_pos = np.array([0])
    y_pos = np.array([0])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    stnplot = StationPlot(ax, x_pos, y_pos)
    with pytest.raises(ValueError):
        stnplot.plot_parameter('C', 50, plot_units='degC')
