# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from matplotlib import style
import pytest

from metpy.plots.station_plot import StationPlot, StationPlotLayout, nws_layout, simple_layout
from metpy.plots.wx_symbols import high_clouds, sky_cover
from metpy.testing import hide_tick_labels, make_figure, test_style
from metpy.units import units


def setup_font():
    'Changes font settings for better test reliability.'
    from matplotlib import rcParams
    rcParams['font.family'] = 'Bitstream Vera Sans'
    rcParams['text.hinting'] = False
    rcParams['text.hinting_factor'] = 8


@pytest.mark.mpl_image_compare(tolerance=6, savefig_kwargs={'dpi': 300})
def test_stationplot_api():
    'Test the StationPlot api'
    setup_font()
    with style.context(test_style):
        fig = make_figure(figsize=(9, 9))

        # testing data
        x = np.array([1, 5])
        y = np.array([2, 4])

        # Make the plot
        sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=16)
        sp.plot_barb([20, 0], [0, -50])
        sp.plot_text('E', ['KOKC', 'ICT'], color='blue')
        sp.plot_parameter('NW', [10.5, 15], color='red')
        sp.plot_symbol('S', [5, 7], high_clouds, color='green')

        sp.ax.set_xlim(0, 6)
        sp.ax.set_ylim(0, 6)
        hide_tick_labels(sp.ax)

        return fig


@pytest.mark.mpl_image_compare(tolerance=8, savefig_kwargs={'dpi': 300})
def test_station_plot_replace():
    'Test that locations are properly replaced'
    setup_font()
    with style.context(test_style):
        fig = make_figure(figsize=(3, 3))

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
        hide_tick_labels(sp.ax)

        return fig


@pytest.mark.mpl_image_compare(tolerance=4, savefig_kwargs={'dpi': 300})
def test_stationlayout_api():
    'Test the StationPlot api'
    setup_font()
    with style.context(test_style):
        fig = make_figure(figsize=(9, 9))

        # testing data
        x = np.array([1, 5])
        y = np.array([2, 4])
        data = dict()
        data['temp'] = np.array([32., 212.]) * units.degF
        data['u'] = np.array([2, 0]) * units.knots
        data['v'] = np.array([0, 5]) * units.knots
        data['stid'] = ['KDEN', 'KSHV']
        data['cover'] = [3, 8]

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
        hide_tick_labels(sp.ax)

        return fig


def test_station_layout_odd_data():
    'Test more corner cases with data passed in'
    setup_font()
    fig = make_figure(figsize=(9, 9))

    # Set up test layout
    layout = StationPlotLayout()
    layout.add_barb('u', 'v')
    layout.add_value('W', 'temperature', units='degF')

    # Now only use data without wind and no units
    data = dict(temperature=[25.])

    # Make the plot
    sp = StationPlot(fig.add_subplot(1, 1, 1), [1], [2], fontsize=12)
    layout.plot(sp, data)
    assert True


def test_station_layout_replace():
    'Test that layout locations are replaced'
    layout = StationPlotLayout()
    layout.add_text('E', 'temperature')
    layout.add_value('E', 'dewpoint')
    assert 'E' in layout
    assert layout['E'][0] is StationPlotLayout.PlotTypes.value
    assert layout['E'][1] == 'dewpoint'


def test_station_layout_names():
    'Test getting station layout names'
    layout = StationPlotLayout()
    layout.add_barb('u', 'v')
    layout.add_text('E', 'stid')
    layout.add_value('W', 'temp')
    layout.add_symbol('C', 'cover', lambda x: x)
    assert sorted(layout.names()) == ['cover', 'stid', 'temp', 'u', 'v']


@pytest.mark.mpl_image_compare(tolerance=6, savefig_kwargs={'dpi': 300})
def test_simple_layout():
    'Test metpy\'s simple layout for station plots'
    setup_font()
    with style.context(test_style):
        fig = make_figure(figsize=(9, 9))

        # testing data
        x = np.array([1, 5])
        y = np.array([2, 4])
        data = dict()
        data['air_temperature'] = np.array([32., 212.]) * units.degF
        data['dew_point_temperature'] = np.array([28., 80.]) * units.degF
        data['air_pressure_at_sea_level'] = np.array([29.92, 28.00]) * units.inHg
        data['eastward_wind'] = np.array([2, 0]) * units.knots
        data['northward_wind'] = np.array([0, 5]) * units.knots
        data['cloud_coverage'] = [3, 8]
        data['present_weather'] = [65, 75]
        data['unused'] = [1, 2]

        # Make the plot
        sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=12)
        simple_layout.plot(sp, data)

        sp.ax.set_xlim(0, 6)
        sp.ax.set_ylim(0, 6)
        hide_tick_labels(sp.ax)

        return fig


@pytest.mark.mpl_image_compare(tolerance=18.5, savefig_kwargs={'dpi': 300})
def test_nws_layout():
    'Test metpy\'s NWS layout for station plots'
    setup_font()
    with style.context(test_style):
        fig = make_figure(figsize=(3, 3))

        # testing data
        x = np.array([1])
        y = np.array([2])
        data = dict()
        data['air_temperature'] = np.array([77]) * units.degF
        data['dew_point_temperature'] = np.array([71]) * units.degF
        data['air_pressure_at_sea_level'] = np.array([999.8]) * units('mbar')
        data['eastward_wind'] = np.array([15.]) * units.knots
        data['northward_wind'] = np.array([15.]) * units.knots
        data['cloud_coverage'] = [7]
        data['present_weather'] = [80]
        data['high_cloud_type'] = [1]
        data['medium_cloud_type'] = [3]
        data['low_cloud_type'] = [2]
        data['visibility_in_air'] = np.array([5.]) * units.mile
        data['tendency_of_air_pressure'] = np.array([-0.3]) * units('mbar')
        data['tendency_of_air_pressure_symbol'] = [8]

        # Make the plot
        sp = StationPlot(fig.add_subplot(1, 1, 1), x, y, fontsize=12, spacing=16)
        nws_layout.plot(sp, data)

        sp.ax.set_xlim(0, 3)
        sp.ax.set_ylim(0, 3)
        hide_tick_labels(sp.ax)

        return fig


def test_layout_str():
    'Test layout string representation'
    layout = StationPlotLayout()
    layout.add_barb('u', 'v')
    layout.add_text('E', 'stid')
    layout.add_value('W', 'temp')
    layout.add_symbol('C', 'cover', lambda x: x)
    assert str(layout) == ('{C: (symbol, cover, ...), E: (text, stid, ...), '
                           'W: (value, temp, ...), barb: (barb, (\'u\', \'v\'), ...)}')
