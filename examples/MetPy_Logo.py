# Copyright (c) 2008-2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
==========
MetPy Logo
==========

Make the MetPy logo.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.patches import Ellipse

from metpy.cbook import get_test_data
from metpy.io import Level2File
from metpy.plots import ctables
from metpy.plots import simple_layout, StationPlot, StationPlotLayout
from metpy.plots.wx_symbols import current_weather, sky_cover
from metpy.units import units
from metpy.io import get_upper_air_data
from metpy.plots import Hodograph, SkewT

from datetime import datetime
import matplotlib.gridspec as gridspec


mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.edgecolor'] = 'gray'


axalpha = 0.05
figcolor = 'white'
dpi = 80
fig = plt.figure(figsize=(6, 6), dpi=dpi)
fig.figurePatch.set_edgecolor(figcolor)
fig.figurePatch.set_facecolor(figcolor)

def add_background():
    ax = fig.add_axes([0., 0., 1., 1.], projection='polar')
    ax.set_facecolor('#275196')
    return ax

def add_overlay():
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.patch.set_alpha(0)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    return ax

def add_skewt(fig):
    dataset = get_upper_air_data(datetime(2002, 11, 11, 0), 'BNA')

    p = dataset.variables['pressure'][:]
    T = dataset.variables['temperature'][:]
    Td = dataset.variables['dewpoint'][:]
    u = dataset.variables['u_wind'][:]
    v = dataset.variables['v_wind'][:]

    # Create a new figure. The dimensions here give a good aspect ratio
    #fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 3)
    gs.update(right=0.78, top=0.75, bottom=0.1)
    skew = SkewT(fig, subplot=gs[1, 2])

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p, T, 'r', linewidth=1)
    skew.plot(p, Td, 'g', linewidth=1)
    #skew.plot_barbs(p, u, v)

    skew.ax.patch.set_alpha(0.3)
    skew.ax.set_ylim(1000,100)

    skew.ax.xaxis.set_ticklabels([])
    skew.ax.yaxis.set_ticklabels([])



def add_metpy_text(ax):
    ax.text(0.5, 0.5, 'MetPy', family='fantasy', color='#f7f9fc', fontsize=135,
            ha='center', va='center', alpha=1.0, transform=ax.transAxes)

def add_anemometer(ax):
    x_center = 0.115
    y_center = 0.26

    # Left Arm
    ax.plot([0 + x_center, 0.08 + x_center], [0 + y_center, -0.035 + y_center], color='#f7f9fc', linewidth=1)

    # Right Arm
    ax.plot([0 + x_center, -0.08 + x_center], [0 + y_center, -0.035 + y_center], color='#f7f9fc', linewidth=1)

    # Vertical Arm
    ax.plot([0 + x_center, 0 + x_center], [0 + y_center, 0.07 + y_center], color='#f7f9fc', linewidth=1)

    # Left Cup
    ax.add_artist(Ellipse(xy=(-0.08+x_center, -0.035+y_center), width=0.06, height=0.07, facecolor='#f7f9fc'))

    # Right Cup
    ax.add_artist(Ellipse(xy=(0.08+x_center, -0.035+y_center), width=0.06, height=0.07, facecolor='#f7f9fc'))

    # Back Cup

    # Front, narrow
    ax.add_artist(Ellipse(xy=(0+x_center+.005, y_center+0.07), width=0.045, height=0.065, facecolor='#f7f9fc'))

    return ax

def add_radar(ax):

    name = get_test_data('KTLX20130520_201643_V06.gz', as_file_obj=False)
    #f = Level2File('KTLX20130520_201643_V06.gz')
    f = Level2File(name)

    # Pull data out of the file
    sweep = 0

    # First item in ray is header, which has azimuth angle
    az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])

    # Pull and mask reflectivity data
    ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
    var_data = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])
    var_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    cmap = ctables.registry.get_colortable('viridis')
    print(np.shape(az))
    print(np.shape(var_range))
    print(np.shape(data))
    ax.pcolormesh(np.radians(45-az), var_range, data.transpose(), cmap=cmap)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Make a "Sweep" Looking Bar
    bars = ax.bar([20],[30], width=np.pi / 128, bottom=0.0)
    for bar in bars:
        bar.set_facecolor('#ffffff')
        bar.set_alpha(0.5)



def add_station_plot(ax):
    # This is our container for the data
    data = dict()

    # Copy out to stage everything together. In an ideal world, this would happen on
    # the data reading side of things, but we're not there yet.
    data['longitude'] = np.array([np.radians(215.)])
    data['latitude'] = np.array([18.])
    data['air_temperature'] = np.array([21.]) * units.degC
    data['dewpoint'] = np.array([18.])# * units.degC
    data['eastward_wind'] = np.array([-50.])
    data['northward_wind'] = np.array([50.])
    data['cloud_coverage'] = np.array([6])
    data['present_weather'] = np.array([17])

    stationplot = StationPlot(ax, data['longitude'], data['latitude'], fontsize=40)

    # Plot the temperature and dew point to the upper and lower left, respectively, of
    # the center point. Each one uses a different color.
    #stationplot.plot_parameter('NW', data['air_temperature'], color='red', fontsize=20)
    #stationplot.plot_parameter('SW', data['dewpoint'], color='darkgreen', fontsize=20)

    # Plot the cloud cover symbols in the center location. This uses the codes made above and
    # uses the `sky_cover` mapper to convert these values to font codes for the
    # weather symbol font.
    stationplot.plot_symbol('C', data['cloud_coverage'], sky_cover)

    # Same this time, but plot current weather to the left of center, using the
    # `current_weather` mapper to convert symbols to the right glyphs.
    stationplot.plot_symbol('W', data['present_weather'], current_weather)

    # Add wind barbs
    stationplot.plot_barb(data['eastward_wind'], data['northward_wind'])



if __name__ == '__main__':
    main_axes = add_background()
    overlay_axes = add_overlay()
    add_metpy_text(main_axes)
    add_radar(main_axes)
    add_station_plot(main_axes)
    main_axes.set_rmax(30)
    add_anemometer(overlay_axes)
    add_skewt(fig)
    plt.savefig('metpylogo.png')
    plt.show()
