# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
'''Make VAD plot from NEXRAD Level 3 VAD files'''

import datetime as dt
from io import StringIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metpy.io import Level3File


def plot_vad(filelist, storm_motion=None):
    '''INPUT: list of NEXRAD VAD files
       OUTPUT: color-coded wind barbs plotted with height for
               the times given in the list of files

        plotting a storm-relative VAD is optional
        (requires a storm motion vector)'''
    vad_pd = pd.DataFrame()
    for files in filelist:
        vad_f = Level3File(files)
        vad_ft = vad_f.tab_pages
        lst = np.arange(0, len(vad_ft)-2)
        vad_list = [pd.read_table(StringIO(vad_ft[i]), sep='\s+', header=1,
                    skiprows=[2]) for i in lst]
        vad = pd.concat(vad_list, axis=0)
        vad['TIME'] = vad_f.metadata['vol_time']
        # convert from altitude in feet to AGL in meters
        vad['AGL'] = ((vad['ALT']*100.)-vad_f.height)*0.3048
        vad_pd = pd.concat([vad_pd, vad], axis=0)

    # Begin Figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    uplot = vad_pd['U']
    vplot = vad_pd['V']
    time_plot = mpl.dates.date2num(vad_pd['TIME'].astype(dt.datetime))

    # Subtract storm motion vector from winds for storm-relative VAD
    if storm_motion is not None:
        uplot, vplot = vad_pd['U']-storm_motion[0], vad_pd['V']-storm_motion[1]

    # Set color params
    c = np.sqrt(uplot**2 + vplot**2)  # color by speed
    bounds = np.arange(0, 100, 1)  # min speed, max speed, interval

    # plot wind barbs
    cmap = plt.cm.gnuplot
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    m = 2  # plot every mth wind barb
    b = plt.barbs(time_plot[::m], vad_pd['AGL'].iloc[::m], uplot.iloc[::m],
                  vplot.iloc[::m], c[::m], cmap=cmap, norm=norm)

    # Assign tick labels for x-axis
    starttime, endtime = vad_pd['TIME'].iloc[0], vad_pd['TIME'].iloc[-1]
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H:%M UTC'))
    ax.set_xlim(starttime-dt.timedelta(minutes=2), endtime+dt.timedelta(minutes=2))

    # Plot asthetics
    fs = 14
    cbar = plt.colorbar(b, cmap=cmap, boundaries=bounds, norm=norm)  # Colorbar
    cbar.set_label('Wind Speed (m/s)', fontsize=fs)
    yticks = np.arange(0, 10500, 500)
    plt.yticks(yticks)
    plt.ylim(-500, 10500)
    plt.xlabel('Time (UTC)', fontsize=fs)
    plt.ylabel('Altitude AGL (m)', fontsize=fs)
    plt.grid('on')
    plt.title(vad_f.siteID+' Velocity Azimuth Display', fontsize=fs+3)
