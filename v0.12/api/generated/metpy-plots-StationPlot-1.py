import matplotlib.pyplot as plt
import numpy as np
from math import ceil

from metpy.plots import StationPlot
from metpy.plots.wx_symbols import current_weather, current_weather_auto
from metpy.plots.wx_symbols import low_clouds, mid_clouds, high_clouds
from metpy.plots.wx_symbols import sky_cover, pressure_tendency


def plot_symbols(mapper, name, nwrap=12, figsize=(10, 1.4)):

    # Determine how many symbols there are and layout in rows of nwrap
    # if there are more than nwrap symbols
    num_symbols = len(mapper)
    codes = np.arange(len(mapper))
    ncols = nwrap
    if num_symbols <= nwrap:
        nrows = 1
        x = np.linspace(0, 1, len(mapper))
        y = np.ones_like(x)
        ax_height = 0.8
    else:
        nrows = int(ceil(num_symbols / ncols))
        x = np.tile(np.linspace(0, 1, ncols), nrows)[:num_symbols]
        y = np.repeat(np.arange(nrows, 0, -1), ncols)[:num_symbols]
        figsize = (10, 1 * nrows + 0.4)
        ax_height = 0.8 + 0.018 * nrows

    fig = plt.figure(figsize=figsize,  dpi=300)
    ax = fig.add_axes([0, 0, 1, ax_height])
    ax.set_title(name, size=20)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)

    # Plot
    sp = StationPlot(ax, x, y, fontsize=36)
    sp.plot_symbol('C', codes, mapper)
    sp.plot_parameter((0, -1), codes, fontsize=18)

    ax.set_ylim(-0.05, nrows + 0.5)

    plt.show()


plot_symbols(current_weather, "Current Weather Symbols")
plot_symbols(current_weather_auto, "Current Weather Auto Reported Symbols")
plot_symbols(low_clouds, "Low Cloud Symbols")
plot_symbols(mid_clouds, "Mid Cloud Symbols")
plot_symbols(high_clouds, "High Cloud Symbols")
plot_symbols(sky_cover, "Sky Cover Symbols")
plot_symbols(pressure_tendency, "Pressure Tendency Symbols")