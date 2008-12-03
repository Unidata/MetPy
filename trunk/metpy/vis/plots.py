__all__ = ['meteogram']

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.dates import DateFormatter

#WORK IN PROGRESS
#TODO:
#   *linestyles should map variable names to dictionary of matplotlib
#       keywords, plus a keyword 'fill'
#   *implement support for mesonet style filled plots
#   *add support for specifying units
#   *add support for specifying data limits
#   *figure out how to specify x-axis limits in time and allow those
#       to be set by user
#   *Update documentation for existing options and documentation for
#       new options
#Elsewhere, need a dewpoint calculation function

def meteogram(data, layout=None, linestyles=None, field_info=None):
    '''
    Plots a meteogram (collection of time series) for a data set. This
    is broken down into a series of panels (defaults to 3), each of which
    can plot multiple variables, with sensible defaults, but can also be
    specified using *layout*.

    *data* : numpy record array
        A numpy record array containing time series for individual variables
        in each field.

    *layout* : dictionary
        A dictionary that maps panel numbers to lists of variables.
        The maximum panel number is used as the total number of panels.

    *linestyles* : dictionary
        A dictionary that maps variable names to linestyles.

    *field_info* : dictionary
        A dictionary that maps standard names, like 'temperature' or
        'dewpoint' to the respective variables in the data record array.
    '''

    if field_info is None:
        field_info = {}

    inv_field_info = dict(zip(field_info.values(), field_info.keys()))

    def map_field(name):
        return field_info.get(name, name)

    #List of variables in each panel.  None denotes that at that point, twinx
    #should be called and the remaining variables plotted on the other axis
    default_layout = {
        0:[map_field('temperature'), None, map_field('relative humidity')],
        1:[map_field('wind speed'), map_field('wind gusts'), None,
            map_field('wind direction')],
        2:[map_field('pressure')],
        3:[map_field('rainfall')],
        4:[map_field('solar radiation')]}

    if layout is not None:
        default_layout.update(layout)
    layout = default_layout

    default_linestyles = {map_field('relative humidity'):('green','fill'),
        map_field('temperature'):('red', 'fill'),
        map_field('pressure'):('brown', 'fill'),
        map_field('dewpoint'):('green' 'fill'),
        map_field('wind speed'):('blue','fill'),
        map_field('wind gusts'):('lightblue', 'fill'),
        map_field('wind direction'):('goldenrod', 'o'),
        map_field('rainfall'):('lightgreen', 'fill'),
        map_field('solar radiation'):('orange', 'fill')}

    if linestyles is not None:
        default_linestyles.update(linestyles)
    linestyles = default_linestyles

    #Get the time variable, using field info if necessary
    dt = field_info.get('datetime', 'datetime')
    time = data[dt]

    #Get the station name, using field info if necessary
    site_name = field_info.get('site', 'site')
    site = data[site_name][0]
    num_panels = max(layout.keys()) + 1

    #Get the date from the first time
    date = time[0].strftime('%y-%m-%d')

    axes = []
    for panel in range(num_panels):
        if panel > 0:
            ax = plt.subplot(num_panels, 1, panel+1, sharex=ax)
        else:
            ax = plt.subplot(num_panels, 1, panel+1)
            ax.set_title('Meteogram for %s on %s' % (site, date))
        for varname in layout[panel]:
            if varname is None:
                ax = plt.twinx(ax)
                continue

            # Get the linestyle for this variable
            color,style = linestyles.get(varname, ('k', '-'))

            #TODO This needs to be implemented
            if style == 'fill':
                style = '-'

            #Get the variable from the data and plot
            var = data[map_field(varname)]
            ax.plot(time, var, style, color=color)

            # Set the label to the title-cased nice-version from the
            # field info.
            ax.set_ylabel(inv_field_info[varname].title())

        # If this is not the last panel, turn off the lower tick labels
        ax.xaxis.set_major_formatter(DateFormatter('%H'))
        axes.append(ax)
        ax.set_xlabel('Time (UTC)')

    return axes

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from matplotlib.cbook import delete_masked_points

def scalar_label(ax, x, y, data, format, loc='C', **kw):
    offset_lookup = dict(C=(0,0), N=(0,1), NE=(1,1), E=(1,0), SE=(1,-1),
        S=(0,-1), SW=(-1,-1), W=(-1,0), NW=(-1,1))
    x,y,data = delete_masked_points(x, y, data)
    trans = ax.transData
    for xi, yi, d in zip(x, y, data):
        ax.text(xi, yi, format % d, transform=trans, ha='center', va='center',
            **kw)

def station_plot(data):
    import matplotlib.pyplot as plt    
    from matplotlib import transforms
    from mpl_toolkits.basemap import Basemap

    kts_per_ms = 1.94384
    
    temp = data['temp']
    rh = data['relh']
    es = 6.112 * np.exp(17.67 * temp/(temp + 243.5))
    e = rh/100. * es
    Td = 243.5/(17.67/np.log(e/6.112) - 1)
    temp = temp * 1.8 + 32
    Td = Td * 1.8 + 32

    wspd = data['wspd']
    wdir = data['wdir']
    mask = (wspd < -900)|(wdir < -900)
    
    wspd = wspd[~mask] * kts_per_ms
    u = ma.array(-wspd * np.sin(wdir * np.pi / 180.), mask=mask)
    v = ma.array(-wspd * np.cos(wdir * np.pi / 180.), mask=mask)

    # stereogrpaphic projection.
    m = Basemap(lon_0=-99, lat_0=35., lat_ts=35, resolution='i',
        projection='stere', urcrnrlat=37, urcrnrlon=-94.25, llcrnrlat=33.7,
        llcrnrlon=-103)
    x,y = m(lon,lat)
    # transform from spherical to map projection coordinates (rotation
    # and interpolation).
    #nxv = 25; nyv = 25
    #udat, vdat, xv, yv = m.transform_vector(u,v,lons1,lats1,nxv,nyv,returnxy=True)
    # create a figure, add an axes.
    fig=plt.figure(figsize=(20,12))
    ax = fig.add_subplot(1,1,1)
#    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # plot barbs.
    ax.barbs(x, y, u, v)
    grid_off = 10
    tran_wspd = ax.transData + transforms.Affine2D().translate(grid_off, 0)
    tran_temp = ax.transData + transforms.Affine2D().translate(-grid_off, grid_off)
    tran_dew = ax.transData + transforms.Affine2D().translate(-grid_off, -grid_off)

    for ind in np.ndindex(lon.shape):
        if not mask[ind]:
            ax.text(x[ind], y[ind], '%.0f' % wspd[ind], transform=tran_wspd,
                color='b', ha='center', va='center')
        if temp[ind] > -900:
            ax.text(x[ind], y[ind], '%.0f' % temp[ind],
                transform=tran_temp, color='r', ha='center', va='center')
        if temp[ind] > -900 and rh[ind] > -900:
            ax.text(x[ind], y[ind], '%.0f' % Td[ind],
                transform=tran_dew, color='g', ha='center', va='center')
    m.bluemarble()
    m.drawstates()
#    m.readshapefile('/home/rmay/mapdata/c_28de04', 'counties')
    # draw parallels
#    m.drawparallels(np.arange(30,40,2),labels=[1,1,0,0])
    # draw meridians
#    m.drawmeridians(np.arange(-110,90,2),labels=[0,0,0,1])
    plt.title('Surface Station Plot')
    plt.savefig('stations.png', dpi=100)
    plt.show()
