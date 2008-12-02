__all__ = ['meteogram']

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def meteogram(site, dt=None, **fields):
    if dt is None:
        import datetime
        dt = datetime.date.today()

    time = fields['time']/60.
    temp = fields['temp']
    relh = fields['relh']
    press = fields['press']
    wspd = fields['wspd']
    
    ax1 = plt.subplot(3,1,1)
    plt.plot(time, relh, 'g')
    plt.ylabel('Rel. Humidity (%)')
    plt.xlim(0,24)
    plt.title('Meteogram for %s on %s' % (site.upper(), dt))
    ax1.xaxis.set_major_formatter(NullFormatter())
    
    ax2 = plt.subplot(3,1,2)
    plt.plot(time, temp, 'r')
    plt.ylabel('Temperature (C)')
    plt.xlim(0,24)
    ax2.xaxis.set_major_formatter(NullFormatter())
    
    ax3 = plt.subplot(3,1,3)
    plt.plot(time, press, 'k')
    plt.xlabel('Time (hour)')
    plt.ylabel('Pressure (mb)')
    ax32 = plt.twinx(ax3)
    ax32.plot(time, wspd, 'b')
    ax32.set_ylabel('Wind Speed (m/s)')
    plt.xlim(0,24)
    plt.xticks(range(0,25,3))
    
    plt.subplots_adjust(hspace=0.1)
    
    return [ax1, ax2, (ax3, ax32)]

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
