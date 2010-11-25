import os
import numpy as np
import scipy.constants as sconsts
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.basemap import Basemap
from metpy import (read_mesonet_data, dewpoint, get_wind_components,
    h_convergence)
from metpy import remote_mesonet_data
from metpy.constants import C2F
from metpy.cbook import append_fields
from metpy.vis import station_plot
from metpy.tools.oban import gaussian_filter

# TODO: Find a way to fix the bad convergence values at the edge of the
# masked data
data = read_mesonet_data('data/200905082110.mdf',
    fields=('STID', 'TIME', 'TAIR', 'RELH', 'WSPD', 'WDIR'))

#Calculate dewpoint in F from relative humidity and temperature
dewpt = C2F(dewpoint(data['TAIR'], data['RELH']/100.))
data = append_fields(data, ('dewpoint',), (dewpt,))

#Convert temperature and dewpoint to Farenheit
data['TAIR'] = C2F(data['TAIR'])

u,v = get_wind_components(data['WSPD'], data['WDIR'])
data = append_fields(data, ('u', 'v'), (u, v))

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(1,1,1)
m = Basemap(lon_0=-99, lat_0=35, lat_ts=35, resolution='i',
    projection='stere', urcrnrlat=37., urcrnrlon=-94.25, llcrnrlat=33.7,
    llcrnrlon=-103., ax=ax)
m.bluemarble()

#Objectively analyze wind components
sig = 15000
lon_grid, lat_grid, x_grid, y_grid = m.makegrid(125, 50, returnxy=True)
x,y = m(data['longitude'], data['latitude'])
u_grid = griddata(x, y, data['u'], x_grid, y_grid)
u_grid = gaussian_filter(x_grid.T, y_grid.T, u_grid.T, sig, sig)
v_grid = griddata(x, y, data['v'], x_grid, y_grid)
v_grid = gaussian_filter(x_grid.T, y_grid.T, v_grid.T, sig, sig)
conv = h_convergence(u_grid.filled(0), v_grid.filled(0), x_grid[0, 1], y_grid[1, 0])
conv = np.ma.array(conv, mask=u_grid.mask)

plt.pcolor(x_grid.T, y_grid.T, conv, zorder=0, cmap=plt.get_cmap('RdBu'),
    antialiased=False, norm=plt.Normalize(-2e-4, 2e-4))

#Convert wind speeds to MPH
data['u'] *= sconsts.hour / sconsts.mile
data['v'] *= sconsts.hour / sconsts.mile
station_plot(data, ax=ax, proj=m,
    styles=dict(dewpoint=dict(color='lightgreen')), zorder=10)
m.drawstates(ax=ax, zorder=1)
mapfile = os.path.join(os.environ['HOME'], 'mapdata', 'c_03oc08.shp')
if os.path.exists(mapfile):
    m.readshapefile(os.path.splitext(mapfile)[0], 'counties', zorder=0)

plt.title(data['datetime'][0].strftime('%H%MZ %d %b %Y'))
plt.show()
