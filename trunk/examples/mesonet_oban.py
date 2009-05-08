import scipy.constants as sconsts
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.basemap import Basemap
from metpy import read_mesonet_data, dewpoint, get_wind_components
from metpy.constants import C2F
from metpy.cbook import append_fields
from metpy.vis import station_plot
from metpy.tools.oban import gaussian_filter

# stereogrpaphic projection
data = read_mesonet_data('data/200811210030.mdf',
    fields=('STID', 'TIME', 'TAIR', 'RELH', 'WSPD', 'WDIR'))

#Calculate dewpoint in F from relative humidity and temperature
dewpt = C2F(dewpoint(data['TAIR'], data['RELH']/100.))
data = append_fields(data, ('dewpoint',), (dewpt,))

#Convert temperature and dewpoint to Farenheit
data['TAIR'] = C2F(data['TAIR'])

#Convert wind speeds to MPH
data['WSPD'] *= sconsts.hour / sconsts.mile
u,v = get_wind_components(data['WSPD'], data['WDIR'])
data = append_fields(data, ('u', 'v'), (u, v))

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(1,1,1)
m = Basemap(lon_0=-99, lat_0=35, lat_ts=35, resolution='i',
    projection='stere', urcrnrlat=37., urcrnrlon=-94.25, llcrnrlat=33.7,
    llcrnrlon=-103., ax=ax)
m.bluemarble()

#Objectively analyze dewpoint
lon_grid, lat_grid, x_grid, y_grid = m.makegrid(125, 50, returnxy=True)
x,y = m(data['longitude'], data['latitude'])
dew_grid = griddata(x, y, data['dewpoint'], x_grid, y_grid)
dew_grid = gaussian_filter(x_grid.T, y_grid.T, dew_grid.T, 10000, 10000)
plt.pcolor(x_grid.T, y_grid.T, dew_grid, zorder=0, cmap=plt.get_cmap('Greens'),
    antialiased=False)

station_plot(data, ax=ax, proj=m,
    styles=dict(dewpoint=dict(color='lightgreen')), zorder=10)
m.drawstates(ax=ax, zorder=1)

plt.title(data['datetime'][0].strftime('%H%MZ %d %b %Y'))
plt.show()
