import scipy.constants as sconsts
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from metpy import read_mesonet_data, dewpoint, get_wind_components
from metpy.readers.mesonet import mesonet_var_map
from metpy.constants import C2F
from metpy.cbook import rec_append_fields
from metpy.vis import station_plot

# stereogrpaphic projection
data = read_mesonet_data('data/200811210030.mdf',
    fields=('STID', 'TIME', 'TAIR', 'RELH', 'WSPD', 'WDIR'))

#Calculate dewpoint in F from relative humidity and temperature
dewpt = C2F(dewpoint(data['TAIR'], data['RELH']/100.))
data = rec_append_fields(data, ('dewpoint',), (dewpt,))

#Convert temperature and dewpoint to Farenheit
data['TAIR'] = C2F(data['TAIR'])

#Convert wind speeds to MPH
data['WSPD'] *= sconsts.hour / sconsts.mile
u,v = get_wind_components(data['WSPD'], data['WDIR'])
data = rec_append_fields(data, ('u', 'v'), (u, v))

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(1,1,1)
m = Basemap(lon_0=-99, lat_0=35, lat_ts=35, resolution='i',
    projection='stere', urcrnrlat=37., urcrnrlon=-94.25, llcrnrlat=33.7,
    llcrnrlon=-103., ax=ax)
m.bluemarble()
station_plot(data, ax=ax, proj=m, field_info=mesonet_var_map,
    styles=dict(dewpoint=dict(color='lightgreen')))
m.drawstates(ax=ax, zorder=0)
m.readshapefile('/home/rmay/mapdata/c_28de04', 'counties', zorder=0)
plt.title(data['datetime'][0].strftime('%H%MZ %d %b %Y'))
plt.show()
