from datetime import datetime
from pytz import UTC
import numpy as np
from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt
from metpy.cbook import ndfromtxt
from metpy.tools.oban import grid_data, cressman_weights
from metpy.vis import station_plot

def convert_date(s):
    return datetime.strptime(s,'%Y-%m-%d %H:%M:%SZ').replace(tzinfo=UTC)
converter = dict(date=convert_date, longitude=lambda s:-float(s))

# Extracts data from IDV output and masks out stations outside of North America
data = ndfromtxt('12Zrun_0203.csv', delimiter=',', skiprows=2,
    names='date, stid, latitude, longitude, height, pressure, temperature',
    dtype=None, converters=converter)

mask = (data['stid'] >= 72000) & (data['stid'] < 75000)
data = data[mask]

lat = data['latitude']
lon = data['longitude']
height = data['height']

# Generate a map plotting
bm = Basemap(projection='tmerc', lat_0=90.0, lon_0=-100.0, lat_ts=40.0,
    llcrnrlon=-121, llcrnrlat=24, urcrnrlon=-64, urcrnrlat=46,
    resolution='l')

# Transform ob locations to locations on map
obx, oby = bm(lon, lat)

# Generate grid of x,y positions
lon_grid, lat_grid, x_grid, y_grid = bm.makegrid(130, 60, returnxy=True)

# Perform analysis of height obs using Cressman weights
heights_cress = grid_data(height, x_grid, y_grid, obx, oby, cressman_weights,
    600000.)

# Mask out values over the ocean so that we don't draw contours there
heights_cress = maskoceans(lon_grid, lat_grid, heights_cress)

# Map plotting
contours = np.arange(4800., 5900., 60.0)
bm.drawstates()
bm.drawcountries()
bm.drawcoastlines()

station_plot(data, proj=bm, layout=dict(C='height', NW=None, SW=None), zorder=10)
cp = bm.contour(x_grid, y_grid, heights_cress, contours)

plt.clabel(cp, fmt='%.0f', inline_spacing=0)
plt.title('500mb Height map')
plt.show()
