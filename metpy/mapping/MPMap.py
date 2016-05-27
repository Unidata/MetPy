import numpy as np

from mpl_toolkits import basemap
from metpy.cbook import get_test_data
from metpy.io import *

from netCDF4 import Dataset

from metpy.plots.ctables import registry

import matplotlib.pyplot as plt

class MPMap(object):

    def __init__(self, settings):

        filename = settings['filename']
        var = settings['variable']


        print(filename)
        type = filename.split(".")[-1]
        print(type)
        data = None

        if type.upper()=='GINI':

            data = GiniFile(get_test_data(filename)).to_dataset()

        elif type.upper() == 'NC':

            data = Dataset(get_test_data(filename))

        else:

            print("Data type not supported")

        try:

            lons = data.variables['lon'][:]
            lats = data.variables['lat'][:]

            self.z = data.variables[var][:]

            self.view = basemap.Basemap(projection='aea', resolution='l', lat_1=28.5, lat_2=44.5, lat_0=38.5,
                            lon_0=-97., area_thresh=5000,  llcrnrlon=np.min(lons[0,:]), llcrnrlat=np.min(lats),
                            urcrnrlon=np.max(lons[0,:]), urcrnrlat=np.max(lats))

            #basemap.Basemap(
                        #width=4800000, height=3100000, projection='aea', resolution='l',
                        #lat_1=28.5, lat_2=44.5, lat_0=38.5, lon_0=-97.,area_thresh=10000)

            self.x_p, self.y_p = self.view(lons, lats)# basemap.pyproj.transform(aea, lcc, lons, lats)



        except Exception as e:
            print(e)

    def show(self):

        #wv_norm, wv_cmap = registry.get_with_steps('WVCIMSS', 0, 1)

        self.view.pcolormesh(self.x_p, self.y_p, np.flipud(self.z), cmap="Greys_r") #, norm=wv_norm)
        self.view.drawcoastlines()


