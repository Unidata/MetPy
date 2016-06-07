import numpy as np

import cartopy.crs as ccrs
from metpy.cbook import get_test_data
from metpy.io import *

from netCDF4 import Dataset

from metpy.plots.ctables import registry

import matplotlib.pyplot as plt

class MPMap(object):

    def __init__(self, settings):

        filename = settings['filename']
        var = settings['variable']
        self.to_proj = settings['to_proj']
        self.from_proj = settings['from_proj']

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

            self.lons = data.variables['lon'][:]
            self.lats = data.variables['lat'][:]

            self.z = data.variables[var][:]

            self.proj_points = self.to_proj.transform_points(self.from_proj, self.lons, self.lats)

            self.x_p, self.y_p = self.proj_points[:,:,0], self.proj_points[:,:,1]

        except Exception as e:
            print(e)

    def show(self):

        #wv_norm, wv_cmap = registry.get_with_steps('WVCIMSS', 0, 1)
        view = plt.axes([0, 0, 1, 1], projection=self.to_proj)
        view.set_extent([-120, -60, 20, 50])
        view.pcolormesh(self.x_p, self.y_p, np.flipud(self.z), cmap="Greys_r") #, norm=wv_norm)

        return view


