import numpy as np

from scipy.interpolate import griddata
from metpy.mapping import _points, _triangles, interpolation

try:
    natgrid_available = True
    from matplotlib.mlab import griddata as mpl_gridding
except ImportError:
    natgrid_available = False

def interp_points(x, y, z, interp_type="linear", xres=50000, yres=50000, buffer=1000):

    x = x + xres
    y = y - yres

    grid_x, grid_y = _points.generate_grid(xres, yres, _points.get_boundary_coords(x, y), buffer)

    if interp_type in ["linear", "nearest", "cubic"]:

        points_zip = np.array(list(zip(x, y)))
        img = griddata(points_zip, z, (grid_x, grid_y), method=interp_type)

    elif interp_type == "natural_neighbor":

        grids = _points.generate_grid_coords(grid_x, grid_y)
        img = interpolation.natural_neighbor(x, y, z, grids)

    elif interp_type == "nngrid":

        if natgrid_available:
            grids = _points.generate_grid_coords(grid_x, grid_y)
            img = mpl_gridding(x, y, z, grid_x, grid_y, interp='nn')
        else:
            raise ValueError("Natgrid not installed.  Please use another interpolation choice: \n"
                             "linear, nearest, cubic, natural_neighbor")

    elif interp_type == "barnes":

        img = _interpolation.barnes()

    elif interp_type == "cressman":

        img = _interpolation.cressman()

    elif interp_type == "rbf":

        img = _interpolation.radial_basis_functions()

    else:

        print("Interpolation option not available\n" +
              "Try: linear, nearest, cubic, natural_neighbor, nngrid, barnes, cressman, rbf")

    img = np.ma.masked_where(np.isnan(img), img)

    return grid_x, grid_y, img

# class MPMap(object):
#
#     def __init__(self, settings):
#
#         filename = settings['filename']
#         var = settings['variable']
#         self.to_proj = settings['to_proj']
#         self.from_proj = settings['from_proj']
#
#         print(filename)
#         type = filename.split(".")[-1]
#         print(type)
#         data = None
#
#         if type.upper()=='GINI':
#
#             data = GiniFile(get_test_data(filename)).to_dataset()
#
#         elif type.upper() == 'NC':
#
#             data = Dataset(get_test_data(filename))
#
#         else:
#
#             print("Data type not supported")
#
#         try:
#
#             self.lons = data.variables['lon'][:]
#             self.lats = data.variables['lat'][:]
#
#             self.z = data.variables[var][:]
#
#             self.proj_points = self.to_proj.transform_points(self.from_proj, self.lons, self.lats)
#
#             self.x_p, self.y_p = self.proj_points[:,:,0], self.proj_points[:,:,1]
#
#         except Exception as e:
#             print(e)
#
#     def show(self):
#
#         #wv_norm, wv_cmap = registry.get_with_steps('WVCIMSS', 0, 1)
#         view = plt.axes([0, 0, 1, 1], projection=self.to_proj)
#         view.set_extent([-120, -60, 20, 50])
#         view.pcolormesh(self.x_p, self.y_p, np.flipud(self.z), cmap="Greys_r") #, norm=wv_norm)
#
#         return view


