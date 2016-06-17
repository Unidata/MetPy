import numpy as np

from scipy.interpolate import griddata
from metpy.mapping import c_points, c_triangles, c_interpolation

try:
    natgrid_available = True
    from matplotlib.mlab import griddata as mpl_gridding
except ImportError:
    natgrid_available = False

def interpolate(x, y, z, interp_type="linear", xres=50000, yres=50000, buffer=1000):

    x = x + xres
    y = y - yres

    grid_x, grid_y = c_points.generate_grid(xres, yres, c_points.get_boundary_coords(x, y), buffer)

    if interp_type in ["linear", "nearest", "cubic"]:

        points_zip = np.array(list(zip(x, y)))
        img = griddata(points_zip, z, (grid_x, grid_y), method=interp_type)

    elif interp_type == "natural_neighbor":

        grids = c_points.generate_grid_coords(grid_x, grid_y)
        img = c_interpolation.natural_neighbor(x, y, z, grids)

    elif interp_type == "nngrid":

        if natgrid_available:
            grids = c_points.generate_grid_coords(grid_x, grid_y)
            img = mpl_gridding(x, y, z, grid_x, grid_y, interp='nn')
        else:
            raise ValueError("Natgrid not installed.  Please use another interpolation choice: \n"
                             "linear, nearest, cubic, natural_neighbor")

    elif interp_type == "barnes":

        img = c_interpolation.barnes()

    elif interp_type == "cressman":

        img = c_interpolation.cressman()

    elif interp_type == "rbf":

        img = c_interpolation.radial_basis_functions()

    else:

        print("Interpolation option not available\n" +
              "Try: linear, nearest, cubic, natural_neighbor, nngrid, barnes, cressman, rbf")

    img = np.ma.masked_where(np.isnan(img), img)

    return grid_x, grid_y, img

