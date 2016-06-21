import numpy as np

from scipy.interpolate import griddata
try:
    from metpy.mapping import c_points, c_triangles, c_interpolation as points, triangles, interpolation
except ImportError:
    from metpy.mapping import points, triangles, interpolation

try:
    natgrid_available = True
    from matplotlib.mlab import griddata as mpl_gridding
except ImportError:
    natgrid_available = False


def remove_nan_observations(x, y, z):
    '''Given (x,y) coordinates and an associated observation (z),
    remove all x, y, and z where z is nan. Will not destroy
    original values.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
        observation value

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        nan valued observations.
    '''

    x_ = x[~np.isnan(z)]
    y_ = y[~np.isnan(z)]
    z_ = z[~np.isnan(z)]

    return x_, y_, z_


def remove_repeat_coordinates(x, y, z):
    '''Given x,y coordinates and an associated observation (z),
    remove all x, y, and z where (x,y) is repeated. Will not
    destroy original values.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
        observation value

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        repeated coordinates.
    '''

    coords = []
    variable = []

    for (x_, y_, t_) in zip(x, y, z):
        if (x_, y_) not in coords:
            coords.append((x_, y_))
            variable.append(t_)

    x_ = np.array(list(coords))[:, 0]
    y_ = np.array(list(coords))[:, 1]

    z_ = np.array(variable)

    return x_, y_, z_


def remove_nans_and_repeats(x, y, z):
    '''Given x,y coordinates and an associated observation (z),
    remove all x, y, and z where (x,y) is repeated and where z
    is nan. Will not destroy original values.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
        observation value

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        repeated coordinates and nan valued observations.
    '''

    x_, y_, z_ = remove_repeat_coordinates(x, y, z)
    x_, y_, z_ = remove_nan_observations(x_, y_, z_)

    return x_, y_, z_


def interpolate(x, y, z, interp_type="linear", hres=50000, buffer=1000, search_radius=None):
    '''Interpolate given (x,y), observation (z) pairs to a grid based on given parameters.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
        observation value
    interp_type: str
        What type of interpolation to use. Available options include "linear", "nearest", "cubic",
        "natural_neighbor", "nngrid", "barnes", "cressman", "rbf".  Default "linear".
    hres: float
        The horizontal resolution of the generated grid. Default 50000.
    buffer: float
        How many meters to add to the bounds of the grid. Default 1000.
    search_radius: float
        A search radius to use for certain interpolation schemes. Default None.

    Returns
    -------
    grid_x: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the x dimension
    grid_y: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the y dimension ndarray
    img: (M, M) ndarray
        List of coordinate observation pairs without
        repeated coordinates and nan valued observations.
    '''

    grid_x, grid_y = points.generate_grid(hres, points.get_boundary_coords(x, y), buffer)

    if interp_type in ["linear", "nearest", "cubic"]:
        points_zip = np.array(list(zip(x, y)))
        img = griddata(points_zip, z, (grid_x, grid_y), method=interp_type)

    elif interp_type == "natural_neighbor":
        img = interpolation.natural_neighbor(x, y, z, grid_x, grid_y)
        img = img.reshape(grid_x.shape)

    elif interp_type == "nngrid":
        if natgrid_available:
            img = mpl_gridding(x, y, z, grid_x, grid_y, interp='nn')
        else:
            raise ValueError("Natgrid not installed.  Please use another interpolation choice: \n"
                             "linear, nearest, cubic, natural_neighbor")

    elif interp_type == "barnes":
        img = interpolation.barnes()

    elif interp_type == "cressman":
        if search_radius == None:
            raise ValueError("You must provide a search radius for cressman interpolation.")
        else:
            img = interpolation.cressman(x, y, z, grid_x, grid_y, search_radius)
            img = img.reshape(grid_x.shape)

    elif interp_type == "rbf":
        img = interpolation.radial_basis_functions()

    else:
        raise ValueError("Interpolation option not available\n" +
                         "Try: linear, nearest, cubic, natural_neighbor, nngrid, barnes, cressman, rbf")

    img = np.ma.masked_where(np.isnan(img), img)

    return grid_x, grid_y, img
