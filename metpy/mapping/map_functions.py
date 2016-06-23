import numpy as np

from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import cdist

from metpy.mapping import interpolation
from metpy.mapping import points

try:
    natgrid_available = True
    from matplotlib.mlab import griddata as mpl_gridding
except ImportError:
    natgrid_available = False

def calc_kappa(spacing, kappa_star=5.052):

    return kappa_star * (2.0 * spacing / np.pi)**2

def threshold_value(x, y, z, val=0):

    x_ = x[z >= val]
    y_ = y[z >= val]
    z_ = z[z >= val]

    return x_, y_, z_

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


def interpolate(x, y, z, interp_type='linear', hres=50000, buffer=1000, minimum_neighbors=3,
                gamma=0.25, kappa_star=5.052, search_radius=None, rbf_func='linear', rbf_smooth=0):
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
        What type of interpolation to use. Available options include:
        1) "linear", "nearest", "cubic", or "rbf" from Scipy.interpolate.
        2) "natural_neighbor", "barnes", or "cressman" from Metpy.mapping .
        3) "nngrid", if installed, from Matplotlib.natgrid.
        Default "linear".
    hres: float
        The horizontal resolution of the generated grid in meters. Default 50000.
    buffer: float
        How many meters to add to the bounds of the grid. Default 1000.
    minimum_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation for a point. Default is 3.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 0.25.
    kappa_star: float
        Response parameter for barnes interpolation, specified nondimensionally
        in terms of the Nyquist. Default 5.052
    search_radius: float
        A search radius to use for the barnes and cressman interpolation schemes.
        If search_radius is not specified, it will default to the average spacing of observations.
    rbf_func: str
        Specifies which function to use for Rbf interpolation.
        Options include: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
        'quintic', and 'thin_plate'. Defualt 'linear'. See scipy.interpolate.Rbf for more information.
    rbf_smooth: float
        Smoothing value applied to rbf interpolation.  Higher values result in more smoothing.

    Returns
    -------
    grid_x: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the x dimension
    grid_y: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the y dimension ndarray
    img: (M, M) ndarray
        2-dimensional array representing the interpolated values for each grid.
    '''

    ave_spacing = np.mean((cdist(list(zip(x, y)), list(zip(x, y)))))

    if search_radius == None:
        search_radius = ave_spacing

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
            raise ValueError("Natgrid not installed.  Please use another interpolation choice.")

    elif interp_type == "cressman":

        img = interpolation.inverse_distance(x, y, z, grid_x, grid_y, search_radius,
                                             min_neighbors=minimum_neighbors, kind=interp_type)
        img = img.reshape(grid_x.shape)

    elif interp_type == "barnes":

        kappa = calc_kappa(ave_spacing, kappa_star)
        img = interpolation.inverse_distance(x, y, z, grid_x, grid_y, search_radius, gamma, kappa,
                                             min_neighbors=minimum_neighbors, kind=interp_type)
        img = img.reshape(grid_x.shape)

    elif interp_type == "rbf":

        #3-dimensional support not yet included. Assign a zero to each z dimension for observations.
        h = np.zeros((len(x)))

        rbfi = Rbf(x, y, h, z, function=rbf_func, smooth=rbf_smooth)

        #3-dimensional support not yet included. Assign a zero to each z dimension grid cell position.
        hi = np.zeros((grid_x.shape))
        img = rbfi(grid_x, grid_y, hi)

    else:
        raise ValueError("Interpolation option not available\n" +
                         "Try: linear, nearest, cubic, natural_neighbor, nngrid, barnes, cressman, rbf")

    img = np.ma.masked_where(np.isnan(img), img)

    return grid_x, grid_y, img
