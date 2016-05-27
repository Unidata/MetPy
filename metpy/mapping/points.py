import numpy as np

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, Delaunay, cKDTree
from scipy.spatial.distance import euclidean

from collections import deque


def interp_points(x, y, z, interp_type="linear", xres=1000, yres=1000, buffer=1000):

    #xmin = np.min(x) - buffer*1000
    #xmax = np.max(x) + buffer*1000
    #ymin = np.min(y) - buffer*1000
    #ymax = np.max(y) + buffer*1000

    grid_x, grid_y = generate_grid(xres, yres, get_boundary_coords(x, y), buffer)

    # np.mgrid[xmin:xmax:xres*1j, ymin:ymax:yres*1j]

    if interp_type in ["linear", "nearest", "cubic"]:

        #griddata requires (x, y) array
        points_zip = np.array(list(zip(x, y)))

        img = griddata(points_zip, z, (grid_x, grid_y), method=interp_type)

    elif interp_type == "natural_neighbor":

        img = natural_neighbor(x, y, z, grid_x, grid_y)

    elif interp_type == "barnes":

        img = barnes()

    elif interp_type == "cressman":

        img = cressman()

    elif interp_type == "rbf":

        img = radial_basis_functions()

    else:

        print("Interpolation option not available\n" +
              "Try: linear, nearest, cubic, natural_neighbor, barnes, cressman, rbf")


    return grid_x, grid_y, img

def natural_neighbor(x, y, z, gx, gy):

    points = list(zip(x, y))

    grid_points = list(zip(generate_grid_coords(gx, gy)))

    triangles = Delaunay(points)

    tri_match = triangles.find_simplex(grid_points)





    return 0

def barnes():

    return 0

def cressman():

    return 0

def radial_basis_functions():

    return 0

def get_points_within_r(center_point, target_points, r, return_idx=False):
    '''Get all target_points within a specified radius
    of a center point.  All data must be in same coord-
    inate system, or you will get unpredictable results.

    Parameters
    ----------
    center_points: (X, Y) ndarray
        location from which to grab surrounding points within r
    target_points: (X, Y) ndarray
        points from which to return if they are within r of center_points
    r: integer
        search radius around center_points to grab target_points
    return_idx: bool
        If true, function will return indices of winning points
        If false (default), function will return list of winning points

    Returns
    -------
    (X, Y) ndarray
        A list of points within r distance of, and in the same
        order as, center_points
    '''

    tree = cKDTree(target_points)
    indices = tree.query_ball_point(center_point, r)
    return tree.data[indices].T


def get_point_count_within_r(center_points, target_points, r):
    '''Get count of target points within a specified radius
    from center points.  All data must be in same coord-
    inate system, or you will get unpredictable results.

    Parameters
    ----------
    center_points: (X, Y) ndarray
        locations from which to grab surrounding points within r
    target_points: (X, Y) ndarray
        points from which to return if they are within r of center_points
    r: integer
        search radius around center_points to grab target_points

    Returns
    -------
        A list of point counts within r distance of, and in the same
        order as, center_points
    '''

    tree = cKDTree(target_points)
    indices = tree.query_ball_point(center_points, r)
    return np.array([len(x) for x in indices])


def smoothed_freq_map(x_points, y_points, bbox, x_steps, y_steps, gaussian):
    '''Create smoothed spatial frequency map of points per user
    defined grid cell within a specified extent.  All values are
    assumed to be in the same coordinate system.

    Parameters
    ----------
    x_points: array-like
        x_coordinates used to calculate counts per grid cell
    y_points: array-like
        y_coordinates used to calculate counts per grid cell
    bbox: dictionary of boundary coordinates
        spatial bounding box of histogram
    steps: (X_size, Y_size) ndarray
        size of the grid cells
    gaussian: floating point
        size of smoothing window

    Returns
    -------
        A smoothed frequency grid
    '''

    #    west = bbox['southwest'][0]
    #    north = bbox['northeast'][1]
    #    east = bbox['northeast'][0]
    #    south = bbox['southwest'][1]

    grid, _, _ = np.histogram2d(y_points, x_points, bins=(y_steps, x_steps))
    grid = np.flipud(grid)
    return gaussian_filter(grid, sigma=gaussian)


def generate_grid(x_dim, y_dim, bbox, ignore_warnings=False):
    '''Generate a meshgrid based on bounding box and x & y resolution

    Parameters
    ----------
    x_dim: integer
        x resolution in meters
    y_dim: integer
        y resolution in meters
    bbox: dictionary
        dictionary containing coordinates for corners of study area

    Returns
    -------
    (X, Y) ndarray
        meshgrid defined by given bounding box
    '''
    if not ignore_warnings and (x_dim < 10000 or y_dim < 10000):
        print("Grids less than 10km may be slow to load at synoptic scale.")
        print("Set ignore_warnings to True to run anyway. Defaulting to 10km")
        x_dim = y_dim = 10000

    x_steps, y_steps = get_xy_steps(bbox, x_dim, y_dim)

    grid_x = np.linspace(bbox['southwest'][0], bbox['northeast'][0], x_steps)
    grid_y = np.linspace(bbox['southwest'][1], bbox['northeast'][1], y_steps)

    gx, gy = np.meshgrid(grid_x, grid_y)

    return gx, gy


def generate_grid_coords(gx, gy):
    '''Calculate x,y coordinates of each grid cell

    Parameters
    ----------
    gx: numeric
        x coordinates in meshgrid
    gy: numeric
        y coordinates in meshgrid

    Returns
    -------
    (X, Y) ndarray
        List of coordinates in meshgrid
    '''

    return np.vstack([gx.ravel(), gy.ravel()]).T


def get_xy_range(bbox):
    '''Returns x and y ranges in meters based on bounding box

    bbox: dictionary
        dictionary containing coordinates for corners of study area

    Returns
    -------
    X, Y: numeric
        X and Y ranges in meters
    '''

    x_range = bbox['northeast'][0] - bbox['southwest'][0]
    y_range = bbox['northeast'][1] - bbox['southwest'][1]

    return x_range, y_range


def get_xy_steps(bbox, x_dim, y_dim):
    '''Return meshgrid spacing based on bounding box

    bbox: dictionary
        dictionary containing coordinates for corners of study area
    x_dim: integer
        x resolution in meters
    y_dim: integer
        y resolution in meters

    Returns
    -------
    (X,Y): ndarray
        List of all X and Y centers used to create a meshgrid
    '''

    x_range, y_range = get_xy_range(bbox)

    x_steps = np.ceil(x_range / x_dim)
    y_steps = np.ceil(y_range / y_dim)
    
    return int(x_steps), int(y_steps)


def get_boundary_coords(x, y, spatial_pad = 0):
    '''Return bounding box based on given x and y coordinates
       assuming northern hemisphere.

    x: numeric
        x coordinates
    y: numeric
        y coordinates
    spatial_pad: numeric
        Number of meters to add to the x and y dimensions to reduce
        edge effects.

    Returns
    -------
    bbox: dictionary
        dictionary containing coordinates for corners of study area
    '''


    west = np.min(x) - spatial_pad
    east = np.max(x) + spatial_pad
    north = np.max(y) + spatial_pad
    south = np.min(y) - spatial_pad

    return {'southwest': (west, south), 'northeast': (east, north)}