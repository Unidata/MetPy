# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Interpolate data valid at one set of points to another in multiple dimensions."""

from __future__ import division

import logging

import numpy as np
from scipy.interpolate import griddata, Rbf
from scipy.spatial import cKDTree, ConvexHull, Delaunay, qhull
from scipy.spatial.distance import cdist

from . import geometry, tools
from ..package_tools import Exporter

exporter = Exporter(globals())

log = logging.getLogger(__name__)


def cressman_point(sq_dist, values, radius):
    r"""Generate a Cressman interpolation value for a point.

    The calculated value is based on the given distances and search radius.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distance between observations and grid point
    values: (N, ) ndarray
        Observation values in same order as sq_dist
    radius: float
        Maximum distance to search for observations to use for
        interpolation.

    Returns
    -------
    value: float
        Interpolation value for grid point.

    """
    weights = tools.cressman_weights(sq_dist, radius)
    total_weights = np.sum(weights)

    return sum(v * (w / total_weights) for (w, v) in zip(weights, values))


def barnes_point(sq_dist, values, kappa, gamma=None):
    r"""Generate a single pass barnes interpolation value for a point.

    The calculated value is based on the given distances, kappa and gamma values.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distance between observations and grid point
    values: (N, ) ndarray
        Observation values in same order as sq_dist
    kappa: float
        Response parameter for barnes interpolation.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 1.

    Returns
    -------
    value: float
        Interpolation value for grid point.

    """
    if gamma is None:
        gamma = 1
    weights = tools.barnes_weights(sq_dist, kappa, gamma)
    total_weights = np.sum(weights)

    return sum(v * (w / total_weights) for (w, v) in zip(weights, values))


def natural_neighbor_point(xp, yp, variable, grid_loc, tri, neighbors, triangle_info):
    r"""Generate a natural neighbor interpolation of the observations to the given point.

    This uses the Liang and Hale approach [Liang2010]_. The interpolation will fail if
    the grid point has no natural neighbors.

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations
    yp: (N, ) ndarray
        y-coordinates of observations
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i])
    grid_loc: (float, float)
        Coordinates of the grid point at which to calculate the
        interpolation.
    tri: object
        Delaunay triangulation of the observations.
    neighbors: (N, ) ndarray
        Simplex codes of the grid point's natural neighbors. The codes
        will correspond to codes in the triangulation.
    triangle_info: dictionary
        Pre-calculated triangle attributes for quick look ups. Requires
        items 'cc' (circumcenters) and 'r' (radii) to be associated with
        each simplex code key from the delaunay triangulation.

    Returns
    -------
    value: float
       Interpolated value for the grid location

    """
    edges = geometry.find_local_boundary(tri, neighbors)
    edge_vertices = [segment[0] for segment in geometry.order_edges(edges)]
    num_vertices = len(edge_vertices)

    p1 = edge_vertices[0]
    p2 = edge_vertices[1]

    c1 = geometry.circumcenter(grid_loc, tri.points[p1], tri.points[p2])
    polygon = [c1]

    area_list = []
    total_area = 0.0

    for i in range(num_vertices):

        p3 = edge_vertices[(i + 2) % num_vertices]

        try:

            c2 = geometry.circumcenter(grid_loc, tri.points[p3], tri.points[p2])
            polygon.append(c2)

            for check_tri in neighbors:
                if p2 in tri.simplices[check_tri]:
                    polygon.append(triangle_info[check_tri]['cc'])

            pts = [polygon[i] for i in ConvexHull(polygon).vertices]
            value = variable[(tri.points[p2][0] == xp) & (tri.points[p2][1] == yp)]

            cur_area = geometry.area(pts)

            total_area += cur_area

            area_list.append(cur_area * value[0])

        except (ZeroDivisionError, qhull.QhullError) as e:
            message = ('Error during processing of a grid. '
                       'Interpolation will continue but be mindful '
                       'of errors in output. ') + str(e)

            log.warning(message)
            return np.nan

        polygon = [c2]

        p2 = p3

    return sum(x / total_area for x in area_list)


@exporter.export
def natural_neighbor_to_points(points, values, xi):
    r"""Generate a natural neighbor interpolation to the given points.

    This assigns values to the given interpolation points using the Liang and Hale
    [Liang2010]_. approach.

    Parameters
    ----------
    points: array_like, shape (n, 2)
        Coordinates of the data points.
    values: array_like, shape (n,)
        Values of the data points.
    xi: array_like, shape (M, 2)
        Points to interpolate the data onto.

    Returns
    -------
    img: (M,) ndarray
        Array representing the interpolated values for each input point in `xi`

    See Also
    --------
    natural_neighbor_to_grid

    """
    tri = Delaunay(points)

    members, triangle_info = geometry.find_natural_neighbors(tri, xi)

    img = np.empty(shape=(xi.shape[0]), dtype=values.dtype)
    img.fill(np.nan)

    for ind, (grid, neighbors) in enumerate(members.items()):

        if len(neighbors) > 0:

            points_transposed = np.array(points).transpose()
            img[ind] = natural_neighbor_point(points_transposed[0], points_transposed[1],
                                              values, xi[grid], tri, neighbors, triangle_info)

    return img


@exporter.export
def inverse_distance_to_points(points, values, xi, r, gamma=None, kappa=None, min_neighbors=3,
                               kind='cressman'):
    r"""Generate an inverse distance weighting interpolation to the given points.

    Values are assigned to the given interpolation points based on either [Cressman1959]_ or
    [Barnes1964]_. The Barnes implementation used here based on [Koch1983]_.

    Parameters
    ----------
    points: array_like, shape (n, 2)
        Coordinates of the data points.
    values: array_like, shape (n,)
        Values of the data points.
    xi: array_like, shape (M, 2)
        Points to interpolate the data onto.
    r: float
        Radius from grid center, within which observations
        are considered and weighted.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default None.
    kappa: float
        Response parameter for barnes interpolation. Default None.
    min_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation
        for a point. Default is 3.
    kind: str
        Specify what inverse distance weighting interpolation to use.
        Options: 'cressman' or 'barnes'. Default 'cressman'

    Returns
    -------
    img: (M,) ndarray
        Array representing the interpolated values for each input point in `xi`

    See Also
    --------
    inverse_distance_to_grid

    """
    obs_tree = cKDTree(points)

    indices = obs_tree.query_ball_point(xi, r=r)

    img = np.empty(shape=(xi.shape[0]), dtype=values.dtype)
    img.fill(np.nan)

    for idx, (matches, grid) in enumerate(zip(indices, xi)):
        if len(matches) >= min_neighbors:

            x1, y1 = obs_tree.data[matches].T
            values_subset = values[matches]
            dists = geometry.dist_2(grid[0], grid[1], x1, y1)

            if kind == 'cressman':
                img[idx] = cressman_point(dists, values_subset, r)
            elif kind == 'barnes':
                img[idx] = barnes_point(dists, values_subset, kappa, gamma)

            else:
                raise ValueError(str(kind) + ' interpolation not supported.')

    return img


@exporter.export
def interpolate_to_points(points, values, xi, interp_type='linear', minimum_neighbors=3,
                          gamma=0.25, kappa_star=5.052, search_radius=None, rbf_func='linear',
                          rbf_smooth=0):
    r"""Interpolate unstructured point data to the given points.

    This function interpolates the given `values` valid at `points` to the points `xi`. This is
    modeled after `scipy.interpolate.griddata`, but acts as a generalization of it by including
    the following types of interpolation:

    - Linear
    - Nearest Neighbor
    - Cubic
    - Radial Basis Function
    - Natural Neighbor (2D Only)
    - Barnes (2D Only)
    - Cressman (2D Only)

    Parameters
    ----------
    points: array_like, shape (n, D)
        Coordinates of the data points.
    values: array_like, shape (n,)
        Values of the data points.
    xi: array_like, shape (M, D)
        Points to interpolate the data onto.
    interp_type: str
        What type of interpolation to use. Available options include:
        1) "linear", "nearest", "cubic", or "rbf" from `scipy.interpolate`.
        2) "natural_neighbor", "barnes", or "cressman" from `metpy.interpolate`.
        Default "linear".
    minimum_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation for a
        point. Default is 3.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 0.25.
    kappa_star: float
        Response parameter for barnes interpolation, specified nondimensionally
        in terms of the Nyquist. Default 5.052
    search_radius: float
        A search radius to use for the barnes and cressman interpolation schemes.
        If search_radius is not specified, it will default to the average spacing of
        observations.
    rbf_func: str
        Specifies which function to use for Rbf interpolation.
        Options include: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
        'quintic', and 'thin_plate'. Defualt 'linear'. See `scipy.interpolate.Rbf` for more
        information.
    rbf_smooth: float
        Smoothing value applied to rbf interpolation.  Higher values result in more smoothing.

    Returns
    -------
    values_interpolated: (M,) ndarray
        Array representing the interpolated values for each input point in `xi`.

    Notes
    -----
    This function primarily acts as a wrapper for the individual interpolation routines. The
    individual functions are also available for direct use.

    See Also
    --------
    interpolate_to_grid

    """
    # If this is a type that `griddata` handles, hand it along to `griddata`
    if interp_type in ['linear', 'nearest', 'cubic']:
        return griddata(points, values, xi, method=interp_type)

    # If this is natural neighbor, hand it along to `natural_neighbor`
    elif interp_type == 'natural_neighbor':
        return natural_neighbor_to_points(points, values, xi)

    # If this is Barnes/Cressman, determine search_radios and hand it along to
    # `inverse_distance`
    elif interp_type in ['cressman', 'barnes']:
        ave_spacing = cdist(points, points).mean()

        if search_radius is None:
            search_radius = ave_spacing

        if interp_type == 'cressman':
            return inverse_distance_to_points(points, values, xi, search_radius,
                                              min_neighbors=minimum_neighbors,
                                              kind=interp_type)
        else:
            kappa = tools.calc_kappa(ave_spacing, kappa_star)
            return inverse_distance_to_points(points, values, xi, search_radius, gamma, kappa,
                                              min_neighbors=minimum_neighbors,
                                              kind=interp_type)

    # If this is radial basis function, make the interpolator and apply it
    elif interp_type == 'rbf':

        points_transposed = np.array(points).transpose()
        xi_transposed = np.array(xi).transpose()
        rbfi = Rbf(points_transposed[0], points_transposed[1], values, function=rbf_func,
                   smooth=rbf_smooth)
        return rbfi(xi_transposed[0], xi_transposed[1])

    else:
        raise ValueError('Interpolation option not available. '
                         'Try: linear, nearest, cubic, natural_neighbor, '
                         'barnes, cressman, rbf')
