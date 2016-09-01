# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division

import numpy as np

import warnings

from scipy.spatial import Delaunay, ConvexHull, cKDTree, qhull

from metpy.gridding import triangles, polygons, points

from ..package_tools import Exporter

exporter = Exporter(globals())


@exporter.export
def natural_neighbor(xp, yp, variable, grid_x, grid_y):
    r"""Generate a natural neighbor interpolation of the given
    points to the given grid using the Liang and Hale (2010)
    approach.

    Liang, Luming, and Dave Hale. "A stable and fast implementation
    of natural neighbor interpolation." (2010).

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations
    yp: (N, ) ndarray
        y-coordinates of observations
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i])
    grid_x: (M, 2) ndarray
        Meshgrid associated with x dimension
    grid_y: (M, 2) ndarray
        Meshgrid associated with y dimension

    Returns
    -------
    img: (M, N) ndarray
        Interpolated values on a 2-dimensional grid
    """

    tri = Delaunay(list(zip(xp, yp)))

    grid_points = points.generate_grid_coords(grid_x, grid_y)

    members, triangle_info = triangles.find_natural_neighbors(tri, grid_points)

    img = np.empty(shape=(grid_points.shape[0]), dtype=variable.dtype)
    img.fill(np.nan)

    for ind, (grid, neighbors) in enumerate(members.items()):

        if len(neighbors) > 0:

            img[ind] = nn_point(xp, yp, variable, grid_points[grid],
                                tri, neighbors, triangle_info)

    img = img.reshape(grid_x.shape)
    return img


def nn_point(xp, yp, variable, grid_loc, tri, neighbors, triangle_info):
    r"""Generate a natural neighbor interpolation of the given
    observations to the given point using the Liang and Hale (2010)
    approach. The interpolation will fail if the grid point has no
    natural neighbors.

    Liang, Luming, and Dave Hale. A stable and fast implementation
    of natural neighbor interpolation. (2010).

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations
    yp: (N, ) ndarray
        y-coordinates of observations
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i])
    grid_loc: (N, 2) ndarray
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

    edges = triangles.find_local_boundary(tri, neighbors)
    edge_vertices = [segment[0] for segment in polygons.order_edges(edges)]
    num_vertices = len(edge_vertices)

    p1 = edge_vertices[0]
    p2 = edge_vertices[1]

    polygon = list()
    c1 = triangles.circumcenter(grid_loc, tri.points[p1], tri.points[p2])
    polygon.append(c1)

    area_list = []
    total_area = 0.0

    for i in range(num_vertices):

        p3 = edge_vertices[(i + 2) % num_vertices]

        try:

            c2 = triangles.circumcenter(grid_loc, tri.points[p3], tri.points[p2])
            polygon.append(c2)

            for check_tri in neighbors:
                if p2 in tri.simplices[check_tri]:
                    polygon.append(triangle_info[check_tri]['cc'])

            pts = [polygon[i] for i in ConvexHull(polygon).vertices]
            value = variable[(tri.points[p2][0] == xp) & (tri.points[p2][1] == yp)]

            cur_area = polygons.area(pts)

            total_area += cur_area

            area_list.append(cur_area * value[0])

        except (ZeroDivisionError, qhull.QhullError) as e:
            message = ('Error during processing of a grid. '
                       'Interpolation will continue but be mindful '
                       'of errors in output. ') + str(e)

            warnings.warn(message)
            return np.nan

        polygon = list()
        polygon.append(c2)

        p2 = p3

    return sum([x / total_area for x in area_list])


def barnes_weights(sq_dist, kappa, gamma):
    r"""Calculate the barnes weights for observation points
    based on their distance from an interpolation point.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distances from interpolation point
        associated with each observation in meters.
    kappa: float
        Response parameter for barnes interpolation. Default None.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default None.

    Returns
    -------
    weights: (N, ) ndarray
        Calculated weights for the given observations determined by their distance
        to the interpolation point.
    """

    return np.exp(-1.0 * sq_dist / (kappa * gamma))


def cressman_weights(sq_dist, r):
    r"""Calculate the cressman weights for observation points
    based on their distance from an interpolation point.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distances from interpolation point
        associated with each observation in meters.
    r: float
        Maximum distance an observation can be from an
        interpolation point to be considered in the inter-
        polation calculation.

    Returns
    -------
    weights: (N, ) ndarray
        Calculated weights for the given observations determined by their distance
        to the interpolation point.
    """

    return (r * r - sq_dist) / (r * r + sq_dist)


@exporter.export
def inverse_distance(xp, yp, variable, grid_x, grid_y, r, gamma=None, kappa=None,
                     min_neighbors=3, kind='cressman'):
    r"""Generate an inverse distance weighting interpolation of the given
    points to the given grid based on either Cressman (1959) or Barnes (1964).
    The Barnes implementation used here based on Koch et al. (1983).

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations.
    yp: (N, ) ndarray
        y-coordinates of observations.
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i]).
    grid_x: (M, 2) ndarray
        Meshgrid associated with x dimension.
    grid_y: (M, 2) ndarray
        Meshgrid associated with y dimension.
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
    img: (M, N) ndarray
        Interpolated values on a 2-dimensional grid
    """

    obs_tree = cKDTree(list(zip(xp, yp)))

    grid_points = points.generate_grid_coords(grid_x, grid_y)

    indices = obs_tree.query_ball_point(grid_points, r=r)

    img = np.empty(shape=(grid_points.shape[0]), dtype=variable.dtype)
    img.fill(np.nan)

    for idx, (matches, grid) in enumerate(zip(indices, grid_points)):
        if len(matches) >= min_neighbors:

            x1, y1 = obs_tree.data[matches].T
            values = variable[matches]
            dists = triangles.dist_2(grid[0], grid[1], x1, y1)

            if kind == 'cressman':
                img[idx] = cressman_point(dists, values, r)
            elif kind == 'barnes':
                img[idx] = barnes_point(dists, values, kappa)

            else:
                raise ValueError(str(kind) + ' interpolation not supported.')

    img = img.reshape(grid_x.shape)
    return img


def cressman_point(sq_dist, values, radius):
    r"""Generate a cressman interpolation value for a point based on
    the given distances and search radius.

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

    weights = cressman_weights(sq_dist, radius)
    total_weights = np.sum(weights)

    return sum([v * (w / total_weights) for (w, v) in zip(weights, values)])


def barnes_point(sq_dist, values, kappa, gamma=1):
    r"""Generate a single pass barnes interpolation value
    for a point based on the given distances, kappa and
    gamma values.

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

    weights = barnes_weights(sq_dist, kappa, gamma)
    total_weights = np.sum(weights)

    return sum([v * (w / total_weights) for (w, v) in zip(weights, values)])
