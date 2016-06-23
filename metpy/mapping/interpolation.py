# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from scipy.spatial import Delaunay, ConvexHull, cKDTree

from metpy.mapping import triangles, polygons, points


def natural_neighbor(xp, yp, variable, grid_x, grid_y):
    '''Generate a natural neighbor interpolation of the given
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
    '''

    tri = Delaunay(list(zip(xp, yp)))

    grid_points = points.generate_grid_coords(grid_x, grid_y)

    tri_match = tri.find_simplex(grid_points)

    img = np.empty(shape=(grid_points.shape[0]), dtype=variable.dtype)
    img.fill(np.nan)

    for ind, (cur_tri, grid) in enumerate(zip(tri_match, grid_points)):
        total_area = 0.0

        if cur_tri != -1:
            neighbors = triangles.find_nn_triangles(tri, cur_tri, grid)

            new_tri = tri.simplices[neighbors]

            edges = triangles.find_local_boundary(tri, neighbors)

            starting_indices = [segment[0] for segment in polygons.order_edges(edges)]

            edge_vertices = tri.points[starting_indices]

            area_list = []
            num_vertices = len(edge_vertices)

            for i in range(num_vertices):
                p1 = edge_vertices[i]
                p2 = edge_vertices[(i + 1) % num_vertices]
                p3 = edge_vertices[(i + 2) % num_vertices]

                polygon = []

                polygon.append(triangles.circumcenter(grid, p1, p2))
                polygon.append(triangles.circumcenter(grid, p3, p2))

                for new in new_tri:
                    pts = tri.points[new]
                    if p2 in pts:
                        polygon.append(triangles.circumcenter(pts[0], pts[1], pts[2]))

                pts = [polygon[i] for i in ConvexHull(polygon).vertices]
                value = variable[(p2[0] == xp) & (p2[1] == yp)]

                cur_area = polygons.area(pts)
                total_area += cur_area

                area_list.append(cur_area * value[0])

            img[ind] = sum([x / total_area for x in area_list])

    img = img.reshape(grid_x.shape)
    return img


def barnes_weights(dist, kappa, gamma=1.0):
    '''Calculate the barnes weights for observation points
    based on their distance from an interpolation point.

    Parameters
    ----------
    dist: (N, ) ndarray
        Distances from interpolation point
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
    '''

    return np.exp(-dist / (kappa * gamma))


def cressman_weights(dist, r):
    '''Calculate the cressman weights for observation points
    based on their distance from an interpolation point.

    Parameters
    ----------
    dist: (N, ) ndarray
        Distances from interpolation point
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
    '''

    return (r * r - dist) / (r * r + dist)


def inverse_distance(xp, yp, variable, grid_x, grid_y, r, gamma=None, kappa=None,
                     min_neighbors=3, kind='cressman'):
    '''Generate an inverse distance weighting interpolation of the given
    points to the given grid based on either Cressman (1959) or Barnes (1964).
    The Barnes implementation used here based on Koch et al. (1983).

    Cressman, George P. "An operational objective analysis system."
        Mon. Wea. Rev 87, no. 10 (1959): 367-374.

    Barnes, S. L., 1964: A technique for maximizing details in numerical
        weather map analysis. J. Appl. Meteor., 3, 396–409.

    Koch, S. E., M. DesJardins, and P. J. Kocin, 1983: An interactive
        Barnes objective analysis scheme for use with satellite and conventional
        data. J. Climate Appl. Meteor., 22, 1487–1503.

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
        Minimum number of neighbors needed to perform barnes or cressman interpolation for a point. Default is 3.
    kind: str
        Specify what inverse distance weighting interpolation to use.
        Options: 'cressman' or 'barnes'. Default 'cressman'

    Returns
    -------
    img: (M, N) ndarray
        Interpolated values on a 2-dimensional grid
    '''

    obs_tree = cKDTree(list(zip(xp, yp)))

    grid_points = points.generate_grid_coords(grid_x, grid_y)

    indices = obs_tree.query_ball_point(grid_points, r=r)

    img = np.empty(shape=(grid_points.shape[0]), dtype=variable.dtype)
    img.fill(np.nan)

    for idx, (matches, grid) in enumerate(zip(indices, grid_points)):
        if len(matches) >= min_neighbors:
            x0, y0 = grid
            x1, y1 = obs_tree.data[matches].T

            values = variable[matches]

            dist = triangles.dist_2(x0, y0, x1, y1)

            if kind == 'cressman':
                weights = cressman_weights(dist, r)
                total_weights = np.sum(weights)
                img[idx] = sum([v * (w / total_weights) for (w, v) in zip(weights, values)])

            elif kind == 'barnes':

                weights = barnes_weights(dist, kappa)
                weights_ = barnes_weights(dist, kappa, gamma)

                total_weights = np.sum(weights)
                img[idx] = np.sum(values * (weights / total_weights))

                total_weights_ = np.sum(weights_)
                img[idx] += np.sum((values - img[idx]) * (weights_ / total_weights_))

    img.reshape(grid_x.shape)
    return img

