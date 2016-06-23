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
                value = variable[(p2[0]==xp) & (p2[1]==yp)]

                cur_area = polygons.area(pts)
                total_area += cur_area

                area_list.append(cur_area * value[0])

            img[ind] = sum([x / total_area for x in area_list])

    img = img.reshape(grid_x.shape)
    return img


def barnes_weights(dist, kappa, gamma=1.0):

    weights = np.exp(-dist / (kappa * gamma))
    return weights


def cressman_weights(dist, r):

    return (r * r - dist) / (r * r + dist)


def inverse_distance(xp, yp, variable, grid_x, grid_y, r, gamma=None, kappa=None,
                     min_neighbors=3, kind='cressman'):
    '''Generate a cressman weights interpolation of the given
    points to the given grid based on Cressman (1959).

    Cressman, George P. "An operational objective analysis system."
    Mon. Wea. Rev 87, no. 10 (1959): 367-374.

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

