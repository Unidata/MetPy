import numpy as np

from metpy.mapping import _triangles, _polygons
from scipy.spatial import Delaunay, ConvexHull


def natural_neighbor(xp, yp, variable, grid_points):

    tri = Delaunay(list(zip(xp, yp)))
    tri_match = tri.find_simplex(grid_points)

    img = np.empty(shape=(grid_points.shape[0]), dtype=variable.dtype)
    img.fill(np.nan)

    for ind, (cur_tri, grid) in enumerate(zip(tri_match, grid_points)):

        total_area = 0.0

        if cur_tri != -1:

            neighbors = _triangles._find_nn_triangles(tri, cur_tri, grid)

            new_tri = tri.simplices[neighbors]

            edges = _triangles._find_local_boundary(tri, neighbors)

            starting_indices = [segment[0] for segment in _polygons._order_edges(edges)]

            edge_vertices = tri.points[starting_indices]

            area_list = []
            num_vertices = len(edge_vertices)
            for i in range(num_vertices):

                p1 = edge_vertices[i]
                p2 = edge_vertices[(i + 1) % num_vertices]
                p3 = edge_vertices[(i + 2) % num_vertices]

                polygon = []

                polygon.append(_triangles._circumcenter(grid[0], grid[1], p1[0], p1[1], p2[0], p2[1]))
                polygon.append(_triangles._circumcenter(grid[0], grid[1], p2[0], p2[1], p3[0], p3[1]))

                for new in new_tri:
                    points = tri.points[new]
                    if p2 in points:
                        polygon.append(_triangles._circumcenter(points[0, 0], points[0, 1],
                                                     points[1, 0], points[1, 1],
                                                     points[2, 0], points[2, 1]))

                pts = [polygon[i] for i in ConvexHull(polygon).vertices]
                value = variable[(p2[0]==xp) & (p2[1]==yp)]

                cur_area = _polygons._area(pts)
                total_area += cur_area

                area_list.append(cur_area * value[0])

            img[ind] = sum([x / total_area for x in area_list])

    return img