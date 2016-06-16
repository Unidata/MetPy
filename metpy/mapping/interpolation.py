import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, ConvexHull
from metpy.mapping import triangles, points, polygons

#from https://github.com/metpy/MetPy/files/138653/cwp-657.pdf
def natural_neighbor(xp, yp, variable, grid_points):

    tri = Delaunay(list(zip(xp, yp)))
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
                    points = tri.points[new]
                    if p2 in points:
                        polygon.append(triangles.circumcenter(points[0], points[1], points[2]))

                pts = [polygon[i] for i in ConvexHull(polygon).vertices]
                value = variable[(p2[0]==xp) & (p2[1]==yp)]

                cur_area = polygons.area(pts)
                total_area += cur_area

                area_list.append(cur_area * value[0])

            img[ind] = sum([x / total_area for x in area_list])
                #np.dot(area_list[:, 0], area_list[:, 1]) / total_area

    return img

def barnes():

    return 0

def cressman():

    return 0

def radial_basis_functions():

    return 0