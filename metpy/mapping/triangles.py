import numpy as np
from numba import jit
from scipy.spatial import Delaunay
#from scipy.spatial.distance import euclidean

from shapely.geometry import Polygon

#class Triangles(object):

#    def __init__(self, points):

#        self.delaunay = Delaunay(points)


#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise
# .euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances
def distance(x, y):
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))


def circumcircle_radius(triangle):

    a = distance(triangle[0], triangle[1])
    b = distance(triangle[1], triangle[2])
    c = distance(triangle[2], triangle[0])

    s = (a + b + c) * 0.5

    prod = s*(a+b-s)*(a+c-s)*(b+c-s)

    radius = (a*b*c) / (4*np.sqrt(prod))

    return radius


def circumcenter(triangle):

    a_x = triangle[0, 0]
    a_y = triangle[0, 1]
    b_x = triangle[1, 0]
    b_y = triangle[1, 1]
    c_x = triangle[2, 0]
    c_y = triangle[2, 1]

    d = 2 * ((a_x * (b_y - c_y)) + (b_x * (c_y - a_y)) + (c_x * (a_y - b_y)))

    cx = ((a_x ** 2 + a_y ** 2) * (b_y - c_y) + (b_x ** 2 + b_y ** 2) * (c_y - a_y) + (c_x ** 2 + c_y ** 2) * (a_y - b_y)) / d
    cy = ((a_x ** 2 + a_y ** 2) * (c_x - b_x) + (b_x ** 2 + b_y ** 2) * (a_x - c_x) + (c_x ** 2 + c_y ** 2) * (b_x - a_x)) / d

    return cx, cy

def area(triangle):

    return Polygon(triangle).area


def find_nn_triangles(tri, cur_tri, position):

    nn = []

    for adjacent_neighbor in tri.neighbors[cur_tri]:

        if not adjacent_neighbor in nn:
            triangle = tri.points[tri.simplices[adjacent_neighbor]]
            cur_x, cur_y = circumcenter(triangle)
            r = circumcircle_radius(triangle)

            if distance([position[0], position[1]], [cur_x, cur_y]) < r:
                nn.append(adjacent_neighbor)

        for second_neighbor in tri.neighbors[adjacent_neighbor]:
            if not second_neighbor in nn:
                triangle = tri.points[tri.simplices[second_neighbor]]
                cur_x, cur_y = circumcenter(triangle)
                r = circumcircle_radius(triangle)

                if distance([position[0], position[1]], [cur_x, cur_y]) < r:
                    nn.append(second_neighbor)

    return list(nn)


def plot_triangle(plt, triangle):
    x = [triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]]
    y = [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]]

    plt.plot(x, y, "k-")


def plot_voronoi_lines(plt, vor):

    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)

        if np.all(simplex >= 0):
            plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k--')