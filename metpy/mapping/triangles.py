import math
import numpy as np
from numba import jit
from scipy.spatial import Delaunay
#from scipy.spatial.distance import euclidean

from shapely.geometry import Polygon, Point

#class Triangles(object):

#    def __init__(self, points):

#        self.delaunay = Delaunay(points)

#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise
# .euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distance
def dist_2(x0, y0, x1, y1):
    d0 = x1 - x0
    d1 = y1 - y0
    return d0 * d0 + d1 * d1

def distance(p0, p1):
    return math.sqrt(dist_2(p0[0], p0[1], p1[0], p1[1]))

def circumcircle_radius(triangle):

    a = distance(triangle[0], triangle[1])
    b = distance(triangle[1], triangle[2])
    c = distance(triangle[2], triangle[0])

    s = (a + b + c) * 0.5

    prod = s*(a+b-s)*(a+c-s)*(b+c-s)

    prod2 = a*b*c
    radius = prod2 * prod2 / (16*prod)

    return radius

#This function assumes that only natural neighbor triangles
#are passed.
def find_local_boundary(triangulation, triangles):

    edges = []

    for triangle in triangles:

        for i in range(3):

            pt1 = triangulation.simplices[triangle][i]
            pt2 = triangulation.simplices[triangle][(i + 1) % 3]

            if (pt1, pt2) in edges:
                edges.remove((pt1, pt2))

            elif (pt2, pt1) in edges:

                edges.remove((pt2, pt1))
            else:
                edges.append((pt1, pt2))

    return edges

def circumcenter(pt0, pt1, pt2):

    a_x = pt0[0]
    a_y = pt0[1]
    b_x = pt1[0]
    b_y = pt1[1]
    c_x = pt2[0]
    c_y = pt2[1]

    bc_y_diff = b_y - c_y
    ca_y_diff = c_y - a_y
    ab_y_diff = a_y - b_y
    d_inv = 0.5 / (a_x * bc_y_diff + b_x * ca_y_diff + c_x * ab_y_diff)

    a_mag = a_x * a_x + a_y * a_y
    b_mag = b_x * b_x + b_y * b_y
    c_mag = c_x * c_x + c_y * c_y
    cx = (a_mag * bc_y_diff + b_mag * ca_y_diff + c_mag * ab_y_diff) * d_inv
    cy = (a_mag * (c_x - b_x) + b_mag * (a_x - c_x) + c_mag * (b_x - a_x))* d_inv

    return cx, cy

def find_nn_triangles(tri, cur_tri, position):

    nn = []

    tri_queue = set(tri.neighbors[cur_tri])

    tri_queue |= set(tri.neighbors[tri.neighbors[cur_tri]].flat)
    tri_queue.discard(-1)

    for neighbor in tri_queue:

        triangle = tri.points[tri.simplices[neighbor]]
        cur_x, cur_y = circumcenter(triangle[0], triangle[1], triangle[2])
        r = circumcircle_radius(triangle)

        if dist_2(position[0], position[1], cur_x, cur_y) < r:

            nn.append(neighbor)

    return nn


