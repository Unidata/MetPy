import numpy as np

from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean


#class Triangles(object):

#    def __init__(self, points):

#        self.delaunay = Delaunay(points)


def circumcircle_radius(triangle):

    a = euclidean(triangle[0], triangle[1])
    b = euclidean(triangle[1], triangle[2])
    c = euclidean(triangle[2], triangle[0])

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