from libc.math cimport sqrt
import numpy as np

cdef double _dist_2(double x0, double y0, double x1, double y1):

    cdef double d0
    cdef double d1

    d0 = x1 - x0
    d1 = y1 - y0

    return d0 * d0 + d1 * d1

cdef double _circumcircle_radius(double x0, double y0, double x1, double y1, double x2, double y2):

    cdef double a
    cdef double b
    cdef double c

    cdef double s
    cdef double prod
    cdef double prod2
    cdef double radius

    a = _distance(x0, y0, x1, y1)
    b = _distance(x1, y1, x2, y2)
    c = _distance(x2, y2, x0, y0)

    s = (a + b + c) * 0.5

    prod = s*(a+b-s)*(a+c-s)*(b+c-s)

    prod2 = a*b*c
    radius = prod2 * prod2 / (16*prod)

    return radius

def _circumcenter(double x0, double y0, double x1, double y1, double x2, double y2):

    cdef double bc_y_diff
    cdef double ca_y_diff
    cdef double ab_y_diff
    cdef double d_inv
    cdef double a_mag
    cdef double b_mag
    cdef double c_mag

    bc_y_diff = y1 - y2
    ca_y_diff = y2 - y0
    ab_y_diff = y0 - y1

    d_inv = 0.5 / (x0 * bc_y_diff + x1 * ca_y_diff + x2 * ab_y_diff)

    a_mag = x0 * x0 + y0 * y0
    b_mag = x1 * x1 + y1 * y1
    c_mag = x2 * x2 + y2 * y2

    cx = (a_mag * bc_y_diff + b_mag * ca_y_diff + c_mag * ab_y_diff) * d_inv
    cy =(a_mag * (x2 - x1) + b_mag * (x0 - x2) + c_mag * (x1 - x0)) * d_inv

    return cx, cy

cdef double _distance(double x0, double y0, double x1, double y1):
    return sqrt(_dist_2(x0, y0, x1, y1))

def _find_nn_triangles(tri, cur_tri, position):

    cdef:
        double x0
        double y0
        double x1
        double y1
        double x2
        double y2
        double triangle[3][2]
        list nn = []
        set tri_queue

    #nn = []

    tri_queue = set(tri.neighbors[cur_tri])

    tri_queue |= set(tri.neighbors[tri.neighbors[cur_tri]].flat)
    tri_queue.discard(-1)

    for neighbor in tri_queue:

        triangle = tri.points[tri.simplices[neighbor]]

        x0 = triangle[0][0]
        y0 = triangle[0][1]
        x1 = triangle[1][0]
        y1 = triangle[1][1]
        x2 = triangle[2][0]
        y2 = triangle[2][1]


        cur_x, cur_y = _circumcenter(x0, y0, x1, y1, x2, y2)
        r = _circumcircle_radius(x0, y0, x1, y1, x2, y2)

        if _dist_2(position[0], position[1], cur_x, cur_y) < r:

            nn.append(neighbor)

    return nn

def _find_local_boundary(triangulation, triangles):

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

def _order_edges(edges):

    edge = edges[0]
    edges = edges[1:]

    ordered_edges = []
    ordered_edges.append(edge)

    num_max = len(edges)
    while len(edges) > 0 and num_max > 0:

        match = edge[1]

        for search_edge in edges:
            vertex = search_edge[0]
            if match == vertex:
                edge = search_edge
                edges.remove(edge)
                ordered_edges.append(search_edge)
                break
        num_max -= 1

    return ordered_edges

def _area(triangle):
    return np.sum(np.cross(triangle, triangle[1:] + [triangle[0]])) / 2.

