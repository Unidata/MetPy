

def _area(poly):

    return abs(A(poly))

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

cdef double A(poly):

    cdef double A = 0.0
    cdef int n = len(poly)
    cdef int i = 0

    for i in range(n):
        A += poly[i][0] * poly[(i+1) % n][1] - poly[(i+1) % n][0] * poly[i][1]

    return A / 2.0



# def find_outer_edges(polygon):
#
#     edges = []
#
#     tri = Delaunay(polygon)
#
#     for triangle in tri.simplices:
#
#         for i in range(3):
#
#             pt1 = triangle[i]
#             pt2 = triangle[(i + 1) % 3]
#
#             if (pt1, pt2) in edges:
#                 edges.remove((pt1, pt2))
#
#             elif (pt2, pt1) in edges:
#
#                 edges.remove((pt2, pt1))
#             else:
#                 edges.append((pt1, pt2))
#
#     print(edges)
#     print(tri.points)
#     return [[tri.points[x], tri.points[y]] for x, y in edges]