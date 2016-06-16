from shapely.geometry import Polygon
#shoelace algorithm
def area(poly):
    return Polygon(poly).area
    #A = 0.0
    #n = len(poly)

    #for i in range(n):
    #    A += poly[i][0] * poly[(i+1) % n][1] - poly[(i+1) % n][0] * poly[i][1]

    #return abs(A) / 2.0

def order_edges(edges):

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