# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division

from ..package_tools import Exporter

exporter = Exporter(globals())


def area(poly):
    r"""Find the area of a given polygon using the shoelace algorithm.

    Parameters
    ----------
    poly: (2, N) ndarray
        2-dimensional coordinates representing an ordered
        traversal around the edge a polygon.

    Returns
    -------
    area: float
    """

    a = 0.0
    n = len(poly)

    for i in range(n):
        a += poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]

    return abs(a) / 2.0


def order_edges(edges):
    r"""Return an ordered traversal of the edges of a
    two-dimensional polygon.

    Parameters
    ----------
    edges: (2, N) ndarray
        List of unordered line segments, where each
        line segment is represented by two unique
        vertex codes.

    Returns
    -------
    ordered_edges: (2, N) ndarray
    """

    edge = edges[0]
    edges = edges[1:]

    ordered_edges = list()
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
