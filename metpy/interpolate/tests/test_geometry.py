# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `geometry` module."""

from __future__ import division

import logging

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from scipy.spatial import Delaunay

from metpy.interpolate.geometry import (area, circumcenter, circumcircle_radius,
                                        circumcircle_radius_2, dist_2, distance,
                                        find_local_boundary, find_natural_neighbors,
                                        find_nn_triangles_point, get_point_count_within_r,
                                        get_points_within_r, order_edges, triangle_area)

logging.getLogger('metpy.interpolate.geometry').setLevel(logging.ERROR)


def test_get_points_within_r():
    r"""Test get points within a radius function."""
    x = list(range(10))
    y = list(range(10))

    center = [1, 5]

    radius = 5

    matches = get_points_within_r(center, list(zip(x, y)), radius).T

    truth = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]

    assert_array_almost_equal(truth, matches)


def test_get_point_count_within_r():
    r"""Test get point count within a radius function."""
    x = list(range(10))
    y = list(range(10))

    center1 = [1, 5]
    center2 = [12, 10]

    radius = 5

    count = get_point_count_within_r([center1, center2], list(zip(x, y)), radius)

    truth = np.array([5, 2])

    assert_array_almost_equal(truth, count)


def test_triangle_area():
    r"""Test area of triangle function."""
    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    truth = 50.0

    t_area = triangle_area(pt0, pt1, pt2)

    assert_almost_equal(truth, t_area)

    # what if two points are the same? Its a line!
    pt0 = [0, 0]
    pt1 = [0, 0]
    pt2 = [10, 0]

    truth = 0

    t_area = triangle_area(pt0, pt1, pt2)

    assert_almost_equal(truth, t_area)


def test_dist_2():
    r"""Test squared distance function."""
    x0 = 0
    y0 = 0

    x1 = 10
    y1 = 10

    truth = 200

    dist2 = dist_2(x0, y0, x1, y1)

    assert_almost_equal(truth, dist2)


def test_distance():
    r"""Test distance function."""
    pt0 = [0, 0]
    pt1 = [10, 10]

    truth = 14.14213562373095

    dist = distance(pt0, pt1)

    assert_almost_equal(truth, dist)


def test_circumcircle_radius_2():
    r"""Test squared circumcircle radius function."""
    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    cc_r2 = circumcircle_radius_2(pt0, pt1, pt2)

    truth = 50

    assert_almost_equal(truth, cc_r2, decimal=2)


def test_circumcircle_radius():
    r"""Test circumcircle radius function."""
    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    cc_r = circumcircle_radius(pt0, pt1, pt2)

    truth = 7.07

    assert_almost_equal(truth, cc_r, decimal=2)


def test_circumcenter():
    r"""Test circumcenter function."""
    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    cc = circumcenter(pt0, pt1, pt2)

    truth = [5., 5.]

    assert_array_almost_equal(truth, cc)


def test_find_natural_neighbors():
    r"""Test find natural neighbors function."""
    x = list(range(0, 20, 4))
    y = list(range(0, 20, 4))
    gx, gy = np.meshgrid(x, y)
    pts = np.vstack([gx.ravel(), gy.ravel()]).T
    tri = Delaunay(pts)

    test_points = np.array([[2, 2], [5, 10], [12, 13.4], [12, 8], [20, 20]])

    neighbors, tri_info = find_natural_neighbors(tri, test_points)

    neighbors_truth = [[0, 1],
                       [24, 25],
                       [16, 17, 30, 31],
                       [18, 19, 20, 21, 22, 23, 26, 27],
                       []]

    for i, true_neighbor in enumerate(neighbors_truth):
        assert_array_almost_equal(true_neighbor, neighbors[i])

    cc_truth = np.array([(2.0, 2.0), (2.0, 2.0), (14.0, 2.0),
                         (14.0, 2.0), (6.0, 2.0), (6.0, 2.0),
                         (10.0, 2.0), (10.0, 2.0), (2.0, 14.0),
                         (2.0, 14.0), (6.0, 6.0), (6.0, 6.0),
                         (2.0, 6.0), (2.0, 6.0), (2.0, 10.0),
                         (2.0, 10.0), (14.0, 14.0), (14.0, 14.0),
                         (10.0, 6.0), (10.0, 6.0), (14.0, 6.0),
                         (14.0, 6.0), (14.0, 10.0), (14.0, 10.0),
                         (6.0, 10.0), (6.0, 10.0), (10.0, 10.0),
                         (10.0, 10.0), (6.0, 14.0), (6.0, 14.0),
                         (10.0, 14.0), (10.0, 14.0)])

    r_truth = np.empty((32,))
    r_truth.fill(2.8284271247461916)

    for key in tri_info:
        assert_almost_equal(cc_truth[key], tri_info[key]['cc'])
        assert_almost_equal(r_truth[key], tri_info[key]['r'])


def test_find_nn_triangles_point():
    r"""Test find natural neighbors for a point function."""
    x = list(range(10))
    y = list(range(10))
    gx, gy = np.meshgrid(x, y)
    pts = np.vstack([gx.ravel(), gy.ravel()]).T
    tri = Delaunay(pts)

    tri_match = tri.find_simplex([4.5, 4.5])

    truth = [62, 63]

    nn = find_nn_triangles_point(tri, tri_match, [4.5, 4.5])

    assert_array_almost_equal(truth, nn)


def test_find_local_boundary():
    r"""Test find edges of natural neighbor triangle group function."""
    x = list(range(10))
    y = list(range(10))
    gx, gy = np.meshgrid(x, y)
    pts = np.vstack([gx.ravel(), gy.ravel()]).T
    tri = Delaunay(pts)

    tri_match = tri.find_simplex([4.5, 4.5])

    nn = find_nn_triangles_point(tri, tri_match, [4.5, 4.5])

    edges = find_local_boundary(tri, nn)

    truth = [(45, 55), (44, 45), (55, 54), (54, 44)]

    assert_array_almost_equal(truth, edges)


def test_area():
    r"""Test get area of polygon function."""
    pt0 = [0, 0]
    pt1 = [5, 5]
    pt2 = [5, 0]

    truth = 12.5

    assert_almost_equal(area([pt0, pt1, pt2]), truth)


def test_order_edges():
    r"""Test order edges of polygon function."""
    edges = [[1, 2], [5, 6], [4, 5], [2, 3], [6, 1], [3, 4]]

    truth = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]]

    assert_array_equal(truth, order_edges(edges))
