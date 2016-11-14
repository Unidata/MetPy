# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division

from metpy.gridding.triangles import (dist_2, distance, circumcircle_radius_2,
                                      circumcircle_radius, circumcenter,
                                      find_natural_neighbors, find_nn_triangles_point,
                                      find_local_boundary, triangle_area)

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from scipy.spatial import Delaunay

import numpy as np


def test_triangle_area():
    r"""Tests area of triangle function"""

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
    r"""Tests squared distance function"""

    x0 = 0
    y0 = 0

    x1 = 10
    y1 = 10

    truth = 200

    dist2 = dist_2(x0, y0, x1, y1)

    assert_almost_equal(truth, dist2)


def test_distance():
    r"""Tests distance function"""

    pt0 = [0, 0]
    pt1 = [10, 10]

    truth = 14.14213562373095

    dist = distance(pt0, pt1)

    assert_almost_equal(truth, dist)


def test_circumcircle_radius_2():
    r"""Tests squared circumcircle radius function"""

    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    cc_r2 = circumcircle_radius_2(pt0, pt1, pt2)

    truth = 50

    assert_almost_equal(truth, cc_r2, decimal=2)


def test_circumcircle_radius():
    r"""Tests circumcircle radius function"""

    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    cc_r = circumcircle_radius(pt0, pt1, pt2)

    truth = 7.07

    assert_almost_equal(truth, cc_r, decimal=2)


def test_circumcenter():
    r"""Tests circumcenter function"""

    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    cc = circumcenter(pt0, pt1, pt2)

    truth = [5., 5.]

    assert_array_almost_equal(truth, cc)


def test_find_natural_neighbors():
    r"""Tests find natural neighbors function"""

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
    r"""Tests find natural neighbors for a point function"""

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
    r"""Tests find edges of natural neighbor triangle group function"""

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
