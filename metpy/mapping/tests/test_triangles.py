# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from metpy.mapping.triangles import *
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from scipy.spatial import Delaunay

import numpy as np


def test_dist_2():

    x0 = 0
    y0 = 0

    x1 = 10
    y1 = 10

    truth = 200

    dist2 = dist_2(x0, y0, x1, y1)

    assert_almost_equal(truth, dist2)


def test_distance():

    pt0 = [0, 0]
    pt1 = [10, 10]

    truth = 14.14213562373095

    dist = distance(pt0, pt1)

    assert_almost_equal(truth, dist)


def test_circumcircle_radius_2():

    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    cc_r2 = circumcircle_radius_2(pt0, pt1, pt2)

    truth = 50

    assert_almost_equal(truth, cc_r2, decimal=2)


def test_circumcircle_radius():

    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    cc_r = circumcircle_radius(pt0, pt1, pt2)

    truth = 7.07

    assert_almost_equal(truth, cc_r, decimal=2)


def test_circumcenter():

    pt0 = [0, 0]
    pt1 = [10, 10]
    pt2 = [10, 0]

    cc = circumcenter(pt0, pt1, pt2)

    truth = [5., 5.]

    assert_array_almost_equal(truth, cc)


def test_find_nn_triangles():

    # creates a triangulation where all the triangles are the
    # same size but with different orientations and positions.
    # the circumcircle radius is 0.707 for each triangle and
    # the point 4.5, 4.5 is equal to the circumcenter for exactly
    # two triangles.  This was tested and verified by hand.

    x = list(range(10))
    y = list(range(10))
    gx, gy = np.meshgrid(x, y)
    pts = np.vstack([gx.ravel(), gy.ravel()]).T
    tri = Delaunay(pts)

    tri_match = tri.find_simplex([4.5, 4.5])

    truth = [62, 63]

    nn = find_nn_triangles(tri, tri_match, [4.5, 4.5])

    assert_array_almost_equal(truth, nn)


def test_find_local_boundary():

    # creates a triangulation where all the triangles are the
    # same size but with different orientations and positions.
    # the circumcircle radius is 0.707 for each triangle and
    # the point 4.5, 4.5 is equal to the circumcenter for exactly
    # two triangles.  This was verified by pen and paper.

    x = list(range(10))
    y = list(range(10))
    gx, gy = np.meshgrid(x, y)
    pts = np.vstack([gx.ravel(), gy.ravel()]).T
    tri = Delaunay(pts)

    tri_match = tri.find_simplex([4.5, 4.5])

    truth = [62, 63]

    nn = find_nn_triangles(tri, tri_match, [4.5, 4.5])

    # point codes for 2d coordinates in triangulation
    edges = find_local_boundary(tri, nn)

    # point codes for 2d coordinates in triangulation
    # ( ([5.0, 4.0], [5.0, 5.0]),
    #   ([4.0, 4.0], [5.0, 4.0]),
    #   ([5.0, 5.0], [4.0, 5.0]),
    #   ([4.0, 5.0], [4.0, 4.0]) )
    truth = [(45, 55), (44, 45), (55, 54), (54, 44)]

    assert_array_almost_equal(truth, edges)
