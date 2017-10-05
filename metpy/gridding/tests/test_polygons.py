# Copyright (c) 2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `polygons` module."""

from __future__ import division

from numpy.testing import assert_almost_equal, assert_array_equal

from metpy.gridding.polygons import (area, order_edges)


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
