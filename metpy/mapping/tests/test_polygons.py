# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division

from metpy.mapping.polygons import (area, order_edges)
from numpy import isclose
from numpy.testing import assert_array_equal


def test_area():
    r"""Tests get area of polygon function"""

    pt0 = [0, 0]
    pt1 = [5, 5]
    pt2 = [5, 0]

    truth = 12.5

    assert isclose(truth, area([pt0, pt1, pt2]))


def test_order_edges():
    r"""Tests order edges of polygon function"""

    edges = [[1, 2], [5, 6], [4, 5], [2, 3], [6, 1], [3, 4]]

    truth = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]]

    assert_array_equal(truth, order_edges(edges))
