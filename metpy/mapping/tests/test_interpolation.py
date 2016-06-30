# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from metpy.mapping.interpolation import *
from metpy.mapping.triangles import find_natural_neighbors
from metpy.mapping.points import generate_grid, get_boundary_coords

import numpy as np

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal

def test_nn_point():

    xp = np.array([ 8, 67, 79, 10, 52, 53, 98, 34, 15, 58])
    yp = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16])

    z = np.array([0.064, 4.489, 6.241, 0.1, 2.704, 2.809, 9.604, 1.156,
                  0.225,  3.364])

    tri = Delaunay(list(zip(xp, yp)))

    sim_gridx = [30]
    sim_gridy = [30]

    members, tri_info = find_natural_neighbors(tri, list(zip(sim_gridx, sim_gridy)))

    val = nn_point(xp, yp, z, [sim_gridx[0], sim_gridy[0]], tri, members[0], tri_info)

    truth = 1.009

    assert_almost_equal(truth, val, 3)

