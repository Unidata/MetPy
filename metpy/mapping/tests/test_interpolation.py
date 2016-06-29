# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from metpy.mapping.interpolation import *
from metpy.mapping.points import generate_grid, get_boundary_coords

import numpy as np
np.random.seed(2)

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal

def test_natural_neighbor():

    pts = np.random.randint(100, 900, (20, 2)) * .981234

    xp = pts[:, 0]
    yp = pts[:, 1]

    z = np.array(xp * yp, dtype=float)

    xg, yg = points.generate_grid(10, points.get_boundary_coords(xp, yp), ignore_warnings=True)

    pts = points.generate_grid_coords(xg, yg)

    img = natural_neighbor(xp, yp, z, xg, yg)

