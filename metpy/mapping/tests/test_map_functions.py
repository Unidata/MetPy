# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from metpy.mapping.map_functions import *

import numpy as np

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal


def test_calc_kappa():

    x = np.array([8, 67, 79, 10, 52, 53, 98, 34, 15, 58])
    y = np.array([24, 87, 48, 94, 98, 66, 14, 24, 60, 16])

    spacing = np.mean((cdist(list(zip(x, y)), list(zip(x, y)))))

    value = calc_kappa(spacing)

