# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test path effects."""

import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest

from metpy.plots import patheffects
# Fixtures to make sure we have the right backend and consistent round
from metpy.testing import set_agg_backend  # noqa: F401, I202


@pytest.mark.mpl_image_compare(savefig_kwargs={'dpi': 300}, remove_text=True)
def test_scalloped_stroke():
    """Test ScallopedStroke path effect."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)

    # test data
    x = [-0.172, 1.078, 0.428, 0.538, 0.178,
         -0.212, -0.482, -0.722, -0.462, -0.172]
    y = [1.264, 0.784, -0.076, -0.846, -1.126,
         -1.246, -1.006, 0.234, 0.754, 1.264]
    verts = np.array([[x, y] for x, y in zip(x, y)])
    codes = np.repeat(Path.LINETO, len(x))
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='#d10000', edgecolor='#000000',
                              path_effects=[patheffects.ScallopedStroke(side='left',
                                                                        spacing=10,
                                                                        length=1.15)])

    ax.add_patch(patch)
    ax.axis('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    return fig
