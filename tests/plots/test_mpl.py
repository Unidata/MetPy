# Copyright (c) 2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `_mpl` module."""

from tempfile import TemporaryFile

import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import numpy as np

# Needed to trigger scattertext monkey-patching
import metpy.plots  # noqa: F401, I202


# Avoiding an image-based test here since that would involve text, which can be tricky
# to handle robustly
def test_scattertext_patheffect_empty():
    """Test scattertext with empty strings and PathEffects (Issue #245)."""
    strings = ['abc', '', 'def']
    x, y = np.arange(6).reshape(2, 3)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scattertext(x, y, strings, color='white',
                   path_effects=[mpatheffects.withStroke(linewidth=1, foreground='black')])

    # Need to trigger a render
    with TemporaryFile('wb') as fobj:
        fig.savefig(fobj)
