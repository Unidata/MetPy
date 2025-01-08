# Copyright (c) 2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `metpy.plots.text` module."""

from tempfile import TemporaryFile

import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import numpy as np
import pytest

from metpy.plots import scattertext
from metpy.testing import autoclose_figure


# Avoiding an image-based test here since that would involve text, which can be tricky
# to handle robustly
def test_scattertext_patheffect_empty():
    """Test scattertext with empty strings and PathEffects (Issue #245)."""
    strings = ['abc', '', 'def']
    x, y = np.arange(6).reshape(2, 3)
    with autoclose_figure() as fig:
        ax = fig.add_subplot(1, 1, 1)
        scattertext(ax, x, y, strings, color='white',
                    path_effects=[mpatheffects.withStroke(linewidth=1, foreground='black')])

        # Need to trigger a render
        with TemporaryFile('wb') as fobj:
            fig.savefig(fobj)


@pytest.mark.mpl_image_compare(remove_text=True, style='default', tolerance=0.069)
def test_scattertext_scalar_text():
    """Test that scattertext can work properly with multiple points but single text."""
    x, y = np.arange(6).reshape(2, 3)
    fig, ax = plt.subplots()
    scattertext(ax, x, y, 'H')
    return fig


def test_scattertext_formatter():
    """Test that scattertext supports formatting arguments."""
    x, y = np.arange(6).reshape(2, 3)
    vals = [1, 2, 3]
    with autoclose_figure() as fig:
        ax = fig.add_subplot(1, 1, 1)
        tc = scattertext(ax, x, y, vals, formatter='02d')
        assert tc.text == ['01', '02', '03']
