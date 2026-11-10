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


@pytest.mark.parametrize('layout_size', [3, 4])
def test_scattertext_layout_rendering(monkeypatch, layout_size):
    """Both Matplotlib layout shapes render like individual Text artists."""
    images = []
    for use_collection in (False, True):
        fig = plt.figure(figsize=(3, 2), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
        x, y, labels = [.15, .65], [.25, .7], ['A', 'B\nC']
        options = {'fontsize': 18, 'color': 'black', 'horizontalalignment': 'left',
                   'verticalalignment': 'baseline'}
        if use_collection:
            collection = scattertext(ax, x, y, labels, **options)
            native_layout = collection._get_layout

            def version_layout(renderer, native_layout=native_layout):
                bbox, rows, descent = native_layout(renderer)
                adjusted = []
                for row in rows:
                    line, metrics = row[:2]
                    xy = row[2] if len(row) == 3 else row[2:]
                    adjusted.append((line, metrics, xy) if layout_size == 3
                                    else (line, metrics, *xy))
                return bbox, adjusted, descent

            monkeypatch.setattr(collection, '_get_layout', version_layout)
        else:
            for px, py, label in zip(x, y, labels, strict=True):
                ax.text(px, py, label, **options)
        fig.canvas.draw()
        images.append(np.array(fig.canvas.buffer_rgba()))
        plt.close(fig)
    np.testing.assert_array_equal(images[0], images[1])
