# Copyright (c) 2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `patheffects` module."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import interp1d

from metpy.plots import ColdFront, OccludedFront, StationaryFront, WarmFront


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.)
def test_fronts():
    """Basic test of plotting fronts using path effects."""
    x = np.linspace(0, 80, 5)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(x, x ** 3, linewidth=2, path_effects=[ColdFront(flip=True, spacing=1.0)])
    ax.plot(x, 60000 * np.sqrt(x), linewidth=0.5,
            path_effects=[WarmFront(spacing=1.0, color='brown')])
    ax.plot(x, 75000 * np.sqrt(x), linewidth=1, path_effects=[OccludedFront(spacing=1.0)])
    ax.plot(x, 6500 * x, linewidth=1, path_effects=[StationaryFront(spacing=1.0)])
    return fig


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.)
def test_curved_stationary():
    """Test that curved stationary fronts don't have weird filled artifacts."""
    n = 100
    front_lat = [31.08, 31.76, 32.18, 32.01, 31.7, 32.44]
    front_lon = [-88.08, -89.23, -90.83, -92.08, -93.27, -95.55]

    index = np.arange(len(front_lon))
    ii = np.linspace(0, index.max(), n)

    xcurve = interp1d(index, front_lon, 'cubic')(ii)
    ycurve = interp1d(index, front_lat, 'cubic')(ii)

    fig, ax = plt.subplots()
    ax.plot(xcurve, ycurve, path_effects=[StationaryFront(spacing=3, size=6)], linewidth=2)
    return fig


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.)
def test_stationary_spacing():
    """Test various aspects of spacing around stationary fronts."""
    x = [0, 1]
    y = [1, 1]

    fig, ax = plt.subplots()
    ax.plot(x, y, path_effects=[StationaryFront(spacing=3, size=15)], linewidth=2)
    ax.plot(x, [0.98, 0.98], path_effects=[StationaryFront(spacing=3, size=15)], linewidth=15)
    ax.plot(x, [0.96, 0.96], path_effects=[StationaryFront(spacing=3, size=15, flip=True)],
            linewidth=15)
    ax.set_ylim(0.95, 1.05)
    ax.set_xlim(-0.05, 1.05)

    return fig
