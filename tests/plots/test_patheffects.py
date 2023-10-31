# Copyright (c) 2022,2023 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `patheffects` module."""
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import interp1d

from metpy.plots import (ColdFront, ColdFrontogenesis, ColdFrontolysis, Dryline, OccludedFront,
                         OccludedFrontogenesis, OccludedFrontolysis, RidgeAxis,
                         ScallopedStroke, Squall, StationaryFront, StationaryFrontogenesis,
                         StationaryFrontolysis, WarmFront, WarmFrontogenesis, WarmFrontolysis)


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.01)
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
def test_frontogenesis():
    """Test plotting forming front path effects."""
    x = np.linspace(0, 80, 5)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(x, np.ones_like(x), path_effects=[ColdFrontogenesis(spacing=1.0)])
    ax.plot(x, np.ones_like(x) * 3, path_effects=[WarmFrontogenesis(flip=True, spacing=1.0)])
    ax.plot(
        x, np.ones_like(x) * 5,
        path_effects=[OccludedFrontogenesis(spacing=1.0, color='black')]
    )
    ax.plot(x, np.ones_like(x) * 7, path_effects=[StationaryFrontogenesis(spacing=1.0)])
    ax.set_ylim(-0.5, 8)
    return fig


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.)
def test_frontolysis():
    """Test plotting dissipating front path effects."""
    x = np.linspace(0, 80, 5)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(x, np.ones_like(x), path_effects=[ColdFrontolysis(spacing=1.0)])
    ax.plot(
        x, np.ones_like(x) * 3,
        path_effects=[WarmFrontolysis(spacing=1.0, color='orange')]
    )
    ax.plot(x, np.ones_like(x) * 5, path_effects=[OccludedFrontolysis(spacing=1.0)])
    ax.plot(
        x, np.ones_like(x) * 7,
        path_effects=[StationaryFrontolysis(flip=True, spacing=1.0)]
    )
    ax.set_ylim(-0.5, 8)
    return fig


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.023)
def test_special():
    """Test plotting special path effects."""
    x = np.linspace(0, 20, 5)
    y = np.linspace(0, 20, 5)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(x, y, path_effects=[Dryline(flip=True)])
    ax.plot(x + 5, y, linewidth=3, path_effects=[RidgeAxis()])
    ax.plot(x + 10, y, path_effects=[Squall(color='red')])
    ax.set_ylim(-1, 21)
    ax.set_xlim(-1, 31)
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


@pytest.mark.mpl_image_compare(savefig_kwargs={'dpi': 300}, remove_text=True)
def test_scalloped_stroke_closed():
    """Test ScallopedStroke path effect."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)

    # test data
    x = [-0.172, 1.078, 0.428, 0.538, 0.178,
         -0.212, -0.482, -0.722, -0.462, -0.172]
    y = [1.264, 0.784, -0.076, -0.846, -1.126,
         -1.246, -1.006, 0.234, 0.754, 1.264]
    verts = np.array([[x, y] for x, y in zip(x, y)])
    codes = np.repeat(mpath.Path.LINETO, len(x))
    codes[0] = mpath.Path.MOVETO
    codes[-1] = mpath.Path.CLOSEPOLY

    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='#d10000', edgecolor='#000000',
                               path_effects=[
                                   ScallopedStroke(side='left', spacing=10, length=1.15)])

    ax.add_patch(patch)
    ax.axis('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    return fig


@pytest.mark.mpl_image_compare(savefig_kwargs={'dpi': 300}, remove_text=True)
def test_scalloped_stroke_segment():
    """Test ScallopedStroke path effect."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)

    # test data
    x = np.arange(9)
    y = np.concatenate([np.arange(5), np.arange(3, -1, -1)])
    verts = np.array([[x, y] for x, y in zip(x, y)])
    codes = np.repeat(mpath.Path.LINETO, len(x))
    codes[0] = mpath.Path.MOVETO

    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor='#000000',
                               path_effects=[
                                   ScallopedStroke(side='left', spacing=10, length=1.15)])

    ax.add_patch(patch)
    ax.axis('equal')
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 4.5)

    return fig
