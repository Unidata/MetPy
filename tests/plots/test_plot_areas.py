# Copyright (c) 2023 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the handling of plot areas outside of declarative."""

import matplotlib.pyplot as plt
import pytest

from metpy.testing import needs_cartopy


@pytest.mark.mpl_image_compare(tolerance=0.005)
@needs_cartopy
def test_uslcc_plotting(ccrs, cfeature):
    """Test plotting the uslcc area with projection."""
    from metpy.plots.plot_areas import areas
    area = 'uslcc'

    # Get the extent and project for the selected area
    extent = areas.loc[area].bounds
    proj = areas.loc[area].proj

    # Plot a simple figure for the selected area
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.005)
@needs_cartopy
def test_au_plotting(ccrs, cfeature):
    """Test plotting the au area with projection."""
    from metpy.plots.plot_areas import areas
    area = 'au'

    # Get the extent and project for the selected area
    extent = areas.loc[area].bounds
    proj = areas.loc[area].proj

    # Plot a simple figure for the selected area
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.008)
@needs_cartopy
def test_cn_plotting(ccrs, cfeature):
    """Test plotting the cn area with projection."""
    from metpy.plots.plot_areas import areas
    area = 'cn'

    # Get the extent and project for the selected area
    extent = areas.loc[area].bounds
    proj = areas.loc[area].proj

    # Plot a simple figure for the selected area
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='black')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.005)
@needs_cartopy
def test_hi_plotting(ccrs, cfeature):
    """Test plotting the hi area with projection."""
    from metpy.plots.plot_areas import areas
    area = 'hi'

    # Get the extent and project for the selected area
    extent = areas.loc[area].bounds
    proj = areas.loc[area].proj

    # Plot a simple figure for the selected area
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='black')

    return fig


@pytest.mark.mpl_image_compare(tolerance=0.005)
@needs_cartopy
def test_wpac_plotting(ccrs, cfeature):
    """Test plotting the wpac area with projection."""
    from metpy.plots.plot_areas import areas
    area = 'wpac'

    # Get the extent and project for the selected area
    extent = areas.loc[area].bounds
    proj = areas.loc[area].proj

    # Plot a simple figure for the selected area
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='black')

    return fig
