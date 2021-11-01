# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the cartopy utilities."""
import matplotlib.pyplot as plt
import pytest

import metpy.plots as mpplots


@pytest.mark.mpl_image_compare(tolerance=0.053, remove_text=True)
def test_us_county_defaults(ccrs):
    """Test the default US county plotting."""
    proj = ccrs.LambertConformal(central_longitude=-85.0, central_latitude=45.0)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([270.25, 270.9, 38.15, 38.75], ccrs.Geodetic())
    ax.add_feature(mpplots.USCOUNTIES)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0.092, remove_text=True)
def test_us_county_scales(ccrs):
    """Test US county plotting with all scales."""
    proj = ccrs.LambertConformal(central_longitude=-85.0, central_latitude=45.0)

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(1, 3, 1, projection=proj)
    ax2 = fig.add_subplot(1, 3, 2, projection=proj)
    ax3 = fig.add_subplot(1, 3, 3, projection=proj)

    for scale, axis in zip(['20m', '5m', '500k'], [ax1, ax2, ax3]):
        axis.set_extent([270.25, 270.9, 38.15, 38.75], ccrs.Geodetic())
        axis.add_feature(mpplots.USCOUNTIES.with_scale(scale))
    return fig


@pytest.mark.mpl_image_compare(tolerance=0.053, remove_text=True)
def test_us_states_defaults(ccrs):
    """Test the default US States plotting."""
    proj = ccrs.LambertConformal(central_longitude=-85.0, central_latitude=45.0)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([270, 280, 28, 39], ccrs.Geodetic())
    ax.add_feature(mpplots.USSTATES)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0.092, remove_text=True)
def test_us_states_scales(ccrs):
    """Test the default US States plotting with all scales."""
    proj = ccrs.LambertConformal(central_longitude=-85.0, central_latitude=45.0)

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(1, 3, 1, projection=proj)
    ax2 = fig.add_subplot(1, 3, 2, projection=proj)
    ax3 = fig.add_subplot(1, 3, 3, projection=proj)

    for scale, axis in zip(['20m', '5m', '500k'], [ax1, ax2, ax3]):
        axis.set_extent([270, 280, 28, 39], ccrs.Geodetic())
        axis.add_feature(mpplots.USSTATES.with_scale(scale))
    return fig


def test_cartopy_stub(monkeypatch):
    """Test that the CartoPy stub will issue an error if CartoPy is not present."""
    import sys

    # This makes sure that cartopy is not found
    monkeypatch.setitem(sys.modules, 'cartopy.crs', None)

    ccrs = mpplots.cartopy_utils.import_cartopy()
    with pytest.raises(AttributeError, match='without Cartopy'):
        ccrs.PlateCarree()


def test_plots_getattr(monkeypatch):
    """Ensure the module-level getattr works."""
    # Make sure the feature is missing
    monkeypatch.delattr(mpplots.cartopy_utils, 'USSTATES', raising=False)
    with pytest.raises(AttributeError, match='Cannot use USSTATES without Cartopy'):
        assert not mpplots.USSTATES  # Should fail on attribute lookup before assert


def test_plots_dir():
    """Ensure dir() on metpy.plots works."""
    assert 'USSTATES' in dir(mpplots)
