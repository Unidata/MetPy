# Copyright (c) 2017,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `_util` module."""

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from metpy.plots import add_metpy_logo, add_timestamp, add_unidata_logo, convert_gempak_color
from metpy.testing import get_test_data

MPL_VERSION = matplotlib.__version__[:3]


@pytest.mark.mpl_image_compare(tolerance=2.638, remove_text=True)
def test_add_timestamp():
    """Test adding a timestamp to an axes object."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    add_timestamp(ax, time=datetime(2017, 1, 1))
    return fig


@pytest.mark.mpl_image_compare(tolerance=2.635, remove_text=True)
def test_add_timestamp_custom_format():
    """Test adding a timestamp to an axes object with custom time formatting."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    add_timestamp(ax, time=datetime(2017, 1, 1), time_format='%H:%M:%S %Y/%m/%d')
    return fig


@pytest.mark.mpl_image_compare(tolerance=5.389, remove_text=True)
def test_add_timestamp_pretext():
    """Test adding a timestamp to an axes object with custom pre-text."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    add_timestamp(ax, time=datetime(2017, 1, 1), pretext='Valid: ')
    return fig


@pytest.mark.mpl_image_compare(tolerance=0.844, remove_text=True)
def test_add_timestamp_high_contrast():
    """Test adding a timestamp to an axes object."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    add_timestamp(ax, time=datetime(2017, 1, 1), high_contrast=True)
    return fig


def test_add_timestamp_xarray():
    """Test that add_timestamp can work with xarray datetime accessor."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ds = xr.open_dataset(get_test_data('AK-REGIONAL_8km_3.9_20160408_1445.gini'),
                         engine='gini')
    txt = add_timestamp(ax, ds.time.dt, pretext='')
    assert txt.get_text() == '2016-04-08T14:45:20Z'


@pytest.mark.mpl_image_compare(tolerance=0.004, remove_text=True)
def test_add_metpy_logo_small():
    """Test adding a MetPy logo to a figure."""
    fig = plt.figure(figsize=(9, 9))
    add_metpy_logo(fig)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0.004, remove_text=True)
def test_add_metpy_logo_large():
    """Test adding a large MetPy logo to a figure."""
    fig = plt.figure(figsize=(9, 9))
    add_metpy_logo(fig, size='large')
    return fig


@pytest.mark.mpl_image_compare(tolerance=0.004, remove_text=True)
def test_add_unidata_logo():
    """Test adding a Unidata logo to a figure."""
    fig = plt.figure(figsize=(9, 9))
    add_unidata_logo(fig)
    return fig


def test_add_logo_invalid_size():
    """Test adding a logo to a figure with an invalid size specification."""
    fig = plt.figure(figsize=(9, 9))
    with pytest.raises(ValueError):
        add_metpy_logo(fig, size='jumbo')


@pytest.mark.mpl_image_compare(tolerance={'3.3': 1.072}.get(MPL_VERSION, 0),
                               remove_text=True)
def test_gempak_color_image_compare():
    """Test creating a plot with all the GEMPAK colors."""
    c = range(32)
    mplc = convert_gempak_color(c)

    delta = 0.025
    x = y = np.arange(-3.0, 3.01, delta)
    xx, yy = np.meshgrid(x, y)
    z1 = np.exp(-xx**2 - yy**2)
    z2 = np.exp(-(xx - 1)**2 - (yy - 1)**2)
    z = (z1 - z2) * 2

    fig = plt.figure(figsize=(9, 9))
    cs = plt.contourf(xx, yy, z, levels=np.linspace(-1.8, 1.8, 33), colors=mplc)
    plt.colorbar(cs)
    return fig


@pytest.mark.mpl_image_compare(tolerance={'3.3': 1.215}.get(MPL_VERSION, 0),
                               remove_text=True)
def test_gempak_color_xw_image_compare():
    """Test creating a plot with all the GEMPAK colors using xw style."""
    c = range(32)
    mplc = convert_gempak_color(c, style='xw')

    delta = 0.025
    x = y = np.arange(-3.0, 3.01, delta)
    xx, yy = np.meshgrid(x, y)
    z1 = np.exp(-xx**2 - yy**2)
    z2 = np.exp(-(xx - 1)**2 - (yy - 1)**2)
    z = (z1 - z2) * 2

    fig = plt.figure(figsize=(9, 9))
    cs = plt.contourf(xx, yy, z, levels=np.linspace(-1.8, 1.8, 33), colors=mplc)
    plt.colorbar(cs)
    return fig


def test_gempak_color_invalid_style():
    """Test converting a GEMPAK color with an invalid style parameter."""
    c = range(32)
    with pytest.raises(ValueError):
        convert_gempak_color(c, style='plt')


def test_gempak_color_quirks():
    """Test converting some unusual GEMPAK colors."""
    c = [-5, 95, 101]
    mplc = convert_gempak_color(c)
    truth = ['white', 'bisque', 'white']
    assert mplc == truth


def test_gempak_color_scalar():
    """Test converting a single GEMPAK color."""
    mplc = convert_gempak_color(6)
    truth = 'cyan'
    assert mplc == truth
