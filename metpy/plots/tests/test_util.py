# Copyright (c) 2017,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `_util` module."""

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import pytest

from metpy.plots import add_metpy_logo, add_timestamp, add_unidata_logo
# Fixture to make sure we have the right backend
from metpy.testing import set_agg_backend  # noqa: F401, I202

MPL_VERSION = matplotlib.__version__[:3]


@pytest.mark.mpl_image_compare(tolerance={'1.4': 5.58}.get(MPL_VERSION, 0.01),
                               remove_text=True)
def test_add_timestamp():
    """Test adding a timestamp to an axes object."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    add_timestamp(ax, time=datetime(2017, 1, 1))
    return fig


@pytest.mark.mpl_image_compare(tolerance={'1.4': 6.03}.get(MPL_VERSION, 0.01),
                               remove_text=True)
def test_add_timestamp_custom_format():
    """Test adding a timestamp to an axes object with custom time formatting."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    add_timestamp(ax, time=datetime(2017, 1, 1), time_format='%H:%M:%S %Y/%m/%d')
    return fig


@pytest.mark.mpl_image_compare(tolerance={'1.4': 5.58}.get(MPL_VERSION, 0.01),
                               remove_text=True)
def test_add_timestamp_pretext():
    """Test adding a timestamp to an axes object with custom pre-text."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    add_timestamp(ax, time=datetime(2017, 1, 1), pretext='Valid: ')
    return fig


@pytest.mark.mpl_image_compare(tolerance={'1.4': 9.51}.get(MPL_VERSION, 0.01),
                               remove_text=True)
def test_add_timestamp_high_contrast():
    """Test adding a timestamp to an axes object."""
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    add_timestamp(ax, time=datetime(2017, 1, 1), high_contrast=True)
    return fig


@pytest.mark.mpl_image_compare(tolerance={'1.4': 0.004}.get(MPL_VERSION, 0.01),
                               remove_text=True)
def test_add_metpy_logo_small():
    """Test adding a MetPy logo to a figure."""
    fig = plt.figure(figsize=(9, 9))
    add_metpy_logo(fig)
    return fig


@pytest.mark.mpl_image_compare(tolerance={'1.4': 0.004}.get(MPL_VERSION, 0.01),
                               remove_text=True)
def test_add_metpy_logo_large():
    """Test adding a large MetPy logo to a figure."""
    fig = plt.figure(figsize=(9, 9))
    add_metpy_logo(fig, size='large')
    return fig


@pytest.mark.mpl_image_compare(tolerance={'1.4': 0.004}.get(MPL_VERSION, 0.01),
                               remove_text=True)
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
