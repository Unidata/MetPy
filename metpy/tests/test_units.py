# Copyright (c) 2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Tests the operation of MetPy's unit support code."""

import matplotlib.pyplot as plt
import pytest

from metpy.testing import set_agg_backend  # noqa: F401
from metpy.units import units


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_axhline():
    r"""Ensure that passing a quantity to axhline does not error."""
    fig, ax = plt.subplots()
    ax.axhline(930 * units('mbar'))
    ax.set_ylim(900, 950)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_axvline():
    r"""Ensure that passing a quantity to axvline does not error."""
    fig, ax = plt.subplots()
    ax.axvline(0 * units('degC'))
    ax.set_xlim(-1, 1)
    return fig
