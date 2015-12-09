# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_agg import FigureCanvasAgg
from metpy.plots.skewt import *  # noqa
from metpy.units import units


# TODO: Need at some point to do image-based comparison, but that's a lot to
# bite off right now
class TestSkewT(object):
    'Test SkewT'
    @staticmethod
    def test_api():
        'Test the SkewT api'
        fig = Figure(figsize=(9, 9))
        skew = SkewT(fig)

        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        p = np.linspace(1000, 100, 10)
        t = np.linspace(20, -20, 10)
        u = np.linspace(-10, 10, 10)
        skew.plot(p, t, 'r')
        skew.plot_barbs(p, u, u)

        # Add the relevant special lines
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()

        with tempfile.NamedTemporaryFile() as f:
            FigureCanvasAgg(fig).print_png(f.name)

    @staticmethod
    def test_subplot():
        'Test using SkewT on a sub-plot'
        fig = Figure(figsize=(9, 9))
        SkewT(fig, subplot=(2, 2, 1))
        with tempfile.NamedTemporaryFile() as f:
            FigureCanvasAgg(fig).print_png(f.name)

    @staticmethod
    def test_gridspec():
        'Test using SkewT on a sub-plot'
        fig = Figure(figsize=(9, 9))
        gs = GridSpec(1, 2)
        SkewT(fig, subplot=gs[0, 1])
        with tempfile.NamedTemporaryFile() as f:
            FigureCanvasAgg(fig).print_png(f.name)


class TestHodograph(object):
    'Test Hodograph'
    @staticmethod
    def test_basic_api():
        'Basic test of Hodograph API'
        fig = Figure(figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1)
        hodo = Hodograph(ax, component_range=60)
        hodo.add_grid(increment=5, color='k')
        hodo.plot([1, 10], [1, 10], color='red')
        hodo.plot_colormapped([1, 3, 5, 10], [2, 4, 6, 11], [0.1, 0.3, 0.5, 0.9], cmap='Greys')

    @staticmethod
    def test_units():
        'Test passing unit-ed quantities to Hodograph'
        fig = Figure(figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1)
        hodo = Hodograph(ax)
        u = np.arange(10) * units.kt
        v = np.arange(10) * units.kt
        hodo.plot(u, v)
        hodo.plot_colormapped(u, v, np.sqrt(u * u + v * v), cmap='Greys')
