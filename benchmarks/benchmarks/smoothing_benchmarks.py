# Copyright (c) 2025 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark the functions in the moist thermo section of metpy's calc module.

Uses Airspeed Velocity for benchmarking and uses artificial dataset to ensure consistent and
reliable data for results.

"""

import os

import numpy as np
import xarray as xr

import metpy.calc as mpcalc


class TimeSuite:
    """Benchmark moist thermo functions in time using Airspeed Velocity and xarray datasets.

    Uses ASV's benchmarking format to load in data and run benchmarks to measure time
     performance

    """

    # NOTE: I'm using CalVer https://calver.org/ YYYY.MM.DD
    version = '2025.07.02'

    def setup_cache(self):
        """Collect the sample dataset from the filepath and opens it as an xarray.

        Returns
        -------
        ds
            Dataset with artificial meteorology data for testing
        """
        base_path = os.path.dirname(__file__)  # path to current file
        file_path = os.path.join(base_path, '..', 'data_array_compressed.nc')
        file_path = os.path.abspath(file_path)
        ds = xr.open_dataset(file_path)
        return ds

    def setup(self, ds):
        """Set up the appropriate slices from the sample dataset for testing.

        Parameters
        ----------
        ds : dataset
            The dataset made in setup_cache which contains the testing data
        """
        self.pressureslice = ds.isel(pressure=0, time=0)
        self.timeslice = ds.isel(time=0)

    def time_smooth_gaussian(self, pressureslice):
        """Benchmarking the gaussian smoothing of a 2d grid."""
        mpcalc.smooth_gaussian(self.pressureslice.relative_humidity, 5)

    def time_smooth_window(self, pressureslice):
        """Benchmarking the window smoothing of a 2d grid."""
        mpcalc.smooth_window(self.pressureslice.relative_humidity, np.diag(np.ones(5)))

    def time_smooth_rectangular(self, pressureslice):
        """Benchmarking the rectangular smoothing of a 2d grid."""
        mpcalc.smooth_rectangular(self.pressureslice.relative_humidity, (3, 7))

    def time_smooth_circular(self, pressureslice):
        """Benchmarking the circular smoothing of a 2d grid."""
        mpcalc.smooth_circular(self.pressureslice.relative_humidity, 2)

    def time_smooth_n_point(self, pressureslice):
        """Benchmarking the 5 point smoothing of a 2d grid."""
        mpcalc.smooth_n_point(self.pressureslice.relative_humidity)

    def time_zoom_xarray(self, pressureslice):
        """Benchmarking the zoom xarray function."""
        mpcalc.zoom_xarray(self.pressureslice.temperature, zoom=3.0)
