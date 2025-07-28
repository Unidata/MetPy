# Copyright (c) 2025 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark the functions in the moist thermo section of metpy's calc module.

Uses Airspeed Velocity for benchmarking and uses artificial dataset to ensure consistent and
reliable data for results.

"""

import os

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

    def time_apparent_temperature(self, pressureslice):
        """Benchmarking calculating apparent temperature on a 2d grid."""
        mpcalc.apparent_temperature(self.pressureslice.temperature,
                                    self.pressureslice.relative_humidity,
                                    self.pressureslice.windspeed)

    def time_heat_index(self, timeslice):
        """Benchmarking calculating heat index on a 3d cube."""
        mpcalc.heat_index(self.timeslice.temperature, self.timeslice.relative_humidity)

    def time_windchill(self, timeslice):
        """Benchmarking calculating windchill on a 3d cube."""
        mpcalc.windchill(self.timeslice.temperature, self.timeslice.windspeed)
