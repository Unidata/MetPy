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
from metpy.units import units


class TimeSuite:
    """Benchmark moist thermo functions in time using Airspeed Velocity and xarray datasets.

    Uses ASV's benchmarking format to load in data and run benchmarks to measure time
     performance

    """

    # NOTE: I'm using CalVer https://calver.org/ YYYY.MM.DD
    version = '2025.07.07'

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

    def time_height_to_pressure_std(self, timeslice):
        """Benchmarking the height to pressure calculation in a std atm on a 3d cube."""
        mpcalc.height_to_pressure_std(self.timeslice.height)

    def time_pressure_to_height_std(self, timeslice):
        """Benchmarking the pressure to height calculation in a std atm on a 3d cube."""
        mpcalc.pressure_to_height_std(self.timeslice.pressure)

    def time_altimeter_to_sea_level_pressure(self, timeslice):
        """Benchmarking altimeter to slp on a 3d cube."""
        mpcalc.altimeter_to_sea_level_pressure(self.timeslice.pressure.values * units('hPa'),
                                               self.timeslice.height.values * units('km'),
                                               self.timeslice.temperature * units('K'))
