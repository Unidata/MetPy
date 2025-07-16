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
        self.timeslice = ds.isel(time=0)

    def time_brunt_vaisala_frequency(self, timeslice):
        """Benchmark Brunt Vaisala frequency calculation on a cube."""
        mpcalc.brunt_vaisala_frequency(self.timeslice.height, self.timeslice.theta)

    def time_gradient_richardson_number(self, timeslice):
        """Benchmark Gradient Richardson Number on a cube."""
        mpcalc.gradient_richardson_number(self.timeslice.height, self.timeslice.theta,
                                          self.timeslice.uwind, self.timeslice.vwind)

    def time_tke(self, ds):
        """Benchmarking turbulent kinetic energy calculation on a cube."""
        mpcalc.tke(ds.uwind.values * units('m/s'), ds.vwind.values * units('m/s'),
                   ds.wwind.values * units('m/s'))

    def time_brunt_vaisala_period(self, timeslice):
        """Benchmark Brunt Vaisala frequency calculation on a cube."""
        mpcalc.brunt_vaisala_period(self.timeslice.height, self.timeslice.theta)
