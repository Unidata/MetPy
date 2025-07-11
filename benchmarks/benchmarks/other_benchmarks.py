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
        self.ds = ds
        self.pressureslice = ds.isel(pressure=0, time=0)
        self.timeslice = ds.isel(time=0)
        self.lineslice = ds.isel(pressure=0, time=0, lat=0)
        self.profileslice = ds.isel(time=0, lat=0, lon=0)

    def time_find_intersections(self, lineslice):
        """Benchmarking finding intersections calculation."""
        mpcalc.find_intersections(self.lineslice.lon, self.lineslice.temperature,
                                  self.lineslice.dewpoint)

    def time_find_peaks(self, pressureslice):
        """Benchmarking finding peaks of 2d dewpoint slice."""
        mpcalc.find_peaks(self.pressureslice.dewpoint)

    def time_get_perturbation(self, ds):
        """Benchmarking getting the perturbation of a time series."""
        mpcalc.get_perturbation(self.ds.temperature)

    def time_peak_persistence(self, pressureslice):
        """Benchmarking calculating persistence of of maxima point in 3d."""
        mpcalc.peak_persistence(self.pressureslice.dewpoint)

    def time_isentropic_interpolation_as_dataset(self, timeslice):
        """Benchmarking the isentropic interpolation as dataset calculation on a 3d cube."""
        mpcalc.isentropic_interpolation_as_dataset([265.] * units.kelvin,
                                                   self.timeslice.temperature)

    def time_isentropic_interpolation(self, timeslice):
        """Bencharking the isentropic interpolation calculation on a 3d cube."""
        mpcalc.isentropic_interpolation([265.] * units.kelvin, self.timeslice.pressure,
                                        self.timeslice.temperature)
