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
        self.profileslice = ds.isel(time=0, lat=0, lon=0)

    def time_density(self, pressureslice):
        """Benchmarking density calculation on a 2d surface."""
        mpcalc.density(self.pressureslice.pressure, self.pressureslice.temperature,
                       self.pressureslice.mixing_ratio)

    def time_height_to_geopotential(self, timeslice):
        """Benchmarking the height to geopotenial calculation on a 3d cube."""
        mpcalc.height_to_geopotential(self.timeslice.height)

    def time_potential_temperature(self, timeslice):
        """Benchmarking the potential temperature calculation on a 3d cube."""
        mpcalc.potential_temperature(self.timeslice.pressure, self.timeslice.temperature)

    def time_static_stability(self, timeslice):
        """Benchmarking static stability calculation on a 3d cube."""
        mpcalc.static_stability(self.timeslice.pressure, self.timeslice.temperature)

    def time_thickness_hydrostatic(self, timeslice):
        """Benchmarking hydrostatic thickness calculation on a 3d cube."""
        mpcalc.thickness_hydrostatic(self.timeslice.pressure, self.timeslice.temperature,
                                     self.timeslice.mixing_ratio)

    def time_dry_lapse(self, timeslice):
        """Benchmarking the dry lapse calculation on a 3d cube."""
        mpcalc.dry_lapse(self.timeslice.pressure, self.timeslice.temperature)

    def time_sigma_to_pressure(self, timeslice):
        """Benchmarking the sigma to pressure calculation on a 3d cube."""
        mpcalc.sigma_to_pressure(self.timeslice.sigma, self.timeslice.pressure[0],
                                 self.timeslice.pressure[49])

    def time_geopotential_to_height(self, timeslice):
        """Benchmarking the geopotential to height calculation on a 3d cube."""
        mpcalc.geopotential_to_height(self.timeslice.geopotential)

    def time_add_pressure_to_height(self, timeslice):
        """Benchmarking adding pressure to height on a 3d cube."""
        mpcalc.add_pressure_to_height(self.timeslice.height, self.timeslice.pressure)

    def time_add_height_to_pressure(self, timeslice):
        """Benchmarking adding height to pressure on a 3d cube."""
        mpcalc.add_height_to_pressure(self.timeslice.pressure.values * units('hPa'),
                                      self.timeslice.height.values * units('km'))

    def time_temperature_from_potential_temperature(self, timeslice):
        """Benchmarking calculating temperature from potential temperature on a 3d cube."""
        mpcalc.temperature_from_potential_temperature(self.timeslice.pressure,
                                                      self.timeslice.theta)

    def time_mean_pressure_weighted(self, profileslice):
        """Benchmarking calculating weighted mean of pressure with temp on one profile."""
        mpcalc.mean_pressure_weighted(self.profileslice.pressure,
                                      self.profileslice.temperature)

    def time_weighted_continuous_average(self, profileslice):
        """Bencharmking calculating weighted continuous average on one profile."""
        mpcalc.weighted_continuous_average(self.profileslice.pressure,
                                           self.profileslice.temperature)

    def time_dry_static_energy(self, timeslice):
        """Benchmarking dry static energy calculation on a 3d cube."""
        mpcalc.dry_static_energy(self.timeslice.height, self.timeslice.temperature)
