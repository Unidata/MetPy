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
import metpy.interpolate as mpinter


class TimeSuite:
    """Benchmark moist thermo functions in time using Airspeed Velocity and xarray datasets.

    Uses ASV's benchmarking format to load in data and run benchmarks to measure time
     performance

    """

    # NOTE: I'm using CalVer https://calver.org/ YYYY.MM.DD
    version = '2025.07.03'

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
        ds = ds.metpy.parse_cf()
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
        start = (30., 260.)
        end = (40., 270.)
        self.cross = mpinter.cross_section(self.timeslice,
                                           start, end).set_coords(('lat', 'lon'))

    def time_geospatial_gradient(self, pressureslice):
        """Benchmarking calculating the geospatial gradient of temp on a 2d array."""
        mpcalc.geospatial_gradient(self.pressureslice.temperature)

    def time_geospatial_laplacian(self, pressureslice):
        """Benchmarking calculating the geospatial laplacian of temp on a 2d array."""
        mpcalc.geospatial_laplacian(self.pressureslice.temperature)

    def time_gradient(self, timeslice):
        """Benchmarking calculating the gradient of temp on a 3d cube."""
        mpcalc.gradient(self.timeslice.temperature)

    def time_vector_derivative(self, pressureslice):
        """Benchmarking calculating the vector derivative of wind on a 2d slice."""
        mpcalc.vector_derivative(self.pressureslice.uwind, self.pressureslice.vwind)

    def time_tangential_component(self, cross):
        """Benchmarking calculation of the tangential component of wind on a slice."""
        mpcalc.tangential_component(self.cross.uwind, self.cross.vwind)

    def time_cross_section_components(self, cross):
        """Benchmarking the cross section components of a wind grid."""
        mpcalc.cross_section_components(self.cross.uwind, self.cross.vwind)

    def time_normal_component(self, cross):
        """Benchmarking the calculating normal components times."""
        mpcalc.normal_component(self.cross.uwind, self.cross.vwind)
