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
        self.profileslice = ds.isel(time=0, lat=0, lon=0)
        start = (30., 260.)
        end = (40., 270.)
        self.cross = mpinter.cross_section(self.timeslice, start, end).set_coords(('lat', 'lon'
                                                                                   ))

    def time_absolute_vorticity(self, pressureslice):
        """Benchmarking absolute momentum calculation on a 2d surface."""
        mpcalc.absolute_vorticity(self.pressureslice.uwind, self.pressureslice.vwind)

    def time_advection(self, timeslice):
        """Benchmarking the advection calculation of t on a 3d cube."""
        mpcalc.advection(self.timeslice.temperature, self.timeslice.uwind,
                         self.timeslice.vwind)

    def time_ageostrophic_wind(self, pressureslice):
        """Benchmarking ageostrophic wind calculation on a 2d surface."""
        mpcalc.ageostrophic_wind(self.pressureslice.height, self.pressureslice.uwind,
                                 self.pressureslice.vwind)

    def time_frontogenesis(self, pressureslice):
        """Benchmarking the calculation of frontogenesis of a 2d field."""
        mpcalc.frontogenesis(self.pressureslice.theta, self.pressureslice.uwind,
                             self.pressureslice.vwind)

    def time_potential_vorticity_barotropic(self, timeslice):
        """Benchmarking the barotropic potential vorticity calculation on a cube."""
        mpcalc.potential_vorticity_barotropic(self.timeslice.height, self.timeslice.uwind,
                                              self.timeslice.vwind)

    def time_q_vector(self, pressureslice):
        """Benchmarking q vector calculation on a 2d slice."""
        mpcalc.q_vector(self.pressureslice.uwind, self.pressureslice.vwind,
                        self.pressureslice.temperature, self.pressureslice.pressure)

    def time_total_deformation(self, pressureslice):
        """Benchmarking total deformation calculation on a 2d slice."""
        mpcalc.total_deformation(self.pressureslice.uwind, self.pressureslice.vwind)

    def time_vorticity(self, pressureslice):
        """Benchmarking vorticity calculation on a 2d slice."""
        mpcalc.vorticity(self.pressureslice.uwind, self.pressureslice.vwind)

    def time_shear_vorticity(self, pressureslice):
        """Benchmarking shear vorticity on a 2d slice."""
        mpcalc.shear_vorticity(self.pressureslice.uwind, self.pressureslice.vwind)

    def time_absolute_momentum(self, cross):
        """Benchmarking absolute momentum calculation."""
        mpcalc.absolute_momentum(self.cross.uwind, self.cross.vwind)

    def time_potential_vorticity_baroclinic(self, timeslice):
        """Benchmarking potential vorticity baroclinic on a 3d cube."""
        mpcalc.potential_vorticity_baroclinic(self.timeslice.theta, self.timeslice.pressure,
                                              self.timeslice.uwind, self.timeslice.vwind)

    def time_inertal_advective_wind(self, timeslice):
        """Benchmarking inertal advective wind calculation on a 3d cube."""
        mpcalc.inertial_advective_wind(self.timeslice.uwind, self.timeslice.vwind,
                                       self.timeslice.uwind, self.timeslice.vwind)

    def time_curvature_vorticity(self, timeslice):
        """Benchmarking the curvature vorticity calculation on a 3d cube."""
        mpcalc.curvature_vorticity(self.timeslice.uwind, self.timeslice.vwind)

    def time_montgomery_streamfunction(self, pressureslice):
        """Benchmarking the montgomery streamfunction calculation on a 2d grid."""
        mpcalc.montgomery_streamfunction(self.pressureslice.height,
                                         self.pressureslice.temperature)

    def time_wind_direction(self, timeslice):
        """Benchmarking the wind direction calculation on a 3d cube."""
        mpcalc.wind_direction(self.timeslice.uwind, self.timeslice.vwind)

    def time_wind_components(self, timeslice):
        """Benchmarking the wind components calculation on a 3d cube."""
        mpcalc.wind_components(self.timeslice.windspeed, self.timeslice.winddir)

    def time_divergence(self, timeslice):
        """Benchmarking divergence on a 3d cube."""
        mpcalc.divergence(self.timeslice.uwind, self.timeslice.vwind)

    def time_stretching_deformation(self, timeslice):
        """Benchmarking stretching deformation on a 3d cube."""
        mpcalc.stretching_deformation(self.timeslice.uwind, self.timeslice.vwind)

    def time_shearing_deformation(self, timeslice):
        """Benchmarking shearing deformation on a 3d cube."""
        mpcalc.shearing_deformation(self.timeslice.uwind, self.timeslice.vwind)

    def time_geostrophic_wind(self, timeslice):
        """Benchmarking the geostrophic wind calculation on a 3d cube."""
        mpcalc.geostrophic_wind(self.timeslice.height, latitude=self.timeslice.lat)

    def time_coriolis_parameter(self, timeslice):
        """Benchmarking coriolis parameter calculation on a 3d cube."""
        mpcalc.coriolis_parameter(self.timeslice.lat)

    def time_wind_speed(self, timeslice):
        """Benchmarking wind speed calculation on a 3d cube."""
        mpcalc.wind_speed(self.timeslice.uwind, self.timeslice.vwind)

    def time_exner_function(self, timeslice):
        """Benchmark exner function calculation on a cube."""
        mpcalc.exner_function(self.timeslice.pressure)
