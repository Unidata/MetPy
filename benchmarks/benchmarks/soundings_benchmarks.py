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
    version = '2025.06.16'

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
        self.pressureslice = ds.isel(time=0, pressure=0)
        self.profileslice = ds.isel(lat=25, lon=25, time=0)
        self.parcelprofile = mpcalc.parcel_profile(self.profileslice.pressure,
                                                   self.profileslice.temperature[0],
                                                   self.profileslice.dewpoint[0])
        self.sbcape, _ = mpcalc.surface_based_cape_cin(self.profileslice.pressure,
                                                       self.profileslice.temperature,
                                                       self.profileslice.dewpoint)
        self.sblcl, _ = mpcalc.lcl(self.profileslice.pressure,
                                   self.profileslice.temperature,
                                   self.profileslice.dewpoint)
        self.sblclheight = mpcalc.pressure_to_height_std(self.sblcl)
        _, _, self.relhel = mpcalc.storm_relative_helicity(self.profileslice.height,
                                                           self.profileslice.uwind,
                                                           self.profileslice.vwind,
                                                           1 * units('km'))
        self.shearu, self.shearv = mpcalc.bulk_shear(self.profileslice.pressure,
                                                     self.profileslice.uwind,
                                                     self.profileslice.vwind)
        self.shear = mpcalc.wind_speed(self.shearu, self.shearv)

    def time_bulk_shear(self, profileslice):
        """Benchmarking calculating the bulk shear of a profile."""
        mpcalc.bulk_shear(self.profileslice.pressure, self.profileslice.uwind,
                          self.profileslice.vwind)

    def time_ccl(self, profileslice):
        """Benchmarking calculating the convective condensation level of a profile."""
        mpcalc.ccl(self.profileslice.pressure, self.profileslice.temperature,
                   self.profileslice.dewpoint)

    def time_parcel_profile(self, profileslice):
        """Benchmarking the atmospheric parcel profile for one profile."""
        mpcalc.parcel_profile(self.profileslice.pressure, self.profileslice.temperature[0],
                              self.profileslice.dewpoint[0])

    def time_most_unstable_parcel(self, profileslice):
        """Benchmarking the calculation to find the most unstable parcel for one profile."""
        mpcalc.most_unstable_parcel(self.profileslice.pressure, self.profileslice.temperature,
                                    self.profileslice.dewpoint)

    def time_cape_cin(self, profileslice):
        """Benchmarking cape_cin calculation for one profile."""
        mpcalc.cape_cin(self.profileslice.pressure, self.profileslice.temperature,
                        self.profileslice.dewpoint, self.parcelprofile)

    def time_lcl(self, timeslice):
        """Benchmarks lcl on a 3d cube - many profiles."""
        mpcalc.lcl(self.pressureslice.pressure, self.pressureslice.temperature,
                   self.pressureslice.dewpoint)

    def time_el(self, profileslice):
        """Benchmarks el calculation on one profile."""
        mpcalc.el(self.profileslice.pressure, self.profileslice.temperature,
                  self.profileslice.dewpoint)

    def time_storm_relative_helicity(self, profileslice):
        """Benchmarks storm relative helicity over one profile."""
        mpcalc.storm_relative_helicity(self.profileslice.height, self.profileslice.uwind,
                                       self.profileslice.vwind, 1 * units('km'))

    def time_vertical_totals(self, timeslice):
        """Benchmarking vertical totals for many profiles."""
        mpcalc.vertical_totals(self.timeslice.pressure, self.timeslice.temperature)

    def time_supercell_composite(self, profileslice):
        """Benchmarks supercell composite calculation for one calculation."""
        mpcalc.supercell_composite(2500 * units('J/kg'), 125 * units('m^2/s^2'),
                                   50 * units.knot)

    def time_critical_angle(self, profileslice):
        """Benchmarking critical angle on one profile."""
        mpcalc.critical_angle(self.profileslice.pressure, self.profileslice.uwind,
                              self.profileslice.vwind, self.profileslice.height,
                              0 * units('m/s'), 0 * units('m/s'))

    def time_bunkers_storm_motion(self, profileslice):
        """Benchmarking bunkers storm motion on one profile."""
        mpcalc.bunkers_storm_motion(self.profileslice.pressure, self.profileslice.uwind,
                                    self.profileslice.vwind, self.profileslice.height)

    def time_corfidi_storm_motion(self, profileslice):
        """Benchmarking corfidi storm motion on one profile."""
        mpcalc.corfidi_storm_motion(self.profileslice.pressure, self.profileslice.uwind,
                                    self.profileslice.vwind)

    def time_sweat_index(self, timeslice):
        """Benchmarking SWEAT index on many profiles."""
        mpcalc.sweat_index(self.timeslice.pressure, self.timeslice.temperature,
                           self.timeslice.dewpoint, self.timeslice.windspeed,
                           self.timeslice.winddir)

    def time_most_unstable_cape_cin(self, profileslice):
        """Benchmarking most unstable cape cin calculation on one profile."""
        mpcalc.most_unstable_cape_cin(self.profileslice.pressure,
                                      self.profileslice.temperature,
                                      self.profileslice.dewpoint)

    def time_surface_based_cape_cin(self, profileslice):
        """Benchmarking surface based cape cin calculation on one profile."""
        mpcalc.surface_based_cape_cin(self.profileslice.pressure,
                                      self.profileslice.temperature,
                                      self.profileslice.dewpoint)

    def time_lifted_index(self, profileslice):
        """Benchmarking lifted index calculation on one profile."""
        mpcalc.lifted_index(self.profileslice.pressure, self.profileslice.temperature,
                            self.parcelprofile)

    def time_k_index(self, timeslice):
        """Benchmarking k index calculation on many profiles."""
        mpcalc.k_index(self.timeslice.pressure, self.timeslice.temperature,
                       self.timeslice.dewpoint)

    def time_mixed_layer_cape_cin(self, profileslice):
        """Benchmarking mixed layer cape cin calculation for one profile."""
        mpcalc.mixed_layer_cape_cin(self.profileslice.pressure, self.profileslice.temperature,
                                    self.profileslice.dewpoint)

    def time_cross_totals(self, timeslice):
        """Benchmarking cross totals calculation on many profiles."""
        mpcalc.cross_totals(self.timeslice.pressure, self.timeslice.temperature,
                            self.timeslice.dewpoint)

    def time_downdraft_cape(self, profileslice):
        """Benchmarking downdraft cape calculation on one profile."""
        mpcalc.downdraft_cape(self.profileslice.pressure, self.profileslice.temperature,
                              self.profileslice.dewpoint)

    def time_parcel_profile_with_lcl_as_dataset(self, profileslice):
        """Benchmarking parcel profile with lcl as dataset one on profile."""
        mpcalc.parcel_profile_with_lcl_as_dataset(self.profileslice.pressure,
                                                  self.profileslice.temperature,
                                                  self.profileslice.dewpoint)

    def time_showalter_index(self, profileslice):
        """Benchmarking calculating the showalter index on one profiles."""
        mpcalc.showalter_index(self.profileslice.pressure, self.profileslice.temperature,
                               self.profileslice.dewpoint)

    def time_galvez_davison_index(self, timeslice):
        """Benchmarking calculating the galvez davison index on many profiles."""
        mpcalc.galvez_davison_index(self.timeslice.pressure, self.timeslice.temperature,
                                    self.timeslice.mixing_ratio, self.timeslice.pressure[0])

    def time_significant_tornado(self, profileslice):
        """Benchmarking significant tornado param for one profile."""
        mpcalc.significant_tornado(self.sbcape, self.sblclheight, self.relhel, self.shear)

    def time_total_totals_index(self, timeslice):
        """Benchmarking total totals index for many profiles."""
        mpcalc.total_totals_index(self.timeslice.pressure, self.timeslice.temperature,
                                  self.timeslice.dewpoint)

    def time_lfc(self, profileslice):
        """Benchmarking level of free convection calculation for one profile."""
        mpcalc.lfc(self.profileslice.pressure, self.profileslice.temperature,
                   self.profileslice.dewpoint)

    def time_mixed_parcel(self, profileslice):
        """Benchmarking mixed parcel for one profile."""
        mpcalc.mixed_parcel(self.profileslice.pressure, self.profileslice.temperature,
                            self.profileslice.dewpoint)

    def time_mixed_layer(self, profileslice):
        """Benchmarking mixed layer of temperature for one profile."""
        mpcalc.mixed_layer(self.profileslice.pressure, self.profileslice.temperature)

    def time_parcel_profile_with_lcl(self, profileslice):
        """Benchmarking parcel profile with lcl calculation."""
        mpcalc.parcel_profile_with_lcl(self.profileslice.pressure,
                                       self.profileslice.temperature,
                                       self.profileslice.dewpoint)
