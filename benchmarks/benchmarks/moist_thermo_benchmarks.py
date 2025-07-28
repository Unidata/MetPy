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
        self.pressureslice = ds.isel(pressure=0, time=0)
        self.timeslice = ds.isel(time=0)
        self.upperslice = ds.isel(pressure=49, time=0)
        self.profileslice = ds.isel(time=0, lat=25, lon=25)

    def time_virtual_temperature(self, timeslice):
        """Benchmark virtual temperature on a 3d cube."""
        mpcalc.virtual_temperature(self.timeslice.temperature, self.timeslice.mixing_ratio)

    def time_dewpoint(self, timeslice):
        """Benchmarking dewpoint from vapor pressure on a 3d cube."""
        mpcalc.dewpoint(self.timeslice.vapor_pressure)

    def time_rh_from_mixing_ratio(self, timeslice):
        """Benchmarking relative humidity from mixing ratio on a 3d cube."""
        mpcalc.relative_humidity_from_mixing_ratio(self.timeslice.pressure,
                                                   self.timeslice.temperature,
                                                   self.timeslice.mixing_ratio)

    def time_dewpoint_from_rh(self, timeslice):
        """Benchmarking dewpoint from calculated on a 3d cube."""
        mpcalc.dewpoint_from_relative_humidity(self.timeslice.temperature,
                                               self.timeslice.relative_humidity)

    def time_precipitable_water(self, timeslice):
        """Benchmarking precipitable water calculation for one column."""
        mpcalc.precipitable_water(self.timeslice.pressure, self.timeslice.dewpoint[0][0])

    def time_wet_bulb_temperature(self, pressureslice):
        """Benchmarking wet bulb temperature calculation on on a slice."""
        mpcalc.wet_bulb_temperature(self.pressureslice.pressure,
                                    self.pressureslice.temperature,
                                    self.pressureslice.dewpoint)

    def time_scale_height(self, pressureslice):
        """Benchmarking the calculation for the scale height of a layer for 2 surfaces."""
        mpcalc.scale_height(self.upperslice.temperature, self.pressureslice.temperature)

    def time_moist_lapse(self, profileslice):
        """Benchmarking the calculation for the moist lapse rate for one profile."""
        mpcalc.moist_lapse(self.profileslice.pressure.values * units('hPa'),
                           self.profileslice.temperature[0].values * units('K'))

    def time_saturation_vapor_pressure(self, timeslice):
        """Benchmarking the saturation vapor pressure calculation for a 3d cube."""
        mpcalc.saturation_vapor_pressure(self.timeslice.temperature)

    def time_water_latent_heat_vaporization(self, timeslice):
        """Benchmarking the vaporization latent heat calculation on a 3d cube."""
        mpcalc.water_latent_heat_vaporization(self.timeslice.temperature)

    def time_water_latent_heat_sublimation(self, timeslice):
        """Benchmarking the sublimation latent heat calculation on a 3d cube."""
        mpcalc.water_latent_heat_sublimation(self.timeslice.temperature)

    def time_water_latent_heat_melting(self, timeslice):
        """Benchmarking the melting latent heat calculation on a 3d cube."""
        mpcalc.water_latent_heat_melting(self.timeslice.temperature)

    def time_specific_humidity_from_dewpoint(self, timeslice):
        """Benchmarking specific humidity from dewpoint calculation on a 3d cube."""
        mpcalc.specific_humidity_from_dewpoint(self.timeslice.pressure,
                                               self.timeslice.temperature)

    def time_relative_humidity_from_dewpoint(self, timeslice):
        """Benchmarking relative humidity from dewpoint calculation on a 3d cube."""
        mpcalc.relative_humidity_from_dewpoint(self.timeslice.temperature,
                                               self.timeslice.dewpoint)

    def time_moist_static_energy(self, timeslice):
        """Benchmarking moist static energy calculation on a 3d cube."""
        mpcalc.moist_static_energy(self.timeslice.height, self.timeslice.temperature,
                                   self.timeslice.specific_humidity)

    def time_dewpoint_from_specific_humidity(self, timeslice):
        """Benchmarking dewpoint from specific humidity calculation on a 3d cube."""
        mpcalc.dewpoint_from_specific_humidity(self.timeslice.pressure,
                                               self.timeslice.temperature,
                                               self.timeslice.specific_humidity)

    def time_moist_air_specific_heat_pressure(self, timeslice):
        """Benchmarking moist air specific heat pressure calculation on a 3d cube."""
        mpcalc.moist_air_specific_heat_pressure(self.timeslice.specific_humidity)

    def time_moist_air_poisson_exponent(self, timeslice):
        """Benchmarking moist air poisson exponent calculation on a cube."""
        mpcalc.moist_air_poisson_exponent(self.timeslice.specific_humidity)

    def time_relative_humidity_wet_psychrometric(self, timeslice):
        """Benchmarking the relative humidity from psychometric calculation on a cube."""
        mpcalc.relative_humidity_wet_psychrometric(self.timeslice.pressure,
                                                   self.timeslice.temperature,
                                                   self.timeslice.wet_bulb_temperature)

    def time_thickness_hydrostatic_from_relative_humidity(self, profileslice):
        """Benchmarking thickness  calculation from relative humidity on one profile."""
        mpcalc.thickness_hydrostatic_from_relative_humidity(self.profileslice.pressure,
                                                            self.profileslice.temperature,
                                                            self.profileslice.relative_humidity
                                                            )

    def time_relative_humidity_from_specific_humidity(self, timeslice):
        """Benchmarking relative humidity from specific humidity calculation on a 3d cube."""
        mpcalc.relative_humidity_from_specific_humidity(self.timeslice.pressure,
                                                        self.timeslice.temperature,
                                                        self.timeslice.specific_humidity)

    def time_wet_bulb_potential_temperature(self, timeslice):
        """Benchmarking the wet bulb potential temperature calculation on a 3d cube."""
        mpcalc.wet_bulb_potential_temperature(self.timeslice.pressure,
                                              self.timeslice.temperature,
                                              self.timeslice.dewpoint)

    def time_vertical_velocity_pressure(self, timeslice):
        """Benchmarking vertical velocity wrt pressure calculation on a 3d cube."""
        mpcalc.vertical_velocity_pressure(self.timeslice.wwind, self.timeslice.pressure,
                                          self.timeslice.temperature,
                                          self.timeslice.mixing_ratio)

    def time_vertical_velocity(self, timeslice):
        """Benchmarking vertical velocity calculation on a 3d cube."""
        mpcalc.vertical_velocity(self.timeslice.omega, self.timeslice.pressure,
                                 self.timeslice.temperature,
                                 self.timeslice.mixing_ratio)

    def time_saturation_equivalent_potential_temperature(self, timeslice):
        """Benchmarking saturation equivalent potential temperature on 3d cube."""
        mpcalc.saturation_equivalent_potential_temperature(self.timeslice.pressure,
                                                           self.timeslice.temperature)

    def time_virtual_potential_temperature(self, timeslice):
        """Benchmarking virtual potential temperature calculation on a 3d cube."""
        mpcalc.virtual_potential_temperature(self.timeslice.pressure,
                                             self.timeslice.temperature,
                                             self.timeslice.mixing_ratio)

    def time_psychrometric_vapor_pressure_wet(self, timeslice):
        """Benchmarking psychrometric vapor pressure calculation on a 3d cube."""
        mpcalc.psychrometric_vapor_pressure_wet(self.timeslice.pressure,
                                                self.timeslice.temperature,
                                                self.timeslice.wet_bulb_temperature)

    def time_mixing_ratio_from_relative_humidity(self, timeslice):
        """Benchmarking mixing ratio from relative humidity calculation on a 3d cube."""
        mpcalc.mixing_ratio_from_relative_humidity(self.timeslice.pressure,
                                                   self.timeslice.temperature,
                                                   self.timeslice.relative_humidity)

    def time_mixing_ratio_from_specific_humidity(self, timeslice):
        """Benchmarking calculating mixing rato from specific humidity on a 3d cube."""
        mpcalc.mixing_ratio_from_specific_humidity(self.timeslice.specific_humidity)

    def time_relative_humidity_from_mixing_ratio(self, timeslice):
        """Benchmarking relative humidity from mixing ratio calculation on a 3d cube."""
        mpcalc.relative_humidity_from_mixing_ratio(self.timeslice.pressure,
                                                   self.timeslice.temperature,
                                                   self.timeslice.mixing_ratio)

    def time_equivalent_potential_temperature(self, timeslice):
        """Benchmarking equivalent potential temperature calculation on 3d cube."""
        mpcalc.equivalent_potential_temperature(self.timeslice.pressure,
                                                self.timeslice.temperature,
                                                self.timeslice.dewpoint)

    def time_virtual_temperature_from_dewpoint(self, timeslice):
        """Benchmarking virtual temperature from dewpoint calculation on 3d cube."""
        mpcalc.virtual_temperature_from_dewpoint(self.timeslice.pressure,
                                                 self.timeslice.temperature,
                                                 self.timeslice.dewpoint)
