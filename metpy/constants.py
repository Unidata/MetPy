r'''
This is a collection of meteorologically significant constants.

Earth
-----
======================== ===== ======== =======================================
Name                     Abbr. Units    Description
------------------------ ----- -------- ---------------------------------------
earth_avg_radius         Re    m        Avg. radius of the Earth
earth_gravity            g     m s^-2   Avg. gravity acceleration on Earth
earth_avg_angular_vel    omega rad s^-1 Avg. angular velocity of Earth
earth_sfc_avg_dist_sun   d     m        Avg. distance of the Earth from the Sun
earth_solar_irradiance   S     W m^-2   Avg. solar irradiance of Earth
earth_max_declination    delta degrees  Max. solar declination angle of Earth
earth_orbit_eccentricity       None     Avg. eccentricity of Earth's orbit
======================== ===== ======== =======================================

Water
-----
======================= ===== =========== =====================================
Name                    Abbr. Units       Description
----------------------- ----- ----------- -------------------------------------
water_molecular_weight  Mw    g mol^-1    Molecular weight of water
water_gas_constant      Rv    J (K kg)^-1 Gas constant for water vapor
density_water           rho_l kg m^-3     Nominal density of liquid water at 0C
wv_specific_heat_press  Cp_v  J (K kg)^-1 Specific heat at constant pressure
                                          for water vapor
wv_specific_heat_vol    Cv_v  J (K kg)^-1 Specific heat at constant volume for
                                          water vapor
water_specific_heat     Cp_l  J (K kg)^-1 Specific heat of liquid water at 0C
water_heat_vaporization Lv    J kg^-1     Latent heat of vaporization for
                                          liquid water at 0C
water_heat_fusion       Lf    J kg^-1     Latent heat of fusion for liquid
                                          water at 0C
ice_specific_heat       Cp_i  J (K kg)^-1 Specific heat of ice at 0C
density_ice             rho_i kg m^-3     Density of ice at 0C
======================= ===== =========== =====================================

Dry Air
-------
======================== ===== =========== ====================================
Name                     Abbr. Units       Description
------------------------ ----- ----------- ------------------------------------
dry_air_molecular_weight Md    g / mol     Nominal molecular weight of dry air
                                           at the surface of th Earth
dry_air_gas_constant     Rd    J (K kg)^-1 Gas constant for dry air at the
                                           surface of the Earth
dry_air_spec_heat_press  Cp_d  J (K kg)^-1 Specific heat at constant pressure
                                           for dry air
dry_air_spec_heat_vol    Cv_d  J (K kg)^-1 Specific heat at constant volume
                                           for dry air
dry_air_density_stp      rho_d kg m^-3     Density of dry air at 0C and 1000mb
======================== ===== =========== ====================================

General Meteorology Constants
-----------------------------
======================== ======= ======== =====================================
Name                     Abbr.   Units    Description
------------------------ ------- -------- -------------------------------------
pot_temp_ref_press       P0      Pa       Reference pressure for potential
                                          temperature
poisson_exponent         kappa   None     Exponent in Poisson's equation
                                          (Rd/Cp_d)
dry_adiabatic_lapse_rate gamma_d K km^-1  The dry adiabatic lapse rate
molecular_weight_ratio   epsilon None     Ratio of molecular weight of water to
                                          that of dry air
======================== ======= ======== =====================================
'''

# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from .units import units
from .package_tools import Exporter

exporter = Exporter(globals())

# Export all the variables defined in this block
with exporter:
    # Earth
    earth_gravity = g = units.gravity.to_base_units()
    Re = earth_avg_radius = 6.37e6 * units.m
    omega = earth_avg_angular_vel = 2 * units.pi / units.day
    d = earth_sfc_avg_dist_sun = 1.496e11 * units.m
    S = earth_solar_irradiance = units.Quantity(1.368e3, 'W / m^2')
    delta = earth_max_declination = 23.45 * units.deg
    earth_orbit_eccentricity = 0.0167

    # Water
    Mw = water_molecular_weight = units.Quantity(18.016, 'g / mol')
    Rv = water_gas_constant = units.R.to_base_units() / Mw
    # Nominal density of liquid water at 0C
    rho_l = density_water = units.Quantity(1e3, 'kg / m^3')
    Cp_v = wv_specific_heat_press = units.Quantity(1952., 'm^2 / s^2 / K')
    Cv_v = wv_specific_heat_vol = units.Quantity(1463., 'm^2 / s^2 / K')
    Cp_l = water_specific_heat = units.Quantity(4218., 'm^2 / s^2 / K')  # at 0C
    Lv = water_heat_vaporization = units.Quantity(2.5e6, 'm^2 / s^2')  # at 0C
    Lf = water_heat_fusion = units.Quantity(3.34e5, 'm^2 / s^2')  # at 0C
    Cp_i = ice_specific_heat = units.Quantity(2106, 'm^2 / s^2 / K')  # at 0C
    rho_i = density_ice = units.Quantity(917, 'kg / m^3')  # at 0C

    # Dry air
    Md = dry_air_molecular_weight = units.Quantity(28.97, 'g / mol')  # at the sfc
    Rd = dry_air_gas_constant = units.R.to_base_units() / Md
    Cp_d = dry_air_spec_heat_press = units.Quantity(1004., 'm^2 / s^2 / K')
    Cv_d = dry_air_spec_heat_vol = units.Quantity(717., 'm^2 / s^2 / K')
    rho_d = dry_air_density_stp = units.Quantity(1.275, 'kg / m^3')  # at 0C 1000mb

    # General meteorology constants
    P0 = pot_temp_ref_press = 1000. * units.mbar
    kappa = poisson_exponent = Rd / Cp_d
    gamma_d = dry_adiabatic_lapse_rate = g / Cp_d
    epsilon = molecular_weight_ratio = Mw / Md
