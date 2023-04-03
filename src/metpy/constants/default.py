# Copyright (c) 2008,2015,2016,2018,2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Constant and thermophysical property values expressed as quantities."""

from ..package_tools import Exporter
from ..units import units

exporter = Exporter(globals())

# Export all the variables defined in this block
with exporter:
    # Earth
    earth_gravity = g = units.Quantity(9.80665, 'm / s^2')
    Re = earth_avg_radius = units.Quantity(6371008.7714, 'm')
    G = gravitational_constant = units.Quantity(6.67430e-11, 'm^3 / kg / s^2')
    GM = geocentric_gravitational_constant = units.Quantity(3986005e8, 'm^3 / s^2')
    omega = earth_avg_angular_vel = units.Quantity(7292115e-11, 'rad / s')
    d = earth_sfc_avg_dist_sun = units.Quantity(149597870700., 'm')
    S = earth_solar_irradiance = units.Quantity(1360.8, 'W / m^2')
    delta = earth_max_declination = units.Quantity(23.45, 'degrees')
    earth_orbit_eccentricity = units.Quantity(0.0167, 'dimensionless')
    earth_mass = me = geocentric_gravitational_constant / gravitational_constant

    # molar gas constant
    R = units.Quantity(8.314462618, 'J / mol / K')

    # Water
    Mw = water_molecular_weight = units.Quantity(18.015268, 'g / mol')
    Rv = water_gas_constant = (R / Mw).to('J / kg / K')
    rho_l = density_water = units.Quantity(999.97495, 'kg / m^3')
    wv_specific_heat_ratio = units.Quantity(1.330, 'dimensionless')
    Cp_v = wv_specific_heat_press = (
        wv_specific_heat_ratio * Rv / (wv_specific_heat_ratio - 1)
    )
    Cv_v = wv_specific_heat_vol = Cp_v / wv_specific_heat_ratio
    Cp_l = water_specific_heat = units.Quantity(4.2194, 'kJ / kg / K').to('J / kg / K')
    Lv = water_heat_vaporization = units.Quantity(2.50084e6, 'J / kg')
    Lf = water_heat_fusion = units.Quantity(3.337e5, 'J / kg')
    Cp_i = ice_specific_heat = units.Quantity(2090, 'J / kg / K')
    rho_i = density_ice = units.Quantity(917, 'kg / m^3')
    sat_pressure_0c = units.Quantity(6.112, 'millibar')

    # Dry air
    Md = dry_air_molecular_weight = units.Quantity(28.96546e-3, 'kg / mol')
    Rd = dry_air_gas_constant = R / Md
    dry_air_spec_heat_ratio = units.Quantity(1.4, 'dimensionless')
    Cp_d = dry_air_spec_heat_press = (
        dry_air_spec_heat_ratio * Rd / (dry_air_spec_heat_ratio - 1)
    )
    Cv_d = dry_air_spec_heat_vol = Cp_d / dry_air_spec_heat_ratio
    rho_d = dry_air_density_stp = (
        units.Quantity(1000., 'mbar') / (Rd * units.Quantity(273.15, 'K'))
    ).to('kg / m^3')

    # General meteorology constants
    P0 = pot_temp_ref_press = units.Quantity(1000., 'mbar')
    kappa = poisson_exponent = (Rd / Cp_d).to('dimensionless')
    gamma_d = dry_adiabatic_lapse_rate = g / Cp_d
    epsilon = molecular_weight_ratio = (Mw / Md).to('dimensionless')

del Exporter
