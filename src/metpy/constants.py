# Copyright (c) 2008,2015,2016,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

r"""A collection of meteorologically significant constants.

Earth
-----
======================== =============== =========== ======================================= ========================================
Name                     Symbol          Short Name  Units                                   Description
------------------------ --------------- ----------- --------------------------------------- ----------------------------------------
earth_avg_radius         :math:`R_e`     Re          :math:`\text{m}`                        Avg. radius of the Earth
earth_gravity            :math:`g`       g           :math:`\text{m s}^{-2}`                 Avg. gravity acceleration on Earth
gravitational_constant   :math:`G`       G           :math:`\text{m}^{3} {kg}^{-1} {s}^{-2}` Gravitational constant
earth_avg_angular_vel    :math:`\Omega`  omega       :math:`\text{rad s}^{-1}`               Avg. angular velocity of Earth
earth_sfc_avg_dist_sun   :math:`d`       d           :math:`\text{m}`                        Avg. distance of the Earth from the Sun
earth_solar_irradiance   :math:`S`       S           :math:`\text{W m}^{-2}`                 Avg. solar irradiance of Earth
earth_max_declination    :math:`\delta`  delta       :math:`\text{degrees}`                  Max. solar declination angle of Earth
earth_orbit_eccentricity :math:`e`                   :math:`\text{None}`                     Avg. eccentricity of Earth's orbit
earth_mass               :math:`m_e`     me          :math:`\text{kg}`                       Total mass of the Earth (approx)
======================== =============== =========== ======================================= ========================================

Water
-----
======================= ================ ========== ============================ ====================================================
Name                    Symbol           Short Name Units                        Description
----------------------- ---------------- ---------- ---------------------------- ----------------------------------------------------
water_molecular_weight  :math:`M_w`      Mw         :math:`\text{g mol}^{-1}`    Molecular weight of water
water_gas_constant      :math:`R_v`      Rv         :math:`\text{J (K kg)}^{-1}` Gas constant for water vapor
density_water           :math:`\rho_l`   rho_l      :math:`\text{kg m}^{-3}`     Nominal density of liquid water at 0C
wv_specific_heat_press  :math:`C_{pv}`   Cp_v       :math:`\text{J (K kg)}^{-1}` Specific heat at constant pressure for water vapor
wv_specific_heat_vol    :math:`C_{vv}`   Cv_v       :math:`\text{J (K kg)}^{-1}` Specific heat at constant volume for water vapor
water_specific_heat     :math:`Cp_l`     Cp_l       :math:`\text{J (K kg)}^{-1}` Specific heat of liquid water at 0C
water_heat_vaporization :math:`L_v`      Lv         :math:`\text{J kg}^{-1}`     Latent heat of vaporization for liquid water at 0C
water_heat_fusion       :math:`L_f`      Lf         :math:`\text{J kg}^{-1}`     Latent heat of fusion for liquid water at 0C
ice_specific_heat       :math:`C_{pi}`   Cp_i       :math:`\text{J (K kg)}^{-1}` Specific heat of ice at 0C
density_ice             :math:`\rho_i`   rho_i      :math:`\text{kg m}^{-3}`     Density of ice at 0C
======================= ================ ========== ============================ ====================================================

Dry Air
-------
======================== ================ ============= ============================ ===============================================================
Name                     Symbol           Short Name    Units                        Description
------------------------ ---------------- ------------- ---------------------------- ---------------------------------------------------------------
dry_air_molecular_weight :math:`M_d`      Md            :math:`\text{g / mol}`       Nominal molecular weight of dry air at the surface of th Earth
dry_air_gas_constant     :math:`R_d`      Rd            :math:`\text{J (K kg)}^{-1}` Gas constant for dry air at the surface of the Earth
dry_air_spec_heat_press  :math:`C_{pd}`   Cp_d          :math:`\text{J (K kg)}^{-1}` Specific heat at constant pressure for dry air
dry_air_spec_heat_vol    :math:`C_{vd}`   Cv_d          :math:`\text{J (K kg)}^{-1}` Specific heat at constant volume for dry air
dry_air_density_stp      :math:`\rho_d`   rho_d         :math:`\text{kg m}^{-3}`     Density of dry air at 0C and 1000mb
======================== ================ ============= ============================ ===============================================================

General Meteorology Constants
-----------------------------
======================== ================= =========== ========================= =======================================================
Name                     Symbol            Short Name   Units                    Description
------------------------ ----------------- ----------- ------------------------- -------------------------------------------------------
pot_temp_ref_press       :math:`P_0`       P0          :math:`\text{Pa}`         Reference pressure for potential temperature
poisson_exponent         :math:`\kappa`    kappa       :math:`\text{None}`       Exponent in Poisson's equation (Rd/Cp_d)
dry_adiabatic_lapse_rate :math:`\gamma_d`  gamma_d     :math:`\text{K km}^{-1}`  The dry adiabatic lapse rate
molecular_weight_ratio   :math:`\epsilon`  epsilon     :math:`\text{None}`       Ratio of molecular weight of water to that of dry air
======================== ================= =========== ========================= =======================================================
"""  # noqa: E501

from .package_tools import Exporter
from .units import units

exporter = Exporter(globals())

# Export all the variables defined in this block
with exporter:
    # Earth
    earth_gravity = g = units.Quantity(1.0, units.gravity).to('m / s^2')
    # Taken from GEMPAK constants
    Re = earth_avg_radius = 6.3712e6 * units.m
    G = gravitational_constant = (units.Quantity(1, units.
                                                 newtonian_constant_of_gravitation)
                                  .to('m^3 / kg / s^2'))
    omega = earth_avg_angular_vel = 2 * units.pi / units.sidereal_day
    d = earth_sfc_avg_dist_sun = 1.496e11 * units.m
    S = earth_solar_irradiance = units.Quantity(1.368e3, 'W / m^2')
    delta = earth_max_declination = 23.45 * units.deg
    earth_orbit_eccentricity = 0.0167
    earth_mass = me = 5.9722e24 * units.kg

    # molar gas constant
    R = units.Quantity(1.0, units.R).to('J / K / mol')

    #
    # Water
    #
    # From: https://pubchem.ncbi.nlm.nih.gov/compound/water
    Mw = water_molecular_weight = units.Quantity(18.01528, 'g / mol')
    Rv = water_gas_constant = R / Mw
    # Nominal density of liquid water at 0C
    rho_l = density_water = units.Quantity(1e3, 'kg / m^3')
    Cp_v = wv_specific_heat_press = units.Quantity(1952., 'm^2 / s^2 / K')
    Cv_v = wv_specific_heat_vol = units.Quantity(1463., 'm^2 / s^2 / K')
    Cp_l = water_specific_heat = units.Quantity(4218., 'm^2 / s^2 / K')  # at 0C
    Lv = water_heat_vaporization = units.Quantity(2.501e6, 'm^2 / s^2')  # at 0C
    Lf = water_heat_fusion = units.Quantity(3.34e5, 'm^2 / s^2')  # at 0C
    Cp_i = ice_specific_heat = units.Quantity(2106, 'm^2 / s^2 / K')  # at 0C
    rho_i = density_ice = units.Quantity(917, 'kg / m^3')  # at 0C

    # Dry air -- standard atmosphere
    Md = dry_air_molecular_weight = units.Quantity(28.9644, 'g / mol')
    Rd = dry_air_gas_constant = R / Md
    dry_air_spec_heat_ratio = 1.4
    Cp_d = dry_air_spec_heat_press = units.Quantity(1005, 'm^2 / s^2 / K')  # Bolton 1980
    Cv_d = dry_air_spec_heat_vol = Cp_d / dry_air_spec_heat_ratio
    rho_d = dry_air_density_stp = ((1000. * units.mbar)
                                   / (Rd * 273.15 * units.K)).to('kg / m^3')

    # General meteorology constants
    P0 = pot_temp_ref_press = 1000. * units.mbar
    kappa = poisson_exponent = (Rd / Cp_d).to('dimensionless')
    gamma_d = dry_adiabatic_lapse_rate = g / Cp_d
    epsilon = molecular_weight_ratio = (Mw / Md).to('dimensionless')

del Exporter
