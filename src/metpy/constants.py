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
=======
======================== =============== =========== ======================================== ========================== ======================================= ================
Name                     Symbol          Short Name  Value                                    Units                      Description                              Reference
------------------------ --------------- ----------- ---------------------------------------- -------------------------- --------------------------------------- ----------------
earth_gravity            :math:`g`       g           .. autodata:: g                          :math:`\text{m s}^{-2}`    Avg. gravity acceleration on Earth      [Pint]_
earth_avg_radius         :math:`R_e`     Re          .. autodata:: Re                         :math:`\text{m}`           Avg. radius of the Earth                [NASA2016]_
earth_avg_angular_vel    :math:`\Omega`  omega       .. autodata:: omega                      :math:`\text{rad s}^{-1}`  Avg. angular velocity of Earth          [Pint]_
earth_sfc_avg_dist_sun   :math:`d`       d           .. autodata:: d                          :math:`\text{m}`           Avg. distance of the Earth from the Sun [NASA2016]_
earth_solar_irradiance   :math:`S`       S           .. autodata:: S                          :math:`\text{W m}^{-2}`    Avg. solar irradiance of Earth          [Coddington2016]_
earth_max_declination    :math:`\delta`  delta       .. autodata:: delta                      :math:`\text{degrees}`     Max. solar declination angle of Earth   [NASA2017]_
earth_orbit_eccentricity :math:`e`                   .. autodata:: earth_orbit_eccentricity   :math:`\text{None}`        Avg. eccentricity of Earth's orbit      [NASA2017]_
======================== =============== =========== ======================================== ========================== ======================================= ================

Water
-----
======================= ================ ========== ============================ ==================================================== ==========================
Name                    Symbol           Short Name Units                        Description                                          Reference
----------------------- ---------------- ---------- ---------------------------- ---------------------------------------------------- --------------------------
water_molecular_weight  :math:`M_w`      Mw         :math:`\text{g mol}^{-1}`    Molecular weight of water                            [Chase1998]_
water_gas_constant      :math:`R_v`      Rv         :math:`\text{J (K kg)}^{-1}` Gas constant for water vapor                         [Chase1998]_
density_water           :math:`\rho_l`   rho_l      :math:`\text{kg m}^{-3}`     Nominal density of liquid water at 0C, 1013.25 hPa   [NIST]_
wv_specific_heat_press  :math:`C_{pv}`   Cp_v       :math:`\text{J (K kg)}^{-1}` Specific heat at constant pressure for water vapor   [WMO-1966]_
wv_specific_heat_vol    :math:`C_{vv}`   Cv_v       :math:`\text{J (K kg)}^{-1}` Specific heat at constant volume for water vapor     [Hobbs2006]_
water_specific_heat     :math:`Cp_l`     Cp_l       :math:`\text{J (K kg)}^{-1}` Specific heat of liquid water at 0C                  [Wagner2002]_
water_heat_vaporization :math:`L_v`      Lv         :math:`\text{J kg}^{-1}`     Latent heat of vaporization for liquid water at 0C   [WMO-1966]_
water_heat_fusion       :math:`L_f`      Lf         :math:`\text{J kg}^{-1}`     Latent heat of fusion for liquid water at 0C         [WMO-1966]_
ice_specific_heat       :math:`C_{pi}`   Cp_i       :math:`\text{J (K kg)}^{-1}` Specific heat of ice at 0C                           [WMO-1966]_
density_ice             :math:`\rho_i`   rho_i      :math:`\text{kg m}^{-3}`     Density of ice at 0C                                 [Hobbs2006]_
======================= ================ ========== ============================ ==================================================== ==========================

Dry Air
-------
======================== ================ ============= =================== ============================ =============================================================== ======================
Name                     Symbol           Short Name                        Units                        Description                                                     Reference
------------------------ ---------------- ------------- ------------------- ---------------------------- --------------------------------------------------------------- ----------------------
dry_air_molecular_weight :math:`M_d`      Md            .. autodata:: Md    :math:`\text{g / mol}`       Nominal molecular weight of dry air at the surface of th Earth  [ICAO]_
dry_air_gas_constant     :math:`R_d`      Rd            .. autodata:: Rd    :math:`\text{J (K kg)}^{-1}` Gas constant for dry air at the surface of the Earth            [Hobbs2006]_
dry_air_spec_heat_press  :math:`C_{pd}`   Cp_d          .. autodata:: Cp_d  :math:`\text{J (K kg)}^{-1}` Specific heat at constant pressure for dry air at 273K          [WMO-1966]_
dry_air_spec_heat_vol    :math:`C_{vd}`   Cv_d          .. autodata:: Cv_d  :math:`\text{J (K kg)}^{-1}` Specific heat at constant volume for dry air at 273K            [WMO-1966]_
dry_air_density_stp      :math:`\rho_d`   rho_d         .. autodata:: rho_d :math:`\text{kg m}^{-3}`     Density of dry air at 0C and 1013.25mb                          [WMO-1966]_
======================== ================ ============= =================== ============================ =============================================================== ======================

General Meteorology Constants
-----------------------------
======================== ================= =========== ========================= ======================================================= ==================
Name                     Symbol            Short Name   Units                    Description                                             Reference
------------------------ ----------------- ----------- ------------------------- ------------------------------------------------------- ------------------
pot_temp_ref_press       :math:`P_0`       P0          :math:`\text{Pa}`         Reference pressure for potential temperature            [Hobbs2006]_
poisson_exponent         :math:`\kappa`    kappa       :math:`\text{None}`       Exponent in Poisson's equation (Rd/Cp_d)                [Hobbs2006]_
dry_adiabatic_lapse_rate :math:`\gamma_d`  gamma_d     :math:`\text{K km}^{-1}`  The dry adiabatic lapse rate                            [Hobbs2006]_
molecular_weight_ratio   :math:`\epsilon`  epsilon     :math:`\text{None}`       Ratio of molecular weight of water to that of dry air   [WMO-1966]_
======================== ================= =========== ========================= ======================================================= ==================
"""  # noqa: E501

from .package_tools import Exporter
from .units import units

exporter = Exporter(globals())

# Export all the variables defined in this block
with exporter:
    # Earth
    earth_gravity = g = units.Quantity(1.0, units.gravity).to('m / s^2')
    G = gravitational_constant = (units.Quantity(1, units.
                                                 newtonian_constant_of_gravitation)
                                  .to('m^3 / kg / s^2'))
    earth_mass = me = 5.9722e24 * units.kg
    Re = earth_avg_radius = 6.371008e6 * units.m  # NASA
    omega = earth_avg_angular_vel = 2 * units.pi / units.sidereal_day
    d = earth_sfc_avg_dist_sun = 1.496e11 * units.m  # NASA
    S = earth_solar_irradiance = units.Quantity(1.3645e3, 'W / m^2')  # Coddington
    delta = earth_max_declination = 23.4393 * units.deg  # NASA
    earth_orbit_eccentricity = 0.01671022  # NASA

    # molar gas constant
    R = units.Quantity(1.0, units.R).to('J / K / mol')

    # Water
    Mw = water_molecular_weight = units.Quantity(18.01528, 'g / mol')
    Rv = water_gas_constant = R / Mw
    # Nominal density of liquid water at 0C, 1 atm
    rho_l = density_water = units.Quantity(999.84, 'kg / m^3')
    Cp_v = wv_specific_heat_press = units.Quantity(1859.0, 'm^2 / s^2 / K')  # For 0 C, WMO
    Cv_v = wv_specific_heat_vol = units.Quantity(1463., 'm^2 / s^2 / K')  # Hobbs
    Cp_l = water_specific_heat = units.Quantity(4219.4, 'm^2 / s^2 / K')  # at 0C, Wagner 2002
    Lv = water_heat_vaporization = units.Quantity(2.50084e6, 'm^2 / s^2')  # at 0C, WMO tables
    Lf = water_heat_fusion = units.Quantity(3.337e5, 'm^2 / s^2')  # at 0C, WMO tables
    Cp_i = ice_specific_heat = units.Quantity(2090, 'm^2 / s^2 / K')  # at 0C, WMO tables
    rho_i = density_ice = units.Quantity(917, 'kg / m^3')  # at 0C, Hobbs

    # Dry air
    Md = dry_air_molecular_weight = units.Quantity(28.964420, 'g / mol')  # ICAO
    Rd = dry_air_gas_constant = R / Md  # ICAO standard atmosphere

    # WMO tables for 1013.25 hPa and 0 C
    rho_d = dry_air_density_stp = (1013.25 * units.hPa /
                                   (Rd * 273.15 * units.kelvin)).to('kg/m^3')
    dry_air_spec_heat_ratio = 1.4  # ICAO
    Cp_d = dry_air_spec_heat_press = units.Quantity(1005, 'm^2 / s^2 / K')  # WMO Tables
    Cv_d = dry_air_spec_heat_vol = Cp_d / dry_air_spec_heat_ratio  # WMO Tables, Hobbs

    # General meteorology constants
    P0 = pot_temp_ref_press = 1000. * units.mbar  # Hobbs
    kappa = poisson_exponent = (Rd / Cp_d).to('dimensionless')  # Hobbs, for dry air
    gamma_d = dry_adiabatic_lapse_rate = g / Cp_d  # Hobbs
    epsilon = molecular_weight_ratio = (Mw / Md).to('dimensionless')

del Exporter
