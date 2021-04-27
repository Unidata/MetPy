# Copyright (c) 2008,2015,2016,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

r"""A collection of meteorologically significant constant and thermophysical property values.

Earth
-----
======================== =============== =========== ======================================= ===============================================================
Name                     Symbol          Short Name  Units                                   Description
------------------------ --------------- ----------- --------------------------------------- ---------------------------------------------------------------
earth_avg_radius         :math:`R_e`     Re          :math:`\text{m}`                        Avg. radius of the Earth [1]_
earth_gravity            :math:`g`       g           :math:`\text{m s}^{-2}`                 Avg. gravity acceleration on Earth [2]_
gravitational_constant   :math:`G`       G           :math:`\text{m}^{3} {kg}^{-1} {s}^{-2}` Gravitational constant [2]_
earth_avg_angular_vel    :math:`\Omega`  omega       :math:`\text{rad s}^{-1}`               Avg. angular velocity of Earth [1]_
earth_sfc_avg_dist_sun   :math:`d`       d           :math:`\text{m}`                        Avg. distance of the Earth from the Sun [3]_
earth_solar_irradiance   :math:`S`       S           :math:`\text{W m}^{-2}`                 Avg. solar irradiance of Earth [4]_
earth_max_declination    :math:`\delta`  delta       :math:`\text{degrees}`                  Max. solar declination angle of Earth
earth_orbit_eccentricity :math:`e`                   :math:`\text{None}`                     Avg. eccentricity of Earth's orbit
earth_mass               :math:`m_e`     me          :math:`\text{kg}`                       Total mass of the Earth (approx) [1]_ [2]_
======================== =============== =========== ======================================= ===============================================================

Water
-----
======================= ================ ========== ============================ ====================================================
Name                    Symbol           Short Name Units                        Description
----------------------- ---------------- ---------- ---------------------------- ----------------------------------------------------
water_molecular_weight  :math:`M_w`      Mw         :math:`\text{g mol}^{-1}`    Molecular weight of water [5]_
water_gas_constant      :math:`R_v`      Rv         :math:`\text{J (K kg)}^{-1}` Gas constant for water vapor [2]_ [5]_
density_water           :math:`\rho_l`   rho_l      :math:`\text{kg m}^{-3}`     Maximum recommended density of liquid water, 0-40C [5]_
wv_specific_heat_press  :math:`C_{pv}`   Cp_v       :math:`\text{J (K kg)}^{-1}` Specific heat at constant pressure for water vapor
wv_specific_heat_vol    :math:`C_{vv}`   Cv_v       :math:`\text{J (K kg)}^{-1}` Specific heat at constant volume for water vapor
water_specific_heat     :math:`Cp_l`     Cp_l       :math:`\text{J (K kg)}^{-1}` Specific heat of liquid water at 0C [6]_
water_heat_vaporization :math:`L_v`      Lv         :math:`\text{J kg}^{-1}`     Latent heat of vaporization for liquid water at 0C [7]_
water_heat_fusion       :math:`L_f`      Lf         :math:`\text{J kg}^{-1}`     Latent heat of fusion for liquid water at 0C [7]_
ice_specific_heat       :math:`C_{pi}`   Cp_i       :math:`\text{J (K kg)}^{-1}` Specific heat of ice at 0C [7]_
density_ice             :math:`\rho_i`   rho_i      :math:`\text{kg m}^{-3}`     Density of ice at 0C
======================= ================ ========== ============================ ====================================================

Dry Air
-------
======================== ================ ============= ============================ ====================================================================
Name                     Symbol           Short Name    Units                        Description
------------------------ ---------------- ------------- ---------------------------- --------------------------------------------------------------------
dry_air_molecular_weight :math:`M_d`      Md            :math:`\text{g / mol}`       Nominal molecular weight of dry air at the surface of th Earth [8]_
dry_air_gas_constant     :math:`R_d`      Rd            :math:`\text{J (K kg)}^{-1}` Gas constant for dry air at the surface of the Earth
dry_air_spec_heat_press  :math:`C_{pd}`   Cp_d          :math:`\text{J (K kg)}^{-1}` Specific heat at constant pressure for dry air
dry_air_spec_heat_vol    :math:`C_{vd}`   Cv_d          :math:`\text{J (K kg)}^{-1}` Specific heat at constant volume for dry air
dry_air_density_stp      :math:`\rho_d`   rho_d         :math:`\text{kg m}^{-3}`     Density of dry air at 0C and 1000mb
======================== ================ ============= ============================ ====================================================================

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

.. [1] [Moritz2000]_
.. [2] [CODATA2018]_
.. [3] [IAU2012]_
.. [4] [Kopp2011]_
.. [5] [IAPWS2001]_
.. [6] [IAPWS1995]_
.. [7] [WMO1966]_
.. [8] [Picard2008]_
"""  # noqa: E501

from .package_tools import Exporter
from .units import units

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
    Rv = water_gas_constant = R / Mw
    rho_l = density_water = units.Quantity(999.97495, 'kg / m^3')
    wv_specific_heat_ratio = units.Quantity(1.330, 'dimensionless')
    Cp_v = wv_specific_heat_press = (
        wv_specific_heat_ratio * Rv / (wv_specific_heat_ratio - 1)
    )
    Cv_v = wv_specific_heat_vol = Cp_v / wv_specific_heat_ratio
    Cp_l = water_specific_heat = units.Quantity(4.2194, 'kJ / kg / K')
    Lv = water_heat_vaporization = units.Quantity(2.50084e6, 'J / kg')
    Lf = water_heat_fusion = units.Quantity(3.337e5, 'J / kg')
    Cp_i = ice_specific_heat = units.Quantity(2090, 'J / kg / K')
    rho_i = density_ice = units.Quantity(917, 'kg / m^3')

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
