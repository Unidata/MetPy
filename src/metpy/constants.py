# Copyright (c) 2008,2015,2016,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

r"""A collection of meteorologically significant constant and thermophysical property values.

Earth
-----
======================== =============== =========== ======================================= ===============================================================
Name                     Symbol          Short Name  Units                                   Description
------------------------ --------------- ----------- --------------------------------------- ---------------------------------------------------------------
earth_avg_radius         :math:`R_e`     Re          :math:`\text{m}`                        Avg. radius of the Earth
earth_gravity            :math:`g`       g           :math:`\text{m s}^{-2}`                 Avg. gravity acceleration on Earth
gravitational_constant   :math:`G`       G           :math:`\text{m}^{3} {kg}^{-1} {s}^{-2}` Gravitational constant
earth_avg_angular_vel    :math:`\Omega`  omega       :math:`\text{rad s}^{-1}`               Avg. angular velocity of Earth
earth_sfc_avg_dist_sun   :math:`d`       d           :math:`\text{m}`                        Avg. distance of the Earth from the Sun
earth_solar_irradiance   :math:`S`       S           :math:`\text{W m}^{-2}`                 Avg. solar irradiance of Earth
earth_max_declination    :math:`\delta`  delta       :math:`\text{degrees}`                  Max. solar declination angle of Earth
earth_orbit_eccentricity :math:`e`                   :math:`\text{None}`                     Avg. eccentricity of Earth's orbit
earth_mass               :math:`m_e`     me          :math:`\text{kg}`                       Total mass of the Earth (approx)
======================== =============== =========== ======================================= ===============================================================

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
    earth_gravity = g = 9.80665 * units('m / s^2')                                  # TODO: doc: codata18, unc exact
    Re = earth_avg_radius = 6371008.7714 * units('m')                               # TODO: doc: GRS80
    G = gravitational_constant = 6.67430e-11 * units('m^3 / kg / s^2')              # TODO: doc: codata18, unc +-.00015e-11
    GM = geocentric_gravitational_constant = 3986005e8 * units('m^3 / s^2')         # TODO: doc: GRS80
    omega = earth_avg_angular_vel = 7292115e-11 * units('rad / s')                  # TODO: doc: GRS80
    d = earth_sfc_avg_dist_sun = 149597870700. * units('m')                         # TODO: doc: IAU2012 https://www.iau.org/static/resolutions/IAU2012_English.pdf
    S = earth_solar_irradiance = 1360.8 * units('W / m^2')                          # TODO: doc: kopp2011 https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2010GL045777
    delta = earth_max_declination = 23.45 * units('degrees')
    earth_orbit_eccentricity = 0.0167 * units('dimensionless')
    earth_mass = me = geocentric_gravitational_constant / gravitational_constant    # TODO: doc: GRS80 / codata18 derived, "with atmo"

    # molar gas constant
    R = 8.314462618 * units('J / mol / K')                                          # TODO: doc: codata18, unc exact


    # Water
    Mw = water_molecular_weight = 18.015268 * units('g / mol')                      # TODO: doc: IAPWS const http://www.iapws.org/relguide/fundam.pdf
    Rv = water_gas_constant = R / Mw
    rho_l = density_water = 999.97495 * units('kg / m^3')                           # TODO: doc: IAPWS const and update description, describe as max 
    wv_specific_heat_ratio = 1.330 * units('dimensionless')                         # at ~25C from dof
    Cp_v = wv_specific_heat_press = (
        wv_specific_heat_ratio * Rv / (wv_specific_heat_ratio - 1)
    )
    Cv_v = wv_specific_heat_vol = Cp_v / wv_specific_heat_ratio
    Cp_l = water_specific_heat = 4.2194 * units('kJ / kg / K')                      # TODO: doc iapws95 wagner2002?
    Lv = water_heat_vaporization = 2.50084e6 * units('J / kg')                      # TODO: doc wmo1966
    Lf = water_heat_fusion = 3.337e5 * units('J / kg')                              # TODO: doc wmo1966
    Cp_i = ice_specific_heat = 2090 * units('J / kg / K')                           # TODO: doc wmo1966
    rho_i = density_ice = 917 * units('kg / m^3')  # at 0C

    # Dry air
    Md = dry_air_molecular_weight = 28.96546e-3 * units('kg / mol')                 # TODO: doc: CIPM2007 https://www.nist.gov/system/files/documents/calibrations/CIPM-2007.pdf
    Rd = dry_air_gas_constant = R / Md
    dry_air_spec_heat_ratio = 1.4 * units('dimensionless')                          # at ~20C NEEDS CITATION
    Cp_d = dry_air_spec_heat_press = (
        dry_air_spec_heat_ratio * Rd / (dry_air_spec_heat_ratio - 1)
    )
    Cv_d = dry_air_spec_heat_vol = Cp_d / dry_air_spec_heat_ratio
    rho_d = dry_air_density_stp = (
        (1000. * units('mbar')) / (Rd * 273.15 * units('K'))
    ).to('kg / m^3')

    # General meteorology constants
    P0 = pot_temp_ref_press = 1000. * units('mbar')
    kappa = poisson_exponent = (Rd / Cp_d).to('dimensionless')
    gamma_d = dry_adiabatic_lapse_rate = g / Cp_d
    epsilon = molecular_weight_ratio = (Mw / Md).to('dimensionless')

del Exporter
