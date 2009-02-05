'''
This is a collection of meteorologically significant constants.

Earth
-------------
Name                     Abbr. Units    Description
======================== ===== ======== ==================
earth_avg_radius         Re    m        Avg. radius of the Earth
earth_gravity            g     m s^-2   Avg. gravity acceleration on Earth
earth_avg_angular_vel    omega rad s^-1 Avg. angular velocity of Earth
earth_sfc_avg_dist_sun   d     m        Avg. distance of the Earth from the Sun
earth_solar_irradiance   S     W m^-2   Avg. solar irradiance of Earth
earth_max_declination    delta degrees  Max. solar declination angle of Earth
earth_orbit_eccentricity       None     Avg. eccentricity of Earth's orbit
======================== ===== ======== ==================

Water
-------------
Name                    Abbr. Units       Description
======================= ===== =========== ==================
water_molecular_weight  Mw    g mol^-1    Molecular weight of water
water_gas_constant      Rv    J (K kg)^-1 Gas constant for water vapor
density_water           rho_l kg m^-3     Nominal density of liquid water at 0C
wv_specific_heat_press  Cp_v  J (K kg)^-1 Specific heat at constant pressure for
                                          water vapor
wv_specific_heat_vol    Cv_v  J (K kg)^-1 Specific heat at constant volume for
                                          water vapor
water_specific_heat     Cp_l  J (K kg)^-1 Specific heat of liquid water at 0C
water_heat_vaporization Lv    J kg^-1     Latent heat of vaporization for liquid
                                          water at 0C
water_heat_fusion       Lf    J kg^-1     Latent heat of fusion for liquid water
                                          at 0C
ice_specific_heat       Cp_i  J (K kg)^-1 Specific heat of ice at 0C
density_ice             rho_i kg m^-3     Density of ice at 0C
======================= ===== =========== ==================

Dry Air
-------------
Name                     Abbr. Units       Description
======================== ===== =========== ==================
dry_air_molecular_weight Md    g / mol     Nominal molecular weight of dry air
                                           at the surface of th Earth
dry_air_gas_constant     Rd    J (K kg)^-1 Gas constant for dry air at the
                                           surface of the Earth
dry_air_spec_heat_press  Cp_d  J (K kg)^-1 Specific heat at constant pressure
                                           for dry air
dry_air_spec_heat_vol    Cv_d  J (K kg)^-1 Specific heat at constant volume
                                           for dry air
dry_air_density_stp      rho_d kg m^-3     Density of dry air at 0C and 1000mb
======================== ===== ======== ==================

General Meteorology Constants
-------------
Name                     Abbr.   Units    Description
======================== ======= ======== ==================
pot_temp_ref_press       P0      Pa       Reference pressure for potential
                                          temperature
poisson_exponent         kappa   None     Exponent in Poisson's equation (Rd/Cp_d)
dry_adiabatic_lapse_rate gamma_d K km^-1  The dry adiabatic lapse rate
molecular_weight_ratio   epsilon None     Ratio of molecular weight of water to
                                          that of dry air
======================== ======= ======== ==================

Temperature Conversion Functions
-------------

F2C :
    Convert temperature in degrees Farenheit to degrees Celsius

F2K :
    Convert temperature in degrees Farenheit to Kelvin

C2F :
    Convert temperature in degrees Celsius to degrees Farenheit

K2F :
    Convert temperature in Kelvin to degrees Farenheit

C2K :
    Convert temperature in degrees Celsius to Kelvin

K2C :
    Convert temperature in Kelvin to degrees Celsius
'''

__all__ = ['C2F', 'C2K', 'F2K', 'K2C', 'K2F', 'F2C', 'Re', 'earth_avg_radius',
    'g', 'earth_avg_gravity', 'omega', 'earth_avg_angular_vel',
    'd', 'earth_sfc_avg_dist_sun', 'S', 'earth_solar_irradiance',
    'Mw', 'water_molecular_weight', 'Rv', 'water_gas_constant',
    'rho_l', 'density_water', 'Cp_v', 'wv_specific_heat_press',
    'Cv_v', 'wv_specific_heat_vol', 'Cp_l', 'water_specific_heat',
    'Lv', 'water_heat_vaporization', 'Lf', 'water_heat_fusion',
    'Cp_i', 'ice_specific_heat', 'rho_i', 'density_ice',
    'Md', 'dry_air_molecular_weight', 'Rd', 'dry_air_gas_constant',
    'Cp_d', 'dry_air_spec_heat_press', 'Cv_d', 'dry_air_spec_heat_vol',
    'rho_d', 'dry_air_density_stp', 'P0', 'pot_temp_ref_press',
    'kappa', 'poisson_exponent', 'gamma_d', 'dry_adiabatic_lapse_rate',
    'epsilon', 'molecular_weight_ratio', 'delta', 'earth_max_declination',
    'earth_orbit_eccentricity']

from datetime import date
try:
    from scipy.constants import pi, day, value, kilo
    from scipy.constants import C2F, F2C, K2F, F2K, C2K, K2C
except ImportError:
    # Use internal copy
    from scipy_consants import pi, day, value, kilo
    from scipy_constants import C2F, F2C, K2F, F2K, C2K, K2C
R = value('molar gas constant')
del value

#Earth
Re = earth_avg_radius = 6.37e6 # m
g = earth_avg_gravity = 9.81 # m s^-2
omega = earth_avg_angular_vel = 2 * pi / day # rad s^-1
d = earth_sfc_avg_dist_sun = 1.496e11 # m
S = earth_solar_irradiance = 1.368e3 # W m^-2
delta = earth_max_declination = 23.45 # degrees
earth_orbit_eccentricity = 0.0167

#Water
Mw = water_molecular_weight = 18.016 # g / mol
Rv = water_gas_constant = R / Mw * kilo #J K^-1 kg^-1
rho_l = density_water = 1e3 # Nominal density of liquid water at 0C in kg m^-3
Cp_v = wv_specific_heat_press = 1952. # J K^-1 kg^-1
Cv_v = wv_specific_heat_vol = 1463. # J K^-1 kg^-1
Cp_l = water_specific_heat = 4218. # at 0C J K^-1 kg^-1
Lv = water_heat_vaporization = 2.5e6 #0C J kg^-1
Lf = water_heat_fusion = 3.34e5 #0C J kg^-1
Cp_i = ice_specific_heat = 2106 # at 0C J K^-1 kg^-1
rho_i = density_ice = 917 # at 0C in kg m^-3

#Dry air
Md = dry_air_molecular_weight = 28.97 # g / mol at the sfc
Rd = dry_air_gas_constant = R / Md * kilo # J K^-1 kg^-1
Cp_d = dry_air_spec_heat_press = 1004. # J K^-1 kg^-1
Cv_d = dry_air_spec_heat_vol = 717. # J K^-1 kg^-1
rho_d = dry_air_density_stp = 1.275 # at 0C 1000mb in kg m^-3

#General meteorology constants
P0 = pot_temp_ref_press = 100000. # Pa
kappa = poisson_exponent = Rd / Cp_d
gamma_d = dry_adiabatic_lapse_rate = g / Cp_d * kilo # K km^-1
epsilon = molecular_weight_ratio = Mw / Md

del pi, day, R, kilo
