'''This is a collection of meteorologically significant constants.'''

#This should go away once scipy 0.7 is released and can be depended upon
try:
    from scipy.constants import (pi, day, value, kilo
    from scipy.constants import C2F, F2C, K2F, F2K, C2K,K2C
    R = value('molar gas constant')
    del value
except ImportError:
    from math import pi
    day = 86400.
    kilo = 1000.0
    R = 8.314472
    zero_Celsius = 273.15

    def C2K(C):
        """Convert Celcius to Kelvin"""
        return C + zero_Celsius

    def K2C(K):
        """Convert Kelvin to Celcius"""
        return K - zero_Celsius

    def F2C(F):
        """Convert Fahrenheit to Celcius"""
        return (F - 32) / 1.8

    def C2F(C):
        """Convert Celcius to Fahrenheit"""
        return 1.8 * C + 32

    def F2K(F):
        """Convert Fahrenheit to Kelvin"""
        return C2K(F2C(F))

    def K2F(K):
        """Convert Kelvin to Fahrenheit"""
        return C2F(K2C(K))

#Earth
Re = earth_avg_radious = 6.37e6 # m
g = earth_avg_gravity = 9.81 # m s^-2
omega = earth_avg_angular_vel = 2 * pi / day # rad s^-1
d = earth_sfc_avg_dist_sun = 1.50e11 # m
S = earth_solar_irradiance = 1.38e3 # W m^-2

#Water
Mw = water_molecular_weight = 18.016 # g / mol
Rv = water_gas_constant = R / Mw * kilo #J K^-1 kg^-1
rho_l = density_water = 1e3 # Nominal density of liquid water at 0C in kg m^-3
Cp_v = water_vapor_specific_heat_press = 1952. # J K^-1 kg^-1
Cv_v = water_vapor_specific_heat_vol = 1463. # J K^-1 kg^-1
Cp_l = water_specific_heat = 4218. # at 0C J K^-1 kg^-1
Lv = water_latent_heat_vaporization = 2.5e6 #0C J kg^-1
Lf = water_latent_heat_fustion = 3.34e5 #0C J kg^-1
Cp_i = ice_specific_heat = 2106 # at 0C J K^-1 kg^-1
rho_i = density_ice = 917 # at 0C in kg m^-3

#Dry air
Md = dry_air_molecular_weight = 28.97 # g / mol at the sfc
Rd = dry_air_gas_constant = R / Md * kilo # J K^-1 kg^-1
Cp_d = dry_air_spec_heat_press = 1004. # J K^-1 kg^-1
Cv_d = dry_air_spec_heat_vol = 717. # J K^-1 kg^-1
dry_air_density_stp = 1.275 # at 0C 1000mb in kg m^-3

#General meteorology constants
P0 = pot_temp_ref_press = 100000. # Pa
kappa = poisson_exponent = Rd / Cp_d
gamma_d = dry_adiabatic_lapse_rate = g / Cp_d * kilo # K km^-1
epsilon = molecular_weight_ratio = Mw / Md

del pi, day, R, kilo
