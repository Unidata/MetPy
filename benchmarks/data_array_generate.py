# Copyright (c) 2025 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Create a sample xarray Dataset with 3D variables.

The generated dataset is used as consistent data for benchmarking
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import metpy.calc as mpcalc
from metpy.units import units

# Make lat/lon data over the mid-latitudes
lats = np.linspace(30, 40, 50)
lons = np.linspace(360 - 100, 360 - 90, 50)
pressure = np.linspace(1000, 250, 50) * units.hPa
p_3d = pressure[:, np.newaxis, np.newaxis]

times = pd.date_range('2024/01/01', '2024/06/01', freq='ME')

# Initialize a random number generator
rng = np.random.default_rng()

# Adding height
z = mpcalc.pressure_to_height_std(p_3d)
height = np.tile(z, (1, len(lats), len(lons)))


# make data based on Matplotlib example data for wind barbs
x, y = np.meshgrid(np.linspace(-3, 3, 51), np.linspace(-3, 3, 51))
z = (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)


# make u and v out of the z equation
u = -np.diff(z[:, 1:], axis=0) * 100 + 10
v = np.diff(z[1:, :], axis=1) * 100 + 10
w = np.full([50, 50], 1)

# Make them 3D
# Note: to make some of the linters happy you have to use the second variable in the loop
# which is why there are some random self / self things going on
u_3d = np.zeros((len(pressure), 50, 50))
for i, p in enumerate(pressure):
    u_3d[i, :, :] = u * (1002 - p.magnitude)**.3  # 1002 ensures the entire lower layer isn't 0

v_3d = np.zeros((len(pressure), 50, 50))
for i, p in enumerate(pressure):
    v_3d[i, :, :] = v * (1002 - p.magnitude)**.3

w_3d = np.zeros((len(pressure), 50, 50))
for i, p in enumerate(pressure):
    w_3d[i, :, :] = w * (p.magnitude / p.magnitude) * rng.random()

# Then make them 4D
u_4d = np.zeros((50, 50, 50, len(times)))
for i, tm in enumerate(times):
    if (tm == tm):
        u_4d[:, :, :, i] = u_3d * rng.uniform(-2, 2)
v_4d = np.zeros((50, 50, 50, len(times)))
for i, tm in enumerate(times):
    if (tm == tm):
        v_4d[:, :, :, i] = v_3d * rng.uniform(-2, 2)

w_4d = np.zeros((50, 50, 50, len(times)))
for i, tm in enumerate(times):
    if (tm == tm):
        w_4d[:, :, :, i] = w_3d * rng.uniform(-2, 2)


windspeed = mpcalc.wind_speed(u_4d * units('m/s'), v_4d * units('m/s'))
winddir = mpcalc.wind_direction(u_4d * units('m/s'), v_4d * units('m/s'))

# setting up for annual temperature cycle
days = np.arange(len(times))
annual_cycle = np.sin(2 * np.pi * days / len(times))

seasonal_amplitude = 10 + 273.15  # K difference from summer to winter
seasonal_variation = seasonal_amplitude * annual_cycle


lapse_rate = 6.5 * units('K/km')  # avg env gamma

# make t as colder air to the north and 3d
t_sfc = (np.linspace(15, 5, 50) * np.ones((50, 50)))
t_sfc = (t_sfc + 273.15)

t_3d = np.zeros((len(pressure), 50, 50))  # (pressure, lat, lon)
for i, p in enumerate(pressure):
    t_3d[i, :, :] = (t_sfc * units.K) - (lapse_rate * height[i, :, :])
    p


# Make t colder in the winter, warmer in the summer
t_4d = np.zeros((50, 50, 50, len(times)))
for i, tm in enumerate(times):
    t_4d[:, :, :, i] = t_3d + seasonal_variation[i]
    tm

t_4d = t_4d * units.K


# Generate potential temperature
theta_4d = mpcalc.potential_temperature(p, t_4d[::-1, :, :, :])

# Generate mixing ratio
surface_w = .015  # dimensionless
top_w = .001  # dimensionless (kg/kg)

# constants for mixing ratio calculation
a = (surface_w - top_w) / (pressure[0] - pressure[49])
b = surface_w - a * pressure[49]

w_profile = a * pressure + b


mixingratio_3d = np.zeros((50, len(lats), len(lons)))
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        lat
        lon
        mixingratio_3d[:, i, j] = w_profile

mixingratio_4d = np.zeros((50, 50, 50, len(times)))
for i, tm in enumerate(times):
    mixingratio_4d[:, :, :, i] = mixingratio_3d
    tm


# Generate vapor pressure
vapor_pressure_4d = mpcalc.vapor_pressure(p, mixingratio_4d)

# Generate dewpoint
td_sfc = (np.linspace(10, 0, 50) * np.ones((50, 50)))
td_3d = np.zeros((len(pressure), 50, 50)) * units.kelvin
for i, p in enumerate(pressure):
    # Scale dewpoint colder and drier with height
    scale = (p / 1000.0).magnitude ** 0.8
    td_3d[i, :, :] = (td_sfc * scale) * units.K


td_4d = np.zeros((50, 50, 50, len(times)))
for i, tm in enumerate(times):
    tm
    td_4d[:, :, :, i] = t_3d + seasonal_variation[i] + rng.uniform(-6, 3, size=(50, 50, 50))

td_4d = td_4d * units.K
td_4d = np.minimum(td_4d, t_4d)

# Generate relative humidity from dewpoint
rh = mpcalc.relative_humidity_from_dewpoint(t_4d, td_4d)

# Generate sigma values
sigma_3d = (p_3d - (250 * units.hPa)) / ((1000 * units.hPa) - (250 * units.hPa))

sigma_4d = np.zeros((50, 50, 50, len(times)))
for i, tm in enumerate(times):
    tm
    sigma_4d[:, :, :, i] = sigma_3d[:, :, :]

# Generate geopotential values
geopotential_3d = mpcalc.height_to_geopotential(height)

geopotential_4d = np.zeros((50, 50, 50, len(times)))
for i, tm in enumerate(times):
    tm
    geopotential_4d[:, :, :, i] = geopotential_3d[:, :, :]

# generate specific humidity values
q = mpcalc.specific_humidity_from_mixing_ratio(mixingratio_4d)

# generate wet bulb temperature
wet_bulb_4d = np.zeros((50, 50, 50, len(times)))
for i, tm in enumerate(times):
    tm
    wet_bulb_4d[:, :, :, i] = mpcalc.wet_bulb_temperature(p_3d[:, :, :], t_4d[:, :, :, i],
                                                          td_4d[:, :, :, i])

# generate omega
omega_4d = mpcalc.vertical_velocity_pressure(w_4d * units('m/s'), p_3d, t_4d, mixingratio_4d)

# place data into an xarray dataset object
lat_da = xr.DataArray(lats, dims='lat', attrs={'standard_name': 'latitude',
                                               'units': 'degrees_north'})
lon_da = xr.DataArray(lons, dims='lon', attrs={'standard_name': 'longitude',
                                               'units': 'degrees_east'})
pressure_level = xr.DataArray(pressure.magnitude, dims='pressure',
                              attrs={'standard_name': 'pressure', 'units': 'hPa'})
time_da = xr.DataArray(times, dims='time', attrs={'standard name': 'time'})

coords = {'lat': lat_da, 'lon': lon_da, 'pressure': pressure_level, 'time': time_da}


uwind = xr.DataArray(u_4d, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                     attrs={'standard_name': 'u-component_of_wind', 'units': 'm s-1'})
vwind = xr.DataArray(v_4d, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                     attrs={'standard_name': 'v-component_of_wind', 'units': 'm s-1'})
wwind = xr.DataArray(w_4d, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                     attrs={'standard_name': 'w-component_of_wind', 'units': 'm s-1'})
temperature = xr.DataArray(t_4d, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                           attrs={'standard_name': 'temperature', 'units': 'K'})
height = xr.DataArray(height, dims=['pressure', 'lat', 'lon'],
                      attrs={'standard_name': 'z dimension', 'units': 'km'})
theta = xr.DataArray(theta_4d, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                     attrs={'standard_name': 'Potential temperature', 'units': 'K'})
mixing_ratio = xr.DataArray(mixingratio_4d, coords=coords,
                            dims=['pressure', 'lat', 'lon', 'time'],
                            attrs={'standard_name': 'Mixing Ratio', 'units': 'dimensionless'})
vapor_pressure = xr.DataArray(vapor_pressure_4d, coords=coords,
                              dims=['pressure', 'lat', 'lon', 'time'],
                              attrs={'standard name': 'Vapor pressure', 'units': 'hPa'})
dewpoint = xr.DataArray(td_4d, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                        attrs={'standard name': 'dewpoint', 'units': 'K'})
relative_humidity = xr.DataArray(rh, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                                 attrs={'standard name': 'relative humidity', 'units': '%'})
windspeed = xr.DataArray(windspeed, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                         attrs={'standard_name': 'windspeed', 'units': 'm s-1'})
winddir = xr.DataArray(winddir, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                       attrs={'standard_name': 'wind direction', 'units': 'degrees'})
sigma = xr.DataArray(sigma_4d, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                     attrs={'standard_name': 'sigma', 'units': 'dimensionless'})
geopotential = xr.DataArray(geopotential_4d, coords=coords,
                            dims=['pressure', 'lat', 'lon', 'time'],
                            attrs={'standard_name': 'geopotential', 'units': 'm2 s-2'})
specific_humidity = xr.DataArray(q, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                                 attrs={'standard_name': 'specific humidity',
                                        'units': 'dimensionless'})
wet_bulb_temperature = xr.DataArray(wet_bulb_4d, coords=coords,
                                    dims=['pressure', 'lat', 'lon', 'time'],
                                    attrs={'standard_name': 'wet_bulb_temperature',
                                           'units': 'K'})
omega = xr.DataArray(omega_4d, coords=coords, dims=['pressure', 'lat', 'lon', 'time'],
                     attrs={'standard_name': 'omega component of wind', 'units': 'pascal/s'})

ds = xr.Dataset({'uwind': uwind,
                 'vwind': vwind,
                 'wwind': wwind,
                 'temperature': temperature,
                 'height': height,
                 'theta': theta,
                 'mixing_ratio': mixing_ratio,
                 'vapor_pressure': vapor_pressure,
                 'dewpoint': dewpoint,
                 'relative_humidity': relative_humidity,
                 'windspeed': windspeed,
                 'winddir': winddir,
                 'sigma': sigma,
                 'geopotential': geopotential,
                 'specific_humidity': specific_humidity,
                 'wet_bulb_temperature': wet_bulb_temperature,
                 'omega': omega})

# Step 1: Initialize encoding dict for data variables
encoding = {
    var: {'zlib': True, 'complevel': 9, 'dtype': 'float32'}
    for var in ds.data_vars
}

# Step 2: Add compression settings for coordinates (no dtype conversion)
for coord in ds.coords:
    encoding[coord] = {'zlib': True, 'complevel': 9}


if os.path.exists('data_array_compressed.nc'):
    os.remove('data_array_compressed.nc')

ds.to_netcdf('data_array_compressed.nc', format='NETCDF4', encoding=encoding)
