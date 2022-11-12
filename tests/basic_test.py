from metpy.calc import (add_height_to_pressure, add_pressure_to_height,
                        altimeter_to_sea_level_pressure, altimeter_to_station_pressure,
                        apparent_temperature, coriolis_parameter, geopotential_to_height,
                        heat_index, height_to_geopotential, height_to_pressure_std,
                        pressure_to_height_std, sigma_to_pressure, smooth_circular,
                        smooth_gaussian, smooth_n_point, smooth_rectangular, smooth_window,
                        wind_components, wind_direction, wind_speed, windchill, zoom_xarray)
from metpy.units import units

d = wind_direction(3. * units('m/s'), 4. * units('m/s'), convention="to")
print(d)

u, v = wind_components(20 * units('m/s'), 50 * units.deg)
print(u)
print(v)

wc = windchill(51 * units.degF, 1 * units('m/s'), face_level_winds="true", mask_undefined="true")
print(wc)

hi = heat_index(30 * units.degC, 50 * units.percent)
print(hi)

temperature = 25 * units.degC
humidity = 80 * units.percent
wind = 20 * units.mph
res = apparent_temperature(30 * units.degC, 20 * units.percent, 20 * units.mph, face_level_winds=True)
print(res)