from basic import altimeter_to_station_pressure, altimeter_to_sea_level_pressure
from metpy.units import units

altim = 990 * units.hectopascal
elev = 1000 * units.m
temperature = 25 * units.degC

value = altimeter_to_station_pressure(altim, elev)
print(value)

value1 = altimeter_to_sea_level_pressure(altim, elev, temperature)
print(value1)
