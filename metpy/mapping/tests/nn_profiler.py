from cProfile import Profile
from pstats import Stats
import cartopy.crs as ccrs
import numpy as np
from metpy.mapping import MPMap
from metpy.mapping.tests.test_MPGridding import station_test_data

from_proj = ccrs.Geodetic()
to_proj = ccrs.AlbersEqualArea(central_longitude=-97.0000, central_latitude=38.0000)

prof = Profile()

x, y, t = station_test_data("air_temperature", from_proj, to_proj)

x = x[~np.isnan(t)]
y = y[~np.isnan(t)]
t = t[~np.isnan(t)]
prof.enable()

#MPMap.interp_points(x, y, t, "natural_neighbor", 100000, 100000)
#MPMap.interp_points(x, y, t, "natural_neighbor", 50000, 50000)
gx, gy, a = MPMap.interp_points(x, y, t, "natural_neighbor", 100000, 100000)

prof.disable()
print(a.shape)
prof.dump_stats('mystats.stats')
with open('mystats_output.txt', 'wt') as output:
    stats = Stats('mystats.stats', stream=output)
    stats.sort_stats('cumulative', 'time')
    stats.print_stats()