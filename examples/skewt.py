#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import C2K, K2C

from metpy.calc import get_wind_components, lcl, dry_lapse, parcel_profile
from metpy.plots import SkewT

# Parse the data
p, T, Td, direc, spd = np.loadtxt('testdata/sounding_data.txt',
        usecols=(0, 2, 3, 6, 7), unpack=True)
u,v = get_wind_components(spd, direc)

# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(6.5875, 6.2125))
skew = SkewT(fig)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot_barbs(p, u, v)

l = lcl(p[0], T[0], Td[0])
skew.plot(l, K2C(dry_lapse(l, C2K(T[0]), p[0])), 'ko',
        markerfacecolor='black')

prof = parcel_profile(p, T[0], Td[0])
skew.plot(p, prof, 'k', linewidth=2)

# An example of a slanted line at constant T -- in this case the 0
# isotherm
l = skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000, 100)

# Show the plot
plt.show()
