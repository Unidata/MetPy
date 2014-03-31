#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from metpy.plots import SkewT

# Parse the data
p, h, T, Td = np.loadtxt('testdata/sounding_data.txt', usecols=range(0, 4),
        unpack=True)

# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(6.5875, 6.2125))
skew = SkewT(fig)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')

# Example of coloring area between profiles
skew.ax.fill_betweenx(p, T, 0, where=T>=0, facecolor='red', alpha=0.4)

# An example of a slanted line at constant T -- in this case the 0
# isotherm
l = skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

skew.plot_dry_adiabats()
skew.plot_mixing_lines()
plt.show()
