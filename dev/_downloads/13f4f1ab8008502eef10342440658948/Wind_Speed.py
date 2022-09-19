# Copyright (c) 2015-2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
==========
Wind Speed
==========

Use `metpy.calc.wind_speed`.

This example demonstrates the calculation of wind speed using the example xarray Dataset and
plotting using Matplotlib.
"""
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.cbook import example_data

# load example data
ds = example_data()

# Calculate the total deformation of the flow
wind_speed = mpcalc.wind_speed(ds.uwind, ds.vwind)

# start figure and set axis
fig, ax = plt.subplots(figsize=(5, 5))

# plot wind speed
cf = ax.contourf(ds.lon, ds.lat, wind_speed, range(5, 80, 5), cmap=plt.cm.BuPu)
plt.colorbar(cf, pad=0, aspect=50)
ax.barbs(ds.lon.values, ds.lat.values, ds.uwind, ds.vwind, color='black', length=5, alpha=0.5)
ax.set(xlim=(260, 270), ylim=(30, 40))
ax.set_title('Wind Speed Calculation')

plt.show()
