# Copyright (c) 2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
========
Q-Vector
========

Use `metpy.calc.q_vector`.

This example demonstrates the q_vector calculation by computing them from the example xarray
Dataset and plotting using Matplotlib.
"""
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.cbook import example_data
from metpy.units import units

# load example data
ds = example_data()

# Calculate the temperature advection of the flow
tadv = mpcalc.advection(ds.temperature, ds.uwind, ds.vwind)

# Calculate the q-vectors
u_qvect, v_qvect = mpcalc.q_vector(ds.uwind, ds.vwind, ds.temperature, 850 * units.hPa)

# start figure and set axis
fig, ax = plt.subplots(figsize=(5, 5))

# plot isotherms
cs = ax.contour(ds.lon, ds.lat, ds.temperature, range(4, 26, 2), colors='tab:red',
                linestyles='dashed', linewidths=3)
plt.clabel(cs, fmt='%d', fontsize=16)

# plot temperature advection in Kelvin per 3 hours
cf = ax.contourf(ds.lon, ds.lat, tadv.metpy.convert_units('kelvin/hour') * 3, range(-6, 7, 1),
                 cmap=plt.cm.bwr, alpha=0.75)
plt.colorbar(cf, pad=0, aspect=50)

# plot Q-vectors as arrows, every other arrow
qvec = ax.quiver(ds.lon.values[::2], ds.lat.values[::2],
                 u_qvect[::2, ::2] * 1e13, v_qvect[::2, ::2] * 1e13,
                 color='black', scale=1000, alpha=0.5, width=0.01)

qk = ax.quiverkey(qvec, 0.8, 0.9, 200, r'$200 m^2/kg/s$', labelpos='E',
                  coordinates='figure')

ax.set(xlim=(260, 270), ylim=(30, 40))
ax.set_title('Q-Vector Calculation')

plt.show()
