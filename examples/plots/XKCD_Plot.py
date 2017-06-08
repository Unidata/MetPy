# Copyright (c) 2008-2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

"""
Stove Ownership
===============

Make a very important plot in the XKCD style.

This is a simple example PR.
"""

import matplotlib.pyplot as plt
import numpy as np


###########################################
# The setup
# ---------
#
# First we setup a figure
# Based on "Stove Ownership" from XKCD by Randall Monroe
# http://xkcd.com/418/

plt.xkcd()
fig = plt.figure()
ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax.set_ylim([-30, 10])

data = np.ones(100)
data[70:] -= np.arange(30)

plt.annotate(
    'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
    xy=(70, 1), arrowprops={'arrowstyle': '->'}, xytext=(15, -10))

plt.plot(data, color='tab:red')

plt.xlabel('time')
plt.ylabel('my overall health')
fig.text(
    0.5, 0.05,
    '"Stove Ownership" from xkcd by Randall Monroe',
    ha='center')

plt.show()

###########################################
# More code could go here
data = np.ones(100)
data[70:] -= np.arange(30)
