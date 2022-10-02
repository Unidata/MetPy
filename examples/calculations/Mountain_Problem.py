# Copyright (c) 2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
================
Mountain Problem
================

Use functions from `metpy.calc` to perform calculations for a rising and sinking parcel.

The code below explores a common problem in meteorology where a parcel can be defined and
lifted initiatlly dry adibatically until saturation is reached. It then ascends moist
adiabatically to a desired level before descending back to the original level from which the
parcel started.
"""
import numpy as np

from metpy.calc import dry_lapse, lcl, moist_lapse
from metpy.units import units

###########################################
# To set up the classic mountain problem, let's define that the parcel will start at 1000-hPa
# and ascend to 700-hPa with an initial temperature of 25 Celsius and a dewpoint of 10 Celsius

p = np.linspace(1000, 700, 301) * units.hPa
T = 25 * units.degC
Td = 10 * units.degC

###########################################
# We first need to determine the maximum level of dry ascent. For this we can use the LCL
# function and retain the pressure level and temperature of the parcel at the LCL
lclp, lclt = lcl(p[0], T, Td)
print('Initial dry ascent yields:')
print(f'  LCL Pressure: {lclp:.2f}')
print(f'  LCL Temperature: {lclt:.2f}')
print()

###########################################
# Knowing that the calculation of the dry ascent is accomplished by the LCL calculation, we
# know how to begin our moist ascent. Begin by subsetting the pressure to begin at levels
# less than or equal to the LCL pressure and use the moist_lapse to find the temperature
# at the top of our ascent (700-hPa in our case)
moist_ascent_p = p[p <= lclp]
moist_ascent_t = moist_lapse(moist_ascent_p, lclt)
print('After moist ascent:')
print(f'  Temperature at top of ascent: {moist_ascent_t[-1]:.2f}')
print()

###########################################
# Now to come "down the mountain" our parcel will warm dry adiabatically, so we can use the
# dry_laspe function to descend from the lowest pressure (using [::-1] to reverse the order)
# and convert our solution to Celsius
dry_descent = dry_lapse(p[::-1], moist_ascent_t[-1]).to('degC')

###########################################
# Pulling it all together
print(f'Starting Temperature: {T:.2f}')
print(f'Starting Dewpoint: {Td:.2f}', end='\n\n')
print(f'Final Temperature: {dry_descent[-1]:.2f}')
print(f'Final Dewpoint: {moist_ascent_t[-1]:.2f}')

###########################################
# So as we expect the parcel has warmed and dried out through the combined dry and moist ascent
# due to the release of latent heat and subsequent precipitating out of moisture from the
# parcel
