# Copyright (c) 2015-2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
Dewpoint and Mixing Ratio
=========================

Use functions from `metpy.calc` as well as pint's unit support to perform calculations.

The code below converts the mixing ratio value into
a value for vapor pressure assuming both 1000mb and 850mb ambient air
pressure values. It also demonstrates converting the resulting dewpoint
temperature to degrees Fahrenheit.
"""
import metpy.calc as mpcalc
from metpy.units import units

###########################################
# Create a test value of mixing ratio in grams per kilogram
mixing = 10 * units('g/kg')
print(mixing)

###########################################
# Now throw that value with units into the function to calculate
# the corresponding vapor pressure, given a surface pressure of 1000 mb
e = mpcalc.vapor_pressure(1000. * units.mbar, mixing)
print(e)

###########################################
# Take the odd units and force them to millibars
print(e.to(units.mbar))

###########################################
# Take the raw vapor pressure and throw into the dewpoint function
td = mpcalc.dewpoint(e)
print(td)

###########################################
# Which can of course be converted to Fahrenheit
print(td.to('degF'))

###########################################
# Now do the same thing for 850 mb, approximately the pressure of Denver
e = mpcalc.vapor_pressure(850. * units.mbar, mixing)
print(e.to(units.mbar))

###########################################
# And print the corresponding dewpoint
td = mpcalc.dewpoint(e)
print(td, td.to('degF'))
