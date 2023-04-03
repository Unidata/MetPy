# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
==================
Working With Units
==================

Early in our scientific careers we all learn about the importance of paying
attention to units in our calculations. Unit conversions can still get the best
of us and have caused more than one major technical disaster, including the
crash and complete loss of the $327 million Mars Climate Orbiter.

In MetPy, we use the ``pint`` library and a custom unit registry to help prevent
unit mistakes in calculations. That means that every quantity you pass to MetPy
should have units attached, just like if you were doing the calculation on
paper! This simplifies the MetPy API by eliminating the need to specify units
various functions. Instead, only the final results need to be converted to desired units. For
more information on unit support, see the documentation for
`Pint <https://pint.readthedocs.io>`_. Particular attention should be paid to the support
for `temperature units <https://pint.readthedocs.io/en/stable/user/nonmult.html>`_.

In this tutorial we'll show some examples of working with units and get you on your way to
utilizing the computation functions in MetPy.
"""

#########################################################################
# Getting Started
# ---------------
# To use units, the first step is to import the default MetPy units registry from the
# :mod:`~metpy.units` module:
import numpy as np

import metpy.calc as mpcalc
from metpy.units import units

#########################################################################
# The unit registry encapsulates all of the available units, as well as any pertinent settings.
# The registry also understands unit prefixes and suffixes; this allows the registry to
# understand ``'kilometer'`` and ``'meters'`` in addition to the base ``'meter'`` unit.
#
# In general, using units is only a small step on top of using the :class:`numpy.ndarray`
# object.
#
# Adding Units to Data
# --------------------
# The easiest way to attach units to an array (or integer, float, etc.) is to multiply by the
# units:
distance = np.arange(1, 5) * units.meters

#########################################################################
# It is also possible to directly construct a :class:`pint.Quantity`, with a full units string:
time = units.Quantity(np.arange(2, 10, 2), 'sec')

#########################################################################
# Compound units can be constructed by the direct mathematical operations necessary:
9.81 * units.meter / (units.second * units.second)

#########################################################################
# This verbose syntax can be reduced by using the unit registry's support for parsing units:
9.81 * units('m/s^2')

#########################################################################
# Operations With Units
# ---------------------
# With units attached, it is possible to perform mathematical operations, resulting in the
# proper units:
print(distance / time)

#########################################################################
# For multiplication and division, units can combine and cancel. For addition and subtraction,
# instead the operands must have compatible units. For instance, this works:
print(distance + distance)

#########################################################################
# But for instance, `distance + time` would not work; instead it gives an error:
#
# `DimensionalityError: Cannot convert from 'meter' ([length]) to 'second' ([time])`
#
# Even if the units are not identical, as long as they are dimensionally equivalent, the
# operation can be performed:
print(3 * units.inch + 5 * units.cm)

#########################################################################
# Converting Units
# ----------------
#
# Converting a :class:`~pint.Quantity` between units can be accomplished by using the
# :meth:`~pint.Quantity.to` method call, which constructs a new :class:`~pint.Quantity` in the
# desired units:

print((1 * units.inch).to(units.mm))

#########################################################################
# There is also the :meth:`~pint.Quantity.ito` method which performs the same operation
# in-place:
a = np.arange(5.) * units.meter
a.ito('feet')
print(a)

#########################################################################
# To simplify units, there is also the :meth:`~pint.Quantity.to_base_units` method,
# which converts a quantity to SI units, performing any needed cancellation:
Lf = 3.34e6 * units('J/kg')
print(Lf, Lf.to_base_units(), sep='\n')

#########################################################################
# :meth:`~pint.Quantity.to_base_units` can also be done in-place via the
# :meth:`~pint.Quantity.ito_base_units` method.
#
# By default Pint does not do any more than simple unit simplification, so when you perform
# operations you could get some seemingly odd results:
length = 10.4 * units.inch
width = 5 * units.cm
area = length * width
print(area)

#########################################################################
# This is another place where :meth:`~pint.Quantity.to` comes in handy:
print(area.to('m^2'))

#########################################################################
# Temperature
# -----------
# Temperature units are actually relatively tricky (more like absolutely tricky as
# you'll see). Temperature is a non-multiplicative unit - they are in a system
# with a reference point. That means that not only is there a scaling factor, but
# also an offset. This makes the math and unit book-keeping a little more complex.
# Imagine adding 10 degrees Celsius to 100 degrees Celsius. Is the answer 110
# degrees Celsius or 383.15 degrees Celsius (283.15 K + 373.15 K)? That's why
# there are delta degrees units in the unit registry for offset units. For more
# examples and explanation you can watch MetPy Monday #13:
# https://www.youtube.com/watch?v=iveJCqxe3Z4.
#
# Let's take a look at how this works and fails:
#
# We would expect this to fail because we cannot add two offset units (and it does
# fail as an "Ambiguous operation with offset unit").
#
# `10 * units.degC + 5 * units.degC`
#
# On the other hand, we can subtract two offset quantities and get a delta. A delta unit is
# pint's way of representing a relative change in two offset units, indicating that this is
# not an absolute value of 5 degrees Celsius, but a relative change of 5 degrees Celsius.
print(10 * units.degC - 5 * units.degC)

#########################################################################
# We can add a delta to an offset unit as well since it is a relative change.
print(25 * units.degC + 5 * units.delta_degF)

#########################################################################
# Absolute temperature scales like Kelvin and Rankine do not have an offset
# and therefore can be used in addition/subtraction without the need for a
# delta version of the unit.
print(273 * units.kelvin + 10 * units.kelvin)

#########################################################################
print(273 * units.kelvin - 10 * units.kelvin)

#########################################################################
# MetPy Calculations
# ------------------
# All MetPy calculations are unit-aware and rely on this information to ensure
# that the calculations operate correctly. For example, we can use units to take
# an observation in whatever units are most convenient and let MetPy handle everything
# under the hood. Below we calculate dewpoint from the temperature and relative humidity:
temperature = 73.2 * units.degF
rh = 64 * units.percent
dewpoint = mpcalc.dewpoint_from_relative_humidity(temperature, rh)

print(dewpoint)

#########################################################################
# or back to Fahrenheit:
print(dewpoint.to('degF'))

#########################################################################
# Dropping Units
# --------------
# While units are part of the MetPy ecosystem, they can be a headache after we have
# computed the desired quantities with MetPy and would like to move on. For example,
# we might have computed the dewpoint temperature for two points, say A and B, and
# would like to compute the average:
temperature_a = 73.2 * units.degF
rh_a = 64 * units.percent
dewpoint_a = mpcalc.dewpoint_from_relative_humidity(temperature_a, rh_a)

temperature_b = 71.1 * units.degF
rh_b = 52 * units.percent
dewpoint_b = mpcalc.dewpoint_from_relative_humidity(temperature_b, rh_b)

#########################################################################
# Per our previous discussion on temperature units, adding two temperatures together
# won't work. In this case, the easiest way to add two quantities and compute
# an average is by dropping the units attached to the values via ``.magnitude``:
print(dewpoint_b.magnitude)

#########################################################################
dewpoint_mean = (dewpoint_a.magnitude + dewpoint_b.magnitude) / 2.
print(dewpoint_mean)

#########################################################################
# Common Mistakes
# ---------------
# There are a few common mistakes the new users often make. Be sure to check
# these when you're having issues.
#
# * Pressure units are `mbar` or `hPa` for common atmospheric measurements. The
#   unit `mb` is actually millibarns--a unit used in particle physics.
# * When using masked arrays, units must be multiplied on the left side. This
#   will be addressed in the future, but is a current limitation in the
#   ecosystem. The expected error will be
#   `AttributeError: 'MaskedArray' object has no attribute 'units'` or calculation
#   functions complaining about expecting a units and getting "dimensionless".
