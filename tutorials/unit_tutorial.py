# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
==============
Units Tutorial
==============

Early in our scientific careers we all learn about the importance of paying
attention to units in our calculations. Unit conversions can still get the best
of us and have caused more than one major technical disaster, including the
crash and complete loss of the $327 million Mars Climate Orbiter.

In MetPy, we use the pint library and a custom unit registry to help prevent
unit mistakes in calculations. That means that every quantity you pass to MetPy
should have units attached, just like if you were doing the calculation on
paper!

In MetPy units are attached by multiplying them with the integer, float, array,
etc. In this tutorial we'll show some examples of working with units and get
you on your way to utilizing the computation functions in MetPy.
"""

import numpy as np

from metpy.units import units

#########################################################################
# Simple Calculation
# ------------------
#
# Let's say we want to calculate the area of a rectangle. It so happens that
# one of our colleagues measures their side of the rectangle in imperial units
# and the other in metric units. No problem! First we need to attach units to
# our measurements. For many units the easiest way is by find the unit as an
# attribute of the unit registry:

length = 10.4 * units.inches
width = 20 * units.meters
print(length, width)

#########################################################################
# Don't forget that you can use tab completion to see what units are available!
# Just about every imaginable quantity is there, but if you find one that isn't,
# we're happy to talk about adding it.
#
# While it may seem like a lot of trouble, let's compute the area of a rectangle
# defined by our length and width variables above. Without units attached, you'd
# need to remember to perform a unit conversion before multiplying or you would
# end up with an area in inch-meters and likely forget about it. With units
# attached, the units are tracked for you.

area = length * width
print(area)

#########################################################################
# That's great, now we have an area, but it is not in a very useful unit still.
# Units can be converted using the `to()` method. While you won't see square meters in
# the units list, we can parse complex/compound units as strings:

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

#########################################################################
# Compound Units
# --------------
# We can create compound units for things like speed by parsing a string of
# units. Abbreviations or full unit names are acceptable.

u = np.random.randint(0, 15, 10) * units('m/s')
v = np.random.randint(0, 15, 10) * units('meters/second')

print(u)
print(v)

#########################################################################
# Common Mistakes
# ---------------
# There are a few common mistakes the new users often make. Be sure to check
# these when you're having issues
#
# * Pressure units are `mbar` or `hPa` for common atmospheric measurements. The
#   unit `mb` is actually millibarns.
# * When using masked arrays, units must be multiplied on the left side. This
#   will be addressed in the future, but is a current limitation in the
#   ecosystem. The expected error will be
#   `AttributeError: 'MaskedArray' object has no attribute 'units'`
