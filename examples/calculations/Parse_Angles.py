# Copyright (c) 2015-2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
Parse angles
=====================================

Use functions from `metpy.calc` as well as pint's unit support to perform calculations.

The code below shows how to parse directional text into angles.
It also demonstrates converting the resulting angles from degrees
into radians. Lastly, it demonstrates the function's flexibility
in handling various formats.
"""
import metpy.calc as mpcalc
from metpy.units import units

###########################################
# Create a test value of a directional text
dir_str = 'SOUTH SOUTH EAST'
print(dir_str)

###########################################
# Now throw that string into the function to calculate
# the corresponding angle
angle_deg = mpcalc.parse_angle(dir_str)
print(angle_deg)

###########################################
# Take the odd units and force them to millibars
angle_rad = angle_deg.to(units.radians)
print(angle_rad)

###########################################
# The function can also handle arrays of string
# in many different abbrieviations and capitalizations
dir_str_list = ['ne', 'NE', 'NORTHEAST', 'NORTH_EAST', 'NORTH east']
angle_deg_list = mpcalc.parse_angle(dir_str_list)
print(angle_deg_list)
