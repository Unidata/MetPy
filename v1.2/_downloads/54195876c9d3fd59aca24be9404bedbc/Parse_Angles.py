# Copyright (c) 2015-2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
============
Parse angles
============

Demonstrate how to convert direction strings to angles.

The code below shows how to parse directional text into angles.
It also  demonstrates the function's flexibility
in handling various string formatting.
"""
import metpy.calc as mpcalc

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
# The function can also handle arrays of strings
# with different abbreviations and capitalizations
dir_str_list = ['ne', 'NE', 'NORTHEAST', 'NORTH_EAST', 'NORTH east']
angle_deg_list = mpcalc.parse_angle(dir_str_list)
print(angle_deg_list)
