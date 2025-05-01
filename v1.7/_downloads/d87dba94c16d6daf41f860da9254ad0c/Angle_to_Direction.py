# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
==================
Angle to Direction
==================

Demonstrate how to convert angles to direction strings.

The code below shows how to convert angles into directional text.
It also  demonstrates the function's flexibility.
"""
import metpy.calc as mpcalc
from metpy.units import units

###########################################
# Create a test value of an angle
angle_deg = 70 * units('degree')
print(angle_deg)

###########################################
# Now throw that angle into the function to
# get the corresponding direction
dir_str = mpcalc.angle_to_direction(angle_deg)
print(dir_str)

###########################################
# The function can also handle array of angles,
# rounding to the nearest direction, handling angles > 360,
# and defaulting to degrees if no units are specified.
angle_deg_list = [0, 361, 719]
dir_str_list = mpcalc.angle_to_direction(angle_deg_list)
print(dir_str_list)

###########################################
# If you want the unabbrieviated version, input full=True
full_dir_str_list = mpcalc.angle_to_direction(angle_deg_list, full=True)
print(full_dir_str_list)
