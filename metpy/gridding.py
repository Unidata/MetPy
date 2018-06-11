# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Provides tools for interpolating irregularly spaced data onto a regular grid.

Deprecated in 0.9 in favor of the more general `interpolate` subpackage.
"""

import warnings

from .deprecation import metpyDeprecation
from .interpolate import *  # noqa: F401, F403

__all__ = ['interpolate', 'inverse_distance', 'natural_neighbor', 'remove_nan_observations',
           'remove_observations_below_value', 'remove_repeat_coordinates']

# Any use of this module should raise a deprecation warning
warnings.warn('The use of the "gridding" subpackage has been deprecated, and will be removed '
              'in 0.12. Use the "interpolate" subpackage instead.', metpyDeprecation)
