# Copyright (c) 2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for reading, calculating, and plotting with weather data."""

# What do we want to pull into the top-level namespace?

from ._version import get_versions
from .xarray import *  # noqa: F401, F403
__version__ = get_versions()['version']
del get_versions
