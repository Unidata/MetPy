# Copyright (c) 2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for reading, calculating, and plotting with weather data."""

# What do we want to pull into the top-level namespace?

import logging
import warnings

# Must occur before below imports
warnings.filterwarnings('ignore', 'numpy.dtype size changed')

from ._version import get_versions  # noqa: E402
from .xarray import *  # noqa: F401, F403
__version__ = get_versions()['version']
del get_versions

try:
    # Added in Python 3.2, will log anything warning or higher to stderr
    logging.lastResort
except AttributeError:
    # Add our own for MetPy on Python 2.7
    logging.getLogger(__name__).addHandler(logging.StreamHandler())
