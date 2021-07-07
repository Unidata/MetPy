# Copyright (c) 2015,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for reading, calculating, and plotting with weather data."""

# What do we want to pull into the top-level namespace?
import os
import sys
import warnings

if sys.version_info < (3,):
    raise ImportError(
        """You are running MetPy 0.12.0 or greater on Python 2.

        MetPy 0.12.0 and above are no longer compatible with Python 2, but this version was
        still installed. Sorry about that; it should not have happened. Make sure you have
        pip >= 9.0 to avoid this kind of issue, as well as setuptools >= 24.2:

        $ pip install pip setuptools --upgrade

        Your choices:

        - Upgrade to Python 3.

        - Install an older version of MetPy:

        $ pip install 'metpy=0.11.1'
        """)

# Must occur before below imports
warnings.filterwarnings('ignore', 'numpy.dtype size changed')
os.environ['PINT_ARRAY_PROTOCOL_FALLBACK'] = '0'

from ._version import get_version  # noqa: E402
from .xarray import *  # noqa: F401, F403, E402

__version__ = get_version()
del get_version
