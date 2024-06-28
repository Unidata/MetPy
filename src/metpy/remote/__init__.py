# Copyright (c) 2015,2016,2018,2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for reading various file formats.

Classes supporting formats are written to take both file names (for local files) or file-like
objects; this allows reading files that are already in memory
(using :class:`python:io.StringIO`) or remote files
(using :func:`~python:urllib.request.urlopen`).

``station_info`` is an instance of `StationLookup` to find information about station locations
(e.g. latitude, longitude, altitude) from various sources.
"""

from .aws import *  # noqa: F403
from ..package_tools import set_module

__all__ = aws.__all__[:]  # pylint: disable=undefined-variable

set_module(globals())
