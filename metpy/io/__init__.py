# Copyright (c) 2015,2016,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for reading various file formats.

These classes are written to take both file names (for local files) or file-like objects;
this allows reading files that are already in memory (using :class:`python:io.StringIO`)
or remote files (using :func:`~python:urllib.request.urlopen`).
"""

from .gini import *  # noqa: F403
from .nexrad import *  # noqa: F403

__all__ = gini.__all__[:]  # pylint: disable=undefined-variable
__all__.extend(nexrad.__all__)  # pylint: disable=undefined-variable
