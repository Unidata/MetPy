# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""MetPy's IO module contains classes for reading files. These classes are written
to take both file names (for local files) or file-like objects; this allows reading files
that are already in memory (using :class:`python:io.StringIO`) or remote files
(using :func:`~python:urllib.request.urlopen`).

There are also classes to implement concepts from the Common Data Model (CDM). The
purpose of these is to simplify data access by proving an interface similar to that
of netcdf4-python.
"""

from .gini import *  # noqa: F403
from .nexrad import *  # noqa: F403
from .upperair import *  # noqa: F403

__all__ = []
__all__.extend(gini.__all__)  # pylint: disable=undefined-variable
__all__.extend(nexrad.__all__)  # pylint: disable=undefined-variable
__all__.extend(upperair.__all__)  # pylint: disable=undefined-variable
