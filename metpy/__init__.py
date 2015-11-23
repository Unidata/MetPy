# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# What do we want to pull into the top-level namespace?

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
