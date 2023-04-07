# Copyright (c) 2023 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Provide tools for accessing data from remote sources.

This currently includes clients for searching and downloading data from public cloud buckets.
"""

from .aws import *  # noqa: F403
from ..package_tools import set_module

__all__ = aws.__all__[:]  # pylint: disable=undefined-variable

set_module(globals())
