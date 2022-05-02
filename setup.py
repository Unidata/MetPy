# Copyright (c) 2008,2010,2015,2016,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Setup script for installing MetPy."""

import sys

from setuptools import setup

if sys.version_info[0] < 3:
    error = """
    If you're using Python 2.7, please install MetPy v0.11.1,
    which is the last release of MetPy that supports Python 2.7,
    but it is no longer maintained.

    Python {py} detected.
    """.format(py='.'.join(str(v) for v in sys.version_info[:3]))

    print(error)  # noqa: T201
    sys.exit(1)

setup(use_scm_version={'version_scheme': 'post-release'})
