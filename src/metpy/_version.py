# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for versioning."""
import os
from importlib.metadata import PackageNotFoundError, version


def get_version():
    """Get MetPy's version.

    If the package is installed (read: metpy is in site-packages), use package
    metadata. If not, use what is provided by setuptools_scm and default to
    package data if that also fails.
    """
    # Inspect where this file's parent directory is located
    moddirname = os.path.dirname(os.path.dirname(__file__))
    # If we're in site-packages, try using package metadata
    if moddirname.endswith('site-packages'):
        try:
            return version(__package__)
        except PackageNotFoundError:
            pass
    try:
        from setuptools_scm import get_version
        return get_version(root='../..', relative_to=__file__,
                           version_scheme='post-release')
    except (ImportError, LookupError):
        try:
            return version(__package__)
        except PackageNotFoundError:
            return 'Unknown'
