# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for versioning."""


def get_version():
    """Get MetPy's version.

    Either get it from package metadata, or get it using version control information if
    an editable installation.
    """
    from importlib.metadata import distribution, PackageNotFoundError

    try:
        dist = distribution(__package__)

        # First see if we can find this file from pip to check for an editable install
        if direct := dist.read_text('direct_url.json'):
            import json

            # Parse file and look for editable key
            info = json.loads(direct)
            if info.get('dir_info', {}).get('editable'):
                import contextlib

                # If editable try to get version using setuptools_scm
                with contextlib.suppress(ImportError, LookupError):
                    from setuptools_scm import get_version
                    return get_version(root='../..', relative_to=__file__,
                                       version_scheme='post-release')

        # With any error or not an editable install, we use the version from the metadata
        return dist.version
    except PackageNotFoundError:
        return 'Unknown'
