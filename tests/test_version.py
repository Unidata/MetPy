# Copyright (c) 2024 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test package version."""
import importlib
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import metpy

def test_version():
    """Test that MetPy version is not None."""
    assert metpy.__version__ is not None


def test_version_installed_with_importlib_metadata_failure():
    """Test that version works when importlib_metadata is not available."""
    with patch('importlib.metadata.version', side_effect=PackageNotFoundError):
        importlib.reload(metpy)
        assert metpy.__version__ != 'Unknown'


def test_version_notinstalled():
    """Test that version works when not installed."""
    with patch('os.path.dirname', return_value='/bogus/bogus/bogus'):
        importlib.reload(metpy)
        assert metpy.__version__ != 'Unknown'
