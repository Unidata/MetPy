# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test functionality of MetPy's deprecation code."""

import pytest

from metpy import deprecation


class FakeyMcFakeface:
    """Our faked object for testing."""

    @classmethod
    def dontuse(cls):
        """Don't use."""
        deprecation.warn_deprecated('0.0.1', pending=True)
        return False

    @classmethod
    @deprecation.deprecated('0.0.1')
    def really_dontuse(cls):
        """Really, don't use."""
        return False


def test_deprecation():
    """Test our various warnings."""
    with pytest.warns(deprecation.MetpyDeprecationWarning):
        FakeyMcFakeface.dontuse()
        assert FakeyMcFakeface.dontuse.__doc__ == "Don't use."
    with pytest.warns(deprecation.MetpyDeprecationWarning):
        FakeyMcFakeface.really_dontuse()
