#!/usr/bin/env python
# Copyright (c) 2015-2025 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `metpy.remote.GOESArchive` module."""
from datetime import datetime, timezone

from metpy.remote import GOESArchive
from metpy.testing import needs_aws


@needs_aws
def test_goes_hour_boundary():
    """Test the GOES client's ability to find products across hour boundaries."""
    # Create a GOES client
    goes = GOESArchive(16)
    # Test case 1: Exact hour boundary
    dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    prod = goes.get_product('ABI-L1b-RadC', dt, band=1)
    assert prod.url is not None
    # Test case 2: Just after hour boundary
    dt = datetime(2025, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
    prod = goes.get_product('ABI-L1b-RadC', dt, band=1)
    assert prod.url is not None
    # Test case 3: Just before hour boundary
    dt = datetime(2025, 1, 1, 0, 59, 30, tzinfo=timezone.utc)
    prod = goes.get_product('ABI-L1b-RadC', dt, band=1)
    assert prod.url is not None
    # Test case 4: Day boundary
    dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    prod = goes.get_product('ABI-L1b-RadC', dt, band=1)
    assert prod.url is not None
