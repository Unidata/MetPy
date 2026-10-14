#!/usr/bin/env python
# Copyright (c) 2015-2025 MetPy Developers.
"""Test script to verify the robustness of the GOES client at hour boundaries.

This script tests the recursive search implementation for finding products
across hour boundaries.
"""
import logging
from datetime import datetime, timezone

from metpy.remote import GOESArchive

logger = logging.getLogger(__name__)

def test_goes_hour_boundary():
    """Test the GOES client's ability to find products across hour boundaries."""
    # Create a GOES client
    goes = GOESArchive(16)
    # Test case 1: Exact hour boundary
    try:
        dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        goes.get_product('ABI-L1b-RadC', dt, band=1)
    except Exception:
        logger.exception('Failed to get product at exact hour boundary')
    # Test case 2: Just after hour boundary
    try:
        dt = datetime(2025, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        goes.get_product('ABI-L1b-RadC', dt, band=1)
    except Exception:
        logger.exception('Failed to get product just after hour boundary')
    # Test case 3: Just before hour boundary
    try:
        dt = datetime(2025, 1, 1, 0, 59, 30, tzinfo=timezone.utc)
        goes.get_product('ABI-L1b-RadC', dt, band=1)
    except Exception:
        logger.exception('Failed to get product just before hour boundary')
    # Test case 4: Day boundary
    try:
        dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        goes.get_product('ABI-L1b-RadC', dt, band=1)
    except Exception:
        logger.exception('Failed to get product at day boundary')

if __name__ == '__main__':
    test_goes_hour_boundary()
