#!/usr/bin/env python
"""
Test script to verify the robustness of the GOES client at hour boundaries.
This script tests the recursive search implementation for finding products
across hour boundaries.
"""
import sys
from datetime import datetime, timezone, timedelta

from metpy.remote import GOESArchive

def test_goes_hour_boundary():
    """Test the GOES client's ability to find products across hour boundaries."""
    print("Testing GOES client at hour boundaries...")
    
    # Create a GOES client
    goes = GOESArchive(16)
    
    # Test case 1: Exact hour boundary
    # This would have failed with the old implementation if no products exist in the new hour
    try:
        # Use a time at exactly the top of an hour
        dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        print(f"\nTest 1: Searching at exact hour boundary: {dt}")
        product = goes.get_product('ABI-L1b-RadC', dt, band=1)
        print(f"Success! Found product: {product.name}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 2: Just after hour boundary
    try:
        # Use a time just after the top of an hour
        dt = datetime(2025, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        print(f"\nTest 2: Searching just after hour boundary: {dt}")
        product = goes.get_product('ABI-L1b-RadC', dt, band=1)
        print(f"Success! Found product: {product.name}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 3: Just before hour boundary
    try:
        # Use a time just before the top of an hour
        dt = datetime(2025, 1, 1, 0, 59, 30, tzinfo=timezone.utc)
        print(f"\nTest 3: Searching just before hour boundary: {dt}")
        product = goes.get_product('ABI-L1b-RadC', dt, band=1)
        print(f"Success! Found product: {product.name}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 4: Day boundary
    try:
        # Use a time at day boundary
        dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        print(f"\nTest 4: Searching at day boundary: {dt}")
        product = goes.get_product('ABI-L1b-RadC', dt, band=1)
        print(f"Success! Found product: {product.name}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_goes_hour_boundary()
