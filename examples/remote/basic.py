# Copyright (c) 2023 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
==================
Remote Data Access
==================

Use MetPy to access data hosted in known AWS S3 buckets
"""
from datetime import datetime, timedelta

from metpy.remote import GOESArchive, NEXRADLevel2Archive, NEXRADLevel3Archive

###################
# NEXRAD Level 2

# Get the nearest product to a time
prod = NEXRADLevel2Archive().get_product('KTLX', datetime(2013, 5, 22, 21, 53))

# Open using MetPy's Level2File class
l2 = prod.access()

###################
# NEXRAD Level 3
start = datetime(2022, 10, 30, 15)
end = start + timedelta(hours=2)
products = NEXRADLevel3Archive().get_range('FTG', 'N0B', start, end)

# Get all the file names--could also get a file-like object or open with MetPy Level3File
print([prod.name for prod in products])

################
# GOES Archives
prod = GOESArchive(19).get_product('ABI-L1b-RadC', band=2)

# Retrieve using xarray + netcdf-c's S3 support
nc = prod.access()
