# Copyright (c) 2020 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `_tools` module."""

from metpy.io._tools import NamedStruct


def test_unpack():
    """Test unpacking a NamedStruct from bytes."""
    struct = NamedStruct([('field1', 'i'), ('field2', 'h')], '>')

    s = struct.unpack(b'\x00\x01\x00\x01\x00\x02')
    assert s.field1 == 65537
    assert s.field2 == 2


def test_pack():
    """Test packing a NamedStruct into bytes."""
    struct = NamedStruct([('field1', 'i'), ('field2', 'h')], '>')

    b = struct.pack(field1=8, field2=3)
    assert b == b'\x00\x00\x00\x08\x00\x03'
