# Copyright (c) 2009,2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""A collection of general purpose tools for reading files."""

from __future__ import print_function

import bz2
from collections import namedtuple
import gzip
import logging
from struct import Struct
import zlib

from ..units import UndefinedUnitError, units

log = logging.getLogger(__name__)


# This works around problems on early Python 2.7 where Struct.unpack_from() can't handle
# being given a bytearray; use memoryview on Python 3, since calling bytearray again isn't
# cheap.
try:
    bytearray_to_buff = buffer
except NameError:
    bytearray_to_buff = memoryview


def open_as_needed(filename):
    """Return a file-object given either a filename or an object.

    Handles opening with the right class based on the file extension.

    """
    if hasattr(filename, 'read'):
        return filename

    if filename.endswith('.bz2'):
        return bz2.BZ2File(filename, 'rb')
    elif filename.endswith('.gz'):
        return gzip.GzipFile(filename, 'rb')
    else:
        return open(filename, 'rb')


class UnitLinker(object):
    r"""Wrap a :class:`metpy.io.cdm.Variable` and handle units.

    Converts any attached unit attribute to a class:`pint.Unit`. It also handles converting
    data returns to be instances of class:`pint.Quantity` rather than bare (unit-less) arrays.

    """

    def __init__(self, var):
        r"""Construct a new :class:`UnitLinker`.

        Parameters
        ----------
        var : Variable
            The :class:`metpy.io.cdm.Variable` to be wrapped.

        """
        self._var = var
        try:
            self._unit = units(self._var.units)
        except (AttributeError, UndefinedUnitError):
            self._unit = None

    def __getitem__(self, ind):
        """Get data from the underlying variable and add units."""
        ret = self._var[ind]
        return ret if self._unit is None else ret * self._unit

    def __getattr__(self, item):
        """Forward all attribute access onto underlying variable."""
        return getattr(self._var, item)

    @property
    def units(self):
        """Access the units from the underlying variable as a :class:`pint.Quantity`."""
        return self._unit

    @units.setter
    def units(self, val):
        """Override the units on the underlying variable."""
        if isinstance(val, units.Unit):
            self._unit = val
        else:
            self._unit = units(val)


class NamedStruct(Struct):
    """Parse bytes using :class:`Struct` but provide named fields."""

    def __init__(self, info, prefmt='', tuple_name=None):
        """Initialize the NamedStruct."""
        if tuple_name is None:
            tuple_name = 'NamedStruct'
        names, fmts = zip(*info)
        self.converters = {}
        conv_off = 0
        for ind, i in enumerate(info):
            if len(i) > 2:
                self.converters[ind - conv_off] = i[-1]
            elif not i[0]:  # Skip items with no name
                conv_off += 1
        self._tuple = namedtuple(tuple_name, ' '.join(n for n in names if n))
        super(NamedStruct, self).__init__(prefmt + ''.join(f for f in fmts if f))

    def _create(self, items):
        if self.converters:
            items = list(items)
            for ind, conv in self.converters.items():
                items[ind] = conv(items[ind])
            if len(items) < len(self._tuple._fields):
                items.extend([None] * (len(self._tuple._fields) - len(items)))
        return self.make_tuple(*items)

    def make_tuple(self, *args, **kwargs):
        """Construct the underlying tuple from values."""
        return self._tuple(*args, **kwargs)

    def unpack(self, s):
        """Parse bytes and return a namedtuple."""
        return self._create(super(NamedStruct, self).unpack(s))

    def unpack_from(self, buff, offset=0):
        """Read bytes from a buffer and return as a namedtuple."""
        return self._create(super(NamedStruct, self).unpack_from(buff, offset))

    def unpack_file(self, fobj):
        """Unpack the next bytes from a file object."""
        return self.unpack(fobj.read(self.size))


# This works around times when we have more than 255 items and can't use
# NamedStruct. This is a CPython limit for arguments.
class DictStruct(Struct):
    """Parse bytes using :class:`Struct` but provide named fields using dictionary access."""

    def __init__(self, info, prefmt=''):
        """Initialize the DictStruct."""
        names, formats = zip(*info)

        # Remove empty names
        self._names = [n for n in names if n]

        super(DictStruct, self).__init__(prefmt + ''.join(f for f in formats if f))

    def _create(self, items):
        return dict(zip(self._names, items))

    def unpack(self, s):
        """Parse bytes and return a namedtuple."""
        return self._create(super(DictStruct, self).unpack(s))

    def unpack_from(self, buff, offset=0):
        """Unpack the next bytes from a file object."""
        return self._create(super(DictStruct, self).unpack_from(buff, offset))


class Enum(object):
    """Map values to specific strings."""

    def __init__(self, *args, **kwargs):
        """Initialize the mapping."""
        # Assign values for args in order starting at 0
        self.val_map = {ind: a for ind, a in enumerate(args)}

        # Invert the kwargs dict so that we can map from value to name
        self.val_map.update(zip(kwargs.values(), kwargs.keys()))

    def __call__(self, val):
        """Map an integer to the string representation."""
        return self.val_map.get(val, 'Unknown ({})'.format(val))


class Bits(object):
    """Breaks an integer into a specified number of True/False bits."""

    def __init__(self, num_bits):
        """Initialize the number of bits."""
        self._bits = range(num_bits)

    def __call__(self, val):
        """Convert the integer to the list of True/False values."""
        return [bool((val >> i) & 0x1) for i in self._bits]


class BitField(object):
    """Convert an integer to a string for each bit."""

    def __init__(self, *names):
        """Initialize the list of named bits."""
        self._names = names

    def __call__(self, val):
        """Return a list with a string for each True bit in the integer."""
        if not val:
            return None

        bits = []
        for n in self._names:
            if val & 0x1:
                bits.append(n)
            val >>= 1
            if not val:
                break

        # Return whole list if empty or multiple items, otherwise just single item
        return bits[0] if len(bits) == 1 else bits


class Array(object):
    """Use a Struct as a callable to unpack a bunch of bytes as a list."""

    def __init__(self, fmt):
        """Initialize the Struct unpacker."""
        self._struct = Struct(fmt)

    def __call__(self, buf):
        """Perform the actual unpacking."""
        return list(self._struct.unpack(buf))


class IOBuffer(object):
    """Holds bytes from a buffer to simplify parsing and random access."""

    def __init__(self, source):
        """Initialize the IOBuffer with the source data."""
        self._data = bytearray(source)
        self._offset = 0
        self.clear_marks()

    @classmethod
    def fromfile(cls, fobj):
        """Initialize the IOBuffer with the contents of the file object."""
        return cls(fobj.read())

    def set_mark(self):
        """Mark the current location and return its id so that the buffer can return later."""
        self._bookmarks.append(self._offset)
        return len(self._bookmarks) - 1

    def jump_to(self, mark, offset=0):
        """Jump to a previously set mark."""
        self._offset = self._bookmarks[mark] + offset

    def offset_from(self, mark):
        """Calculate the current offset relative to a marked location."""
        return self._offset - self._bookmarks[mark]

    def clear_marks(self):
        """Clear all marked locations."""
        self._bookmarks = []

    def splice(self, mark, newdata):
        """Replace the data after the marked location with the specified data."""
        self.jump_to(mark)
        self._data = self._data[:self._offset] + bytearray(newdata)

    def read_struct(self, struct_class):
        """Parse and return a structure from the current buffer offset."""
        struct = struct_class.unpack_from(bytearray_to_buff(self._data), self._offset)
        self.skip(struct_class.size)
        return struct

    def read_func(self, func, num_bytes=None):
        """Parse data from the current buffer offset using a function."""
        # only advance if func succeeds
        res = func(self.get_next(num_bytes))
        self.skip(num_bytes)
        return res

    def read_ascii(self, num_bytes=None):
        """Return the specified bytes as ascii-formatted text."""
        return self.read(num_bytes).decode('ascii')

    def read_binary(self, num, item_type='B'):
        """Parse the current buffer offset as the specified code."""
        if 'B' in item_type:
            return self.read(num)

        if item_type[0] in ('@', '=', '<', '>', '!'):
            order = item_type[0]
            item_type = item_type[1:]
        else:
            order = '@'

        return list(self.read_struct(Struct(order + '{:d}'.format(int(num)) + item_type)))

    def read_int(self, code):
        """Parse the current buffer offset as the specified integer code."""
        return self.read_struct(Struct(code))[0]

    def read(self, num_bytes=None):
        """Read and return the specified bytes from the buffer."""
        res = self.get_next(num_bytes)
        self.skip(len(res))
        return res

    def get_next(self, num_bytes=None):
        """Get the next bytes in the buffer without modifying the offset."""
        if num_bytes is None:
            return self._data[self._offset:]
        else:
            return self._data[self._offset:self._offset + num_bytes]

    def skip(self, num_bytes):
        """Jump the ahead the specified bytes in the buffer."""
        if num_bytes is None:
            self._offset = len(self._data)
        else:
            self._offset += num_bytes

    def check_remains(self, num_bytes):
        """Check that the number of bytes specified remains in the buffer."""
        return len(self._data[self._offset:]) == num_bytes

    def truncate(self, num_bytes):
        """Remove the specified number of bytes from the end of the buffer."""
        self._data = self._data[:-num_bytes]

    def at_end(self):
        """Return whether the buffer has reached the end of data."""
        return self._offset >= len(self._data)

    def __getitem__(self, item):
        """Return the data at the specified location."""
        return self._data[item]

    def __str__(self):
        """Return a string representation of the IOBuffer."""
        return 'Size: {} Offset: {}'.format(len(self._data), self._offset)

    def __len__(self):
        """Return the amount of data in the buffer."""
        return len(self._data)


def zlib_decompress_all_frames(data):
    """Decompress all frames of zlib-compressed bytes.

    Repeatedly tries to decompress `data` until all data are decompressed, or decompression
    fails. This will skip over bytes that are not compressed with zlib.

    Parameters
    ----------
    data : bytearray or bytes
        Binary data compressed using zlib.

    Returns
    -------
        bytearray
            All decompressed bytes

    """
    frames = bytearray()
    data = bytes(data)
    while data:
        decomp = zlib.decompressobj()
        try:
            frames.extend(decomp.decompress(data))
            data = decomp.unused_data
        except zlib.error:
            frames.extend(data)
            break
    return frames


def bits_to_code(val):
    """Convert the number of bits to the proper code for unpacking."""
    if val == 8:
        return 'B'
    elif val == 16:
        return 'H'
    else:
        log.warning('Unsupported bit size: %s. Returning "B"', val)
        return 'B'


# For debugging
def hexdump(buf, num_bytes, offset=0, width=32):
    """Perform a hexudmp of the buffer.

    Returns the hexdump as a canonically-formatted string.
    """
    ind = offset
    end = offset + num_bytes
    lines = []
    while ind < end:
        chunk = buf[ind:ind + width]
        actual_width = len(chunk)
        hexfmt = '{:02X}'
        blocksize = 4
        blocks = [hexfmt * blocksize for _ in range(actual_width // blocksize)]

        # Need to get any partial lines
        num_left = actual_width % blocksize  # noqa: S001  Fix false alarm
        if num_left:
            blocks += [hexfmt * num_left + '--' * (blocksize - num_left)]
        blocks += ['--' * blocksize] * (width // blocksize - len(blocks))

        hexoutput = ' '.join(blocks)
        printable = tuple(chunk)
        lines.append('  '.join((hexoutput.format(*printable), str(ind).ljust(len(str(end))),
                                str(ind - offset).ljust(len(str(end))),
                                ''.join(chr(c) if 31 < c < 128 else '.' for c in chunk))))
        ind += width
    return '\n'.join(lines)
