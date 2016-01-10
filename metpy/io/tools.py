'A collection of general purpose tools for reading files'

# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import print_function
import logging
import zlib
from collections import namedtuple
from struct import Struct

log = logging.getLogger("metpy.io.tools")
log.setLevel(logging.WARNING)


class NamedStruct(Struct):
    def __init__(self, info, prefmt='', tuple_name=None):
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
        return self._tuple(*items)

    def unpack(self, s):
        return self._create(super(NamedStruct, self).unpack(s))

    def unpack_from(self, buff, offset=0):
        return self._create(super(NamedStruct, self).unpack_from(buff, offset))

    def unpack_file(self, fobj):
        bytes = fobj.read(self.size)
        return self.unpack(bytes)


# This works around times when we have more than 255 items and can't use
# NamedStruct. This is a CPython limit for arguments.
class DictStruct(Struct):
    def __init__(self, info, prefmt=''):
        names, formats = zip(*info)

        # Remove empty names
        self._names = [n for n in names if n]

        super(DictStruct, self).__init__(prefmt + ''.join(f for f in formats if f))

    def _create(self, items):
        return dict(zip(self._names, items))

    def unpack(self, s):
        return self._create(super(DictStruct, self).unpack(s))

    def unpack_from(self, buff, offset=0):
        return self._create(super(DictStruct, self).unpack_from(buff, offset))


class Enum(object):
    def __init__(self, *args, **kwargs):
        self.val_map = dict()
        # Assign values for args in order starting at 0
        for ind, a in enumerate(args):
            self.val_map[ind] = a

        # Invert the kwargs dict so that we can map from value to name
        for k in kwargs:
            self.val_map[kwargs[k]] = k

    def __call__(self, val):
        return self.val_map.get(val, 'Unknown ({})'.format(val))


class Bits(object):
    def __init__(self, num_bits):
        self._bits = range(num_bits)

    def __call__(self, val):
        return [bool((val >> i) & 0x1) for i in self._bits]


class BitField(object):
    def __init__(self, *names):
        self._names = names

    def __call__(self, val):
        if not val:
            return None

        l = []
        for n in self._names:
            if val & 0x1:
                l.append(n)
            val >>= 1
            if not val:
                break

        # Return whole list if empty or multiple items, otherwise just single item
        return l[0] if len(l) == 1 else l


class Array(object):
    def __init__(self, fmt):
        self._struct = Struct(fmt)

    def __call__(self, buf):
        return list(self._struct.unpack(buf))


class IOBuffer(object):
    def __init__(self, source):
        self._data = bytearray(source)
        self._offset = 0
        self.clear_marks()

    @classmethod
    def fromfile(cls, fobj):
        return cls(fobj.read())

    def set_mark(self):
        self._bookmarks.append(self._offset)
        return len(self._bookmarks) - 1

    def jump_to(self, mark, offset=0):
        self._offset = self._bookmarks[mark] + offset

    def offset_from(self, mark):
        return self._offset - self._bookmarks[mark]

    def clear_marks(self):
        self._bookmarks = []

    def splice(self, mark, newdata):
        self.jump_to(mark)
        self._data = self._data[:self._offset] + bytearray(newdata)

    def read_struct(self, struct_class):
        struct = struct_class.unpack_from(self._data, self._offset)
        self.skip(struct_class.size)
        return struct

    def read_func(self, func, num_bytes=None):
        # only advance if func succeeds
        res = func(self.get_next(num_bytes))
        self.skip(num_bytes)
        return res

    def read_ascii(self, num_bytes=None):
        return self.read(num_bytes).decode('ascii')

    def read_binary(self, num, item_type='B'):
        if 'B' in item_type:
            return self.read(num)

        if item_type[0] in ('@', '=', '<', '>', '!'):
            order = item_type[0]
            item_type = item_type[1:]
        else:
            order = '@'

        return list(self.read_struct(Struct(order + '%d' % num + item_type)))

    def read_int(self, code):
        return self.read_struct(Struct(code))[0]

    def read(self, num_bytes=None):
        res = self.get_next(num_bytes)
        self.skip(len(res))
        return res

    def get_next(self, num_bytes=None):
        if num_bytes is None:
            return self._data[self._offset:]
        else:
            return self._data[self._offset:self._offset + num_bytes]

    def skip(self, num_bytes):
        if num_bytes is None:
            self._offset = len(self._data)
        else:
            self._offset += num_bytes

    def check_remains(self, num_bytes):
        return len(self._data[self._offset:]) == num_bytes

    def truncate(self, num_bytes):
        self._data = self._data[:-num_bytes]

    def at_end(self):
        return self._offset >= len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __str__(self):
        return 'Size: {} Offset: {}'.format(len(self._data), self._offset)

    def print_next(self, num_bytes):
        print(' '.join('%02x' % c for c in self.get_next(num_bytes)))

    def __len__(self):
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
    if val == 8:
        return 'B'
    elif val == 16:
        return 'H'
    else:
        log.warning('Unsupported bit size: %s. Returning "B"', val)
        return 'B'


# For debugging
def hexdump(buf, num_bytes, offset=0, width=32):
    ind = offset
    end = offset + num_bytes
    while ind < end:
        chunk = buf[ind:ind + width]
        actual_width = len(chunk)
        hexfmt = '%02X'
        blocksize = 4
        blocks = [hexfmt * blocksize for _ in range(actual_width // blocksize)]

        # Need to get any partial lines
        num_left = actual_width % blocksize
        if num_left:
            blocks += [hexfmt * num_left + '--' * (blocksize - num_left)]
        blocks += ['--' * blocksize] * (width // blocksize - len(blocks))

        hexoutput = ' '.join(blocks)
        printable = tuple(chunk)
        print(hexoutput % printable, str(ind).ljust(len(str(end))),
              str(ind - offset).ljust(len(str(end))),
              ''.join(chr(c) if 31 < c < 128 else '.' for c in chunk), sep='  ')
        ind += width
