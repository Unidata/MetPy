import bz2
import datetime
import gzip
import re
import struct
import warnings
import zlib
from struct import Struct
from cStringIO import StringIO
from collections import defaultdict

import numpy as np
from scipy.constants import day, milli
from collections import namedtuple
from metpy.cbook import is_string_like

class NamedStruct(Struct):
    def __init__(self, info, prefmt='', tuple_name=None):
        if tuple_name is None:
            tuple_name = 'NamedStruct'
        names, fmts = zip(*info)
        self.converters = {}
        conv_off = 0
        for ind,i in enumerate(info):
            if len(i) > 2:
                self.converters[ind - conv_off] = i[-1]
            elif not i[0]: # Skip items with no name
                conv_off += 1
        self._tuple = namedtuple(tuple_name, ' '.join(n for n in names if n))
        Struct.__init__(self, prefmt + ''.join(fmts))

    def _create(self, items):
        if self.converters:
            items = list(items)
            for ind,conv in self.converters.items():
                items[ind] = conv(items[ind])
        return self._tuple(*items)

    def unpack(self, s):
        return self._create(Struct.unpack(self, s))

    def unpack_from(self, buff, offset=0):
        return self._create(Struct.unpack_from(self, buff, offset))

    def unpack_file(self, fobj):
        bytes = fobj.read(self.size)
        return self.unpack(bytes)

class Bits(object):
    def __init__(self, num_bits):
        self._bits = range(num_bits)

    def __call__(self, val):
        return [bool((val>>i) & 0x1) for i in self._bits]

class BitField(Bits):
    def __init__(self, *names):
        Bits.__init__(self, len(names))
        self._names = names

    def __call__(self, val):
        if not val: return None

        l = []
        for n in self._names:
            if val & 0x1:
                l.append(n)
            val = val >> 1
            if not val:
                break

        return l if len(l) > 1 else l[0]

def version(val):
    return '{:.1f}'.format(val / 10.)

def scaler(scale):
    def inner(val):
        return val * scale
    return inner


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
        return str(self.read(num_bytes))

    def read_binary(self, num, type='B'):
        if 'B' in type:
            return self.read(num)

        if type[0] in ('@', '=', '<', '>', '!'):
            order = type[0]
            type = type[1:]
        else:
            order = '@'

        return list(self.read_struct(Struct(order + '%d' % num + type)))

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
        print ' '.join('%02x' % ord(c) for c in self.get_next(num_bytes))

    def __len__(self):
        return len(self._data)


def bzip_blocks_decompress_all(data):
    frames = []
    offset = 0
    while offset < len(data):
        size_bytes = data[offset:offset + 4]
        offset += 4
        block_cmp_bytes = abs(Struct('>l').unpack(size_bytes)[0])
        if block_cmp_bytes:
            frames.append(bz2.decompress(data[offset:offset+block_cmp_bytes]))
            offset += block_cmp_bytes
        else:
            frames.append(size_bytes)
            frames.append(data[offset:])
    return ''.join(frames)


def nexrad_to_datetime(julian_date, ms_midnight):
    #Subtracting one from julian_date is because epoch date is 1
    return datetime.datetime.fromtimestamp((julian_date - 1) * day
        + ms_midnight * milli)

def bits_to_code(val):
    if val == 8:
        return 'B'
    elif val == 16:
        return 'H'
    else:
        warnings.warn('Unsuported bit size: %s' % val)
        return 'B'

START_ELEVATION = 0x1
END_ELEVATION = 0x2
START_VOLUME = 0x4
END_VOLUME = 0x8
LAST_ELEVATION = 0x10
BAD_DATA = 0x20
def remap_status(val):
    bad = BAD_DATA if val & 0xF0 else 0
    val = val & 0x0F
    if val == 0:
        status = START_ELEVATION
    elif val == 1:
        status = 0
    elif val == 2:
        status = END_ELEVATION
    elif val == 3:
        status = START_ELEVATION | START_VOLUME
    elif val == 4:
        status = END_ELEVATION | END_VOLUME
    elif val == 5:
        status = START_ELEVATION | LAST_ELEVATION

    return status | bad

class Level2File(object):
    #Number of bytes
    AR2_BLOCKSIZE = 2432
    CTM_HEADER_SIZE = 12
    def __init__(self, filename):
        if is_string_like(filename):
            if filename.endswith('.bz2'):
                fobj = bz2.BZ2File(filename, 'rb')
            elif filename.endswith('.gz'):
                fobj = gzip.GzipFile(filename, 'rb')
            else:
                fobj = file(filename, 'rb')
        else:
            fobj = filename

        self._buffer = IOBuffer.fromfile(fobj)
        self._read_volume_header()

        self._buffer = IOBuffer(self._buffer.read_func(bzip_blocks_decompress_all))

        #Now we're all initialized, we can proceed with reading in data
        self._read_data()

    vol_hdr_fmt = NamedStruct([('version', '9s'), ('vol_num', '3s'),
        ('date', 'L'), ('time_ms', 'L'), ('stid', '4s')], '>', 'VolHdr')

    def _read_volume_header(self):
        self.vol_hdr = self._buffer.read_struct(self.vol_hdr_fmt)
        print self.vol_hdr
        self.dt = nexrad_to_datetime(self.vol_hdr.date, self.vol_hdr.time_ms)
        self.stid = self.vol_hdr.stid
        print self.dt

    msg_hdr_fmt = NamedStruct([('size_hw', 'H'), ('rda_channel', 'B'),
        ('msg_type', 'B'), ('seq_num', 'H'), ('date', 'H'), ('time_ms', 'I'),
        ('num_segments', 'H'), ('segment_num', 'H')], '>', 'MsgHdr')

    def _read_data(self):
        self.sweeps = []
        while not self._buffer.at_end():
            # Clear old file book marks and set the start of message for
            # easy jumping to the end
            self._buffer.clear_marks()
            msg_start = self._buffer.set_mark()

            # Skip CTM
            self._buffer.skip(self.CTM_HEADER_SIZE)

            # Read the message header
            msg_hdr = self._buffer.read_struct(self.msg_hdr_fmt)

            # If the size is 0, this is just padding, which is for certain
            # done in the metadata messages. Just handle generally here
            if msg_hdr.size_hw:
                print msg_hdr, str(nexrad_to_datetime(msg_hdr.date, msg_hdr.time_ms))

                # Try to handle the message. If we don't handle it, skipping
                # past it is handled at the end anyway.
                try:
                    getattr(self, '_decode_msg%d' % msg_hdr.msg_type)(msg_hdr)
                except AttributeError:
                    warnings.warn("Unknown message: {0.msg_type}".format(msg_hdr))

            # Jump to the start of the next message. This depends on whether
            # the message was legacy with fixed block size or not.
            if msg_hdr.msg_type != 31:
                # The AR2_BLOCKSIZE includes accounts for the CTM header
                self._buffer.jump_to(msg_start, self.AR2_BLOCKSIZE)
            else:
                # Need to include the CTM header
                self._buffer.jump_to(msg_start,
                        self.CTM_HEADER_SIZE + 2 * msg_hdr.size_hw)

    msg31_data_hdr_fmt = NamedStruct([('stid', '4s'), ('time_ms', 'L'),
        ('date', 'H'), ('az_num', 'H'), ('az_angle', 'f'),
        ('compression', 'B'), (None, 'x'), ('rad_length', 'H'),
        ('az_spacing', 'B'), ('rad_status', 'B', remap_status), ('el_num', 'B'),
        ('sector_num', 'B'), ('el_angle', 'f'),
        ('spot_blanking', 'B', BitField('Radial', 'Elevation', 'Volume')),
        ('az_index_mode', 'B', scaler(0.01)), ('num_data_blks', 'H'), ('vol_const_ptr', 'L'),
        ('el_const_ptr', 'L'), ('rad_const_ptr', 'L')], '>', 'Msg31DataHdr')

    msg31_vol_const_fmt = NamedStruct([('type', 's'), ('name', '3s'),
        ('size', 'H'), ('major', 'B'), ('minor', 'B'), ('lat', 'f'),
        ('lon', 'f'), ('site_amsl', 'h'), ('feedhorn_agl', 'H'),
        ('calib_dbz', 'f'), ('txpower_h', 'f'), ('txpower_v', 'f'),
        ('sys_zdr', 'f'), ('phidp0', 'f'), ('vcp', 'H'), (None, '2x')],
        '>', 'VolConsts')

    msg31_el_const_fmt = NamedStruct([('type', 's'), ('name', '3s'),
        ('size', 'H'), ('atmos_atten', 'h', scaler(0.001)),
        ('calib_dbz0', 'f')], '>', 'ElConsts')

    rad_const_fmt = NamedStruct([('type', 's'), ('name', '3s'), ('size', 'H'),
        ('unamb_range', 'H', scaler(0.1)), ('noise_h', 'f'), ('noise_v', 'f'),
        ('nyq_vel', 'H', scaler(0.01)), (None, '2x')], '>', 'RadConsts')

    data_block_fmt = NamedStruct([('type', 's'), ('name', '3s'),
        ('reserved', 'L'), ('num_gates', 'H'),
        ('first_range_gate', 'H', scaler(0.001)),
        ('gate_width', 'H', scaler(0.001)), ('tover', 'H', scaler(0.1)),
        ('snr_thresh', 'h', scaler(0.1)),
        ('recombined', 'B', BitField('Azimuths', 'Gates')),
        ('data_size', 'B', bits_to_code), ('scale', 'f'), ('offset', 'f')],
        '>', 'DataBlockHdr')

    def _decode_msg31(self, msg_hdr):
        msg_start = self._buffer.set_mark()
        data_hdr = self._buffer.read_struct(self.msg31_data_hdr_fmt)
        print data_hdr

        # Read all the data block pointers separately. This simplifies just
        # iterating over them
        ptrs = self._buffer.read_binary(6, '>L')

        assert data_hdr.compression == 0, 'Compressed message 31 not supported!'

        self._buffer.jump_to(msg_start, data_hdr.vol_const_ptr)
        vol_consts = self._buffer.read_struct(self.msg31_vol_const_fmt)
        print vol_consts

        self._buffer.jump_to(msg_start, data_hdr.el_const_ptr)
        el_consts = self._buffer.read_struct(self.msg31_el_const_fmt)
        print el_consts

        self._buffer.jump_to(msg_start, data_hdr.rad_const_ptr)
        rad_consts = self._buffer.read_struct(self.rad_const_fmt)
        print rad_consts

        data = dict()
        block_count = 3
        for ptr in ptrs:
            if ptr:
                block_count += 1
                self._buffer.jump_to(msg_start, ptr)
                hdr = self._buffer.read_struct(self.data_block_fmt)
                vals = self._buffer.read_binary(hdr.num_gates,
                        '>' + hdr.data_size)
                vals = (np.array(vals) - hdr.offset) / hdr.scale
                print hdr, len(vals)
                data[hdr.name] = (hdr, vals)

        if not self.sweeps and not data_hdr.rad_status & START_VOLUME:
            warnings.warn('Missed start of volume!')

        if data_hdr.rad_status & START_ELEVATION:
            self.sweeps.append([])

        if len(self.sweeps) != data_hdr.el_num:
            warnings.warn('Missed elevation -- Have %d but data on %d.'
                    ' Compensating...' % (len(self.sweeps), data_hdr.el_num))
            self.sweeps.append([])

        self.sweeps[-1].append((data_hdr, vol_consts, el_consts, rad_consts, data))

        if data_hdr.num_data_blks != block_count:
            warnings.warn('Incorrect number of blocks detected -- Got %d'
                    'instead of %d' % (block_count, data_hdr.num_data_blks))
        assert data_hdr.rad_length == self._buffer.offset_from(msg_start)

    def _decode_msg1(self, msg_hdr):
        pass


def reduce_lists(d):
    for field in d:
        old_data = d[field]
        if len(old_data) == 1:
            d[field] = old_data[0]

def two_comp16(val):
    if val>>15:
        val =  -(~val & 0x7fff) - 1
    return val

def float16(val):
    # Fraction is 10 LSB, Exponent middle 5, and Sign the MSB
    frac = val & 0x03ff
    exp = (val >> 10) & 0x1F
    sign = val >> 15

    if exp:
        value = 2 ** (exp - 16) * (1 + float(frac) / 2**10)
    else:
        value = float(frac) / 2**9

    if sign:
        value *= -1

    return value

def float32(short1, short2):
    return struct.unpack('>f', struct.pack('>hh', short1, short2))[0]

def date_elem(ind_days, ind_minutes):
    def inner(seq):
        return nexrad_to_datetime(seq[ind_days], seq[ind_minutes] * 60 * 1000)
    return inner

def scaled_elem(index, scale):
    def inner(seq):
        return seq[index] * scale
    return inner

def combine_elem(ind1, ind2):
    def inner(seq):
        shift = 2**16
        if seq[ind1] < 0:
            seq[ind1] += shift
        if seq[ind2] < 0:
            seq[ind2] += shift
        return (seq[ind1] << 16) | seq[ind2]
    return inner

def float_elem(ind1, ind2):
    return lambda seq: float32(seq[ind1], seq[ind2])

def high_byte(ind):
    def inner(seq):
        return seq[ind] >> 8
    return inner

def low_byte(ind):
    def inner(seq):
        return seq[ind] & 0x00FF
    return inner

def zlib_decompress_all_frames(data):
    frames = []
    while data:
        decomp = zlib.decompressobj()
        try:
            frames.append(decomp.decompress(data))
        except zlib.error:
            break
        data = decomp.unused_data
    return ''.join(frames) + data

# Data mappers used to take packed data and turn into physical units
# Default is to use numpy array indexing to use LUT to change data bytes
# into physical values. Can also have a 'labels' attribute to give 
# categorical labels
class DataMapper(object):
    # Need to find way to handle range folded
    #RANGE_FOLD = -9999
    RANGE_FOLD = float('nan')
    MISSING = float('nan')

    def __call__(self, data):
        return self.lut[data]

class DigitalMapper(DataMapper):
    _min_scale = 0.1
    _inc_scale = 0.1
    _min_data = 2
    _max_data = 255
    range_fold = False
    def __init__(self, prod):
        min_val = two_comp16(prod.thresholds[0]) * self._min_scale
        inc = prod.thresholds[1] * self._inc_scale
        num_levels = prod.thresholds[2]
        self.lut = [self.MISSING] * 256

        # Generate lookup table -- sanity check on num_levels handles
        # the fact that DHR advertises 256 levels, which *includes*
        # missing, differing from other products
        num_levels = min(num_levels, self._max_data - self._min_data + 1)
        for i in range(num_levels):
            self.lut[i + self._min_data] = min_val + i * inc

        self.lut = np.array(self.lut)

class DigitalRefMapper(DigitalMapper):
    units = 'dBZ'

class DigitalVelMapper(DigitalMapper):
    units = 'm/s'
    range_fold = True

class DigitalSPWMapper(DigitalVelMapper):
    _min_data = 129
    _max_data = 149

class PrecipArrayMapper(DigitalMapper):
    _inc_scale = 0.001
    _min_data = 1
    _max_data = 254
    units = 'dBA'

class DigitalStormPrecipMapper(DigitalMapper):
    units = 'inches'
    _inc_scale = 0.01

class DigitalVILMapper(DataMapper):
    def __init__(self, prod):
        lin_scale = float16(prod.thresholds[0])
        lin_offset = float16(prod.thresholds[1])
        log_start = prod.thresholds[2]
        log_scale = float16(prod.thresholds[3])
        log_offset = float16(prod.thresholds[4])
        self.lut = np.empty((256,), dtype=np.float)
        self.lut.fill(self.MISSING)

        # VIL is allowed to use 2 through 254 inclusive. 0 is thresholded,
        # 1 is flagged, and 255 is reserved
        ind = np.arange(255)
        self.lut[2:log_start] = (ind[2:log_start] - lin_offset) / lin_scale
        self.lut[log_start:-1] = np.exp((ind[log_start:] - log_offset) / log_scale)

class DigitalEETMapper(DataMapper):
    def __init__(self, prod):
        data_mask = prod.thresholds[0]
        scale = prod.thresholds[1]
        offset = prod.thresholds[2]
        topped_mask = prod.thresholds[3]
        self.lut = [self.MISSING] * 256
        self.topped_lut = [False] * 256
        for i in range(2, 256):
            self.lut[i] = ((i & data_mask) - offset) / scale
            self.topped_lut[i] = bool(i & topped_mask)

        self.lut = np.array(self.lut)
        self.topped_lut = np.array(self.topped_lut)

    def __call__(self, data_vals):
        return self.lut[data_vals], self.topped_lut[data_vals]

class GenericDigitalMapper(DataMapper):
    def __init__(self, prod):
        scale = float32(prod.thresholds[0], prod.thresholds[1])
        offset = float32(prod.thresholds[2], prod.thresholds[3])
        max_data_val = prod.thresholds[5]
        leading_flags = prod.thresholds[6]
        trailing_flags = prod.thresholds[7]
        self.lut = [self.MISSING] * max_data_val

        if leading_flags > 1:
            self.lut[1] = self.RANGE_FOLD

        for i in range(leading_flags, max_data_val - trailing_flags):
            self.lut[i] = (i - offset) / scale

        self.lut = np.array(self.lut)

class DigitalHMCMapper(DataMapper):
    labels = ['ND', 'BI', 'GC', 'IC', 'DS', 'WS', 'RA', 'HR',
              'BD', 'GR', 'HA', 'UK', 'RF']
    def __init__(self, prod):
        self.lut = [self.MISSING] * 256
        for i in range(10, 256):
            self.lut[i] = i // 10
        self.lut[150] = self.RANGE_FOLD
        self.lut = np.array(self.lut)

#156, 157
class EDRMapper(DataMapper):
    def __init__(self, prod):
        scale = prod.thresholds[0] / 1000.
        offset = prod.thresholds[1] / 1000.
        data_levels = prod.thresholds[2]
        leading_flags = prod.thresholds[3]
        self.lut = [self.MISSING] * data_levels
        for i in range(leading_flags, data_levels):
            self.lut = scale * i + offset
        self.lut = np.array(self.lut)

class LegacyMapper(DataMapper):
    lut_names = ['Blank', 'TH', 'ND', 'RF', 'BI', 'GC', 'IC', 'GR', 'WS',
                 'DS', 'RA', 'HR', 'BD', 'HA', 'UK']
    def __init__(self, prod):
        self.labels = []
        self.lut = []
        for t in prod.thresholds:
            codes,val = t>>8, t & 0xFF
            label=''
            if codes>>7:
                label = self.lut_names[val]
                if label in ('Blank', 'TH', 'ND'):
                    val = self.MISSING
                elif label == 'RF':
                    val = self.RANGE_FOLD

            elif codes>>6:
                val *= 0.01
                label = '%.2f' % val
            elif codes>>5:
                val *= 0.05
                label = '%.2f' % val
            elif codes>>4:
                val *= 0.1
                label = '%.1f' % val

            if codes & 0x1:
                val *= -1
                label = '-' + label
            elif (codes >> 1) & 0x1:
                label = '+' + label

            if (codes >> 2) & 0x1:
                label = '<' + label
            elif (codes >> 3) & 0x1:
                label = '>' + label

            if not label:
                label = str(val)

            self.lut.append(val)
            self.labels.append(label)
        self.lut = np.array(self.lut)

class Level3File(object):
    ij_to_km = 0.25
    wmo_finder = re.compile('((?:NX|SD|NO)US)\d{2}[\s\w\d]+\w*(\w{3})\r\r\n')
    header_fmt = NamedStruct([('code', 'H'), ('date', 'H'), ('time', 'l'),
        ('msg_len', 'L'), ('src_id', 'h'), ('dest_id', 'h'),
        ('num_blks', 'H')], '>', 'MsgHdr')
    # See figure 3-17 in 2620001 document for definition of status bit fields
    gsm_fmt = NamedStruct([('divider', 'h'), ('block_len', 'H'),
        ('op_mode', 'h', BitField('Clear Air', 'Precip')),
        ('rda_op_status', 'h', BitField('Spare', 'Online', 'Maintenance Required',
            'Maintenance Mandatory', 'Commanded Shutdown', 'Inoperable',
            'Spare', 'Wideband Disconnect')),
        ('vcp', 'h'), ('num_el', 'h'),
        ('el1', 'h', scaler(0.1)), ('el2', 'h', scaler(0.1)),
        ('el3', 'h', scaler(0.1)), ('el4', 'h', scaler(0.1)),
        ('el5', 'h', scaler(0.1)), ('el6', 'h', scaler(0.1)),
        ('el7', 'h', scaler(0.1)), ('el8', 'h', scaler(0.1)),
        ('el9', 'h', scaler(0.1)), ('el10', 'h', scaler(0.1)),
        ('el11', 'h', scaler(0.1)), ('el12', 'h', scaler(0.1)),
        ('el13', 'h', scaler(0.1)), ('el14', 'h', scaler(0.1)),
        ('el15', 'h', scaler(0.1)), ('el16', 'h', scaler(0.1)),
        ('el17', 'h', scaler(0.1)), ('el18', 'h', scaler(0.1)),
        ('el19', 'h', scaler(0.1)), ('el20', 'h', scaler(0.1)),
        ('rda_status', 'h', BitField('Spare', 'Startup', 'Standby', 'Restart', 'Operate', 'Off-line Operate')),
        ('rda_alarms', 'h', BitField('Indeterminate', 'Tower/Utilities',
            'Pedestal', 'Transmitter', 'Receiver', 'RDA Control', 'RDA Communications',
            'Signal Processor')),
        ('tranmission_enable', 'h', BitField('Spare', 'None', 'Reflectivity',
            'Velocity', 'Spectrum Width', 'Dual Pol')),
        ('rpg_op_status', 'h', BitField('Loadshed', 'Online', 'Maintenance Required',
            'Maintenance Mandatory', 'Commanded shutdown')),
        ('rpg_alarms', 'h', BitField('None', 'Node Connectivity', 'Wideband Failure',
            'RPG Control Task Failure', 'Data Base Failure', 'Spare',
            'RPG Input Buffer Loadshed', 'Spare', 'Product Storage Loadshed'
            'Spare', 'Spare', 'Spare', 'RPG/RPG Intercomputer Link Failure',
            'Redundant Channel Error', 'Task Failure', 'Media Failure')),
        ('rpg_status', 'h', BitField('Restart', 'Operate', 'Standby')),
        ('rpg_narrowband_status', 'h', BitField('Commanded Disconnect', 'Narrowband Loadshed')),
        ('h_ref_calib', 'h', scaler(0.25)),
        ('prod_avail', 'h', BitField('Product Availability', 'Degraded Availability', 'Not Available')),
        ('super_res_cuts', 'h', Bits(16)), ('cmd_status', 'h', Bits(6)),
        ('v_ref_calib', 'h', scaler(0.25)),
        ('rda_build', 'h', version), ('rda_channel', 'h'), ('reserved', 'h'),
        ('reserved2', 'h'), ('build_version', 'h', version)], '>', 'GSM')
    # Build 14.0 added more bytes to the GSM
    additional_gsm_fmt = NamedStruct([('el21', 'h', scaler(0.1)),
        ('el22', 'h', scaler(0.1)), ('el23', 'h', scaler(0.1)),
        ('el24', 'h', scaler(0.1)), ('el25', 'h', scaler(0.1)),
        ('vcp_supplemental', 'H', BitField('AVSET', 'SAILS', 'site_vcp')),
        ('spare', '84s')], '>', 'GSM')
    prod_desc_fmt = NamedStruct([('divider', 'h'), ('lat', 'l'), ('lon', 'l'),
        ('height', 'h'), ('prod_code', 'h'), ('op_mode', 'h'),
        ('vcp', 'h'), ('seq_num', 'H'), ('vol_num', 'H'),
        ('vol_date', 'H'), ('vol_start_time', 'l'), ('prod_gen_date', 'H'),
        ('prod_gen_time', 'l'), ('dep1', 'h'), ('dep2', 'h'), ('el_num', 'H'),
        ('dep3', 'h'), ('thr1', 'H'), ('thr2', 'H'), ('thr3', 'H'),
        ('thr4', 'H'), ('thr5', 'H'), ('thr6', 'H'), ('thr7', 'H'),
        ('thr8', 'H'), ('thr9', 'H'), ('thr10', 'H'), ('thr11', 'H'),
        ('thr12', 'H'), ('thr13', 'H'), ('thr14', 'H'), ('thr15', 'H'),
        ('thr16', 'H'), ('dep4', 'h'), ('dep5', 'h'), ('dep6', 'h'),
        ('dep7', 'h'), ('dep8', 'h'), ('dep9', 'h'), ('dep10', 'h'),
        ('version', 'b'), ('spot_blank', 'b'), ('sym_off', 'L'), ('graph_off', 'L'),
        ('tab_off', 'L')], '>', 'ProdDesc')
    sym_block_fmt = NamedStruct([('divider', 'h'), ('block_id', 'h'),
        ('block_len', 'L'), ('nlayer', 'H')], '>', 'SymBlock')
    tab_header_fmt = NamedStruct([('divider', 'h'), ('block_id', 'h'), ('block_len', 'L')], '>', 'TabHeader')
    tab_block_fmt = NamedStruct([('divider', 'h'), ('num_pages', 'h')], '>', 'TabBlock')
    sym_layer_fmt = NamedStruct([('divider', 'h'), ('length', 'L')], '>',
        'SymLayer')
    graph_block_fmt = NamedStruct([('divider', 'h'), ('block_id', 'h'),
        ('block_len', 'L'), ('num_pages', 'H')], '>', 'GraphBlock')
    standalone_tabular = [62, 73, 75, 82]
    prod_spec_map = {16  : ('Base Reflectivity', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     17  : ('Base Reflectivity', 460., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     18  : ('Base Reflectivity', 460., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     19  : ('Base Reflectivity', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     20  : ('Base Reflectivity', 460., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     21  : ('Base Reflectivity', 460., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     22  : ('Base Velocity', 60., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4))),
                     23  : ('Base Velocity', 115., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4))),
                     24  : ('Base Velocity', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4))),
                     25  : ('Base Velocity', 60., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4))),
                     26  : ('Base Velocity', 115., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4))),
                     27  : ('Base Velocity', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4))),
                     28  : ('Base Spectrum Width', 60., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3))),
                     29  : ('Base Spectrum Width', 115., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3))),
                     30  : ('Base Spectrum Width', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3))),
                     31  : ('User Selectable Storm Total Precipitation', 230., LegacyMapper,
                            (('end_hour', 0), ('hour_span', 1), ('null_product', 2),
                             ('max_rainfall', scaled_elem(3, 0.1)), ('rainfall_begin', date_elem(4, 5)),
                             ('rainfall_end', date_elem(6, 7)), ('bias', scaled_elem(8, 0.01)),
                             ('gr_pairs', scaled_elem(5, 0.01)))),
                     32  : ('Digital Hybrid Scan Reflectivity', 230., DigitalRefMapper,
                            (('max', 3), ('avg_time', date_elem(4, 5)), ('compression', 7),
                             ('uncompressed_size', combine_elem(8, 9)))),
                     33  : ('Hybrid Scan Reflectivity', 230., LegacyMapper,
                            (('max', 3), ('avg_time', date_elem(4, 5)))),
                     34  : ('Clutter Filter Control', 230., LegacyMapper,
                            (('clutter_bitmap', 0), ('cmd_map', 1), ('bypass_map_date', date_elem(4, 5)),
                             ('notchwidth_map_date', date_elem(6, 7)))),
                     35  : ('Composite Reflectivity', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     36  : ('Composite Reflectivity', 460., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     37  : ('Composite Reflectivity', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     38  : ('Composite Reflectivity', 460., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     41  : ('Echo Tops', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', scaled_elem(3, 1000)))), # Max in ft
                     48  : ('VAD Wind Profile', None, LegacyMapper,
                            (('max', 3), ('dir_max', 4), ('alt_max', scaled_elem(5, 10)))), # Max in ft
                     55  : ('Storm Relative Mean Radial Velocity', 50., LegacyMapper,
                            (('window_az', scaled_elem(0, 0.1)), ('window_range', scaled_elem(1, 0.1)),
                             ('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4),
                             ('source', 5), ('height', 6), ('avg_speed', scaled_elem(7, 0.1)),
                             ('avg_dir', scaled_elem(8, 0.1)), ('alert_category', 9))),
                     56  : ('Storm Relative Mean Radial Velocity', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4),
                             ('source', 5), ('avg_speed', scaled_elem(7, 0.1)), ('avg_dir', scaled_elem(8, 0.1)))),
                     57  : ('Vertically Integrated Liquid', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3))), # Max in kg / m^2
                     58  : ('Storm Tracking Information', 460., LegacyMapper, (('num_storms', 3),)),
                     59  : ('Hail Index', 230., LegacyMapper, ()),
                     61  : ('Tornado Vortex Signature', 230., LegacyMapper, (('num_tvs', 3), ('num_etvs', 4))),
                     62  : ('Storm Structure', 460., LegacyMapper, ()),
                     63  : ('Layer Composite Reflectivity (Layer 1 Average)', 230., LegacyMapper,
                            (('max', 3),
                             ('layer_bottom', scaled_elem(4, 1000.)), ('layer_top', scaled_elem(5, 1000.)),
                             ('calib_const', float_elem(7, 8)))),
                     64  : ('Layer Composite Reflectivity (Layer 2 Average)', 230., LegacyMapper,
                            (('max', 3),
                             ('layer_bottom', scaled_elem(4, 1000.)), ('layer_top', scaled_elem(5, 1000.)),
                             ('calib_const', float_elem(7, 8)))),
                     65  : ('Layer Composite Reflectivity (Layer 1 Max)', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                             ('layer_bottom', scaled_elem(4, 1000.)), ('layer_top', scaled_elem(5, 1000.)),
                             ('calib_const', float_elem(7, 8)))),
                     66  : ('Layer Composite Reflectivity (Layer 2 Max)', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                             ('layer_bottom', scaled_elem(4, 1000.)), ('layer_top', scaled_elem(5, 1000.)),
                             ('calib_const', float_elem(7, 8)))),
                     67  : ('Layer Composite Reflectivity - AP Removed', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                             ('layer_bottom', scaled_elem(4, 1000.)), ('layer_top', scaled_elem(5, 1000.)),
                             ('calib_const', float_elem(7, 8)))),
                     74  : ('Radar Coded Message', 460., LegacyMapper, ()),
                     78  : ('Surface Rainfall Accumulation (1 hour)', 230., LegacyMapper,
                            (('max_rainfall', scaled_elem(3, 0.1)), ('bias', scaled_elem(4, 0.01)),
                             ('gr_pairs', scaled_elem(5, 0.01)), ('rainfall_end', date_elem(6, 7)))),
                     79  : ('Surface Rainfall Accumulation (3 hour)', 230., LegacyMapper,
                            (('max_rainfall', scaled_elem(3, 0.1)), ('bias', scaled_elem(4, 0.01)),
                             ('gr_pairs', scaled_elem(5, 0.01)), ('rainfall_end', date_elem(6, 7)))),
                     80  : ('Storm Total Rainfall Accumulation', 230., LegacyMapper,
                            (('max_rainfall', scaled_elem(3, 0.1)), ('rainfall_begin', date_elem(4, 5)),
                             ('rainfall_end', date_elem(6, 7)), ('bias', scaled_elem(8, 0.01)),
                             ('gr_pairs', scaled_elem(9, 0.01)))),
                     81  : ('Hourly Digital Precipitation Array', 230., PrecipArrayMapper,
                            (('max_rainfall', scaled_elem(3, 0.001)), ('bias', scaled_elem(4, 0.01)),
                             ('gr_pairs', scaled_elem(5, 0.01)), ('rainfall_end', date_elem(6, 7)))),
                     82  : ('Supplemental Precipitation Data', None, LegacyMapper, ()),
                     89  : ('Layer Composite Reflectivity (Layer 3 Average)', 230., LegacyMapper,
                            (('max', 3),
                             ('layer_bottom', scaled_elem(4, 1000.)), ('layer_top', scaled_elem(5, 1000.)),
                             ('calib_const', float_elem(7, 8)))),
                     90  : ('Layer Composite Reflectivity (Layer 3 Max)', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                             ('layer_bottom', scaled_elem(4, 1000.)), ('layer_top', scaled_elem(5, 1000.)),
                             ('calib_const', float_elem(7, 8)))),
                     93  : ('ITWS Digital Base Velocity', 115., DigitalVelMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3),
                             ('max', 4), ('precision', 6))),
                     94  : ('Base Reflectivity Data Array', 460., DigitalRefMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     95  : ('Composite Reflectivity Edited for AP', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     96  : ('Composite Reflectivity Edited for AP', 460., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     97  : ('Composite Reflectivity Edited for AP', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     98  : ('Composite Reflectivity Edited for AP', 460., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8)))),
                     99  : ('Base Velocity Data Array', 300., DigitalVelMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     132 : ('Clutter Likelihood Reflectivity', 230., LegacyMapper, (('el_angle', scaled_elem(2, 0.1)),)),
                     133 : ('Clutter Likelihood Doppler', 230., LegacyMapper, (('el_angle', scaled_elem(2, 0.1)),)),
                     134 : ('High Resolution VIL', 460., DigitalVILMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('num_edited', 4),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     135 : ('Enhanced Echo Tops', 345., DigitalEETMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', scaled_elem(3, 1000.)), # Max in ft
                             ('num_edited', 4), ('ref_thresh', 5), ('points_removed', 6),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     138 : ('Digital Storm Total Precipitation', 230., DigitalStormPrecipMapper,
                            (('rainfall_begin', date_elem(0, 1)), ('bias', scaled_elem(2, 0.01)),
                             ('max', scaled_elem(3, 0.01)), ('rainfall_end', date_elem(4, 5)),
                             ('gr_pairs', scaled_elem(6, 0.01)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     141 : ('Mesocyclone Detection', 230., LegacyMapper,
                            (('min_ref_thresh', 0), ('overlap_display_filter', 1), ('min_strength_rank', 2))),
                     152 : ('Archive III Status Product', None, LegacyMapper,
                            (('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     153 : ('Super Resolution Reflectivity Data Array', 460., DigitalRefMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     154 : ('Super Resolution Velocity Data Array', 300., DigitalVelMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     155 : ('Super Resolution Spectrum Width Data Array', 300., DigitalSPWMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     156 : ('Turbulence Detection (Eddy Dissipation Rate)', 230., EDRMapper,
                            (('el_start_time', 0), ('el_end_time', 1),
                             ('el_angle', scaled_elem(2, 0.1)), ('min_el', scaled_elem(3, 0.01)),
                             ('mean_el', scaled_elem(4, 0.01)), ('max_el', scaled_elem(5, 0.01)))),
                     157 : ('Turbulence Detection (Eddy Dissipation Rate Confidence)', 230., EDRMapper,
                            (('el_start_time', 0), ('el_end_time', 1),
                             ('el_angle', scaled_elem(2, 0.1)), ('min_el', scaled_elem(3, 0.01)),
                             ('mean_el', scaled_elem(4, 0.01)), ('max_el', scaled_elem(5, 0.01)))),
                     158 : ('Differential Reflectivity', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.1)), ('max', scaled_elem(4, 0.1)))),
                     159 : ('Digital Differential Reflectivity', 300., GenericDigitalMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.1)), ('max', scaled_elem(4, 0.1)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     160 : ('Correlation Coefficient', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.00333)), ('max', scaled_elem(4, 0.00333)))),
                     161 : ('Digital Correlation Coefficient', 300., GenericDigitalMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.00333)), ('max', scaled_elem(4, 0.00333)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     162 : ('Specific Differential Phase', 230., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.05)), ('max', scaled_elem(4, 0.05)))),
                     163 : ('Digital Specific Differential Phase', 300., GenericDigitalMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.05)), ('max', scaled_elem(4, 0.05)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     164 : ('Hydrometeor Classification', 230., LegacyMapper, (('el_angle', scaled_elem(2, 0.1)),)),
                     165 : ('Digital Hydrometeor Classification', 300., DigitalHMCMapper, (('el_angle', scaled_elem(2, 0.1)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     166 : ('Melting Layer', 230., LegacyMapper, (('el_angle', scaled_elem(2, 0.1)),)),
                     169 : ('One Hour Accumulation', 230., LegacyMapper,
                            (('null_product', low_byte(2)), ('max', scaled_elem(3, 0.1)),
                             ('rainfall_end', date_elem(4, 5)), ('bias', scaled_elem(6, 0.01)),
                             ('gr_pairs', scaled_elem(7, 0.01)))),
                     170 : ('Digital Accumulation Array', 230., GenericDigitalMapper,
                            (('null_product', low_byte(2)), ('max', scaled_elem(3, 0.1)),
                             ('rainfall_end', date_elem(4, 5)), ('bias', scaled_elem(6, 0.01)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     171 : ('Storm Total Accumulation', 230., LegacyMapper,
                            (('rainfall_begin', date_elem(0, 1)), ('null_product', low_byte(2)),
                             ('max', scaled_elem(3, 0.1)), ('rainfall_end', date_elem(4, 5)),
                             ('bias', scaled_elem(6, 0.01)), ('gr_pairs', scaled_elem(7, 0.01)))),
                     172 : ('Digital Storm total Accumulation', 230., GenericDigitalMapper,
                            (('rainfall_begin', date_elem(0, 1)), ('null_product', low_byte(2)),
                             ('max', scaled_elem(3, 0.1)), ('rainfall_end', date_elem(4, 5)), ('bias', scaled_elem(6, 0.01)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     173 : ('Digital User-Selectable Accumulation', 230., GenericDigitalMapper,
                            (('period', 1), ('missing_period', high_byte(2)),
                             ('null_product', low_byte(2)), ('max', scaled_elem(3, 0.1)),
                             ('rainfall_end', date_elem(4, 0)), ('start_time', 5), ('bias', scaled_elem(6, 0.01)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     174 : ('Digital One-Hour Difference Accumulation', 230., GenericDigitalMapper,
                            (('max', scaled_elem(3, 0.1)), ('rainfall_end', date_elem(4, 5)), ('min', scaled_elem(6, 0.1)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     175 : ('Digital Storm Total Difference Accumulation', 230., GenericDigitalMapper,
                            (('rainfall_begin', date_elem(0, 1)), ('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)), ('rainfall_end', date_elem(4, 5)), ('min', scaled_elem(6, 0.1)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     176 : ('Digital Instantaneous Precipitation Rate', 230., GenericDigitalMapper,
                            (('rainfall_begin', date_elem(0, 1)), ('precip_detected', high_byte(2)), ('need_bias', low_byte(2)),
                             ('max', 3), ('percent_filled', scaled_elem(4, 0.01)), ('max_elev', scaled_elem(5, 0.1)),
                             ('bias', scaled_elem(6, 0.01)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     177 : ('Hybrid Hydrometeor Classification', 230., DigitalHMCMapper,
                            (('mode_filter_size', 3), ('hybrid_percent_filled', 4), ('max_elev', scaled_elem(5, 0.1)),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     181 : ('TDWR Base Reflectivity', 90., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
#                            ('calib_const', float_elem(7, 8))),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     182 : ('TDWR Base Velocity', 90., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))),
                     186 : ('TDWR Long Range Base Reflectivity', 416., LegacyMapper,
                            (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                             ('compression', 7), ('uncompressed_size', combine_elem(8, 9))))}

    def __init__(self, fname):
        # Just read in the entire set of data at once
        self.filename = fname
        self._buffer = IOBuffer.fromfile(open(fname, 'rb'))

        # Pop off the WMO header if we find it
        self._process_WMO_header()

        # Pop off last 4 bytes if necessary
        self._process_end_bytes()

        # Handle free text message products that are pure text
        if self.wmo_code == 'NOUS':
            self.text = ''.join(self._buffer.read_ascii())
            return

        # Decompress the data if necessary, and if so, pop off new header
        self._buffer = IOBuffer(self._buffer.read_func(zlib_decompress_all_frames))
        self._process_WMO_header()

        # Check for empty product
        if len(self._buffer) == 0:
            warnings.warn("{}: Empty product!".format(self.filename))
            return

        # Set up places to store data and metadata
#        self.data = []
        self.metadata = dict()

        # Unpack the message header and the product description block
        msg_start = self._buffer.set_mark()
        self.header = self._buffer.read_struct(self.header_fmt)
        #print self.header, len(self._buffer), self.header.msg_len - self.header_fmt.size
        assert self._buffer.check_remains(self.header.msg_len - self.header_fmt.size), 'Insufficient bytes remaining.'

        # Handle GSM and jump out
        if self.header.code == 2:
            self.gsm = self._buffer.read_struct(self.gsm_fmt)
            assert self.gsm.divider == -1
            if self.gsm.block_len > 82:
                self.gsm_additional = self._buffer.read_struct(self.additional_gsm_fmt)
                assert self.gsm.block_len == 178
            else:
                assert self.gsm.block_len == 82
            return

        self.prod_desc = self._buffer.read_struct(self.prod_desc_fmt)

        # Convert thresholds and dependent values to lists of values
        self.thresholds = [getattr(self.prod_desc, 'thr%d' % i) for i in range(1, 17)]
        self.depVals = [getattr(self.prod_desc, 'dep%d' % i) for i in range(1, 11)]

        # Set up some time/location metadata
        self.metadata['msg_time'] = nexrad_to_datetime(self.header.date, self.header.time*1000)
        self.metadata['vol_time'] = nexrad_to_datetime(self.prod_desc.vol_date, self.prod_desc.vol_start_time*1000)
        self.metadata['prod_time'] = nexrad_to_datetime(self.prod_desc.prod_gen_date, self.prod_desc.prod_gen_time*1000)
        self.lat = self.prod_desc.lat * 0.001
        self.lon = self.prod_desc.lon * 0.001
        self.height = self.prod_desc.height

        # Handle product-specific blocks. Default to compression and elevation angle
        # Also get other product specific information, like name,
        # maximum range, and how to map data bytes to values
        self.product_name, self.max_range, mapper, meta = self.prod_spec_map.get(self.header.code,
                                                 ('Unknown Product', 230., LegacyMapper, (('el_angle', scaled_elem(2, 0.1)),
                                                  ('compression', 7), ('uncompressed_size', combine_elem(8, 9)),
                                                  ('defaultVals', 0))))
        for name,block in meta:
            if callable(block):
                self.metadata[name] = block(self.depVals)
            else:
                self.metadata[name] = self.depVals[block]

        # Now that we have the header, we have everything needed to make tables
        # Store as class that can be called
        self.map_data = mapper(self)

        # Process compression if indicated. We need to fail
        # gracefully here since we default to it being on
        if self.metadata.get('compression', False):
            try:
                comp_start = self._buffer.set_mark()
                decomp_data = self._buffer.read_func(bz2.decompress)
                self._buffer.splice(comp_start, decomp_data)
                assert self._buffer.check_remains(self.metadata['uncompressed_size'])
            except IOError:
                pass

        # Unpack the various blocks, if present.  The factor of 2 converts from
        # 'half-words' to bytes
        # Check to see if this is one of the "special" products that uses
        # header-free blocks and re-assigns the offests
        if self.header.code in self.standalone_tabular:
            if self.prod_desc.sym_off:
                # For standalone tabular alphanumeric, symbology offset is actually
                # tabular
                self._unpack_tabblock(msg_start, 2 * self.prod_desc.sym_off, False)
            if self.prod_desc.graph_off:
                # Offset seems to be off by 1 from where we're counting, but
                # it's not clear why.
                self._unpack_standalone_graphblock(msg_start, 2 * (self.prod_desc.graph_off - 1))
        # Need special handling for (old) radar coded message format
        elif self.header.code == 74:
            self._unpack_rcm(msg_start, 2 * self.prod_desc.sym_off)
        else:
            if self.prod_desc.sym_off:
                self._unpack_symblock(msg_start, 2 * self.prod_desc.sym_off)
            if self.prod_desc.graph_off:
                self._unpack_graphblock(msg_start, 2 * self.prod_desc.graph_off)
            if self.prod_desc.tab_off:
                self._unpack_tabblock(msg_start, 2 * self.prod_desc.tab_off)

        if 'defaultVals' in self.metadata:
            warnings.warn("{}: Using default metadata for product {}".format(self.filename, self.header.code))

    def _process_WMO_header(self):
        # Read off the WMO header if necessary
        data = self._buffer.get_next(64)
        match = self.wmo_finder.match(data)
        if match:
            self.wmo_code = match.groups()[0]
            self.siteID = match.groups()[-1]
            self._buffer.skip(match.end())

    def _process_end_bytes(self):
        if self._buffer[-4:-1] == '\r\r\n':
            self._buffer.truncate(4)

    def _unpack_rle_data(self, data):
        # Unpack Run-length encoded data
        unpacked = []
        for run in data:
            num,val = run>>4, run&0x0F
            unpacked.extend([val]*num)
        return unpacked

    def pos_scale(self, isSymBlock):
        return 0.25 if isSymBlock else 1

    def _unpack_rcm(self, start, offset):
        self._buffer.jump_to(start, offset)
        header = self._buffer.read_ascii(10)
        assert header == '1234 ROBUU'
        #warnings.warn("{}: RCM decoding not supported.".format(self.filename))

    def _unpack_symblock(self, start, offset):
        self._buffer.jump_to(start, offset)
        blk = self._buffer.read_struct(self.sym_block_fmt)

        self.sym_block = []
        assert blk.divider == -1, 'Bad divider for symbology block: %d should be -1' % blk.divider
        assert blk.block_id == 1, 'Bad block ID for symbology block: %d should be 1' % blk.block_id
        for l in range(blk.nlayer):
            layer_hdr = self._buffer.read_struct(self.sym_layer_fmt)
            assert layer_hdr.divider == -1
            layer = []
            self.sym_block.append(layer)
            #print blk, layer_hdr, ' '.join('%02x' % ord(c) for c in self._buffer.get_next(64))
            layer_start = self._buffer.set_mark()
            while self._buffer.offset_from(layer_start) < layer_hdr.length:
                packet_code = self._buffer.read_int('>H')
                if packet_code in self.packet_map:
                    layer.append(self.packet_map[packet_code](self, packet_code, True))
                else:
                    warnings.warn('{0}: Unknown symbology packet type {1}/{1:#x}.'.format(self.filename, packet_code))
                    self._buffer.jump_to(layer_start, layer_hdr.length)
            assert self._buffer.offset_from(layer_start) == layer_hdr.length

    def _unpack_graphblock(self, start, offset):
        self._buffer.jump_to(start, offset)
        hdr = self._buffer.read_struct(self.graph_block_fmt)
        assert hdr.divider == -1, 'Bad divider for graphical block: %d should be -1' % hdr.divider
        assert hdr.block_id == 2, 'Bad block ID for graphical block: %d should be 1' % hdr.block_id
        self.graph_pages = []
        for page in range(hdr.num_pages):
            page_num = self._buffer.read_int('>H')
            assert page + 1 == page_num
            page_size = self._buffer.read_int('>H')
            page_start = self._buffer.set_mark()
            packets = []
            while self._buffer.offset_from(page_start) < page_size:
                packet_code = self._buffer.read_int('>H')
                if packet_code in self.packet_map:
                    packets.append(self.packet_map[packet_code](self, packet_code, False))
                else:
                    warnings.warn('{0}: Unknown graphical packet type {1}/{1:#x}.'.format(self.filename, packet_code))
                    self._buffer.skip(page_size)
            self.graph_pages.append(packets)

    def _unpack_standalone_graphblock(self, start, offset):
        self._buffer.jump_to(start, offset)
        packets = []
        while not self._buffer.at_end():
            packet_code = self._buffer.read_int('>H')
            if packet_code in self.packet_map:
                packets.append(self.packet_map[packet_code](self, packet_code, False))
            else:
                warnings.warn('{0}: Unknown standalone graphical packet type {1}/{1:#x}.'.format(self.filename, packet_code))
                # Assume next 2 bytes is packet length and try skipping
                num_bytes = self._buffer.read_int('>H')
                self._buffer.skip(num_bytes)
        self.graph_pages = [packets]

    def _unpack_tabblock(self, start, offset, haveHeader=True):
        self._buffer.jump_to(start, offset)
        block_start = self._buffer.set_mark()

        # Read the header and validate if needed
        if haveHeader:
            header = self._buffer.read_struct(self.tab_header_fmt)
            assert header.divider == -1
            assert header.block_id == 3

            # Read off secondary message and product description blocks,
            # but as far as I can tell, all we really need is the text that follows
            msg_header2 = self._buffer.read_struct(self.header_fmt)
            prod_desc2 = self._buffer.read_struct(self.prod_desc_fmt)

        # Get the start of the block with number of pages and divider
        blk = self._buffer.read_struct(self.tab_block_fmt)
        assert blk.divider == -1

        # Read the pages line by line, break pages on a -1 character count
        self.tab_pages = []
        for page in range(blk.num_pages):
            lines = []
            num_chars = self._buffer.read_int('>h')
            while num_chars != -1:
                lines.append(''.join(self._buffer.read_ascii(num_chars)))
                num_chars = self._buffer.read_int('>h')
            self.tab_pages.append('\n'.join(lines))

        if haveHeader:
            assert self._buffer.offset_from(block_start) == header.block_len

    def __repr__(self):
        return self.filename + ': ' + '\n'.join(map(str, [self.product_name, self.header, self.prod_desc, self.thresholds,
                                                           self.depVals, self.metadata,
                                                           (self.siteID, self.lat, self.lon, self.height)]))

    def _unpack_packet_radial_data(self, code, inSymBlock):
        hdr_fmt = NamedStruct([('ind_first_bin', 'H'), ('nbins', 'H'),
            ('i_center', 'h'), ('j_center', 'h'), ('scale_factor', 'h'),
            ('num_rad', 'H')], '>', 'RadialHeader')
        rad_fmt = NamedStruct([('num_hwords', 'H'), ('start_angle', 'h'),
            ('angle_delta', 'h')], '>', 'RadialData')
        hdr = self._buffer.read_struct(hdr_fmt)
        rads = []
        for i in range(hdr.num_rad):
            rad = self._buffer.read_struct(rad_fmt)
            start_az = rad.start_angle * 0.1
            end_az = start_az + rad.angle_delta * 0.1
            rads.append((start_az, end_az,
                         self._unpack_rle_data(self._buffer.read_binary(2 * rad.num_hwords))))
        start,end,vals = zip(*rads)
        return dict(start_az=list(start), end_az=list(end), data=list(vals),
                center=(hdr.i_center * self.pos_scale(inSymBlock), hdr.j_center * self.pos_scale(inSymBlock)),
                gate_scale=hdr.scale_factor * 0.001, first=hdr.ind_first_bin)

    def _unpack_packet_digital_radial(self, code, inSymBlock):
        hdr_fmt = NamedStruct([('ind_first_bin', 'H'), ('nbins', 'H'),
            ('i_center', 'h'), ('j_center', 'h'), ('scale_factor', 'h'),
            ('num_rad', 'H')], '>', 'DigitalRadialHeader')
        rad_fmt = NamedStruct([('num_bytes', 'H'), ('start_angle', 'h'),
            ('angle_delta', 'h')], '>', 'DigitalRadialData')
        hdr = self._buffer.read_struct(hdr_fmt)
        rads = []
        for i in range(hdr.num_rad):
            rad = self._buffer.read_struct(rad_fmt)
            start_az = rad.start_angle * 0.1
            end_az = start_az + rad.angle_delta * 0.1
            rads.append((start_az, end_az, self._buffer.read_binary(rad.num_bytes)))
        start,end,vals = zip(*rads)
        return dict(start_az=list(start), end_az=list(end), data=list(vals),
                center=(hdr.i_center * self.pos_scale(inSymBlock),
                        hdr.j_center * self.pos_scale(inSymBlock)),
                        gate_scale=hdr.scale_factor * 0.001, first=hdr.ind_first_bin)

    def _unpack_packet_raster_data(self, code, inSymBlock):
        hdr_fmt = NamedStruct([('code', 'L'), ('i_start', 'h'), ('j_start', 'h'), # start in km/4
                               ('xscale_int', 'h'), ('xscale_frac', 'h'),
                               ('yscale_int', 'h'), ('yscale_frac', 'h'),
                               ('num_rows', 'h'), ('packing', 'h')], '>', 'RasterData')
        hdr = self._buffer.read_struct(hdr_fmt)
        assert hdr.code == 0x800000C0
        assert hdr.packing == 2
        rows = []
        for row in range(hdr.num_rows):
            num_bytes = self._buffer.read_int('>H')
            rows.append(self._unpack_rle_data(self._buffer.read_binary(num_bytes)))
        return dict(start_x=hdr.i_start * hdr.xscale_int,
                    start_y=hdr.j_start * hdr.yscale_int, data=rows)

    def _unpack_packet_uniform_text(self, code, inSymBlock):
        # By not using a struct, we can handle multiple codes
        num_bytes = self._buffer.read_int('>H')
        if code == 8:
            value = self._buffer.read_int('>H')
            read_bytes = 6
        else:
            value = None
            read_bytes = 4
        i_start = self._buffer.read_int('>h')
        j_start = self._buffer.read_int('>h')

        # Text is what remains beyond what's been read, not including byte count
        text = ''.join(self._buffer.read_ascii(num_bytes - read_bytes))
        return dict(x=i_start * self.pos_scale(inSymBlock), y=j_start * self.pos_scale(inSymBlock),
                    color=value, text=text)

    def _unpack_packet_special_text_symbol(self, code, inSymBlock):
        d = self._unpack_packet_uniform_text(code, inSymBlock)

        # Translate special characters to their meaning
        ret = dict()
        symbol_map = {'!': 'past storm position', '"': 'current storm position', '#': 'forecast storm position',
                      '$': 'past MDA position', '%': 'forecast MDA position', ' ': None}

        # Use this meaning as the key in the returned packet
        for c in d['text']:
            if c not in symbol_map:
                warnings.warn('{0}: Unknown special symbol {1}/{2:#x}.'.format(self.filename, c, ord(c)))
            else:
                key = symbol_map[c]
                if key:
                    ret[key] = d['x'], d['y']
        del d['text']

        return ret

    def _unpack_packet_special_graphic_symbol(self, code, inSymBlock):
        type_map = {3: 'Mesocyclone', 11: '3D Correlated Shear', 12: 'TVS', 26:'ETVS', 13:'Positive Hail',
                    14: 'Probable Hail', 15: 'Storm ID', 19: 'HDA', 25: 'STI Circle'}
        point_feature_map = {1: 'Mesocyclone (ext.)', 3: 'Mesocyclone', 5: 'TVS (Ext.)', 6: 'ETVS (Ext.)',
                             7: 'TVS', 8: 'ETVS', 9: 'MDA', 10: 'MDA (Elev.)', 11: 'MDA (Weak)'}

        # Read the number of bytes and set a mark for sanity checking
        num_bytes = self._buffer.read_int('>H')
        packet_data_start = self._buffer.set_mark()

        scale = self.pos_scale(inSymBlock)

        # Loop over the bytes we have
        ret = defaultdict(list)
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            # Read position
            ret['x'].append(self._buffer.read_int('>h') * scale)
            ret['y'].append(self._buffer.read_int('>h') * scale)

            # Handle any types that have additional info
            if code in (3, 11, 25):
                ret['radius'].append(self._buffer.read_int('>h') * scale)
            elif code == 15:
                ret['id'].append(''.join(self._buffer.read_ascii(2)))
            elif code == 19:
                ret['POH'].append(self._buffer.read_int('>h'))
                ret['POSH'].append(self._buffer.read_int('>h'))
                ret['Max Size'].append(self._buffer.read_int('>H'))
            elif code == 20:
                kind = self._buffer.read_int('>H')
                attr = self._buffer.read_int('>H')
                if kind < 5 or kind > 8:
                    ret['radius'].append(attr * scale)

                if kind not in point_feature_map:
                    warnings.warn('{0}: Unknown graphic symbol point kind {1}/{1:#x}.'.format(self.filename, kind))
                    ret['type'].append('Unknown (%d)' % kind)
                else:
                    ret['type'].append(point_feature_map[kind])

        # Map the code to a name for this type of symbol
        if code != 20:
            if code not in type_map:
                warnings.warn('{0}: Unknown graphic symbol type {1}/{1:#x}.'.format(self.filename, code))
                ret['type'] = 'Unknown'
            else:
                ret['type'] = type_map[code]

        # Check and return
        assert self._buffer.offset_from(packet_data_start) == num_bytes

        # Reduce dimensions of lists if possible
        reduce_lists(ret)

        return ret

    def _unpack_packet_scit(self, code, inSymBlock):
        num_bytes = self._buffer.read_int('>H')
        packet_data_start = self._buffer.set_mark()
        ret = defaultdict(list)
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            next_code = self._buffer.read_int('>H')
            if next_code not in self.packet_map:
                warnings.warn('{0}: Unknown packet in SCIT {1}/{1:#x}.'.format(self.filename, next_code))
                self._buffer.jump_to(packet_data_start, num_bytes)
                return ret
            else:
                next_packet = self.packet_map[next_code](self, next_code, inSymBlock)
                if next_code == 6:
                    ret['track'].append(next_packet['vectors'])
                elif next_code == 25:
                    ret['STI Circle'].append(next_packet)
                elif next_code == 2:
                    ret['markers'].append(next_packet)
                else:
                    warnings.warn('{0}: Unsupported packet in SCIT {1}/{1:#x}.'.format(self.filename, next_code))
                    ret['data'].append(next_packet)
        reduce_lists(ret)
        return ret

    def _unpack_packet_digital_precipitation(self, code, inSymBlock):
        # Read off a couple of unused spares
        spare1 = self._buffer.read_int('>H')
        spare2 = self._buffer.read_int('>H')

        # Get the size of the grid
        lfm_boxes = self._buffer.read_int('>H')
        num_rows = self._buffer.read_int('>H')
        rows = []

        # Read off each row and decode the RLE data
        for row_num in range(num_rows):
            row_num_bytes = self._buffer.read_int('>H')
            row_bytes = self._buffer.read_binary(row_num_bytes)
            if code == 18:
                row = self._unpack_rle_data(row_bytes)
            else:
                row = []
                for run,level in zip(row_bytes[::2], row_bytes[1::2]):
                    row.extend([level] * run)
            assert len(row) == lfm_boxes
            rows.append(row)

        return dict(data=rows)

    def _unpack_packet_linked_vector(self, code, inSymBlock):
        num_bytes = self._buffer.read_int('>h')
        if code == 9:
            value = self._buffer.read_int('>h')
            num_bytes -= 2
        else:
            value = None
        scale = self.pos_scale(inSymBlock)
        pos = [b * scale for b in self._buffer.read_binary(num_bytes / 2, '>h')]
        vectors = zip(pos[::2], pos[1::2])
        return dict(vectors=vectors, color=value)

    def _unpack_packet_vector(self, code, inSymBlock):
        num_bytes = self._buffer.read_int('>h')
        if code == 10:
            value = self._buffer.read_int('>h')
            num_bytes -= 2
        else:
            value = None
        scale = self.pos_scale(inSymBlock)
        pos = [p * scale for p in self._buffer.read_binary(num_bytes / 2, '>h')]
        vectors = zip(pos[::4], pos[1::4], pos[2::4], pos[3::4])
        return dict(vectors=vectors, color=value)

    def _unpack_packet_contour_color(self, code, inSymBlock):
        # Check for color value indicator
        assert self._buffer.read_int('>H') == 0x0002

        # Read and return value (level) of contour
        return dict(color=self._buffer.read_int('>H'))

    def _unpack_packet_linked_contour(self, code, inSymBlock):
        # Check for initial point indicator
        assert self._buffer.read_int('>H') == 0x8000

        scale = self.pos_scale(inSymBlock)
        startx = self._buffer.read_int('>h') * scale
        starty = self._buffer.read_int('>h') * scale
        vectors = [(startx, starty)]
        num_bytes = self._buffer.read_int('>H')
        pos = [b * scale for b in self._buffer.read_binary(num_bytes / 2, '>h')]
        vectors.extend(zip(pos[::2], pos[1::2]))
        return dict(vectors=vectors)

    def _unpack_packet_wind_barbs(self, code, inSymBlock):
        # Figure out how much to read
        num_bytes = self._buffer.read_int('>h')
        packet_data_start = self._buffer.set_mark()
        ret = defaultdict(list)

        # Read while we have data, then return
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            ret['color'].append(self._buffer.read_int('>h'))
            ret['x'].append(self._buffer.read_int('>h') * self.pos_scale(inSymBlock))
            ret['y'].append(self._buffer.read_int('>h') * self.pos_scale(inSymBlock))
            ret['direc'].append(self._buffer.read_int('>h'))
            ret['speed'].append(self._buffer.read_int('>h'))
        return ret

    def _unpack_packet_generic(self, code, inSymBlock):
        # Reserved HW
        assert self._buffer.read_int('>h') == 0

        # Read number of bytes (2 HW) and return
        num_bytes = self._buffer.read_int('>l')
        data = ''.join(self._buffer.read_ascii(num_bytes))
        return dict(xdrdata=data)

    def _unpack_packet_trend_times(self, code, inSymBlock):
        num_bytes = self._buffer.read_int('>h')
        return dict(times=self._read_trends())

    def _unpack_packet_cell_trend(self, code, inSymBlock):
        code_map = ['Cell Top', 'Cell Base', 'Max Reflectivity Height',
                    'Probability of Hail', 'Probability of Severe Hail',
                    'Cell-based VIL', 'Maximum Reflectivity',
                    'Centroid Height']
        code_scales = [100, 100, 100, 1, 1, 1, 1, 100]
        num_bytes = self._buffer.read_int('>h')
        packet_data_start = self._buffer.set_mark()
        cell_id = ''.join(self._buffer.read_ascii(2))
        x = self._buffer.read_int('>h') * self.pos_scale(inSymBlock)
        y = self._buffer.read_int('>h') * self.pos_scale(inSymBlock)
        ret = dict(id=cell_id, x=x, y=y)
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            code = self._buffer.read_int('>h')
            try:
                ind = code - 1
                key = code_map[ind]
                scale = code_scales[ind]
            except IndexError:
                warnings.warn('{0}: Unsupported trend code {1}/{1:#x}.'.format(self.filename, code))
                key = 'Unknown'
                scale = 1
            vals = self._read_trends()
            if code in (1,2):
                ret['%s Limited' % key] = [True if v > 700 else False for v in vals]
                vals = [v - 1000 if v > 700 else v for v in vals]
            ret[key] = [v * scale for v in vals]

        return ret

    def _read_trends(self):
        num_vols = self._buffer.read_int('b')
        latest = self._buffer.read_int('b')
        vals = [self._buffer.read_int('>h') for i in range(num_vols)]

        # Wrap the circular buffer so that latest is last
        vals = vals[latest:] + vals[:latest]
        return vals


    packet_map = {1      : _unpack_packet_uniform_text,
                  2      : _unpack_packet_special_text_symbol,
                  3      : _unpack_packet_special_graphic_symbol,
                  4      : _unpack_packet_wind_barbs,
                  6      : _unpack_packet_linked_vector,
                  8      : _unpack_packet_uniform_text,
#                  9      : _unpack_packet_linked_vector,
                  10     : _unpack_packet_vector,
                  11     : _unpack_packet_special_graphic_symbol,
                  12     : _unpack_packet_special_graphic_symbol,
                  13     : _unpack_packet_special_graphic_symbol,
                  14     : _unpack_packet_special_graphic_symbol,
                  15     : _unpack_packet_special_graphic_symbol,
                  16     : _unpack_packet_digital_radial,
                  17     : _unpack_packet_digital_precipitation,
                  18     : _unpack_packet_digital_precipitation,
                  19     : _unpack_packet_special_graphic_symbol,
                  20     : _unpack_packet_special_graphic_symbol,
                  21     : _unpack_packet_cell_trend,
                  22     : _unpack_packet_trend_times,
                  23     : _unpack_packet_scit,
                  24     : _unpack_packet_scit,
                  25     : _unpack_packet_special_graphic_symbol,
                  26     : _unpack_packet_special_graphic_symbol,
                  28     : _unpack_packet_generic,
#                  29     : _unpack_packet_generic,
                  0x0802 : _unpack_packet_contour_color,
                  0x0E03 : _unpack_packet_linked_contour,
                  0xaf1f : _unpack_packet_radial_data,
                  0xba07 : _unpack_packet_raster_data}

def is_precip_mode(vcp_num):
    return vcp_num // 10 == 3
