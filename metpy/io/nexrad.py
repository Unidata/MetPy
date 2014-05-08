import bz2
import datetime
import gzip
import re
import warnings
import zlib
from struct import Struct
from cStringIO import StringIO

import numpy as np
from scipy.constants import day, milli
from collections import namedtuple
from metpy.cbook import is_string_like

class NamedStruct(Struct):
    def __init__(self, info, prefmt='', tuple_name=None):
        if tuple_name is None:
            tuple_name = 'NamedStruct'
        names, fmts = zip(*info)
        self._tuple = namedtuple(tuple_name, ' '.join(n for n in names if n))
        Struct.__init__(self, prefmt + ''.join(fmts))

    def unpack(self, s):
        return self._tuple(*Struct.unpack(self, s))

    def unpack_from(self, buff, offset=0):
        return self._tuple(*Struct.unpack_from(self, buff, offset))

    def unpack_file(self, fobj):
        bytes = fobj.read(self.size)
        return self.unpack(bytes)

class FileStruct(Struct):
    'Wraps unpacking data directly from a file.'
    def __init__(self, fmt, fobj):
        self._fobj = fobj
        Struct.__init__(self, fmt)
    def read(self):
        bytes = self._fobj.read(self.size)
        return self.unpack(bytes)

class CompressedBlockFile(object):
    '''
    A class to wrap a file that consists of a series of compressed blocks
    and their sizes.
    '''
    def __init__(self, fobj):
        self._fobj = fobj
        self._buffer = StringIO(self._read_next_compressed_block())

    def read(self, nbytes):
        bytes = self._buffer.read(nbytes)
        if not bytes:
            self._buffer = StringIO(self._read_next_compressed_block())
            bytes = self._buffer.read(nbytes)
        return bytes

    def close(self):
        self._fobj.close()
        self._buffer.close()

    def _read_next_compressed_block(self):
        num_bytes, = FileStruct('>l', self._fobj).read()
        print 'num_bytes: %d' % num_bytes
        cmp_block = self._fobj.read(num_bytes)
        return bz2.decompress(cmp_block)

def nexrad_to_datetime(julian_date, ms_midnight):
    #Subtracting one from julian_date is because epoch date is 1
    return datetime.datetime.fromtimestamp((julian_date - 1) * day
        + ms_midnight * milli)

class Level2File(object):
    #Number of bytes
    AR2_BLOCKSIZE = 2416
    def __init__(self, filename):
        if is_string_like(filename):
            if filename.endswith('.bz2'):
                self._fobj = bz2.BZ2File(filename, 'rb')
            elif filename.endswith('.gz'):
                self._fobj = gzip.GzipFile(filename, 'rb')
            else:
                self._fobj = file(filename, 'rb')
        else:
            self._fobj = filename

        self._read_volume_header()

        #If this comes up with an actual number of bytes, we have a file with
        #bz2'd blocks.  If it's zero, that means we're in the 12-bytes
        fmt = FileStruct('>l', self._fobj)
        bz2_test, = fmt.read()

        #Rewind the bytes we just read
        self._fobj.seek(-fmt.size, 1)

        if bz2_test:
            self._fobj = CompressedBlockFile(self._fobj)

        #Now we're all initialized, we can proceed with reading in data
        self._read_data()

    def _read_volume_header(self):
        vol_fields = [('version', '9s'), ('vol_num', '3s'), ('date', 'L'),
                      ('time_ms', 'L'), ('stid', '4s')]
        vol_hdr_fmt = NamedStruct(vol_fields, '>', 'VolHdr')
        vol_hdr = vol_hdr_fmt.unpack_file(self._fobj)
        print vol_hdr
        self.dt = nexrad_to_datetime(vol_hdr.date, vol_hdr.time_ms)
        self._version = vol_hdr.version
        self.stid = vol_hdr.stid
        print self.dt

    def _read_data(self):
        ctm_hdr = FileStruct('12c', self._fobj)
        ctm_hdr.read()
        msg_hdr_fields = [('size_hw', 'H'), ('rda_channel', 'B'),
                          ('msg_type', 'B'), ('seq_num', 'H'),
                          ('date', 'H'), ('time_ms', 'I'),
                          ('num_segments', 'H'), ('segment_num', 'H')]
        msg_hdr_fmt = NamedStruct(msg_hdr_fields, '>', 'MsgHdr')
        while True:
            msg_hdr = msg_hdr_fmt.unpack_file(self._fobj)
            if msg_hdr.msg_type in (1,31):
                dt = nexrad_to_datetime(msg_hdr.date, msg_hdr.time_ms)
                print msg_hdr, str(dt)

            msg = self._fobj.read(2 * msg_hdr.size_hw)
            try:
                getattr(self, '_decode_msg%d' % msg_hdr.msg_type)(msg)
            except AttributeError:
                pass
#                print 'Unknown Message Type: %d' % msg_hdr.msg_type
            if msg_hdr.msg_type != 31:
                self._fobj.read(self.AR2_BLOCKSIZE - len(msg))

    msg31_data_hdr_fields = [('stid', '4s'), ('time_ms', 'L'), ('date', 'H'),
                             ('az_num', 'H'), ('az_angle', 'f'),
                             ('compression', 'B'), (None, 'x'),
                             ('rad_length', 'H'), ('az_spacing', 'B'),
                             ('rad_status', 'B'), ('el_num', 'B'),
                             ('sector_num', 'B'), ('el_angle', 'f'),
                             ('spot_blanking', 'B'), ('az_index_mode', 'B'),
                             ('num_data_blks', 'H'), ('vol_const_ptr', 'L'),
                             ('el_const_ptr', 'L'), ('rad_const_ptr', 'L'),
                             ('ref_ptr', 'L'), ('vel_ptr', 'L'),
                             ('sw_ptr', 'L'), ('zdr_ptr', 'L'),
                             ('phi_ptr', 'L'), ('rho_ptr', 'L')]
    msg31_data_hdr_fmt = NamedStruct(msg31_data_hdr_fields, '>', 'Msg31DataHdr')

    msg31_vol_const_fields = [('type', 's'), ('name', '3s'), ('size', 'H'),
                              ('major', 'B'), ('minor', 'B'), ('lat', 'f'),
                              ('lon', 'f'), ('site_amsl', 'h'),
                              ('feedhorn_agl', 'H'), ('calib_dbz', 'f'),
                              ('txpower_h', 'f'), ('txpower_v', 'f'),
                              ('sys_zdr', 'f'), ('phidp0', 'f'),
                              ('vcp', 'H'), (None, '2x')]
    msg31_vol_const_fmt = NamedStruct(msg31_vol_const_fields, '>', 'VolConsts')

    msg31_el_const_fields = [('type', 's'), ('name', '3s'), ('size', 'H'),
                             ('atmos_atten', 'h'), ('calib_dbz0', 'f')]
    msg31_el_const_fmt = NamedStruct(msg31_el_const_fields, '>', 'ElConsts')

    rad_const_fields = [('type', 's'), ('name', '3s'), ('size', 'H'),
                        ('unamb_range', 'H'), ('noise_h', 'f'),
                        ('noise_v', 'f'), ('nyq_vel', 'H'), (None, '2x')]
    rad_const_fmt = NamedStruct(rad_const_fields, '>', 'RadConsts')
    def _decode_msg31(self, msg):
        print len(msg)
        data_hdr = self.msg31_data_hdr_fmt.unpack_from(msg)
        print data_hdr
        vol_consts = self.msg31_vol_const_fmt.unpack_from(msg,
            data_hdr.vol_const_ptr)
        print vol_consts
        el_consts = self.msg31_el_const_fmt.unpack_from(msg,
            data_hdr.el_const_ptr)
        print el_consts
        rad_consts = self.rad_const_fmt.unpack_from(msg, data_hdr.rad_const_ptr)
        print rad_consts

    def _decode_msg1(self, msg):
        pass

class IOBuffer(object):
    def __init__(self, source):
        self._data = source
        self._offset = 0
        self._bookmarks = []
        self._int_codes = {}

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

    def splice(self, mark, newdata):
        self.jump_to(mark)
        self._data = self._data[:self._offset] + newdata

    def read_struct(self, struct_class):
        struct = struct_class.unpack_from(self._data, self._offset)
        self.skip(struct_class.size)
        return struct

    def read_func(self, func, num_bytes=None):
        # only advance if func succeeds
        res = func(self.get_next(num_bytes))
        self.skip(num_bytes)
        return res

    def read_binary(self, num_bytes=None, signed=True):
        return map(ord, self.read(num_bytes))

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

    def __len__(self):
        return len(self._data)


class NamedStruct(Struct):
    def __init__(self, info, prefmt='', tuple_name=None):
        if tuple_name is None:
            tuple_name = 'NamedStruct'
        names, fmts = zip(*info)
        self._tuple = namedtuple(tuple_name, ' '.join(n for n in names if n))
        Struct.__init__(self, prefmt + ''.join(fmts))

    def unpack(self, s):
        return self._tuple(*Struct.unpack(self, s))

    def unpack_from(self, buff, offset=0):
        return self._tuple(*Struct.unpack_from(self, buff, offset))

    def unpack_file(self, fobj):
        bytes = fobj.read(self.size)
        return self.unpack(bytes)


def nexrad_to_datetime(julian_date, ms_midnight):
    #Subtracting one from julian_date is because epoch date is 1
    return datetime.datetime.fromtimestamp((julian_date - 1) * day
        + ms_midnight * milli)

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
    comb = combine_elem(ind1, ind2)
    return lambda x: float(comb(x))

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

class Level3File(object):
    wmo_finder = re.compile('(?:NX|SD)US\d{2}[\s\w\d]+\w*(\w{3})\r\r\n')
    header_fmt = NamedStruct([('code', 'H'), ('date', 'H'), ('time', 'l'),
        ('msg_len', 'L'), ('src_id', 'h'), ('dest_id', 'h'),
        ('num_blks', 'H')], '>', 'MsgHdr')
    # See figure 3-17 in 2620001 document for definition of status bit fields
    gsm_fmt = NamedStruct([('divider', 'h'), ('block_len', 'H'),
        ('op_mode', 'h'), ('rda_op_status', 'h'), ('vcp', 'h'), ('num_el', 'h'),
        ('el1', 'h'), ('el2', 'h'), ('el3', 'h'), ('el4', 'h'), ('el5', 'h'),
        ('el6', 'h'), ('el7', 'h'), ('el8', 'h'), ('el9', 'h'), ('el10', 'h'),
        ('el11', 'h'), ('el12', 'h'), ('el13', 'h'), ('el14', 'h'), ('el15', 'h'),
        ('el16', 'h'), ('el17', 'h'), ('el18', 'h'), ('el19', 'h'), ('el20', 'h'),
        ('rda_status', 'h'), ('rda_alarms', 'h'), ('tranmission_enable', 'h'),
        ('rpg_op_status', 'h'), ('rpg_alarms', 'h'), ('rpg_status', 'h'),
        ('rpg_narrowband_status', 'h'), ('h_ref_calib', 'h'), ('prod_avail', 'h'),
        ('super_res_cuts', 'h'), ('cmd_status', 'h'), ('v_ref_calib', 'h'),
        ('rda_build', 'h'), ('rda_channel', 'h'), ('reserved', 'h'),
        ('reserved2', 'h'), ('build_version', 'h')], '>', 'GSM')
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
    standalone_tabular = [73, 62, 75, 82]
    prod_spec_map = {16  : (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8))),
                     17  : (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8))),
                     18  : (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8))),
                     19  : (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8))),
                     20  : (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8))),
                     21  : (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8))),
                     27  : (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4)),
                     32  : (('max', 3), ('avg_time', date_elem(4, 5)), ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9))),
                     37  : (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('calib_const', float_elem(7, 8))),
                     41  : (('el_angle', scaled_elem(2, 0.1)), ('max', scaled_elem(3, 1000))), # Max in ft
                     48  : (('max', 3), ('dir_max', 4), ('alt_max', scaled_elem(5, 10))), # Max in ft
                     56  : (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4),
                            ('source', 5), ('avg_speed', scaled_elem(7, 0.1)), ('avg_dir', scaled_elem(8, 0.1))),
                     57  : (('el_angle', scaled_elem(2, 0.1)), ('max', 3)), # Max in kg / m^2
                     58  : (('num_storms', 3),),
                     61  : (('num_tvs', 3), ('num_etvs', 4)),
                     62  : (),
                     78  : (('max_rainfall', scaled_elem(3, 0.1)), ('bias', scaled_elem(4, 0.01)),
                            ('gr_pairs', scaled_elem(5, 0.01)), ('rainfall_end', date_elem(6, 7))),
                     79  : (('max_rainfall', scaled_elem(3, 0.1)), ('bias', scaled_elem(4, 0.01)),
                            ('gr_pairs', scaled_elem(5, 0.01)), ('rainfall_end', date_elem(6, 7))),
                     80  : (('max_rainfall', scaled_elem(3, 0.1)), ('rainfall_begin', date_elem(4, 5)),
                            ('rainfall_end', date_elem(6, 7)), ('bias', scaled_elem(8, 0.01)),
                            ('gr_pairs', scaled_elem(5, 0.01))),
                     81  : (('max_rainfall', scaled_elem(3, 0.001)), ('bias', scaled_elem(4, 0.01)),
                            ('gr_pairs', scaled_elem(5, 0.01)), ('rainfall_end', date_elem(6, 7))),
                     94  : (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     99  : (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     134 : (('el_angle', scaled_elem(2, 0.1)), ('max', 3), ('num_edited', 4),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     135 : (('el_angle', scaled_elem(2, 0.1)), ('max', scaled_elem(3, 1000)), # Max in ft
                            ('num_edited', 4), ('ref_thresh', 5), ('points_removed', 6),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     138 : (('rainfall_begin', date_elem(0, 1)), ('bias', scaled_elem(2, 0.01)),
                            ('max', scaled_elem(3, 0.01)), ('rainfall_end', date_elem(4, 5)),
                            ('gr_pairs', scaled_elem(6, 0.01)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     141 : (('min_ref_thresh', 0), ('overlap_display_filter', 1), ('min_strength_rank', 2)),
                     158 : (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.1)), ('max', scaled_elem(4, 0.1))),
                     159 : (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.1)), ('max', scaled_elem(4, 0.1)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     161 : (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.00333)), ('max', scaled_elem(4, 0.00333)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     163 : (('el_angle', scaled_elem(2, 0.1)), ('min', scaled_elem(3, 0.05)), ('max', scaled_elem(4, 0.05)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     165 : (('el_angle', scaled_elem(2, 0.1)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     166 : (('el_angle', scaled_elem(2, 0.1)),),
                     169 : (('null_product', low_byte(2)), ('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 5)), ('bias', scaled_elem(6, 0.01)),
                            ('gr_pairs', scaled_elem(7, 0.01))),
                     170 : (('null_product', low_byte(2)), ('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 5)), ('bias', scaled_elem(6, 0.01)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     171 : (('rainfall_begin', date_elem(0, 1)), ('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)), ('rainfall_end', date_elem(4, 5)),
                            ('bias', scaled_elem(6, 0.01)), ('gr_pairs', scaled_elem(7, 0.01))),
                     172 : (('rainfall_begin', date_elem(0, 1)), ('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)), ('rainfall_end', date_elem(4, 5)), ('bias', scaled_elem(6, 0.01)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     173 : (('period', 1), ('missing_period', high_byte(2)),
                            ('null_product', low_byte(2)), ('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 0)), ('start_time', 5), ('bias', scaled_elem(6, 0.01)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     174 : (('max', scaled_elem(3, 0.1)), ('rainfall_end', date_elem(4, 5)), ('min', scaled_elem(6, 0.1)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     175 : (('rainfall_begin', date_elem(0, 1)), ('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)), ('rainfall_end', date_elem(4, 5)), ('min', scaled_elem(6, 0.1)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     176 : (('rainfall_begin', date_elem(0, 1)), ('precip_detected', high_byte(2)), ('need_bias', low_byte(2)),
                            ('max', 3), ('percent_filled', scaled_elem(4, 0.01)), ('max_elev', scaled_elem(5, 0.1)),
                            ('bias', scaled_elem(6, 0.01)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     177 : (('mode_filter_size', 3), ('hybrid_percent_filled', 4), ('max_elev', scaled_elem(5, 0.1)),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     181 : (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
#                            ('calib_const', float_elem(7, 8))),
                            ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     182 : (('el_angle', scaled_elem(2, 0.1)), ('min', 3), ('max', 4),
                           ('compression', 7), ('uncompressed_size', combine_elem(8, 9))),
                     186 : (('el_angle', scaled_elem(2, 0.1)), ('max', 3),
                           ('compression', 7), ('uncompressed_size', combine_elem(8, 9)))}

    def __init__(self, fname):
        # Just read in the entire set of data at once
        self._filename = fname
        self._buffer = IOBuffer.fromfile(open(fname, 'rb'))

        # Pop off the WMO header if we find it
        self._process_WMO_header()

        # Pop off last 4 bytes if necessary
        self._process_end_bytes()

        # Decompress the data if necessary, and if so, pop off new header
        self._buffer = IOBuffer(self._buffer.read_func(zlib_decompress_all_frames))
        self._process_WMO_header()

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
        for name,block in self.prod_spec_map.get(self.header.code,
                                                 (('el_angle', scaled_elem(2, 0.1)),
                                                  ('compression', 7), ('uncompressed_size', combine_elem(8, 9)),
                                                  ('defaultVals', 0))):
            if callable(block):
                self.metadata[name] = block(self.depVals)
            else:
                self.metadata[name] = self.depVals[block]

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
        else:
            if self.prod_desc.sym_off:
                self._unpack_symblock(msg_start, 2 * self.prod_desc.sym_off)
            if self.prod_desc.graph_off:
                self._unpack_graphblock(msg_start, 2 * self.prod_desc.graph_off)
            if self.prod_desc.tab_off:
                self._unpack_tabblock(msg_start, 2 * self.prod_desc.tab_off)

        if 'defaultVals' in self.metadata:
            warnings.warn("{}: Using default metadata for product {}".format(self._filename, self.header.code))

    def _process_WMO_header(self):
        # Read off the WMO header if necessary
        data = self._buffer.get_next(64)
        match = self.wmo_finder.match(data)
        if match:
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
            unpacked.extend([self.thresholds[val]]*num)
        return unpacked

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
                    layer.append(self.packet_map[packet_code](self, packet_code))
                else:
                    warnings.warn('{0}: Unknown symbology packet type {1}/{1:#x}.'.format(self._filename, packet_code))
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
                    packets.append(self.packet_map[packet_code](self, packet_code))
                else:
                    warnings.warn('{0}: Unknown graphical packet type {1}/{1:#x}.'.format(self._filename, packet_code))
                    self._buffer.skip(page_size)
            self.graph_pages.append(packets)

    def _unpack_standalone_graphblock(self, start, offset):
        self._buffer.jump_to(start, offset)
        packets = []
        while not self._buffer.at_end():
            packet_code = self._buffer.read_int('>H')
            if packet_code in self.packet_map:
                packets.append(self.packet_map[packet_code](self, packet_code))
            else:
                warnings.warn('{0}: Unknown standalone graphical packet type {1}/{1:#x}.'.format(self._filename, packet_code))
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
                lines.append(''.join(self._buffer.read(num_chars)))
                num_chars = self._buffer.read_int('>h')
            self.tab_pages.append('\n'.join(lines))

        if haveHeader:
            assert self._buffer.offset_from(block_start) == header.block_len

    def __repr__(self):
        return self._filename + ': ' + '\n'.join(map(str, [self.header, self.prod_desc, self.thresholds,
                                                           self.depVals, self.metadata,
                                                           (self.siteID, self.lat, self.lon, self.height)]))

    def _unpack_packet_radial_data(self, code):
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
        return dict(start_az=list(start), end_az=list(end), data=list(vals))

    def _unpack_packet_digital_radial(self, code):
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
        return dict(start_az=list(start), end_az=list(end), data=list(vals))

    def _unpack_packet_raster_data(self, code):
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
        return dict(start_x=hdr.j_start * hdr.xscale_int,
                    start_y=hdr.i_start * hdr.yscale_int, data=rows)

    def _unpack_packet_uniform_text(self, code):
        # By not using a struct, we can handle multiple codes
        num_bytes = self._buffer.read_int('>H')
        if code == 8:
            value = self._buffer.read_int('>H')
            read_bytes = 6
        else:
            value = None
            read_bytes = 4
        i_start = self._buffer.read_int('>H')
        j_start = self._buffer.read_int('>H')

        # Text is what remains beyond what's been read, not including byte count
        text = ''.join(self._buffer.read(num_bytes - read_bytes))
        return dict(x=j_start, y=i_start, color=value, text=text)

    def _unpack_packet_special_text_symbol(self, code):
        d = self._unpack_packet_uniform_text(code)

        # Translate special characters to their meaning
        ret = dict()
        symbol_map = {'!': 'past storm position', '"': 'current storm position', '#': 'forecast storm position',
                      '$': 'past MDA position', '%': 'forecast MDA position', ' ': None}

        # Use this meaning as the key in the returned packet
        for c in d['text']:
            if c not in symbol_map:
                warnings.warn('{0}: Unknown special symbol {1}/{2:#x}.'.format(self._filename, c, ord(c)))
            else:
                key = symbol_map[c]
                if key:
                    ret[key] = d['x'], d['y']
        del d['text']

        return ret

    def _unpack_packet_special_graphic_symbol(self, code):
        type_map = {3: 'Mesocyclone', 11: '3D Correlated Shear', 12: 'TVS', 26:'ETVS', 13:'Positive Hail',
                    14: 'Probable Hail', 15: 'Storm ID', 19: 'HDA', 25: 'STI Circle'}
        point_feature_map = {1: 'Mesocyclone (ext.)', 3: 'Mesocyclone', 5: 'TVS (Ext.)', 6: 'ETVS (Ext.)',
                             7: 'TVS', 8: 'ETVS', 9: 'MDA', 10: 'MDA (Elev.)', 11: 'MDA (Weak)'}

        # Read the number of bytes and set a mark for sanity checking
        num_bytes = self._buffer.read_int('>H')
        packet_data_start = self._buffer.set_mark()

        # Loop over the bytes we have
        ret = dict()
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            # Read position
            ret.setdefault('y', list()).append(self._buffer.read_int('>h'))
            ret.setdefault('x', list()).append(self._buffer.read_int('>h'))

            # Handle any types that have additional info
            if code in (3, 11, 25):
                ret.setdefault('radius', list()).append(self._buffer.read_int('>h'))
            elif code == 15:
                ret.setdefault('id', list()).append(''.join(self._buffer.read(2)))
            elif code == 19:
                ret.setdefault('POH', list()).append(self._buffer.read_int('>h'))
                ret.setdefault('POSH', list()).append(self._buffer.read_int('>h'))
                ret.setdefault('Max Size', list()).append(self._buffer.read_int('>H'))
            elif code == 20:
                kind = self._buffer.read_int('>H')
                attr = self._buffer.read_int('>H')
                if kind < 5 or kind > 8:
                    ret.setdefault('radius', list()).append(attr)
                    if kind not in point_feature_map:
                        warnings.warn('{0}: Unknown graphic symbol point kind {1}/{1:#x}.'.format(self._filename, kind))
                        ret.setdefault('type', list()).append('Unknown')
                    else:
                        ret.setdefault('type', list()).append(point_feature_map[kind])

                ret.setdefault('type', list()).append(kind)

        # Map the code to a name for this type of symbol
        if code != 20:
            if code not in type_map:
                warnings.warn('{0}: Unknown graphic symbol type {1}/{1:#x}.'.format(self._filename, code))
                ret['type'] = 'Unknown'
            else:
                ret['type'] = type_map[code]

        # Check and return
        assert self._buffer.offset_from(packet_data_start) == num_bytes
        return ret

    def _unpack_packet_scit(self, code):
        num_bytes = self._buffer.read_int('>H')
        packet_data_start = self._buffer.set_mark()
        ret = dict()
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            next_code = self._buffer.read_int('>H')
            if next_code not in self.packet_map:
                warnings.warn('{0}: Unknown packet in SCIT {1}/{1:#x}.'.format(self._filename, next_code))
                self._buffer.jump_to(packet_data_start, num_bytes)
                return ret
            else:
                next_packet = self.packet_map[next_code](self, next_code)
                if next_code == 6:
                    ret['track'] = next_packet['vectors']
                elif next_code == 25:
                    ret['STI Circle'] = next_packet
                elif next_code == 2:
                    ret['marker'] = next_packet
                else:
                    warnings.warn('{0}: Unsupported packet in SCIT {1}/{1:#x}.'.format(self._filename, next_code))
                    ret['data'] = next_packet
        return ret

    def _unpack_packet_digital_precipitation(self, code):
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

    def _unpack_packet_linked_vector(self, code):
        num_bytes = self._buffer.read_int('>h')
        if code == 9:
            value = self._buffer.read_int('>h')
            num_bytes -= 2
        else:
            value = None
        bytes = self._buffer.read_binary(num_bytes)
        vectors = zip(bytes[::2], bytes[1::2])
        return dict(vectors=vectors, color=value)

    def _unpack_packet_vector(self, code):
        num_bytes = self._buffer.read_int('>h')
        if code == 10:
            value = self._buffer.read_int('>h')
            num_bytes -= 2
        else:
            value = None
        bytes = self._buffer.read_binary(num_bytes)
        vectors = zip(bytes[::4], bytes[1::4], bytes[2::4], bytes[3::4])
        return dict(vectors=vectors, color=value)

    def _unpack_packet_contour_color(self, code):
        # Check for color value indicator
        assert self._buffer.read_int('>H') == 0x0002

        # Read and return value (level) of contour
        return dict(color=self._buffer.read_int('>H'))

    def _unpack_packet_linked_contour(self, code):
        # Check for color value indicator
        assert self._buffer.read_int('>H') == 0x8000

        starty = self._buffer.read_int('>h')
        startx = self._buffer.read_int('>h')
        vectors = [(startx, starty)]
        num_vecs = self._buffer.read_int('>H') / 4
        for i in range(num_vecs):
            y = self._buffer.read_int('>h')
            x = self._buffer.read_int('>h')
            vectors.append((x, y))
        return dict(vectors=vectors)

    def _unpack_packet_wind_barbs(self, code):
        # Figure out how much to read
        num_bytes = self._buffer.read_int('>h')
        packet_data_start = self._buffer.set_mark()
        ret = dict()

        # Read while we have data, then return
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            ret.setdefault('color', list()).append(self._buffer.read_int('>h'))
            ret.setdefault('x', list()).append(self._buffer.read_int('>h'))
            ret.setdefault('y', list()).append(self._buffer.read_int('>h'))
            ret.setdefault('direc', list()).append(self._buffer.read_int('>h'))
            ret.setdefault('speed', list()).append(self._buffer.read_int('>h'))
        return ret

    def _unpack_packet_generic(self, code):
        # Reserved HW
        assert self._buffer.read_int('>h') == 0

        # Read number of bytes (2 HW) and return
        num_bytes = self._buffer.read_int('>l')
        data = ''.join(self._buffer.read(num_bytes))
        return dict(xdrdata=data)

    def _unpack_packet_trend_times(self, code):
        num_bytes = self._buffer.read_int('>h')
        return dict(times=self._read_trends())

    def _unpack_packet_cell_trend(self, code):
        code_map = ['Cell Top', 'Cell Base', 'Max Reflectivity Height',
                    'Probability of Hail', 'Probability of Severe Hail',
                    'Cell-based VIL', 'Maximum Reflectivity',
                    'Centroid Height']
        code_scales = [100, 100, 100, 1, 1, 1, 1, 100]
        num_bytes = self._buffer.read_int('>h')
        packet_data_start = self._buffer.set_mark()
        cell_id = ''.join(self._buffer.read(2))
        y = self._buffer.read_int('>h')
        x = self._buffer.read_int('>h')
        ret = dict(id=cell_id, x=x, y=y)
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            code = self._buffer.read_int('>h')
            try:
                ind = code - 1
                key = code_map[ind]
                scale = code_scales[ind]
            except IndexError:
                warnings.warn('{0}: Unsupported trend code {1}/{1:#x}.'.format(self._filename, code))
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

