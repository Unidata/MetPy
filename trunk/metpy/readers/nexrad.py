import bz2
import datetime
import gzip
from struct import Struct
from cStringIO import StringIO

import numpy as np
from scipy.constants import day, milli
from metpy.cbook import is_string_like, namedtuple

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

class Level3File(object):
    header_fmt = NamedStruct([('code', 'H'), ('date', 'H'), ('time', 'l'),
        ('msg_len', 'L'), ('src_id', 'h'), ('dest_id', 'h'),
        ('num_blks', 'H')], '>', 'MsgHdr')
    prod_desc_fmt = NamedStruct([('divider', 'h'), ('lat', 'l'), ('lon', 'l'),
        ('height', 'h'), ('prod_code', 'h'), ('op_mode', 'h'),
        ('vcp', 'h'), ('seq_num', 'H'), ('vol_num', 'H'),
        ('vol_date', 'H'), ('vol_start_time', 'l'), ('prod_gen_date', 'H'),
        ('prod_gen_time', 'l'), ('dep1', 'h'), ('dep2', 'h'), ('el_num', 'H'),
        ('dep3', 'h'), ('thr1', 'h'), ('thr2', 'h'), ('thr3', 'h'),
        ('thr4', 'h'), ('thr5', 'h'), ('thr6', 'h'), ('thr7', 'h'),
        ('thr8', 'h'), ('thr9', 'h'), ('thr10', 'h'), ('thr11', 'h'),
        ('thr12', 'h'), ('thr13', 'h'), ('thr14', 'h'), ('thr15', 'h'),
        ('thr16', 'h'), ('dep4', 'h'), ('dep5', 'h'), ('dep6', 'h'),
        ('dep7', 'h'), ('dep8', 'h'), ('dep9', 'h'), ('dep10', 'h'),
        ('n_maps', 'h'), ('sym_off', 'L'), ('graph_off', 'L'),
        ('tab_off', 'L')], '>', 'ProdDesc')
    sym_block_fmt = NamedStruct([('divider', 'h'), ('block_id', 'h'),
        ('block_len', 'L'), ('nlayer', 'H')], '>', 'SymBlock')
    sym_layer_fmt = NamedStruct([('divider', 'h'), ('length', 'L')], '>',
        'SymLayer')
    graph_block_fmt = NamedStruct([('divider', 'h'), ('block_id', 'h'),
        ('block_len', 'L'), ('npages', 'H'), ('pagenum', 'H'),
        ('page_len', 'H')], '>', 'GraphBlock')
    def __init__(self, fname):
        self.packet_map = {0xaf1f:self._unpack_packet_radial_data}

        self._fobj = open(fname, 'rb')

        # Read off the UNISYS header, the first 30 bytes
        self._start_offset = 30
        self._fobj.seek(self._start_offset)

        # Set up places to store data and metadata
        self.data = []
        self.metadata = dict()

        # Unpack the message header and the product description block
        hdr = self.header_fmt.unpack_file(self._fobj)
        desc = self.prod_desc_fmt.unpack_file(self._fobj)

        print hdr
        print desc
        self.metadata['datetime'] = nexrad_to_datetime(hdr.date, hdr.time*1000)

        # Unpack the various blocks, if present.  The factor of 2 converts from
        # 'half-words' to bytes
        if desc.sym_off:
            self._unpack_symblock(2 * desc.sym_off)
        if desc.graph_off:
            self._unpack_graphblock(2 * desc.graph_off)
        if desc.tab_off:
            self._unpack_tabblock(2 * desc.tab_off)

    def _unpack_symblock(self, offset):
        code = Struct('>H')
        self._fobj.seek(self._start_offset + offset)
        blk = self.sym_block_fmt.unpack_file(self._fobj)
        print blk
        assert blk.divider == -1
        assert blk.block_id == 1
        blk_data = self._fobj.read(blk.block_len - 6) # 6 for len and nlayers
        layer_off = 0
        for l in range(blk.nlayer):
            layer_hdr = self.sym_layer_fmt.unpack_from(blk_data, layer_off)
            print layer_hdr
            assert layer_hdr.divider == -1
            layer_off += self.sym_layer_fmt.size
            layer_data = blk_data[layer_off:layer_off + layer_hdr.length]
            layer_off += layer_hdr.length
            
            data_off = 0
            while data_off < len(layer_data):
                packet_code, = code.unpack_from(layer_data, data_off)
                data_off += code.size
                print packet_code, '%x' % packet_code
                data,size = self.packet_map[packet_code](layer_data[data_off:])
                self.data.append(data)
                data_off += size

    def _unpack_graphblock(self, offset):
        self._fobj.seek(self._start_offset + offset)
        raise NotImplementedError('Graphic block not implemented.')

    def _unpack_tabblock(self, offset):
        self._fobj.seek(self._start_offset + offset)
        raise NotImplementedError('Tabular block not implemented.')

    def _unpack_packet_radial_data(self, data):
        hdr_fmt = NamedStruct([('ind_first_bin', 'H'), ('nbins', 'H'),
            ('i_center', 'h'), ('j_center', 'h'), ('scale_factor', 'h'),
            ('num_rad', 'H')], '>', 'RadialHeader')
        rad_fmt = NamedStruct([('num_hwords', 'H'), ('start_angle', 'h'),
            ('angle_delta', 'h')], '>', 'RadialData')
        hdr = hdr_fmt.unpack_from(data)
        print hdr
        size = hdr_fmt.size
        rads = []
        for i in range(hdr.num_rad):
            rad = rad_fmt.unpack_from(data, size)
            size += rad_fmt.size
            start_az = rad.start_angle * 0.1
            end_az = start_az + rad.angle_delta * 0.1

            rad_data = data[size:size + rad.num_hwords * 2]
            # Unpack Run-length encoded data
            unpacked = []
            for run in map(ord, rad_data):
                num,val = run>>4, run&0x0F
                unpacked.extend([val]*num)
            rads.append((start_az, end_az, unpacked))
            size += rad.num_hwords * 2
        start,end,vals = zip(*rads)
        return dict(start_az=start, end_az=end, data=vals), size 

if __name__ == '__main__':
    import numpy as np
    from numpy.ma import masked_array
    import matplotlib.pyplot as plt
    from scipy.constants import degree
#    name = '/home/rmay/test_radar/KTLX20081110_220148_V03'
#    f = Level2File(name)
    name = '/home/rmay/test_radar/nids/KOUN_SDUS54_N0RTLX_200811101410'
    f = Level3File(name)
    datadict = f.data[0]
    
    ref = np.array(datadict['data'])
    ref = masked_array(ref, mask=(ref==0))
    az = np.array(datadict['start_az'])
    rng = np.arange(ref.shape[1])

    xlocs = rng * np.sin(az[:, np.newaxis] * degree)
    ylocs = rng * np.cos(az[:, np.newaxis] * degree)
    plt.pcolormesh(xlocs, ylocs, ref)
    plt.axis('equal')
    plt.show()
