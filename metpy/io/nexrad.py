# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import print_function
import bz2
import datetime
import gzip
import logging
import re
import struct
from collections import defaultdict, namedtuple, OrderedDict
from struct import Struct
from xdrlib import Unpacker

import numpy as np
from scipy.constants import day, milli
from ..cbook import is_string_like
from ..package_tools import Exporter
from .tools import (Array, BitField, Bits, DictStruct, Enum, IOBuffer, NamedStruct,
                    bits_to_code, zlib_decompress_all_frames)

exporter = Exporter(globals())

log = logging.getLogger('metpy.io.nexrad')
log.addHandler(logging.StreamHandler())  # Python 2.7 needs a handler set
log.setLevel(logging.WARNING)


def version(val):
    if val / 100. > 2.:
        ver = val / 100.
    else:
        ver = val / 10.
    return '{:.1f}'.format(ver)


def scaler(scale):
    def inner(val):
        return val * scale
    return inner


def angle(val):
    return val * 360. / 2**16


def az_rate(val):
    return val * 90. / 2**16


def bzip_blocks_decompress_all(data):
    frames = bytearray()
    offset = 0
    while offset < len(data):
        size_bytes = data[offset:offset + 4]
        offset += 4
        block_cmp_bytes = abs(Struct('>l').unpack(size_bytes)[0])
        if block_cmp_bytes:
            frames.extend(bz2.decompress(data[offset:offset + block_cmp_bytes]))
            offset += block_cmp_bytes
        else:
            frames.extend(size_bytes)
            frames.extend(data[offset:])
    return frames


def nexrad_to_datetime(julian_date, ms_midnight):
    # Subtracting one from julian_date is because epoch date is 1
    return datetime.datetime.fromtimestamp((julian_date - 1) * day +
                                           ms_midnight * milli)


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

START_ELEVATION = 0x1
END_ELEVATION = 0x2
START_VOLUME = 0x4
END_VOLUME = 0x8
LAST_ELEVATION = 0x10
BAD_DATA = 0x20


@exporter.export
class Level2File(object):
    r'''A class that handles reading the NEXRAD Level 2 data and the various
    messages that are contained within.

    This class attempts to decode every byte that is in a given data file.
    It supports both external compression, as well as the internal BZ2
    compression that is used.

    Attributes
    ----------
    stid : str
        The ID of the radar station
    dt : Datetime instance
        The date and time of the data
    vol_hdr : namedtuple
        The unpacked volume header
    sweeps : list of tuples
        Data for each of the sweeps found in the file
    rda_status : namedtuple, optional
        Unpacked RDA status information, if found
    maintenance_data : namedtuple, optional
        Unpacked maintenance data information, if found
    maintenance_data_desc : dict, optional
        Descriptions of maintenance data fields, if maintenance data present
    vcp_info : namedtuple, optional
        Unpacked VCP information, if found
    clutter_filter_bypass_map : dict, optional
        Unpacked clutter filter bypass map, if present
    rda : dict, optional
        Unpacked RDA adaptation data, if present
    rda_adaptation_desc : dict, optional
        Descriptions of RDA adaptation data, if adaptation data present

    Notes
    -----
    The internal data structure that things are decoded into is still to be
    determined.
    '''

    # Number of bytes
    AR2_BLOCKSIZE = 2432  # 12 (CTM) + 2416 (Msg hdr + data) + 4 (FCS)
    CTM_HEADER_SIZE = 12

    MISSING = float('nan')
    RANGE_FOLD = float('nan')  # TODO: Need to separate from missing

    def __init__(self, filename):
        r'''Create instance of `Level2File`.

        Parameters
        ----------
        filename : str or file-like object
            If str, the name of the file to be opened. Gzip-ed files are
            recognized with the extension '.gz', as are bzip2-ed files with
            the extension `.bz2` If `fname` is a file-like object,
            this will be read from directly.
        '''

        if is_string_like(filename):
            if filename.endswith('.bz2'):
                fobj = bz2.BZ2File(filename, 'rb')
            elif filename.endswith('.gz'):
                fobj = gzip.GzipFile(filename, 'rb')
            else:
                fobj = open(filename, 'rb')
        else:
            fobj = filename

        self._buffer = IOBuffer.fromfile(fobj)
        self._read_volume_header()
        start = self._buffer.set_mark()

        # See if we need to apply bz2 decompression
        try:
            self._buffer = IOBuffer(self._buffer.read_func(bzip_blocks_decompress_all))
        except:
            self._buffer.jump_to(start)

        # Now we're all initialized, we can proceed with reading in data
        self._read_data()

    vol_hdr_fmt = NamedStruct([('version', '9s'), ('vol_num', '3s'),
                               ('date', 'L'), ('time_ms', 'L'), ('stid', '4s')], '>', 'VolHdr')

    def _read_volume_header(self):
        self.vol_hdr = self._buffer.read_struct(self.vol_hdr_fmt)
        self.dt = nexrad_to_datetime(self.vol_hdr.date, self.vol_hdr.time_ms)
        self.stid = self.vol_hdr.stid

    msg_hdr_fmt = NamedStruct([('size_hw', 'H'),
                               ('rda_channel', 'B', BitField('Redundant Channel 1',
                                                             'Redundant Channel 2',
                                                             None, 'ORDA')),
                               ('msg_type', 'B'), ('seq_num', 'H'), ('date', 'H'),
                               ('time_ms', 'I'), ('num_segments', 'H'), ('segment_num', 'H')],
                              '>', 'MsgHdr')

    def _read_data(self):
        self._msg_buf = {}
        self.sweeps = []
        self.rda_status = []
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
                # Try to handle the message. If we don't handle it, skipping
                # past it is handled at the end anyway.
                if hasattr(self, '_decode_msg%d' % msg_hdr.msg_type):
                    getattr(self, '_decode_msg%d' % msg_hdr.msg_type)(msg_hdr)
                else:
                    log.warning('Unknown message: %d', msg_hdr.msg_type)

            # Jump to the start of the next message. This depends on whether
            # the message was legacy with fixed block size or not.
            if msg_hdr.msg_type != 31:
                # The AR2_BLOCKSIZE accounts for the CTM header before the
                # data, as well as the Frame Check Sequence (4 bytes) after
                # the end of the data
                self._buffer.jump_to(msg_start, self.AR2_BLOCKSIZE)
            else:
                # Need to include the CTM header but not FCS
                self._buffer.jump_to(msg_start,
                                     self.CTM_HEADER_SIZE + 2 * msg_hdr.size_hw)

        # Check if we have any message segments still in the buffer
        if self._msg_buf:
            log.warning('Remaining buffered messages segments for message type(s): %s',
                        ' '.join(map(str, self._msg_buf.keys())))

        del self._msg_buf

    msg1_fmt = NamedStruct([('time_ms', 'L'), ('date', 'H'),
                            ('unamb_range', 'H', scaler(0.1)), ('az_angle', 'H', angle),
                            ('az_num', 'H'), ('rad_status', 'H', remap_status),
                            ('el_angle', 'H', angle), ('el_num', 'H'),
                            ('surv_first_gate', 'h', scaler(0.001)),
                            ('doppler_first_gate', 'h', scaler(0.001)),
                            ('surv_gate_width', 'H', scaler(0.001)),
                            ('doppler_gate_width', 'H', scaler(0.001)),
                            ('surv_num_gates', 'H'), ('doppler_num_gates', 'H'),
                            ('cut_sector_num', 'H'), ('calib_dbz0', 'f'),
                            ('ref_offset', 'H'), ('vel_offset', 'H'), ('sw_offset', 'H'),
                            ('dop_res', 'H', BitField(None, 0.5, 1.0)), ('vcp', 'H'),
                            (None, '14x'), ('nyq_vel', 'H', scaler(0.01)),
                            ('atmos_atten', 'H', scaler(0.001)), ('tover', 'H', scaler(0.1)),
                            ('spot_blanking', 'B', BitField('Radial', 'Elevation', 'Volume')),
                            (None, '32x')], '>', 'Msg1Fmt')

    msg1_data_hdr = namedtuple('Msg1DataHdr',
                               'name first_gate gate_width num_gates scale offset')

    def _decode_msg1(self, msg_hdr):
        msg_start = self._buffer.set_mark()
        hdr = self._buffer.read_struct(self.msg1_fmt)
        data_dict = dict()

        # Process all data pointers:
        read_info = []
        if hdr.surv_num_gates and hdr.ref_offset:
            read_info.append((hdr.ref_offset,
                              self.msg1_data_hdr('REF', hdr.surv_first_gate,
                                                 hdr.surv_gate_width,
                                                 hdr.surv_num_gates, 2.0, 66.0)))

        if hdr.vel_offset:
            read_info.append((hdr.vel_offset,
                              self.msg1_data_hdr('VEL', hdr.doppler_first_gate,
                                                 hdr.doppler_gate_width,
                                                 hdr.doppler_num_gates,
                                                 1. / hdr.dop_res, 129.0)))

        if hdr.sw_offset:
            read_info.append((hdr.sw_offset,
                              self.msg1_data_hdr('SW', hdr.doppler_first_gate,
                                                 hdr.doppler_gate_width,
                                                 hdr.doppler_num_gates, 2.0, 129.0)))

        for ptr, data_hdr in read_info:
            # Jump and read
            self._buffer.jump_to(msg_start, ptr)
            vals = np.array(self._buffer.read_binary(data_hdr.num_gates, 'B'))

            # Scale and flag data
            scaled_vals = (vals - data_hdr.offset) / data_hdr.scale
            scaled_vals[vals == 0] = self.MISSING
            scaled_vals[vals == 1] = self.RANGE_FOLD

            # Store
            data_dict[data_hdr.name] = (data_hdr, scaled_vals)

        self._add_sweep(hdr)
        self.sweeps[-1].append((hdr, data_dict))

    msg2_fmt = NamedStruct([
        ('rda_status', 'H', BitField('None', 'Start-Up', 'Standby', 'Restart',
                                     'Operate', 'Spare', 'Off-line Operate')),
        ('op_status', 'H', BitField('Disabled', 'On-Line',
                                    'Maintenance Action Required',
                                    'Maintenance Action Mandatory',
                                    'Commanded Shut Down', 'Inoperable',
                                    'Automatic Calibration')),
        ('control_status', 'H', BitField('None', 'Local Only',
                                         'RPG (Remote) Only', 'Either')),
        ('aux_power_gen_state', 'H', BitField('Switch to Aux Power',
                                              'Utility PWR Available',
                                              'Generator On',
                                              'Transfer Switch Manual',
                                              'Commanded Switchover')),
        ('avg_tx_pwr', 'H'), ('ref_calib_cor', 'h'),
        ('data_transmission_enabled', 'H', BitField('None', 'None',
                                                    'Reflectivity', 'Velocity', 'Width')),
        ('vcp_num', 'h'), ('rda_control_auth', 'H', BitField('No Action',
                                                             'Local Control Requested',
                                                             'Remote Control Enabled')),
        ('rda_build', 'H', version), ('op_mode', 'H', BitField('None', 'Test',
                                                               'Operational', 'Maintenance')),
        ('super_res_status', 'H', BitField('None', 'Enabled', 'Disabled')),
        ('cmd_status', 'H', Bits(6)),
        ('avset_status', 'H', BitField('None', 'Enabled', 'Disabled')),
        ('rda_alarm_status', 'H', BitField('No Alarms', 'Tower/Utilities',
                                           'Pedestal', 'Transmitter', 'Receiver',
                                           'RDA Control', 'Communication',
                                           'Signal Processor')),
        ('command_acknowledge', 'H', BitField('Remote VCP Received',
                                              'Clutter Bypass map received',
                                              'Redundant Chan Ctrl Cmd received')),
        ('channel_control_status', 'H'),
        ('spot_blanking', 'H', BitField('Enabled', 'Disabled')),
        ('bypass_map_gen_date', 'H'), ('bypass_map_gen_time', 'H'),
        ('clutter_filter_map_gen_date', 'H'), ('clutter_filter_map_gen_time', 'H'),
        (None, '2x'),
        ('transition_pwr_src_state', 'H', BitField('Off', 'OK')),
        ('RMS_control_status', 'H', BitField('RMS in control', 'RDA in control')),
        # See Table IV-A for definition of alarms
        (None, '2x'), ('alarms', '28s', Array('>14H'))], '>', 'Msg2Fmt')

    def _decode_msg2(self, msg_hdr):
        self.rda_status.append(self._buffer.read_struct(self.msg2_fmt))
        self._check_size(msg_hdr, self.msg2_fmt.size)

    def _decode_msg3(self, msg_hdr):
        from .nexrad_msgs.msg3 import descriptions, fields
        self.maintenance_data_desc = descriptions
        msg_fmt = DictStruct(fields, '>')
        self.maintenance_data = self._buffer.read_struct(msg_fmt)
        self._check_size(msg_hdr, msg_fmt.size)

    vcp_fmt = NamedStruct([('size_hw', 'H'), ('pattern_type', 'H'),
                           ('num', 'H'), ('num_el_cuts', 'H'), ('clutter_map_group', 'H'),
                           ('dop_res', 'B', BitField(None, 0.5, 1.0)),
                           ('pulse_width', 'B', BitField('None', 'Short', 'Long')),
                           (None, '10x'), ('els', None)], '>', 'VCPFmt')

    vcp_el_fmt = NamedStruct([('el_angle', 'H', angle),
                              ('channel_config', 'B', Enum('Constant Phase', 'Random Phase',
                                                           'SZ2 Phase')),
                              ('waveform', 'B', Enum('None', 'Contiguous Surveillance',
                                                     'Contig. Doppler with Ambiguity Res.',
                                                     'Contig. Doppler without Ambiguity Res.',
                                                     'Batch', 'Staggered Pulse Pair')),
                              ('super_res', 'B', BitField('0.5 azimuth and 0.25km range res.',
                                                          'Doppler to 300km',
                                                          'Dual Polarization Control',
                                                          'Dual Polarization to 300km')),
                              ('surv_prf_num', 'B'), ('surv_pulse_count', 'H'),
                              ('az_rate', 'h', az_rate),
                              ('ref_thresh', 'h', scaler(0.125)),
                              ('vel_thresh', 'h', scaler(0.125)),
                              ('sw_thresh', 'h', scaler(0.125)),
                              ('zdr_thresh', 'h', scaler(0.125)),
                              ('phidp_thresh', 'h', scaler(0.125)),
                              ('rhohv_thresh', 'h', scaler(0.125)),
                              ('sector1_edge', 'H', angle),
                              ('sector1_doppler_prf_num', 'H'),
                              ('sector1_pulse_count', 'H'), (None, '2x'),
                              ('sector2_edge', 'H', angle),
                              ('sector2_doppler_prf_num', 'H'),
                              ('sector2_pulse_count', 'H'), (None, '2x'),
                              ('sector3_edge', 'H', angle),
                              ('sector3_doppler_prf_num', 'H'),
                              ('sector3_pulse_count', 'H'), (None, '2x')], '>', 'VCPEl')

    def _decode_msg5(self, msg_hdr):
        vcp_info = self._buffer.read_struct(self.vcp_fmt)
        els = [self._buffer.read_struct(self.vcp_el_fmt) for _ in range(vcp_info.num_el_cuts)]
        self.vcp_info = vcp_info._replace(els=els)
        self._check_size(msg_hdr,
                         self.vcp_fmt.size + vcp_info.num_el_cuts * self.vcp_el_fmt.size)

    def _decode_msg13(self, msg_hdr):
        data = self._buffer_segment(msg_hdr)
        if data:
            data = list(Struct('>%dh' % (len(data) / 2)).unpack(data))
            bmap = dict()
            date, time, num_el = data[:3]
            bmap['datetime'] = nexrad_to_datetime(date, time)

            offset = 3
            bmap['data'] = []
            bit_conv = Bits(16)
            for e in range(num_el):
                seg_num = data[offset]
                offset += 1
                assert seg_num == (e + 1), ('Message 13 segments out of sync --'
                                            ' read %d but on %d' % (seg_num, e + 1))

                az_data = []
                for _ in range(360):
                    gates = []
                    for _ in range(32):
                        gates.extend(bit_conv(data[offset]))
                        offset += 1
                    az_data.append(gates)
                bmap['data'].append(az_data)

            self.clutter_filter_bypass_map = bmap

            if offset != len(data):
                log.warning('Message 13 left data -- Used: %d Avail: %d', offset, len(data))

    msg15_code_map = {0: 'Bypass Filter', 1: 'Bypass map in Control',
                      2: 'Force Filter'}

    def _decode_msg15(self, msg_hdr):
        # buffer the segments until we have the whole thing. The data
        # will be returned concatenated when this is the case
        data = self._buffer_segment(msg_hdr)
        if data:
            data = list(Struct('>%dh' % (len(data) / 2)).unpack(data))
            cmap = dict()
            date, time, num_el = data[:3]
            cmap['datetime'] = nexrad_to_datetime(date, time)

            offset = 3
            cmap['data'] = []
            for _ in range(num_el):
                az_data = []
                for _ in range(360):
                    num_rng = data[offset]
                    offset += 1

                    codes = data[offset:2 * num_rng + offset:2]
                    offset += 1

                    ends = data[offset:2 * num_rng + offset:2]
                    offset += 2 * num_rng - 1
                    az_data.append(list(zip(ends, codes)))
                cmap['data'].append(az_data)

            self.clutter_filter_map = cmap
            if offset != len(data):
                log.warning('Message 15 left data -- Used: %d Avail: %d', offset, len(data))

    def _decode_msg18(self, msg_hdr):
        # buffer the segments until we have the whole thing. The data
        # will be returned concatenated when this is the case
        data = self._buffer_segment(msg_hdr)
        if data:
            from .nexrad_msgs.msg18 import descriptions, fields
            self.rda_adaptation_desc = descriptions

            # Can't use NamedStruct because we have more than 255 items--this
            # is a CPython limit for arguments.
            msg_fmt = DictStruct(fields, '>')
            self.rda = msg_fmt.unpack(data)
            for num in (11, 21, 31, 32, 300, 301):
                attr = 'VCPAT%d' % num
                dat = self.rda[attr]
                vcp_hdr = self.vcp_fmt.unpack_from(dat, 0)
                off = self.vcp_fmt.size
                els = []
                for i in range(vcp_hdr.num_el_cuts):
                    els.append(self.vcp_el_fmt.unpack_from(dat, off))
                    off += self.vcp_el_fmt.size
                self.rda[attr] = vcp_hdr._replace(els=els)

    msg31_data_hdr_fmt = NamedStruct([('stid', '4s'), ('time_ms', 'L'),
                                      ('date', 'H'), ('az_num', 'H'),
                                      ('az_angle', 'f'), ('compression', 'B'),
                                      (None, 'x'), ('rad_length', 'H'),
                                      ('az_spacing', 'B'),
                                      ('rad_status', 'B', remap_status),
                                      ('el_num', 'B'), ('sector_num', 'B'),
                                      ('el_angle', 'f'),
                                      ('spot_blanking', 'B', BitField('Radial', 'Elevation',
                                                                      'Volume')),
                                      ('az_index_mode', 'B', scaler(0.01)),
                                      ('num_data_blks', 'H'),
                                      ('vol_const_ptr', 'L'), ('el_const_ptr', 'L'),
                                      ('rad_const_ptr', 'L')], '>', 'Msg31DataHdr')

    msg31_vol_const_fmt = NamedStruct([('type', 's'), ('name', '3s'),
                                       ('size', 'H'), ('major', 'B'),
                                       ('minor', 'B'), ('lat', 'f'), ('lon', 'f'),
                                       ('site_amsl', 'h'), ('feedhorn_agl', 'H'),
                                       ('calib_dbz', 'f'), ('txpower_h', 'f'),
                                       ('txpower_v', 'f'), ('sys_zdr', 'f'),
                                       ('phidp0', 'f'), ('vcp', 'H'),
                                       ('processing_status', 'H', BitField('RxR Noise',
                                                                           'CBT'))],
                                      '>', 'VolConsts')

    msg31_el_const_fmt = NamedStruct([('type', 's'), ('name', '3s'),
                                      ('size', 'H'), ('atmos_atten', 'h', scaler(0.001)),
                                      ('calib_dbz0', 'f')], '>', 'ElConsts')

    rad_const_fmt_v1 = NamedStruct([('type', 's'), ('name', '3s'), ('size', 'H'),
                                    ('unamb_range', 'H', scaler(0.1)),
                                    ('noise_h', 'f'), ('noise_v', 'f'),
                                    ('nyq_vel', 'H', scaler(0.01)),
                                    (None, '2x')], '>', 'RadConstsV1')
    rad_const_fmt_v2 = NamedStruct([('type', 's'), ('name', '3s'), ('size', 'H'),
                                    ('unamb_range', 'H', scaler(0.1)),
                                    ('noise_h', 'f'), ('noise_v', 'f'),
                                    ('nyq_vel', 'H', scaler(0.01)),
                                    (None, '2x'), ('calib_dbz0_h', 'f'),
                                    ('calib_dbz0_v', 'f')], '>', 'RadConstsV2')

    data_block_fmt = NamedStruct([('type', 's'), ('name', '3s'),
                                  ('reserved', 'L'), ('num_gates', 'H'),
                                  ('first_gate', 'H', scaler(0.001)),
                                  ('gate_width', 'H', scaler(0.001)),
                                  ('tover', 'H', scaler(0.1)),
                                  ('snr_thresh', 'h', scaler(0.1)),
                                  ('recombined', 'B', BitField('Azimuths', 'Gates')),
                                  ('data_size', 'B', bits_to_code),
                                  ('scale', 'f'), ('offset', 'f')], '>', 'DataBlockHdr')

    def _decode_msg31(self, msg_hdr):
        msg_start = self._buffer.set_mark()
        data_hdr = self._buffer.read_struct(self.msg31_data_hdr_fmt)

        # Read all the data block pointers separately. This simplifies just
        # iterating over them
        ptrs = self._buffer.read_binary(6, '>L')

        assert data_hdr.compression == 0, 'Compressed message 31 not supported!'

        self._buffer.jump_to(msg_start, data_hdr.vol_const_ptr)
        vol_consts = self._buffer.read_struct(self.msg31_vol_const_fmt)

        self._buffer.jump_to(msg_start, data_hdr.el_const_ptr)
        el_consts = self._buffer.read_struct(self.msg31_el_const_fmt)

        self._buffer.jump_to(msg_start, data_hdr.rad_const_ptr)
        # Major version jumped with Build 14.0
        if vol_consts.major < 2:
            rad_consts = self._buffer.read_struct(self.rad_const_fmt_v1)
        else:
            rad_consts = self._buffer.read_struct(self.rad_const_fmt_v2)

        data = dict()
        block_count = 3
        for ptr in ptrs:
            if ptr:
                block_count += 1
                self._buffer.jump_to(msg_start, ptr)
                hdr = self._buffer.read_struct(self.data_block_fmt)
                vals = np.array(self._buffer.read_binary(hdr.num_gates,
                                                         '>' + hdr.data_size))
                scaled_vals = (vals - hdr.offset) / hdr.scale
                scaled_vals[vals == 0] = self.MISSING
                scaled_vals[vals == 1] = self.RANGE_FOLD
                data[hdr.name.strip()] = (hdr, scaled_vals)

        self._add_sweep(data_hdr)

        self.sweeps[-1].append((data_hdr, vol_consts, el_consts, rad_consts, data))

        if data_hdr.num_data_blks != block_count:
            log.warning('Incorrect number of blocks detected -- Got %d'
                        'instead of %d', block_count, data_hdr.num_data_blks)
        assert data_hdr.rad_length == self._buffer.offset_from(msg_start)

    def _buffer_segment(self, msg_hdr):
        # Add to the buffer
        bufs = self._msg_buf.setdefault(msg_hdr.msg_type, dict())
        bufs[msg_hdr.segment_num] = self._buffer.read(2 * msg_hdr.size_hw -
                                                      self.msg_hdr_fmt.size)

        # Warn for badly formatted data
        if len(bufs) != msg_hdr.segment_num:
            log.warning('Segment out of order (Got: %d Count: %d) for message type %d.',
                        msg_hdr.segment_num, len(bufs), msg_hdr.msg_type)

        # If we're complete, return the full collection of data
        if msg_hdr.num_segments == len(bufs):
            self._msg_buf.pop(msg_hdr.msg_type)
            return b''.join(bytes(item[1]) for item in sorted(bufs.items()))

    def _add_sweep(self, hdr):
        if not self.sweeps and not hdr.rad_status & START_VOLUME:
            log.warning('Missed start of volume!')

        if hdr.rad_status & START_ELEVATION:
            self.sweeps.append([])

        if len(self.sweeps) != hdr.el_num:
            log.warning('Missed elevation -- Have %d but data on %d.'
                        ' Compensating...', len(self.sweeps), hdr.el_num)
            while len(self.sweeps) < hdr.el_num:
                self.sweeps.append([])

    def _check_size(self, msg_hdr, size):
        hdr_size = msg_hdr.size_hw * 2 - self.msg_hdr_fmt.size
        assert size == hdr_size, ('Message type %d should be %d bytes but got %d' %
                                  (msg_hdr.msg_type, size, hdr_size))


def reduce_lists(d):
    for field in d:
        old_data = d[field]
        if len(old_data) == 1:
            d[field] = old_data[0]


def two_comp16(val):
    if val >> 15:
        val = -(~val & 0x7fff) - 1
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
    return struct.unpack('>f', struct.pack('>HH', short1, short2))[0]


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
    # Masking below in python will properly convert signed values to unsigned
    return lambda seq: float32(seq[ind1] & 0xFFFF, seq[ind2] & 0xFFFF)


def high_byte(ind):
    def inner(seq):
        return seq[ind] >> 8
    return inner


def low_byte(ind):
    def inner(seq):
        return seq[ind] & 0x00FF
    return inner


# Data mappers used to take packed data and turn into physical units
# Default is to use numpy array indexing to use LUT to change data bytes
# into physical values. Can also have a 'labels' attribute to give
# categorical labels
class DataMapper(object):
    # Need to find way to handle range folded
    # RANGE_FOLD = -9999
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


# 156, 157
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
            codes, val = t >> 8, t & 0xFF
            label = ''
            if codes >> 7:
                label = self.lut_names[val]
                if label in ('Blank', 'TH', 'ND'):
                    val = self.MISSING
                elif label == 'RF':
                    val = self.RANGE_FOLD

            elif codes >> 6:
                val *= 0.01
                label = '%.2f' % val
            elif codes >> 5:
                val *= 0.05
                label = '%.2f' % val
            elif codes >> 4:
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


@exporter.export
class Level3File(object):
    r'''A class that handles reading the wide array of NEXRAD Level 3 (NIDS)
    product files.

    This class attempts to decode every byte that is in a given product file.
    It supports all of the various compression formats that exist for these
    products in the wild.

    Attributes
    ----------
    metadata : dict
        Various general metadata available from the product
    header : namedtuple
        Decoded product header
    prod_desc : namedtuple
        Decoded product description block
    siteID : str
        ID of the site found in the header, empty string if none found
    lat : float
        Radar site latitude
    lon : float
        Radar site longitude
    height : float
        Radar site height AMSL
    product_name : str
        Name of the product contained in file
    max_range : float
        Maximum range of the product, taken from the NIDS ICD
    map_data : Mapper
        Class instance mapping data int values to proper floating point values
    sym_block : list, optional
        Any symbology block packets that were found
    tab_pages : list, optional
        Any tabular pages that were found
    graph_pages : list, optional
        Any graphical pages that were found

    Notes
    -----
    The internal data structure that things are decoded into is still to be
    determined.
    '''

    ij_to_km = 0.25
    wmo_finder = re.compile('((?:NX|SD|NO)US)\d{2}[\s\w\d]+\w*(\w{3})\r\r\n')
    header_fmt = NamedStruct([('code', 'H'), ('date', 'H'), ('time', 'l'),
                              ('msg_len', 'L'), ('src_id', 'h'), ('dest_id', 'h'),
                              ('num_blks', 'H')], '>', 'MsgHdr')
    # See figure 3-17 in 2620001 document for definition of status bit fields
    gsm_fmt = NamedStruct([('divider', 'h'), ('block_len', 'H'),
                           ('op_mode', 'h', BitField('Clear Air', 'Precip')),
                           ('rda_op_status', 'h', BitField('Spare', 'Online',
                                                           'Maintenance Required',
                                                           'Maintenance Mandatory',
                                                           'Commanded Shutdown', 'Inoperable',
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
                           ('rda_status', 'h', BitField('Spare', 'Startup', 'Standby',
                                                        'Restart', 'Operate',
                                                        'Off-line Operate')),
                           ('rda_alarms', 'h', BitField('Indeterminate', 'Tower/Utilities',
                                                        'Pedestal', 'Transmitter', 'Receiver',
                                                        'RDA Control', 'RDA Communications',
                                                        'Signal Processor')),
                           ('tranmission_enable', 'h', BitField('Spare', 'None',
                                                                'Reflectivity',
                                                                'Velocity', 'Spectrum Width',
                                                                'Dual Pol')),
                           ('rpg_op_status', 'h', BitField('Loadshed', 'Online',
                                                           'Maintenance Required',
                                                           'Maintenance Mandatory',
                                                           'Commanded shutdown')),
                           ('rpg_alarms', 'h', BitField('None', 'Node Connectivity',
                                                        'Wideband Failure',
                                                        'RPG Control Task Failure',
                                                        'Data Base Failure', 'Spare',
                                                        'RPG Input Buffer Loadshed',
                                                        'Spare', 'Product Storage Loadshed'
                                                        'Spare', 'Spare', 'Spare',
                                                        'RPG/RPG Intercomputer Link Failure',
                                                        'Redundant Channel Error',
                                                        'Task Failure', 'Media Failure')),
                           ('rpg_status', 'h', BitField('Restart', 'Operate', 'Standby')),
                           ('rpg_narrowband_status', 'h', BitField('Commanded Disconnect',
                                                                   'Narrowband Loadshed')),
                           ('h_ref_calib', 'h', scaler(0.25)),
                           ('prod_avail', 'h', BitField('Product Availability',
                                                        'Degraded Availability',
                                                        'Not Available')),
                           ('super_res_cuts', 'h', Bits(16)),
                           ('cmd_status', 'h', Bits(6)),
                           ('v_ref_calib', 'h', scaler(0.25)),
                           ('rda_build', 'h', version), ('rda_channel', 'h'),
                           ('reserved', 'h'), ('reserved2', 'h'),
                           ('build_version', 'h', version)], '>', 'GSM')
    # Build 14.0 added more bytes to the GSM
    additional_gsm_fmt = NamedStruct([('el21', 'h', scaler(0.1)),
                                      ('el22', 'h', scaler(0.1)),
                                      ('el23', 'h', scaler(0.1)),
                                      ('el24', 'h', scaler(0.1)),
                                      ('el25', 'h', scaler(0.1)),
                                      ('vcp_supplemental', 'H', BitField('AVSET',
                                                                         'SAILS',
                                                                         'site_vcp',
                                                                         'RxR Noise',
                                                                         'CBT')),
                                      ('spare', '84s')], '>', 'GSM')
    prod_desc_fmt = NamedStruct([('divider', 'h'), ('lat', 'l'), ('lon', 'l'),
                                 ('height', 'h'), ('prod_code', 'h'),
                                 ('op_mode', 'h'), ('vcp', 'h'), ('seq_num', 'H'),
                                 ('vol_num', 'H'), ('vol_date', 'H'),
                                 ('vol_start_time', 'l'), ('prod_gen_date', 'H'),
                                 ('prod_gen_time', 'l'), ('dep1', 'h'),
                                 ('dep2', 'h'), ('el_num', 'H'), ('dep3', 'h'),
                                 ('thr1', 'H'), ('thr2', 'H'), ('thr3', 'H'),
                                 ('thr4', 'H'), ('thr5', 'H'), ('thr6', 'H'),
                                 ('thr7', 'H'), ('thr8', 'H'), ('thr9', 'H'),
                                 ('thr10', 'H'), ('thr11', 'H'), ('thr12', 'H'),
                                 ('thr13', 'H'), ('thr14', 'H'), ('thr15', 'H'),
                                 ('thr16', 'H'), ('dep4', 'h'), ('dep5', 'h'),
                                 ('dep6', 'h'), ('dep7', 'h'), ('dep8', 'h'),
                                 ('dep9', 'h'), ('dep10', 'h'), ('version', 'b'),
                                 ('spot_blank', 'b'), ('sym_off', 'L'), ('graph_off', 'L'),
                                 ('tab_off', 'L')], '>', 'ProdDesc')
    sym_block_fmt = NamedStruct([('divider', 'h'), ('block_id', 'h'),
                                 ('block_len', 'L'), ('nlayer', 'H')], '>', 'SymBlock')
    tab_header_fmt = NamedStruct([('divider', 'h'), ('block_id', 'h'),
                                  ('block_len', 'L')], '>', 'TabHeader')
    tab_block_fmt = NamedStruct([('divider', 'h'), ('num_pages', 'h')], '>', 'TabBlock')
    sym_layer_fmt = NamedStruct([('divider', 'h'), ('length', 'L')], '>',
                                'SymLayer')
    graph_block_fmt = NamedStruct([('divider', 'h'), ('block_id', 'h'),
                                   ('block_len', 'L'), ('num_pages', 'H')], '>', 'GraphBlock')
    standalone_tabular = [62, 73, 75, 82]
    prod_spec_map = {16: ('Base Reflectivity', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     17: ('Base Reflectivity', 460., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     18: ('Base Reflectivity', 460., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     19: ('Base Reflectivity', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     20: ('Base Reflectivity', 460., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     21: ('Base Reflectivity', 460., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     22: ('Base Velocity', 60., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3), ('max', 4))),
                     23: ('Base Velocity', 115., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3), ('max', 4))),
                     24: ('Base Velocity', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3), ('max', 4))),
                     25: ('Base Velocity', 60., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3), ('max', 4))),
                     26: ('Base Velocity', 115., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3), ('max', 4))),
                     27: ('Base Velocity', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3), ('max', 4))),
                     28: ('Base Spectrum Width', 60., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3))),
                     29: ('Base Spectrum Width', 115., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3))),
                     30: ('Base Spectrum Width', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3))),
                     31: ('User Selectable Storm Total Precipitation', 230., LegacyMapper,
                          (('end_hour', 0),
                           ('hour_span', 1),
                           ('null_product', 2),
                           ('max_rainfall', scaled_elem(3, 0.1)),
                           ('rainfall_begin', date_elem(4, 5)),
                           ('rainfall_end', date_elem(6, 7)),
                           ('bias', scaled_elem(8, 0.01)),
                           ('gr_pairs', scaled_elem(5, 0.01)))),
                     32: ('Digital Hybrid Scan Reflectivity', 230., DigitalRefMapper,
                          (('max', 3),
                           ('avg_time', date_elem(4, 5)),
                           ('compression', 7),
                           ('uncompressed_size', combine_elem(8, 9)))),
                     33: ('Hybrid Scan Reflectivity', 230., LegacyMapper,
                          (('max', 3), ('avg_time', date_elem(4, 5)))),
                     34: ('Clutter Filter Control', 230., LegacyMapper,
                          (('clutter_bitmap', 0),
                           ('cmd_map', 1),
                           ('bypass_map_date', date_elem(4, 5)),
                           ('notchwidth_map_date', date_elem(6, 7)))),
                     35: ('Composite Reflectivity', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     36: ('Composite Reflectivity', 460., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     37: ('Composite Reflectivity', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     38: ('Composite Reflectivity', 460., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     41: ('Echo Tops', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', scaled_elem(3, 1000)))),  # Max in ft
                     48: ('VAD Wind Profile', None, LegacyMapper,
                          (('max', 3),
                           ('dir_max', 4),
                           ('alt_max', scaled_elem(5, 10)))),  # Max in ft
                     55: ('Storm Relative Mean Radial Velocity', 50., LegacyMapper,
                          (('window_az', scaled_elem(0, 0.1)),
                           ('window_range', scaled_elem(1, 0.1)),
                           ('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3),
                           ('max', 4),
                           ('source', 5),
                           ('height', 6),
                           ('avg_speed', scaled_elem(7, 0.1)),
                           ('avg_dir', scaled_elem(8, 0.1)),
                           ('alert_category', 9))),
                     56: ('Storm Relative Mean Radial Velocity', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3),
                           ('max', 4),
                           ('source', 5),
                           ('avg_speed', scaled_elem(7, 0.1)),
                           ('avg_dir', scaled_elem(8, 0.1)))),
                     57: ('Vertically Integrated Liquid', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3))),  # Max in kg / m^2
                     58: ('Storm Tracking Information', 460., LegacyMapper,
                          (('num_storms', 3),)),
                     59: ('Hail Index', 230., LegacyMapper, ()),
                     61: ('Tornado Vortex Signature', 230., LegacyMapper,
                          (('num_tvs', 3), ('num_etvs', 4))),
                     62: ('Storm Structure', 460., LegacyMapper, ()),
                     63: ('Layer Composite Reflectivity (Layer 1 Average)', 230., LegacyMapper,
                          (('max', 3),
                           ('layer_bottom', scaled_elem(4, 1000.)),
                           ('layer_top', scaled_elem(5, 1000.)),
                           ('calib_const', float_elem(7, 8)))),
                     64: ('Layer Composite Reflectivity (Layer 2 Average)', 230., LegacyMapper,
                          (('max', 3),
                           ('layer_bottom', scaled_elem(4, 1000.)),
                           ('layer_top', scaled_elem(5, 1000.)),
                           ('calib_const', float_elem(7, 8)))),
                     65: ('Layer Composite Reflectivity (Layer 1 Max)', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('layer_bottom', scaled_elem(4, 1000.)),
                           ('layer_top', scaled_elem(5, 1000.)),
                           ('calib_const', float_elem(7, 8)))),
                     66: ('Layer Composite Reflectivity (Layer 2 Max)', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('layer_bottom', scaled_elem(4, 1000.)),
                           ('layer_top', scaled_elem(5, 1000.)),
                           ('calib_const', float_elem(7, 8)))),
                     67: ('Layer Composite Reflectivity - AP Removed', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('layer_bottom', scaled_elem(4, 1000.)),
                           ('layer_top', scaled_elem(5, 1000.)),
                           ('calib_const', float_elem(7, 8)))),
                     74: ('Radar Coded Message', 460., LegacyMapper, ()),
                     78: ('Surface Rainfall Accumulation (1 hour)', 230., LegacyMapper,
                          (('max_rainfall', scaled_elem(3, 0.1)),
                           ('bias', scaled_elem(4, 0.01)),
                           ('gr_pairs', scaled_elem(5, 0.01)),
                           ('rainfall_end', date_elem(6, 7)))),
                     79: ('Surface Rainfall Accumulation (3 hour)', 230., LegacyMapper,
                          (('max_rainfall', scaled_elem(3, 0.1)),
                           ('bias', scaled_elem(4, 0.01)),
                           ('gr_pairs', scaled_elem(5, 0.01)),
                           ('rainfall_end', date_elem(6, 7)))),
                     80: ('Storm Total Rainfall Accumulation', 230., LegacyMapper,
                          (('max_rainfall', scaled_elem(3, 0.1)),
                           ('rainfall_begin', date_elem(4, 5)),
                           ('rainfall_end', date_elem(6, 7)),
                           ('bias', scaled_elem(8, 0.01)),
                           ('gr_pairs', scaled_elem(9, 0.01)))),
                     81: ('Hourly Digital Precipitation Array', 230., PrecipArrayMapper,
                          (('max_rainfall', scaled_elem(3, 0.001)),
                           ('bias', scaled_elem(4, 0.01)),
                           ('gr_pairs', scaled_elem(5, 0.01)),
                           ('rainfall_end', date_elem(6, 7)))),
                     82: ('Supplemental Precipitation Data', None, LegacyMapper, ()),
                     89: ('Layer Composite Reflectivity (Layer 3 Average)', 230., LegacyMapper,
                          (('max', 3),
                           ('layer_bottom', scaled_elem(4, 1000.)),
                           ('layer_top', scaled_elem(5, 1000.)),
                           ('calib_const', float_elem(7, 8)))),
                     90: ('Layer Composite Reflectivity (Layer 3 Max)', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('layer_bottom', scaled_elem(4, 1000.)),
                           ('layer_top', scaled_elem(5, 1000.)),
                           ('calib_const', float_elem(7, 8)))),
                     93: ('ITWS Digital Base Velocity', 115., DigitalVelMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3),
                           ('max', 4), ('precision', 6))),
                     94: ('Base Reflectivity Data Array', 460., DigitalRefMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('compression', 7),
                           ('uncompressed_size', combine_elem(8, 9)))),
                     95: ('Composite Reflectivity Edited for AP', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     96: ('Composite Reflectivity Edited for AP', 460., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     97: ('Composite Reflectivity Edited for AP', 230., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     98: ('Composite Reflectivity Edited for AP', 460., LegacyMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('max', 3),
                           ('calib_const', float_elem(7, 8)))),
                     99: ('Base Velocity Data Array', 300., DigitalVelMapper,
                          (('el_angle', scaled_elem(2, 0.1)),
                           ('min', 3),
                           ('max', 4),
                           ('compression', 7),
                           ('uncompressed_size', combine_elem(8, 9)))),
                     132: ('Clutter Likelihood Reflectivity', 230., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),)),
                     133: ('Clutter Likelihood Doppler', 230., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),)),
                     134: ('High Resolution VIL', 460., DigitalVILMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', 3),
                            ('num_edited', 4),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     135: ('Enhanced Echo Tops', 345., DigitalEETMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', scaled_elem(3, 1000.)),  # Max in ft
                            ('num_edited', 4),
                            ('ref_thresh', 5),
                            ('points_removed', 6),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     138: ('Digital Storm Total Precipitation', 230., DigitalStormPrecipMapper,
                           (('rainfall_begin', date_elem(0, 1)),
                            ('bias', scaled_elem(2, 0.01)),
                            ('max', scaled_elem(3, 0.01)),
                            ('rainfall_end', date_elem(4, 5)),
                            ('gr_pairs', scaled_elem(6, 0.01)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     141: ('Mesocyclone Detection', 230., LegacyMapper,
                           (('min_ref_thresh', 0),
                            ('overlap_display_filter', 1),
                            ('min_strength_rank', 2))),
                     152: ('Archive III Status Product', None, LegacyMapper,
                           (('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     153: ('Super Resolution Reflectivity Data Array', 460., DigitalRefMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', 3),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     154: ('Super Resolution Velocity Data Array', 300., DigitalVelMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', 3),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     155: ('Super Resolution Spectrum Width Data Array', 300.,
                           DigitalSPWMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', 3),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     156: ('Turbulence Detection (Eddy Dissipation Rate)', 230., EDRMapper,
                           (('el_start_time', 0),
                            ('el_end_time', 1),
                            ('el_angle', scaled_elem(2, 0.1)),
                            ('min_el', scaled_elem(3, 0.01)),
                            ('mean_el', scaled_elem(4, 0.01)),
                            ('max_el', scaled_elem(5, 0.01)))),
                     157: ('Turbulence Detection (Eddy Dissipation Rate Confidence)', 230.,
                           EDRMapper,
                           (('el_start_time', 0),
                            ('el_end_time', 1),
                            ('el_angle', scaled_elem(2, 0.1)),
                            ('min_el', scaled_elem(3, 0.01)),
                            ('mean_el', scaled_elem(4, 0.01)),
                            ('max_el', scaled_elem(5, 0.01)))),
                     158: ('Differential Reflectivity', 230., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('min', scaled_elem(3, 0.1)),
                            ('max', scaled_elem(4, 0.1)))),
                     159: ('Digital Differential Reflectivity', 300., GenericDigitalMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('min', scaled_elem(3, 0.1)),
                            ('max', scaled_elem(4, 0.1)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     160: ('Correlation Coefficient', 230., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('min', scaled_elem(3, 0.00333)),
                            ('max', scaled_elem(4, 0.00333)))),
                     161: ('Digital Correlation Coefficient', 300., GenericDigitalMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('min', scaled_elem(3, 0.00333)),
                            ('max', scaled_elem(4, 0.00333)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     162: ('Specific Differential Phase', 230., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('min', scaled_elem(3, 0.05)),
                            ('max', scaled_elem(4, 0.05)))),
                     163: ('Digital Specific Differential Phase', 300., GenericDigitalMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('min', scaled_elem(3, 0.05)),
                            ('max', scaled_elem(4, 0.05)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     164: ('Hydrometeor Classification', 230., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),)),
                     165: ('Digital Hydrometeor Classification', 300., DigitalHMCMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     166: ('Melting Layer', 230., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),)),
                     169: ('One Hour Accumulation', 230., LegacyMapper,
                           (('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 5)),
                            ('bias', scaled_elem(6, 0.01)),
                            ('gr_pairs', scaled_elem(7, 0.01)))),
                     170: ('Digital Accumulation Array', 230., GenericDigitalMapper,
                           (('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 5)),
                            ('bias', scaled_elem(6, 0.01)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     171: ('Storm Total Accumulation', 230., LegacyMapper,
                           (('rainfall_begin', date_elem(0, 1)),
                            ('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 5)),
                            ('bias', scaled_elem(6, 0.01)),
                            ('gr_pairs', scaled_elem(7, 0.01)))),
                     172: ('Digital Storm total Accumulation', 230., GenericDigitalMapper,
                           (('rainfall_begin', date_elem(0, 1)),
                            ('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 5)),
                            ('bias', scaled_elem(6, 0.01)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     173: ('Digital User-Selectable Accumulation', 230., GenericDigitalMapper,
                           (('period', 1),
                            ('missing_period', high_byte(2)),
                            ('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 0)),
                            ('start_time', 5),
                            ('bias', scaled_elem(6, 0.01)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     174: ('Digital One-Hour Difference Accumulation', 230.,
                           GenericDigitalMapper,
                           (('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 5)),
                            ('min', scaled_elem(6, 0.1)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     175: ('Digital Storm Total Difference Accumulation', 230.,
                           GenericDigitalMapper,
                           (('rainfall_begin', date_elem(0, 1)),
                            ('null_product', low_byte(2)),
                            ('max', scaled_elem(3, 0.1)),
                            ('rainfall_end', date_elem(4, 5)),
                            ('min', scaled_elem(6, 0.1)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     176: ('Digital Instantaneous Precipitation Rate', 230.,
                           GenericDigitalMapper,
                           (('rainfall_begin', date_elem(0, 1)),
                            ('precip_detected', high_byte(2)),
                            ('need_bias', low_byte(2)),
                            ('max', 3),
                            ('percent_filled', scaled_elem(4, 0.01)),
                            ('max_elev', scaled_elem(5, 0.1)),
                            ('bias', scaled_elem(6, 0.01)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     177: ('Hybrid Hydrometeor Classification', 230., DigitalHMCMapper,
                           (('mode_filter_size', 3),
                            ('hybrid_percent_filled', 4),
                            ('max_elev', scaled_elem(5, 0.1)),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     180: ('TDWR Base Reflectivity', 90., DigitalRefMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', 3),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     181: ('TDWR Base Reflectivity', 90., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', 3))),
                     182: ('TDWR Base Velocity', 90., DigitalVelMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('min', 3),
                            ('max', 4),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     183: ('TDWR Base Velocity', 90., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('min', 3),
                            ('max', 4))),
                     185: ('TDWR Base Spectrum Width', 90., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', 3))),
                     186: ('TDWR Long Range Base Reflectivity', 416., DigitalRefMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', 3),
                            ('compression', 7),
                            ('uncompressed_size', combine_elem(8, 9)))),
                     187: ('TDWR Long Range Base Reflectivity', 416., LegacyMapper,
                           (('el_angle', scaled_elem(2, 0.1)),
                            ('max', 3)))}

    def __init__(self, filename):
        r'''Create instance of `Level3File`.

        Parameters
        ----------
        filename : str or file-like object
            If str, the name of the file to be opened. If file-like object,
            this will be read from directly.
        '''

        if is_string_like(filename):
            fobj = open(filename, 'rb')
            self.filename = filename
        else:
            fobj = filename
            self.filename = 'No Filename'

        # Just read in the entire set of data at once
        self._buffer = IOBuffer.fromfile(fobj)

        # Pop off the WMO header if we find it
        self._process_wmo_header()

        # Pop off last 4 bytes if necessary
        self._process_end_bytes()

        # Set up places to store data and metadata
#        self.data = []
        self.metadata = dict()

        # Handle free text message products that are pure text
        if self.wmo_code == 'NOUS':
            self.header = None
            self.prod_desc = None
            self.thresholds = None
            self.depVals = None
            self.product_name = 'Free Text Message'
            self.text = ''.join(self._buffer.read_ascii())
            return

        # Decompress the data if necessary, and if so, pop off new header
        self._buffer = IOBuffer(self._buffer.read_func(zlib_decompress_all_frames))
        self._process_wmo_header()

        # Check for empty product
        if len(self._buffer) == 0:
            log.warning('%s: Empty product!', self.filename)
            return

        # Unpack the message header and the product description block
        msg_start = self._buffer.set_mark()
        self.header = self._buffer.read_struct(self.header_fmt)
        # print(self.header, len(self._buffer), self.header.msg_len - self.header_fmt.size)
        assert self._buffer.check_remains(self.header.msg_len - self.header_fmt.size)

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
        self.metadata['msg_time'] = nexrad_to_datetime(self.header.date,
                                                       self.header.time * 1000)
        self.metadata['vol_time'] = nexrad_to_datetime(self.prod_desc.vol_date,
                                                       self.prod_desc.vol_start_time * 1000)
        self.metadata['prod_time'] = nexrad_to_datetime(self.prod_desc.prod_gen_date,
                                                        self.prod_desc.prod_gen_time * 1000)
        self.lat = self.prod_desc.lat * 0.001
        self.lon = self.prod_desc.lon * 0.001
        self.height = self.prod_desc.height

        # Handle product-specific blocks. Default to compression and elevation angle
        # Also get other product specific information, like name,
        # maximum range, and how to map data bytes to values
        default = ('Unknown Product', 230., LegacyMapper,
                   (('el_angle', scaled_elem(2, 0.1)), ('compression', 7),
                    ('uncompressed_size', combine_elem(8, 9)), ('defaultVals', 0)))
        self.product_name, self.max_range, mapper, meta = self.prod_spec_map.get(
            self.header.code, default)
        for name, block in meta:
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

        # Unpack the various blocks, if present. The factor of 2 converts from
        # 'half-words' to bytes
        # Check to see if this is one of the "special" products that uses
        # header-free blocks and re-assigns the offsets
        if self.header.code in self.standalone_tabular:
            if self.prod_desc.sym_off:
                # For standalone tabular alphanumeric, symbology offset is
                # actually tabular
                self._unpack_tabblock(msg_start, 2 * self.prod_desc.sym_off, False)
            if self.prod_desc.graph_off:
                # Offset seems to be off by 1 from where we're counting, but
                # it's not clear why.
                self._unpack_standalone_graphblock(msg_start,
                                                   2 * (self.prod_desc.graph_off - 1))
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
            log.warning('%s: Using default metadata for product %d',
                        self.filename, self.header.code)

    def _process_wmo_header(self):
        # Read off the WMO header if necessary
        data = self._buffer.get_next(64).decode('utf-8', 'ignore')
        match = self.wmo_finder.search(data)
        if match:
            self.wmo_code = match.groups()[0]
            self.siteID = match.groups()[-1]
            self._buffer.skip(match.end())
        else:
            self.wmo_code = ''

    def _process_end_bytes(self):
        check_bytes = self._buffer[-4:-1]
        if check_bytes == b'\r\r\n' or check_bytes == b'\xff\xff\n':
            self._buffer.truncate(4)

    @staticmethod
    def _unpack_rle_data(data):
        # Unpack Run-length encoded data
        unpacked = []
        for run in data:
            num, val = run >> 4, run & 0x0F
            unpacked.extend([val] * num)
        return unpacked

    @staticmethod
    def pos_scale(is_sym_block):
        return 0.25 if is_sym_block else 1

    def _unpack_rcm(self, start, offset):
        self._buffer.jump_to(start, offset)
        header = self._buffer.read_ascii(10)
        assert header == '1234 ROBUU'
        text_data = self._buffer.read_ascii()
        end = 0
        # Appendix B of ICD tells how to interpret this stuff, but that just
        # doesn't seem worth it.
        for marker, name in [('AA', 'ref'), ('BB', 'vad'), ('CC', 'remarks')]:
            start = text_data.find('/NEXR' + marker, end)
            # For part C the search for end fails, but returns -1, which works
            end = text_data.find('/END' + marker, start)
            setattr(self, 'rcm_' + name, text_data[start:end])

    def _unpack_symblock(self, start, offset):
        self._buffer.jump_to(start, offset)
        blk = self._buffer.read_struct(self.sym_block_fmt)

        self.sym_block = []
        assert blk.divider == -1, ('Bad divider for symbology block: %d should be -1' %
                                   blk.divider)
        assert blk.block_id == 1, ('Bad block ID for symbology block: %d should be 1' %
                                   blk.block_id)
        for _ in range(blk.nlayer):
            layer_hdr = self._buffer.read_struct(self.sym_layer_fmt)
            assert layer_hdr.divider == -1
            layer = []
            self.sym_block.append(layer)
            layer_start = self._buffer.set_mark()
            while self._buffer.offset_from(layer_start) < layer_hdr.length:
                packet_code = self._buffer.read_int('>H')
                if packet_code in self.packet_map:
                    layer.append(self.packet_map[packet_code](self, packet_code, True))
                else:
                    log.warning('%s: Unknown symbology packet type %d/%x.',
                                self.filename, packet_code, packet_code)
                    self._buffer.jump_to(layer_start, layer_hdr.length)
            assert self._buffer.offset_from(layer_start) == layer_hdr.length

    def _unpack_graphblock(self, start, offset):
        self._buffer.jump_to(start, offset)
        hdr = self._buffer.read_struct(self.graph_block_fmt)
        assert hdr.divider == -1, ('Bad divider for graphical block: %d should be -1' %
                                   hdr.divider)
        assert hdr.block_id == 2, ('Bad block ID for graphical block: %d should be 1' %
                                   hdr.block_id)
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
                    log.warning('%s: Unknown graphical packet type %d/%x.',
                                self.filename, packet_code, packet_code)
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
                log.warning('%s: Unknown standalone graphical packet type %d/%x.',
                            self.filename, packet_code, packet_code)
                # Assume next 2 bytes is packet length and try skipping
                num_bytes = self._buffer.read_int('>H')
                self._buffer.skip(num_bytes)
        self.graph_pages = [packets]

    def _unpack_tabblock(self, start, offset, have_header=True):
        self._buffer.jump_to(start, offset)
        block_start = self._buffer.set_mark()

        # Read the header and validate if needed
        if have_header:
            header = self._buffer.read_struct(self.tab_header_fmt)
            assert header.divider == -1
            assert header.block_id == 3

            # Read off secondary message and product description blocks,
            # but as far as I can tell, all we really need is the text that follows
            self._buffer.read_struct(self.header_fmt)
            self._buffer.read_struct(self.prod_desc_fmt)

        # Get the start of the block with number of pages and divider
        blk = self._buffer.read_struct(self.tab_block_fmt)
        assert blk.divider == -1

        # Read the pages line by line, break pages on a -1 character count
        self.tab_pages = []
        for _ in range(blk.num_pages):
            lines = []
            num_chars = self._buffer.read_int('>h')
            while num_chars != -1:
                lines.append(''.join(self._buffer.read_ascii(num_chars)))
                num_chars = self._buffer.read_int('>h')
            self.tab_pages.append('\n'.join(lines))

        if have_header:
            assert self._buffer.offset_from(block_start) == header.block_len

    def __repr__(self):
        items = [self.product_name, self.header, self.prod_desc, self.thresholds,
                 self.depVals, self.metadata, self.siteID]
        return self.filename + ': ' + '\n'.join(map(str, items))

    def _unpack_packet_radial_data(self, code, in_sym_block):
        hdr_fmt = NamedStruct([('ind_first_bin', 'H'), ('nbins', 'H'),
                               ('i_center', 'h'), ('j_center', 'h'),
                               ('scale_factor', 'h'), ('num_rad', 'H')],
                              '>', 'RadialHeader')
        rad_fmt = NamedStruct([('num_hwords', 'H'), ('start_angle', 'h'),
                               ('angle_delta', 'h')], '>', 'RadialData')
        hdr = self._buffer.read_struct(hdr_fmt)
        rads = []
        for _ in range(hdr.num_rad):
            rad = self._buffer.read_struct(rad_fmt)
            start_az = rad.start_angle * 0.1
            end_az = start_az + rad.angle_delta * 0.1
            rads.append((start_az, end_az,
                         self._unpack_rle_data(
                             self._buffer.read_binary(2 * rad.num_hwords))))
        start, end, vals = zip(*rads)
        return dict(start_az=list(start), end_az=list(end), data=list(vals),
                    center=(hdr.i_center * self.pos_scale(in_sym_block),
                            hdr.j_center * self.pos_scale(in_sym_block)),
                    gate_scale=hdr.scale_factor * 0.001, first=hdr.ind_first_bin)

    def _unpack_packet_digital_radial(self, code, in_sym_block):
        hdr_fmt = NamedStruct([('ind_first_bin', 'H'), ('nbins', 'H'),
                               ('i_center', 'h'), ('j_center', 'h'),
                               ('scale_factor', 'h'), ('num_rad', 'H')],
                              '>', 'DigitalRadialHeader')
        rad_fmt = NamedStruct([('num_bytes', 'H'), ('start_angle', 'h'),
                               ('angle_delta', 'h')], '>', 'DigitalRadialData')
        hdr = self._buffer.read_struct(hdr_fmt)
        rads = []
        for i in range(hdr.num_rad):
            rad = self._buffer.read_struct(rad_fmt)
            start_az = rad.start_angle * 0.1
            end_az = start_az + rad.angle_delta * 0.1
            rads.append((start_az, end_az, self._buffer.read_binary(rad.num_bytes)))
        start, end, vals = zip(*rads)
        return dict(start_az=list(start), end_az=list(end), data=list(vals),
                    center=(hdr.i_center * self.pos_scale(in_sym_block),
                            hdr.j_center * self.pos_scale(in_sym_block)),
                    gate_scale=hdr.scale_factor * 0.001, first=hdr.ind_first_bin)

    def _unpack_packet_raster_data(self, code, in_sym_block):
        hdr_fmt = NamedStruct([('code', 'L'),
                               ('i_start', 'h'), ('j_start', 'h'),  # start in km/4
                               ('xscale_int', 'h'), ('xscale_frac', 'h'),
                               ('yscale_int', 'h'), ('yscale_frac', 'h'),
                               ('num_rows', 'h'), ('packing', 'h')], '>', 'RasterData')
        hdr = self._buffer.read_struct(hdr_fmt)
        assert hdr.code == 0x800000C0
        assert hdr.packing == 2
        rows = []
        for _ in range(hdr.num_rows):
            num_bytes = self._buffer.read_int('>H')
            rows.append(self._unpack_rle_data(self._buffer.read_binary(num_bytes)))
        return dict(start_x=hdr.i_start * hdr.xscale_int,
                    start_y=hdr.j_start * hdr.yscale_int, data=rows)

    def _unpack_packet_uniform_text(self, code, in_sym_block):
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
        return dict(x=i_start * self.pos_scale(in_sym_block),
                    y=j_start * self.pos_scale(in_sym_block), color=value, text=text)

    def _unpack_packet_special_text_symbol(self, code, in_sym_block):
        d = self._unpack_packet_uniform_text(code, in_sym_block)

        # Translate special characters to their meaning
        ret = dict()
        symbol_map = {'!': 'past storm position', '"': 'current storm position',
                      '#': 'forecast storm position', '$': 'past MDA position',
                      '%': 'forecast MDA position', ' ': None}

        # Use this meaning as the key in the returned packet
        for c in d['text']:
            if c not in symbol_map:
                log.warning('%s: Unknown special symbol %d/%x.', self.filename, c, ord(c))
            else:
                key = symbol_map[c]
                if key:
                    ret[key] = d['x'], d['y']
        del d['text']

        return ret

    def _unpack_packet_special_graphic_symbol(self, code, in_sym_block):
        type_map = {3: 'Mesocyclone', 11: '3D Correlated Shear', 12: 'TVS',
                    26: 'ETVS', 13: 'Positive Hail', 14: 'Probable Hail',
                    15: 'Storm ID', 19: 'HDA', 25: 'STI Circle'}
        point_feature_map = {1: 'Mesocyclone (ext.)', 3: 'Mesocyclone',
                             5: 'TVS (Ext.)', 6: 'ETVS (Ext.)', 7: 'TVS',
                             8: 'ETVS', 9: 'MDA', 10: 'MDA (Elev.)', 11: 'MDA (Weak)'}

        # Read the number of bytes and set a mark for sanity checking
        num_bytes = self._buffer.read_int('>H')
        packet_data_start = self._buffer.set_mark()

        scale = self.pos_scale(in_sym_block)

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
                    log.warning('%s: Unknown graphic symbol point kind %d/%x.',
                                self.filename, kind, kind)
                    ret['type'].append('Unknown (%d)' % kind)
                else:
                    ret['type'].append(point_feature_map[kind])

        # Map the code to a name for this type of symbol
        if code != 20:
            if code not in type_map:
                log.warning('%s: Unknown graphic symbol type %d/%x.',
                            self.filename, code, code)
                ret['type'] = 'Unknown'
            else:
                ret['type'] = type_map[code]

        # Check and return
        assert self._buffer.offset_from(packet_data_start) == num_bytes

        # Reduce dimensions of lists if possible
        reduce_lists(ret)

        return ret

    def _unpack_packet_scit(self, code, in_sym_block):
        num_bytes = self._buffer.read_int('>H')
        packet_data_start = self._buffer.set_mark()
        ret = defaultdict(list)
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            next_code = self._buffer.read_int('>H')
            if next_code not in self.packet_map:
                log.warning('%s: Unknown packet in SCIT %d/%x.',
                            self.filename, next_code, next_code)
                self._buffer.jump_to(packet_data_start, num_bytes)
                return ret
            else:
                next_packet = self.packet_map[next_code](self, next_code, in_sym_block)
                if next_code == 6:
                    ret['track'].append(next_packet['vectors'])
                elif next_code == 25:
                    ret['STI Circle'].append(next_packet)
                elif next_code == 2:
                    ret['markers'].append(next_packet)
                else:
                    log.warning('%s: Unsupported packet in SCIT %d/%x.',
                                self.filename, next_code, next_code)
                    ret['data'].append(next_packet)
        reduce_lists(ret)
        return ret

    def _unpack_packet_digital_precipitation(self, code, in_sym_block):
        # Read off a couple of unused spares
        self._buffer.read_int('>H')
        self._buffer.read_int('>H')

        # Get the size of the grid
        lfm_boxes = self._buffer.read_int('>H')
        num_rows = self._buffer.read_int('>H')
        rows = []

        # Read off each row and decode the RLE data
        for _ in range(num_rows):
            row_num_bytes = self._buffer.read_int('>H')
            row_bytes = self._buffer.read_binary(row_num_bytes)
            if code == 18:
                row = self._unpack_rle_data(row_bytes)
            else:
                row = []
                for run, level in zip(row_bytes[::2], row_bytes[1::2]):
                    row.extend([level] * run)
            assert len(row) == lfm_boxes
            rows.append(row)

        return dict(data=rows)

    def _unpack_packet_linked_vector(self, code, in_sym_block):
        num_bytes = self._buffer.read_int('>h')
        if code == 9:
            value = self._buffer.read_int('>h')
            num_bytes -= 2
        else:
            value = None
        scale = self.pos_scale(in_sym_block)
        pos = [b * scale for b in self._buffer.read_binary(num_bytes / 2, '>h')]
        vectors = list(zip(pos[::2], pos[1::2]))
        return dict(vectors=vectors, color=value)

    def _unpack_packet_vector(self, code, in_sym_block):
        num_bytes = self._buffer.read_int('>h')
        if code == 10:
            value = self._buffer.read_int('>h')
            num_bytes -= 2
        else:
            value = None
        scale = self.pos_scale(in_sym_block)
        pos = [p * scale for p in self._buffer.read_binary(num_bytes / 2, '>h')]
        vectors = list(zip(pos[::4], pos[1::4], pos[2::4], pos[3::4]))
        return dict(vectors=vectors, color=value)

    def _unpack_packet_contour_color(self, code, in_sym_block):
        # Check for color value indicator
        assert self._buffer.read_int('>H') == 0x0002

        # Read and return value (level) of contour
        return dict(color=self._buffer.read_int('>H'))

    def _unpack_packet_linked_contour(self, code, in_sym_block):
        # Check for initial point indicator
        assert self._buffer.read_int('>H') == 0x8000

        scale = self.pos_scale(in_sym_block)
        startx = self._buffer.read_int('>h') * scale
        starty = self._buffer.read_int('>h') * scale
        vectors = [(startx, starty)]
        num_bytes = self._buffer.read_int('>H')
        pos = [b * scale for b in self._buffer.read_binary(num_bytes / 2, '>h')]
        vectors.extend(zip(pos[::2], pos[1::2]))
        return dict(vectors=vectors)

    def _unpack_packet_wind_barbs(self, code, in_sym_block):
        # Figure out how much to read
        num_bytes = self._buffer.read_int('>h')
        packet_data_start = self._buffer.set_mark()
        ret = defaultdict(list)

        # Read while we have data, then return
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            ret['color'].append(self._buffer.read_int('>h'))
            ret['x'].append(self._buffer.read_int('>h') * self.pos_scale(in_sym_block))
            ret['y'].append(self._buffer.read_int('>h') * self.pos_scale(in_sym_block))
            ret['direc'].append(self._buffer.read_int('>h'))
            ret['speed'].append(self._buffer.read_int('>h'))
        return ret

    def _unpack_packet_generic(self, code, in_sym_block):
        # Reserved HW
        assert self._buffer.read_int('>h') == 0

        # Read number of bytes (2 HW) and return
        num_bytes = self._buffer.read_int('>l')
        hunk = self._buffer.read(num_bytes)
        xdrparser = Level3XDRParser(hunk)
        return xdrparser(code)

    def _unpack_packet_trend_times(self, code, in_sym_block):
        self._buffer.read_int('>h')  # number of bytes, not needed to process
        return dict(times=self._read_trends())

    def _unpack_packet_cell_trend(self, code, in_sym_block):
        code_map = ['Cell Top', 'Cell Base', 'Max Reflectivity Height',
                    'Probability of Hail', 'Probability of Severe Hail',
                    'Cell-based VIL', 'Maximum Reflectivity',
                    'Centroid Height']
        code_scales = [100, 100, 100, 1, 1, 1, 1, 100]
        num_bytes = self._buffer.read_int('>h')
        packet_data_start = self._buffer.set_mark()
        cell_id = ''.join(self._buffer.read_ascii(2))
        x = self._buffer.read_int('>h') * self.pos_scale(in_sym_block)
        y = self._buffer.read_int('>h') * self.pos_scale(in_sym_block)
        ret = dict(id=cell_id, x=x, y=y)
        while self._buffer.offset_from(packet_data_start) < num_bytes:
            code = self._buffer.read_int('>h')
            try:
                ind = code - 1
                key = code_map[ind]
                scale = code_scales[ind]
            except IndexError:
                log.warning('%s: Unsupported trend code %d/%x.', self.filename, code, code)
                key = 'Unknown'
                scale = 1
            vals = self._read_trends()
            if code in (1, 2):
                ret['%s Limited' % key] = [True if v > 700 else False for v in vals]
                vals = [v - 1000 if v > 700 else v for v in vals]
            ret[key] = [v * scale for v in vals]

        return ret

    def _read_trends(self):
        num_vols = self._buffer.read_int('b')
        latest = self._buffer.read_int('b')
        vals = [self._buffer.read_int('>h') for _ in range(num_vols)]

        # Wrap the circular buffer so that latest is last
        vals = vals[latest:] + vals[:latest]
        return vals

    packet_map = {1: _unpack_packet_uniform_text,
                  2: _unpack_packet_special_text_symbol,
                  3: _unpack_packet_special_graphic_symbol,
                  4: _unpack_packet_wind_barbs,
                  6: _unpack_packet_linked_vector,
                  8: _unpack_packet_uniform_text,
                  # 9: _unpack_packet_linked_vector,
                  10: _unpack_packet_vector,
                  11: _unpack_packet_special_graphic_symbol,
                  12: _unpack_packet_special_graphic_symbol,
                  13: _unpack_packet_special_graphic_symbol,
                  14: _unpack_packet_special_graphic_symbol,
                  15: _unpack_packet_special_graphic_symbol,
                  16: _unpack_packet_digital_radial,
                  17: _unpack_packet_digital_precipitation,
                  18: _unpack_packet_digital_precipitation,
                  19: _unpack_packet_special_graphic_symbol,
                  20: _unpack_packet_special_graphic_symbol,
                  21: _unpack_packet_cell_trend,
                  22: _unpack_packet_trend_times,
                  23: _unpack_packet_scit,
                  24: _unpack_packet_scit,
                  25: _unpack_packet_special_graphic_symbol,
                  26: _unpack_packet_special_graphic_symbol,
                  28: _unpack_packet_generic,
                  29: _unpack_packet_generic,
                  0x0802: _unpack_packet_contour_color,
                  0x0E03: _unpack_packet_linked_contour,
                  0xaf1f: _unpack_packet_radial_data,
                  0xba07: _unpack_packet_raster_data}


class Level3XDRParser(Unpacker):
    def __call__(self, code):
        xdr = OrderedDict()

        if code == 28:
            xdr.update(self._unpack_prod_desc())
        else:
            log.warning('XDR: code %d not implemented', code)

        # Check that we got it all
        self.done()
        return xdr

    def unpack_string(self):
        return Unpacker.unpack_string(self).decode('ascii')

    def _unpack_prod_desc(self):
        xdr = OrderedDict()

        # NOTE: The ICD (262001U) incorrectly lists op-mode, vcp, el_num, and
        # spare as int*2. Changing to int*4 makes things parse correctly.
        xdr['name'] = self.unpack_string()
        xdr['description'] = self.unpack_string()
        xdr['code'] = self.unpack_int()
        xdr['type'] = self.unpack_int()
        xdr['prod_time'] = self.unpack_uint()
        xdr['radar_name'] = self.unpack_string()
        xdr['latitude'] = self.unpack_float()
        xdr['longitude'] = self.unpack_float()
        xdr['height'] = self.unpack_float()
        xdr['vol_time'] = self.unpack_uint()
        xdr['el_time'] = self.unpack_uint()
        xdr['el_angle'] = self.unpack_float()
        xdr['vol_num'] = self.unpack_int()
        xdr['op_mode'] = self.unpack_int()
        xdr['vcp_num'] = self.unpack_int()
        xdr['el_num'] = self.unpack_int()
        xdr['compression'] = self.unpack_int()
        xdr['uncompressed_size'] = self.unpack_int()
        xdr['parameters'] = self._unpack_parameters()
        xdr['components'] = self._unpack_components()

        return xdr

    def _unpack_parameters(self):
        num = self.unpack_int()

        # ICD documents a "pointer" here, that seems to be garbage. Just read
        # and use the number, starting the list immediately.
        self.unpack_int()

        if num == 0:
            return None

        ret = list()
        for i in range(num):
            ret.append((self.unpack_string(), self.unpack_string()))
            if i < num - 1:
                self.unpack_int()  # Another pointer for the 'list' ?

        if num == 1:
            ret = ret[0]

        return ret

    def _unpack_components(self):
        num = self.unpack_int()

        # ICD documents a "pointer" here, that seems to be garbage. Just read
        # and use the number, starting the list immediately.
        self.unpack_int()

        ret = list()
        for i in range(num):
            try:
                code = self.unpack_int()
                ret.append(self._component_lookup[code](self))
                if i < num - 1:
                    self.unpack_int()  # Another pointer for the 'list' ?
            except KeyError:
                log.warning('Unknown XDR Component: %d', code)
                break

        if num == 1:
            ret = ret[0]

        return ret

    radial_fmt = namedtuple('RadialComponent', ['description', 'gate_width',
                                                'first_gate', 'parameters',
                                                'radials'])
    radial_data_fmt = namedtuple('RadialData', ['azimuth', 'elevation', 'width',
                                                'num_bins', 'attributes',
                                                'data'])

    def _unpack_radial(self):
        ret = self.radial_fmt(description=self.unpack_string(),
                              gate_width=self.unpack_float(),
                              first_gate=self.unpack_float(),
                              parameters=self._unpack_parameters(),
                              radials=None)
        num_rads = self.unpack_int()
        rads = list()
        for _ in range(num_rads):
            # ICD is wrong, says num_bins is float, should be int
            rads.append(self.radial_data_fmt(azimuth=self.unpack_float(),
                                             elevation=self.unpack_float(),
                                             width=self.unpack_float(),
                                             num_bins=self.unpack_int(),
                                             attributes=self.unpack_string(),
                                             data=self.unpack_array(self.unpack_int)))
        return ret._replace(radials=rads)

    text_fmt = namedtuple('TextComponent', ['parameters', 'text'])

    def _unpack_text(self):
        return self.text_fmt(parameters=self._unpack_parameters(),
                             text=self.unpack_string())

    _component_lookup = {1: _unpack_radial, 4: _unpack_text}


@exporter.export
def is_precip_mode(vcp_num):
    r'''Determine if the NEXRAD radar is operating in precipitation mode

    Parameters
    ----------
    vcp_num : int
        The NEXRAD volume coverage pattern (VCP) number

    Returns
    -------
    bool
        True if the VCP corresponds to precipitation mode, False otherwise
    '''
    return not vcp_num // 10 == 3
