"""
This module provides routines to turns a directory tree of WDSSII radar sweepfiles
into a Radar Software Library (trmm-fc.gsfc.nasa.gov/trmm_gv/software/rsl/index.html)
radar structures, one per volume scan. Uses a ctypes wrapper for RSL.

The RSL library may be used to write this radar structure to disk as a Universal Format
or HDF radar volume scan file.

This module also serves as an example of how to use wrangle_wdssii.py 
to organize volume scans and loop through those volumes to read from the
WDSSII NetCDF product files. Provision is made for plain-array sweeps and for
sweeps with WDSSII's ad-hoc run-length-encoding of data arrays. Ideally, the
WDSSII product IO knoweldge would be separated from the RSL specfic bits, but I didn't
feel like creating the necessary intermediary data structure when the NetCDF API was
already pretty obvious.


EXAMPLE
-------
def directory_to_uf(directory, outdir):
    
    vol_scans = read_directory(directory)
    scan_times = vol_scans.keys()
    scan_times.sort()
    
    for time in scan_times:
        print '-----------------------------------------------'
        print 'Processing volume scan from', time
        vol_scan = vol_scans[time]
        radar = RSL_wdssii_to_radar(vol_scan)
        filename = 'PAR_%s.uf' % (time.strftime('%Y%m%d_%H%M%S'),)
        RSL.RSL_radar_to_uf(radar, os.path.join(outdir, filename))

directory_to_uf('/Volumes/Backup/fulgurite/data/2006/PAR_new', '/Users/ebruning/out/py_rsl_par')


KNOWN ISSUES
------------
Pupynere (1.0.10) seems to fail sporadically when opening from a gzip file object. 
It may have something to do with unsupported file.seek() calls. At the moment, it is
recommended to do a gunzip -r root_sweep_directory before running this code.

Only the basic set of polarimetric fields is supported, but can be expanded by revising
FIELD_CODES to match WDSSII product names with RSL field codes.


Copyright 2009, Eric Bruning, eric@deeplycloudy.com
License: BSD-like

"""

import os, sys
import time
import datetime
import gzip
import ctypes

import numpy as np
import pupynere as netcdf

import wrangle_wdssii as wdssii
from metpy.readers import RSL

def progress(current, total, pre_text=''):
    percent = int(100.0*float(current)/total)
    sys.stdout.write("\r" + pre_text + " ... %d%%" % percent)
    sys.stdout.flush()

# wdssii sweepfiles use UNIX time. This module assumes system is UNIX.
system_epoch_delta = time.mktime(datetime.datetime(1970, 1, 1, 0, 0, 0).timetuple()) - time.timezone
assert system_epoch_delta == 0.0
def unix_time_to_datetime(unix):
    # could add or subtract system_epoch_delta to unix to make this universal, but I'm
    # too lazy right now to figure out the right sign.
    t = time.gmtime(unix)
    d = datetime.datetime(*t[0:6])
    return d

def decimal_to_deg_min_sec(decimal):
    degrees = int(decimal)
    minutes = int((decimal - degrees)*60.0)
    seconds = 3600.0 * (decimal-degrees-minutes/60.0)
    return degrees, minutes, seconds

RSL_field_types = RSL.fieldTypes()

FIELD_CODES = {'Reflectivity':'DZ',  'Velocity':'VR',
    'Zdr':'DR', 'RhoHV':'RH',
    'Kdp':'KD', 'PhiDP':'PH', 
    }

def read_directory(directory):
    """ Returns volume scan structure for a directory structure of WDSSII
        product files.
    """
    sweeps    = wdssii.gather_sweeps_by_field(directory)
    volumes   = wdssii.volumeize_sweeps(sweeps)
    vol_scans = wdssii.align_volumes_across_fields(volumes)
    
    return vol_scans
    
def open_ncfile(nc_filename):
    """ Opens nc_filename, unzipping if necessary.
    """
    if nc_filename.find('.gz') >= 0:
        gztemp= gzip.open(nc_filename)
        gztemp.closed=False # pupynere needs this attribute, but gzip objects don't have it.
        ncdf = netcdf.netcdf_file(gztemp, 'r')
    else:
        ncdf = netcdf.netcdf_file(nc_filename,'r')
    return ncdf

def RSL_wdssii_to_radar(vol_scan, radar_name='MPAR'):
    """ Returns an RSL radar structure for a single volume scan structure
        as assembled by the wrangle_wdssii module.
    """
    fields = vol_scan.keys()
    n_fields = len(fields)
    
    field_ids = [getattr(RSL_field_types, FIELD_CODES[field]) for field in fields]
    
    # Grab universal data from the first sweep in this field
    if 'Reflectivity' in fields:
        ref_field = 'Reflectivity'
    else:
        ref_field = fields[0]
    
    
    # Need to read metadata from all sweeps to query the number of gates
    max_gates = 1
    max_rays  = 1
    max_sweeps = 1
    
    # sweep_files = {}
    # for field in fields:
    #     sweep_files[field] = []
    #     sweeps = vol_scan[field]
    #     max_sweeps = max(len(sweeps), max_sweeps)
    #     for date, nominal_el, nc_filename in sweeps:
    #         if nc_filename.find('.gz') >= 0:
    #             ncdf = netcdf.netcdf_file(gzip.open(nc_filename),'r')
    #         else:
    #             ncdf = netcdf.netcdf_file(nc_filename,'r')
    #         max_gates = max(ncdf.dimensions['Gate'], max_gates)
    #         max_rays  = max(ncdf.dimensions['Azimuth'], max_rays)
    #         sweep_files[field].append(ncdf)

    # ----- Allocate the radar structure -----
    radar = RSL.RSL_new_radar(max(field_ids)+1) #, max_swps, max_rays, max_gates)
    
    # ----- Add radar header information -----
    ref_sweep = open_ncfile(vol_scan[ref_field][0][2]) #sweep_files[ref_field][0]
    start_time = unix_time_to_datetime(ref_sweep.Time)
    lat, lon = ref_sweep.Latitude, ref_sweep.Longitude
    latd, latm, lats = decimal_to_deg_min_sec(lat)
    lond, lonm, lons = decimal_to_deg_min_sec(lon)
    height = ref_sweep.Height
    if height < 50:
        print 'WARNING: Treating altitude in sweep file as AGL, and adding to 355 m MSL ground altitude at KOUN site'
        height += 355.0
    ref_sweep.close()
    
    radar.contents.h.year       = start_time.year
    radar.contents.h.month      = start_time.month
    radar.contents.h.day        = start_time.day
    radar.contents.h.hour       = start_time.hour
    radar.contents.h.minute     = start_time.minute
    radar.contents.h.sec        = start_time.second
    radar.contents.h.nvolumes   = n_fields
    radar.contents.h.radar_name = radar_name
    radar.contents.h.latd       = latd
    radar.contents.h.latm       = latm
    radar.contents.h.lats       = int(lats)
    radar.contents.h.lond       = lond
    radar.contents.h.lonm       = lonm
    radar.contents.h.lons       = int(lons)
    radar.contents.h.height     = int(height)
    
    # ----- Add data for each field (volume in RSL lingo) and sweep
    for i_field, field in enumerate(fields):
        progress(0, 1, field)
        field_id = getattr(RSL_field_types, FIELD_CODES[field])
        field_f, field_invf = RSL.conversion_functions[FIELD_CODES[field]]
        n_sweeps = len(vol_scan[field])
        vol = RSL.RSL_new_volume(n_sweeps)
        radar.contents.v[field_id] = vol
        
        vol.contents.h.field_type = FIELD_CODES[field]
        vol.contents.h.nsweeps = n_sweeps
        vol.contents.h.calibr_const = 0
        vol.contents.h.f    = field_f
        vol.contents.h.invf = field_invf
        # vol.h.no_data_flag = sweep_files[field][0].MissingData
        no_data_flag = None
        
        for i_swp, swp_data in  enumerate(vol_scan[field]):
            progress(i_swp, n_sweeps, field)    
            # we assume that every sweep has the same missing data value, which
            # is stored in the no_data_flag volume header attribute
            ncdf = open_ncfile(swp_data[2])
            
            
            if no_data_flag == None:
                no_data_flag = ncdf.MissingData # vol.contents.h.no_data_flag
                # vol.contents.h.no_data_flag = no_data_flag
            else:
                assert no_data_flag == ncdf.MissingData
            
            n_rays = ncdf.dimensions['Azimuth']
            n_gates = ncdf.dimensions['Gate']
            
            sweep = RSL.RSL_new_sweep(n_rays)
            vol.contents.sweep[i_swp] = sweep
            
            sweep.contents.h.field_type  = FIELD_CODES[ncdf.TypeName]
            sweep.contents.h.sweep_num   = i_swp  + 1 # one-indexed
            sweep.contents.h.elev        = ncdf.Elevation
            sweep.contents.h.beam_width  = np.mean(ncdf.variables['BeamWidth'].data)
            sweep.contents.h.nrays       = n_rays
            sweep.contents.h.f           = field_f
            sweep.contents.h.invf        = field_invf
            
            sweep_time = unix_time_to_datetime(ncdf.Time)
            sweep_az = ncdf.variables['Azimuth'].data
            sweep_beam_width = ncdf.variables['BeamWidth'].data
            sweep_gate_width = ncdf.variables['GateWidth'].data
            sweep_nyquist    = ncdf.variables['NyquistVelocity'].data
            
            # WDSSII sometimes run-length-encodes the 2-D radial data array 
            # (e.g., the Reflectivity variable). This is indicated by the 
            # DataType = "SparseRadialSet" vs. DataType = "RadialSet" attribute
            if ncdf.DataType == 'SparseRadialSet':
                rle_sweep = ncdf.variables[field].data.astype(np.float32)
                sweep_data = np.zeros((n_rays, n_gates), dtype=np.float32) + no_data_flag
                pixel_x = ncdf.variables['pixel_x'].data
                pixel_y = ncdf.variables['pixel_y'].data
                pixel_id = ncdf.variables['pixel_count'].data
                for i_px in range(ncdf.dimensions['pixel']):
                    y_low  = pixel_y[i_px]
                    y_high = pixel_y[i_px]+pixel_id[i_px]-1
                    sweep_data[pixel_x[i_px], y_low:y_high] = rle_sweep[i_px]
            else:
                sweep_data = ncdf.variables[field].data.astype(np.float32)
            
            for i_ray in range(sweep.contents.h.nrays):
                ray = RSL.RSL_new_ray(n_gates)
                sweep.contents.ray[i_ray] = ray
                
                ray.contents.h.nbins     = n_gates
                ray.contents.h.year      = sweep_time.year
                ray.contents.h.month     = sweep_time.month
                ray.contents.h.day       = sweep_time.day
                ray.contents.h.hour      = sweep_time.hour
                ray.contents.h.minute    = sweep_time.minute
                ray.contents.h.sec       = sweep_time.second
                ray.contents.h.azimuth   = sweep_az[i_ray]
                ray.contents.h.ray_num   = i_ray + 1 # one-indexed
                ray.contents.h.elev      = ncdf.Elevation
                ray.contents.h.elev_num  = i_swp + 1 # one-indexed
                ray.contents.h.range_bin1= int(ncdf.RangeToFirstGate)
                ray.contents.h.gate_size = int(sweep_gate_width[i_ray])
                ray.contents.h.fix_angle = ncdf.Elevation
                ray.contents.h.lat       = ncdf.Latitude
                ray.contents.h.lon       = ncdf.Longitude
                ray.contents.h.alt       = radar.contents.h.height
                ray.contents.h.beam_width= sweep_beam_width[i_ray]
                ray.contents.h.nyq_vel   = sweep_nyquist[i_ray]
                ray.contents.h.f         = field_f
                ray.contents.h.invf      = field_invf
                
                # Always have enough space for the available data since we allocated for the maximum number of gates
                
                # this is the slowest possible way to write the data, working gate by gate.
                # anything fancier requires a deeper understanding of how ctypes deals with
                # arrays and how to interface them with numpy.
                # print 'adding data to ray with bin count of ', ray.contents.h.nbins
                for i_gate in range(0, ray.contents.h.nbins):
                    # according the the RSL docs, the data values should be of type float (presumably 32 bit)
                    # when passed to invf.
                    float_value = ctypes.c_float(sweep_data[i_ray,i_gate])
                    # print field, i_gate, ray.contents.h.invf, RSL.DZ_INVF, RSL.VR_INVF
                    # print ray.contents.h.invf
                    # raise AttributeError, 'invf sucks'

                    # range_value = RSL.Range()
                    # if field_id == 1:
                        # range_value = RSL.VR_INVF(float_value) #ray.contents.h.invf(float_value)
                    # if field_id == 0:
                        # range_value = RSL.DZ_INVF(float_value)
                    # print 'assigning'
                    range_value = ray.contents.h.invf(float_value)
                    ray.contents.range[i_gate] = range_value
                # # ray.range[0:ray.h.nbins] = sweep_data[i_ray, :]
                # if max_gates != ray.h.nbins:
                #     # total number of gates for this ray is less than the maximum.
                #     for i_gate in range(ray.h.nbins)
                #     ray.range[ray.h.nbins:] = vol.h.no_data_flag
            
            # print 'closing sweep', swp_data[2]
            ncdf.close()
        print '\r'+ field +' ... done'
                    
    return radar
    
        
    
if __name__ == '__main__':
    def directory_to_uf(directory, outdir='/Users/ebruning/out/py_rsl_par'):
        vol_scans = read_directory(directory)
        scan_times = vol_scans.keys()
        scan_times.sort()

        # print vol_scans[scan_times[0]]['Reflectivity'][0] #date, el, filename

        for time in scan_times:
            print '-----------------------------------------------'
            print 'Processing volume scan from', time
            vol_scan = vol_scans[time]
            radar = RSL_wdssii_to_radar(vol_scan)
            # print 'made a radar structure for', time
            filename = 'PAR_%s.uf' % (time.strftime('%Y%m%d_%H%M%S'),)
            RSL.RSL_radar_to_uf(radar, os.path.join(outdir, filename))

    directory_to_uf('/Volumes/Backup/fulgurite/data/2006/PAR_new')