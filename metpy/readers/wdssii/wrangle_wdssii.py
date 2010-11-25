""" 
This module provides routines to organize a directory structure full of
WDSSII sweepfiles. Within a top-level directory, subdirectories are laid out 
as follows:
    Reflectivity/
        0.5/
            20060815-220901.netcdf
            20060815-220926.010.netcdf[.gz]
            ...
        1.5/
        .../
    Velocity
        0.5/
        1.5/
        .../
    etc. for other fields
This directory structure was in effect for data obtained in May 2009, and may vary
substantially if your radar data were obtained prior to (or after) that date.

A dictionary of volume scans, keyed by scan time, is produced. Scan times are taken
from the file names, without examining the contents of each file.

Example
-------
>>> sweeps    = gather_sweeps_by_field('/Volumes/Backup/fulgurite/data/2006/PAR_new')
>>> volumes   = volumeize_sweeps(sweeps)
>>> vol_scans = align_volumes_across_fields(volumes)

>>> # first volume of reflectivity data
>>> print volumes['Reflectivity'][0]
>>> scan_times = vol_scans.keys()
>>> scan_times.sort() #no guarantee of order in a dictionary
>>> print vol_scans[scan_times[0]]['Reflectivity'][0] #date, el, filename

Copyright 2009, Eric Bruning, eric@deeplycloudy.com
License: BSD-like

"""

import os
import datetime
import numpy as np

def verify_numeric_string(s):
    try:
        return float(s)
    except ValueError:
        return None
        
def parse_filename(fn):
    """Should return a sortable date-time (string or otherwise)"""
    simple = fn.split('.')[0]
    date_part, time_part = simple.split('-')
     # = datetime_parts[0], datetime_parts[1]
    assert len(date_part) == 8 #YYYYMMDD
    assert len(time_part) == 6 #hhmmss
    Y,M,D = date_part[0:4], date_part[4:6], date_part[6:8]
    h,m,s = time_part[0:2], time_part[2:4], time_part[4:6]
    
    Y,M,D,h,m,s = map(int, (Y,M,D,h,m,s))
    
    date = datetime.datetime(Y,M,D,h,m,s)
    
    # return simple
    return date
    
def find_large_changes(sweep_list, thresh=5.0):
    els = np.fromiter((swp[1] for swp in sweep_list), float)
    el_changes = els[1:] - els[:-1]
    
    large_change, = np.where((el_changes > thresh))
    
    return large_change

def fix_sweep_ordering(sweeps):
    """ Assumes the volume is collected in the usual way, with sequentially
        increasing elevation angles with time until a reset to the lowest elev
        angle.
        
        Phased array radars can collect data so fast that sweeps can have the
        same time to the nearest second. There is enough metadata in each individual
        sweep file make this determination, but this algorithm handles it on the
        basis of what a volume scan should look like, without having to open each
        sweepfile individually.
        There should be no large positive elevation angle changes."""
    
    for field_type in sweeps:
        # sorts by time, then elevation angle. Necessary preliminary ordering.
        sweeps[field_type].sort()        

    # Sometimes the last scan in a volume and the first of the next volume will
    # be tagged with the same time. This causes the first scant of the new volume
    # to sort before the last scan of the old volume. e.g, 38, 40, 0.5, 41, 0.8, 1.5
    for field in sweeps.keys():
        large_change = find_large_changes(sweeps[field])
        
        # swap the index with the large change with the next index
        for idx in large_change:
            sweeps[field][idx:idx+2] = sweeps[field][idx+1] , sweeps[field][idx]
            
        # Throw an error if the previous step didn't take care of all the large
        # positive changes in elevation angle
        large_change = find_large_changes(sweeps[field])
        assert len(large_change) == 0
    
    
def volumeize_sweeps(sweeps):
    """ Returns a dictionary keyed by field type. The value for each field is a list of volume
        sweep lists contained for each field. The number of volumes for each field
        isn't necessarily the same, and require alignment before combination into a multi-field
        radar volume scan, in the traditional sense.
        
        Assumes the volume is collected in the usual way, with sequentially
        increasing elevation angles with time until a reset to the lowest elev
        angle.
        
        The following example shows how to examine the sorted elevation angles for some
        field in the sweeps structure
        >>> import matplotlib.pyplot as plt
        >>> els = np.fromiter((swp[1] for swp in sweeps['Reflectivity']), float)
        >>> el_changes = els[1:] - els[:-1]
        >>> plt.figure()
        >>> plt.plot(els)
        >>> plt.plot(el_changes)
        >>> plt.show
    """
    
    fix_sweep_ordering(sweeps)
    
    dataset = {}
    
    for field in sweeps:
        volumes = []
        sweep_list = sweeps[field]
        els = np.fromiter((swp[1] for swp in sweep_list), float)
        el_changes = els[1:] - els[:-1]
        
        neg_change, = np.where((el_changes < 0.0))
        low, high = [0]+list(neg_change+1), list(neg_change+1)+[len(els)]
        
        for low_idx, high_idx in zip(low, high):
            vol_this_field = sweep_list[low_idx:high_idx]
            volumes.append(vol_this_field)
        dataset[field] = volumes
    
    fields = sweeps.keys()
    
    
    return dataset

        
def gather_sweeps_by_field(directory):
    """ Walks directory containing subdirectories for field types and 
        sub-sub directories for elevation angles. Returns dictionary of 
        lists of tuples of (date-time, elevation) keyed by each field.
    """
    directory = os.path.normpath(directory)
    path_sep = os.path.sep
    
    root_path_parts = directory.split(path_sep)
    root_path_depth = len(root_path_parts)
    
    sweeps={}
    
    for dirpath, dirnames, filenames in os.walk(directory):
        path_parts = dirpath.split(path_sep)
        # os.path.join(dirpath, name)
                
        # first level directory is a field, with dirnames as the vcp
        if len(path_parts) - root_path_depth == 1:
            field = path_parts[-1]
            
            vcp = [verify_numeric_string(dirname) for dirname in dirnames if verify_numeric_string(dirname)]

            if field not in sweeps.keys():
                sweeps[field] = []
            print 'Found field', field, 'with VCP'
            print vcp

        if len(path_parts) - root_path_depth == 2:
            vcp_el = verify_numeric_string(path_parts[-1])
            
            if vcp_el is not None:
                # create a list of files, with tuple of (date-time, elevation, path name), which
                # will be sorted to create a list of sweeps adjacent in time and ascending in elevation
                nc_files = [(parse_filename(fn), vcp_el, os.path.join(dirpath, fn)) 
                                for fn in filenames if (fn.find('netcdf') >= 0)]
                sweeps[field].extend(nc_files)
                    
    return sweeps
    
    
def align_volumes_across_fields(dataset):
    """ Matches volume scans collected at the same time across fields in dataset.
        Assumes that the scan strategy is mostly sane, with no checking to see if
        some field-volumes are used more than once.
        
        Returns a dictionary of volume scans, keyed by start time.
        Each volume scan is further keyed by field name.     
        # scan_times = vol_scans.keys()
        # print vol_scans[scan_times[0]]['Reflectivity'][0] #date, el, filename
    """
    
    fields = dataset.keys()
    if 'Reflectivity' in fields:
        key_field = 'Reflectivity'
    else:
        key_field = fields[0]
    print 'Aligning volumes using', key_field
    
    # Align all fields in one volume scan using the list of volumes attached in the
    # to the key_field data.
    
    # date/time of first sweep in each volume scan
    vol_starts = {}
    for field in fields:
        vol_starts[field] = [(vol[0][0], vol) for vol in dataset[field]]
    
    vol_scans = {}

    # zero indexing gives time, 1-indexing gives the volume scan list
    # for each volume in the key field, find the volume in each other field
    # that begins the closest (before or after) to the key start time.
    key_starts = vol_starts[key_field]
    for key_start in key_starts:
        vol_scans[key_start[0]] = {}
        for field in fields:
            deltas = [(abs(start[0] - key_start[0]), start[1]) for start in vol_starts[field]]
            deltas.sort()
            vol_scans[key_start[0]][field] = deltas[0][1]
    
    return vol_scans
    
                
        
if __name__ == '__main__':
    sweeps    = gather_sweeps_by_field('/Volumes/Backup/fulgurite/data/2006/PAR_new')
    volumes   = volumeize_sweeps(sweeps)
    vol_scans = align_volumes_across_fields(volumes)
    
    # first volume of reflectivity data
    print volumes['Reflectivity'][0]
    scan_times = vol_scans.keys()
    print vol_scans[scan_times[0]]['Reflectivity'][0] #date, el, filename
    
