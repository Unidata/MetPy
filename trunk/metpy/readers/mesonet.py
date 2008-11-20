#!/usr/bin/env python
import numpy as np
from numpy.ma import MaskedArray

mesonet_vars = ['STID', 'STNM', 'TIME', 'RELH', 'TAIR', 'WSPD', 'WVEC', 'WDIR',
    'WDSD', 'WSSD', 'WMAX', 'RAIN', 'PRES', 'SRAD', 'TA9M', 'WS2M', 'TS10',
    'TB10', 'TS05', 'TB05', 'TS30', 'TR05', 'TR25', 'TR60', 'TR75']

def remote_mesonet_ts(site, date=None, fields=None, unpack=True):
    '''
    Reads in Oklahoma Mesonet timeseries data directly from their servers.

    site : string
        The station id for the data to be fetched.  This is
        case-insensitive.

    date : datetime object
        A python :class:`datetime` object specify that date
        for which that data should be downloaded.

    fields : sequence
        A list of the variables which should be returned.  See
        :func:`read_mesonet_ts` for a list of valid fields.

    unpack : bool
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = remote_mesonet_ts(...)``. Defaults to True.

    Returns : array
        A nfield by ntime masked array.  nfield is the number of fields
        requested and ntime is the number of times in the file.  Each
        variable is a row in the array.  The variables are returned in
        the order given in *fields*.
    '''
    import urllib2
    
    if date is None:
        import datetime
        date = datetime.date.today()
    
    #Create the various parts of the URL and assemble them together
    path = '/mts/%d/%d/%d/' % (date.year, date.month, date.day)
    fname = '%s%s.mts' % (dt.strftime('%Y%m%d'), site.lower())
    baseurl='http://www.mesonet.org/public/data/getfile.php?dir=%s&filename=%s'
    
    #Open the remote location
    datafile = urllib2.urlopen(baseurl % (path+fname, fname))
    print datafile.url
    
    #Read the data 
    #Numpy.loadtxt checks prohibit actually doing this, though there's no
    #reason it can't work.  I'll file a bug.  The next two lines work around it
    from cStringIO import StringIO
    datafile = StringIO(datafile.read())
    return read_mesonet_ts(datafile, fields, unpack)

def read_mesonet_ts(filename, fields=None, unpack=True):
    '''
    Reads an Oklahoma Mesonet time series file from *filename*.

    filename : string or file-like object
        Location of data. Can be anything compatible with
        :func:`numpy.loadtxt`, including a filename or a file-like
        object.

    fields : sequence
        List of fields to read from file.  (Case insensitive)
        Valid fields are:
            STID, STNM, TIME, RELH, TAIR, WSPD, WVEC, WDIR, WDSD,
            WSSD, WMAX, RAIN, PRES, SRAD, TA9M, WS2M, TS10, TB10,
            TS05, TB05, TS30, TR05, TR25, TR60, TR75
        The default is to return all fields.

    unpack : bool
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = read_mesonet_ts(...)``. Defaults to True.


    Returns : array
        A nfield by ntime masked array.  nfield is the number of fields
        requested and ntime is the number of times in the file.  Each
        variable is a row in the array.  The variables are returned in
        the order given in *fields*.
    '''
    MISSING = -995
    FUTURE = -996
    
    if fields is None:
        cols = None
    else:
        cols = [mesonet_vars.index(f.upper()) for f in fields]

    data = np.loadtxt(filename, skiprows=3, usecols=cols, unpack=unpack)

    #Mask out data that are missing or have not yet been collected
    mask = (data == MISSING) | (data == FUTURE)
    return MaskedArray(data, mask=mask)

if __name__ == '__main__':
    import datetime
    from optparse import OptionParser

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    #Create a command line option parser so we can pass in site and/or date
    parser = OptionParser()
    parser.add_option('-s', '--site', dest='site', help='get data for SITE',
        metavar='SITE', default='nrmn')
    parser.add_option('-d', '--date', dest='date', help='get data for YYYYMMDD',
        metavar='YYYYMMDD', default=None)
    
    #Parse the command line options and convert them to useful values
    opts,args = parser.parse_args()
    if opts.date is not None:
        dt = datetime.datetime.strptime(opts.date, '%Y%m%d')
    else:
        dt = None
    
    time, relh, temp, wspd, press = remote_mesonet_ts(opts.site, dt,
        ['time', 'relh', 'tair', 'wspd', 'pres'])

    time = time/60.
    
    ax = plt.subplot(3,1,1)
    plt.plot(time, relh, 'g')
    plt.ylabel('Rel. Humidity (%)')
    plt.xlim(0,24)
    plt.title('Meteogram for %s on %s' % (opts.site.upper(), dt.date()))
    ax.xaxis.set_major_formatter(NullFormatter())
    
    ax = plt.subplot(3,1,2)
    plt.plot(time, temp, 'r')
    plt.ylabel('Temperature (C)')
    plt.xlim(0,24)
    ax.xaxis.set_major_formatter(NullFormatter())
    
    ax = plt.subplot(3,1,3)
    plt.plot(time, press, 'k')
    plt.xlabel('Time (hour)')
    plt.ylabel('Pressure (mb)')
    ax2 = plt.twinx(ax)
    ax2.plot(time, wspd, 'b')
    ax2.set_ylabel('Wind Speed (m/s)')
    plt.xlim(0,24)
    plt.xticks(range(0,25,3))
    
    plt.subplots_adjust(hspace=0.1)    
    plt.show()   
