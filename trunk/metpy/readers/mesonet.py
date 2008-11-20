#!/usr/bin/env python
import numpy as np
from numpy.ma import MaskedArray

def read_mts(filename):
    '''
    Reads the Mesonet time series file *filename*.  *filename* can be anything
    compatible wity :func:`numpy.loadtxt`.
    '''
    MISSING = -995
    FUTURE = -996
    data = np.loadtxt(filename, skiprows=3, usecols=(2,3,4,5,12),
        unpack=True)

    #Mask out data that are missing or have not yet been collected
    mask = (data == MISSING) | (data == FUTURE)
    return MaskedArray(data, mask=mask)

if __name__ == '__main__':
    import datetime, os.path, urllib2
    from optparse import OptionParser

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    #Create a command line option parser so we can pass in site and/or date
    parser = OptionParser()
    parser.add_option('-s', '--site', dest='site', help='get data for SITE',
        metavar='SITE', default='nrmn')
    
    #This creates a string for today's date to use as a default
    today_str = datetime.date.today().strftime('%Y%m%d')
    parser.add_option('-d', '--date', dest='date', help='get data for YYYYMMDD',
        metavar='YYYYMMDD', default=today_str)
    
    #Parse the command line options and convert them to useful values
    opts,args = parser.parse_args()
    dt = datetime.datetime.strptime(opts.date, '%Y%m%d')
    site = opts.site.lower()
    
    #Create the various parts of the URL and assemble them together
    path = '/mts/%d/%d/%d/' % (dt.year, dt.month, dt.day)
    fname = '%s%s.mts' % (dt.strftime('%Y%m%d'), site)
    baseurl='http://www.mesonet.org/public/data/getfile.php?dir=%s&filename=%s'
    
    #Open the remote location
    datafile = urllib2.urlopen(baseurl % (path+fname, fname))
    print datafile.url
    
    #Read the data    
    #Numpy.loadtxt checks prohibit actually doing this, though there's no
    #reason it can't work.  I'll file a bug.
    #time, relh, temp, wspd, press = read_mts(datafile)
    from cStringIO import StringIO
    time, relh, temp, wspd, press = read_mts(StringIO(datafile.read()))

    time = time/60.
    
    ax = plt.subplot(3,1,1)
    plt.plot(time, relh, 'g')
    plt.ylabel('Rel. Humidity (%)')
    plt.xlim(0,24)
    plt.title('Meteogram for %s on %s' % (site.upper(), dt.date()))
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
