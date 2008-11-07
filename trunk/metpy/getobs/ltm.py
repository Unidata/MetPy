#!/usr/bin/python
import numpy as np
import os
import datetime

def pull_date(x):
    year=int(x[1:5])
    month=int(x[6:8])
    day=int(x[9:11])
    hour=int(x[12:14])
    minute=int(x[15:17])
    second=int(x[18:20])

    return datetime.datetime(year,month,day,hour,minute,second)


def sonic_2005(filename):
    '''
        Read in RMY Sonic Anemometer Data from the Lake Thunderbird Micronet Spring 2005
    '''
#
#   Determine file type
#
    if os.path.splitext(os.path.basename(filename))[1]=='.txt':
#
#       Open File
#
        try:
            date = np.loadtxt(filename,delimiter=',',skiprows=2,usecols=np.arange(0,1),dtype=datetime.datetime,converters={0:pull_date})
            u1,u2,u3,u4,u5,v1,v2,v3,v4,v5,w1,w2,w3,w4,w5,T1,T2,T3,T4,T5,voltage= \
                np.loadtxt(filename, usecols=np.arange(1,22), comments='%', delimiter=',',unpack=True)
            u=np.r_[u1,u2,u3,u4,u5]
            v=np.r_[v1,v2,v3,v4,v5]
            w=np.r_[w1,w2,w3,w4,w5]
            T=np.r_[T1,T2,T3,T4,T5]

            dt = np.dtype([('u',np.float),('v',np.float),('w',np.float),
                      ('T',np.float)])       
            data=np.array(zip(u,v,w,T), dtype=dt).reshape(5,-1)

            dt = np.dtype([('date',object),('voltage',np.float)])
            ext = np.array(zip(date,voltage), dtype=dt) 

        except IOError:
            print '%s does not exist\n'%filename
            raise
    
        return data,ext

    elif os.path.splitext(os.path.basename(filename))[1]=='.nc':
        import nio

        f=nio.open_file(filename,mode='r')
        u=f.variables['u'][:]
        v=f.variables['v'][:]
        w=f.variables['w'][:]
        T=f.variables['T'][:]
        f.close()

        data = np.rec.fromarrays([u,v,w,T],names='u,v,w,T')

        return data

def sonic(filename, L5_fix=True):
    '''
        Read in RMY Sonic Anemometer Data from the Lake Thunderbird Micronet year >=2007
          --includes fix for level 5 alignment as default (controlled with keyword
            L5_fix (L5_fix=True default)
    '''
    from metpy.generic import horizontal_align_fix as haf
#
#   Determine file type
#
    if ((os.path.splitext(os.path.basename(filename))[1]=='.dat')|(os.path.splitext(os.path.basename(filename))[1]=='.txt')|(os.path.splitext(os.path.basename(filename))[1]=='.gz')):
#
#       Open File
#
        try:
            date = np.loadtxt(filename,delimiter=',',skiprows=2,usecols=np.arange(0,1),dtype=datetime.datetime,converters={0:pull_date})
            u1,v1,w1,T1,u2,v2,w2,T2,u3,v3,w3,T3,u4,v4,w4,T4,u5,v5,w5,T5= \
                np.loadtxt(filename, usecols=np.arange(2,22), comments='%', delimiter=',',skiprows=2,unpack=True)
            u5,v5=haf(u5,v5,-15.3)
            u=np.r_[u1,u2,u3,u4,u5]
            v=np.r_[v1,v2,v3,v4,v5]
            w=np.r_[w1,w2,w3,w4,w5]
            T=np.r_[T1,T2,T3,T4,T5]

            dt = np.dtype([('u',np.float),('v',np.float),('w',np.float),
                      ('T',np.float)])       
            data=np.array(zip(u,v,w,T), dtype=dt).reshape(5,-1)

        except IOError:
            print '%s does not exist\n'%filename
            raise
    
        return date,data

    elif os.path.splitext(os.path.basename(filename))[1]=='.nc':
        import nio

        f=nio.open_file(filename,mode='r')
        u=f.variables['u'][:]
        v=f.variables['v'][:]
        w=f.variables['w'][:]
        T=f.variables['T'][:]
        f.close()

        data = np.rec.fromarrays([u,v,w,T],names='u,v,w,T')

        return data

if __name__=='__main__':
    import sys
    filename = sys.argv[1]

    data,ext=ltm_sonic(filename)
