#!/usr/bin/python
import numpy as np
import os
import datetime

def pull_date(x):
    year=int(x[0:4])
    month=int(x[5:7])
    day=int(x[8:10])
    hour=int(x[11:13])
    minute=int(x[14:16])
    second=int(x[17:19])
    if len(x) == 19:
        return datetime.datetime(year,month,day,hour,minute,second)
    else:
        microsecond = int(x[20:])*100000
        return datetime.datetime(year,month,day,hour,minute,second,microsecond)

def sonics(top_dir,filedate):
    '''
        Read in all RMY Sonic Anemometer Data from the Joint Urban 2003 field campagin for a specific hour.
        top dir = path to directory where tower1 and tower 2 reside
        filedate = yyyy-mm-dd-hhhh
    '''
#
#       Open File
#
#CST,UTC,Julian,Ux1,Ux2,Ux3,Ux4,Ux5,Uy1,Uy2,Uy3,Uy4,Uy5,Uz1,Uz2,Uz3,Uz4,Uz5,
#Ts1,Ts2,Ts3,Ts4,Ts5,L1 Flag,L2 Flag,L3 Flag,L4 Flag,L5 Flag [0:27]
    filename = []
    filename.append(top_dir+'/tower1/'+filedate+'-tower1-ts.txt.gz')
    filename.append(top_dir+'/tower2/'+filedate+'-tower2-ts.txt.gz')
    try:

        date1 = np.loadtxt(filename[0], usecols=[1], dtype=datetime.datetime,converters={1:pull_date}, comments='%', delimiter=',')
        date2 = np.loadtxt(filename[1], usecols=[1], dtype=datetime.datetime,converters={1:pull_date}, comments='%', delimiter=',')
    except IOError:
        print '%s does not exist\n'%filename
        raise

    for i in range(0,2):

        data = np.loadtxt(filename[i], usecols=np.arange(3,28), comments='%', delimiter=',')

        u=data[:,0:5]
        v=data[:,5:10]
        w=data[:,10:15]
        T=data[:,15:20]
#        flags = data[:,20:25].astype(int)

        dt = np.dtype([('u1',np.float,(36000,5)),('v1',np.float,(36000,5)),
                       ('w1',np.float,(36000,5)),('T1',np.float,(36000,5))])
        if i == 0:
            rdata1=np.rec.fromarrays([u,v,w,T],names='u,v,w,T')
#            rdata1=np.array((u,v,w,T), dtype=dt)
        elif i == 1:
            rdata2=np.rec.fromarrays([u,v,w,T],names='u,v,w,T')
#            rdata2=np.array((u,v,w,T), dtype=dt)

    return (date1,date2,rdata1,rdata2)


if __name__=='__main__':
    import sys
    filename = sys.argv[1]

    data,ext=ltm_sonic(filename)
