#!/usr/bin/python

import numpy as np
import pylab as P

def block(a,splits=1):
    '''
computes 'splits' number of equal length averaging blocks of the time series a

example:  return 15 minute blocks out of an one hour long time series -

          block(a,4) (4 splits because there are 4 15-minute averages in an hour)
    '''
    if splits < 1:
        splits = 1
        print "don't be stupid...splits cannot be less than 1"
#
#   try to find shape of array by first assuming more than one sensor
#
    try:
        r=np.shape(a)[0]
        c=np.shape(a)[1]
    except (ValueError,IndexError):
#
#       assume only one sensor
# 
        r=1
        c=np.shape(a)[0]
#
#   create block assuming one of two cases, in order: multiple sensors or 
#     one sensor
# 
    if r != 1:
        sp_a=a.reshape(a.shape[0],splits,-1).swapaxes(1,2)
    else:
        sp_a=a.reshape(splits,-1).swapaxes(0,1)

    return sp_a

def block_apply(func,*blocks):
    '''
Apply the function func() to the arrays of blocks (arrays returned from block(a,splits)
    '''
#
#   try to find shape of array by first assuming more than one sensor
#
    try:
        sensors,obs,blks=blocks[0].shape
    except ValueError:
        sensors=None
        blks,obs=blocks[0].shape
#
#   apply function to each block
#
    for s in range(0,blks):
        blocks2=[]
        for arg in blocks:
            try:
                blocks2.append(arg.swapaxes(0,2)[s].swapaxes(0,1))
            except ValueError:
                blocks2.append(arg[s].swapaxes(0,1))
        blocks2=tuple(blocks2)
        if s !=0:
            b=np.c_[b,func(*blocks2)]
        else:
            b = func(*blocks2)
#
#   reshape the arrays for consistancy
#
    try:
        b=b.reshape(blks,-1,sensors).swapaxes(0,2)
    except NameError,ValueError:
        b=b.reshape(blks,-1)

    return b

def find_critical_points(x,mode="all",adj=0.01):
    """
    Find the critical points for the timeseries x (1-D array).
    """
#
#   approximate 1st and 2nd derivitives
#
    dx=[]
    for i in range(0,len(x)-1):
        dx.append(x[i+1]-x[i])
    dx=np.array(dx)

    dx2=find_extrema(dx,mode=mode)
#
# look for sign change in dx2
#
    return find_zero_crossings(dx2,mode=mode)


def find_extrema(x,mode="all",adj=0.01):
    """
    Find the local extreama for the timeseries x (1-D array).
    """
#
#   approximate derivitive
#
    dx=[]
    for i in range(0,len(x)-1):
        dx.append(x[i+1]-x[i])
    dx=np.array(dx)
#
# look for sign change in dx
#
    return find_zero_crossings(dx,mode=mode,adj=adj)

def find_zero_crossings(x,mode='all',adj = 0.01):

    if mode == 'negative':
        cond = np.sign(-1.)
    elif mode == 'positive':
        cond = np.sign(1.)
    elif mode == 'all':
        cond = True
    elif mode == 'close':
        cond = True
        x = x - adj
    else: 
        print('rtfm')

    s = np.sign(x)
    current_sign=s[0]

    zc=[]
    for i in range(1,len(s)):
        if (((s[i]!=current_sign) and ((cond==True) or (s[i]==cond)))):
            zc.append(i)
            current_sign=s[i]
    return np.array(zc)
        

def get_dims(*args):
    '''
Get the dimensions from the arguments.  If the dimensions are not consistant, raise error.
    '''
#
#   try to find shape of array by first assuming more than one sensor
#
    sen = []
    ob = []
    blk = []
    flg_error = False

    for i in range(0,len(args)):
        try:
            sensors,obs,blks=args[i].shape

        except ValueError:
            try:
                blks=1
                sensors,obs=args[i].shape
            except ValueError:
                sensors=1
                blks=1
                obs=args[i].shape[0]
        if i != 0:
            if not((sen == sensors) & (ob == obs) & (blk==blks)):
                flg_error = True

        else:
            sen = sensors
            ob = obs
            blk = blks         
              
    if not flg_error:
        return (sensors,obs,blks)
    else:
        print "diminsions not consistant"
        print "generic/getdims.py"

def get_pert(a):
    '''
Compute the time series of the perturbations of a
    '''
    sensors,obs,blks = get_dims(a)
    ap=np.ones((sensors,obs,blks))


    for blk in range(0,blks):
        for sen in range(0,sensors):
            ap[sen,:,blk] = a[sen,:,blk] - np.average(a[sen,:,blk])
    '''
    try:
        sensors,obs = np.shape(a)
        ap = np.ones((sensors,obs))
        for sen in range(0,sensors):
            ap[sen,:] = a[sen,:] - np.average(a[sen,:])
    except ValueError:
        ap = a - np.average(a)
    '''
    return ap

def wswd(u,v,w):
    '''
Compute the wind speed (horizontal and vector) and wind direction given u,v and w.
    '''
    hws = np.sqrt(u*u+v*v)
    vws = np.sqrt(u*u+v*v+w*w)
    wd = np.arctan2(-u,-v)*180./np.pi
    wd[wd<0]=360+wd[wd<0]

    return hws,vws,wd

def horizontal_align_fix(u,v,h_offset):
    '''
Compute a new u and v after applying a horizontal vertical offset (angle \
in degrees)
    '''
    hws = np.sqrt(u*u+v*v)
    wd = np.arctan2(-u,-v)*180./np.pi
    wd[wd<0]=360+wd[wd<0]

    wd_new = wd + h_offset
    wd_new[wd_new<0]=360+wd_new[wd_new<0]
    wd_new[wd_new>360]=wd_new[wd_new>360]-360

    u_new = -hws*np.sin(wd_new*np.pi/180.)
    v_new = -hws*np.cos(wd_new*np.pi/180.)

    return u_new, v_new

def running_average(x,length):
    '''

    x: data vector
    length: number of points in moving average

    '''
    window=np.ones(int(length))
    tmp=np.convolve(window,x,mode='same')/length
    for i in range(0,length):
        tmp[i]=x[i:i+length].mean()
        tmp[x.shape[0]-i-1]=x[x.shape[0]-length-i:x.shape[0]-i].mean()

    return tmp

def congruence(x,y):
    tx = np.array(x).flatten()
    ty = np.array(y).flatten()
    tmp = np.sum(tx*ty)/np.sqrt(np.sum(tx*tx)*np.sum(ty*ty))

    return tmp
#
# TEST ZONE!
#
def wswd_test():
    import pylab as P
    x=np.arange(0,2*np.pi,0.001)
    u=np.cos(x)
    v=np.sin(x)
    w=2.*np.sin(x*2.)*np.cos(x*2.)
    hws,vws,wd=wswd(u,v,w)
    ax1 = P.subplot(1,1,1)
    P.plot(x,u,'r')
    P.plot(x,v,'b')
    ax2 = P.twinx()
    P.plot(x,wd,'g')
    ax2.yaxis.tick_right()
    P.show()    

