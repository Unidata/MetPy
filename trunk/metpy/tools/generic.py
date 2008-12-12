#!/usr/bin/python

import numpy as np

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
