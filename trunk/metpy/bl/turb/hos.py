#!/usr/bin/python
import numpy as N
from metpy.generic import get_pert

def kurtosis(a):
    '''
Compute the kurtosis of the timeseries a
    '''
    ax=1

    ap = get_pert(a)
    K = N.average(N.power(ap,4),axis=ax)/N.power(N.std(ap),4)

    return K

def skewness(a):
    '''
Compute the skewness of the timeseries a
    '''
    ax = 1

    ap = get_pert(a)
    S = N.average(N.power(ap,3),axis=ax)/N.power(N.std(ap),3)

    return S

def tke(u,v,w):
    '''
Compute the turbulence kinetic energy from the time series of the velocity \
components u,v, and w.
    '''
    ax=1

    up = get_pert(u)
    vp = get_pert(v)
    wp = get_pert(w)

    tke = N.power(N.average(N.power(up,2),axis=ax)+\
                  N.average(N.power(vp,2),axis=ax)+\
                  N.average(N.power(wp,2),axis=ax),0.5)

    return tke

