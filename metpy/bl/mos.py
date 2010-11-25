#!/usr/bin/python
import numpy as np

__all__ = ['u_star', 'theta_star', 'obu_length']

def u_star(u,v,w):
    '''
    Compute the friction velocity, u_star, from the timeseries of the velocity \
    components u, v, and w (an nD array)
    '''
    from metpy.bl.turb.fluxes import rs as R
    rs = R(u,v,w)
    uw = rs[3]
    vw = rs[4]

    us = np.power(np.power(uw,2)+np.power(vw,2),0.25)

    return us

def theta_star(u,v,w,T):
    '''
    Compute the friction temperature, theta_star, from the timeseries of the velocity \
    components u, v, and w, and temperature (an nD array)
    '''
    from metpy.bl.turb.fluxes import turb_covar as TC

    ts = -TC(w,T)/u_star(u,v,w)

    return ts

def obu_length(u,v,w,T):
    '''
    Compute the Obukhov Length, L, using the timeseries of the velocity \
    components u, v, and w, and temperature (an nD array)
    '''
    from metpy.constants import g
    L = np.power(u_star(u,v,w),2)*np.average(T)/(0.4*g*theta_star(u,v,w,T))

    return L
