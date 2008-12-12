#!/usr/bin/python

__all__ = ['csd_welch','coherence','ogive','psd_welch']

import numpy as np
from matplotlib.mlab import psd as psd_welch
from matplotlib.mlab import csd as csd_welch
from matplotlib.mlab import cohere as coherence
from metpy.bl.generic import get_dims
from scipy.integrate import cumtrapz

def ogive(coPxx,delf):
    sensors,obs,blks = get_dims(coPxx)
    coPxx=coPxx.reshape(sensors,obs,blks)
    og = np.zeros((sensors,obs-1,blks))

    ogt=cumtrapz(coPxx[:,::-1,:],dx=delf,axis=1)
    og=ogt[:,::-1,:]

    return og
