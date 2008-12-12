#!/usr/bin/python

__all__ = ['turb_covar', 'rs']

import numpy as np
from generic import get_dims
from generic import get_pert

def cs_contrib(a,b,bins,subsamp=10):
    from generic import get_dims
    '''
a,b: raw time series (not perturbations)

Compute the contribution of coherent structures to the total flux of a'b'.

bins: list of arrays that contain np.arange(start_index,stop_index) for CS events
      the list is returned from metpy.vis.wavelet_plt() as the first member.

subsamp: number of samples in original time series for each sample in subsampled
timeseres (if timeseries was subsampled to calculate wavelet).
    '''
    sensors,obs,blks=get_dims(a,b)
    a=a.reshape((sensors,obs,blks))
    b=a.reshape((sensors,obs,blks))
    cohf_n=[]
    tcoh=[]
    ap=get_pert(a)
    bp=get_pert(b)
    for i in range(0,len(bins[0])):
        tcoh.append(len(bins[0][i])*subsamp)
        cohf_n.append(tcoh[i]*np.average(
                   ap[0,bins[0][i][0]*subsamp:bins[0][i][-1]*subsamp+subsamp-1,0]*
                   bp[0,bins[0][i][0]*subsamp:bins[0][i][-1]*subsamp+subsamp-1,0]))
        
    F_coh=np.array(cohf_n).squeeze().sum()/(a.shape[1]*turb_covar(a[0,:,0],b[0,:,0]))
    TE=F_coh/((1./a.shape[1])*np.array(tcoh).sum())
    return (F_coh,TE,tcoh)


def turb_covar(a,b,remove_mean=True):
    '''
Compute a turbulence covariance term (i.e. u'w', w'T'...)
    '''
    sensors,obs,blks = get_dims(a,b)
    a=a.reshape(sensors,obs,blks)
    b=b.reshape(sensors,obs,blks)
    t_cov = np.ones((sensors,blks))
    ax = 1
    for blk in range(0,blks):
        for sen in range(0,sensors):
            if remove_mean:
                t_cov[sen,blk] = np.average(a[sen,:,blk]*b[sen,:,blk])-\
                    np.average(a[sen,:,blk])*np.average(b[sen,:,blk])
            else:
                t_cov[sen,blk] = np.average(a[sen,:,blk]*b[sen,:,blk])

    return t_cov

def rs(u,v,w):
    '''
Compute the and return the Reynolds Stress Tensor. u,v,w:  matricies where \
the number of rows indicates the number off individual sensors and the number \
of columns indicates the number of points in the time series.

rs returns a tuple containing the six distinct terms found in the Reynolds \
Stress tensor (uu,vv,ww,uw,vw,uv), where each member of the array is a 1D \
numpy array of length sensor.
    '''

    uw = turb_covar(u,w)
    vw = turb_covar(v,w)
    uv = turb_covar(u,v)
    uu = turb_covar(u,u)
    vv = turb_covar(v,v)
    ww = turb_covar(w,w)

    return np.array((uu,vv,ww,uw,vw,uv)).swapaxes(0,1)
