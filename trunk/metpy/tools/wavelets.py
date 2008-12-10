#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

class Wavelet(object):

    def __init__(self,wt,wavelet):
        self.coefs = wt
        self.motherwavelet = wavelet

    def get_gws(self,valid=False):
        if valid:
            self=np.ma.array(self,mask=self.motherwavelet.get_mask())

        gws = np.power(np.abs(self.coefs),2).mean(axis=1); #Torrence and Compo 1998

        return gws

def cwt(x,wavelet):
    """
    cwt: compute the continuous wavelet transform of x using the mother wavelet `wavelet`.
        x: signal to be transformed
        wavelet: an instance of the class MotherWavelet as defined in motherwavelet.py
    """
    # Transform the signal and motherwavelet into the Fourier domain
    xf=fft(x)
    mwf=fft(wavelet.coefs,axis=1)
    # Convolve (mult. in Fourier space) and bring back
    #wt_tmp=ifft(mwf*xf[np.newaxis,:],axis=0)
    wt_tmp=ifft(mwf*xf.reshape(1,-1),axis=1).transpose()
    # Scale by the square root of the scales used in the wavelet transform
    wt = (np.fft.fftshift(wt_tmp,axes=[0])/np.sqrt(wavelet.scales)).transpose()
    # if motherwavelet was real, only keep real part of transform
    if np.all(np.negative(np.iscomplex(wavelet.coefs))):
        wt=wt.real

    return Wavelet(wt,wavelet)

def cxwt(x1,x2,wavelet,scales):
    '''
    Compute the cross-wavelet transform of 'x1' and 'x2' using the 
    'wavelet' basis set from scales 1 to 'max_scale'.

    x1,x2 - data to which the cross-wavelet transform is applied

    wavelet - instance of class MotherWavelet from motherwavelet.py

    
    '''
    print "working on x1"
    cwt1=cwt(x1,wavelet)
    print "working on x2"
    cwt2=cwt(x2,wavelet)

    print "working on xwt"
    xwt=cwt1*np.conjugate(cwt2)

    return xwt



def test():
    import time
    import motherwavelets as mw
#
#   create signal
#
    x=np.arange(0,100,0.1)/10*np.pi
    y1 = np.sin(x[0:200])
    y2=np.sin(x[200:400]*2)
    y3=np.sin(x[400:600]*4)
    y4=np.sin(x[600:800]*8)
    y5=np.sin(x[800:1000]*16)
    y=np.r_[y1,y2,y3,y4,y5]
#
#   get wavelet coefficients
#
    wt=cwt(y,mw.SDG,np.arange(1,150))


    return wt


"""
References

Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
  and Francis Group, New York/London. 353 pp.

Collineau, S. and Y. brunet, 1993: Detection of turbulent coherent motions in a
  forest canopy part 1: wavelet analysis.  Boundary-Layer Meteorology, 65,
  pp 357-379.

Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavlet Analysis.
  Bulletin of the American Meteorological Society, 79, 1, pp. 61-78.

"""
#def cwt(x,wavelet,scales,calc_gws=False,gws_valid=False,sampf=1):
#    """
#    cwt: compute the continuous wavelet transform of x using the wavelet `wavelet`.
#        x: signal to be transformed
#        wavelet: wavelet from motherwavelets.py
#        scales: 1D numpy array of scales
#        calc_gws: compute and return the global wavelet specturm (bool)
#        gws_valid: compute gws using only valid data (defined by cone of influence)
#        sampf: sample frequency of x in samples per unit time
#    """
#    # get mother wavelets at scale a
#    mw=wavelet.coefs(len(x),scales,sampf=sampf)
#    # Fourier, baby!
#    xf=fft(x)
#    mwf=fft(mw,axis=0)
#    # Convolve (mult. in Fourier space) and bring it back!
#    """
#    wt_tmp=np.abs(np.fft.ifft(mwf*xf.reshape(-1,1),axis=0).transpose())
#    """
#    wt_tmp=ifft(mwf*xf[np.newaxis,:],axis=0)
#    """
#    Compute cone of influence
#    """
#    y1=wavelet.coi_coef(sampf)*np.arange(0,len(x)/2)
#    y2=-wavelet.coi_coef(sampf)*np.arange(0,len(x)/2)+y1[-1]
#    coi = np.r_[y1,y2] 

#    wt = np.fft.fftshift(wt_tmp,axes=[1])/np.sqrt(scales[:,np.newaxis])

#    if np.all(np.negative(np.iscomplex(mw))):
#        wt=wt.real

#    if calc_gws is False:
#        return coi,wt
#    else:
##       Compute Global Wavelet Specturm
#        mask=get_mask(wt.shape,coi_coef,scales)

#        if gws_valid is True:
#            gws = get_gws(wt,mask=mask)
#        else:
#            gws = get_gws(wt)

#        return coi,gws,wt
#def get_mask(wavelet_shape,coi_coef,scales):
#    """
#    get mask for cone of influence.

#    input:

#        wavelet_shape : tuple containing shape of wavelet

#        coi_coef : cone of influence coefficient 

#        scales : 1d array of scales used in wavelet transform

#    return

#        mask : array of bools for use in np.ma.array('',mask=mask)

#    """
#    mask = np.ones(wavelet_shape)
#    masks = coi_coef*scales
#    for s in range(0,len(scales)):
#        if (s != 0) and (int(np.ceil(masks[s])) < mask.shape[1]):
#            mask[s,np.ceil(int(masks[s])):-np.ceil(int(masks[s]))]=0

#    return mask.astype(bool)


#def SDG(len_wavelet,scales,normalize=True,return_wdu=False,sampf=1):
#    """
#    Compute the Second Derivative of a Gaussian (SDG) wavelet of length
#    'len_wavelet' at scales 'scales' (numpy array) in time domain.

#    for normalize = True (give wavelet unit energy)
#        W(a)=2/(sqrt(3)*PI^(0.25))*(1-x^2)*exp(-x^2/2) (Addison 2002)
#    for normalize = False
#        W(a)=(1-x^2)*exp(-x^2/2) (Addison 2002)

#    len_wavelet: length of wavelet

#    return_wdu : return the wavelet durration unit (relates scale to period)

#    The returned coefficients are centered (x=0) at len_w/2
#    """
#    coi_coef=np.sqrt(2.)/sampf ;#Torrence and Compo 1998
#    wdu = np.pi/np.sqrt(2.); # Collineau and Brunet 1993
##
##   Create array containing values used to evaluate the wavelet function
##
#    xi=np.arange(-len_wavelet/2.,len_wavelet/2.)
##
##   Compute wavelet function
##
#    if normalize is True:
#        c=2./(np.sqrt(3.)*np.power(np.pi,0.25))
#    else:
#        c=1.

#    for s in np.arange(0,len(scales)):
#        xsd=xi/scales[s]
#        if s != 0:
#            mw_tmp = c*(1.-xsd*xsd)*np.exp(-xsd*xsd/2.)
#            mw=np.c_[mw,mw_tmp]
#        else:
#            mw = c*(1.-xsd*xsd)*np.exp(-xsd*xsd/2.)
#    if return_wdu:
#        return coi_coef,wdu,mw
#    else:
#        return coi_coef,mw

#def cwt_time(x,wavelet,max_scale):
#    """
#    Compute the wavelet transform (in the time domain using direct convolution)
#    of 'x' using the 'wavelet' basis set from scales 1 to 'max_scale'.

#    x - data to which the wavelet transform is applied

#    wavelet - wavelet function (for example, SDG)

#    max_scale - maximum scale used to compute the wavelet transform
#    """
#    wt=[]
##
##   loop over scales from 1 to max_scale
##
#    for a in range(1,max_scale+1):
##        if a in range(0,max_scale,100):
##            print '    working on ' +`a` + "'s scale"
##
##       length of wavelet function (based on scale to mitigate truncation of
##         wavelet function - 10*a seems to work well for the SDG wavelet, may need
##         to be tweaked for others.
##
#        len_w=len(x)+10*a
##
##       indicies to pull out array of len(x) centered on the full wavelet vector
##
#        indexa=(len_w-len(x))/2
#        indexb=(len_w+len(x))/2
##
##       compute wavelet coefficients at scale a
##
#        wt.append((1./np.sqrt(a)*np.convolve(x,np.conjugate(wavelet(len_w,a)),mode='same'))[indexa:indexb])

#    return np.array(wt)

