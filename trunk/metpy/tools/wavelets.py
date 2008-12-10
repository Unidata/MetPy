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

class MotherWavelet(object):

    @staticmethod
    def get_coi_coef(sampf):
        """
        raise error if Cone of Influence coefficient is not set in subclass wavelet
        """
        raise NotImplementedError('coi_coef needs to be implemented in subclass wavelet')

    #add methods for computing cone of influence and mask
    def get_coi(self):
        """
        Compute cone of influence
        """
        y1 =  self.coi_coef*np.arange(0,self.coefs.shape[1]/2)
        y2 = -self.coi_coef*np.arange(0,self.coefs.shape[1]/2)+y1[-1]
        coi = np.r_[y1,y2] 
        self.coi = coi
        return coi

    def get_mask(self):
        """
        get mask for cone of influence.

        input:

            wavelet_shape : tuple containing shape of wavelet

            coi_coef : cone of influence coefficient 

            scales : 1d array of scales used in wavelet transform

        return

            mask : array of bools for use in np.ma.array('',mask=mask)

        """
        mask = np.ones(self.coefs.shape)
        masks = self.coi_coef*self.scales
        for s in range(0,len(self.scales)):
            if (s != 0) and (int(np.ceil(masks[s])) < mask.shape[1]):
                mask[s,np.ceil(int(masks[s])):-np.ceil(int(masks[s]))]=0
        self.mask = mask.astype(bool)
        return self.mask

class SDG(MotherWavelet):
    """
    Class for the SDG wavelet

    References

    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    Collineau, S. and Y. brunet, 1993: Detection of turbulent coherent motions in a
      forest canopy part 1: wavelet analysis.  Boundary-Layer Meteorology, 65,
      pp 357-379.

    Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavlet Analysis.
      Bulletin of the American Meteorological Society, 79, 1, pp. 61-78.

    """
    def __init__(self,len_wavelet=None,scales=None,sampf=1,normalize=True):
        self.sampf = sampf
        self.scales = scales
        self.len_wavelet = len_wavelet
        self.normalize = normalize
        self.coi_coef = self.get_coi_coef()
        self.coefs = self.get_coefs()
        self.name='second degree of a gaussian mother wavelet'    
        self.wdu = np.pi/np.sqrt(2.); # Collineau and Brunet 1993

    def get_coi_coef(self):
        """
        Compute the cone of influence coefficient
        """
        coi_coef=np.sqrt(2.)/self.sampf ;#Torrence and Compo 1998
        self.coi_coef = coi_coef
        return coi_coef

    def get_coefs(self):
        """
        Calculate the coefficients for the mother wavelet SDG
        """

        """
        Create array containing values used to evaluate the wavelet function
        """
        xi=np.arange(-self.len_wavelet/2.,self.len_wavelet/2.)

        if self.normalize is True:
            c=2./(np.sqrt(3.)*np.power(np.pi,0.25))
        else:
            c=1.
        """
        find coefficient at each scale
        """
        mw = np.empty((len(self.scales),self.len_wavelet))

        for s in range(len(self.scales)):
            xsd = -xi * xi / (self.scales[s] * self.scales[s])
            mw[s] = c * (1. + xsd) * np.exp(xsd / 2.)

        self.coefs = mw

        return mw


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
