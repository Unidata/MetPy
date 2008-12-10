#!/usr/bin/python

import numpy as np

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


#class SDG(MotherWavelet):
#    """
#    Class for the SDG wavelet

#    References

#    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
#      and Francis Group, New York/London. 353 pp.

#    Collineau, S. and Y. brunet, 1993: Detection of turbulent coherent motions in a
#      forest canopy part 1: wavelet analysis.  Boundary-Layer Meteorology, 65,
#      pp 357-379.

#    Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavlet Analysis.
#      Bulletin of the American Meteorological Society, 79, 1, pp. 61-78.

#    """

#    @staticmethod
#    def coi_coef(sampf):
#        """
#        Compute the cone of influence coefficient
#        """
#        coi_coef=np.sqrt(2.)/sampf ;#Torrence and Compo 1998
#        return coi_coef

#    @staticmethod
#    def coefs(len_wavelet,scales,normalize=True,sampf=1):
#        """
#        Calculate the coefficients for the mother wavelet SDG
#        """

#        """
#        Create array containing values used to evaluate the wavelet function
#        """
#        xi=np.arange(-len_wavelet/2.,len_wavelet/2.)

#        if normalize is True:
#            c=2./(np.sqrt(3.)*np.power(np.pi,0.25))
#        else:
#            c=1.
#        """
#        find coefficient at each scale
#        """
#        mw = np.empty((len(scales),len_wavelet))

#        for s in range(len(scales)):
#            xsd = -xi*xi / (scales[s] * scales[s])
#            mw[s] = c * (1. + xsd) * np.exp(xsd / 2.)

#        return mw.transpose()

#    name='second degree of a gaussian mother wavelet'    
#    wdu = np.pi/np.sqrt(2.); # Collineau and Brunet 1993
