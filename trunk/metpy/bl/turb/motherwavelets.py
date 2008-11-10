#!/usr/bin/python

import numpy as np

class MotherWavelet(object):
    @staticmethod
    def coi_coef(sampf):
        """
        raise error if Cone of Influence coefficient is not set in subclass wavelet
        """
        raise NotImplementedError('coi_coef needs to be implemented in subclass wavelet')

    #add methods for computing cone of influence and mask

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

    @staticmethod
    def coi_coef(sampf):
        """
        Compute the cone of influence coefficient
        """
        coi_coef=np.sqrt(2.)/sampf ;#Torrence and Compo 1998
        return coi_coef

    @staticmethod
    def coefs(len_wavelet,scales,normalize=True,sampf=1):
        """
        Calculate the coefficients for the mother wavelet SDG
        """

        """
        Create array containing values used to evaluate the wavelet function
        """
        xi=np.arange(-len_wavelet/2.,len_wavelet/2.)

        if normalize is True:
            c=2./(np.sqrt(3.)*np.power(np.pi,0.25))
        else:
            c=1.
        """
        find coefficient at each scale
        """
        for s in np.arange(0,len(scales)):
            xsd=xi/scales[s]
            if s != 0:
                mw_tmp = c*(1.-xsd*xsd)*np.exp(-xsd*xsd/2.)
                mw=np.c_[mw,mw_tmp]
            else:
                mw = c*(1.-xsd*xsd)*np.exp(-xsd*xsd/2.)
        return mw

    name='second degree of a gaussian mother wavelet'    
    wdu = np.pi/np.sqrt(2.); # Collineau and Brunet 1993
