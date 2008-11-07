#!/usr/bin/python
import numpy as np
#import scipy as S
from matplotlib.mlab import csd as csd_welch
from matplotlib.mlab import cohere as coherence
from metpy.generic import get_dims
from scipy.integrate import cumtrapz

from matplotlib import cbook
def detrend_none(x):
    "Return x: no detrending"
    return x

def window_hanning(x):
    "return x times the hanning window of len(x)"
    return np.hanning(len(x))*x

def psd(x, blocksize=256, NFFT=256, Fs=1, detrend=detrend_none,
        window=window_hanning, noverlap=0):
    """
    The power spectral density by Welches average periodogram method.
    The vector x is divided into blocklength length segments.  Each segment
    is detrended by function detrend and windowed by function window.
    noperlap gives the length of the overlap between segments.  The
    absolute(fft(segment))**2 of each segment are averaged to compute Pxx,
    with a scaling to correct for power loss due to windowing.

    Fs is the sampling frequency (samples per time unit).  It is used
    to calculate the Fourier frequencies, freqs, in cycles per time
    unit.

    -- detrend is a function, unlike in matlab where it is a vector.
    -- window can be a function or a vector of length blocksize. To create window
       vectors see numpy.blackman, numpy.hamming, numpy.bartlett,
       scipy.signal, scipy.signal.get_window etc.
    -- if length x < blocksize, it will be zero padded to blocksize


    Returns the tuple Pxx, freqs

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """

    x = np.asarray(x) # make sure we're dealing with a numpy array

    blocksize=int(blocksize)
    NFFT=int(NFFT)

    # zero pad x up to blocksize if it is shorter than blocksize
    if len(x)<blocksize:
        n = len(x)
        x = np.resize(x, (blocksize,))    # Can't use resize method.
        x[n:] = 0

    # for real x, ignore the negative frequencies
    if np.iscomplexobj(x): numFreqs = blocksize
    else: numFreqs = blocksize//2+1

    if cbook.iterable(window):
        assert(len(window) == blocksize)
        windowVals = window
    else:
        windowVals = window(np.ones((blocksize,),x.dtype))

    step = blocksize-noverlap
    ind = range(0,len(x)-blocksize+1,step)
    n = len(ind)
    Pxx = np.zeros((numFreqs,n), np.float_)
    # do the ffts of the slices
    for i in range(n):
        thisX = x[ind[i]:ind[i]+blocksize]
        thisX = windowVals * detrend(thisX)
        fx = (np.absolute(np.fft.fft(thisX,n=NFFT))**2)/(blocksize)
        Pxx[:,i] = fx[:numFreqs]

    if n>1:
        Pxx = Pxx.mean(axis=1)
    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2
    Pxx /= (np.abs(windowVals)**2).sum()/(blocksize)

    freqs = Fs/np.float(NFFT) * np.arange(numFreqs)

    return Pxx, freqs


def ogive(coPxx,delf):
    sensors,obs,blks = get_dims(coPxx)
    coPxx=coPxx.reshape(sensors,obs,blks)
    og = np.zeros((sensors,obs-1,blks))

    ogt=cumtrapz(coPxx[:,::-1,:],dx=delf,axis=1)
    og=ogt[:,::-1,:]

    return og

def psd_test():
    import matplotlib.pyplot as plt
    from random import random

    #create simple signal
    #x=np.arange(0,400,np.pi/10.)
    x=np.arange(0,80,np.pi/80.)
    y=np.sin(5*x)+np.sin(15*x)
    for i in range(0,len(y)):
        y[i]=y[i]+random()

    fig=plt.figure()
    ax1=fig.add_subplot(2,1,1)
    ax1.plot(x,y)

    Pxx1,f1=psd(y,blocksize=len(x),NFFT=len(x))
    Pxx2,f2=psd(y,blocksize=len(x),NFFT=len(x)*2)
    Pxx4,f4=psd(y,blocksize=len(x),NFFT=len(x)*4)

    ax2=fig.add_subplot(2,3,4)
    ax2.semilogy(f1,Pxx1)
    ax2.semilogy(f2,Pxx2)
    ax2.semilogy(f4,Pxx4)

    Pxx1,f1=psd(y,blocksize=len(x),NFFT=len(x))
    Pxx2,f2=psd(y,blocksize=len(x)/2.,NFFT=len(x))
    Pxx4,f4=psd(y,blocksize=len(x)/4.,NFFT=len(x))
    ax3=fig.add_subplot(2,3,5,sharex=ax2,sharey=ax2)
    ax3.semilogy(f1,Pxx1)
    ax3.semilogy(f2,Pxx2)
    ax3.semilogy(f4,Pxx4)

    Pxx1,f1=psd(y,blocksize=len(x)/2.,NFFT=len(x),noverlap=0)
    Pxx2,f2=psd(y,blocksize=len(x)/2.,NFFT=len(x),noverlap=int(0.05*len(x)/2.))
    Pxx4,f4=psd(y,blocksize=len(x)/2.,NFFT=len(x),noverlap=int(0.2*len(x)/2.))
    ax4=fig.add_subplot(2,3,6,sharex=ax2,sharey=ax2)
    ax4.semilogy(f1,Pxx1)
    ax4.semilogy(f2,Pxx2)
    ax4.semilogy(f4,Pxx4)

    plt.show()

