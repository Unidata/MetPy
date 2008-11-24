#!/usr/bin/python

import numpy as np
import metpy
import metobs
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from pylab import poly_between

def find_peaks(coefs,peak,cs_threshold=0.40,mask=None):
    gws = metobs.bl.turb.wavelets.get_gws(coefs,mask)
#   find all extrema in the gws
    gws_extrema=metpy.generic.find_extrema(gws,mode='all')
    if gws_extrema.shape[0] == 0:
        print 'trying close'
        gws_extrema=metpy.generic.find_extrema(gws,mode='close',adj=1)

    if gws_extrema.shape[0] != 0:
#       look for extrema and zero crossings at particular scale
        cs_scale_extrema=metpy.generic.find_extrema(coefs[gws_extrema[peak],:],mode='all')
        cs_scale_zero_crossings=metpy.generic.find_zero_crossings(coefs[gws_extrema[peak],:],mode='all')
    #   threshold for calling an event as CS event - 100*cs_threshold% of peak coefficient
        threshold=np.abs(coefs[gws_extrema[peak],cs_scale_extrema]).max()*cs_threshold
    #   find start/stop indicies for cs events
        start=[]
        stop=[]
    #   for scale in extrema along peak in gws
        for i in range(1,len(cs_scale_extrema)):
    #       print coefs[gws_extrema[peak],cs_scale_extrema[i]]>threshold,coefs[gws_extrema[peak],cs_scale_extrema[i-1]]<0
    #       if coef at extrema is larger than threshold and coef at previous extrema is negative
            if (coefs[gws_extrema[peak],cs_scale_extrema[i]]>threshold) & (coefs[gws_extrema[peak],cs_scale_extrema[i-1]]<0):
                try:
    #               stop indicated by zero crossing following extrema of interest
                    stop.append(cs_scale_zero_crossings[cs_scale_zero_crossings>cs_scale_extrema[i]][0])
    #               start indicated by extrema previous to zero crossing
                    start.append(cs_scale_extrema[i-1])
                except IndexError:
                    pass

        return start,stop,gws_extrema
    else:
        print 'no extrema found'
        return 0,0,0
    
def wavelet_plt(ts,c,**kwargs):

    fontsize=18

    keys=kwargs.keys()

    if 'ax1_title' in keys:
        ax1_title=kwargs['ax1_title']
    else:
        ax1_title=None

    if 'cbarnorm' in keys:
        cbarnorm = kwargs['cbarnorm']
    else:
        cbarnorm = 2

    if 'coi' in keys:
        coi = kwargs['coi']
    else:
        coi = None

    if 'cs_threshold' in keys:
        cs_threshold = kwargs['cs_threshold']
    else:
        cs_threshold = 0.4

    if 'fname' in keys:
        fname = kwargs['fname']
        mkplot = True
    else:
        fname = None
        mkplot=False

    if 'peak' in keys:
        peak = kwargs['peak']
    else:
        peak = 0

    if 'scales' in keys:
        scales = kwargs['scales']
        if 'sampf' in keys:
            sampf = kwargs['sampf']
            scales=scales/sampf
    else:
        scales = None

    if 'show_plot' in keys:
        show_plot = kwargs['show_plot']
    else:
        show_plot = True

    if 'mask' in keys:
        mask = kwargs['mask']
    else:
        mask = None

    if 'use_valid_gws' in keys:
        use_valid_gws=kwargs['use_valid_gws']
    else:
        use_valid_gws=False

    if 'wdu' in keys:
        wdu=kwargs['wdu']
    else:
        wdu=1

    T=scales*wdu
    time = np.arange(0,c.shape[1])
    gws = metobs.bl.turb.wavelets.get_gws(c,mask=None)

    if mkplot:
        fig = plt.figure(num=1,figsize=(16,12))
        axt = fig.add_subplot(2,4,3)
        fig.clear()

        fig = plt.figure(num=1,figsize=(16,12))
        ax1 = fig.add_subplot(2,1,1)
        ax1.clear()
        t=ax1.get_position()
        t2=axt.get_position()
        ax1pos=t.get_points()
        axtpos=t2.get_points()
        ax1pos[1][0]=axtpos[1][0]*0.95
        t.set_points(ax1pos)
        ax1.set_position(t)
        #ax1.pcolormesh(time,T,c)
        ax1.imshow(c,aspect='auto',origin = 'bottom')
        if coi is not None:
            xs,ys = poly_between(np.arange(0,len(coi)),np.max(coi),coi)
            ax1.fill(xs,ys,'k',alpha=0.2)
            ax1.set_xlim(0,c.shape[1])
            ax1.set_ylim(0,c.shape[0])             

        #ax2 = fig.add_subplot(2,5,5, sharey=ax1)
        ax2 = fig.add_subplot(2,5,5)
        ax2.semilogx(gws,T,'k+')
        if mask is not None:
            ax2.semilogx(metpy.bl.turb.wavelets.get_gws(c,mask=mask),T,'g+')
            ax2.legend(('all','valid'))
        #else:
        #    ax2.legend(('all'),)
        ax2.grid(b=True,color='k',linestyle='-',linewidth=1)
        ax1.set_yticks(ax2.get_yticks()/(wdu))
        ax1.set_yticklabels((ax1.get_yticks()*wdu).astype(int))
        ax1.set_xticklabels((ax1.get_xticks()/sampf).astype(int))
        ax1.set_xlim(0,c.shape[1])
        ax1.set_ylim(0,c.shape[0])
        ax1.set_ylabel('Period (s)',fontsize=fontsize)
        ax2.set_title('Global Wavelet Power Specturm')
        if ax1_title is not None:
            ax1.set_title(ax1_title,fontsize=fontsize)
#        norm = Normalize(-cbarnorm,cbarnorm)
        lpos = ax1.get_position()
        t = lpos.get_points()       
        t[0][0] = t[1][0]
        t[1][0] = 1.02*t[1][0]
        lpos.set_points(t)
        cax = fig.add_axes(lpos)
#        cbar = matplotlib.colorbar.ColorbarBase(ax=cax,
#           norm=norm, orientation='vertical')
        cbar=ax1.figure.colorbar(ax1.images[0],cax=cax)
        cbar.set_label("T'")

        ax3 = fig.add_subplot(2,1,2, sharex=ax1)
        t = ax3.get_position()
        ax3pos=t.get_points()
        ax3pos[1][0]=ax1.get_position().get_points()[1][0]
        t.set_points(ax3pos)
        ax3.set_position(t)
        ax3.plot(ts,'k')

###########################################################
#
# Detect Coherent Structures
#
###########################################################
#   find extreama in global wavelet spectrum
    if use_valid_gws:
        start,stop,gws_extrema =  metpy.vis.wavelets.find_peaks(c,peak,cs_threshold,mask=mask)
    else:
        start,stop,gws_extrema =  metpy.vis.wavelets.find_peaks(c,peak,cs_threshold)
     
    if start == 0:
        highlight_cs=False
    else:
        highlight_cs = True

#   Create and plot bins for shading
    x=[]
    y=[]
    xa=[]
    ya=[]
    if highlight_cs:
        for i in range(0,len(start)):
            x=np.arange(start[i],stop[i])
            y=np.repeat(np.max(np.abs(ts)),len(x))
            xa.append(x)
            ya.append(y)
            if mkplot:
                xs,ys=poly_between(x,-np.max(np.abs(ts)),y)
                ax3.fill(xs,ys,'k',alpha=0.2)
    else:
        xa=None
        ya=None

    if mkplot:
        ax3.set_ylim((np.min(ts),np.max(ts)))
        if highlight_cs:
            ax4=ax3.twinx()
            ax4.plot(c[gws_extrema[peak],:],'r',linewidth=3)
            ax4.legend(('peak scale: '+`np.int(gws_extrema[peak]*wdu)`+' s',))
            ax4.set_xlim((0,c.shape[1]))
            ax4.set_xticklabels((ax4.get_xticks()/sampf).astype(int))
            ax4.set_xlabel('time (s)',fontsize=fontsize)
            ax4.set_ylabel('Wavelet Coefficient\nalong GWPS Peak Scale',fontsize=fontsize)
        else:
            ax3.set_xlim((0,c.shape[1]))
        ax3.grid()
        ax3.set_ylabel('Temperature(detrended) (K)',fontsize=fontsize)

    if (fname is None) and (show_plot):
        plt.show()
        plt.clf()
        plt.close()
    elif (fname is not None):
        plt.savefig(fname,dpi=300)
        if show_plot is True:
            plt.show()
        plt.clf()
        plt.close()

    return(xa,ya),gws_extrema*wdu

def xwavelet_plt(c1,c2,c3,max_period=None,min_time=0,max_time=None,
               cbarnorm=2,fname='',show=True,xlabs=None,ylabs=None,main_title='',
               titles=['','',''],peak=[0,0,0]):

    if max_period == None: max_period = c1.shape[0]
    if max_time == None: max_period = c1.shape[1]

    fig = plt.figure(num=1,figsize=(16,12))
    ax1a = fig.add_subplot(3,2,1)
    ax2a = fig.add_subplot(3,2,3,sharex=ax1a,sharey=ax1a)
    ax3a = fig.add_subplot(3,2,5,sharex=ax1a,sharey=ax1a)
    ax1b = fig.add_subplot(3,2,2,sharey=ax1a)
    ax2b = fig.add_subplot(3,2,4,sharey=ax1a)
    ax3b = fig.add_subplot(3,2,6,sharey=ax1a)

    wavelet(c1,fig,ax1a,max_period=max_period,
            min_time=min_time,max_time=max_time,
            cbarnorm=cbarnorm,xlabs=xlabs[0],ylabs=ylabs[0],title=titles[0])
    wavelet(c2,fig,ax2a,max_period=max_period,
            min_time=min_time,max_time=max_time,
            cbarnorm=cbarnorm,xlabs=xlabs[2],ylabs=ylabs[2],title=titles[1])
    wavelet(c3,fig,ax3a,max_period=max_period,
            min_time=min_time,max_time=max_time,
            cbarnorm=cbarnorm,xlabs=xlabs[4],ylabs=ylabs[4],title=titles[2])

    gws_plot(c1,ax1b,max_period=max_period,min_time=min_time,max_time=max_time,
             xlabs=xlabs[1],ylabs=ylabs[1],peak=peak[0])
    gws_plot(c2,ax2b,max_period=max_period,min_time=min_time,max_time=max_time,
             xlabs=xlabs[3],ylabs=ylabs[3],peak=peak[0])
    gws_plot(c3,ax3b,max_period=max_period,min_time=min_time,max_time=max_time,
             xlabs=xlabs[5],ylabs=ylabs[5],peak=peak[0])
    if show:
        plt.show()

    if fname is not None:
        plt.savefig(fname,dpi=300)

    plt.close(1)

def wavelet(c,fig,ax,max_period=None,min_time=0,max_time=None,
            cbarnorm=2,xlabs=None,ylabs=None,title='',cloc=0.48):

    scales = np.arange(1,max_period+1)
    T = scales*np.pi/np.sqrt(2.) #(eq 22 Collineau and Brunet 1993a)
    time = np.arange(min_time,max_time)

    ax.pcolormesh(time,T[0:max_period],c[0:max_period,min_time:max_time])
    if title != '':
        ax.set_title(title)
    ax.set_xlim(min_time,max_time-1)
    ax.set_ylim(T[0],T[max_period-2])
    if xlabs is not None: ax.set_xlabel(xlabs)
    if ylabs is not None: ax.set_ylabel(ylabs)
    norm = Normalize(-cbarnorm,cbarnorm)
    lpos = ax.get_position()
#    lpos[0] = cloc
#    lpos[2] = 0.01
#    cax = fig.add_axes(lpos)
#    cbar = matplotlib.colorbar.ColorbarBase(ax=cax,
#       norm=norm, orientation='vertical')

def gws_plot(c,ax,max_period=None,min_time=0,max_time=None,
            xlabs=None,ylabs=None,peak=-1):

    scales = np.arange(1,max_period+1)
    T = scales*np.pi/np.sqrt(2.) #(eq 22 Collineau and Brunet 1993a)
    time = np.arange(min_time,max_time-1)

    gws = metobs.bl.turb.wavelets.gws(c[0:max_period-1,min_time:max_time-1])

    ax.semilogx(gws[0:max_period-1],T[0:max_period-1],'+')
    ax.set_ylim(T[0],T[max_period-2])
    ax.grid(b=True,color='k',linestyle='-',linewidth=1)
    if xlabs!=None: ax.set_xlabel(xlabs)
    if ylabs!=None: ax.set_ylabel(ylabs)
    if peak > 0:
        gws_extrema=metpy.generic.find_extrema(gws,mode='all')
        ax.text(gws[gws_extrema[peak]],T[gws_extrema[peak]],'x')
        ax.text(gws[gws_extrema[peak]]+2,T[gws_extrema[peak]],`np.round(T[gws_extrema[peak]],1)`,
            fontsize=16,backgroundcolor='white',bbox=dict(facecolor='white', alpha=1.0))

"""

References

Collineau, S. and Y. brunet, 1993: Detection of turbulent coherent motions in a
  forest canopy part 1: wavelet analysis.  Boundary-Layer Meteorology, 65,
  pp 357-379.

"""
