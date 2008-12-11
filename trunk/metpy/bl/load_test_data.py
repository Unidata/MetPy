#!/usr/bin/python

import metpy
import numpy as N

filename = '/home/lesserwhirls/python_packages/metpy/ts_dat_2008_03_09_1900.dat'

date,data=metpy.getobs.ltm.sonic(filename)

u=data['u']
v=data['v']
w=data['w']
T=data['T']
#ub=metpy.generic.block(data['u'],2)
#vb=metpy.generic.block(data['v'],2)
#wb=metpy.generic.block(data['w'],2)
#Tb=metpy.generic.block(data['T'],2)

#ur,vr,wr=metpy.bl.rotations.two_three_rot(u,v,w)

#ubr,vbr,wbr=metpy.bl.rotations.two_three_rot(ub,vb,wb)

#s=metpy.generic.block_apply(metpy.bl.sim.mos.theta_star,ub,vb,wb,Tb)
