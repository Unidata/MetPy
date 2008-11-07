#!/usr/bin/python

import metobs
import numpy as N

filename = '/home/lesserwhirls/python_packages/metobs/ts_dat_2008_03_09_1900.dat'

date,data=metobs.getobs.ltm.sonic(filename)

u=data['u']
v=data['v']
w=data['w']
T=data['T']
#ub=metobs.generic.block(data['u'],2)
#vb=metobs.generic.block(data['v'],2)
#wb=metobs.generic.block(data['w'],2)
#Tb=metobs.generic.block(data['T'],2)

#ur,vr,wr=metobs.bl.rotations.two_three_rot(u,v,w)

#ubr,vbr,wbr=metobs.bl.rotations.two_three_rot(ub,vb,wb)

#s=metobs.generic.block_apply(metobs.bl.sim.mos.theta_star,ub,vb,wb,Tb)
