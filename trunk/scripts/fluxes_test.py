#!/usr/bin/python

from metpy.readers import ltm

filename = '/micronet/python/data/2005-03-12-2100-ts.txt'

data,ext=ltm.tower.read_data.sonic(filename)

rs=ltm.tower.fluxes.rs(data['u'],data['v'],data['w'])
