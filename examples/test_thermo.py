# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 00:09:51 2016

@author: Usuario
"""

import metpy.calc as mcalc
from metpy.units import units

P = 1010.50 * units.mbar
print ' Pressure = ',P

Tdb = 23.9 * units.degC
print ' Temperature = ',Tdb

Twb = 20.8 * units.degC
print ' Wet temperature = ',Twb

rh =  mcalc.relative_humidity_psychrometric(Tdb, Twb,P)

print ' Relative humidity = ',rh
