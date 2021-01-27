##############################################################################
# Import packages:
import numpy as np
import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc

import geocat.datafiles as gdf

###############################################################################
# Read in data from GeoCAT-datafiles (for validation of code)

# Open dataset
da = pd.read_csv(gdf.get('ascii_files/sounding.testdata'),
                 delimiter='\\s+',
                 header=None)

# Extract the data
p = da[1].values * units.hPa  # Pressure [mb/hPa]
tc = (da[5].values + 2) * units.degC  # Temperature [C]
tdc = (da[9].values + 2) * units.degC  # Dew pt temp  [C]

ta = mpcalc.parcel_profile(p, tc[0], tdc[0])  # Parcel profile
tac = (ta.magnitude - 273.15) * units.degC  # Parcel temp in C


# Specify temperature and dewpt at 850mb and temp at 500mb
t850 = da.loc[da[1] == 850]
t850 = (t850[5].values + 2) * units.degC

dpt850 = da.loc[da[1] == 850]
dpt850 = (dpt850[9].values + 2) * units.degC

t500 = da.loc[da[1] == 500]
t500 = (t500[5].values + 2) * units.degC

# Calculate lcl at the 850 hPa level
lcl = mpcalc.lcl(850 * units.hPa, t850[0], dpt850[0])
lcl = lcl[0]

##############################################################################


def showalter_index(pressure, temperature, lcl850, t500): 
    """Calculate Showalter Index from pressure temperature and 850 hPa lcl
    
    Showalter Index derived from [Galway1956]_:
    SI = T500 - Tp500
    
    where:
    T500 is the measured temperature at 500 hPa
    Tp500 is the temperature of the lifted parcel at 500 hPa
    
   Parameters
   ----------
        
        pressure : `pint.Quantity`
            Atmospheric pressure level(s) of interest, in order from highest to
        lowest pressure
        
        temperature : `pint.Quantity`
            Parcel temperature for corresponding pressure 
        
        lcl850 (:class: `pint.Quantity`):
            Calculated lcl taken at 850 hPa
        
        t500 (:class: ``pint.Quantity`):
            Environmental temperature at 500 hPa 
            
    Returns
    -------
    `pint.Quantity`
        Showalter Index
        
    """
    
    # Define start and end heights for dry and moist lapse rate calculations
    p_strt = 1000 * units.hPa
    p_end = 500 * units.hPa
    
    # Calculate parcel temp when raised dry adiabatically from surface to lcl

    dl = mpcalc.dry_lapse(lcl850, tc[0], p[0])
    dl = (dl.magnitude - 273.15) * units.degC  # Change units to C

    
    # Calculate parcel temp when raised moist adiabatically from lcl to 500mb
    ml = mpcalc.moist_lapse(p_end, dl, lcl)
    
    # Calculate the Showalter index
    shox = t500 - ml
    return shox

# Uncomment to validate the code will 'catch' if an lcl is lower than the 850mb height
showalter = showalter_index(p, tc, lcl, t500) 

print(showalter)


