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

##############################################################################


def showalter_index(pressure, temperature, dewpt): 
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
        
        dewpt (:class: `pint.Quantity`):
            Parcel dew point temperatures for corresponding pressure
        

    Returns
    -------
    `pint.Quantity`
        Showalter Index
        
    """
    # find the measured temperature and dew point temperature at 850 hPa.
    idx850 = np.where(pressure == 850 * units.hPa)
    T850 = temperature[idx850]
    Td850 = dewpt[idx850]
    
    # find the parcel profile temperature at 500 hPa.
    idx500 = np.where(pressure == 500 * units.hPa)
    Tp500 = temperature[idx500]
    
    # Calculate lcl at the 850 hPa level
    lcl = mpcalc.lcl(850 * units.hPa, T850[0], Td850[0])
    lcl = lcl[0]
    
    # Define start and end heights for dry and moist lapse rate calculations
    p_strt = 1000 * units.hPa
    p_end = 500 * units.hPa
    
    # Calculate parcel temp when raised dry adiabatically from surface to lcl

    dl = mpcalc.dry_lapse(lcl, tc[0], p[0])
    dl = (dl.magnitude - 273.15) * units.degC  # Change units to C

    # Calculate parcel temp when raised moist adiabatically from lcl to 500mb
    ml = mpcalc.moist_lapse(p_end, dl, lcl)
    
    # Calculate the Showalter index
    shox = Tp500 - ml
    return shox

showalter = showalter_index(p, tc, tdc) 

print(showalter)


