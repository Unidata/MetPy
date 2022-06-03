import metpy.calc as mpcalc
from metpy.units import units
import  xarray as xr
from netCDF4 import Dataset
import pandas as pd

from metpy.cbook import get_test_data
data = Dataset(get_test_data('wrf_example.nc', False))
df = pd.read_fwf(get_test_data('may4_sounding.txt', as_file_obj=False),
                 skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)
lat = data.variables['lat'][:]
temperature = units.Quantity(data.variables['temperature'][:], 'degC')
hgt = units.Quantity(data.variables['height'][:], 'meter')
print(data)
x=[[99, 99,100], [99, 99,100],[99, 99,100],[99, 99,100]]
#temp = xr.DataArray(data=x) * units('degC')
u= xr.DataArray(data=x)* units['m/s']
#h= xr.DataArray(data=x)* units.meter
#lat =xr.DataArray(data=x)* units('degree')

edr=mpcalc.kinematics.eady_growth_rate(temperature,u,hgt,lat)
#print(u)
