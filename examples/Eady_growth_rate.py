import metpy.calc as mpcalc
from metpy.units import units
import  xarray as xr
from netCDF4 import Dataset
import pandas as pd

from metpy.cbook import get_test_data
data = xr.open_dataset(get_test_data('narr_example.nc', False))
data = data.metpy.parse_cf().squeeze()
edr=mpcalc.kinematics.eady_growth_rate(data['Temperature'],data['u_wind'],data['Geopotential_height'],data['lat'])