import datetime
import matplotlib.pyplot as plt
import scipy.constants as sconsts
from metpy.readers.mesonet import (read_mesonet_data, get_last_time,
    mesonet_units)
from metpy.vis import meteogram
from metpy.constants import C2F
from metpy.calc import dewpoint, windchill
from metpy.cbook import rec_append_fields

fields = ('stid', 'time', 'relh', 'tair', 'wspd', 'wmax', 'wdir', 'pres',
    'srad', 'rain')

data = read_mesonet_data('data/20080117nrmn.mts', fields, lookup_stids=False)

#Add a reasonable time range if we're doing current data
end = get_last_time(data)
times = (end - datetime.timedelta(hours=24), end)

#Calculate dewpoint in F from relative humidity and temperature
dewpt = C2F(dewpoint(data['TAIR'], data['RELH']/100.))
data = rec_append_fields(data, ('dewpoint',), (dewpt,))

#Convert temperature and dewpoint to Farenheit
mod_units = mesonet_units.copy()
mod_units['TAIR'] = 'F'
mod_units['dewpoint'] = 'F'
data['TAIR'] = C2F(data['TAIR'])

#Convert wind speeds to MPH
data['WSPD'] *= sconsts.hour / sconsts.mile
data['WMAX'] *= sconsts.hour / sconsts.mile
mod_units['WSPD'] = 'MPH'
mod_units['WMAX'] = 'MPH'

#Convert rainfall to inches
data['RAIN'] *= sconsts.milli / sconsts.inch
mod_units['RAIN'] = 'in.'

#Calculate windchill
wchill = windchill(data['TAIR'], data['WSPD'], metric=False)
data = rec_append_fields(data, ('windchill',), (wchill,))

fig = plt.figure(figsize=(8,10))
layout = {0:['temperature', 'dewpoint', 'windchill']}
axs = meteogram(data, fig, num_panels=5, units=mod_units, time_range=times,
    layout=layout)
axs[0].set_ylabel('Temperature (F)')
plt.show()
