import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import SkewT
from metpy.units import units

fig = plt.figure(figsize=(4, 4), dpi=150)
skew = SkewT(fig, rotation=45)

col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']

df = pd.read_fwf(get_test_data('may4_sounding.txt', as_file_obj=False),
                 skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)

# Drop any rows with all NaN values for T, Td, winds
df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='all'
               ).reset_index(drop=True)

p = df['pressure'].values * units.hPa
t = df['temperature'].values * units.degC
td = df['dewpoint'].values * units.degC
wind_speed = df['speed'].values * units.knots
wind_dir = df['direction'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)
p_prof, t, td, prof = mpcalc.parcel_profile_with_lcl(p, t, td)

skew.plot(p_prof, t, 'r')
skew.plot(p_prof, td, 'g')
skew.plot(p_prof, prof, 'k')  # Plot parcel profile
skew.plot_barbs(p, u, v, y_clip_radius=0.11)

skew.ax.set_title('Sounding for 00Z 4 May 1999')
skew.ax.set_xlim(-10, 35)
skew.ax.set_xlabel(f'Temperature ({t.units:~})')
skew.ax.set_ylim(1000, 250)
skew.ax.set_ylabel(f'Pressure ({p_prof.units:~})')

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.shade_cin(p_prof, t, prof)
skew.shade_cape(p_prof, t, prof)

plt.show()