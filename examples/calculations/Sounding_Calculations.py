# Copyright (c) 2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
=============================
Sounding Calculation Examples
=============================

Use functions from `metpy.calc` to perform a number of calculations using sounding data.

The code below uses example data to perform many sounding calculations for a severe weather
event on May 22, 2011 from the Norman, OK sounding.
"""
import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.units import units

###########################################
# Effective Shear Algorithm for use in Supercell Composite Calculation


def effective_layer(p, t, td, h, height_layer=False):
    """A function that determines the effective inflow layer for a convective sounding.

    Uses the default values of Thompason et al. (2004) for CAPE (100 J/kg) and CIN (-250 J/kg).

    Input:
      - p: sounding pressure with units
      - T: sounding temperature with units
      - Td: sounding dewpoint temperature with units
      - h: sounding heights with units

    Returns:
      - pbot/hbot, ptop/htop: pressure/height of the bottom level,
                              pressure/height of the top level
    """
    from metpy.calc import cape_cin, parcel_profile
    from metpy.units import units

    pbot = None

    for i in range(p.shape[0]):
        prof = parcel_profile(p[i:], t[i], td[i])
        sbcape, sbcin = cape_cin(p[i:], t[i:], td[i:], prof)
        if sbcape >= 100 * units('J/kg') and sbcin > -250 * units('J/kg'):
            pbot = p[i]
            hbot = h[i]
            bot_idx = i
            break
    if not pbot:
        return None, None

    for i in range(bot_idx + 1, p.shape[0]):
        prof = parcel_profile(p[i:], t[i], td[i])
        sbcape, sbcin = cape_cin(p[i:], t[i:], td[i:], prof)
        if sbcape < 100 * units('J/kg') or sbcin < -250 * units('J/kg'):
            ptop = p[i]
            htop = h[i]
            break

    if height_layer:
        return hbot, htop
    else:
        return pbot, ptop


###########################################
# Upper air data can be obtained using the siphon package, but for this example we will use
# some of MetPy's sample data.
col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']

df = pd.read_fwf(get_test_data('20110522_OUN_12Z.txt', as_file_obj=False),
                 skiprows=7, usecols=[0, 1, 2, 3, 6, 7], names=col_names)

# Drop any rows with all NaN values for T, Td, winds
df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'
                       ), how='all').reset_index(drop=True)

###########################################
# Isolate needed variables from our data file and attach units
p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
wdir = df['direction'].values * units.degree
sped = df['speed'].values * units.knot
height = df['height'].values * units.meter

###########################################
# Compute the wind components
u, v = mpcalc.wind_components(sped, wdir)

###########################################
# Compute common sounding index parameters
ctotals = mpcalc.cross_totals(p, T, Td)
kindex = mpcalc.k_index(p, T, Td)
showalter = mpcalc.showalter_index(p, T, Td)
total_totals = mpcalc.total_totals_index(p, T, Td)
vert_totals = mpcalc.vertical_totals(p, T)

###########################################
# Compture the parcel profile for a surface-based parcel
prof = mpcalc.parcel_profile(p, T[0], Td[0])

###########################################
# Compute the corresponding LI, CAPE, CIN values for a surface parcel
lift_index = mpcalc.lifted_index(p, T, prof)
cape, cin = mpcalc.cape_cin(p, T, Td, prof)

###########################################
# Determine the LCL, LFC, and EL for our surface parcel
lclp, lclt = mpcalc.lcl(p[0], T[0], Td[0])
lfcp, _ = mpcalc.lfc(p, T, Td)
el_pressure, _ = mpcalc.el(p, T, Td, prof)

###########################################
# Compute the characteristics of a mean layer parcel (50-hPa depth)
ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=50 * units.hPa)
ml_p, _, _ = mpcalc.mixed_parcel(p, T, Td, depth=50 * units.hPa)
mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, prof, depth=50 * units.hPa)

###########################################
# Compute the characteristics of the most unstable parcel (50-hPa depth)
mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, T, Td, depth=50 * units.hPa)
mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=50 * units.hPa)

###########################################
# Compute the Bunkers Storm Motion vector and use to calculate the critical angle
(u_storm, v_storm), *_ = mpcalc.bunkers_storm_motion(p, u, v, height)
critical_angle = mpcalc.critical_angle(p, u, v, height, u_storm, v_storm)

###########################################
# Work on the calculations needed to compute the significant tornado parameter

# Estimate height of LCL in meters from hydrostatic thickness
new_p = np.append(p[p > lclp], lclp)
new_t = np.append(T[p > lclp], lclt)
lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

# Compute Surface-based CAPE
sbcape, _ = mpcalc.surface_based_cape_cin(p, T, Td)

# Compute SRH, given a motion vector toward the NE at 9.9 m/s
*_, total_helicity = mpcalc.storm_relative_helicity(height, u, v, depth=1 * units.km,
                                                    storm_u=u_storm, storm_v=v_storm)

# Copmute Bulk Shear components and then magnitude
ubshr, vbshr = mpcalc.bulk_shear(p, u, v, height=height, depth=6 * units.km)
bshear = mpcalc.wind_speed(ubshr, vbshr)

# Use all computed pieces to calculate the Significant Tornado parameter
sig_tor = mpcalc.significant_tornado(sbcape, lcl_height,
                                     total_helicity, bshear).to_base_units()

###########################################
# Compute the supercell composite parameter, if possible

# Determine the top and bottom of the effective layer using our own function
hbot, htop = effective_layer(p, T, Td, height, height_layer=True)

# Perform the calculation of supercell composite if an effective layer exists
if hbot:
    esrh = mpcalc.storm_relative_helicity(height, u, v, depth=htop - hbot, bottom=hbot)
    eubshr, evbshr = mpcalc.bulk_shear(p, u, v, height=height, depth=htop - hbot, bottom=hbot)
    ebshear = mpcalc.wind_speed(eubshr, evbshr)

    super_comp = mpcalc.supercell_composite(mucape, esrh[0], ebshear)
else:
    super_comp = np.nan

###########################################
# Print Important Sounding Parameters
print('Important Sounding Parameters for KOUN on 22 Mary 2011 12 UTC')
print()
print(f'        CAPE: {cape:.2f}')
print(f'         CIN: {cin:.2f}')
print(f'LCL Pressure: {lclp:.2f}')
print(f'LFC Pressure: {lfcp:.2f}')
print(f' EL Pressure: {el_pressure:.2f}')
print()
print(f'   Lifted Index: {lift_index:.2f}')
print(f'        K-Index: {kindex:.2f}')
print(f'Showalter Index: {showalter:.2f}')
print(f'   Cross Totals: {ctotals:.2f}')
print(f'   Total Totals: {total_totals:.2f}')
print(f'Vertical Totals: {vert_totals:.2f}')
print()
print('Mixed Layer - Lowest 50-hPa')
print(f'     ML Temp: {ml_t:.2f}')
print(f'     ML Dewp: {ml_td:.2f}')
print(f'     ML CAPE: {mlcape:.2f}')
print(f'      ML CIN: {mlcin:.2f}')
print()
print('Most Unstable - Lowest 50-hPa')
print(f'     MU Temp: {mu_t:.2f}')
print(f'     MU Dewp: {mu_td:.2f}')
print(f' MU Pressure: {mu_p:.2f}')
print(f'     MU CAPE: {mucape:.2f}')
print(f'      MU CIN: {mucin:.2f}')
print()
print('Bunkers Storm Motion Vector')
print(f'  u_storm: {u_storm:.2f}')
print(f'  v_storm: {v_storm:.2f}')
print(f'Critical Angle: {critical_angle:.2f}')
print()
print(f'Storm Relative Helicity: {total_helicity:.2f}')
print(f'Significant Tornado Parameter: {sig_tor:.2f}')
print(f'Supercell Composite Parameter: {super_comp:.2f}')
