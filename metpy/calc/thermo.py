# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division
import numpy as np
import scipy.integrate as si
from .tools import find_intersections
from ..package_tools import Exporter
from ..constants import epsilon, kappa, P0, Rd, Lv, Cp_d
from ..units import atleast_1d, concatenate, units

exporter = Exporter(globals())

sat_pressure_0c = 6.112 * units.millibar


@exporter.export
def potential_temperature(pressure, temperature):
    r'''Calculate the potential temperature.

    Uses the Poisson equation to calculation the potential temperature
    given `pressure` and `temperature`.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The total atmospheric pressure
    temperature : `pint.Quantity`
        The temperature

    Returns
    -------
    `pint.Quantity`
        The potential temperature corresponding to the the temperature and
        pressure.

    See Also
    --------
    dry_lapse

    Notes
    -----
    Formula:

    .. math:: \Theta = T (P_0 / P)^\kappa

    Examples
    --------
    >>> from metpy.units import units
    >>> metpy.calc.potential_temperature(800. * units.mbar, 273. * units.kelvin)
    290.9814150577374

    '''

    return temperature * (P0 / pressure).to('dimensionless')**kappa


@exporter.export
def dry_lapse(pressure, temperature):
    r'''Calculate the temperature at a level assuming only dry processes
    operating from the starting point.

    This function lifts a parcel starting at `temperature`, conserving
    potential temperature. The starting pressure should be the first item in
    the `pressure` array.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure level(s) of interest
    temperature : `pint.Quantity`
        The starting temperature

    Returns
    -------
    `pint.Quantity`
       The resulting parcel temperature at levels given by `pressure`

    See Also
    --------
    moist_lapse : Calculate parcel temperature assuming liquid saturation
                  processes
    parcel_profile : Calculate complete parcel profile
    potential_temperature
    '''

    return temperature * (pressure / pressure[0])**kappa


@exporter.export
def moist_lapse(pressure, temperature):
    r'''
    Calculate the temperature at a level assuming liquid saturation processes
    operating from the starting point.

    This function lifts a parcel starting at `temperature`. The starting
    pressure should be the first item in the `pressure` array. Essentially,
    this function is calculating moist pseudo-adiabats.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure level(s) of interest
    temperature : `pint.Quantity`
        The starting temperature

    Returns
    -------
    `pint.Quantity`
       The temperature corresponding to the the starting temperature and
       pressure levels.

    See Also
    --------
    dry_lapse : Calculate parcel temperature assuming dry adiabatic processes
    parcel_profile : Calculate complete parcel profile

    Notes
    -----
    This function is implemented by integrating the following differential
    equation:

    .. math:: \frac{dT}{dP} = \frac{1}{P} \frac{R_d T + L_v r_s}
                                {C_{pd} + \frac{L_v^2 r_s \epsilon}{R_d T^2}}

    This equation comes from [1]_.

    References
    ----------

    .. [1] Bakhshaii, A. and R. Stull, 2013: Saturated Pseudoadiabats--A
           Noniterative Approximation. J. Appl. Meteor. Clim., 52, 5-15.
    '''

    def dt(t, p):
        t = units.Quantity(t, temperature.units)
        p = units.Quantity(p, pressure.units)
        rs = saturation_mixing_ratio(p, t)
        frac = ((Rd * t + Lv * rs) /
                (Cp_d + (Lv * Lv * rs * epsilon / (Rd * t * t)))).to('kelvin')
        return frac / p
    return units.Quantity(si.odeint(dt, atleast_1d(temperature).squeeze(),
                                    pressure.squeeze()).T.squeeze(), temperature.units)


@exporter.export
def lcl(pressure, temperature, dewpt, max_iters=50, eps=1e-2):
    r'''Calculate the lifted condensation level (LCL) using from the starting
    point.

    The starting state for the parcel is defined by `temperature`, `dewpt`,
    and `pressure`.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The starting atmospheric pressure
    temperature : `pint.Quantity`
        The starting temperature
    dewpt : `pint.Quantity`
        The starting dew point

    Returns
    -------
    `pint.Quantity`
        The LCL

    Other Parameters
    ----------------
    max_iters : int, optional
        The maximum number of iterations to use in calculation, defaults to 50.
    eps : float, optional
        The desired absolute error in the calculated value, defaults to 1e-2.

    See Also
    --------
    parcel_profile

    Notes
    -----
    This function is implemented using an iterative approach to solve for the
    LCL. The basic algorithm is:
    1. Find the dew point from the LCL pressure and starting mixing ratio
    2. Find the LCL pressure from the starting temperature and dewpoint
    3. Iterate until convergence

    The function is guaranteed to finish by virtue of the `maxIters` counter.
    '''

    w = mixing_ratio(saturation_vapor_pressure(dewpt), pressure)
    new_p = p = pressure
    eps = units.Quantity(eps, p.units)
    while max_iters:
        td = dewpoint(vapor_pressure(p, w))
        new_p = pressure * (td / temperature) ** (1. / kappa)
        if np.abs(new_p - p).max() < eps:
            break
        p = new_p
        max_iters -= 1
    return new_p


@exporter.export
def lfc(pressure, temperature, dewpt):
    r'''Calculate the level of free convection (LFC) by finding the first
    intersection of the ideal parcel path and the measured parcel
    temperature.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The starting atmospheric pressure
    temperature : `pint.Quantity`
        The starting temperature
    dewpt : `pint.Quantity`
        The starting dew point

    Returns
    -------
    `pint.Quantity`
        The LFC

    See Also
    --------
    parcel_profile
    '''

    ideal_profile = parcel_profile(pressure, temperature[0], dewpt[0]).to('degC')
    intersections = find_intersections(pressure, ideal_profile, temperature)
    return intersections[0]


@exporter.export
def parcel_profile(pressure, temperature, dewpt):
    r'''Calculate the profile a parcel takes through the atmosphere, lifting
    from the starting point.

    The parcel starts at `temperature`, and `dewpt`, lifted up
    dry adiabatically to the LCL, and then moist adiabatically from there.
    `pressure` specifies the pressure levels for the profile.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure level(s) of interest. The first entry should be the starting
        point pressure.
    temperature : `pint.Quantity`
        The starting temperature
    dewpt : `pint.Quantity`
        The starting dew point

    Returns
    -------
    `pint.Quantity`
        The parcel temperatures at the specified pressure levels.

    See Also
    --------
    lcl, moist_lapse, dry_lapse
    '''

    # Find the LCL
    l = lcl(pressure[0], temperature, dewpt).to(pressure.units)

    # Find the dry adiabatic profile, *including* the LCL. We need >= the LCL in case the
    # LCL is included in the levels. It's slightly redundant in that case, but simplifies
    # the logic for removing it later.
    press_lower = concatenate((pressure[pressure >= l], l))
    t1 = dry_lapse(press_lower, temperature)

    # Find moist pseudo-adiabatic profile starting at the LCL
    press_upper = concatenate((l, pressure[pressure < l]))
    t2 = moist_lapse(press_upper, t1[-1]).to(t1.units)

    # Return LCL *without* the LCL point
    return concatenate((t1[:-1], t2[1:]))


@exporter.export
def vapor_pressure(pressure, mixing):
    r'''Calculate water vapor (partial) pressure

    Given total `pressure` and water vapor `mixing` ratio, calculates the
    partial pressure of water vapor.

    Parameters
    ----------
    pressure : `pint.Quantity`
        total atmospheric pressure
    mixing : `pint.Quantity`
        dimensionless mass mixing ratio

    Returns
    -------
    `pint.Quantity`
        The ambient water vapor (partial) pressure in the same units as
        `pressure`.

    See Also
    --------
    saturation_vapor_pressure, dewpoint
    '''

    return pressure * mixing / (epsilon + mixing)


@exporter.export
def saturation_vapor_pressure(temperature):
    r'''Calculate the saturation water vapor (partial) pressure

    Parameters
    ----------
    temperature : `pint.Quantity`
        The temperature

    Returns
    -------
    `pint.Quantity`
        The saturation water vapor (partial) pressure

    See Also
    --------
    vapor_pressure, dewpoint

    Notes
    -----
    Instead of temperature, dewpoint may be used in order to calculate
    the actual (ambient) water vapor (partial) pressure.

    The formula used is that from Bolton 1980 [2] for T in degrees Celsius:

    .. math:: 6.112 e^\frac{17.67T}{T + 243.5}

    References
    ----------
    .. [2] Bolton, D., 1980: The Computation of Equivalent Potential
           Temperature. Mon. Wea. Rev., 108, 1046-1053.
    '''

    # Converted from original in terms of C to use kelvin. Using raw absolute values of C in
    # a formula plays havoc with units support.
    return sat_pressure_0c * np.exp(17.67 * (temperature - 273.15 * units.kelvin) /
                                    (temperature - 29.65 * units.kelvin))


@exporter.export
def dewpoint_rh(temperature, rh):
    r'''Calculate the ambient dewpoint given air temperature and relative
    humidity.

    Parameters
    ----------
    temperature : `pint.Quantity`
        Air temperature
    rh : `pint.Quantity`
        Relative humidity expressed as a ratio in the range [0, 1]

    Returns
    -------
    `pint.Quantity`
        The dew point temperature

    See Also
    --------
    dewpoint, saturation_vapor_pressure
    '''

    return dewpoint(rh * saturation_vapor_pressure(temperature))


@exporter.export
def dewpoint(e):
    r'''Calculate the ambient dewpoint given the vapor pressure.

    Parameters
    ----------
    e : `pint.Quantity`
        Water vapor partial pressure

    Returns
    -------
    `pint.Quantity`
        Dew point temperature

    See Also
    --------
    dewpoint_rh, saturation_vapor_pressure, vapor_pressure

    Notes
    -----
    This function inverts the Bolton 1980 [3] formula for saturation vapor
    pressure to instead calculate the temperature. This yield the following
    formula for dewpoint in degrees Celsius:

    .. math:: T = \frac{243.5 log(e / 6.112)}{17.67 - log(e / 6.112)}

    References
    ----------
    .. [3] Bolton, D., 1980: The Computation of Equivalent Potential
           Temperature. Mon. Wea. Rev., 108, 1046-1053.
    '''

    val = np.log(e / sat_pressure_0c)
    return 0. * units.degC + 243.5 * units.delta_degC * val / (17.67 - val)


@exporter.export
def mixing_ratio(part_press, tot_press):
    r'''Calculates the mixing ratio of gas given its partial pressure
    and the total pressure of the air.

    There are no required units for the input arrays, other than that
    they have the same units.

    Parameters
    ----------
    part_press : `pint.Quantity`
        Partial pressure of the constituent gas
    tot_press : `pint.Quantity`
        Total air pressure

    Returns
    -------
    `pint.Quantity`
        The (mass) mixing ratio, dimensionless (e.g. Kg/Kg or g/g)

    See Also
    --------
    vapor_pressure
    '''

    return epsilon * part_press / (tot_press - part_press)


@exporter.export
def saturation_mixing_ratio(tot_press, temperature):
    r'''Calculates the saturation mixing ratio given total pressure
    and the temperature.

    The implementation uses the formula outlined in [4]

    Parameters
    ----------
    tot_press: `pint.Quantity`
        Total atmospheric pressure
    temperature: `pint.Quantity`
        The temperature

    Returns
    -------
    `pint.Quantity`
        The saturation mixing ratio, dimensionless

    References
    ----------
    .. [4] Hobbs, Peter V. and Wallace, John M., 1977: Atmospheric Science, an Introductory
            Survey. 73.
    '''

    return mixing_ratio(saturation_vapor_pressure(temperature), tot_press)


@exporter.export
def equivalent_potential_temperature(pressure, temperature):
    r'''Calculates equivalent potential temperature given an air parcel's
    pressure and temperature.

    The implementation uses the formula outlined in [5]

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure
    temperature: `pint.Quantity`
        The temperature

    Returns
    -------
    `pint.Quantity`
        The corresponding equivalent potential temperature of the parcel

    Notes
    -----
    .. math:: \Theta_e = \Theta e^\frac{L_v r_s}{C_{pd} T}

    References
    ----------
    .. [5] Hobbs, Peter V. and Wallace, John M., 1977: Atmospheric Science, an Introductory
            Survey. 78-79.
    '''

    pottemp = potential_temperature(pressure, temperature)
    smixr = saturation_mixing_ratio(pressure, temperature)
    return pottemp * np.exp(Lv * smixr / (Cp_d * temperature))
