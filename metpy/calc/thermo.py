# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division
import numpy as np
import scipy.integrate as si
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
    pressure : array_like
        The total atmospheric pressure
    temperature : array_like
        The temperature

    Returns
    -------
    array_like
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

    return temperature * (P0 / pressure)**kappa


@exporter.export
def dry_lapse(pressure, temperature):
    r'''Calculate the temperature at a level assuming only dry processes
    operating from the starting point.

    This function lifts a parcel starting at `temperature`, conserving
    potential temperature. The starting pressure should be the first item in
    the `pressure` array.

    Parameters
    ----------
    pressure : array_like
        The atmospheric pressure level(s) of interest
    temperature : array_like
        The starting temperature

    Returns
    -------
    array_like
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
    pressure : array_like
        The atmospheric pressure level(s) of interest
    temperature : array_like
        The starting temperature

    Returns
    -------
    array_like
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
        rs = mixing_ratio(saturation_vapor_pressure(t), p)
        return (1. / p) * ((Rd * t + Lv * rs) /
                           (Cp_d + (Lv * Lv * rs * epsilon / (Rd * t * t))))
    return units.Quantity(si.odeint(dt, atleast_1d(temperature).squeeze(),
                                    pressure.squeeze()).T.squeeze(), temperature.units)


@exporter.export
def lcl(pressure, temperature, dewpt, max_iters=50, eps=1e-2):
    r'''Calculate the lifted condensation level (LCL) using from the starting
    point.

    The starting state for the parcel is defined by `temperature`, `dewpoint`,
    and `pressure`.

    Parameters
    ----------
    pressure : array_like
        The starting atmospheric pressure
    temperature : array_like
        The starting temperature
    dewpt : array_like
        The starting dew point

    Returns
    -------
    array_like
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
    p = pressure
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
def parcel_profile(pressure, temperature, dewpt):
    r'''Calculate the profile a parcel takes through the atmosphere, lifting
    from the starting point.

    The parcel starts at `temperature`, and `dewpt`, lifted up
    dry adiabatically to the LCL, and then moist adiabatically from there.
    `pressure` specifies the pressure levels for the profile.

    Parameters
    ----------
    pressure : array_like
        The atmospheric pressure level(s) of interest. The first entry should be the starting
        point pressure.
    temperature : array_like
        The starting temperature
    dewpt : array_like
        The starting dew point

    Returns
    -------
    array_like
        The parcel temperatures at the specified pressure levels.

    See Also
    --------
    lcl, moist_lapse, dry_lapse
    '''

    # Find the LCL
    l = lcl(pressure[0], temperature, dewpt).to(pressure.units)

    # Find the dry adiabatic profile, *including* the LCL
    press_lower = concatenate((pressure[pressure > l], l))
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
    pressure : array_like
        total atmospheric pressure
    mixing : array_like
        dimensionless mass mixing ratio

    Returns
    -------
    array_like
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
    temperature : array_like
        The temperature

    Returns
    -------
    array_like
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
    temperature : array_like
        Air temperature
    rh : array_like
        Relative humidity expressed as a ratio in the range [0, 1]

    Returns
    -------
    array_like
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
    e : array_like
        Water vapor partial pressure

    Returns
    -------
    array_like
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
    part_press : array_like
        Partial pressure of the constituent gas
    tot_press : array_like
        Total air pressure

    Returns
    -------
    array_like
        The (mass) mixing ratio, dimensionless (e.g. Kg/Kg or g/g)

    See Also
    --------
    vapor_pressure
    '''

    return epsilon * part_press / (tot_press - part_press)
