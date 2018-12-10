# Copyright (c) 2008,2015,2016,2017,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains a collection of thermodynamic calculations."""

from __future__ import division

import warnings

import numpy as np
import scipy.integrate as si
import scipy.optimize as so

from .tools import (_greater_or_close, _less_or_close, find_bounding_indices,
                    find_intersections, first_derivative, get_layer)
from .. import constants as mpconsts
from ..cbook import broadcast_indices
from ..interpolate.one_dimension import interpolate_1d
from ..package_tools import Exporter
from ..units import atleast_1d, check_units, concatenate, units
from ..xarray import preprocess_xarray

exporter = Exporter(globals())

sat_pressure_0c = 6.112 * units.millibar


@exporter.export
@preprocess_xarray
@check_units('[temperature]', '[temperature]')
def relative_humidity_from_dewpoint(temperature, dewpt):
    r"""Calculate the relative humidity.

    Uses temperature and dewpoint in celsius to calculate relative
    humidity using the ratio of vapor pressure to saturation vapor pressures.

    Parameters
    ----------
    temperature : `pint.Quantity`
        The temperature
    dew point : `pint.Quantity`
        The dew point temperature

    Returns
    -------
    `pint.Quantity`
        The relative humidity

    See Also
    --------
    saturation_vapor_pressure

    """
    e = saturation_vapor_pressure(dewpt)
    e_s = saturation_vapor_pressure(temperature)
    return (e / e_s)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[pressure]')
def exner_function(pressure, reference_pressure=mpconsts.P0):
    r"""Calculate the Exner function.

    .. math:: \Pi = \left( \frac{p}{p_0} \right)^\kappa

    This can be used to calculate potential temperature from temperature (and visa-versa),
    since

    .. math:: \Pi = \frac{T}{\theta}

    Parameters
    ----------
    pressure : `pint.Quantity`
        The total atmospheric pressure
    reference_pressure : `pint.Quantity`, optional
        The reference pressure against which to calculate the Exner function, defaults to P0

    Returns
    -------
    `pint.Quantity`
        The value of the Exner function at the given pressure

    See Also
    --------
    potential_temperature
    temperature_from_potential_temperature

    """
    return (pressure / reference_pressure).to('dimensionless')**mpconsts.kappa


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]')
def potential_temperature(pressure, temperature):
    r"""Calculate the potential temperature.

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
        The potential temperature corresponding to the temperature and
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
    <Quantity(290.96653180346203, 'kelvin')>

    """
    return temperature / exner_function(pressure)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]')
def temperature_from_potential_temperature(pressure, theta):
    r"""Calculate the temperature from a given potential temperature.

    Uses the inverse of the Poisson equation to calculate the temperature from a
    given potential temperature at a specific pressure level.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The total atmospheric pressure
    theta : `pint.Quantity`
        The potential temperature

    Returns
    -------
    `pint.Quantity`
        The temperature corresponding to the potential temperature and pressure.

    See Also
    --------
    dry_lapse
    potential_temperature

    Notes
    -----
    Formula:

    .. math:: T = \Theta (P / P_0)^\kappa

    Examples
    --------
    >>> from metpy.units import units
    >>> from metpy.calc import temperature_from_potential_temperature
    >>> # potential temperature
    >>> theta = np.array([ 286.12859679, 288.22362587]) * units.kelvin
    >>> p = 850 * units.mbar
    >>> T = temperature_from_potential_temperature(p,theta)

    """
    return theta * exner_function(pressure)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[pressure]')
def dry_lapse(pressure, temperature, ref_pressure=None):
    r"""Calculate the temperature at a level assuming only dry processes.

    This function lifts a parcel starting at `temperature`, conserving
    potential temperature. The starting pressure can be given by `ref_pressure`.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure level(s) of interest
    temperature : `pint.Quantity`
        The starting temperature
    ref_pressure : `pint.Quantity`, optional
        The reference pressure. If not given, it defaults to the first element of the
        pressure array.

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

    """
    if ref_pressure is None:
        ref_pressure = pressure[0]
    return temperature * (pressure / ref_pressure)**mpconsts.kappa


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[pressure]')
def moist_lapse(pressure, temperature, ref_pressure=None):
    r"""Calculate the temperature at a level assuming liquid saturation processes.

    This function lifts a parcel starting at `temperature`. The starting pressure can
    be given by `ref_pressure`. Essentially, this function is calculating moist
    pseudo-adiabats.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure level(s) of interest
    temperature : `pint.Quantity`
        The starting temperature
    ref_pressure : `pint.Quantity`, optional
        The reference pressure. If not given, it defaults to the first element of the
        pressure array.

    Returns
    -------
    `pint.Quantity`
       The temperature corresponding to the starting temperature and
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

    This equation comes from [Bakhshaii2013]_.

    """
    def dt(t, p):
        t = units.Quantity(t, temperature.units)
        p = units.Quantity(p, pressure.units)
        rs = saturation_mixing_ratio(p, t)
        frac = ((mpconsts.Rd * t + mpconsts.Lv * rs)
                / (mpconsts.Cp_d + (mpconsts.Lv * mpconsts.Lv * rs * mpconsts.epsilon
                                    / (mpconsts.Rd * t * t)))).to('kelvin')
        return frac / p

    if ref_pressure is None:
        ref_pressure = pressure[0]

    pressure = pressure.to('mbar')
    ref_pressure = ref_pressure.to('mbar')
    temperature = atleast_1d(temperature)

    side = 'left'

    pres_decreasing = (pressure[0] > pressure[-1])
    if pres_decreasing:
        # Everything is easier if pressures are in increasing order
        pressure = pressure[::-1]
        side = 'right'

    ref_pres_idx = np.searchsorted(pressure.m, ref_pressure.m, side=side)

    ret_temperatures = np.empty((0, temperature.shape[0]))

    if ref_pressure > pressure.min():
        # Integrate downward in pressure
        pres_down = np.append(ref_pressure, pressure[(ref_pres_idx - 1)::-1])
        trace_down = si.odeint(dt, temperature.squeeze(), pres_down.squeeze())
        ret_temperatures = np.concatenate((ret_temperatures, trace_down[:0:-1]))

    if ref_pressure < pressure.max():
        # Integrate upward in pressure
        pres_up = np.append(ref_pressure, pressure[ref_pres_idx:])
        trace_up = si.odeint(dt, temperature.squeeze(), pres_up.squeeze())
        ret_temperatures = np.concatenate((ret_temperatures, trace_up[1:]))

    if pres_decreasing:
        ret_temperatures = ret_temperatures[::-1]

    return units.Quantity(ret_temperatures.T.squeeze(), temperature.units)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]')
def lcl(pressure, temperature, dewpt, max_iters=50, eps=1e-5):
    r"""Calculate the lifted condensation level (LCL) using from the starting point.

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
    `(pint.Quantity, pint.Quantity)`
        The LCL pressure and temperature

    Other Parameters
    ----------------
    max_iters : int, optional
        The maximum number of iterations to use in calculation, defaults to 50.
    eps : float, optional
        The desired relative error in the calculated value, defaults to 1e-5.

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

    The function is guaranteed to finish by virtue of the `max_iters` counter.

    """
    def _lcl_iter(p, p0, w, t):
        td = dewpoint(vapor_pressure(units.Quantity(p, pressure.units), w))
        return (p0 * (td / t) ** (1. / mpconsts.kappa)).m

    w = mixing_ratio(saturation_vapor_pressure(dewpt), pressure)
    fp = so.fixed_point(_lcl_iter, pressure.m, args=(pressure.m, w, temperature),
                        xtol=eps, maxiter=max_iters)
    lcl_p = fp * pressure.units
    return lcl_p, dewpoint(vapor_pressure(lcl_p, w))


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]', '[temperature]')
def lfc(pressure, temperature, dewpt, parcel_temperature_profile=None):
    r"""Calculate the level of free convection (LFC).

    This works by finding the first intersection of the ideal parcel path and
    the measured parcel temperature.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure
    temperature : `pint.Quantity`
        The temperature at the levels given by `pressure`
    dewpt : `pint.Quantity`
        The dew point at the levels given by `pressure`
    parcel_temperature_profile: `pint.Quantity`, optional
        The parcel temperature profile from which to calculate the LFC. Defaults to the
        surface parcel profile.

    Returns
    -------
    `pint.Quantity`
        The LFC pressure and temperature

    See Also
    --------
    parcel_profile

    """
    # Default to surface parcel if no profile or starting pressure level is given
    if parcel_temperature_profile is None:
        new_stuff = parcel_profile_with_lcl(pressure, temperature, dewpt)
        pressure, temperature, _, parcel_temperature_profile = new_stuff
        temperature = temperature.to('degC')
        parcel_temperature_profile = parcel_temperature_profile.to('degC')

    # The parcel profile and data have the same first data point, so we ignore
    # that point to get the real first intersection for the LFC calculation.
    x, y = find_intersections(pressure[1:], parcel_temperature_profile[1:],
                              temperature[1:], direction='increasing')

    # The LFC could:
    # 1) Not exist
    # 2) Exist but be equal to the LCL
    # 3) Exist and be above the LCL

    # LFC does not exist or is LCL
    if len(x) == 0:
        if np.all(_less_or_close(parcel_temperature_profile, temperature)):
            # LFC doesn't exist
            return np.nan * pressure.units, np.nan * temperature.units
        else:  # LFC = LCL
            x, y = lcl(pressure[0], temperature[0], dewpt[0])
            return x, y

    # LFC exists and is not LCL. Make sure it is above the LCL.
    else:
        idx = x < lcl(pressure[0], temperature[0], dewpt[0])[0]
        x = x[idx]
        y = y[idx]
        return x[0], y[0]


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]', '[temperature]')
def el(pressure, temperature, dewpt, parcel_temperature_profile=None):
    r"""Calculate the equilibrium level.

    This works by finding the last intersection of the ideal parcel path and
    the measured environmental temperature. If there is one or fewer intersections, there is
    no equilibrium level.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure
    temperature : `pint.Quantity`
        The temperature at the levels given by `pressure`
    dewpt : `pint.Quantity`
        The dew point at the levels given by `pressure`
    parcel_temperature_profile: `pint.Quantity`, optional
        The parcel temperature profile from which to calculate the EL. Defaults to the
        surface parcel profile.

    Returns
    -------
    `pint.Quantity, pint.Quantity`
        The EL pressure and temperature

    See Also
    --------
    parcel_profile

    """
    # Default to surface parcel if no profile or starting pressure level is given
    if parcel_temperature_profile is None:
        new_stuff = parcel_profile_with_lcl(pressure, temperature, dewpt)
        pressure, temperature, _, parcel_temperature_profile = new_stuff
        temperature = temperature.to('degC')
        parcel_temperature_profile = parcel_temperature_profile.to('degC')

    # If the top of the sounding parcel is warmer than the environment, there is no EL
    if parcel_temperature_profile[-1] > temperature[-1]:
        return np.nan * pressure.units, np.nan * temperature.units

    # Otherwise the last intersection (as long as there is one) is the EL
    x, y = find_intersections(pressure[1:], parcel_temperature_profile[1:], temperature[1:])
    if len(x) > 0:
        return x[-1], y[-1]
    else:
        return np.nan * pressure.units, np.nan * temperature.units


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]')
def parcel_profile(pressure, temperature, dewpt):
    r"""Calculate the profile a parcel takes through the atmosphere.

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

    """
    _, _, _, t_l, _, t_u = _parcel_profile_helper(pressure, temperature, dewpt)
    return concatenate((t_l, t_u))


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]')
def parcel_profile_with_lcl(pressure, temperature, dewpt):
    r"""Calculate the profile a parcel takes through the atmosphere.

    The parcel starts at `temperature`, and `dewpt`, lifted up
    dry adiabatically to the LCL, and then moist adiabatically from there.
    `pressure` specifies the pressure levels for the profile. This function returns
    a profile that includes the LCL.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure level(s) of interest. The first entry should be the starting
        point pressure.
    temperature : `pint.Quantity`
        The atmospheric temperature at the levels in `pressure`. The first entry should be the
        starting point temperature.
    dewpt : `pint.Quantity`
        The atmospheric dew point at the levels in `pressure`. The first entry should be the
        starting dew point.

    Returns
    -------
    pressure : `pint.Quantity`
        The parcel profile pressures, which includes the specified levels and the LCL
    ambient_temperature : `pint.Quantity`
        The atmospheric temperature values, including the value interpolated to the LCL level
    ambient_dew_point : `pint.Quantity`
        The atmospheric dew point values, including the value interpolated to the LCL level
    profile_temperature : `pint.Quantity`
        The parcel profile temperatures at all of the levels in the returned pressures array,
        including the LCL.

    See Also
    --------
    lcl, moist_lapse, dry_lapse, parcel_profile

    """
    p_l, p_lcl, p_u, t_l, t_lcl, t_u = _parcel_profile_helper(pressure, temperature[0],
                                                              dewpt[0])
    new_press = concatenate((p_l, p_lcl, p_u))
    prof_temp = concatenate((t_l, t_lcl, t_u))
    new_temp = _insert_lcl_level(pressure, temperature, p_lcl)
    new_dewp = _insert_lcl_level(pressure, dewpt, p_lcl)
    return new_press, new_temp, new_dewp, prof_temp


def _parcel_profile_helper(pressure, temperature, dewpt):
    """Help calculate parcel profiles.

    Returns the temperature and pressure, above, below, and including the LCL. The
    other calculation functions decide what to do with the pieces.

    """
    # Find the LCL
    press_lcl, temp_lcl = lcl(pressure[0], temperature, dewpt)
    press_lcl = press_lcl.to(pressure.units)

    # Find the dry adiabatic profile, *including* the LCL. We need >= the LCL in case the
    # LCL is included in the levels. It's slightly redundant in that case, but simplifies
    # the logic for removing it later.
    press_lower = concatenate((pressure[pressure >= press_lcl], press_lcl))
    temp_lower = dry_lapse(press_lower, temperature)

    # If the pressure profile doesn't make it to the lcl, we can stop here
    if _greater_or_close(np.nanmin(pressure), press_lcl.m):
        return (press_lower[:-1], press_lcl, np.array([]) * press_lower.units,
                temp_lower[:-1], temp_lcl, np.array([]) * temp_lower.units)

    # Find moist pseudo-adiabatic profile starting at the LCL
    press_upper = concatenate((press_lcl, pressure[pressure < press_lcl]))
    temp_upper = moist_lapse(press_upper, temp_lower[-1]).to(temp_lower.units)

    # Return profile pieces
    return (press_lower[:-1], press_lcl, press_upper[1:],
            temp_lower[:-1], temp_lcl, temp_upper[1:])


def _insert_lcl_level(pressure, temperature, lcl_pressure):
    """Insert the LCL pressure into the profile."""
    interp_temp = interpolate_1d(lcl_pressure, pressure, temperature)

    # Pressure needs to be increasing for searchsorted, so flip it and then convert
    # the index back to the original array
    loc = pressure.size - pressure[::-1].searchsorted(lcl_pressure)
    return np.insert(temperature.m, loc, interp_temp.m) * temperature.units


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[dimensionless]')
def vapor_pressure(pressure, mixing):
    r"""Calculate water vapor (partial) pressure.

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

    Notes
    -----
    This function is a straightforward implementation of the equation given in many places,
    such as [Hobbs1977]_ pg.71:

    .. math:: e = p \frac{r}{r + \epsilon}

    See Also
    --------
    saturation_vapor_pressure, dewpoint

    """
    return pressure * mixing / (mpconsts.epsilon + mixing)


@exporter.export
@preprocess_xarray
@check_units('[temperature]')
def saturation_vapor_pressure(temperature):
    r"""Calculate the saturation water vapor (partial) pressure.

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

    The formula used is that from [Bolton1980]_ for T in degrees Celsius:

    .. math:: 6.112 e^\frac{17.67T}{T + 243.5}

    """
    # Converted from original in terms of C to use kelvin. Using raw absolute values of C in
    # a formula plays havoc with units support.
    return sat_pressure_0c * np.exp(17.67 * (temperature - 273.15 * units.kelvin)
                                    / (temperature - 29.65 * units.kelvin))


@exporter.export
@preprocess_xarray
@check_units('[temperature]', '[dimensionless]')
def dewpoint_rh(temperature, rh):
    r"""Calculate the ambient dewpoint given air temperature and relative humidity.

    Parameters
    ----------
    temperature : `pint.Quantity`
        Air temperature
    rh : `pint.Quantity`
        Relative humidity expressed as a ratio in the range 0 < rh <= 1

    Returns
    -------
    `pint.Quantity`
        The dew point temperature

    See Also
    --------
    dewpoint, saturation_vapor_pressure

    """
    if np.any(rh > 1.2):
        warnings.warn('Relative humidity >120%, ensure proper units.')
    return dewpoint(rh * saturation_vapor_pressure(temperature))


@exporter.export
@preprocess_xarray
@check_units('[pressure]')
def dewpoint(e):
    r"""Calculate the ambient dewpoint given the vapor pressure.

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
    This function inverts the [Bolton1980]_ formula for saturation vapor
    pressure to instead calculate the temperature. This yield the following
    formula for dewpoint in degrees Celsius:

    .. math:: T = \frac{243.5 log(e / 6.112)}{17.67 - log(e / 6.112)}

    """
    val = np.log(e / sat_pressure_0c)
    return 0. * units.degC + 243.5 * units.delta_degC * val / (17.67 - val)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[pressure]', '[dimensionless]')
def mixing_ratio(part_press, tot_press, molecular_weight_ratio=mpconsts.epsilon):
    r"""Calculate the mixing ratio of a gas.

    This calculates mixing ratio given its partial pressure and the total pressure of
    the air. There are no required units for the input arrays, other than that
    they have the same units.

    Parameters
    ----------
    part_press : `pint.Quantity`
        Partial pressure of the constituent gas
    tot_press : `pint.Quantity`
        Total air pressure
    molecular_weight_ratio : `pint.Quantity` or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air
        (:math:`\epsilon\approx0.622`).

    Returns
    -------
    `pint.Quantity`
        The (mass) mixing ratio, dimensionless (e.g. Kg/Kg or g/g)

    Notes
    -----
    This function is a straightforward implementation of the equation given in many places,
    such as [Hobbs1977]_ pg.73:

    .. math:: r = \epsilon \frac{e}{p - e}

    See Also
    --------
    saturation_mixing_ratio, vapor_pressure

    """
    return (molecular_weight_ratio * part_press
            / (tot_press - part_press)).to('dimensionless')


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]')
def saturation_mixing_ratio(tot_press, temperature):
    r"""Calculate the saturation mixing ratio of water vapor.

    This calculation is given total pressure and the temperature. The implementation
    uses the formula outlined in [Hobbs1977]_ pg.73.

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

    """
    return mixing_ratio(saturation_vapor_pressure(temperature), tot_press)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]')
def equivalent_potential_temperature(pressure, temperature, dewpoint):
    r"""Calculate equivalent potential temperature.

    This calculation must be given an air parcel's pressure, temperature, and dewpoint.
    The implementation uses the formula outlined in [Bolton1980]_:

    First, the LCL temperature is calculated:

    .. math:: T_{L}=\frac{1}{\frac{1}{T_{D}-56}+\frac{ln(T_{K}/T_{D})}{800}}+56

    Which is then used to calculate the potential temperature at the LCL:

    .. math:: \theta_{DL}=T_{K}\left(\frac{1000}{p-e}\right)^k
              \left(\frac{T_{K}}{T_{L}}\right)^{.28r}

    Both of these are used to calculate the final equivalent potential temperature:

    .. math:: \theta_{E}=\theta_{DL}\exp\left[\left(\frac{3036.}{T_{L}}
                                              -1.78\right)*r(1+.448r)\right]

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure
    temperature: `pint.Quantity`
        Temperature of parcel
    dewpoint: `pint.Quantity`
        Dewpoint of parcel

    Returns
    -------
    `pint.Quantity`
        The equivalent potential temperature of the parcel

    Notes
    -----
    [Bolton1980]_ formula for Theta-e is used, since according to
    [DaviesJones2009]_ it is the most accurate non-iterative formulation
    available.

    """
    t = temperature.to('kelvin').magnitude
    td = dewpoint.to('kelvin').magnitude
    p = pressure.to('hPa').magnitude
    e = saturation_vapor_pressure(dewpoint).to('hPa').magnitude
    r = saturation_mixing_ratio(pressure, dewpoint).magnitude

    t_l = 56 + 1. / (1. / (td - 56) + np.log(t / td) / 800.)
    th_l = t * (1000 / (p - e)) ** mpconsts.kappa * (t / t_l) ** (0.28 * r)
    th_e = th_l * np.exp((3036. / t_l - 1.78) * r * (1 + 0.448 * r))

    return th_e * units.kelvin


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]')
def saturation_equivalent_potential_temperature(pressure, temperature):
    r"""Calculate saturation equivalent potential temperature.

    This calculation must be given an air parcel's pressure and temperature.
    The implementation uses the formula outlined in [Bolton1980]_ for the
    equivalent potential temperature, and assumes a saturated process.

    First, because we assume a saturated process, the temperature at the LCL is
    equivalent to the current temperature. Therefore the following equation

    .. math:: T_{L}=\frac{1}{\frac{1}{T_{D}-56}+\frac{ln(T_{K}/T_{D})}{800}}+56

    reduces to

    .. math:: T_{L} = T_{K}

    Then the potential temperature at the temperature/LCL is calculated:

    .. math:: \theta_{DL}=T_{K}\left(\frac{1000}{p-e}\right)^k
              \left(\frac{T_{K}}{T_{L}}\right)^{.28r}

    However, because

    .. math:: T_{L} = T_{K}

    it follows that

    .. math:: \theta_{DL}=T_{K}\left(\frac{1000}{p-e}\right)^k

    Both of these are used to calculate the final equivalent potential temperature:

    .. math:: \theta_{E}=\theta_{DL}\exp\left[\left(\frac{3036.}{T_{K}}
                                              -1.78\right)*r(1+.448r)\right]

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure
    temperature: `pint.Quantity`
        Temperature of parcel

    Returns
    -------
    `pint.Quantity`
        The saturation equivalent potential temperature of the parcel

    Notes
    -----
    [Bolton1980]_ formula for Theta-e is used (for saturated case), since according to
    [DaviesJones2009]_ it is the most accurate non-iterative formulation
    available.

    """
    t = temperature.to('kelvin').magnitude
    p = pressure.to('hPa').magnitude
    e = saturation_vapor_pressure(temperature).to('hPa').magnitude
    r = saturation_mixing_ratio(pressure, temperature).magnitude

    th_l = t * (1000 / (p - e)) ** mpconsts.kappa
    th_es = th_l * np.exp((3036. / t - 1.78) * r * (1 + 0.448 * r))

    return th_es * units.kelvin


@exporter.export
@preprocess_xarray
@check_units('[temperature]', '[dimensionless]', '[dimensionless]')
def virtual_temperature(temperature, mixing, molecular_weight_ratio=mpconsts.epsilon):
    r"""Calculate virtual temperature.

    This calculation must be given an air parcel's temperature and mixing ratio.
    The implementation uses the formula outlined in [Hobbs2006]_ pg.80.

    Parameters
    ----------
    temperature: `pint.Quantity`
        The temperature
    mixing : `pint.Quantity`
        dimensionless mass mixing ratio
    molecular_weight_ratio : `pint.Quantity` or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air.
        (:math:`\epsilon\approx0.622`).

    Returns
    -------
    `pint.Quantity`
        The corresponding virtual temperature of the parcel

    Notes
    -----
    .. math:: T_v = T \frac{\text{w} + \epsilon}{\epsilon\,(1 + \text{w})}

    """
    return temperature * ((mixing + molecular_weight_ratio)
                          / (molecular_weight_ratio * (1 + mixing)))


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[dimensionless]', '[dimensionless]')
def virtual_potential_temperature(pressure, temperature, mixing,
                                  molecular_weight_ratio=mpconsts.epsilon):
    r"""Calculate virtual potential temperature.

    This calculation must be given an air parcel's pressure, temperature, and mixing ratio.
    The implementation uses the formula outlined in [Markowski2010]_ pg.13.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure
    temperature: `pint.Quantity`
        The temperature
    mixing : `pint.Quantity`
        dimensionless mass mixing ratio
    molecular_weight_ratio : `pint.Quantity` or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air.
        (:math:`\epsilon\approx0.622`).

    Returns
    -------
    `pint.Quantity`
        The corresponding virtual potential temperature of the parcel

    Notes
    -----
    .. math:: \Theta_v = \Theta \frac{\text{w} + \epsilon}{\epsilon\,(1 + \text{w})}

    """
    pottemp = potential_temperature(pressure, temperature)
    return virtual_temperature(pottemp, mixing, molecular_weight_ratio)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[dimensionless]', '[dimensionless]')
def density(pressure, temperature, mixing, molecular_weight_ratio=mpconsts.epsilon):
    r"""Calculate density.

    This calculation must be given an air parcel's pressure, temperature, and mixing ratio.
    The implementation uses the formula outlined in [Hobbs2006]_ pg.67.

    Parameters
    ----------
    temperature: `pint.Quantity`
        The temperature
    pressure: `pint.Quantity`
        Total atmospheric pressure
    mixing : `pint.Quantity`
        dimensionless mass mixing ratio
    molecular_weight_ratio : `pint.Quantity` or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air.
        (:math:`\epsilon\approx0.622`).

    Returns
    -------
    `pint.Quantity`
        The corresponding density of the parcel

    Notes
    -----
    .. math:: \rho = \frac{p}{R_dT_v}

    """
    virttemp = virtual_temperature(temperature, mixing, molecular_weight_ratio)
    return (pressure / (mpconsts.Rd * virttemp)).to(units.kilogram / units.meter ** 3)


@exporter.export
@preprocess_xarray
@check_units('[temperature]', '[temperature]', '[pressure]')
def relative_humidity_wet_psychrometric(dry_bulb_temperature, web_bulb_temperature,
                                        pressure, **kwargs):
    r"""Calculate the relative humidity with wet bulb and dry bulb temperatures.

    This uses a psychrometric relationship as outlined in [WMO8-2014]_, with
    coefficients from [Fan1987]_.

    Parameters
    ----------
    dry_bulb_temperature: `pint.Quantity`
        Dry bulb temperature
    web_bulb_temperature: `pint.Quantity`
        Wet bulb temperature
    pressure: `pint.Quantity`
        Total atmospheric pressure

    Returns
    -------
    `pint.Quantity`
        Relative humidity

    Notes
    -----
    .. math:: RH = \frac{e}{e_s}

    * :math:`RH` is relative humidity as a unitless ratio
    * :math:`e` is vapor pressure from the wet psychrometric calculation
    * :math:`e_s` is the saturation vapor pressure

    See Also
    --------
    psychrometric_vapor_pressure_wet, saturation_vapor_pressure

    """
    return (psychrometric_vapor_pressure_wet(dry_bulb_temperature, web_bulb_temperature,
                                             pressure, **kwargs)
            / saturation_vapor_pressure(dry_bulb_temperature))


@exporter.export
@preprocess_xarray
@check_units('[temperature]', '[temperature]', '[pressure]')
def psychrometric_vapor_pressure_wet(dry_bulb_temperature, wet_bulb_temperature, pressure,
                                     psychrometer_coefficient=6.21e-4 / units.kelvin):
    r"""Calculate the vapor pressure with wet bulb and dry bulb temperatures.

    This uses a psychrometric relationship as outlined in [WMO8-2014]_, with
    coefficients from [Fan1987]_.

    Parameters
    ----------
    dry_bulb_temperature: `pint.Quantity`
        Dry bulb temperature
    wet_bulb_temperature: `pint.Quantity`
        Wet bulb temperature
    pressure: `pint.Quantity`
        Total atmospheric pressure
    psychrometer_coefficient: `pint.Quantity`, optional
        Psychrometer coefficient. Defaults to 6.21e-4 K^-1.

    Returns
    -------
    `pint.Quantity`
        Vapor pressure

    Notes
    -----
    .. math:: e' = e'_w(T_w) - A p (T - T_w)

    * :math:`e'` is vapor pressure
    * :math:`e'_w(T_w)` is the saturation vapor pressure with respect to water at temperature
      :math:`T_w`
    * :math:`p` is the pressure of the wet bulb
    * :math:`T` is the temperature of the dry bulb
    * :math:`T_w` is the temperature of the wet bulb
    * :math:`A` is the psychrometer coefficient

    Psychrometer coefficient depends on the specific instrument being used and the ventilation
    of the instrument.

    See Also
    --------
    saturation_vapor_pressure

    """
    return (saturation_vapor_pressure(wet_bulb_temperature) - psychrometer_coefficient
            * pressure * (dry_bulb_temperature - wet_bulb_temperature).to('kelvin'))


@exporter.export
@preprocess_xarray
@check_units('[dimensionless]', '[temperature]', '[pressure]')
def mixing_ratio_from_relative_humidity(relative_humidity, temperature, pressure):
    r"""Calculate the mixing ratio from relative humidity, temperature, and pressure.

    Parameters
    ----------
    relative_humidity: array_like
        The relative humidity expressed as a unitless ratio in the range [0, 1]. Can also pass
        a percentage if proper units are attached.
    temperature: `pint.Quantity`
        Air temperature
    pressure: `pint.Quantity`
        Total atmospheric pressure

    Returns
    -------
    `pint.Quantity`
        Dimensionless mixing ratio

    Notes
    -----
    Formula adapted from [Hobbs1977]_ pg. 74.

    .. math:: w = (RH)(w_s)

    * :math:`w` is mixing ratio
    * :math:`RH` is relative humidity as a unitless ratio
    * :math:`w_s` is the saturation mixing ratio

    See Also
    --------
    relative_humidity_from_mixing_ratio, saturation_mixing_ratio

    """
    return (relative_humidity
            * saturation_mixing_ratio(pressure, temperature)).to('dimensionless')


@exporter.export
@preprocess_xarray
@check_units('[dimensionless]', '[temperature]', '[pressure]')
def relative_humidity_from_mixing_ratio(mixing_ratio, temperature, pressure):
    r"""Calculate the relative humidity from mixing ratio, temperature, and pressure.

    Parameters
    ----------
    mixing_ratio: `pint.Quantity`
        Dimensionless mass mixing ratio
    temperature: `pint.Quantity`
        Air temperature
    pressure: `pint.Quantity`
        Total atmospheric pressure

    Returns
    -------
    `pint.Quantity`
        Relative humidity

    Notes
    -----
    Formula based on that from [Hobbs1977]_ pg. 74.

    .. math:: RH = \frac{w}{w_s}

    * :math:`RH` is relative humidity as a unitless ratio
    * :math:`w` is mixing ratio
    * :math:`w_s` is the saturation mixing ratio

    See Also
    --------
    mixing_ratio_from_relative_humidity, saturation_mixing_ratio

    """
    return mixing_ratio / saturation_mixing_ratio(pressure, temperature)


@exporter.export
@preprocess_xarray
@check_units('[dimensionless]')
def mixing_ratio_from_specific_humidity(specific_humidity):
    r"""Calculate the mixing ratio from specific humidity.

    Parameters
    ----------
    specific_humidity: `pint.Quantity`
        Specific humidity of air

    Returns
    -------
    `pint.Quantity`
        Mixing ratio

    Notes
    -----
    Formula from [Salby1996]_ pg. 118.

    .. math:: w = \frac{q}{1-q}

    * :math:`w` is mixing ratio
    * :math:`q` is the specific humidity

    See Also
    --------
    mixing_ratio, specific_humidity_from_mixing_ratio

    """
    try:
        specific_humidity = specific_humidity.to('dimensionless')
    except AttributeError:
        pass
    return specific_humidity / (1 - specific_humidity)


@exporter.export
@preprocess_xarray
@check_units('[dimensionless]')
def specific_humidity_from_mixing_ratio(mixing_ratio):
    r"""Calculate the specific humidity from the mixing ratio.

    Parameters
    ----------
    mixing_ratio: `pint.Quantity`
        mixing ratio

    Returns
    -------
    `pint.Quantity`
        Specific humidity

    Notes
    -----
    Formula from [Salby1996]_ pg. 118.

    .. math:: q = \frac{w}{1+w}

    * :math:`w` is mixing ratio
    * :math:`q` is the specific humidity

    See Also
    --------
    mixing_ratio, mixing_ratio_from_specific_humidity

    """
    try:
        mixing_ratio = mixing_ratio.to('dimensionless')
    except AttributeError:
        pass
    return mixing_ratio / (1 + mixing_ratio)


@exporter.export
@preprocess_xarray
@check_units('[dimensionless]', '[temperature]', '[pressure]')
def relative_humidity_from_specific_humidity(specific_humidity, temperature, pressure):
    r"""Calculate the relative humidity from specific humidity, temperature, and pressure.

    Parameters
    ----------
    specific_humidity: `pint.Quantity`
        Specific humidity of air
    temperature: `pint.Quantity`
        Air temperature
    pressure: `pint.Quantity`
        Total atmospheric pressure

    Returns
    -------
    `pint.Quantity`
        Relative humidity

    Notes
    -----
    Formula based on that from [Hobbs1977]_ pg. 74. and [Salby1996]_ pg. 118.

    .. math:: RH = \frac{q}{(1-q)w_s}

    * :math:`RH` is relative humidity as a unitless ratio
    * :math:`q` is specific humidity
    * :math:`w_s` is the saturation mixing ratio

    See Also
    --------
    relative_humidity_from_mixing_ratio

    """
    return (mixing_ratio_from_specific_humidity(specific_humidity)
            / saturation_mixing_ratio(pressure, temperature))


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]', '[temperature]')
def cape_cin(pressure, temperature, dewpt, parcel_profile):
    r"""Calculate CAPE and CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile and parcel path. CIN is integrated between the surface and
    LFC, CAPE is integrated between the LFC and EL (or top of sounding). Intersection points of
    the measured temperature profile and parcel profile are linearly interpolated.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure level(s) of interest. The first entry should be the starting
        point pressure.
    temperature : `pint.Quantity`
        The atmospheric temperature corresponding to pressure.
    dewpt : `pint.Quantity`
        The atmospheric dew point corresponding to pressure.
    parcel_profile : `pint.Quantity`
        The temperature profile of the parcel

    Returns
    -------
    `pint.Quantity`
        Convective available potential energy (CAPE).
    `pint.Quantity`
        Convective inhibition (CIN).

    Notes
    -----
    Formula adopted from [Hobbs1977]_.

    .. math:: \text{CAPE} = -R_d \int_{LFC}^{EL} (T_{parcel} - T_{env}) d\text{ln}(p)

    .. math:: \text{CIN} = -R_d \int_{SFC}^{LFC} (T_{parcel} - T_{env}) d\text{ln}(p)


    * :math:`CAPE` Convective available potential energy
    * :math:`CIN` Convective inhibition
    * :math:`LFC` Pressure of the level of free convection
    * :math:`EL` Pressure of the equilibrium level
    * :math:`SFC` Level of the surface or beginning of parcel path
    * :math:`R_d` Gas constant
    * :math:`g` Gravitational acceleration
    * :math:`T_{parcel}` Parcel temperature
    * :math:`T_{env}` Environment temperature
    * :math:`p` Atmospheric pressure

    See Also
    --------
    lfc, el

    """
    # Calculate LFC limit of integration
    lfc_pressure, _ = lfc(pressure, temperature, dewpt,
                          parcel_temperature_profile=parcel_profile)

    # If there is no LFC, no need to proceed.
    if np.isnan(lfc_pressure):
        return 0 * units('J/kg'), 0 * units('J/kg')
    else:
        lfc_pressure = lfc_pressure.magnitude

    # Calculate the EL limit of integration
    el_pressure, _ = el(pressure, temperature, dewpt,
                        parcel_temperature_profile=parcel_profile)

    # No EL and we use the top reading of the sounding.
    if np.isnan(el_pressure):
        el_pressure = pressure[-1].magnitude
    else:
        el_pressure = el_pressure.magnitude

    # Difference between the parcel path and measured temperature profiles
    y = (parcel_profile - temperature).to(units.degK)

    # Estimate zero crossings
    x, y = _find_append_zero_crossings(np.copy(pressure), y)

    # CAPE
    # Only use data between the LFC and EL for calculation
    p_mask = _less_or_close(x, lfc_pressure) & _greater_or_close(x, el_pressure)
    x_clipped = x[p_mask]
    y_clipped = y[p_mask]
    cape = (mpconsts.Rd
            * (np.trapz(y_clipped, np.log(x_clipped)) * units.degK)).to(units('J/kg'))

    # CIN
    # Only use data between the surface and LFC for calculation
    p_mask = _greater_or_close(x, lfc_pressure)
    x_clipped = x[p_mask]
    y_clipped = y[p_mask]
    cin = (mpconsts.Rd
           * (np.trapz(y_clipped, np.log(x_clipped)) * units.degK)).to(units('J/kg'))

    return cape, cin


def _find_append_zero_crossings(x, y):
    r"""
    Find and interpolate zero crossings.

    Estimate the zero crossings of an x,y series and add estimated crossings to series,
    returning a sorted array with no duplicate values.

    Parameters
    ----------
    x : `pint.Quantity`
        x values of data
    y : `pint.Quantity`
        y values of data

    Returns
    -------
    x : `pint.Quantity`
        x values of data
    y : `pint.Quantity`
        y values of data

    """
    # Find and append crossings to the data
    crossings = find_intersections(x[1:], y[1:], np.zeros_like(y[1:]) * y.units)
    x = concatenate((x, crossings[0]))
    y = concatenate((y, crossings[1]))

    # Resort so that data are in order
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Remove duplicate data points if there are any
    keep_idx = np.ediff1d(x, to_end=[1]) > 0
    x = x[keep_idx]
    y = y[keep_idx]
    return x, y


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]')
def most_unstable_parcel(pressure, temperature, dewpoint, heights=None,
                         bottom=None, depth=300 * units.hPa):
    """
    Determine the most unstable parcel in a layer.

    Determines the most unstable parcel of air by calculating the equivalent
    potential temperature and finding its maximum in the specified layer.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Atmospheric pressure profile
    temperature: `pint.Quantity`
        Atmospheric temperature profile
    dewpoint: `pint.Quantity`
        Atmospheric dewpoint profile
    heights: `pint.Quantity`, optional
        Atmospheric height profile. Standard atmosphere assumed when None (the default).
    bottom: `pint.Quantity`, optional
        Bottom of the layer to consider for the calculation in pressure or height.
        Defaults to using the bottom pressure or height.
    depth: `pint.Quantity`, optional
        Depth of the layer to consider for the calculation in pressure or height. Defaults
        to 300 hPa.

    Returns
    -------
    `pint.Quantity`
        Pressure, temperature, and dew point of most unstable parcel in the profile.
    integer
        Index of the most unstable parcel in the given profile

    See Also
    --------
    get_layer

    """
    p_layer, t_layer, td_layer = get_layer(pressure, temperature, dewpoint, bottom=bottom,
                                           depth=depth, heights=heights, interpolate=False)
    theta_e = equivalent_potential_temperature(p_layer, t_layer, td_layer)
    max_idx = np.argmax(theta_e)
    return p_layer[max_idx], t_layer[max_idx], td_layer[max_idx], max_idx


@exporter.export
@preprocess_xarray
@check_units('[temperature]', '[pressure]', '[temperature]')
def isentropic_interpolation(theta_levels, pressure, temperature, *args, **kwargs):
    r"""Interpolate data in isobaric coordinates to isentropic coordinates.

    Parameters
    ----------
    theta_levels : array
        One-dimensional array of desired theta surfaces
    pressure : array
        One-dimensional array of pressure levels
    temperature : array
        Array of temperature
    args : array, optional
        Any additional variables will be interpolated to each isentropic level.

    Returns
    -------
    list
        List with pressure at each isentropic level, followed by each additional
        argument interpolated to isentropic coordinates.

    Other Parameters
    ----------------
    axis : int, optional
        The axis corresponding to the vertical in the temperature array, defaults to 0.
    tmpk_out : bool, optional
        If true, will calculate temperature and output as the last item in the output list.
        Defaults to False.
    max_iters : int, optional
        The maximum number of iterations to use in calculation, defaults to 50.
    eps : float, optional
        The desired absolute error in the calculated value, defaults to 1e-6.
    bottom_up_search : bool, optional
        Controls whether to search for theta levels bottom-up, or top-down. Defaults to
        True, which is bottom-up search.

    Notes
    -----
    Input variable arrays must have the same number of vertical levels as the pressure levels
    array. Pressure is calculated on isentropic surfaces by assuming that temperature varies
    linearly with the natural log of pressure. Linear interpolation is then used in the
    vertical to find the pressure at each isentropic level. Interpolation method from
    [Ziv1994]_. Any additional arguments are assumed to vary linearly with temperature and will
    be linearly interpolated to the new isentropic levels.

    See Also
    --------
    potential_temperature

    """
    # iteration function to be used later
    # Calculates theta from linearly interpolated temperature and solves for pressure
    def _isen_iter(iter_log_p, isentlevs_nd, ka, a, b, pok):
        exner = pok * np.exp(-ka * iter_log_p)
        t = a * iter_log_p + b
        # Newton-Raphson iteration
        f = isentlevs_nd - t * exner
        fp = exner * (ka * t - a)
        return iter_log_p - (f / fp)

    # Change when Python 2.7 no longer supported
    # Pull out keyword arguments
    tmpk_out = kwargs.pop('tmpk_out', False)
    max_iters = kwargs.pop('max_iters', 50)
    eps = kwargs.pop('eps', 1e-6)
    axis = kwargs.pop('axis', 0)
    bottom_up_search = kwargs.pop('bottom_up_search', True)

    # Get dimensions in temperature
    ndim = temperature.ndim

    # Convert units
    pres = pressure.to('hPa')
    temperature = temperature.to('kelvin')

    slices = [np.newaxis] * ndim
    slices[axis] = slice(None)
    slices = tuple(slices)
    pres = np.broadcast_to(pres[slices], temperature.shape) * pres.units

    # Sort input data
    sort_pres = np.argsort(pres.m, axis=axis)
    sort_pres = np.swapaxes(np.swapaxes(sort_pres, 0, axis)[::-1], 0, axis)
    sorter = broadcast_indices(pres, sort_pres, ndim, axis)
    levs = pres[sorter]
    tmpk = temperature[sorter]

    theta_levels = np.asanyarray(theta_levels.to('kelvin')).reshape(-1)
    isentlevels = theta_levels[np.argsort(theta_levels)]

    # Make the desired isentropic levels the same shape as temperature
    shape = list(temperature.shape)
    shape[axis] = isentlevels.size
    isentlevs_nd = np.broadcast_to(isentlevels[slices], shape)

    # exponent to Poisson's Equation, which is imported above
    ka = mpconsts.kappa.m_as('dimensionless')

    # calculate theta for each point
    pres_theta = potential_temperature(levs, tmpk)

    # Raise error if input theta level is larger than pres_theta max
    if np.max(pres_theta.m) < np.max(theta_levels):
        raise ValueError('Input theta level out of data bounds')

    # Find log of pressure to implement assumption of linear temperature dependence on
    # ln(p)
    log_p = np.log(levs.m)

    # Calculations for interpolation routine
    pok = mpconsts.P0 ** ka

    # index values for each point for the pressure level nearest to the desired theta level
    above, below, good = find_bounding_indices(pres_theta.m, theta_levels, axis,
                                               from_below=bottom_up_search)

    # calculate constants for the interpolation
    a = (tmpk.m[above] - tmpk.m[below]) / (log_p[above] - log_p[below])
    b = tmpk.m[above] - a * log_p[above]

    # calculate first guess for interpolation
    isentprs = 0.5 * (log_p[above] + log_p[below])

    # Make sure we ignore any nans in the data for solving; checking a is enough since it
    # combines log_p and tmpk.
    good &= ~np.isnan(a)

    # iterative interpolation using scipy.optimize.fixed_point and _isen_iter defined above
    log_p_solved = so.fixed_point(_isen_iter, isentprs[good],
                                  args=(isentlevs_nd[good], ka, a[good], b[good], pok.m),
                                  xtol=eps, maxiter=max_iters)

    # get back pressure from log p
    isentprs[good] = np.exp(log_p_solved)

    # Mask out points we know are bad as well as points that are beyond the max pressure
    isentprs[~(good & _less_or_close(isentprs, np.max(pres.m)))] = np.nan

    # create list for storing output data
    ret = [isentprs * units.hPa]

    # if tmpk_out = true, calculate temperature and output as last item in list
    if tmpk_out:
        ret.append((isentlevs_nd / ((mpconsts.P0.m / isentprs) ** ka)) * units.kelvin)

    # do an interpolation for each additional argument
    if args:
        others = interpolate_1d(isentlevels, pres_theta.m, *(arr[sorter] for arr in args),
                                axis=axis)
        if len(args) > 1:
            ret.extend(others)
        else:
            ret.append(others)

    return ret


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]')
def surface_based_cape_cin(pressure, temperature, dewpoint):
    r"""Calculate surface-based CAPE and CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile for a surface-based parcel. CIN is integrated
    between the surface and LFC, CAPE is integrated between the LFC and EL (or top of
    sounding). Intersection points of the measured temperature profile and parcel profile are
    linearly interpolated.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile. The first entry should be the starting
        (surface) observation.
    temperature : `pint.Quantity`
        Temperature profile
    dewpoint : `pint.Quantity`
        Dewpoint profile

    Returns
    -------
    `pint.Quantity`
        Surface based Convective Available Potential Energy (CAPE).
    `pint.Quantity`
        Surface based Convective INhibition (CIN).

    See Also
    --------
    cape_cin, parcel_profile

    """
    p, t, td, profile = parcel_profile_with_lcl(pressure, temperature, dewpoint)
    return cape_cin(p, t, td, profile)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]')
def most_unstable_cape_cin(pressure, temperature, dewpoint, **kwargs):
    r"""Calculate most unstable CAPE/CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile and most unstable parcel path. CIN is integrated between the
    surface and LFC, CAPE is integrated between the LFC and EL (or top of sounding).
    Intersection points of the measured temperature profile and parcel profile are linearly
    interpolated.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Pressure profile
    temperature : `pint.Quantity`
        Temperature profile
    dewpoint : `pint.Quantity`
        Dewpoint profile

    Returns
    -------
    `pint.Quantity`
        Most unstable Convective Available Potential Energy (CAPE).
    `pint.Quantity`
        Most unstable Convective INhibition (CIN).

    See Also
    --------
    cape_cin, most_unstable_parcel, parcel_profile

    """
    _, parcel_temperature, parcel_dewpoint, parcel_idx = most_unstable_parcel(pressure,
                                                                              temperature,
                                                                              dewpoint,
                                                                              **kwargs)
    mu_profile = parcel_profile(pressure[parcel_idx:], parcel_temperature, parcel_dewpoint)
    return cape_cin(pressure[parcel_idx:], temperature[parcel_idx:],
                    dewpoint[parcel_idx:], mu_profile)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]')
def mixed_parcel(p, temperature, dewpt, parcel_start_pressure=None,
                 heights=None, bottom=None, depth=100 * units.hPa, interpolate=True):
    r"""Calculate the properties of a parcel mixed from a layer.

    Determines the properties of an air parcel that is the result of complete mixing of a
    given atmospheric layer.

    Parameters
    ----------
    p : `pint.Quantity`
        Atmospheric pressure profile
    temperature : `pint.Quantity`
        Atmospheric temperature profile
    dewpt : `pint.Quantity`
        Atmospheric dewpoint profile
    parcel_start_pressure : `pint.Quantity`, optional
        Pressure at which the mixed parcel should begin (default None)
    heights: `pint.Quantity`, optional
        Atmospheric heights corresponding to the given pressures (default None)
    bottom : `pint.Quantity`, optional
        The bottom of the layer as a pressure or height above the surface pressure
        (default None)
    depth : `pint.Quantity`, optional
        The thickness of the layer as a pressure or height above the bottom of the layer
        (default 100 hPa)
    interpolate : bool, optional
        Interpolate the top and bottom points if they are not in the given data

    Returns
    -------
    `pint.Quantity, pint.Quantity, pint.Quantity`
        The pressure, temperature, and dewpoint of the mixed parcel.

    """
    # If a parcel starting pressure is not provided, use the surface
    if not parcel_start_pressure:
        parcel_start_pressure = p[0]

    # Calculate the potential temperature and mixing ratio over the layer
    theta = potential_temperature(p, temperature)
    mixing_ratio = saturation_mixing_ratio(p, dewpt)

    # Mix the variables over the layer
    mean_theta, mean_mixing_ratio = mixed_layer(p, theta, mixing_ratio, bottom=bottom,
                                                heights=heights, depth=depth,
                                                interpolate=interpolate)

    # Convert back to temperature
    mean_temperature = (mean_theta / potential_temperature(parcel_start_pressure,
                                                           1 * units.kelvin)) * units.kelvin

    # Convert back to dewpoint
    mean_vapor_pressure = vapor_pressure(parcel_start_pressure, mean_mixing_ratio)
    mean_dewpoint = dewpoint(mean_vapor_pressure)

    return (parcel_start_pressure, mean_temperature.to(temperature.units),
            mean_dewpoint.to(dewpt.units))


@exporter.export
@preprocess_xarray
@check_units('[pressure]')
def mixed_layer(p, *args, **kwargs):
    r"""Mix variable(s) over a layer, yielding a mass-weighted average.

    This function will integrate a data variable with respect to pressure and determine the
    average value using the mean value theorem.

    Parameters
    ----------
    p : array-like
        Atmospheric pressure profile
    datavar : array-like
        Atmospheric variable measured at the given pressures
    heights: array-like, optional
        Atmospheric heights corresponding to the given pressures (default None)
    bottom : `pint.Quantity`, optional
        The bottom of the layer as a pressure or height above the surface pressure
        (default None)
    depth : `pint.Quantity`, optional
        The thickness of the layer as a pressure or height above the bottom of the layer
        (default 100 hPa)
    interpolate : bool, optional
        Interpolate the top and bottom points if they are not in the given data

    Returns
    -------
    `pint.Quantity`
        The mixed value of the data variable.

    """
    # Pull out keyword arguments, remove when we drop Python 2.7
    heights = kwargs.pop('heights', None)
    bottom = kwargs.pop('bottom', None)
    depth = kwargs.pop('depth', 100 * units.hPa)
    interpolate = kwargs.pop('interpolate', True)

    layer = get_layer(p, *args, heights=heights, bottom=bottom,
                      depth=depth, interpolate=interpolate)
    p_layer = layer[0]
    datavars_layer = layer[1:]

    ret = []
    for datavar_layer in datavars_layer:
        actual_depth = abs(p_layer[0] - p_layer[-1])
        ret.append((-1. / actual_depth.m) * np.trapz(datavar_layer, p_layer)
                   * datavar_layer.units)
    return ret


@exporter.export
@preprocess_xarray
@check_units('[length]', '[temperature]')
def dry_static_energy(heights, temperature):
    r"""Calculate the dry static energy of parcels.

    This function will calculate the dry static energy following the first two terms of
    equation 3.72 in [Hobbs2006]_.

    Notes
    -----
    .. math::\text{dry static energy} = c_{pd} * T + gz

    * :math:`T` is temperature
    * :math:`z` is height

    Parameters
    ----------
    heights : array-like
        Atmospheric height
    temperature : array-like
        Atmospheric temperature

    Returns
    -------
    `pint.Quantity`
        The dry static energy

    """
    return (mpconsts.g * heights + mpconsts.Cp_d * temperature).to('kJ/kg')


@exporter.export
@preprocess_xarray
@check_units('[length]', '[temperature]', '[dimensionless]')
def moist_static_energy(heights, temperature, specific_humidity):
    r"""Calculate the moist static energy of parcels.

    This function will calculate the moist static energy following
    equation 3.72 in [Hobbs2006]_.
    Notes
    -----
    .. math::\text{moist static energy} = c_{pd} * T + gz + L_v q

    * :math:`T` is temperature
    * :math:`z` is height
    * :math:`q` is specific humidity

    Parameters
    ----------
    heights : array-like
        Atmospheric height
    temperature : array-like
        Atmospheric temperature
    specific_humidity : array-like
        Atmospheric specific humidity

    Returns
    -------
    `pint.Quantity`
        The moist static energy

    """
    return (dry_static_energy(heights, temperature)
            + mpconsts.Lv * specific_humidity.to('dimensionless')).to('kJ/kg')


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]')
def thickness_hydrostatic(pressure, temperature, **kwargs):
    r"""Calculate the thickness of a layer via the hypsometric equation.

    This thickness calculation uses the pressure and temperature profiles (and optionally
    mixing ratio) via the hypsometric equation with virtual temperature adjustment

    .. math:: Z_2 - Z_1 = -\frac{R_d}{g} \int_{p_1}^{p_2} T_v d\ln p,

    which is based off of Equation 3.24 in [Hobbs2006]_.

    This assumes a hydrostatic atmosphere.

    Layer bottom and depth specified in pressure.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile
    temperature : `pint.Quantity`
        Atmospheric temperature profile
    mixing : `pint.Quantity`, optional
        Profile of dimensionless mass mixing ratio. If none is given, virtual temperature
        is simply set to be the given temperature.
    molecular_weight_ratio : `pint.Quantity` or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air.
        (:math:`\epsilon\approx0.622`).
    bottom : `pint.Quantity`, optional
        The bottom of the layer in pressure. Defaults to the first observation.
    depth : `pint.Quantity`, optional
        The depth of the layer in hPa. Defaults to the full profile if bottom is not given,
        and 100 hPa if bottom is given.

    Returns
    -------
    `pint.Quantity`
        The thickness of the layer in meters.

    See Also
    --------
    thickness_hydrostatic_from_relative_humidity, pressure_to_height_std, virtual_temperature

    """
    mixing = kwargs.pop('mixing', None)
    molecular_weight_ratio = kwargs.pop('molecular_weight_ratio', mpconsts.epsilon)
    bottom = kwargs.pop('bottom', None)
    depth = kwargs.pop('depth', None)

    # Get the data for the layer, conditional upon bottom/depth being specified and mixing
    # ratio being given
    if bottom is None and depth is None:
        if mixing is None:
            layer_p, layer_virttemp = pressure, temperature
        else:
            layer_p = pressure
            layer_virttemp = virtual_temperature(temperature, mixing, molecular_weight_ratio)
    else:
        if mixing is None:
            layer_p, layer_virttemp = get_layer(pressure, temperature, bottom=bottom,
                                                depth=depth)
        else:
            layer_p, layer_temp, layer_w = get_layer(pressure, temperature, mixing,
                                                     bottom=bottom, depth=depth)
            layer_virttemp = virtual_temperature(layer_temp, layer_w, molecular_weight_ratio)

    # Take the integral (with unit handling) and return the result in meters
    return (- mpconsts.Rd / mpconsts.g * np.trapz(
        layer_virttemp.to('K'), x=np.log(layer_p / units.hPa)) * units.K).to('m')


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]')
def thickness_hydrostatic_from_relative_humidity(pressure, temperature, relative_humidity,
                                                 **kwargs):
    r"""Calculate the thickness of a layer given pressure, temperature and relative humidity.

    Similar to ``thickness_hydrostatic``, this thickness calculation uses the pressure,
    temperature, and relative humidity profiles via the hypsometric equation with virtual
    temperature adjustment.

    .. math:: Z_2 - Z_1 = -\frac{R_d}{g} \int_{p_1}^{p_2} T_v d\ln p,

    which is based off of Equation 3.24 in [Hobbs2006]_. Virtual temperature is calculated
    from the profiles of temperature and relative humidity.

    This assumes a hydrostatic atmosphere.

    Layer bottom and depth specified in pressure.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile
    temperature : `pint.Quantity`
        Atmospheric temperature profile
    relative_humidity : `pint.Quantity`
        Atmospheric relative humidity profile. The relative humidity is expressed as a
        unitless ratio in the range [0, 1]. Can also pass a percentage if proper units are
        attached.
    bottom : `pint.Quantity`, optional
        The bottom of the layer in pressure. Defaults to the first observation.
    depth : `pint.Quantity`, optional
        The depth of the layer in hPa. Defaults to the full profile if bottom is not given,
        and 100 hPa if bottom is given.

    Returns
    -------
    `pint.Quantity`
        The thickness of the layer in meters.

    See Also
    --------
    thickness_hydrostatic, pressure_to_height_std, virtual_temperature,
    mixing_ratio_from_relative_humidity

    """
    bottom = kwargs.pop('bottom', None)
    depth = kwargs.pop('depth', None)
    mixing = mixing_ratio_from_relative_humidity(relative_humidity, temperature, pressure)

    return thickness_hydrostatic(pressure, temperature, mixing=mixing, bottom=bottom,
                                 depth=depth)


@exporter.export
@preprocess_xarray
@check_units('[length]', '[temperature]')
def brunt_vaisala_frequency_squared(heights, potential_temperature, axis=0):
    r"""Calculate the square of the Brunt-Vaisala frequency.

    Brunt-Vaisala frequency squared (a measure of atmospheric stability) is given by the
    formula:

    .. math:: N^2 = \frac{g}{\theta} \frac{d\theta}{dz}

    This formula is based off of Equations 3.75 and 3.77 in [Hobbs2006]_.

    Parameters
    ----------
    heights : array-like
        One-dimensional profile of atmospheric height
    potential_temperature : array-like
        Atmospheric potential temperature
    axis : int, optional
        The axis corresponding to vertical in the potential temperature array, defaults to 0.

    Returns
    -------
    array-like
        The square of the Brunt-Vaisala frequency.

    See Also
    --------
    brunt_vaisala_frequency, brunt_vaisala_period, potential_temperature

    """
    # Ensure validity of temperature units
    potential_temperature = potential_temperature.to('K')

    # Calculate and return the square of Brunt-Vaisala frequency
    return mpconsts.g / potential_temperature * first_derivative(potential_temperature,
                                                                 x=heights, axis=axis)


@exporter.export
@preprocess_xarray
@check_units('[length]', '[temperature]')
def brunt_vaisala_frequency(heights, potential_temperature, axis=0):
    r"""Calculate the Brunt-Vaisala frequency.

    This function will calculate the Brunt-Vaisala frequency as follows:

    .. math:: N = \left( \frac{g}{\theta} \frac{d\theta}{dz} \right)^\frac{1}{2}

    This formula based off of Equations 3.75 and 3.77 in [Hobbs2006]_.

    This function is a wrapper for `brunt_vaisala_frequency_squared` that filters out negative
    (unstable) quanties and takes the square root.

    Parameters
    ----------
    heights : array-like
        One-dimensional profile of atmospheric height
    potential_temperature : array-like
        Atmospheric potential temperature
    axis : int, optional
        The axis corresponding to vertical in the potential temperature array, defaults to 0.

    Returns
    -------
    array-like
        Brunt-Vaisala frequency.

    See Also
    --------
    brunt_vaisala_frequency_squared, brunt_vaisala_period, potential_temperature

    """
    bv_freq_squared = brunt_vaisala_frequency_squared(heights, potential_temperature,
                                                      axis=axis)
    bv_freq_squared[bv_freq_squared.magnitude < 0] = np.nan

    return np.sqrt(bv_freq_squared)


@exporter.export
@preprocess_xarray
@check_units('[length]', '[temperature]')
def brunt_vaisala_period(heights, potential_temperature, axis=0):
    r"""Calculate the Brunt-Vaisala period.

    This function is a helper function for `brunt_vaisala_frequency` that calculates the
    period of oscilation as in Exercise 3.13 of [Hobbs2006]_:

    .. math:: \tau = \frac{2\pi}{N}

    Returns `NaN` when :math:`N^2 > 0`.

    Parameters
    ----------
    heights : array-like
        One-dimensional profile of atmospheric height
    potential_temperature : array-like
        Atmospheric potential temperature
    axis : int, optional
        The axis corresponding to vertical in the potential temperature array, defaults to 0.

    Returns
    -------
    array-like
        Brunt-Vaisala period.

    See Also
    --------
    brunt_vaisala_frequency, brunt_vaisala_frequency_squared, potential_temperature

    """
    bv_freq_squared = brunt_vaisala_frequency_squared(heights, potential_temperature,
                                                      axis=axis)
    bv_freq_squared[bv_freq_squared.magnitude <= 0] = np.nan

    return 2 * np.pi / np.sqrt(bv_freq_squared)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', '[temperature]')
def wet_bulb_temperature(pressure, temperature, dewpoint):
    """Calculate the wet-bulb temperature using Normand's rule.

    This function calculates the wet-bulb temperature using the Normand method. The LCL is
    computed, and that parcel brought down to the starting pressure along a moist adiabat.
    The Normand method (and others) are described and compared by [Knox2017]_.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Initial atmospheric pressure
    temperature : `pint.Quantity`
        Initial atmospheric temperature
    dewpoint : `pint.Quantity`
        Initial atmospheric dewpoint

    Returns
    -------
    array-like
        Wet-bulb temperature

    See Also
    --------
    lcl, moist_lapse

    """
    if not hasattr(pressure, 'shape'):
        pressure = atleast_1d(pressure)
        temperature = atleast_1d(temperature)
        dewpoint = atleast_1d(dewpoint)

    it = np.nditer([pressure, temperature, dewpoint, None],
                   op_dtypes=['float', 'float', 'float', 'float'],
                   flags=['buffered'])

    for press, temp, dewp, ret in it:
        press = press * pressure.units
        temp = temp * temperature.units
        dewp = dewp * dewpoint.units
        lcl_pressure, lcl_temperature = lcl(press, temp, dewp)
        moist_adiabat_temperatures = moist_lapse(concatenate([lcl_pressure, press]),
                                                 lcl_temperature)
        ret[...] = moist_adiabat_temperatures[-1]

    # If we started with a scalar, return a scalar
    if it.operands[3].size == 1:
        return it.operands[3][0] * moist_adiabat_temperatures.units
    return it.operands[3] * moist_adiabat_temperatures.units


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]')
def static_stability(pressure, temperature, axis=0):
    r"""Calculate the static stability within a vertical profile.

    .. math:: \sigma = -\frac{RT}{p} \frac{\partial \ln \theta}{\partial p}

    This formuala is based on equation 4.3.6 in [Bluestein1992]_.

    Parameters
    ----------
    pressure : array-like
        Profile of atmospheric pressure
    temperature : array-like
        Profile of temperature
    axis : int, optional
        The axis corresponding to vertical in the pressure and temperature arrays, defaults
        to 0.

    Returns
    -------
    array-like
        The profile of static stability.

    """
    theta = potential_temperature(pressure, temperature)

    return - mpconsts.Rd * temperature / pressure * first_derivative(np.log(theta / units.K),
                                                                     x=pressure, axis=axis)


@exporter.export
@preprocess_xarray
@check_units('[dimensionless]', '[temperature]', '[pressure]')
def dewpoint_from_specific_humidity(specific_humidity, temperature, pressure):
    r"""Calculate the dewpoint from specific humidity, temperature, and pressure.

    Parameters
    ----------
    specific_humidity: `pint.Quantity`
        Specific humidity of air
    temperature: `pint.Quantity`
        Air temperature
    pressure: `pint.Quantity`
        Total atmospheric pressure

    Returns
    -------
    `pint.Quantity`
        Dewpoint temperature

    See Also
    --------
    relative_humidity_from_mixing_ratio, dewpoint_rh

    """
    return dewpoint_rh(temperature, relative_humidity_from_specific_humidity(specific_humidity,
                                                                             temperature,
                                                                             pressure))


@exporter.export
@preprocess_xarray
@check_units('[length]/[time]', '[pressure]', '[temperature]')
def vertical_velocity_pressure(w, pressure, temperature, mixing=0):
    r"""Calculate omega from w assuming hydrostatic conditions.

    This function converts vertical velocity with respect to height
    :math:`\left(w = \frac{Dz}{Dt}\right)` to that
    with respect to pressure :math:`\left(\omega = \frac{Dp}{Dt}\right)`
    assuming hydrostatic conditions on the synoptic scale.
    By Equation 7.33 in [Hobbs2006]_,

    .. math: \omega \simeq -\rho g w

    Density (:math:`\rho`) is calculated using the :func:`density` function,
    from the given pressure and temperature. If `mixing` is given, the virtual
    temperature correction is used, otherwise, dry air is assumed.

    Parameters
    ----------
    w: `pint.Quantity`
        Vertical velocity in terms of height
    pressure: `pint.Quantity`
        Total atmospheric pressure
    temperature: `pint.Quantity`
        Air temperature
    mixing: `pint.Quantity`, optional
        Mixing ratio of air

    Returns
    -------
    `pint.Quantity`
        Vertical velocity in terms of pressure (in Pascals / second)

    See Also
    --------
    density, vertical_velocity

    """
    rho = density(pressure, temperature, mixing)
    return (- mpconsts.g * rho * w).to('Pa/s')


@exporter.export
@preprocess_xarray
@check_units('[pressure]/[time]', '[pressure]', '[temperature]')
def vertical_velocity(omega, pressure, temperature, mixing=0):
    r"""Calculate w from omega assuming hydrostatic conditions.

    This function converts vertical velocity with respect to pressure
    :math:`\left(\omega = \frac{Dp}{Dt}\right)` to that with respect to height
    :math:`\left(w = \frac{Dz}{Dt}\right)` assuming hydrostatic conditions on
    the synoptic scale. By Equation 7.33 in [Hobbs2006]_,

    .. math: \omega \simeq -\rho g w

    so that

    .. math w \simeq \frac{- \omega}{\rho g}

    Density (:math:`\rho`) is calculated using the :func:`density` function,
    from the given pressure and temperature. If `mixing` is given, the virtual
    temperature correction is used, otherwise, dry air is assumed.

    Parameters
    ----------
    omega: `pint.Quantity`
        Vertical velocity in terms of pressure
    pressure: `pint.Quantity`
        Total atmospheric pressure
    temperature: `pint.Quantity`
        Air temperature
    mixing: `pint.Quantity`, optional
        Mixing ratio of air

    Returns
    -------
    `pint.Quantity`
        Vertical velocity in terms of height (in meters / second)

    See Also
    --------
    density, vertical_velocity_pressure

    """
    rho = density(pressure, temperature, mixing)
    return (omega / (- mpconsts.g * rho)).to('m/s')
