# Copyright (c) 2008,2015,2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains a collection of thermodynamic calculations."""
import contextlib
import warnings

import numpy as np
import scipy.integrate as si
import scipy.optimize as so
import xarray as xr

from .exceptions import InvalidSoundingError
from .tools import (_greater_or_close, _less_or_close, _remove_nans, find_bounding_indices,
                    find_intersections, first_derivative, get_layer)
from .. import constants as mpconsts
from ..cbook import broadcast_indices
from ..interpolate.one_dimension import interpolate_1d
from ..package_tools import Exporter
from ..units import check_units, concatenate, units
from ..xarray import add_vertical_dim_from_xarray, preprocess_and_wrap

exporter = Exporter(globals())

sat_pressure_0c = units.Quantity(6.112, 'millibar')


@exporter.export
@preprocess_and_wrap(wrap_like='temperature', broadcast=('temperature', 'dewpoint'))
@check_units('[temperature]', '[temperature]')
def relative_humidity_from_dewpoint(temperature, dewpoint):
    r"""Calculate the relative humidity.

    Uses temperature and dewpoint to calculate relative humidity as the ratio of vapor
    pressure to saturation vapor pressures.

    Parameters
    ----------
    temperature : `pint.Quantity`
        Air temperature

    dewpoint : `pint.Quantity`
        Dewpoint temperature

    Returns
    -------
    `pint.Quantity`
        Relative humidity

    Notes
    -----
    .. math:: rh = \frac{e(T_d)}{e_s(T)}

    .. versionchanged:: 1.0
       Renamed ``dewpt`` parameter to ``dewpoint``

    See Also
    --------
    saturation_vapor_pressure

    """
    e = saturation_vapor_pressure(dewpoint)
    e_s = saturation_vapor_pressure(temperature)
    return (e / e_s)


@exporter.export
@preprocess_and_wrap(wrap_like='pressure')
@check_units('[pressure]', '[pressure]')
def exner_function(pressure, reference_pressure=mpconsts.P0):
    r"""Calculate the Exner function.

    .. math:: \Pi = \left( \frac{p}{p_0} \right)^\kappa

    This can be used to calculate potential temperature from temperature (and visa-versa),
    since:

    .. math:: \Pi = \frac{T}{\theta}

    Parameters
    ----------
    pressure : `pint.Quantity`
        Total atmospheric pressure

    reference_pressure : `pint.Quantity`, optional
        The reference pressure against which to calculate the Exner function, defaults to
        metpy.constants.P0

    Returns
    -------
    `pint.Quantity`
        Value of the Exner function at the given pressure

    See Also
    --------
    potential_temperature
    temperature_from_potential_temperature

    """
    return (pressure / reference_pressure).to('dimensionless')**mpconsts.kappa


@exporter.export
@preprocess_and_wrap(wrap_like='temperature', broadcast=('pressure', 'temperature'))
@check_units('[pressure]', '[temperature]')
def potential_temperature(pressure, temperature):
    r"""Calculate the potential temperature.

    Uses the Poisson equation to calculation the potential temperature
    given `pressure` and `temperature`.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Total atmospheric pressure

    temperature : `pint.Quantity`
        Air temperature

    Returns
    -------
    `pint.Quantity`
        Potential temperature corresponding to the temperature and pressure

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
    <Quantity(290.972015, 'kelvin')>

    """
    return temperature / exner_function(pressure)


@exporter.export
@preprocess_and_wrap(
    wrap_like='potential_temperature',
    broadcast=('pressure', 'potential_temperature')
)
@check_units('[pressure]', '[temperature]')
def temperature_from_potential_temperature(pressure, potential_temperature):
    r"""Calculate the temperature from a given potential temperature.

    Uses the inverse of the Poisson equation to calculate the temperature from a
    given potential temperature at a specific pressure level.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Total atmospheric pressure

    potential_temperature : `pint.Quantity`
        Potential temperature

    Returns
    -------
    `pint.Quantity`
        Temperature corresponding to the potential temperature and pressure

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
    >>> T = temperature_from_potential_temperature(p, theta)

    .. versionchanged:: 1.0
       Renamed ``theta`` parameter to ``potential_temperature``

    """
    return potential_temperature * exner_function(pressure)


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'reference_pressure')
)
@check_units('[pressure]', '[temperature]', '[pressure]')
def dry_lapse(pressure, temperature, reference_pressure=None, vertical_dim=0):
    r"""Calculate the temperature at a level assuming only dry processes.

    This function lifts a parcel starting at ``temperature``, conserving
    potential temperature. The starting pressure can be given by ``reference_pressure``.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure level(s) of interest

    temperature : `pint.Quantity`
        Starting temperature

    reference_pressure : `pint.Quantity`, optional
        Reference pressure; if not given, it defaults to the first element of the
        pressure array.

    Returns
    -------
    `pint.Quantity`
       The parcel's resulting temperature at levels given by ``pressure``

    See Also
    --------
    moist_lapse : Calculate parcel temperature assuming liquid saturation processes
    parcel_profile : Calculate complete parcel profile
    potential_temperature

    Notes
    -----
    Only reliably functions on 1D profiles (not higher-dimension vertical cross sections or
    grids) unless reference_pressure is specified.

    .. versionchanged:: 1.0
       Renamed ``ref_pressure`` parameter to ``reference_pressure``

    """
    if reference_pressure is None:
        reference_pressure = pressure[0]
    return temperature * (pressure / reference_pressure)**mpconsts.kappa


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'reference_pressure')
)
@check_units('[pressure]', '[temperature]', '[pressure]')
def moist_lapse(pressure, temperature, reference_pressure=None):
    r"""Calculate the temperature at a level assuming liquid saturation processes.

    This function lifts a parcel starting at `temperature`. The starting pressure can
    be given by `reference_pressure`. Essentially, this function is calculating moist
    pseudo-adiabats.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure level(s) of interest

    temperature : `pint.Quantity`
        Starting temperature

    reference_pressure : `pint.Quantity`, optional
        Reference pressure; if not given, it defaults to the first element of the
        pressure array.

    Returns
    -------
    `pint.Quantity`
       The resulting parcel temperature at levels given by `pressure`

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

    Only reliably functions on 1D profiles (not higher-dimension vertical cross sections or
    grids).

    .. versionchanged:: 1.0
       Renamed ``ref_pressure`` parameter to ``reference_pressure``

    """
    def dt(t, p):
        t = units.Quantity(t, temperature.units)
        p = units.Quantity(p, pressure.units)
        rs = saturation_mixing_ratio(p, t)
        frac = ((mpconsts.Rd * t + mpconsts.Lv * rs)
                / (mpconsts.Cp_d + (mpconsts.Lv * mpconsts.Lv * rs * mpconsts.epsilon
                                    / (mpconsts.Rd * t * t)))).to('kelvin')
        return (frac / p).magnitude

    pressure = np.atleast_1d(pressure)
    if reference_pressure is None:
        reference_pressure = pressure[0]

    if np.isnan(reference_pressure):
        return units.Quantity(np.full(pressure.shape, np.nan), temperature.units)

    pressure = pressure.to('mbar')
    reference_pressure = reference_pressure.to('mbar')
    org_units = temperature.units
    temperature = np.atleast_1d(temperature).to('kelvin')

    side = 'left'

    pres_decreasing = (pressure[0] > pressure[-1])
    if pres_decreasing:
        # Everything is easier if pressures are in increasing order
        pressure = pressure[::-1]
        side = 'right'

    ref_pres_idx = np.searchsorted(pressure.m, reference_pressure.m, side=side)

    ret_temperatures = np.empty((0, temperature.shape[0]))

    if _greater_or_close(reference_pressure, pressure.min()):
        # Integrate downward in pressure
        pres_down = np.append(reference_pressure.m, pressure[(ref_pres_idx - 1)::-1].m)
        trace_down = si.odeint(dt, temperature.m.squeeze(), pres_down.squeeze())
        ret_temperatures = np.concatenate((ret_temperatures, trace_down[:0:-1]))

    if reference_pressure < pressure.max():
        # Integrate upward in pressure
        pres_up = np.append(reference_pressure.m, pressure[ref_pres_idx:].m)
        trace_up = si.odeint(dt, temperature.m.squeeze(), pres_up.squeeze())
        ret_temperatures = np.concatenate((ret_temperatures, trace_up[1:]))

    if pres_decreasing:
        ret_temperatures = ret_temperatures[::-1]

    return units.Quantity(ret_temperatures.T.squeeze(), 'kelvin').to(org_units)


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]')
def lcl(pressure, temperature, dewpoint, max_iters=50, eps=1e-5):
    r"""Calculate the lifted condensation level (LCL) from the starting point.

    The starting state for the parcel is defined by `temperature`, `dewpoint`,
    and `pressure`. If these are arrays, this function will return a LCL
    for every index. This function does work with surface grids as a result.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Starting atmospheric pressure

    temperature : `pint.Quantity`
        Starting temperature

    dewpoint : `pint.Quantity`
        Starting dewpoint

    Returns
    -------
    `pint.Quantity`
        LCL pressure

    `pint.Quantity`
        LCL temperature

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

    1. Find the dewpoint from the LCL pressure and starting mixing ratio
    2. Find the LCL pressure from the starting temperature and dewpoint
    3. Iterate until convergence

    The function is guaranteed to finish by virtue of the `max_iters` counter.

    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``dewpt`` parameter to ``dewpoint``

    """
    def _lcl_iter(p, p0, w, t):
        nonlocal nan_mask
        td = globals()['dewpoint'](vapor_pressure(units.Quantity(p, pressure.units), w))
        p_new = (p0 * (td / t) ** (1. / mpconsts.kappa)).m
        nan_mask = nan_mask | np.isnan(p_new)
        return np.where(np.isnan(p_new), p, p_new)

    # Handle nans by creating a mask that gets set by our _lcl_iter function if it
    # ever encounters a nan, at which point pressure is set to p, stopping iteration.
    nan_mask = False
    w = mixing_ratio(saturation_vapor_pressure(dewpoint), pressure)
    lcl_p = so.fixed_point(_lcl_iter, pressure.m, args=(pressure.m, w, temperature),
                           xtol=eps, maxiter=max_iters)
    lcl_p = np.where(nan_mask, np.nan, lcl_p)

    # np.isclose needed if surface is LCL due to precision error with np.log in dewpoint.
    # Causes issues with parcel_profile_with_lcl if removed. Issue #1187
    lcl_p = units.Quantity(np.where(np.isclose(lcl_p, pressure.m), pressure.m, lcl_p),
                           pressure.units)

    return lcl_p, globals()['dewpoint'](vapor_pressure(lcl_p, w)).to(temperature.units)


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]', '[temperature]')
def lfc(pressure, temperature, dewpoint, parcel_temperature_profile=None, dewpoint_start=None,
        which='top'):
    r"""Calculate the level of free convection (LFC).

    This works by finding the first intersection of the ideal parcel path and
    the measured parcel temperature. If this intersection occurs below the LCL,
    the LFC is determined to be the same as the LCL, based upon the conditions
    set forth in [USAF1990]_, pg 4-14, where a parcel must be lifted dry adiabatically
    to saturation before it can freely rise.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure

    temperature : `pint.Quantity`
        Temperature at the levels given by `pressure`

    dewpoint : `pint.Quantity`
        Dewpoint at the levels given by `pressure`

    parcel_temperature_profile: `pint.Quantity`, optional
        The parcel's temperature profile from which to calculate the LFC. Defaults to the
        surface parcel profile.

    dewpoint_start: `pint.Quantity`, optional
        Dewpoint of the parcel for which to calculate the LFC. Defaults to the surface
        dewpoint.

    which: str, optional
        Pick which LFC to return. Options are 'top', 'bottom', 'wide', 'most_cape', and 'all';
        'top' returns the lowest-pressure LFC (default),
        'bottom' returns the highest-pressure LFC,
        'wide' returns the LFC whose corresponding EL is farthest away,
        'most_cape' returns the LFC that results in the most CAPE in the profile.

    Returns
    -------
    `pint.Quantity`
        LFC pressure, or array of same if which='all'

    `pint.Quantity`
        LFC temperature, or array of same if which='all'

    See Also
    --------
    parcel_profile

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``dewpt``,``dewpoint_start`` parameters to ``dewpoint``, ``dewpoint_start``

    """
    pressure, temperature, dewpoint = _remove_nans(pressure, temperature, dewpoint)
    # Default to surface parcel if no profile or starting pressure level is given
    if parcel_temperature_profile is None:
        new_stuff = parcel_profile_with_lcl(pressure, temperature, dewpoint)
        pressure, temperature, dewpoint, parcel_temperature_profile = new_stuff
        parcel_temperature_profile = parcel_temperature_profile.to(temperature.units)

    if dewpoint_start is None:
        dewpoint_start = dewpoint[0]

    # The parcel profile and data may have the same first data point.
    # If that is the case, ignore that point to get the real first
    # intersection for the LFC calculation. Use logarithmic interpolation.
    if np.isclose(parcel_temperature_profile[0].to(temperature.units).m, temperature[0].m):
        x, y = find_intersections(pressure[1:], parcel_temperature_profile[1:],
                                  temperature[1:], direction='increasing', log_x=True)
    else:
        x, y = find_intersections(pressure, parcel_temperature_profile,
                                  temperature, direction='increasing', log_x=True)

    # Compute LCL for this parcel for future comparisons
    this_lcl = lcl(pressure[0], parcel_temperature_profile[0], dewpoint_start)

    # The LFC could:
    # 1) Not exist
    # 2) Exist but be equal to the LCL
    # 3) Exist and be above the LCL

    # LFC does not exist or is LCL
    if len(x) == 0:
        # Is there any positive area above the LCL?
        mask = pressure < this_lcl[0]
        if np.all(_less_or_close(parcel_temperature_profile[mask], temperature[mask])):
            # LFC doesn't exist
            x = units.Quantity(np.nan, pressure.units)
            y = units.Quantity(np.nan, temperature.units)
        else:  # LFC = LCL
            x, y = this_lcl
        return x, y

    # LFC exists. Make sure it is no lower than the LCL
    else:
        idx = x < this_lcl[0]
        # LFC height < LCL height, so set LFC = LCL
        if not any(idx):
            el_pressure, _ = find_intersections(pressure[1:], parcel_temperature_profile[1:],
                                                temperature[1:], direction='decreasing',
                                                log_x=True)
            if np.min(el_pressure) > this_lcl[0]:
                x = units.Quantity(np.nan, pressure.units)
                y = units.Quantity(np.nan, temperature.units)
            else:
                x, y = this_lcl
            return x, y
        # Otherwise, find all LFCs that exist above the LCL
        # What is returned depends on which flag as described in the docstring
        else:
            return _multiple_el_lfc_options(x, y, idx, which, pressure,
                                            parcel_temperature_profile, temperature,
                                            dewpoint, intersect_type='LFC')


def _multiple_el_lfc_options(intersect_pressures, intersect_temperatures, valid_x,
                             which, pressure, parcel_temperature_profile, temperature,
                             dewpoint, intersect_type):
    """Choose which ELs and LFCs to return from a sounding."""
    p_list, t_list = intersect_pressures[valid_x], intersect_temperatures[valid_x]
    if which == 'all':
        x, y = p_list, t_list
    elif which == 'bottom':
        x, y = p_list[0], t_list[0]
    elif which == 'top':
        x, y = p_list[-1], t_list[-1]
    elif which == 'wide':
        x, y = _wide_option(intersect_type, p_list, t_list, pressure,
                            parcel_temperature_profile, temperature)
    elif which == 'most_cape':
        x, y = _most_cape_option(intersect_type, p_list, t_list, pressure, temperature,
                                 dewpoint, parcel_temperature_profile)
    else:
        raise ValueError('Invalid option for "which". Valid options are "top", "bottom", '
                         '"wide", "most_cape", and "all".')
    return x, y


def _wide_option(intersect_type, p_list, t_list, pressure, parcel_temperature_profile,
                 temperature):
    """Calculate the LFC or EL that produces the greatest distance between these points."""
    # zip the LFC and EL lists together and find greatest difference
    if intersect_type == 'LFC':
        # Find EL intersection pressure values
        lfc_p_list = p_list
        el_p_list, _ = find_intersections(pressure[1:], parcel_temperature_profile[1:],
                                          temperature[1:], direction='decreasing',
                                          log_x=True)
    else:  # intersect_type == 'EL'
        el_p_list = p_list
        # Find LFC intersection pressure values
        lfc_p_list, _ = find_intersections(pressure, parcel_temperature_profile,
                                           temperature, direction='increasing',
                                           log_x=True)
    diff = [lfc_p.m - el_p.m for lfc_p, el_p in zip(lfc_p_list, el_p_list)]
    return (p_list[np.where(diff == np.max(diff))][0],
            t_list[np.where(diff == np.max(diff))][0])


def _most_cape_option(intersect_type, p_list, t_list, pressure, temperature, dewpoint,
                      parcel_temperature_profile):
    """Calculate the LFC or EL that produces the most CAPE in the profile."""
    # Need to loop through all possible combinations of cape, find greatest cape profile
    cape_list, pair_list = [], []
    for which_lfc in ['top', 'bottom']:
        for which_el in ['top', 'bottom']:
            cape, _ = cape_cin(pressure, temperature, dewpoint, parcel_temperature_profile,
                               which_lfc=which_lfc, which_el=which_el)
            cape_list.append(cape.m)
            pair_list.append([which_lfc, which_el])
    (lfc_chosen, el_chosen) = pair_list[np.where(cape_list == np.max(cape_list))[0][0]]
    if intersect_type == 'LFC':
        if lfc_chosen == 'top':
            x, y = p_list[-1], t_list[-1]
        else:  # 'bottom' is returned
            x, y = p_list[0], t_list[0]
    else:  # EL is returned
        if el_chosen == 'top':
            x, y = p_list[-1], t_list[-1]
        else:
            x, y = p_list[0], t_list[0]
    return x, y


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]', '[temperature]')
def el(pressure, temperature, dewpoint, parcel_temperature_profile=None, which='top'):
    r"""Calculate the equilibrium level.

    This works by finding the last intersection of the ideal parcel path and
    the measured environmental temperature. If there is one or fewer intersections, there is
    no equilibrium level.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile

    temperature : `pint.Quantity`
        Temperature at the levels given by `pressure`

    dewpoint : `pint.Quantity`
        Dewpoint at the levels given by `pressure`

    parcel_temperature_profile: `pint.Quantity`, optional
        The parcel's temperature profile from which to calculate the EL. Defaults to the
        surface parcel profile.

    which: str, optional
        Pick which LFC to return. Options are 'top', 'bottom', 'wide', 'most_cape', and 'all'.
        'top' returns the lowest-pressure EL, default.
        'bottom' returns the highest-pressure EL.
        'wide' returns the EL whose corresponding LFC is farthest away.
        'most_cape' returns the EL that results in the most CAPE in the profile.

    Returns
    -------
    `pint.Quantity`
        EL pressure, or array of same if which='all'

    `pint.Quantity`
        EL temperature, or array of same if which='all'

    See Also
    --------
    parcel_profile

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``dewpt`` parameter to ``dewpoint``

    """
    pressure, temperature, dewpoint = _remove_nans(pressure, temperature, dewpoint)
    # Default to surface parcel if no profile or starting pressure level is given
    if parcel_temperature_profile is None:
        new_stuff = parcel_profile_with_lcl(pressure, temperature, dewpoint)
        pressure, temperature, dewpoint, parcel_temperature_profile = new_stuff
        parcel_temperature_profile = parcel_temperature_profile.to(temperature.units)

    # If the top of the sounding parcel is warmer than the environment, there is no EL
    if parcel_temperature_profile[-1] > temperature[-1]:
        return (units.Quantity(np.nan, pressure.units),
                units.Quantity(np.nan, temperature.units))

    # Interpolate in log space to find the appropriate pressure - units have to be stripped
    # and reassigned to allow np.log() to function properly.
    x, y = find_intersections(pressure[1:], parcel_temperature_profile[1:], temperature[1:],
                              direction='decreasing', log_x=True)
    lcl_p, _ = lcl(pressure[0], temperature[0], dewpoint[0])
    idx = x < lcl_p
    if len(x) > 0 and x[-1] < lcl_p:
        return _multiple_el_lfc_options(x, y, idx, which, pressure,
                                        parcel_temperature_profile, temperature, dewpoint,
                                        intersect_type='EL')
    else:
        return (units.Quantity(np.nan, pressure.units),
                units.Quantity(np.nan, temperature.units))


@exporter.export
@preprocess_and_wrap(wrap_like='pressure')
@check_units('[pressure]', '[temperature]', '[temperature]')
def parcel_profile(pressure, temperature, dewpoint):
    r"""Calculate the profile a parcel takes through the atmosphere.

    The parcel starts at `temperature`, and `dewpoint`, lifted up
    dry adiabatically to the LCL, and then moist adiabatically from there.
    `pressure` specifies the pressure levels for the profile.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure level(s) of interest. This array must be from
        high to low pressure.

    temperature : `pint.Quantity`
        Starting temperature

    dewpoint : `pint.Quantity`
        Starting dewpoint

    Returns
    -------
    `pint.Quantity`
        The parcel's temperatures at the specified pressure levels

    See Also
    --------
    lcl, moist_lapse, dry_lapse

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).

    .. versionchanged:: 1.0
       Renamed ``dewpt`` parameter to ``dewpoint``

    """
    _, _, _, t_l, _, t_u = _parcel_profile_helper(pressure, temperature, dewpoint)
    return concatenate((t_l, t_u))


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]')
def parcel_profile_with_lcl(pressure, temperature, dewpoint):
    r"""Calculate the profile a parcel takes through the atmosphere.

    The parcel starts at `temperature`, and `dewpoint`, lifted up
    dry adiabatically to the LCL, and then moist adiabatically from there.
    `pressure` specifies the pressure levels for the profile. This function returns
    a profile that includes the LCL.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure level(s) of interest. This array must be from
        high to low pressure.

    temperature : `pint.Quantity`
        Atmospheric temperature at the levels in `pressure`. The first entry should be at
        the same level as the first `pressure` data point.

    dewpoint : `pint.Quantity`
        Atmospheric dewpoint at the levels in `pressure`. The first entry should be at
        the same level as the first `pressure` data point.

    Returns
    -------
    pressure : `pint.Quantity`
        The parcel profile pressures, which includes the specified levels and the LCL

    ambient_temperature : `pint.Quantity`
        Atmospheric temperature values, including the value interpolated to the LCL level

    ambient_dew_point : `pint.Quantity`
        Atmospheric dewpoint values, including the value interpolated to the LCL level

    profile_temperature : `pint.Quantity`
        The parcel profile temperatures at all of the levels in the returned pressures array,
        including the LCL

    See Also
    --------
    lcl, moist_lapse, dry_lapse, parcel_profile, parcel_profile_with_lcl_as_dataset

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Also, will only return Pint Quantities, even when given xarray DataArray profiles. To
    obtain a xarray Dataset instead, use `parcel_profile_with_lcl_as_dataset` instead.

    .. versionchanged:: 1.0
       Renamed ``dewpt`` parameter to ``dewpoint``

    """
    p_l, p_lcl, p_u, t_l, t_lcl, t_u = _parcel_profile_helper(pressure, temperature[0],
                                                              dewpoint[0])
    new_press = concatenate((p_l, p_lcl, p_u))
    prof_temp = concatenate((t_l, t_lcl, t_u))
    new_temp = _insert_lcl_level(pressure, temperature, p_lcl)
    new_dewp = _insert_lcl_level(pressure, dewpoint, p_lcl)
    return new_press, new_temp, new_dewp, prof_temp


@exporter.export
def parcel_profile_with_lcl_as_dataset(pressure, temperature, dewpoint):
    r"""Calculate the profile a parcel takes through the atmosphere, returning a Dataset.

    The parcel starts at `temperature`, and `dewpoint`, lifted up
    dry adiabatically to the LCL, and then moist adiabatically from there.
    `pressure` specifies the pressure levels for the profile. This function returns
    a profile that includes the LCL.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure level(s) of interest. This array must be from
        high to low pressure.
    temperature : `pint.Quantity`
        The atmospheric temperature at the levels in `pressure`. The first entry should be at
        the same level as the first `pressure` data point.
    dewpoint : `pint.Quantity`
        The atmospheric dewpoint at the levels in `pressure`. The first entry should be at
        the same level as the first `pressure` data point.

    Returns
    -------
    profile : `xarray.Dataset`
        The interpolated profile with three data variables: ambient_temperature,
        ambient_dew_point, and profile_temperature, all of which are on an isobaric
        coordinate.

    See Also
    --------
    lcl, moist_lapse, dry_lapse, parcel_profile, parcel_profile_with_lcl

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).

    """
    p, ambient_temperature, ambient_dew_point, profile_temperature = parcel_profile_with_lcl(
        pressure,
        temperature,
        dewpoint
    )
    return xr.Dataset(
        {
            'ambient_temperature': (
                ('isobaric',),
                ambient_temperature,
                {'standard_name': 'air_temperature'}
            ),
            'ambient_dew_point': (
                ('isobaric',),
                ambient_dew_point,
                {'standard_name': 'dew_point_temperature'}
            ),
            'parcel_temperature': (
                ('isobaric',),
                profile_temperature,
                {'long_name': 'air_temperature_of_lifted_parcel'}
            )
        },
        coords={
            'isobaric': (
                'isobaric',
                p.m,
                {'units': str(p.units), 'standard_name': 'air_pressure'}
            )
        }
    )


def _check_pressure(pressure):
    """Check that pressure decreases monotonically.

    Returns True if the pressure decreases monotonically; otherwise, returns False.
    """
    return np.all(pressure[:-1] > pressure[1:])


def _parcel_profile_helper(pressure, temperature, dewpoint):
    """Help calculate parcel profiles.

    Returns the temperature and pressure, above, below, and including the LCL. The
    other calculation functions decide what to do with the pieces.

    """
    # Check that pressure is monotonic decreasing.
    if not _check_pressure(pressure):
        msg = """
        Pressure does not decrease monotonically in your sounding.
        Using scipy.signal.medfilt may fix this."""
        raise InvalidSoundingError(msg)

    # Find the LCL
    press_lcl, temp_lcl = lcl(pressure[0], temperature, dewpoint)
    press_lcl = press_lcl.to(pressure.units)

    # Find the dry adiabatic profile, *including* the LCL. We need >= the LCL in case the
    # LCL is included in the levels. It's slightly redundant in that case, but simplifies
    # the logic for removing it later.
    press_lower = concatenate((pressure[pressure >= press_lcl], press_lcl))
    temp_lower = dry_lapse(press_lower, temperature)

    # If the pressure profile doesn't make it to the lcl, we can stop here
    if _greater_or_close(np.nanmin(pressure), press_lcl):
        return (press_lower[:-1], press_lcl, units.Quantity(np.array([]), press_lower.units),
                temp_lower[:-1], temp_lcl, units.Quantity(np.array([]), temp_lower.units))

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
    return units.Quantity(np.insert(temperature.m, loc, interp_temp.m), temperature.units)


@exporter.export
@preprocess_and_wrap(wrap_like='mixing_ratio', broadcast=('pressure', 'mixing_ratio'))
@check_units('[pressure]', '[dimensionless]')
def vapor_pressure(pressure, mixing_ratio):
    r"""Calculate water vapor (partial) pressure.

    Given total ``pressure`` and water vapor ``mixing_ratio``, calculates the
    partial pressure of water vapor.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Total atmospheric pressure

    mixing_ratio : `pint.Quantity`
        Dimensionless mass mixing ratio

    Returns
    -------
    `pint.Quantity`
        Ambient water vapor (partial) pressure in the same units as ``pressure``

    Notes
    -----
    This function is a straightforward implementation of the equation given in many places,
    such as [Hobbs1977]_ pg.71:

    .. math:: e = p \frac{r}{r + \epsilon}

    .. versionchanged:: 1.0
       Renamed ``mixing`` parameter to ``mixing_ratio``

    See Also
    --------
    saturation_vapor_pressure, dewpoint

    """
    return pressure * mixing_ratio / (mpconsts.epsilon + mixing_ratio)


@exporter.export
@preprocess_and_wrap(wrap_like='temperature')
@check_units('[temperature]')
def saturation_vapor_pressure(temperature):
    r"""Calculate the saturation water vapor (partial) pressure.

    Parameters
    ----------
    temperature : `pint.Quantity`
        Air temperature

    Returns
    -------
    `pint.Quantity`
        Saturation water vapor (partial) pressure

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
    return sat_pressure_0c * np.exp(17.67 * (temperature - units.Quantity(273.15, 'kelvin'))
                                    / (temperature - units.Quantity(29.65, 'kelvin')))


@exporter.export
@preprocess_and_wrap(wrap_like='temperature', broadcast=('temperature', 'relative_humidity'))
@check_units('[temperature]', '[dimensionless]')
def dewpoint_from_relative_humidity(temperature, relative_humidity):
    r"""Calculate the ambient dewpoint given air temperature and relative humidity.

    Parameters
    ----------
    temperature : `pint.Quantity`
        Air temperature

    relative_humidity : `pint.Quantity`
        Relative humidity expressed as a ratio in the range 0 < relative_humidity <= 1

    Returns
    -------
    `pint.Quantity`
        Dewpoint temperature


    .. versionchanged:: 1.0
       Renamed ``rh`` parameter to ``relative_humidity``

    See Also
    --------
    dewpoint, saturation_vapor_pressure

    """
    if np.any(relative_humidity > 1.2):
        warnings.warn('Relative humidity >120%, ensure proper units.')
    return dewpoint(relative_humidity * saturation_vapor_pressure(temperature))


@exporter.export
@preprocess_and_wrap(wrap_like='vapor_pressure')
@check_units('[pressure]')
def dewpoint(vapor_pressure):
    r"""Calculate the ambient dewpoint given the vapor pressure.

    Parameters
    ----------
    e : `pint.Quantity`
        Water vapor partial pressure

    Returns
    -------
    `pint.Quantity`
        Dewpoint temperature

    See Also
    --------
    dewpoint_from_relative_humidity, saturation_vapor_pressure, vapor_pressure

    Notes
    -----
    This function inverts the [Bolton1980]_ formula for saturation vapor
    pressure to instead calculate the temperature. This yield the following
    formula for dewpoint in degrees Celsius:

    .. math:: T = \frac{243.5 log(e / 6.112)}{17.67 - log(e / 6.112)}

    .. versionchanged:: 1.0
       Renamed ``e`` parameter to ``vapor_pressure``

    """
    val = np.log(vapor_pressure / sat_pressure_0c)
    return (units.Quantity(0., 'degC')
            + units.Quantity(243.5, 'delta_degC') * val / (17.67 - val))


@exporter.export
@preprocess_and_wrap(wrap_like='partial_press', broadcast=('partial_press', 'total_press'))
@check_units('[pressure]', '[pressure]', '[dimensionless]')
def mixing_ratio(partial_press, total_press, molecular_weight_ratio=mpconsts.epsilon):
    r"""Calculate the mixing ratio of a gas.

    This calculates mixing ratio given its partial pressure and the total pressure of
    the air. There are no required units for the input arrays, other than that
    they have the same units.

    Parameters
    ----------
    partial_press : `pint.Quantity`
        Partial pressure of the constituent gas

    total_press : `pint.Quantity`
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

    .. versionchanged:: 1.0
       Renamed ``part_press``, ``tot_press`` parameters to ``partial_press``, ``total_press``

    See Also
    --------
    saturation_mixing_ratio, vapor_pressure

    """
    return (molecular_weight_ratio * partial_press
            / (total_press - partial_press)).to('dimensionless')


@exporter.export
@preprocess_and_wrap(wrap_like='temperature', broadcast=('total_press', 'temperature'))
@check_units('[pressure]', '[temperature]')
def saturation_mixing_ratio(total_press, temperature):
    r"""Calculate the saturation mixing ratio of water vapor.

    This calculation is given total atmospheric pressure and air temperature.

    Parameters
    ----------
    total_press: `pint.Quantity`
        Total atmospheric pressure

    temperature: `pint.Quantity`
        Air temperature

    Returns
    -------
    `pint.Quantity`
        Saturation mixing ratio, dimensionless

    Notes
    -----
    This function is a straightforward implementation of the equation given in many places,
    such as [Hobbs1977]_ pg.73:

    .. math:: r_s = \epsilon \frac{e_s}{p - e_s}

    .. versionchanged:: 1.0
       Renamed ``tot_press`` parameter to ``total_press``

    """
    return mixing_ratio(saturation_vapor_pressure(temperature), total_press)


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'dewpoint')
)
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
        Equivalent potential temperature of the parcel

    Notes
    -----
    [Bolton1980]_ formula for Theta-e is used, since according to
    [DaviesJones2009]_ it is the most accurate non-iterative formulation
    available.

    """
    t = temperature.to('kelvin').magnitude
    td = dewpoint.to('kelvin').magnitude
    r = saturation_mixing_ratio(pressure, dewpoint).magnitude
    e = saturation_vapor_pressure(dewpoint)

    t_l = 56 + 1. / (1. / (td - 56) + np.log(t / td) / 800.)
    th_l = potential_temperature(pressure - e, temperature) * (t / t_l) ** (0.28 * r)
    return th_l * np.exp(r * (1 + 0.448 * r) * (3036. / t_l - 1.78))


@exporter.export
@preprocess_and_wrap(wrap_like='temperature', broadcast=('pressure', 'temperature'))
@check_units('[pressure]', '[temperature]')
def saturation_equivalent_potential_temperature(pressure, temperature):
    r"""Calculate saturation equivalent potential temperature.

    This calculation must be given an air parcel's pressure and temperature.
    The implementation uses the formula outlined in [Bolton1980]_ for the
    equivalent potential temperature, and assumes a saturated process.

    First, because we assume a saturated process, the temperature at the LCL is
    equivalent to the current temperature. Therefore the following equation.

    .. math:: T_{L}=\frac{1}{\frac{1}{T_{D}-56}+\frac{ln(T_{K}/T_{D})}{800}}+56

    reduces to:

    .. math:: T_{L} = T_{K}

    Then the potential temperature at the temperature/LCL is calculated:

    .. math:: \theta_{DL}=T_{K}\left(\frac{1000}{p-e}\right)^k
              \left(\frac{T_{K}}{T_{L}}\right)^{.28r}

    However, because:

    .. math:: T_{L} = T_{K}

    it follows that:

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
        Saturation equivalent potential temperature of the parcel

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

    return units.Quantity(th_es, units.kelvin)


@exporter.export
@preprocess_and_wrap(wrap_like='temperature', broadcast=('temperature', 'mixing_ratio'))
@check_units('[temperature]', '[dimensionless]', '[dimensionless]')
def virtual_temperature(temperature, mixing_ratio, molecular_weight_ratio=mpconsts.epsilon):
    r"""Calculate virtual temperature.

    This calculation must be given an air parcel's temperature and mixing ratio.
    The implementation uses the formula outlined in [Hobbs2006]_ pg.80.

    Parameters
    ----------
    temperature: `pint.Quantity`
        Air temperature

    mixing_ratio : `pint.Quantity`
        Mass mixing ratio (dimensionless)

    molecular_weight_ratio : `pint.Quantity` or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air.
        (:math:`\epsilon\approx0.622`)

    Returns
    -------
    `pint.Quantity`
        Corresponding virtual temperature of the parcel

    Notes
    -----
    .. math:: T_v = T \frac{\text{w} + \epsilon}{\epsilon\,(1 + \text{w})}

    .. versionchanged:: 1.0
       Renamed ``mixing`` parameter to ``mixing_ratio``

    """
    return temperature * ((mixing_ratio + molecular_weight_ratio)
                          / (molecular_weight_ratio * (1 + mixing_ratio)))


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'mixing_ratio')
)
@check_units('[pressure]', '[temperature]', '[dimensionless]', '[dimensionless]')
def virtual_potential_temperature(pressure, temperature, mixing_ratio,
                                  molecular_weight_ratio=mpconsts.epsilon):
    r"""Calculate virtual potential temperature.

    This calculation must be given an air parcel's pressure, temperature, and mixing ratio.
    The implementation uses the formula outlined in [Markowski2010]_ pg.13.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure

    temperature: `pint.Quantity`
        Air temperature

    mixing_ratio : `pint.Quantity`
        Dimensionless mass mixing ratio

    molecular_weight_ratio : `pint.Quantity` or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air.
        (:math:`\epsilon\approx0.622`)

    Returns
    -------
    `pint.Quantity`
        Corresponding virtual potential temperature of the parcel

    Notes
    -----
    .. math:: \Theta_v = \Theta \frac{\text{w} + \epsilon}{\epsilon\,(1 + \text{w})}

    .. versionchanged:: 1.0
       Renamed ``mixing`` parameter to ``mixing_ratio``

    """
    pottemp = potential_temperature(pressure, temperature)
    return virtual_temperature(pottemp, mixing_ratio, molecular_weight_ratio)


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'mixing_ratio')
)
@check_units('[pressure]', '[temperature]', '[dimensionless]', '[dimensionless]')
def density(pressure, temperature, mixing_ratio, molecular_weight_ratio=mpconsts.epsilon):
    r"""Calculate density.

    This calculation must be given an air parcel's pressure, temperature, and mixing ratio.
    The implementation uses the formula outlined in [Hobbs2006]_ pg.67.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure

    temperature: `pint.Quantity`
        Air temperature

    mixing_ratio : `pint.Quantity`
        Mass mixing ratio (dimensionless)

    molecular_weight_ratio : `pint.Quantity` or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air.
        (:math:`\epsilon\approx0.622`)

    Returns
    -------
    `pint.Quantity`
        Corresponding density of the parcel

    Notes
    -----
    .. math:: \rho = \frac{p}{R_dT_v}

    .. versionchanged:: 1.0
       Renamed ``mixing`` parameter to ``mixing_ratio``

    """
    virttemp = virtual_temperature(temperature, mixing_ratio, molecular_weight_ratio)
    return (pressure / (mpconsts.Rd * virttemp)).to('kg/m**3')


@exporter.export
@preprocess_and_wrap(
    wrap_like='dry_bulb_temperature',
    broadcast=('pressure', 'dry_bulb_temperature', 'wet_bulb_temperature')
)
@check_units('[pressure]', '[temperature]', '[temperature]')
def relative_humidity_wet_psychrometric(pressure, dry_bulb_temperature, wet_bulb_temperature,
                                        **kwargs):
    r"""Calculate the relative humidity with wet bulb and dry bulb temperatures.

    This uses a psychrometric relationship as outlined in [WMO8]_, with
    coefficients from [Fan1987]_.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure

    dry_bulb_temperature: `pint.Quantity`
        Dry bulb temperature

    wet_bulb_temperature: `pint.Quantity`
        Wet bulb temperature

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

    .. versionchanged:: 1.0
       Changed signature from
       ``(dry_bulb_temperature, web_bulb_temperature, pressure, **kwargs)``

    See Also
    --------
    psychrometric_vapor_pressure_wet, saturation_vapor_pressure

    """
    return (psychrometric_vapor_pressure_wet(pressure, dry_bulb_temperature,
                                             wet_bulb_temperature, **kwargs)
            / saturation_vapor_pressure(dry_bulb_temperature))


@exporter.export
@preprocess_and_wrap(
    wrap_like='dry_bulb_temperature',
    broadcast=('pressure', 'dry_bulb_temperature', 'wet_bulb_temperature')
)
@check_units('[pressure]', '[temperature]', '[temperature]')
def psychrometric_vapor_pressure_wet(pressure, dry_bulb_temperature, wet_bulb_temperature,
                                     psychrometer_coefficient=None):
    r"""Calculate the vapor pressure with wet bulb and dry bulb temperatures.

    This uses a psychrometric relationship as outlined in [WMO8]_, with
    coefficients from [Fan1987]_.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure

    dry_bulb_temperature: `pint.Quantity`
        Dry bulb temperature

    wet_bulb_temperature: `pint.Quantity`
        Wet bulb temperature

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

    .. versionchanged:: 1.0
       Changed signature from
       ``(dry_bulb_temperature, wet_bulb_temperature, pressure, psychrometer_coefficient)``

    See Also
    --------
    saturation_vapor_pressure

    """
    if psychrometer_coefficient is None:
        psychrometer_coefficient = units.Quantity(6.21e-4, '1/K')
    return (saturation_vapor_pressure(wet_bulb_temperature) - psychrometer_coefficient
            * pressure * (dry_bulb_temperature - wet_bulb_temperature).to('kelvin'))


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'relative_humidity')
)
@check_units('[pressure]', '[temperature]', '[dimensionless]')
def mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity):
    r"""Calculate the mixing ratio from relative humidity, temperature, and pressure.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure

    temperature: `pint.Quantity`
        Air temperature

    relative_humidity: array_like
        The relative humidity expressed as a unitless ratio in the range [0, 1]. Can also pass
        a percentage if proper units are attached.

    Returns
    -------
    `pint.Quantity`
        Mixing ratio (dimensionless)

    Notes
    -----
    Formula adapted from [Hobbs1977]_ pg. 74.

    .. math:: w = (rh)(w_s)

    * :math:`w` is mixing ratio
    * :math:`rh` is relative humidity as a unitless ratio
    * :math:`w_s` is the saturation mixing ratio

    .. versionchanged:: 1.0
       Changed signature from ``(relative_humidity, temperature, pressure)``

    See Also
    --------
    relative_humidity_from_mixing_ratio, saturation_mixing_ratio

    """
    return (relative_humidity
            * saturation_mixing_ratio(pressure, temperature)).to('dimensionless')


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'mixing_ratio')
)
@check_units('[pressure]', '[temperature]', '[dimensionless]')
def relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio):
    r"""Calculate the relative humidity from mixing ratio, temperature, and pressure.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure

    temperature: `pint.Quantity`
        Air temperature

    mixing_ratio: `pint.Quantity`
        Dimensionless mass mixing ratio

    Returns
    -------
    `pint.Quantity`
        Relative humidity

    Notes
    -----
    Formula based on that from [Hobbs1977]_ pg. 74.

    .. math:: rh = \frac{w}{w_s}

    * :math:`rh` is relative humidity as a unitless ratio
    * :math:`w` is mixing ratio
    * :math:`w_s` is the saturation mixing ratio

    .. versionchanged:: 1.0
       Changed signature from ``(mixing_ratio, temperature, pressure)``

    See Also
    --------
    mixing_ratio_from_relative_humidity, saturation_mixing_ratio

    """
    return mixing_ratio / saturation_mixing_ratio(pressure, temperature)


@exporter.export
@preprocess_and_wrap(wrap_like='specific_humidity')
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
    with contextlib.suppress(AttributeError):
        specific_humidity = specific_humidity.to('dimensionless')
    return specific_humidity / (1 - specific_humidity)


@exporter.export
@preprocess_and_wrap(wrap_like='mixing_ratio')
@check_units('[dimensionless]')
def specific_humidity_from_mixing_ratio(mixing_ratio):
    r"""Calculate the specific humidity from the mixing ratio.

    Parameters
    ----------
    mixing_ratio: `pint.Quantity`
        Mixing ratio

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
    with contextlib.suppress(AttributeError):
        mixing_ratio = mixing_ratio.to('dimensionless')
    return mixing_ratio / (1 + mixing_ratio)


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'specific_humidity')
)
@check_units('[pressure]', '[temperature]', '[dimensionless]')
def relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity):
    r"""Calculate the relative humidity from specific humidity, temperature, and pressure.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure

    temperature: `pint.Quantity`
        Air temperature

    specific_humidity: `pint.Quantity`
        Specific humidity of air

    Returns
    -------
    `pint.Quantity`
        Relative humidity

    Notes
    -----
    Formula based on that from [Hobbs1977]_ pg. 74. and [Salby1996]_ pg. 118.

    .. math:: relative_humidity = \frac{q}{(1-q)w_s}

    * :math:`relative_humidity` is relative humidity as a unitless ratio
    * :math:`q` is specific humidity
    * :math:`w_s` is the saturation mixing ratio

    .. versionchanged:: 1.0
       Changed signature from ``(specific_humidity, temperature, pressure)``

    See Also
    --------
    relative_humidity_from_mixing_ratio

    """
    return (mixing_ratio_from_specific_humidity(specific_humidity)
            / saturation_mixing_ratio(pressure, temperature))


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]', '[temperature]')
def cape_cin(pressure, temperature, dewpoint, parcel_profile, which_lfc='bottom',
             which_el='top'):
    r"""Calculate CAPE and CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile and parcel path. CIN is integrated between the surface and
    LFC, CAPE is integrated between the LFC and EL (or top of sounding). Intersection points
    of the measured temperature profile and parcel profile are logarithmically interpolated.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure level(s) of interest, in order from highest to
        lowest pressure

    temperature : `pint.Quantity`
        Atmospheric temperature corresponding to pressure

    dewpoint : `pint.Quantity`
        Atmospheric dewpoint corresponding to pressure

    parcel_profile : `pint.Quantity`
        Temperature profile of the parcel

    which_lfc : str
        Choose which LFC to integrate from. Valid options are 'top', 'bottom', 'wide',
        and 'most_cape'. Default is 'bottom'.

    which_el : str
        Choose which EL to integrate to. Valid options are 'top', 'bottom', 'wide',
        and 'most_cape'. Default is 'top'.

    Returns
    -------
    `pint.Quantity`
        Convective Available Potential Energy (CAPE)

    `pint.Quantity`
        Convective Inhibition (CIN)

    Notes
    -----
    Formula adopted from [Hobbs1977]_.

    .. math:: \text{CAPE} = -R_d \int_{LFC}^{EL} (T_{parcel} - T_{env}) d\text{ln}(p)

    .. math:: \text{CIN} = -R_d \int_{SFC}^{LFC} (T_{parcel} - T_{env}) d\text{ln}(p)


    * :math:`CAPE` is convective available potential energy
    * :math:`CIN` is convective inhibition
    * :math:`LFC` is pressure of the level of free convection
    * :math:`EL` is pressure of the equilibrium level
    * :math:`SFC` is the level of the surface or beginning of parcel path
    * :math:`R_d` is the gas constant
    * :math:`g` is gravitational acceleration
    * :math:`T_{parcel}` is the parcel temperature
    * :math:`T_{env}` is environment temperature
    * :math:`p` is atmospheric pressure

    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``dewpt`` parameter to ``dewpoint``

    See Also
    --------
    lfc, el

    """
    pressure, temperature, dewpoint, parcel_profile = _remove_nans(pressure, temperature,
                                                                   dewpoint, parcel_profile)
    # Calculate LFC limit of integration
    lfc_pressure, _ = lfc(pressure, temperature, dewpoint,
                          parcel_temperature_profile=parcel_profile, which=which_lfc)

    # If there is no LFC, no need to proceed.
    if np.isnan(lfc_pressure):
        return units.Quantity(0, 'J/kg'), units.Quantity(0, 'J/kg')
    else:
        lfc_pressure = lfc_pressure.magnitude

    # Calculate the EL limit of integration
    el_pressure, _ = el(pressure, temperature, dewpoint,
                        parcel_temperature_profile=parcel_profile, which=which_el)

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
    p_mask = _less_or_close(x.m, lfc_pressure) & _greater_or_close(x.m, el_pressure)
    x_clipped = x[p_mask].magnitude
    y_clipped = y[p_mask].magnitude
    cape = (mpconsts.Rd
            * units.Quantity(np.trapz(y_clipped, np.log(x_clipped)), 'K')).to(units('J/kg'))

    # CIN
    # Only use data between the surface and LFC for calculation
    p_mask = _greater_or_close(x.m, lfc_pressure)
    x_clipped = x[p_mask].magnitude
    y_clipped = y[p_mask].magnitude
    cin = (mpconsts.Rd
           * units.Quantity(np.trapz(y_clipped, np.log(x_clipped)), 'K')).to(units('J/kg'))

    # Set CIN to 0 if it's returned as a positive value (#1190)
    if cin > units.Quantity(0, 'J/kg'):
        cin = units.Quantity(0, 'J/kg')
    return cape, cin


def _find_append_zero_crossings(x, y):
    r"""
    Find and interpolate zero crossings.

    Estimate the zero crossings of an x,y series and add estimated crossings to series,
    returning a sorted array with no duplicate values.

    Parameters
    ----------
    x : `pint.Quantity`
        X values of data

    y : `pint.Quantity`
        Y values of data

    Returns
    -------
    x : `pint.Quantity`
        X values of data
    y : `pint.Quantity`
        Y values of data

    """
    crossings = find_intersections(x[1:], y[1:],
                                   units.Quantity(np.zeros_like(y[1:]), y.units), log_x=True)
    x = concatenate((x, crossings[0]))
    y = concatenate((y, crossings[1]))

    # Resort so that data are in order
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Remove duplicate data points if there are any
    keep_idx = np.ediff1d(x.magnitude, to_end=[1]) > 1e-6
    x = x[keep_idx]
    y = y[keep_idx]
    return x, y


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]')
def most_unstable_parcel(pressure, temperature, dewpoint, height=None, bottom=None,
                         depth=None):
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

    height: `pint.Quantity`, optional
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
        Pressure, temperature, and dewpoint of most unstable parcel in the profile

    integer
        Index of the most unstable parcel in the given profile

    See Also
    --------
    get_layer

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height``

    """
    if depth is None:
        depth = units.Quantity(300, 'hPa')
    p_layer, t_layer, td_layer = get_layer(pressure, temperature, dewpoint, bottom=bottom,
                                           depth=depth, height=height, interpolate=False)
    theta_e = equivalent_potential_temperature(p_layer, t_layer, td_layer)
    max_idx = np.argmax(theta_e)
    return p_layer[max_idx], t_layer[max_idx], td_layer[max_idx], max_idx


@exporter.export
@add_vertical_dim_from_xarray
@preprocess_and_wrap()
@check_units('[temperature]', '[pressure]', '[temperature]')
def isentropic_interpolation(levels, pressure, temperature, *args, vertical_dim=0,
                             temperature_out=False, max_iters=50, eps=1e-6,
                             bottom_up_search=True, **kwargs):
    r"""Interpolate data in isobaric coordinates to isentropic coordinates.

    Parameters
    ----------
    levels : array
        One-dimensional array of desired potential temperature surfaces

    pressure : array
        One-dimensional array of pressure levels

    temperature : array
        Array of temperature
    vertical_dim : int, optional
        The axis corresponding to the vertical in the temperature array, defaults to 0.

    temperature_out : bool, optional
        If true, will calculate temperature and output as the last item in the output list.
        Defaults to False.

    max_iters : int, optional
        Maximum number of iterations to use in calculation, defaults to 50.

    eps : float, optional
        The desired absolute error in the calculated value, defaults to 1e-6.

    bottom_up_search : bool, optional
        Controls whether to search for levels bottom-up, or top-down. Defaults to
        True, which is bottom-up search.

    args : array, optional
        Any additional variables will be interpolated to each isentropic level

    Returns
    -------
    list
        List with pressure at each isentropic level, followed by each additional
        argument interpolated to isentropic coordinates.

    Notes
    -----
    Input variable arrays must have the same number of vertical levels as the pressure levels
    array. Pressure is calculated on isentropic surfaces by assuming that temperature varies
    linearly with the natural log of pressure. Linear interpolation is then used in the
    vertical to find the pressure at each isentropic level. Interpolation method from
    [Ziv1994]_. Any additional arguments are assumed to vary linearly with temperature and will
    be linearly interpolated to the new isentropic levels.

    Will only return Pint Quantities, even when given xarray DataArray profiles. To
    obtain a xarray Dataset instead, use `isentropic_interpolation_as_dataset` instead.

    .. versionchanged:: 1.0
       Renamed ``theta_levels``, ``axis`` parameters to ``levels``, ``vertical_dim``

    See Also
    --------
    potential_temperature, isentropic_interpolation_as_dataset

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

    # Get dimensions in temperature
    ndim = temperature.ndim

    # Convert units
    pressure = pressure.to('hPa')
    temperature = temperature.to('kelvin')

    slices = [np.newaxis] * ndim
    slices[vertical_dim] = slice(None)
    slices = tuple(slices)
    pressure = units.Quantity(np.broadcast_to(pressure[slices].magnitude, temperature.shape),
                              pressure.units)

    # Sort input data
    sort_pressure = np.argsort(pressure.m, axis=vertical_dim)
    sort_pressure = np.swapaxes(np.swapaxes(sort_pressure, 0, vertical_dim)[::-1], 0,
                                vertical_dim)
    sorter = broadcast_indices(pressure, sort_pressure, ndim, vertical_dim)
    levs = pressure[sorter]
    tmpk = temperature[sorter]

    levels = np.asarray(levels.m_as('kelvin')).reshape(-1)
    isentlevels = levels[np.argsort(levels)]

    # Make the desired isentropic levels the same shape as temperature
    shape = list(temperature.shape)
    shape[vertical_dim] = isentlevels.size
    isentlevs_nd = np.broadcast_to(isentlevels[slices], shape)

    # exponent to Poisson's Equation, which is imported above
    ka = mpconsts.kappa.m_as('dimensionless')

    # calculate theta for each point
    pres_theta = potential_temperature(levs, tmpk)

    # Raise error if input theta level is larger than pres_theta max
    if np.max(pres_theta.m) < np.max(levels):
        raise ValueError('Input theta level out of data bounds')

    # Find log of pressure to implement assumption of linear temperature dependence on
    # ln(p)
    log_p = np.log(levs.m)

    # Calculations for interpolation routine
    pok = mpconsts.P0 ** ka

    # index values for each point for the pressure level nearest to the desired theta level
    above, below, good = find_bounding_indices(pres_theta.m, levels, vertical_dim,
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
    isentprs[~(good & _less_or_close(isentprs, np.max(pressure.m)))] = np.nan

    # create list for storing output data
    ret = [units.Quantity(isentprs, 'hPa')]

    # if temperature_out = true, calculate temperature and output as last item in list
    if temperature_out:
        ret.append(units.Quantity((isentlevs_nd / ((mpconsts.P0.m / isentprs) ** ka)), 'K'))

    # do an interpolation for each additional argument
    if args:
        others = interpolate_1d(isentlevels, pres_theta.m, *(arr[sorter] for arr in args),
                                axis=vertical_dim, return_list_always=True)
        ret.extend(others)

    return ret


@exporter.export
def isentropic_interpolation_as_dataset(
    levels,
    temperature,
    *args,
    max_iters=50,
    eps=1e-6,
    bottom_up_search=True
):
    r"""Interpolate xarray data in isobaric coords to isentropic coords, returning a Dataset.

    Parameters
    ----------
    levels : `pint.Quantity`
        One-dimensional array of desired potential temperature surfaces
    temperature : `xarray.DataArray`
        Array of temperature
    args : `xarray.DataArray`, optional
        Any other given variables will be interpolated to each isentropic level. Must have
        names in order to have a well-formed output Dataset.
    max_iters : int, optional
        The maximum number of iterations to use in calculation, defaults to 50.
    eps : float, optional
        The desired absolute error in the calculated value, defaults to 1e-6.
    bottom_up_search : bool, optional
        Controls whether to search for levels bottom-up, or top-down. Defaults to
        True, which is bottom-up search.

    Returns
    -------
    xarray.Dataset
        Dataset with pressure, temperature, and each additional argument, all on the specified
        isentropic coordinates.

    Notes
    -----
    Input variable arrays must have the same number of vertical levels as the pressure levels
    array. Pressure is calculated on isentropic surfaces by assuming that temperature varies
    linearly with the natural log of pressure. Linear interpolation is then used in the
    vertical to find the pressure at each isentropic level. Interpolation method from
    [Ziv1994]_. Any additional arguments are assumed to vary linearly with temperature and will
    be linearly interpolated to the new isentropic levels.

    This formulation relies upon xarray functionality. If using Pint Quantities, use
    `isentropic_interpolation` instead.

    See Also
    --------
    potential_temperature, isentropic_interpolation

    """
    # Ensure matching coordinates by broadcasting
    all_args = xr.broadcast(temperature, *args)

    # Obtain result as list of Quantities
    ret = isentropic_interpolation(
        levels,
        all_args[0].metpy.vertical,
        all_args[0].metpy.unit_array,
        *(arg.metpy.unit_array for arg in all_args[1:]),
        vertical_dim=all_args[0].metpy.find_axis_number('vertical'),
        temperature_out=True,
        max_iters=max_iters,
        eps=eps,
        bottom_up_search=bottom_up_search
    )

    # Reconstruct coordinates and dims (add isentropic levels, remove isobaric levels)
    vertical_dim = all_args[0].metpy.find_axis_name('vertical')
    new_coords = {
        'isentropic_level': xr.DataArray(
            levels.m,
            dims=('isentropic_level',),
            coords={'isentropic_level': levels.m},
            name='isentropic_level',
            attrs={
                'units': str(levels.units),
                'positive': 'up'
            }
        ),
        **{
            key: value
            for key, value in all_args[0].coords.items()
            if key != vertical_dim
        }
    }
    new_dims = [
        dim if dim != vertical_dim else 'isentropic_level' for dim in all_args[0].dims
    ]

    # Build final dataset from interpolated Quantities and original DataArrays
    return xr.Dataset(
        {
            'pressure': (
                new_dims,
                ret[0],
                {'standard_name': 'air_pressure'}
            ),
            'temperature': (
                new_dims,
                ret[1],
                {'standard_name': 'air_temperature'}
            ),
            **{
                all_args[i].name: (new_dims, ret[i + 1], all_args[i].attrs)
                for i in range(1, len(all_args))
            }
        },
        coords=new_coords
    )


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]')
def surface_based_cape_cin(pressure, temperature, dewpoint):
    r"""Calculate surface-based CAPE and CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile for a surface-based parcel. CIN is integrated
    between the surface and LFC, CAPE is integrated between the LFC and EL (or top of
    sounding). Intersection points of the measured temperature profile and parcel profile are
    logarithmically interpolated.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile. The first entry should be the starting
        (surface) observation, with the array going from high to low pressure.

    temperature : `pint.Quantity`
        Temperature profile corresponding to the `pressure` profile

    dewpoint : `pint.Quantity`
        Dewpoint profile corresponding to the `pressure` profile

    Returns
    -------
    `pint.Quantity`
        Surface based Convective Available Potential Energy (CAPE)

    `pint.Quantity`
        Surface based Convective Inhibition (CIN)

    See Also
    --------
    cape_cin, parcel_profile

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    """
    pressure, temperature, dewpoint = _remove_nans(pressure, temperature, dewpoint)
    p, t, td, profile = parcel_profile_with_lcl(pressure, temperature, dewpoint)
    return cape_cin(p, t, td, profile)


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]')
def most_unstable_cape_cin(pressure, temperature, dewpoint, **kwargs):
    r"""Calculate most unstable CAPE/CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile and most unstable parcel path. CIN is integrated between the
    surface and LFC, CAPE is integrated between the LFC and EL (or top of sounding).
    Intersection points of the measured temperature profile and parcel profile are
    logarithmically interpolated.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Pressure profile

    temperature : `pint.Quantity`
        Temperature profile

    dewpoint : `pint.Quantity`
        Dew point profile

    kwargs
        Additional keyword arguments to pass to `most_unstable_parcel`

    Returns
    -------
    `pint.Quantity`
        Most unstable Convective Available Potential Energy (CAPE)

    `pint.Quantity`
        Most unstable Convective Inhibition (CIN)

    See Also
    --------
    cape_cin, most_unstable_parcel, parcel_profile

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    """
    pressure, temperature, dewpoint = _remove_nans(pressure, temperature, dewpoint)
    _, _, _, parcel_idx = most_unstable_parcel(pressure, temperature, dewpoint, **kwargs)
    p, t, td, mu_profile = parcel_profile_with_lcl(pressure[parcel_idx:],
                                                   temperature[parcel_idx:],
                                                   dewpoint[parcel_idx:])
    return cape_cin(p, t, td, mu_profile)


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]')
def mixed_layer_cape_cin(pressure, temperature, dewpoint, **kwargs):
    r"""Calculate mixed-layer CAPE and CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile and mixed-layer parcel path. CIN is integrated between the
    surface and LFC, CAPE is integrated between the LFC and EL (or top of sounding).
    Intersection points of the measured temperature profile and parcel profile are
    logarithmically interpolated. Kwargs for `mixed_parcel` can be provided, such as `depth`.
    Default mixed-layer depth is 100 hPa.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Pressure profile

    temperature : `pint.Quantity`
        Temperature profile

    dewpoint : `pint.Quantity`
        Dewpoint profile

    kwargs
        Additional keyword arguments to pass to `mixed_parcel`

    Returns
    -------
    `pint.Quantity`
        Mixed-layer Convective Available Potential Energy (CAPE)
    `pint.Quantity`
        Mixed-layer Convective INhibition (CIN)

    See Also
    --------
    cape_cin, mixed_parcel, parcel_profile

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    """
    depth = kwargs.get('depth', units.Quantity(100, 'hPa'))
    parcel_pressure, parcel_temp, parcel_dewpoint = mixed_parcel(pressure, temperature,
                                                                 dewpoint, **kwargs)

    # Remove values below top of mixed layer and add in the mixed layer values
    pressure_prof = pressure[pressure < (pressure[0] - depth)]
    temp_prof = temperature[pressure < (pressure[0] - depth)]
    dew_prof = dewpoint[pressure < (pressure[0] - depth)]
    pressure_prof = concatenate([parcel_pressure, pressure_prof])
    temp_prof = concatenate([parcel_temp, temp_prof])
    dew_prof = concatenate([parcel_dewpoint, dew_prof])

    p, t, td, ml_profile = parcel_profile_with_lcl(pressure_prof, temp_prof, dew_prof)
    return cape_cin(p, t, td, ml_profile)


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]')
def mixed_parcel(pressure, temperature, dewpoint, parcel_start_pressure=None,
                 height=None, bottom=None, depth=None, interpolate=True):
    r"""Calculate the properties of a parcel mixed from a layer.

    Determines the properties of an air parcel that is the result of complete mixing of a
    given atmospheric layer.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile

    temperature : `pint.Quantity`
        Atmospheric temperature profile

    dewpoint : `pint.Quantity`
        Atmospheric dewpoint profile

    parcel_start_pressure : `pint.Quantity`, optional
        Pressure at which the mixed parcel should begin (default None)

    height: `pint.Quantity`, optional
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
        Pressure of the mixed parcel
    `pint.Quantity`
        Temperature of the mixed parcel

    `pint.Quantity`
        Dewpoint of the mixed parcel

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``p``, ``dewpt``, ``heights`` parameters to
       ``pressure``, ``dewpoint``, ``height``

    """
    # If a parcel starting pressure is not provided, use the surface
    if not parcel_start_pressure:
        parcel_start_pressure = pressure[0]

    if depth is None:
        depth = units.Quantity(100, 'hPa')

    # Calculate the potential temperature and mixing ratio over the layer
    theta = potential_temperature(pressure, temperature)
    mixing_ratio = saturation_mixing_ratio(pressure, dewpoint)

    # Mix the variables over the layer
    mean_theta, mean_mixing_ratio = mixed_layer(pressure, theta, mixing_ratio, bottom=bottom,
                                                height=height, depth=depth,
                                                interpolate=interpolate)

    # Convert back to temperature
    mean_temperature = mean_theta * exner_function(parcel_start_pressure)

    # Convert back to dewpoint
    mean_vapor_pressure = vapor_pressure(parcel_start_pressure, mean_mixing_ratio)

    # Using globals() here allows us to keep the dewpoint parameter but still call the
    # function of the same name.
    mean_dewpoint = globals()['dewpoint'](mean_vapor_pressure)

    return (parcel_start_pressure, mean_temperature.to(temperature.units),
            mean_dewpoint.to(dewpoint.units))


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]')
def mixed_layer(pressure, *args, height=None, bottom=None, depth=None, interpolate=True):
    r"""Mix variable(s) over a layer, yielding a mass-weighted average.

    This function will integrate a data variable with respect to pressure and determine the
    average value using the mean value theorem.

    Parameters
    ----------
    pressure : array-like
        Atmospheric pressure profile

    datavar : array-like
        Atmospheric variable measured at the given pressures

    height: array-like, optional
        Atmospheric heights corresponding to the given pressures (default None)

    bottom : `pint.Quantity`, optional
        The bottom of the layer as a pressure or height above the surface pressure
        (default None)

    depth : `pint.Quantity`, optional
        The thickness of the layer as a pressure or height above the bottom of the layer
        (default 100 hPa)

    interpolate : bool, optional
        Interpolate the top and bottom points if they are not in the given data (default True)

    Returns
    -------
    `pint.Quantity`
        The mixed value of the data variable

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``p``, ``heights`` parameters to ``pressure``, ``height``

    """
    if depth is None:
        depth = units.Quantity(100, 'hPa')
    layer = get_layer(pressure, *args, height=height, bottom=bottom,
                      depth=depth, interpolate=interpolate)
    p_layer = layer[0]
    datavars_layer = layer[1:]

    ret = []
    for datavar_layer in datavars_layer:
        actual_depth = abs(p_layer[0] - p_layer[-1])
        ret.append(units.Quantity(np.trapz(datavar_layer.m, p_layer.m) / -actual_depth.m,
                   datavar_layer.units))
    return ret


@exporter.export
@preprocess_and_wrap(wrap_like='temperature', broadcast=('height', 'temperature'))
@check_units('[length]', '[temperature]')
def dry_static_energy(height, temperature):
    r"""Calculate the dry static energy of parcels.

    This function will calculate the dry static energy following the first two terms of
    equation 3.72 in [Hobbs2006]_.

    Parameters
    ----------
    height : `pint.Quantity`
        Atmospheric height

    temperature : `pint.Quantity`
        Air temperature

    Returns
    -------
    `pint.Quantity`
        Dry static energy

    See Also
    --------
    montgomery_streamfunction

    Notes
    -----
    .. math:: \text{dry static energy} = c_{pd} T + gz

    * :math:`T` is temperature
    * :math:`z` is height

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height``

    """
    return (mpconsts.g * height + mpconsts.Cp_d * temperature).to('kJ/kg')


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('height', 'temperature', 'specific_humidity')
)
@check_units('[length]', '[temperature]', '[dimensionless]')
def moist_static_energy(height, temperature, specific_humidity):
    r"""Calculate the moist static energy of parcels.

    This function will calculate the moist static energy following
    equation 3.72 in [Hobbs2006]_.

    Parameters
    ----------
    height : `pint.Quantity`
        Atmospheric height

    temperature : `pint.Quantity`
        Air temperature

    specific_humidity : `pint.Quantity`
        Atmospheric specific humidity

    Returns
    -------
    `pint.Quantity`
        Moist static energy

    Notes
    -----
    .. math:: \text{moist static energy} = c_{pd} T + gz + L_v q

    * :math:`T` is temperature
    * :math:`z` is height
    * :math:`q` is specific humidity

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height``

    """
    return (dry_static_energy(height, temperature)
            + mpconsts.Lv * specific_humidity.to('dimensionless')).to('kJ/kg')


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]')
def thickness_hydrostatic(pressure, temperature, mixing_ratio=None,
                          molecular_weight_ratio=mpconsts.epsilon, bottom=None, depth=None):
    r"""Calculate the thickness of a layer via the hypsometric equation.

    This thickness calculation uses the pressure and temperature profiles (and optionally
    mixing ratio) via the hypsometric equation with virtual temperature adjustment.

    .. math:: Z_2 - Z_1 = -\frac{R_d}{g} \int_{p_1}^{p_2} T_v d\ln p,

    Which is based off of Equation 3.24 in [Hobbs2006]_.

    This assumes a hydrostatic atmosphere. Layer bottom and depth specified in pressure.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile

    temperature : `pint.Quantity`
        Atmospheric temperature profile

    mixing_ratio : `pint.Quantity`, optional
        Profile of dimensionless mass mixing ratio. If none is given, virtual temperature
        is simply set to be the given temperature.

    molecular_weight_ratio : `pint.Quantity` or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air.
        (:math:`\epsilon\approx0.622`)

    bottom : `pint.Quantity`, optional
        The bottom of the layer in pressure. Defaults to the first observation.

    depth : `pint.Quantity`, optional
        The depth of the layer in hPa. Defaults to the full profile if bottom is not given,
        and 100 hPa if bottom is given.

    Returns
    -------
    `pint.Quantity`
        The thickness of the layer in meters

    See Also
    --------
    thickness_hydrostatic_from_relative_humidity, pressure_to_height_std, virtual_temperature

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``mixing`` parameter to ``mixing_ratio``

    """
    # Get the data for the layer, conditional upon bottom/depth being specified and mixing
    # ratio being given
    if bottom is None and depth is None:
        if mixing_ratio is None:
            layer_p, layer_virttemp = pressure, temperature
        else:
            layer_p = pressure
            layer_virttemp = virtual_temperature(temperature, mixing_ratio,
                                                 molecular_weight_ratio)
    else:
        if mixing_ratio is None:
            layer_p, layer_virttemp = get_layer(pressure, temperature, bottom=bottom,
                                                depth=depth)
        else:
            layer_p, layer_temp, layer_w = get_layer(pressure, temperature, mixing_ratio,
                                                     bottom=bottom, depth=depth)
            layer_virttemp = virtual_temperature(layer_temp, layer_w, molecular_weight_ratio)

    # Take the integral (with unit handling) and return the result in meters
    integral = units.Quantity(np.trapz(layer_virttemp.m_as('K'), np.log(layer_p.m_as('hPa'))),
                              units.K)
    return (-mpconsts.Rd / mpconsts.g * integral).to('m')


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]')
def thickness_hydrostatic_from_relative_humidity(pressure, temperature, relative_humidity,
                                                 bottom=None, depth=None):
    r"""Calculate the thickness of a layer given pressure, temperature and relative humidity.

    Similar to ``thickness_hydrostatic``, this thickness calculation uses the pressure,
    temperature, and relative humidity profiles via the hypsometric equation with virtual
    temperature adjustment

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
        The thickness of the layer in meters

    See Also
    --------
    thickness_hydrostatic, pressure_to_height_std, virtual_temperature,
    mixing_ratio_from_relative_humidity

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    """
    mixing = mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity)

    return thickness_hydrostatic(pressure, temperature, mixing_ratio=mixing, bottom=bottom,
                                 depth=depth)


@exporter.export
@add_vertical_dim_from_xarray
@preprocess_and_wrap(wrap_like='height', broadcast=('height', 'potential_temperature'))
@check_units('[length]', '[temperature]')
def brunt_vaisala_frequency_squared(height, potential_temperature, vertical_dim=0):
    r"""Calculate the square of the Brunt-Vaisala frequency.

    Brunt-Vaisala frequency squared (a measure of atmospheric stability) is given by the
    formula:

    .. math:: N^2 = \frac{g}{\theta} \frac{d\theta}{dz}

    This formula is based off of Equations 3.75 and 3.77 in [Hobbs2006]_.

    Parameters
    ----------
    height : `xarray.DataArray` or `pint.Quantity`
        Atmospheric (geopotential) height

    potential_temperature : `xarray.DataArray` or `pint.Quantity`
        Atmospheric potential temperature

    vertical_dim : int, optional
        The axis corresponding to vertical in the potential temperature array, defaults to 0,
        unless `height` and `potential_temperature` given as `xarray.DataArray`, in which case
        it is automatically determined from the coordinate metadata.

    Returns
    -------
    `pint.Quantity` or `xarray.DataArray`
        The square of the Brunt-Vaisala frequency. Given as `pint.Quantity`, unless both
        `height` and `potential_temperature` arguments are given as `xarray.DataArray`, in
        which case will be `xarray.DataArray`.


    .. versionchanged:: 1.0
       Renamed ``heights``, ``axis`` parameters to ``height``, ``vertical_dim``

    See Also
    --------
    brunt_vaisala_frequency, brunt_vaisala_period, potential_temperature

    """
    # Ensure validity of temperature units
    potential_temperature = potential_temperature.to('K')

    # Calculate and return the square of Brunt-Vaisala frequency
    return mpconsts.g / potential_temperature * first_derivative(
        potential_temperature,
        x=height,
        axis=vertical_dim
    )


@exporter.export
@add_vertical_dim_from_xarray
@preprocess_and_wrap(wrap_like='height', broadcast=('height', 'potential_temperature'))
@check_units('[length]', '[temperature]')
def brunt_vaisala_frequency(height, potential_temperature, vertical_dim=0):
    r"""Calculate the Brunt-Vaisala frequency.

    This function will calculate the Brunt-Vaisala frequency as follows:

    .. math:: N = \left( \frac{g}{\theta} \frac{d\theta}{dz} \right)^\frac{1}{2}

    This formula based off of Equations 3.75 and 3.77 in [Hobbs2006]_.

    This function is a wrapper for `brunt_vaisala_frequency_squared` that filters out negative
    (unstable) quantities and takes the square root.

    Parameters
    ----------
    height : `xarray.DataArray` or `pint.Quantity`
        Atmospheric (geopotential) height

    potential_temperature : `xarray.DataArray` or `pint.Quantity`
        Atmospheric potential temperature

    vertical_dim : int, optional
        The axis corresponding to vertical in the potential temperature array, defaults to 0,
        unless `height` and `potential_temperature` given as `xarray.DataArray`, in which case
        it is automatically determined from the coordinate metadata.

    Returns
    -------
    `pint.Quantity` or `xarray.DataArray`
        Brunt-Vaisala frequency. Given as `pint.Quantity`, unless both
        `height` and `potential_temperature` arguments are given as `xarray.DataArray`, in
        which case will be `xarray.DataArray`.


    .. versionchanged:: 1.0
       Renamed ``heights``, ``axis`` parameters to ``height``, ``vertical_dim``

    See Also
    --------
    brunt_vaisala_frequency_squared, brunt_vaisala_period, potential_temperature

    """
    bv_freq_squared = brunt_vaisala_frequency_squared(height, potential_temperature,
                                                      vertical_dim=vertical_dim)
    bv_freq_squared[bv_freq_squared.magnitude < 0] = np.nan

    return np.sqrt(bv_freq_squared)


@exporter.export
@add_vertical_dim_from_xarray
@preprocess_and_wrap(wrap_like='height', broadcast=('height', 'potential_temperature'))
@check_units('[length]', '[temperature]')
def brunt_vaisala_period(height, potential_temperature, vertical_dim=0):
    r"""Calculate the Brunt-Vaisala period.

    This function is a helper function for `brunt_vaisala_frequency` that calculates the
    period of oscillation as in Exercise 3.13 of [Hobbs2006]_:

    .. math:: \tau = \frac{2\pi}{N}

    Returns `NaN` when :math:`N^2 > 0`.

    Parameters
    ----------
    height : `xarray.DataArray` or `pint.Quantity`
        Atmospheric (geopotential) height

    potential_temperature : `xarray.DataArray` or `pint.Quantity`
        Atmospheric potential temperature

    vertical_dim : int, optional
        The axis corresponding to vertical in the potential temperature array, defaults to 0,
        unless `height` and `potential_temperature` given as `xarray.DataArray`, in which case
        it is automatically determined from the coordinate metadata.

    Returns
    -------
    `pint.Quantity` or `xarray.DataArray`
        Brunt-Vaisala period. Given as `pint.Quantity`, unless both
        `height` and `potential_temperature` arguments are given as `xarray.DataArray`, in
        which case will be `xarray.DataArray`.


    .. versionchanged:: 1.0
       Renamed ``heights``, ``axis`` parameters to ``height``, ``vertical_dim``

    See Also
    --------
    brunt_vaisala_frequency, brunt_vaisala_frequency_squared, potential_temperature

    """
    bv_freq_squared = brunt_vaisala_frequency_squared(height, potential_temperature,
                                                      vertical_dim=vertical_dim)
    bv_freq_squared[bv_freq_squared.magnitude <= 0] = np.nan

    return 2 * np.pi / np.sqrt(bv_freq_squared)


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'dewpoint')
)
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
    `pint.Quantity`
        Wet-bulb temperature

    See Also
    --------
    lcl, moist_lapse

    Notes
    -----
    Since this function iteratively applies a parcel calculation, it should be used with
    caution on large arrays.

    """
    if not hasattr(pressure, 'shape'):
        pressure = np.atleast_1d(pressure)
        temperature = np.atleast_1d(temperature)
        dewpoint = np.atleast_1d(dewpoint)

    lcl_press, lcl_temp = lcl(pressure, temperature, dewpoint)

    it = np.nditer([pressure.magnitude, lcl_press.magnitude, lcl_temp.magnitude, None],
                   op_dtypes=['float', 'float', 'float', 'float'],
                   flags=['buffered'])

    for press, lpress, ltemp, ret in it:
        moist_adiabat_temperatures = moist_lapse(units.Quantity(press, pressure.units),
                                                 units.Quantity(ltemp, lcl_temp.units),
                                                 units.Quantity(lpress, lcl_press.units))
        ret[...] = moist_adiabat_temperatures.m_as(temperature.units)

    # If we started with a scalar, return a scalar
    ret = it.operands[3]
    if ret.size == 1:
        ret = ret[0]
    return units.Quantity(ret, temperature.units)


@exporter.export
@add_vertical_dim_from_xarray
@preprocess_and_wrap(wrap_like='temperature', broadcast=('pressure', 'temperature'))
@check_units('[pressure]', '[temperature]')
def static_stability(pressure, temperature, vertical_dim=0):
    r"""Calculate the static stability within a vertical profile.

    .. math:: \sigma = -\frac{RT}{p} \frac{\partial \ln \theta}{\partial p}

    This formula is based on equation 4.3.6 in [Bluestein1992]_.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Profile of atmospheric pressure

    temperature : `pint.Quantity`
        Profile of temperature

    vertical_dim : int, optional
        The axis corresponding to vertical in the pressure and temperature arrays, defaults
        to 0.

    Returns
    -------
    `pint.Quantity`
        The profile of static stability


    .. versionchanged:: 1.0
       Renamed ``axis`` parameter ``vertical_dim``

    """
    theta = potential_temperature(pressure, temperature)

    return - mpconsts.Rd * temperature / pressure * first_derivative(
        np.log(theta.m_as('K')),
        x=pressure,
        axis=vertical_dim
    )


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature',
    broadcast=('pressure', 'temperature', 'specific_humdiity')
)
@check_units('[pressure]', '[temperature]', '[dimensionless]')
def dewpoint_from_specific_humidity(pressure, temperature, specific_humidity):
    r"""Calculate the dewpoint from specific humidity, temperature, and pressure.

    Parameters
    ----------
    pressure: `pint.Quantity`
        Total atmospheric pressure

    temperature: `pint.Quantity`
        Air temperature

    specific_humidity: `pint.Quantity`
        Specific humidity of air

    Returns
    -------
    `pint.Quantity`
        Dew point temperature


    .. versionchanged:: 1.0
       Changed signature from ``(specific_humidity, temperature, pressure)``

    See Also
    --------
    relative_humidity_from_mixing_ratio, dewpoint_from_relative_humidity

    """
    return dewpoint_from_relative_humidity(temperature,
                                           relative_humidity_from_specific_humidity(
                                               pressure, temperature, specific_humidity))


@exporter.export
@preprocess_and_wrap(wrap_like='w', broadcast=('w', 'pressure', 'temperature'))
@check_units('[length]/[time]', '[pressure]', '[temperature]')
def vertical_velocity_pressure(w, pressure, temperature, mixing_ratio=0):
    r"""Calculate omega from w assuming hydrostatic conditions.

    This function converts vertical velocity with respect to height
    :math:`\left(w = \frac{Dz}{Dt}\right)` to that
    with respect to pressure :math:`\left(\omega = \frac{Dp}{Dt}\right)`
    assuming hydrostatic conditions on the synoptic scale.
    By Equation 7.33 in [Hobbs2006]_,

    .. math:: \omega \simeq -\rho g w

    Density (:math:`\rho`) is calculated using the :func:`density` function,
    from the given pressure and temperature. If `mixing_ratio` is given, the virtual
    temperature correction is used, otherwise, dry air is assumed.

    Parameters
    ----------
    w: `pint.Quantity`
        Vertical velocity in terms of height

    pressure: `pint.Quantity`
        Total atmospheric pressure

    temperature: `pint.Quantity`
        Air temperature

    mixing_ratio: `pint.Quantity`, optional
        Mixing ratio of air

    Returns
    -------
    `pint.Quantity`
        Vertical velocity in terms of pressure (in Pascals / second)

    See Also
    --------
    density, vertical_velocity

    """
    rho = density(pressure, temperature, mixing_ratio)
    return (-mpconsts.g * rho * w).to('Pa/s')


@exporter.export
@preprocess_and_wrap(
    wrap_like='omega',
    broadcast=('omega', 'pressure', 'temperature', 'mixing_ratio')
)
@check_units('[pressure]/[time]', '[pressure]', '[temperature]')
def vertical_velocity(omega, pressure, temperature, mixing_ratio=0):
    r"""Calculate w from omega assuming hydrostatic conditions.

    This function converts vertical velocity with respect to pressure
    :math:`\left(\omega = \frac{Dp}{Dt}\right)` to that with respect to height
    :math:`\left(w = \frac{Dz}{Dt}\right)` assuming hydrostatic conditions on
    the synoptic scale. By Equation 7.33 in [Hobbs2006]_,

    .. math:: \omega \simeq -\rho g w

    so that

    .. math:: w \simeq \frac{- \omega}{\rho g}

    Density (:math:`\rho`) is calculated using the :func:`density` function,
    from the given pressure and temperature. If `mixing_ratio` is given, the virtual
    temperature correction is used, otherwise, dry air is assumed.

    Parameters
    ----------
    omega: `pint.Quantity`
        Vertical velocity in terms of pressure

    pressure: `pint.Quantity`
        Total atmospheric pressure

    temperature: `pint.Quantity`
        Air temperature

    mixing_ratio: `pint.Quantity`, optional
        Mixing ratio of air

    Returns
    -------
    `pint.Quantity`
        Vertical velocity in terms of height (in meters / second)

    See Also
    --------
    density, vertical_velocity_pressure

    """
    rho = density(pressure, temperature, mixing_ratio)
    return (omega / (- mpconsts.g * rho)).to('m/s')


@exporter.export
@preprocess_and_wrap(wrap_like='dewpoint', broadcast=('dewpoint', 'pressure'))
@check_units('[pressure]', '[temperature]')
def specific_humidity_from_dewpoint(pressure, dewpoint):
    r"""Calculate the specific humidity from the dewpoint temperature and pressure.

    Parameters
    ----------
    dewpoint: `pint.Quantity`
        Dewpoint temperature

    pressure: `pint.Quantity`
        Pressure

    Returns
    -------
    `pint.Quantity`
        Specific humidity


    .. versionchanged:: 1.0
       Changed signature from ``(dewpoint, pressure)``

    See Also
    --------
    mixing_ratio, saturation_mixing_ratio

    """
    mixing_ratio = saturation_mixing_ratio(pressure, dewpoint)
    return specific_humidity_from_mixing_ratio(mixing_ratio)


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]')
def lifted_index(pressure, temperature, parcel_profile):
    """Calculate Lifted Index from the pressure temperature and parcel profile.

    Lifted index formula derived from [Galway1956]_ and referenced by [DoswellSchultz2006]_:
    LI = T500 - Tp500

    where:

    T500 is the measured temperature at 500 hPa
    Tp500 is the temperature of the lifted parcel at 500 hPa

    Calculation of the lifted index is defined as the temperature difference between the
    observed 500 hPa temperature and the temperature of a parcel lifted from the
    surface to 500 hPa.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure level(s) of interest, in order from highest to
        lowest pressure

    temperature : `pint.Quantity`
        Atmospheric temperature corresponding to pressure

    parcel_profile : `pint.Quantity`
        Temperature profile of the parcel

    Returns
    -------
    `pint.Quantity`
        Lifted Index

    """
    # find the index for the 500 hPa pressure level.
    idx = np.where(pressure == units.Quantity(500, 'hPa'))
    # find the measured temperature at 500 hPa.
    t500 = temperature[idx]
    # find the parcel profile temperature at 500 hPa.
    tp500 = parcel_profile[idx]
    # calculate the lifted index.
    lifted_index = t500 - tp500.to(units.degC)
    return lifted_index


@exporter.export
@add_vertical_dim_from_xarray
@preprocess_and_wrap(
    wrap_like='potential_temperature',
    broadcast=('height', 'potential_temperature', 'u', 'v')
)
@check_units('[length]', '[temperature]', '[speed]', '[speed]')
def gradient_richardson_number(height, potential_temperature, u, v, vertical_dim=0):
    r"""Calculate the gradient (or flux) Richardson number.

    .. math:: Ri = \frac{g}{\theta} \frac{\left(\partial \theta/\partial z\right)}
             {\left(\partial u / \partial z\right)^2 + \left(\partial v / \partial z\right)^2}

    See [Holton2004]_ pg. 121-122. As noted by [Holton2004]_, flux Richardson
    number values below 0.25 indicate turbulence.

    Parameters
    ----------
    height : `pint.Quantity`
        Atmospheric height

    potential_temperature : `pint.Quantity`
        Atmospheric potential temperature

    u : `pint.Quantity`
        X component of the wind

    v : `pint.Quantity`
        y component of the wind

    vertical_dim : int, optional
        The axis corresponding to vertical, defaults to 0. Automatically determined from
        xarray DataArray arguments.

    Returns
    -------
    `pint.Quantity`
        Gradient Richardson number
    """
    dthetadz = first_derivative(potential_temperature, x=height, axis=vertical_dim)
    dudz = first_derivative(u, x=height, axis=vertical_dim)
    dvdz = first_derivative(v, x=height, axis=vertical_dim)

    return (mpconsts.g / potential_temperature) * (dthetadz / (dudz ** 2 + dvdz ** 2))


@exporter.export
@preprocess_and_wrap(
    wrap_like='temperature_bottom',
    broadcast=('temperature_bottom', 'temperature_top')
)
@check_units('[temperature]', '[temperature]')
def scale_height(temperature_bottom, temperature_top):
    r"""Calculate the scale height of a layer.

    .. math::   H = \frac{R_d \overline{T}}{g}

    This function assumes dry air, but can be used with the virtual temperature
    to account for moisture.

    Parameters
    ----------
    temperature_bottom : `pint.Quantity`
        Temperature at bottom of layer

    temperature_top : `pint.Quantity`
        Temperature at top of layer

    Returns
    -------
    `pint.Quantity`
        Scale height of layer
    """
    t_bar = 0.5 * (temperature_bottom.to('kelvin') + temperature_top.to('kelvin'))

    return (mpconsts.Rd * t_bar) / mpconsts.g


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', '[temperature]')
def showalter_index(pressure, temperature, dewpoint):
    """Calculate Showalter Index.

    Showalter Index derived from [Galway1956]_:
    SI = T500 - Tp500

    where:
    T500 is the measured temperature at 500 hPa
    Tp500 is the temperature of the parcel at 500 hPa when lifted from 850 hPa

    Parameters
    ----------
        pressure : `pint.Quantity`
            Atmospheric pressure, in order from highest to lowest pressure

        temperature : `pint.Quantity`
            Ambient temperature corresponding to ``pressure``

        dewpoint : `pint.Quantity`
            Ambient dew point temperatures corresponding to ``pressure``

    Returns
    -------
    `pint.Quantity`
        Showalter index

    """
    # find the measured temperature and dew point temperature at 850 hPa.
    t850, td850 = interpolate_1d(units.Quantity(850, 'hPa'), pressure, temperature, dewpoint)

    # find the parcel profile temperature at 500 hPa.
    tp500 = interpolate_1d(units.Quantity(500, 'hPa'), pressure, temperature)

    # Lift parcel from 850 to 500, handling any needed dry vs. saturated adiabatic processes
    prof = parcel_profile(units.Quantity([850., 500.], 'hPa'), t850, td850)

    # Calculate the Showalter index
    return tp500 - prof[-1]
