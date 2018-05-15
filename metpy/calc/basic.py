# Copyright (c) 2008,2015,2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains a collection of basic calculations.

These include:

* wind components
* heat index
* windchill
"""

from __future__ import division

import warnings

import numpy as np

from ..constants import G, g, me, omega, Rd, Re
from ..package_tools import Exporter
from ..units import atleast_1d, check_units, masked_array, units
from ..xarray import preprocess_xarray

exporter = Exporter(globals())


@exporter.export
@preprocess_xarray
def get_wind_speed(u, v):
    r"""Compute the wind speed from u and v-components.

    Parameters
    ----------
    u : array_like
        Wind component in the X (East-West) direction
    v : array_like
        Wind component in the Y (North-South) direction

    Returns
    -------
    wind speed: array_like
        The speed of the wind

    See Also
    --------
    get_wind_components

    """
    speed = np.sqrt(u * u + v * v)
    return speed


@exporter.export
@preprocess_xarray
def get_wind_dir(u, v):
    r"""Compute the wind direction from u and v-components.

    Parameters
    ----------
    u : array_like
        Wind component in the X (East-West) direction
    v : array_like
        Wind component in the Y (North-South) direction

    Returns
    -------
    wind direction: `pint.Quantity`
        The direction of the wind, specified as the direction from
        which it is blowing, with 0 being North.

    See Also
    --------
    get_wind_components

    """
    wdir = 90. * units.deg - np.arctan2(-v, -u)
    origshape = wdir.shape
    wdir = atleast_1d(wdir)
    wdir[wdir < 0] += 360. * units.deg
    return wdir.reshape(origshape)


@exporter.export
@preprocess_xarray
def get_wind_components(speed, wdir):
    r"""Calculate the U, V wind vector components from the speed and direction.

    Parameters
    ----------
    speed : array_like
        The wind speed (magnitude)
    wdir : array_like
        The wind direction, specified as the direction from which the wind is
        blowing, with 0 being North.

    Returns
    -------
    u, v : tuple of array_like
        The wind components in the X (East-West) and Y (North-South)
        directions, respectively.

    See Also
    --------
    get_wind_speed
    get_wind_dir

    Examples
    --------
    >>> from metpy.units import units
    >>> metpy.calc.get_wind_components(10. * units('m/s'), 225. * units.deg)
    (<Quantity(7.071067811865475, 'meter / second')>,
     <Quantity(7.071067811865477, 'meter / second')>)

    """
    wdir = _check_radians(wdir, max_radians=4 * np.pi)
    u = -speed * np.sin(wdir)
    v = -speed * np.cos(wdir)
    return u, v


@exporter.export
@preprocess_xarray
@check_units(temperature='[temperature]', speed='[speed]')
def windchill(temperature, speed, face_level_winds=False, mask_undefined=True):
    r"""Calculate the Wind Chill Temperature Index (WCTI).

    Calculates WCTI from the current temperature and wind speed using the formula
    outlined by the FCM [FCMR192003]_.

    Specifically, these formulas assume that wind speed is measured at
    10m.  If, instead, the speeds are measured at face level, the winds
    need to be multiplied by a factor of 1.5 (this can be done by specifying
    `face_level_winds` as `True`.)

    Parameters
    ----------
    temperature : `pint.Quantity`
        The air temperature
    speed : `pint.Quantity`
        The wind speed at 10m.  If instead the winds are at face level,
        `face_level_winds` should be set to `True` and the 1.5 multiplicative
        correction will be applied automatically.
    face_level_winds : bool, optional
        A flag indicating whether the wind speeds were measured at facial
        level instead of 10m, thus requiring a correction.  Defaults to
        `False`.
    mask_undefined : bool, optional
        A flag indicating whether a masked array should be returned with
        values where wind chill is undefined masked.  These are values where
        the temperature > 50F or wind speed <= 3 miles per hour. Defaults
        to `True`.

    Returns
    -------
    `pint.Quantity`
        The corresponding Wind Chill Temperature Index value(s)

    See Also
    --------
    heat_index

    """
    # Correct for lower height measurement of winds if necessary
    if face_level_winds:
        # No in-place so that we copy
        # noinspection PyAugmentAssignment
        speed = speed * 1.5

    temp_limit, speed_limit = 10. * units.degC, 3 * units.mph
    speed_factor = speed.to('km/hr').magnitude ** 0.16
    wcti = units.Quantity((0.6215 + 0.3965 * speed_factor) * temperature.to('degC').magnitude -
                          11.37 * speed_factor + 13.12, units.degC).to(temperature.units)

    # See if we need to mask any undefined values
    if mask_undefined:
        mask = np.array((temperature > temp_limit) | (speed <= speed_limit))
        if mask.any():
            wcti = masked_array(wcti, mask=mask)

    return wcti


@exporter.export
@preprocess_xarray
@check_units('[temperature]')
def heat_index(temperature, rh, mask_undefined=True):
    r"""Calculate the Heat Index from the current temperature and relative humidity.

    The implementation uses the formula outlined in [Rothfusz1990]_. This equation is a
    multi-variable least-squares regression of the values obtained in [Steadman1979]_.

    Parameters
    ----------
    temperature : `pint.Quantity`
        Air temperature
    rh : array_like
        The relative humidity expressed as a unitless ratio in the range [0, 1].
        Can also pass a percentage if proper units are attached.

    Returns
    -------
    `pint.Quantity`
        The corresponding Heat Index value(s)

    Other Parameters
    ----------------
    mask_undefined : bool, optional
        A flag indicating whether a masked array should be returned with
        values where heat index is undefined masked.  These are values where
        the temperature < 80F or relative humidity < 40 percent. Defaults
        to `True`.

    See Also
    --------
    windchill

    """
    delta = temperature - 0. * units.degF
    rh2 = rh * rh
    delta2 = delta * delta

    # Calculate the Heat Index -- constants converted for RH in [0, 1]
    hi = (-42.379 * units.degF + 2.04901523 * delta +
          1014.333127 * units.delta_degF * rh - 22.475541 * delta * rh -
          6.83783e-3 / units.delta_degF * delta2 - 5.481717e2 * units.delta_degF * rh2 +
          1.22874e-1 / units.delta_degF * delta2 * rh + 8.5282 * delta * rh2 -
          1.99e-2 / units.delta_degF * delta2 * rh2)

    # See if we need to mask any undefined values
    if mask_undefined:
        mask = np.array((temperature < 80. * units.degF) | (rh < 40 * units.percent))
        if mask.any():
            hi = masked_array(hi, mask=mask)

    return hi


@exporter.export
@preprocess_xarray
@check_units(temperature='[temperature]', speed='[speed]')
def apparent_temperature(temperature, rh, speed, face_level_winds=False):
    r"""Calculate the current apparent temperature.

    Calculates the current apparent temperature based on the wind chill or heat index
    as appropriate for the current conditions. Follows [NWS10201]_.

    Parameters
    ----------
    temperature : `pint.Quantity`
        The air temperature
    rh : `pint.Quantity`
        The relative humidity expressed as a unitless ratio in the range [0, 1].
        Can also pass a percentage if proper units are attached.
    speed : `pint.Quantity`
        The wind speed at 10m.  If instead the winds are at face level,
        `face_level_winds` should be set to `True` and the 1.5 multiplicative
        correction will be applied automatically.
    face_level_winds : bool, optional
        A flag indicating whether the wind speeds were measured at facial
        level instead of 10m, thus requiring a correction.  Defaults to
        `False`.

    Returns
    -------
    `pint.Quantity`
        The corresponding apparent temperature value(s)

    See Also
    --------
    heat_index, windchill

    """
    is_not_scalar = isinstance(temperature.m, (list, tuple, np.ndarray))

    temperature = atleast_1d(temperature)
    rh = atleast_1d(rh)
    speed = atleast_1d(speed)

    wind_chill_temperature = windchill(temperature, speed, face_level_winds=face_level_winds,
                                       mask_undefined=True).to(temperature.units)

    heat_index_temperature = heat_index(temperature, rh,
                                        mask_undefined=True).to(temperature.units)

    # Combine the heat index and wind chill arrays (no point has a value in both)
    app_temperature = np.ma.where(wind_chill_temperature.mask,
                                  heat_index_temperature,
                                  wind_chill_temperature)

    if is_not_scalar:
        # Fill in missing areas where neither wind chill or heat index are applicable with the
        # ambient temperature.
        app_temperature[app_temperature.mask] = temperature[app_temperature.mask]
        return np.array(app_temperature) * temperature.units
    else:
        if app_temperature.mask:
            app_temperature = temperature.m
        return app_temperature[0] * temperature.units


@exporter.export
@preprocess_xarray
@check_units('[pressure]')
def pressure_to_height_std(pressure):
    r"""Convert pressure data to heights using the U.S. standard atmosphere.

    The implementation uses the formula outlined in [Hobbs1977]_ pg.60-61.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure

    Returns
    -------
    `pint.Quantity`
        The corresponding height value(s)

    Notes
    -----
    .. math:: Z = \frac{T_0}{\Gamma}[1-\frac{p}{p_0}^\frac{R\Gamma}{g}]

    """
    t0 = 288. * units.kelvin
    gamma = 6.5 * units('K/km')
    p0 = 1013.25 * units.mbar
    return (t0 / gamma) * (1 - (pressure / p0).to('dimensionless')**(Rd * gamma / g))


@exporter.export
@preprocess_xarray
@check_units('[length]')
def height_to_geopotential(height):
    r"""Compute geopotential for a given height.

    Parameters
    ----------
    height : `pint.Quantity`
        Height above sea level (array_like)

    Returns
    -------
    `pint.Quantity`
        The corresponding geopotential value(s)

    Examples
    --------
    >>> from metpy.constants import g, G, me, Re
    >>> import metpy.calc
    >>> from metpy.units import units
    >>> height = np.linspace(0,10000, num = 11) * units.m
    >>> geopot = metpy.calc.height_to_geopotential(height)
    >>> geopot
    <Quantity([     0.           9817.46806283  19631.85526579  29443.16305888
    39251.39289118  49056.54621087  58858.62446525  68657.62910064
    78453.56156253  88246.42329545  98036.21574306], 'meter ** 2 / second ** 2')>

    Notes
    -----
    Derived from definition of geopotential in [Hobbs2006]_ pg.14 Eq.1.8.

    """
    # Calculate geopotential
    geopot = G * me * ((1 / Re) - (1 / (Re + height)))

    return geopot


@exporter.export
@preprocess_xarray
def geopotential_to_height(geopot):
    r"""Compute height from a given geopotential.

    Parameters
    ----------
    geopotential : `pint.Quantity`
        Geopotential (array_like)

    Returns
    -------
    `pint.Quantity`
        The corresponding height value(s)

    Examples
    --------
    >>> from metpy.constants import g, G, me, Re
    >>> import metpy.calc
    >>> from metpy.units import units
    >>> height = np.linspace(0,10000, num = 11) * units.m
    >>> geopot = metpy.calc.height_to_geopotential(height)
    >>> geopot
    <Quantity([     0.           9817.46806283  19631.85526579  29443.16305888
    39251.39289118  49056.54621087  58858.62446525  68657.62910064
    78453.56156253  88246.42329545  98036.21574306], 'meter ** 2 / second ** 2')>
    >>> height = metpy.calc.geopotential_to_height(geopot)
    >>> height
    <Quantity([     0.   1000.   2000.   3000.   4000.   5000.   6000.   7000.   8000.
    9000.  10000.], 'meter')>

    Notes
    -----
    Derived from definition of geopotential in [Hobbs2006]_ pg.14 Eq.1.8.

    """
    # Calculate geopotential
    height = (((1 / Re) - (geopot / (G * me))) ** -1) - Re

    return height


@exporter.export
@preprocess_xarray
@check_units('[length]')
def height_to_pressure_std(height):
    r"""Convert height data to pressures using the U.S. standard atmosphere.

    The implementation inverts the formula outlined in [Hobbs1977]_ pg.60-61.

    Parameters
    ----------
    height : `pint.Quantity`
        Atmospheric height

    Returns
    -------
    `pint.Quantity`
        The corresponding pressure value(s)

    Notes
    -----
    .. math:: p = p_0 e^{\frac{g}{R \Gamma} \text{ln}(1-\frac{Z \Gamma}{T_0})}

    """
    t0 = 288. * units.kelvin
    gamma = 6.5 * units('K/km')
    p0 = 1013.25 * units.mbar
    return p0 * (1 - (gamma / t0) * height) ** (g / (Rd * gamma))


@exporter.export
@preprocess_xarray
def coriolis_parameter(latitude):
    r"""Calculate the coriolis parameter at each point.

    The implementation uses the formula outlined in [Hobbs1977]_ pg.370-371.

    Parameters
    ----------
    latitude : array_like
        Latitude at each point

    Returns
    -------
    `pint.Quantity`
        The corresponding coriolis force at each point

    """
    latitude = _check_radians(latitude, max_radians=np.pi / 2)
    return (2. * omega * np.sin(latitude)).to('1/s')


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[length]')
def add_height_to_pressure(pressure, height):
    r"""Calculate the pressure at a certain height above another pressure level.

    This assumes a standard atmosphere.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Pressure level
    height : `pint.Quantity`
        Height above a pressure level

    Returns
    -------
    `pint.Quantity`
        The corresponding pressure value for the height above the pressure level

    See Also
    --------
    pressure_to_height_std, height_to_pressure_std, add_pressure_to_height

    """
    pressure_level_height = pressure_to_height_std(pressure)
    return height_to_pressure_std(pressure_level_height + height)


@exporter.export
@preprocess_xarray
@check_units('[length]', '[pressure]')
def add_pressure_to_height(height, pressure):
    r"""Calculate the height at a certain pressure above another height.

    This assumes a standard atmosphere.

    Parameters
    ----------
    height : `pint.Quantity`
        Height level
    pressure : `pint.Quantity`
        Pressure above height level

    Returns
    -------
    `pint.Quantity`
        The corresponding height value for the pressure above the height level

    See Also
    --------
    pressure_to_height_std, height_to_pressure_std, add_height_to_pressure

    """
    pressure_at_height = height_to_pressure_std(height)
    return pressure_to_height_std(pressure_at_height - pressure)


@exporter.export
@preprocess_xarray
@check_units('[dimensionless]', '[pressure]', '[pressure]')
def sigma_to_pressure(sigma, psfc, ptop):
    r"""Calculate pressure from sigma values.

    Parameters
    ----------
    sigma : ndarray
        The sigma levels to be converted to pressure levels.

    psfc : `pint.Quantity`
        The surface pressure value.

    ptop : `pint.Quantity`
        The pressure value at the top of the model domain.

    Returns
    -------
    `pint.Quantity`
        The pressure values at the given sigma levels.

    Notes
    -----
    Sigma definition adapted from [Philips1957]_.

    .. math:: p = \sigma * (p_{sfc} - p_{top}) + p_{top}

    * :math:`p` is pressure at a given `\sigma` level
    * :math:`\sigma` is non-dimensional, scaled pressure
    * :math:`p_{sfc}` is pressure at the surface or model floor
    * :math:`p_{top}` is pressure at the top of the model domain

    """
    if np.any(sigma < 0) or np.any(sigma > 1):
        raise ValueError('Sigma values should be bounded by 0 and 1')

    if psfc.magnitude < 0 or ptop.magnitude < 0:
        raise ValueError('Pressure input should be non-negative')

    return sigma * (psfc - ptop) + ptop


def _check_radians(value, max_radians=2 * np.pi):
    """Input validation of values that could be in degrees instead of radians.

    Parameters
    ----------
    value : `pint.Quantity`
        The input value to check.

    max_radians : float
        Maximum absolute value of radians before warning.

    Returns
    -------
    `pint.Quantity`
        The input value

    """
    try:
        value = value.to('radians').m
    except AttributeError:
        pass
    if np.greater(np.nanmax(np.abs(value)), max_radians):
        warnings.warn('Input over {} radians. '
                      'Ensure proper units are given.'.format(max_radians))
    return value
