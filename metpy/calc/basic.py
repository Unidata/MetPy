# Copyright (c) 2008,2015,2016,2017,2018 MetPy Developers.
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
from scipy.ndimage import gaussian_filter

from .. import constants as mpconsts
from ..deprecation import deprecated
from ..package_tools import Exporter
from ..units import atleast_1d, check_units, masked_array, units
from ..xarray import preprocess_xarray

exporter = Exporter(globals())


@exporter.export
@preprocess_xarray
def wind_speed(u, v):
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
    wind_components

    """
    speed = np.sqrt(u * u + v * v)
    return speed


@exporter.export
@preprocess_xarray
def wind_direction(u, v):
    r"""Compute the wind direction from u and v-components.

    Parameters
    ----------
    u : array_like
        Wind component in the X (East-West) direction
    v : array_like
        Wind component in the Y (North-South) direction

    Returns
    -------
    direction: `pint.Quantity`
        The direction of the wind in interval [0, 360] degrees, specified as the direction from
        which it is blowing, with 360 being North.

    See Also
    --------
    wind_components

    Notes
    -----
    In the case of calm winds (where `u` and `v` are zero), this function returns a direction
    of 0.

    """
    wdir = 90. * units.deg - np.arctan2(-v, -u)
    origshape = wdir.shape
    wdir = atleast_1d(wdir)
    wdir[wdir <= 0] += 360. * units.deg
    # Need to be able to handle array-like u and v (with or without units)
    # np.any check required for legacy numpy which treats 0-d False boolean index as zero
    calm_mask = (np.asarray(u) == 0.) & (np.asarray(v) == 0.)
    if np.any(calm_mask):
        wdir[calm_mask] = 0. * units.deg
    return wdir.reshape(origshape).to('degrees')


@exporter.export
@preprocess_xarray
def wind_components(speed, wdir):
    r"""Calculate the U, V wind vector components from the speed and direction.

    Parameters
    ----------
    speed : array_like
        The wind speed (magnitude)
    wdir : array_like
        The wind direction, specified as the direction from which the wind is
        blowing (0-2 pi radians or 0-360 degrees), with 360 degrees being North.

    Returns
    -------
    u, v : tuple of array_like
        The wind components in the X (East-West) and Y (North-South)
        directions, respectively.

    See Also
    --------
    wind_speed
    wind_direction

    Examples
    --------
    >>> from metpy.units import units
    >>> metpy.calc.wind_components(10. * units('m/s'), 225. * units.deg)
    (<Quantity(7.071067811865475, 'meter / second')>,
     <Quantity(7.071067811865477, 'meter / second')>)

    """
    wdir = _check_radians(wdir, max_radians=4 * np.pi)
    u = -speed * np.sin(wdir)
    v = -speed * np.cos(wdir)
    return u, v


@exporter.export
@preprocess_xarray
@deprecated('0.9', addendum=' This function has been renamed wind_speed.',
            pending=False)
def get_wind_speed(u, v):
    """Wrap wind_speed for deprecated get_wind_speed function."""
    return wind_speed(u, v)


get_wind_speed.__doc__ = (wind_speed.__doc__
                          + '\n    .. deprecated:: 0.9.0\n        Function has been renamed to'
                            ' `wind_speed` and will be removed from MetPy in 0.12.0.')


@exporter.export
@preprocess_xarray
@deprecated('0.9', addendum=' This function has been renamed wind_direction.',
            pending=False)
def get_wind_dir(u, v):
    """Wrap wind_direction for deprecated get_wind_dir function."""
    return wind_direction(u, v)


get_wind_dir.__doc__ = (wind_direction.__doc__
                        + '\n    .. deprecated:: 0.9.0\n        Function has been renamed to '
                          '`wind_direction` and will be removed from MetPy in 0.12.0.')


@exporter.export
@preprocess_xarray
@deprecated('0.9', addendum=' This function has been renamed wind_components.',
            pending=False)
def get_wind_components(u, v):
    """Wrap wind_components for deprecated get_wind_components function."""
    return wind_components(u, v)


get_wind_components.__doc__ = (wind_components.__doc__
                               + '\n    .. deprecated:: 0.9.0\n        Function has been '
                                 'renamed to `wind_components` and will be removed from MetPy '
                                 'in 0.12.0.')


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
    wcti = units.Quantity((0.6215 + 0.3965 * speed_factor) * temperature.to('degC').magnitude
                          - 11.37 * speed_factor + 13.12, units.degC).to(temperature.units)

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
    delta = temperature.to(units.degF) - 0. * units.degF
    rh2 = rh * rh
    delta2 = delta * delta

    # Calculate the Heat Index -- constants converted for RH in [0, 1]
    hi = (-42.379 * units.degF
          + 2.04901523 * delta
          + 1014.333127 * units.delta_degF * rh
          - 22.475541 * delta * rh
          - 6.83783e-3 / units.delta_degF * delta2
          - 5.481717e2 * units.delta_degF * rh2
          + 1.22874e-1 / units.delta_degF * delta2 * rh
          + 8.5282 * delta * rh2
          - 1.99e-2 / units.delta_degF * delta2 * rh2)

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
    app_temperature = np.ma.where(masked_array(wind_chill_temperature).mask,
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
        return atleast_1d(app_temperature)[0] * temperature.units


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
    return (t0 / gamma) * (1 - (pressure / p0).to('dimensionless')**(
        mpconsts.Rd * gamma / mpconsts.g))


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
    geopot = mpconsts.G * mpconsts.me * ((1 / mpconsts.Re) - (1 / (mpconsts.Re + height)))

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
    height = (((1 / mpconsts.Re) - (geopot / (mpconsts.G * mpconsts.me))) ** -1) - mpconsts.Re

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
    return p0 * (1 - (gamma / t0) * height) ** (mpconsts.g / (mpconsts.Rd * gamma))


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
    return (2. * mpconsts.omega * np.sin(latitude)).to('1/s')


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


@exporter.export
@preprocess_xarray
def smooth_gaussian(scalar_grid, n):
    """Filter with normal distribution of weights.

    Parameters
    ----------
    scalar_grid : `pint.Quantity`
        Some n-dimensional scalar grid. If more than two axes, smoothing
        is only done across the last two.

    n : int
        Degree of filtering

    Returns
    -------
    `pint.Quantity`
        The filtered 2D scalar grid

    Notes
    -----
    This function is a close replication of the GEMPAK function GWFS,
    but is not identical.  The following notes are incorporated from
    the GEMPAK source code:

    This function smoothes a scalar grid using a moving average
    low-pass filter whose weights are determined by the normal
    (Gaussian) probability distribution function for two dimensions.
    The weight given to any grid point within the area covered by the
    moving average for a target grid point is proportional to

                    EXP [ -( D ** 2 ) ],

    where D is the distance from that point to the target point divided
    by the standard deviation of the normal distribution.  The value of
    the standard deviation is determined by the degree of filtering
    requested.  The degree of filtering is specified by an integer.
    This integer is the number of grid increments from crest to crest
    of the wave for which the theoretical response is 1/e = .3679.  If
    the grid increment is called delta_x, and the value of this integer
    is represented by N, then the theoretical filter response function
    value for the N * delta_x wave will be 1/e.  The actual response
    function will be greater than the theoretical value.

    The larger N is, the more severe the filtering will be, because the
    response function for all wavelengths shorter than N * delta_x
    will be less than 1/e.  Furthermore, as N is increased, the slope
    of the filter response function becomes more shallow; so, the
    response at all wavelengths decreases, but the amount of decrease
    lessens with increasing wavelength.  (The theoretical response
    function can be obtained easily--it is the Fourier transform of the
    weight function described above.)

    The area of the patch covered by the moving average varies with N.
    As N gets bigger, the smoothing gets stronger, and weight values
    farther from the target grid point are larger because the standard
    deviation of the normal distribution is bigger.  Thus, increasing
    N has the effect of expanding the moving average window as well as
    changing the values of weights.  The patch is a square covering all
    points whose weight values are within two standard deviations of the
    mean of the two dimensional normal distribution.

    The key difference between GEMPAK's GWFS and this function is that,
    in GEMPAK, the leftover weight values representing the fringe of the
    distribution are applied to the target grid point.  In this
    function, the leftover weights are not used.

    When this function is invoked, the first argument is the grid to be
    smoothed, the second is the value of N as described above:

                        GWFS ( S, N )

    where N > 1.  If N <= 1, N = 2 is assumed.  For example, if N = 4,
    then the 4 delta x wave length is passed with approximate response
    1/e.

    """
    # Compute standard deviation in a manner consistent with GEMPAK
    n = int(round(n))
    if n < 2:
        n = 2
    sgma = n / (2 * np.pi)

    # Construct sigma sequence so smoothing occurs only in horizontal direction
    nax = len(scalar_grid.shape)
    # Assume the last two axes represent the horizontal directions
    sgma_seq = [sgma if i > nax - 3 else 0 for i in range(nax)]

    # Compute smoothed field and reattach units
    res = gaussian_filter(scalar_grid, sgma_seq, truncate=2 * np.sqrt(2))
    if hasattr(scalar_grid, 'units'):
        res = res * scalar_grid.units
    return res


@exporter.export
@preprocess_xarray
def smooth_n_point(scalar_grid, n=5, passes=1):
    """Filter with normal distribution of weights.

    Parameters
    ----------
    scalar_grid : array-like or `pint.Quantity`
        Some 2D scalar grid to be smoothed.

    n: int
        The number of points to use in smoothing, only valid inputs
        are 5 and 9. Defaults to 5.

    passes : int
        The number of times to apply the filter to the grid. Defaults
        to 1.

    Returns
    -------
    array-like or `pint.Quantity`
        The filtered 2D scalar grid.

    Notes
    -----
    This function is a close replication of the GEMPAK function SM5S
    and SM9S depending on the choice of the number of points to use
    for smoothing. This function can be applied multiple times to
    create a more smoothed field and will only smooth the interior
    points, leaving the end points with their original values. If a
    masked value or NaN values exists in the array, it will propagate
    to any point that uses that particular grid point in the smoothing
    calculation. Applying the smoothing function multiple times will
    propogate NaNs further throughout the domain.

    """
    if n == 9:
        p = 0.25
        q = 0.125
        r = 0.0625
    elif n == 5:
        p = 0.5
        q = 0.125
        r = 0.0
    else:
        raise ValueError('The number of points to use in the smoothing '
                         'calculation must be either 5 or 9.')

    smooth_grid = scalar_grid[:].copy()
    for _i in range(passes):
        smooth_grid[1:-1, 1:-1] = (p * smooth_grid[1:-1, 1:-1]
                                   + q * (smooth_grid[2:, 1:-1] + smooth_grid[1:-1, 2:]
                                          + smooth_grid[:-2, 1:-1] + smooth_grid[1:-1, :-2])
                                   + r * (smooth_grid[2:, 2:] + smooth_grid[2:, :-2] +
                                          + smooth_grid[:-2, 2:] + smooth_grid[:-2, :-2]))
    return smooth_grid


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
