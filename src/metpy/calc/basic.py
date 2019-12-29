# Copyright (c) 2008,2015,2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains a collection of basic calculations.

These include:

* wind components
* heat index
* windchill
"""
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from .. import constants as mpconsts
from ..package_tools import Exporter
from ..units import atleast_1d, check_units, masked_array, units
from ..xarray import preprocess_xarray

exporter = Exporter(globals())

# The following variables are constants for a standard atmosphere
t0 = 288. * units.kelvin
p0 = 1013.25 * units.hPa


@exporter.export
@preprocess_xarray
@check_units('[speed]', '[speed]')
def wind_speed(u, v):
    r"""Compute the wind speed from u and v-components.

    Parameters
    ----------
    u : `pint.Quantity`
        Wind component in the X (East-West) direction
    v : `pint.Quantity`
        Wind component in the Y (North-South) direction

    Returns
    -------
    wind speed: `pint.Quantity`
        The speed of the wind

    See Also
    --------
    wind_components

    """
    speed = np.sqrt(u * u + v * v)
    return speed


@exporter.export
@preprocess_xarray
@check_units('[speed]', '[speed]')
def wind_direction(u, v, convention='from'):
    r"""Compute the wind direction from u and v-components.

    Parameters
    ----------
    u : `pint.Quantity`
        Wind component in the X (East-West) direction
    v : `pint.Quantity`
        Wind component in the Y (North-South) direction
    convention : str
        Convention to return direction. 'from' returns the direction the wind is coming from
        (meteorological convention). 'to' returns the direction the wind is going towards
        (oceanographic convention). Default is 'from'.

    Returns
    -------
    direction: `pint.Quantity`
        The direction of the wind in interval [0, 360] degrees, with 360 being North, with the
        direction defined by the convention kwarg.

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

    # Handle oceanographic convection
    if convention == 'to':
        wdir -= 180 * units.deg
    elif convention not in ('to', 'from'):
        raise ValueError('Invalid kwarg for "convention". Valid options are "from" or "to".')

    wdir[wdir <= 0] += 360. * units.deg
    # avoid unintended modification of `pint.Quantity` by direct use of magnitude
    calm_mask = (np.asarray(u.magnitude) == 0.) & (np.asarray(v.magnitude) == 0.)
    # np.any check required for legacy numpy which treats 0-d False boolean index as zero
    if np.any(calm_mask):
        wdir[calm_mask] = 0. * units.deg
    return wdir.reshape(origshape).to('degrees')


@exporter.export
@preprocess_xarray
@check_units('[speed]')
def wind_components(speed, wdir):
    r"""Calculate the U, V wind vector components from the speed and direction.

    Parameters
    ----------
    speed : `pint.Quantity`
        The wind speed (magnitude)
    wdir : `pint.Quantity`
        The wind direction, specified as the direction from which the wind is
        blowing (0-2 pi radians or 0-360 degrees), with 360 degrees being North.

    Returns
    -------
    u, v : tuple of `pint.Quantity`
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

    The implementation uses the formula outlined in [Rothfusz1990]_, which is a
    multi-variable least-squares regression of the values obtained in [Steadman1979]_.
    Additional conditional corrections are applied to match what the National
    Weather Service operationally uses. See Figure 3 of [Anderson2013]_ for a
    depiction of this algorithm and further discussion.

    Parameters
    ----------
    temperature : `pint.Quantity`
        Air temperature
    rh : `pint.Quantity`
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
        values masked where the temperature < 80F. Defaults to `True`.

    See Also
    --------
    windchill

    """
    temperature = atleast_1d(temperature)
    rh = atleast_1d(rh)
    # assign units to rh if they currently are not present
    if not hasattr(rh, 'units'):
        rh = rh * units.dimensionless
    delta = temperature.to(units.degF) - 0. * units.degF
    rh2 = rh * rh
    delta2 = delta * delta

    # Simplifed Heat Index -- constants converted for RH in [0, 1]
    a = -10.3 * units.degF + 1.1 * delta + 4.7 * units.delta_degF * rh

    # More refined Heat Index -- constants converted for RH in [0, 1]
    b = (-42.379 * units.degF
         + 2.04901523 * delta
         + 1014.333127 * units.delta_degF * rh
         - 22.475541 * delta * rh
         - 6.83783e-3 / units.delta_degF * delta2
         - 5.481717e2 * units.delta_degF * rh2
         + 1.22874e-1 / units.delta_degF * delta2 * rh
         + 8.5282 * delta * rh2
         - 1.99e-2 / units.delta_degF * delta2 * rh2)

    # Create return heat index
    hi = np.full(np.shape(temperature), np.nan) * units.degF
    # Retain masked status of temperature with resulting heat index
    if hasattr(temperature, 'mask'):
        hi = masked_array(hi)

    # If T <= 40F, Heat Index is T
    sel = (temperature <= 40. * units.degF)
    if np.any(sel):
        hi[sel] = temperature[sel].to(units.degF)

    # If a < 79F and hi is unset, Heat Index is a
    sel = (a < 79. * units.degF) & np.isnan(hi)
    if np.any(sel):
        hi[sel] = a[sel]

    # Use b now for anywhere hi has yet to be set
    sel = np.isnan(hi)
    if np.any(sel):
        hi[sel] = b[sel]

    # Adjustment for RH <= 13% and 80F <= T <= 112F
    sel = ((rh <= 13. * units.percent) & (temperature >= 80. * units.degF)
           & (temperature <= 112. * units.degF))
    if np.any(sel):
        rh15adj = ((13. - rh * 100.) / 4.
                   * ((17. * units.delta_degF - np.abs(delta - 95. * units.delta_degF))
                      / 17. * units.delta_degF) ** 0.5)
        hi[sel] = hi[sel] - rh15adj[sel]

    # Adjustment for RH > 85% and 80F <= T <= 87F
    sel = ((rh > 85. * units.percent) & (temperature >= 80. * units.degF)
           & (temperature <= 87. * units.degF))
    if np.any(sel):
        rh85adj = 0.02 * (rh * 100. - 85.) * (87. * units.delta_degF - delta)
        hi[sel] = hi[sel] + rh85adj[sel]

    # See if we need to mask any undefined values
    if mask_undefined:
        mask = np.array(temperature < 80. * units.degF)
        if mask.any():
            hi = masked_array(hi, mask=mask)
    return hi


@exporter.export
@preprocess_xarray
@check_units(temperature='[temperature]', speed='[speed]')
def apparent_temperature(temperature, rh, speed, face_level_winds=False, mask_undefined=True):
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
    mask_undefined : bool, optional
        A flag indicating whether a masked array should be returned with
        values where wind chill or heat_index is undefined masked. For wind
        chill, these are values where the temperature > 50F or
        wind speed <= 3 miles per hour. For heat index, these are values
        where the temperature < 80F.
        Defaults to `True`.

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

    # NB: mask_defined=True is needed to know where computed values exist
    wind_chill_temperature = windchill(temperature, speed, face_level_winds=face_level_winds,
                                       mask_undefined=True).to(temperature.units)

    heat_index_temperature = heat_index(temperature, rh,
                                        mask_undefined=True).to(temperature.units)

    # Combine the heat index and wind chill arrays (no point has a value in both)
    # NB: older numpy.ma.where does not return a masked array
    app_temperature = masked_array(
        np.ma.where(masked_array(wind_chill_temperature).mask,
                    heat_index_temperature.to(temperature.units),
                    wind_chill_temperature.to(temperature.units)
                    ), temperature.units)

    # If mask_undefined is False, then set any masked values to the temperature
    if not mask_undefined:
        app_temperature[app_temperature.mask] = temperature[app_temperature.mask]

    # If no values are masked and provided temperature does not have a mask
    # we should return a non-masked array
    if not np.any(app_temperature.mask) and not hasattr(temperature, 'mask'):
        app_temperature = np.array(app_temperature.m) * temperature.units

    if is_not_scalar:
        return app_temperature
    else:
        return atleast_1d(app_temperature)[0]


@exporter.export
@preprocess_xarray
@check_units('[pressure]')
def pressure_to_height_std(pressure):
    r"""Convert pressure data to heights using the U.S. standard atmosphere [NOAA1976]_.

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
    gamma = 6.5 * units('K/km')
    return (t0 / gamma) * (1 - (pressure / p0).to('dimensionless')**(
        mpconsts.Rd * gamma / mpconsts.g))


@exporter.export
@preprocess_xarray
@check_units('[length]')
def height_to_geopotential(height):
    r"""Compute geopotential for a given height above sea level.

    Calculates the geopotential from height above mean sea level using the following formula,
    which is derived from the definition of geopotential as given in [Hobbs2006]_ Pg. 69 Eq
    3.21, along with an approximation for variation of gravity with altitude:

    .. math:: \Phi = \frac{g R_e z}{R_e + z}

    (where :math:`\Phi` is geopotential, :math:`z` is height, :math:`R_e` is average Earth
    radius, and :math:`g` is standard gravity.)

    Parameters
    ----------
    height : `pint.Quantity`
        Height above sea level

    Returns
    -------
    `pint.Quantity`
        The corresponding geopotential value(s)

    Examples
    --------
    >>> import metpy.calc
    >>> from metpy.units import units
    >>> height = np.linspace(0, 10000, num=11) * units.m
    >>> geopot = metpy.calc.height_to_geopotential(height)
    >>> geopot
    <Quantity([     0.           9805.11102602  19607.14506998  29406.10358006
    39201.98800351  48994.79978671  58784.54037509  68571.21121319
    78354.81374467  88135.34941224  97912.81965774], 'meter ** 2 / second ** 2')>

    Notes
    -----
    This calculation approximates :math:`g(z)` as

    .. math:: g(z) = g_0 \left( \frac{R_e}{R_e + z} \right)^2

    where :math:`g_0` is standard gravity. It thereby accounts for the average effects of
    centrifugal force on apparent gravity, but neglects latitudinal variations due to
    centrifugal force and Earth's eccentricity.

    (Prior to MetPy v0.11, this formula instead calculated :math:`g(z)` from Newton's Law of
    Gravitation assuming a spherical Earth and no centrifugal force effects.)

    See Also
    --------
    geopotential_to_height

    """
    return (mpconsts.g * mpconsts.Re * height) / (mpconsts.Re + height)


@exporter.export
@preprocess_xarray
def geopotential_to_height(geopot):
    r"""Compute height above sea level from a given geopotential.

    Calculates the height above mean sea level from geopotential using the following formula,
    which is derived from the definition of geopotential as given in [Hobbs2006]_ Pg. 69 Eq
    3.21, along with an approximation for variation of gravity with altitude:

    .. math:: z = \frac{\Phi R_e}{gR_e - \Phi}

    (where :math:`\Phi` is geopotential, :math:`z` is height, :math:`R_e` is average Earth
    radius, and :math:`g` is standard gravity.)

    Parameters
    ----------
    geopotential : `pint.Quantity`
        Geopotential

    Returns
    -------
    `pint.Quantity`
        The corresponding value(s) of height above sea level

    Examples
    --------
    >>> import metpy.calc
    >>> from metpy.units import units
    >>> height = np.linspace(0, 10000, num=11) * units.m
    >>> geopot = metpy.calc.height_to_geopotential(height)
    >>> geopot
    <Quantity([     0.           9805.11102602  19607.14506998  29406.10358006
    39201.98800351  48994.79978671  58784.54037509  68571.21121319
    78354.81374467  88135.34941224  97912.81965774], 'meter ** 2 / second ** 2')>
    >>> height = metpy.calc.geopotential_to_height(geopot)
    >>> height
    <Quantity([     0.   1000.   2000.   3000.   4000.   5000.   6000.   7000.   8000.
    9000.  10000.], 'meter')>

    Notes
    -----
    This calculation approximates :math:`g(z)` as

    .. math:: g(z) = g_0 \left( \frac{R_e}{R_e + z} \right)^2

    where :math:`g_0` is standard gravity. It thereby accounts for the average effects of
    centrifugal force on apparent gravity, but neglects latitudinal variations due to
    centrifugal force and Earth's eccentricity.

    (Prior to MetPy v0.11, this formula instead calculated :math:`g(z)` from Newton's Law of
    Gravitation assuming a spherical Earth and no centrifugal force effects.)

    See Also
    --------
    height_to_geopotential

    """
    return (geopot * mpconsts.Re) / (mpconsts.g * mpconsts.Re - geopot)


@exporter.export
@preprocess_xarray
@check_units('[length]')
def height_to_pressure_std(height):
    r"""Convert height data to pressures using the U.S. standard atmosphere [NOAA1976]_.

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
    gamma = 6.5 * units('K/km')
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

    This assumes a standard atmosphere [NOAA1976]_.

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

    This assumes a standard atmosphere [NOAA1976]_.

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
@units.wraps('=A', ('=A', None))
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

    # Compute smoothed field
    return gaussian_filter(scalar_grid, sgma_seq, truncate=2 * np.sqrt(2))


@exporter.export
@preprocess_xarray
@units.wraps('=A', ('=A', None, None))
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
        The number of times to apply the filter to the grid. Defaults to 1.

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
    propagate NaNs further throughout the domain.

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


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[length]')
def altimeter_to_station_pressure(altimeter_value, height):
    r"""Convert the altimeter measurement to station pressure.

    This function is useful for working with METARs since they do not provide
    altimeter values, but not sea-level pressure or station pressure.
    The following definitions of altimeter setting and station pressure
    are taken from [Smithsonian1951]_ Altimeter setting is the
    pressure value to which an aircraft altimeter scale is set so that it will
    indicate the altitude above mean sea-level of an aircraft on the ground at the
    location for which the value is determined. It assumes a standard atmosphere [NOAA1976]_.
    Station pressure is the atmospheric pressure at the designated station elevation.
    Finding the station pressure can be helpful for calculating sea-level pressure
    or other parameters.

    Parameters
    ----------
    altimeter_value : `pint.Quantity`
        The altimeter setting value as defined by the METAR or other observation,
        which can be measured in either inches of mercury (in. Hg) or millibars (mb)
    height: `pint.Quantity`
        Elevation of the station measuring pressure.

    Returns
    -------
    `pint.Quantity`
        The station pressure in hPa or in. Hg, which can be used to calculate sea-level
        pressure

    See Also
    --------
    altimeter_to_sea_level_pressure

    Notes
    -----
    This function is implemented using the following equations from the
    Smithsonian Handbook (1951) p. 269

    Equation 1:
     .. math:: A_{mb} = (p_{mb} - 0.3)F

    Equation 3:
     .. math::  F = \left [1 + \left(\frac{p_{0}^n a}{T_{0}} \right)
                   \frac{H_{b}}{p_{1}^n} \right ] ^ \frac{1}{n}

    Where

    :math:`p_{0}` = standard sea-level pressure = 1013.25 mb

    :math:`p_{1} = p_{mb} - 0.3` when :math:`p_{0} = 1013.25 mb`

    gamma = lapse rate in [NOAA1976]_ standard atmosphere below the isothermal layer
    :math:`6.5^{\circ}C. km.^{-1}`

    :math:`t_{0}` = standard sea-level temperature 288 K

    :math:`H_{b} =` station elevation in meters (elevation for which station
      pressure is given)

    :math:`n = \frac{a R_{d}}{g} = 0.190284` where :math:`R_{d}` is the gas
      constant for dry air

    And solving for :math:`p_{mb}` results in the equation below, which is used to
    calculate station pressure :math:`(p_{mb})`

    .. math:: p_{mb} = \left [A_{mb} ^ n - \left (\frac{p_{0} a H_{b}}{T_0}
                       \right) \right] ^ \frac{1}{n} + 0.3

    """
    # Gamma Value for this case
    gamma = 0.0065 * units('K/m')

    # N-Value
    n = (mpconsts.Rd * gamma / mpconsts.g).to_base_units()

    return ((altimeter_value ** n
             - ((p0.to(altimeter_value.units) ** n * gamma * height) / t0)) ** (1 / n)
            + 0.3 * units.hPa)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[length]', '[temperature]')
def altimeter_to_sea_level_pressure(altimeter_value, height, temperature):
    r"""Convert the altimeter setting to sea-level pressure.

    This function is useful for working with METARs since most provide
    altimeter values, but not sea-level pressure, which is often plotted
    on surface maps. The following definitions of altimeter setting, station pressure, and
    sea-level pressure are taken from [Smithsonian1951]_
    Altimeter setting is the pressure value to which an aircraft altimeter scale
    is set so that it will indicate the altitude above mean sea-level of an aircraft
    on the ground at the location for which the value is determined. It assumes a standard
    atmosphere. Station pressure is the atmospheric pressure at the designated station
    elevation. Sea-level pressure is a pressure value obtained by the theoretical reduction
    of barometric pressure to sea level. It is assumed that atmosphere extends to sea level
    below the station and that the properties of the atmosphere are related to conditions
    observed at the station. This value is recorded by some surface observation stations,
    but not all. If the value is recorded, it can be found in the remarks section. Finding
    the sea-level pressure is helpful for plotting purposes and different calculations.

    Parameters
    ----------
    altimeter_value : 'pint.Quantity'
        The altimeter setting value is defined by the METAR or other observation,
        with units of inches of mercury (in Hg) or millibars (hPa)
    height  : 'pint.Quantity'
        Elevation of the station measuring pressure. Often times measured in meters
    temperature : 'pint.Quantity'
        Temperature at the station

    Returns
    -------
    'pint.Quantity'
        The sea-level pressure in hPa and makes pressure values easier to compare
        between different stations

    See Also
    --------
    altimeter_to_station_pressure

    Notes
    -----
    This function is implemented using the following equations from Wallace and Hobbs (1977)

    Equation 2.29:
     .. math::
       \Delta z = Z_{2} - Z_{1}
       = \frac{R_{d} \bar T_{v}}{g_0}ln\left(\frac{p_{1}}{p_{2}}\right)
       = \bar H ln \left (\frac {p_{1}}{p_{2}} \right)

    Equation 2.31:
     .. math::
       p_{0} = p_{g}exp \left(\frac{Z_{g}}{\bar H} \right) \\
       = p_{g}exp \left(\frac{g_{0}Z_{g}}{R_{d}\bar T_{v}} \right)

    Then by substituting :math:`Delta_{Z}` for :math:`Z_{g}` in Equation 2.31:
     .. math:: p_{sea_level} = p_{station} exp\left(\frac{\Delta z}{H}\right)

    where :math:`Delta_{Z}` is the elevation in meters and :math:`H = \frac{R_{d}T}{g}`

    """
    # Calculate the station pressure using function altimeter_to_station_pressure()
    psfc = altimeter_to_station_pressure(altimeter_value, height)

    # Calculate the scale height
    h = mpconsts.Rd * temperature / mpconsts.g

    return psfc * np.exp(height / h)


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
