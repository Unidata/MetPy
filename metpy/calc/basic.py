import numpy as np
import scipy.integrate as si
from numpy.ma import log, exp, cos, sin, masked_array
from scipy.constants import kilo, hour, g, K2C, C2K, C2F, F2C
from ..constants import epsilon, kappa, P0, Rd, Lv, Cp_d

__all__ = ['vapor_pressure', 'saturation_vapor_pressure', 'dewpoint',
           'dewpoint_rh', 'get_speed_dir', 'potential_temperature',
           'get_wind_components', 'mixing_ratio', 'tke', 'windchill',
           'heat_index', 'h_convergence', 'v_vorticity', 'dry_lapse',
           'moist_lapse', 'lcl', 'parcel_profile',
           'convergence_vorticity', 'advection', 'geostrophic_wind']

sat_pressure_0c = 6.112  # mb


def potential_temperature(pressure, temperature):
    r'''Calculate the potential temperature.

    Uses the Poisson equation to calculation the potential temperature
    given `pressure` and `temperature`.

    Parameters
    ----------
    pressure : array_like
        The total atmospheric pressure in mb
    temperature : array_like
        The temperature in Kelvin

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
    >>> metpy.calc.potential_temperature(800., 273.)
    290.9814150577374

    '''

    # Factor of 100 converts mb to Pa. Really need unit support here.
    return temperature * (P0 / (pressure * 100))**kappa


# Dividing P0 by 100 converts to mb
def dry_lapse(pressure, temperature, starting_pressure=P0 / 100):
    r'''Calculate the temperature at a level assuming only dry processes
    operating from the starting point.

    This function lifts a parcel starting at `temperature` and
    `starting_pressure` to the level given by `pressure`, conserving potential
    temperature.

    Parameters
    ----------
    pressure : array_like
        The atmospheric pressure level of interest in mb
    temperature : array_like
        The starting temperature in Kelvin
    starting_pressure : array_like
        The pressure at the starting point. Defaults to P0 (1000 mb).

    Returns
    -------
    array_like
       The resulting parcel temperature, in Kelvin, at level `pressure`

    See Also
    --------
    moist_lapse : Calculate parcel temperature assuming liquid saturation
                  processes
    parcel_profile : Calculate complete parcel profile
    potential_temperature
    '''

    return temperature * (pressure / starting_pressure)**kappa


def moist_lapse(pressure, temperature):
    r'''
    Calculate the temperature at a level assuming liquid saturation processes
    operating from the starting point.

    This function lifts a parcel starting at `temperature`
    this is calculating moist pseudo-adiabats. The starting pressure should be
    the first item in the `pressure` array.

    Parameters
    ----------
    pressure : array_like
        The atmospheric pressure level of interest in mb
    temperature : array_like
        The starting temperature in Kelvin

    Returns
    -------
    array_like
       The temperature corresponding to the the starting temperature and
       pressure levels, with the shape determined by numpy broadcasting rules.

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

    # Factor of 100 converts mb to Pa. Really need unit support here.
    def dT(T, P):
        rs = mixing_ratio(saturation_vapor_pressure(K2C(T)), P)
        return (1. / P) * ((Rd * T + Lv * rs) /
                           (Cp_d + (Lv * Lv * rs * epsilon / (Rd * T * T))))
    return si.odeint(dT, temperature.squeeze(), pressure.squeeze()).T


def lcl(pressure, temperature, dewpt, maxIters=50, eps=1e-2):
    r'''Calculate the lifted condensation level (LCL) using from the starting
    point.

    The starting state for the parcel is defined by `temperature`, `dewpoint`,
    and `pressure`.

    Parameters
    ----------
    pressure : array_like
        The starting atmospheric pressure in mb
    temperature : array_like
        The starting temperature in Kelvin
    dewpt : array_like
        The dew point in Kelvin

    Returns
    -------
    array_like
        The LCL in mb.

    Other Parameters
    ----------------
    maxIters : int, optional
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

    w = mixing_ratio(saturation_vapor_pressure(K2C(dewpt)), pressure)
    P = pressure
    while maxIters:
        Td = C2K(dewpoint(vapor_pressure(P, w)))
        newP = pressure * (Td / temperature) ** (1. / kappa)
        if np.abs(newP - P).max() < eps:
            break
        P = newP
        maxIters -= 1
    return newP


def parcel_profile(pressure, temperature, dewpt):
    r'''Calculate the profile a parcel takes through the atmosphere, lifting
    from the starting point.

    The parcel starts at `temperature`, and `dewpt`, lifed up
    dry adiabatically to the LCL, and then moist adiabatically from there.
    `pressure` specifies the pressure levels for the profile.

    Parameters
    ----------
    pressure : array_like
        The atmospheric pressure in mb. The first entry should be the starting
        point pressure.
    temperature : array_like
        The temperature in Kelvin
    dewpt : array_like
        The dew point in Kelvin

    Returns
    -------
    array_like
        The parcel temperatures at the specified pressure levels.

    See Also
    --------
    lcl, moist_lapse, dry_lapse
    '''

    # Find the LCL
    l = np.atleast_1d(lcl(pressure[0], temperature, dewpt))

    # Find the dry adiabatic profile, *including* the LCL
    press_lower = np.concatenate((pressure[pressure > l], l))
    T1 = dry_lapse(press_lower, temperature, pressure[0])

    # Find moist pseudo-adiabatic; combine and return, making sure to
    # elminate (duplicated) starting point
    T2 = moist_lapse(pressure[pressure < l], T1[-1]).squeeze()
    return np.concatenate((T1, T2[1:]))


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


def saturation_vapor_pressure(temperature):
    r'''Calculate the saturation water vapor (partial) pressure

    Parameters
    ----------
    temperature : array_like
        The temperature in degrees Celsius.

    Returns
    -------
    array_like
        The saturation water vapor (partial) presure in mb.

    See Also
    --------
    vapor_pressure, dewpoint

    Notes
    -----
    Instead of temperature, dewpoint may be used in order to calculate
    the actual (ambient) water vapor (partial) pressure.

    The formula used is that from Bolton 1980 [1] for T in degrees Celsius:

    .. math:: 6.112 e^\frac{17.67T}{T + 243.5}

    References
    ----------
    .. [1] Bolton, D., 1980: The Computation of Equivalent Potential
           Temperature. Mon. Wea. Rev., 108, 1046-1053.
    '''

    return sat_pressure_0c * exp(17.67 * temperature / (temperature + 243.5))


def dewpoint_rh(temperature, rh):
    r'''Calculate the ambient dewpoint given air temperature and relative
    humidity.

    Parameters
    ----------
    temperature : array_like
        Temperature in degrees Celsius
    rh : array_like
        Relative humidity expressed as a ratio in the range [0, 1]

    Returns
    -------
    array_like
        The dew point temperature in degrees Celsius, with the shape
        of the result being determined using numpy's broadcasting rules.

    See Also
    --------
    dewpoint, saturation_vapor_pressure
    '''

    return dewpoint(rh * saturation_vapor_pressure(temperature))


def dewpoint(e):
    r'''Calculate the ambient dewpoint given the vapor pressure.

    Parameters
    ----------
    e : array_like
        Water vapor partial pressure in mb

    Returns
    -------
    array_like
        Dew point temperature in degrees Celsius.

    See Also
    --------
    dewpoint_rh, saturation_vapor_pressure, vapor_pressure

    Notes
    -----
    This function inverts the Bolton 1980 [1] formula for saturation vapor
    pressure to instead calculate the temperature. This yield the following
    formula for dewpoint in degrees Celsius:

    .. math:: T = \frac{243.5 log(e / 6.112)}{17.67 - log(e / 6.112)}

    References
    ----------
    .. [1] Bolton, D., 1980: The Computation of Equivalent Potential
           Temperature. Mon. Wea. Rev., 108, 1046-1053.
    '''

    val = log(e / sat_pressure_0c)
    return 243.5 * val / (17.67 - val)


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
        The (mass) mixing ratio, unitless (e.g. Kg/Kg or g/g)

    See Also
    --------
    vapor_pressure
    '''

    return epsilon * part_press / (tot_press - part_press)


def get_speed_dir(u, v):
    r'''Compute the wind speed and wind direction.

    Parameters
    ----------
    u : array_like
        Wind component in the X (East-West) direction
    v : array_like
        Wind component in the Y (North-South) direction

    Returns
    -------
    speed, direction : tuple of array_like
        The speed and direction of the wind, respectively

    See Also
    --------
    get_wind_components
    '''

    speed = np.sqrt(u * u + v * v)
    wdir = np.rad2deg(np.arctan2(-u, -v))
    wdir[wdir < 0] += 360.
    return speed, wdir


def get_wind_components(speed, wdir):
    r'''Calculate the U, V wind vector components from the speed and
    direction.

    Parameters
    ----------
    speed : array_like
        The wind speed (magnitude)
    wdir : array_like
        The wind direction in degrees, specified as the direction from which the
        wind is blowing.

    Returns
    -------
    u, v : tuple of array_like
        The wind components in the X (East-West) and Y (North-South)
        directions, respectively.

    See Also
    --------
    get_speed_dir
    '''

    wdir = np.deg2rad(wdir)
    u = -speed * sin(wdir)
    v = -speed * cos(wdir)
    return u, v


def tke(u, v, w):
    r'''Compute the turbulence kinetic energy (tke) from the time series of the
    velocity components.

    Parameters
    ----------
    u : array_like
        The wind component along the x-axis
    v : array_like
        The wind component along the y-axis
    w : array_like
        The wind componennt along the z-axis

    Returns
    -------
    array_like
        The corresponding tke value(s)
    '''

    up = u - u.mean()
    vp = v - v.mean()
    wp = w - w.mean()
    return np.sqrt(up * up + vp * vp + wp * wp)


def windchill(temperature, speed, face_level_winds=False, mask_undefined=True):
    r'''Calculate the Wind Chill Temperature Index (WCTI) from the current
    temperature and wind speed.

    Specifically, these formulas assume that wind speed is measured at
    10m.  If, instead, the speeds are measured at face level, the winds
    need to be multiplied by a factor of 1.5 (this can be done by specifying
    `face_level_winds` as True.)

    Parameters
    ----------
    temp : array_like
        The air temperature in degrees Celsius
    speed : array_like
        The wind speed at 10m.  If instead the winds are at face level,
        `face_level_winds` should be set to True and the 1.5 multiplicative
        correction will be applied automatically.  Wind speed should be
        given in units of meters per second.

    Returns
    -------
    array_like
        The corresponding Wind Chill Temperature Index value(s)

    Other Parameters
    ----------------
    face_level_winds : bool, optional
        A flag indicating whether the wind speeds were measured at facial
        level instead of 10m, thus requiring a correction.  Defaults to
        False.
    mask_undefined : bool, optional
        A flag indicating whether a masked array should be returned with
        values where wind chill is undefined masked.  These are values where
        the temperature > 50F or wind speed <= 3 miles per hour. Defaults
        to True.

    See Also
    --------
    heat_index

    References
    ----------
    .. [1] http://www.ofcm.gov/jagti/r19-ti-plan/pdf/03_chap3.pdf
    '''

    # Correct for lower height measurement of winds if necessary
    if face_level_winds:
        speed = speed * 1.5

    # Formula uses wind speed in km/hr, but passing in m/s makes more
    # sense.  Convert here.
    temp_limit, speed_limit = 10., 4.828  # Temp in C, speed in km/h
    speed = speed * hour / kilo
    speed_factor = speed ** 0.16
    wcti = (13.12 + 0.6215 * temperature - 11.37 * speed_factor +
            0.3965 * temperature * speed_factor)

    # See if we need to mask any undefined values
    if mask_undefined:
        mask = np.array((temperature > temp_limit) | (speed <= speed_limit))
        if mask.any():
            wcti = masked_array(wcti, mask=mask)

    return wcti


def heat_index(temperature, rh, mask_undefined=True):
    r'''Calculate the Heat Index from the current temperature and relative
    humidity.

    The implementation uses the formula outlined in [2].

    Parameters
    ----------
    temp : array_like
        Air temperature in degrees Celsuis
    rh : array_like
        The relative humidity expressed as an integer percentage.

    Returns : array_like
        The corresponding Heat Index value(s) in degrees Celsuis

    Other Parameters
    ----------------
    mask_undefined : bool, optional
        A flag indicating whether a masked array should be returned with
        values where heat index is undefined masked.  These are values where
        the temperature < 80F or relative humidity < 40 percent. Defaults
        to True.

    See Also
    --------
    windchill

    References
    ----------
    .. [1] Steadman, R.G., 1979: The assessment of sultriness. Part I: A
           temperature-humidity index based on human physiology and clothing
           science. J. Appl. Meteor., 18, 861-873.

    .. [2] http://www.srh.noaa.gov/ffc/html/studies/ta_htindx.PDF

    '''

    temperature = C2F(temperature)  # Formula in F
    rh2 = rh ** 2
    temp2 = temperature ** 2

    # Calculate the Heat Index
    HI = (-42.379 + 2.04901523 * temperature + 10.14333127 * rh -
          0.22475541 * temperature * rh - 6.83783e-3 * temp2 -
          5.481717e-2 * rh2 + 1.22874e-3 * temp2 * rh +
          8.5282e-4 * temperature * rh2 - 1.99e-6 * temp2 * rh2)

    # See if we need to mask any undefined values
    if mask_undefined:
        mask = np.array((temperature < 80.) | (rh < 40))
        if mask.any():
            HI = masked_array(HI, mask=mask)

    return F2C(HI)


def _get_gradients(u, v, dx, dy):
    # Helper function for getting convergence and vorticity from 2D arrays
    dudx, dudy = np.gradient(u, dx, dy)
    dvdx, dvdy = np.gradient(v, dx, dy)
    return dudx, dudy, dvdx, dvdy


def v_vorticity(u, v, dx, dy):
    r'''Calculate the vertical vorticity of the horizontal wind.

    The grid must have a constant spacing in each direction.

    Parameters
    ----------
    u : (X, Y) ndarray
        x component of the wind
    v : (X, Y) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (X, Y) ndarray
        vertical vorticity

    See Also
    --------
    h_convergence, convergence_vorticity
    '''

    dudx, dudy, dvdx, dvdy = _get_gradients(u, v, dx, dy)
    return dvdx - dudy


def h_convergence(u, v, dx, dy):
    r'''Calculate the horizontal convergence of the horizontal wind.

    The grid must have a constant spacing in each direction.

    Parameters
    ----------
    u : (X, Y) ndarray
        x component of the wind
    v : (X, Y) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (X, Y) ndarray
        The horizontal convergence

    See Also
    --------
    v_vorticity, convergence_vorticity
    '''

    dudx, dudy, dvdx, dvdy = _get_gradients(u, v, dx, dy)
    return dudx + dvdy


def convergence_vorticity(u, v, dx, dy):
    r'''Calculate the horizontal convergence and vertical vorticity of the
    horizontal wind.

    The grid must have a constant spacing in each direction.

    Parameters
    ----------
    u : (X, Y) ndarray
        x component of the wind
    v : (X, Y) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    convergence, vorticity : tuple of (X, Y) ndarrays
        The horizontal convergence and vertical vorticity, respectively

    See Also
    --------
    v_vorticity, h_convergence

    Notes
    -----
    This is a convenience function that will do less work than calculating
    the horizontal convergence and vertical vorticity separately.
    '''

    dudx, dudy, dvdx, dvdy = _get_gradients(u, v, dx, dy)
    return dudx + dvdy, dvdx - dudy


def advection(scalar, wind, deltas):
    r'''Calculate the advection of a scalar field by the wind.

    The order of the dimensions of the arrays must match the order in which
    the wind components are given.  For example, if the winds are given [u, v],
    then the scalar and wind arrays must be indexed as x,y (which puts x as the
    rows, not columns).

    Parameters
    ----------
    scalar : N-dimensional array
        Array (with N-dimensions) with the quantity to be advected.
    wind : sequence of arrays
        Length N sequence of N-dimensional arrays.  Represents the flow,
        with a component of the wind in each dimension.  For example, for
        horizontal advection, this could be a list: [u, v], where u and v
        are each a 2-dimensional array.
    deltas : sequence
        A (length N) sequence containing the grid spacing in each dimension.

    Returns
    -------
    N-dimensional array
        An N-dimensional array containing the advection at all grid points.
    '''

    # Gradient returns a list of derivatives along each dimension.  We convert
    # this to an array with dimension as the first index
    grad = np.asarray(np.gradient(scalar, *deltas))

    # This allows passing in a list of wind components or an array
    wind = np.asarray(wind)

    # Make them be at least 2D (handling the 1D case) so that we can do the
    # multiply and sum below
    grad, wind = np.atleast_2d(grad, wind)

    return (-grad * wind).sum(axis=0)


def geostrophic_wind(heights, f, dx, dy, geopotential=False):
    r'''Calculate the geostrophic wind given from the heights.

    Parameters
    ----------
    heights : (x,y) ndarray
        The height field, given with leading dimensions of x by y.  There
        can be trailing dimensions on the array. These are assumed in meters
        and will be scaled by gravity.
    f : array_like
        The coriolis parameter in s^-1.  This can be a scalar to be applied
        everywhere or an array of values.
    dx : scalar
        The grid spacing in the x-direction in meters.
    dy : scalar
        The grid spacing in the y-direction in meters.

    Returns
    -------
    A 2-item tuple of arrays
        A tuple of the x-component and y-component of the geostropic wind in
        m s^-1.

    Other Parameters
    ----------------
    geopotential : boolean, optional
        If true, the heights are assumed to actually be values of geopotential,
        in units of m^2 s^-2, and the values will not be scaled by gravity.
    '''

    if geopotential:
        norm_factor = 1. / f
    else:
        norm_factor = g / f

    # If heights is has more than 2 dimensions, we need to pass in some dummy
    # grid deltas so that we can still use np.gradient.  It may be better to
    # to loop in this case, but that remains to be done.
    deltas = [dx, dy]
    if heights.ndim > 2:
        deltas = deltas + [1.] * (heights.ndim - 2)

    grad = np.gradient(heights, *deltas)
    dx, dy = grad[0], grad[1]  # This throws away unused gradient components
    return -norm_factor * dy, norm_factor * dx
