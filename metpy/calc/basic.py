import numpy as np
from numpy.ma import masked_array
from ..package_tools import Exporter
from ..units import units

exporter = Exporter(globals())


@exporter.export
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
    wdir = units.Quantity(np.atleast_1d(90. * units.deg - np.arctan2(v, u)), units.deg)
    wdir[wdir < 0] += 360. * units.deg
    return speed, wdir.reshape(speed.shape)


@exporter.export
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

    u = -speed * np.sin(wdir)
    v = -speed * np.cos(wdir)
    return u, v


@exporter.export
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
        The air temperature
    speed : array_like
        The wind speed at 10m.  If instead the winds are at face level,
        `face_level_winds` should be set to True and the 1.5 multiplicative
        correction will be applied automatically.

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
    .. [4] http://www.ofcm.gov/jagti/r19-ti-plan/pdf/03_chap3.pdf
    '''

    # Correct for lower height measurement of winds if necessary
    if face_level_winds:
        speed = speed * 1.5

    temp_limit, speed_limit = 10. * units.degC, 3 * units.mph
    speed_factor = speed.to('km/hr').magnitude ** 0.16
    delta = temperature - 0. * units.degC
    wcti = (13.12 * units.degC + 0.6215 * delta -
            11.37 * units.delta_degC * speed_factor + 0.3965 * delta * speed_factor)

    # See if we need to mask any undefined values
    if mask_undefined:
        mask = np.array((temperature > temp_limit) | (speed <= speed_limit))
        if mask.any():
            wcti = masked_array(wcti, mask=mask)

    return wcti


@exporter.export
def heat_index(temperature, rh, mask_undefined=True):
    r'''Calculate the Heat Index from the current temperature and relative
    humidity.

    The implementation uses the formula outlined in [6].

    Parameters
    ----------
    temp : array_like
        Air temperature
    rh : array_like
        The relative humidity expressed as a percentage in the range [0, 100].

    Returns : array_like
        The corresponding Heat Index value(s)

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
    .. [5] Steadman, R.G., 1979: The assessment of sultriness. Part I: A
           temperature-humidity index based on human physiology and clothing
           science. J. Appl. Meteor., 18, 861-873.

    .. [6] http://www.srh.noaa.gov/ffc/html/studies/ta_htindx.PDF

    '''

    delta = temperature - 0. * units.degF
    rh2 = rh ** 2
    delta2 = delta ** 2

    # Calculate the Heat Index
    hi = (-42.379 * units.degF + 2.04901523 * delta +
          10.14333127 * units.delta_degF * rh - 0.22475541 * delta * rh -
          6.83783e-3 / units.delta_degF * delta2 - 5.481717e-2 * units.delta_degF * rh2 +
          1.22874e-3 / units.delta_degF * delta2 * rh + 8.5282e-4 * delta * rh2 -
          1.99e-6 / units.delta_degF * delta2 * rh2)

    # See if we need to mask any undefined values
    if mask_undefined:
        mask = np.array((temperature < 80. * units.degF) | (rh < 40))
        if mask.any():
            hi = masked_array(hi, mask=mask)

    return hi
