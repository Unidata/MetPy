import numpy as np
from numpy.ma import cos, sin, masked_array
from scipy.constants import kilo, hour, C2F, F2C
from ..package_tools import Exporter

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
    wdir = np.rad2deg(np.arctan2(-u, -v))
    wdir[wdir < 0] += 360.
    return speed, wdir


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

    wdir = np.deg2rad(wdir)
    u = -speed * sin(wdir)
    v = -speed * cos(wdir)
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
    .. [4] http://www.ofcm.gov/jagti/r19-ti-plan/pdf/03_chap3.pdf
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


@exporter.export
def heat_index(temperature, rh, mask_undefined=True):
    r'''Calculate the Heat Index from the current temperature and relative
    humidity.

    The implementation uses the formula outlined in [6].

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
    .. [5] Steadman, R.G., 1979: The assessment of sultriness. Part I: A
           temperature-humidity index based on human physiology and clothing
           science. J. Appl. Meteor., 18, 861-873.

    .. [6] http://www.srh.noaa.gov/ffc/html/studies/ta_htindx.PDF

    '''

    temperature = C2F(temperature)  # Formula in F
    rh2 = rh ** 2
    temp2 = temperature ** 2

    # Calculate the Heat Index
    hi = (-42.379 + 2.04901523 * temperature + 10.14333127 * rh -
          0.22475541 * temperature * rh - 6.83783e-3 * temp2 -
          5.481717e-2 * rh2 + 1.22874e-3 * temp2 * rh +
          8.5282e-4 * temperature * rh2 - 1.99e-6 * temp2 * rh2)

    # See if we need to mask any undefined values
    if mask_undefined:
        mask = np.array((temperature < 80.) | (rh < 40))
        if mask.any():
            hi = masked_array(hi, mask=mask)

    return F2C(hi)
