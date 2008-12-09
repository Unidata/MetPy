'A collection of generic calculation functions.'

import numpy as np
from numpy.ma import log, exp, cos, sin
from metpy.cbook import iterable

sat_pressure_0c = 6.112 # mb
def vapor_pressure(temp):
    '''
    Calculate the saturation water vapor (partial) pressure given
    *temperature*.

    temp : scalar or array
        The temperature in degrees Celsius.

    Returns : scalar or array
        The saturation water vapor (partial) presure in millibars, with
        the same shape as *temp*.

    Instead of temperature, dewpoint may be used in order to calculate
    the actual (ambient) water vapor (partial) pressure.
    '''
    return sat_pressure_0c * exp(17.67 * temp / (temp + 243.5))

def dewpoint(temp, rh):
    '''
    Calculate the ambient dewpoint given air temperature and relative
    humidity.

    temp : scalar or array
        The temperature in degrees Celsius.

    rh : scalar or array
        The relative humidity expressed as a ratio in the range [0, 1]

    Returns : scalar or array
        The dew point temperature in degrees Celsius, with the shape
        of the result being determined using numpy's broadcasting rules.
    '''
    es = vapor_pressure(temp)
    val = log(rh * es/sat_pressure_0c)
    return 243.5 * val / (17.67 - val)

def mixing_ratio(part_press, tot_press):
    '''
    Calculates the mixing ratio of gas given its partial pressure
    and the total pressure of the air.

    part_press : scalar or array
        The partial pressure of the constituent gas.

    tot_press : scalar or array
        The total air pressure.

    Returns : scalar or array
        The (mass) mixing ratio, unitless (e.g. Kg/Kg or g/g)

    There are no required units for the input arrays, other than that
    they have the same units.
    '''
    return part_press / (tot_press - part_press)

def get_wind_components(speed, wdir):
    '''
    Calculate the U, V wind vector components from the speed and
    direction (from which the wind is blowing).

    speed : scalar or array
        The wind speed (magnitude)

    wdir : scalar or array
        The wind direction in degrees

    Returns : tuple of scalars or arrays
        The tuple (U,V) corresponding to the wind components in the
        X (East-West) and Y (North-South) directions, respectively.
    '''
    from scipy.constants import degree
    u = -speed * sin(wdir * degree)
    v = -speed * cos(wdir * degree)
    return u,v
