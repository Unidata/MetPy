'A collection of generic calculation functions.'

import numpy as np
from metpy.cbook import iterable

sat_pressure_0c = 6.112 # mb
def vapor_pressure(temp):
    '''
    Calculate the saturation water vapor (partial) pressure given
    *temperature*.

    temp : scalar or array
        The temperature in degrees Celsius.

    Returns : scalar or array
        The saturation water vapor (partial) presure, with the same
        shape as *temp*.

    Instead of temperature, dewpoint may be used in order to calculate
    the actual (ambient) water vapor (partial) pressure.
    '''
    return sat_pressure_0c * np.exp(17.67 * temp / (temp + 243.5))

def dewpoint(temp, rh):
    '''
    Calculate the ambient dewpoint given air temperature and relative
    humidity.

    temp : scalar or array
        The temperature in degrees Celsius.

    rh : scalar or array
        The relative humidity expressed as a ratio in the range [0, 1]

    Returns : scalar or array
        The dew point temperature, with the shape of the result being
        that which results from broadcasting temp against rh using
        numpy's broadcasting rules.
    '''
    es = vapor_pressure(temp)
    val = np.log(rh * es/sat_pressure_0c)
    return 243.5 * val / (17.67 - val)
