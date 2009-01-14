'''
A collection of functions used for solar calculations.  The core
goal is accurate calculation of the solar position (i.e. the sun's zenith
angle).  This can then be used to calculate the theoretical expected
incoming solar radiation as well as the times of sunrise and sunset.

References:
    Reda, I., Andreas, A., 2004. Solar position algorithm for solar
    radiation applications.  J. Solar Energy 76 (5), 577-289.
'''
from datetime import datetime, timedelta
import numpy as np
from numpy.ma import masked_array
from scipy.constants import degree
from metpy.constants import S, d
from metpy.cbook import iterable

__all__ = ['solar_irradiance', 'solar_declination_angle', 'solar_constant',
    'sunrise', 'sunset']

try:
    import ephem

    def _get_solar_const(dt):
        sun = ephem.Sun(dt)
        R = sun.earth_distance * ephem.meters_per_au
        return S * (d / R)**2

    def _get_declination(dt):
        return ephem.Sun(dt).dec

    def _get_cos_zenith(lat, lon, dt):
        sun = ephem.Sun()
        loc = ephem.Observer()
        loc.lat = lat
        loc.long = lon

        results = []
        for t in dt:
            loc.date = t
            sun.compute(loc)
            results.append(np.pi / 2 - sun.alt)
        return np.cos(np.array(results))

    def _get_sunrise(lat, lon, dt):
        sun = ephem.Sun()
        loc = ephem.Observer()
        loc.lat = lat
        loc.long = lon
        loc.date = dt
        return loc.next_rising(sun).datetime()

    def _get_sunset(lat, lon, dt):
        sun = ephem.Sun()
        loc = ephem.Observer()
        loc.lat = lat
        loc.long = lon
        loc.date = dt
        return loc.next_setting(sun).datetime()

except ImportError:
    import warnings
    warnings.warn('PyEphem not found.  Less accurate versions of the solar '
        'calculations will be used.  You can get PyEphem at: '
        'http://rhodesmill.org/pyephem/index')
    from metpy.constants import earth_max_declination, earth_orbit_eccentricity

    def _get_solar_const(dt):
        perihelion = dt.replace(month=1, day=3)
        ndays = (dt - perihelion).days
        M = 2 * np.pi * ndays / 365.25
        v = (M + 0.0333988 * np.sin(M) + 0.0003486 * np.sin(2 * M) + 0.0000050
            * np.sin(3 * M))
        return S * ((1 + earth_orbit_eccentricity * np.cos(v))
            / (1 - earth_orbit_eccentricity**2))**2

    def _get_declination(dt):
        solstice = dt.replace(month=6, day=22)
        ndays = (dt - solstice).days
        solar_longitude = 2 * np.pi * ndays / 365.

        return earth_max_declination * np.cos(solar_longitude) * degree

    def _get_cos_zenith(lat, lon, dt):
        delta = solar_declination_angle(dt[0])
        hour = np.array([d.hour + d.minute/60. + d.second/3600. for d in dt])
        hour_angle = 15 * hour * degree

        return (np.sin(lat) * np.sin(delta)
            - np.cos(lat) * np.cos(delta) * np.cos(hour_angle + lon))

    def _get_sunrise(lat, lon, dt):
        dec = solar_declination_angle(dt)
        hour = (np.arccos(np.tan(lat) * np.tan(dec)) - lon) / (15 * degree)
        return dt + timedelta(hours=hour)

    def _get_sunset(lat, lon, dt):
        dec = solar_declination_angle(dt)
        hour = ((2 * np.pi - np.arccos(np.tan(lat) * np.tan(dec)) - lon)
            / (15 * degree))
        return dt + timedelta(hours=24. - hour)

def solar_declination_angle(date=None):
    '''
    Calculate the actual solar declination angle for a given date.

    date : datetime.datetime instance
        The date for which the solar declination angle should be calculated.
        The default is today.

    Returns : float
        The solar declination angle in radians
    '''
    if date is None:
        date = datetime.now()

    return _get_declination(date)

def solar_constant(date=None):
    '''
    Calculate the solar irradiance at the edge of the earth's atmosphere,
    without accounting for any spreading of the radiation due to
    the angle of incidince.  This *does* account for changes in the
    distance between the Earth and sun due to the eccentricity of the
    Earth's orbit.

    date : datetime.datetime instance
        The date for which the solar 'constant' should be calculated.
        Defaults to current date.

    Returns : float
        The solar constant value adjusted for the ellipticity of Earth's orbit.
    '''
    if date is None:
        date = datetime.now()

    return _get_solar_const(date)

def solar_irradiance(latitude, longitude, dt=None, optical_depth=0.12):
    '''
    Calculate the solar irradiance for the specified time and location.

    latitude : scalar
        The latitude of the location on the Earth in degrees

    longitude : scalar
        The longitude of the location on the Earth in degrees

    dt : datetime.datetime instance
        The date and time for which the irradiance should be calculated.
        This defaults to the current date.

    optical_depth : scalar
        An overall optical depth value to assume for the atmosphere so
        that the effects of atmospheric absorption can be accounted for.
        Defaults to 0.12.

    Returns : scalar or array
        The solar irradiance in W / m^2 for each value in *hour*.
    '''
    if dt is None:
        dt = datetime.utcnow()

    if not iterable(dt):
        dt = [dt]

    lat_rad = latitude * degree
    lon_rad = longitude * degree

    cos_zenith = _get_cos_zenith(lat_rad, lon_rad, dt)

    s = solar_constant(dt[0])

    # Mask out values for cos < 0
    mask = np.array(cos_zenith < 0.)
    if mask.any():
        cos_zenith = masked_array(cos_zenith, mask=mask)

    return s * cos_zenith * np.exp(-optical_depth / cos_zenith)

def sunrise(latitude, longitude, date=None):
    '''
    Calculate the time of sunrise for a given date and location.

    latitude : scalar
        The latitude of the location on the Earth in degrees

    longitude : scalar
        The longitude of the location on the Earth in degrees

    dt : datetime.datetime instance
        The date for which sunrise should be calculated. Defaults to today.

    Returns : datetime.datetime instance
        The time of sunrise in UTC.
    '''

    if date is None:
        date = datetime.utcnow().replace(hour=0, minute=0, second=0,
            microsecond=0)

    lat_rad = latitude * degree
    lon_rad = longitude * degree

    return _get_sunrise(lat_rad, lon_rad, date)

def sunset(latitude, longitude, date=None):
    '''
    Calculate the time of sunset for a given date and location.

    latitude : scalar
        The latitude of the location on the Earth in degrees

    longitude : scalar
        The longitude of the location on the Earth in degrees

    dt : datetime.datetime instance
        The date for which sunrise should be calculated. Defaults to today.

    Returns : datetime.datetime instance
        The time of sunrise in UTC.
    '''

    if date is None:
        date = datetime.utcnow().replace(hour=0, minute=0, second=0,
            microsecond=0)

    lat_rad = latitude * degree
    lon_rad = longitude * degree

    return _get_sunset(lat_rad, lon_rad, date)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print sunrise(35.2167, -97.433), sunset(35.2167, -97.433)
    times = np.linspace(0, 24, 200)
    basedate = datetime.utcnow().replace(hour=0, minute=0, second=0,
        microsecond=0)
    dts = [basedate + timedelta(hours=t) for t in times]
    data = solar_irradiance(35.2167, -97.433, dts)
    plt.plot(times, data)
    plt.show()
