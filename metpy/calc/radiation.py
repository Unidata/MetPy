# Copyright (c) 2008,2015,2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
""" Contains a collection of radiation calculations.

This file contains Some heavily edited code and documentation from :

     meteolib/evaplib
     Author: Maarten J. Waterloo <m.j.waterloo@vu.nl>
     Version: 1.0
     Date: 2014-11
     License: Unknown - assumed Public Domain
     http://python.hydrology-amsterdam.nl/moduledoc/index.html#

     AND

     scikits-hydroclimpy
     Version: 0.67.1
     Date: 2009-11-25
     License: BSD
     http://hydroclimpy.sourceforge.net/
"""

from __future__ import division

import numpy as np
import pandas as pd

from ..constants import S
from ..package_tools import Exporter
from ..units import units
from .basic import _check_radians

exporter = Exporter(globals())


@exporter.export
def solar_declination(dates):
    r"""Returns the solar declination :math:`\delta` as [Spencer_1971]

    .. math::

       \\frac{\pi}{180^\circ} \delta &= 0.006918 \\\\
                                     &- 0.399912 \cos(\phi) \\\\
                                     &+ 0.070257 \sin(\phi) \\\\
                                     &- 0.006758 \cos(2\phi) \\\\
                                     &+ 0.000907 \sin(2\phi) \\\\
                                     &- 0.002697 \cos(3\phi) \\\\
                                     &+ 0.001480 \sin(3\phi)

    where :math:`\phi= 2 \pi (n-1)/N` is the fractional year in radians,
    :math:`n` the day of the year (1 for Jan, 1st) and :math:`N` the number
    of days in the year.

    Parameters
    ----------
    dates:
        The dates to determine the solar declination.

    Returns
    -------
    solar_declination:
        The solar declination in radians.

    Examples
    --------
        >>> solar_declination('2001-01-01')*180/np.pi
        <Quantity(-23.095402297984275, 'radian')>
        >>> solar_declination(['2000-01-01', '2000-02-01'])
        <Quantity([-0.4030891  -0.30540097], 'radian')>
        >>> solar_declination('2010-06-21T01:00')*180/np.pi
        <Quantity(23.448121292154948, 'radian')>
    """
    dates = pd.to_datetime(dates)

    # Solar declination
    fyr = (2 * np.pi * (dates.dayofyear - 1.0 + (dates.hour - 12.0)/24.0) /
           np.where(dates.is_leap_year, 366, 365))
    declination = (0.006918
                   - 0.399912 * np.cos(fyr)
                   + 0.070257 * np.sin(fyr)
                   - 0.006758 * np.cos(2 * fyr)
                   + 0.000907 * np.sin(2 * fyr)
                   - 0.002697 * np.cos(3 * fyr)
                   + 0.001480 * np.sin(3 * fyr))
    try:
        return declination * units.radian
    except TypeError:
        return declination.values * units.radian


@exporter.export
def solar_day_angle(dates):
    r"""Returns the day angle.

    Parameters
    ----------
    dates:
        The dates to determine the solar declination.

    Returns
    -------
    solar_day_angle:
        The solar day angle in radians.
    """
    dates = pd.to_datetime(dates)
    # Calculate day angle j [radians]
    j = 2 * np.pi / 365.25 * dates.dayofyear
    try:
        return j * units.radian
    except TypeError:
        return j.values * units.radian


@exporter.export
def sunset_hour_angle(dates, latitude):
    r"""Returns sunset hour angle.

    Parameters
    ----------
    dates:
        The dates to determine the solar declination.
    latitude : float
        Latitude in decimal degrees, negative for southern hemisphere.

    Returns
    -------
    sunset_hour_angle:
        The sunset hour angle in radians.

    Examples
    --------
        >>> sunset_hour_angle('2007-04-11', 52.0)
        <Quantity(1.7479732750051813, 'radian')>
    """
    rlatitude = _check_radians(latitude * units.deg,
                               max_radians=np.pi * 67 / 180)
    dt = solar_declination(dates)
    # calculate sunset hour angle [radians]
    ws = np.arccos(np.clip(-np.tan(rlatitude) * np.tan(dt), -1., 1.))
    return ws


@exporter.export
def earth_sun_distance(dates):
    """Calculate the distance between the Earth and the Sun.

    Parameters
    ----------
    dates : str, datetime, np.datetime64, pd.datetime and lists thereof
        The date or dates.

    Examples
    --------
        >>> earth_sun_distance(['2000-01-04', '2000-01-05', '2000-01-06'])
        <Quantity([0.9832914  0.98329387 0.98330129], 'astronomical_unit')>
    """
    dates = pd.to_datetime(dates)
    au = (1 - 0.0167086*np.cos(2*np.pi /
                               365.256363*(dates.dayofyear - 4)))
    try:
        return au * units.au
    except TypeError:
        return au.values * units.au


@exporter.export
def daylight_hours(dates, latitude):
    r"""Calculate the maximum extra-terrestrial sunshine duration.

    Calculate the maximum sunshine duration at the top of the atmosphere for
    day of year and latitude.

    Parameters
    ----------
    dates : str, datetime, np.datetime64, pd.datetime and lists thereof
        The date or dates.
    latitude : float
        Latitude in decimal degrees, negative for southern hemisphere.

    Returns
    -------
    `pint.Quantity`
        (float, array) maximum sunshine hours [h].

    Notes
    -----
    Only valid for latitudes between -67 and 67 degrees (i.e. tropics
    and temperate zone).

    References
    ----------

    R.G. Allen, L.S. Pereira, D. Raes and M. Smith (1998). Crop
    Evaporation - Guidelines for computing crop water requirements,
    FAO - Food and Agriculture Organization of the United Nations.
    Irrigation and drainage paper 56, Chapter 3. Rome, Italy.
    (http://www.fao.org/docrep/x0490e/x0490e07.htm)

    Examples
    --------
        >>> daylight_hours('2001-02-19', 60)
        <Quantity(9.188064873858993, 'hour')>
        >>> days = ['2001-04-09', '2001-07-18', '2001-10-26']
        >>> latitude = 52.
        >>> daylight_hours(days, latitude)
        <Quantity([13.22256353 15.98285786  9.89409615], 'hour')>
    """
    # calculate sunset hour angle [radians]
    ws = sunset_hour_angle(dates, latitude)
    # Calculate sunshine duration N [h]
    N = 24 / np.pi * ws * units.hour / units.radian
    return N


@exporter.export
def extraterrestial_radiation(dates, latitude):
    r"""Calculate the maximum extra-terrestrial solar radiation.

    Calculate the maximum solar radiation at the top of the atmosphere for
    day of year and latitude.

    Parameters
    ----------
    dates : str, datetime, np.datetime64, pd.datetime and lists thereof
        The date or dates.
    latitude : float
        Latitude in decimal degrees, negative for southern hemisphere.

    Returns
    -------
    `pint.Quantity`
        maximum solar radiation hours

    Notes
    -----
    Only valid for latitudes between -67 and 67 degrees (i.e. tropics
    and temperate zone).

    References
    ----------

    R.G. Allen, L.S. Pereira, D. Raes and M. Smith (1998). Crop
    Evaporation - Guidelines for computing crop water requirements,
    FAO - Food and Agriculture Organization of the United Nations.
    Irrigation and drainage paper 56, Chapter 3. Rome, Italy.
    (http://www.fao.org/docrep/x0490e/x0490e07.htm)

    Examples
    --------
        >>> extraterrestial_radiation('2001-02-19', 60)
        <Quantity(105.3873311947177, 'watt / meter ** 2')>
        >>> days = ['2001-04-09', '2001-07-18', '2001-10-26']
        >>> latitude = 52.
        >>> extraterrestial_radiation(days,latitude)
        <Quantity([336.36588496 484.77244183 158.93978282], 'watt / meter ** 2')>

    """
    rlatitude = _check_radians(latitude * units.deg,
                               max_radians=np.pi * 67 / 180)
    # calculate solar declination dt [radians]
    dt = solar_declination(dates)
    # calculate sunset hour angle [radians]
    ws = sunset_hour_angle(dates, latitude)
    # Calculate relative distance to sun
    dr = earth_sun_distance(dates) / units.au
    # Calculate Rext
    Rext = (S /
            np.pi * dr *
            (ws * np.sin(rlatitude) * np.sin(dt) +
             np.sin(ws) * np.cos(rlatitude) * np.cos(dt))) / units.radian
    return Rext


@exporter.export
def solar_shortwave_from_temperatures(dates,
                                      latitude,
                                      tmin,
                                      tmax,
                                      Kt=0.170):
    r"""Computes the incoming solar radiations [W.m\ :sup:`-2`.d\ :sup:`-1`]
    from temperatures, as:

    .. math::
       R_s = K_t \cdot \sqrt{T_{max}-T_{min}} \cdot R_a

    where :math:`K_t` is an empirical parameter.
    By default, :math:`K_t=0.171`. Otherwise, :math:`K_t` is estimated from the
    temperature range :math:`\Delta T=T_{max}-T_{min}` as

    .. math::
       K_t = 0.00185 (\Delta T)^2 - 0.0433 \Delta T + 0.4023

    Parameters
    ----------
    tmin, tmax : TimeSeries
        Minimum and maximum temperatures [\u00B0C].

    Kt : {None, float}, optional
        Regression constant used to approximate solar radiations from
        temperatures.  The default (:math:`K_t=0.171` corresponds to the
        standard value suggested by Hargreaves and Samani
        [Hargreaves_Samani_1985]_.  If None, :math:`K_t` is evaluated from the
        temperatures range :math:`\Delta T` as
        :math:`K_t = 0.00185 {\Delta T}^2 - 0.0433 {\Delta T} + 0.4023`.

    Examples
    --------
        >>> solar_shortwave_from_temperatures('2001-02-19', 60, 5, 20)
        <Quantity(69.38777436512581, 'watt / meter ** 2')>
    """
    trange = (tmax - tmin)
    if Kt is None:
        Kt = 0.00185 * trange ** 2 - 0.0433 * trange + 0.4023
    sol_rad = (Kt *
               np.ma.sqrt(trange) *
               extraterrestial_radiation(dates, latitude))
    return sol_rad


@exporter.export
def solar_shortwave_from_sunshine_hours(dates,
                                        latitude,
                                        sunshine_hours=None,
                                        a_s=0.25,
                                        b_s=0.50):
    r"""Computes the incoming solar (shortwave) radiations :math:`Rs` from
    actual sunshine hours as :

    .. math::
       R_{s} = \left(a_s + b_s \\frac{D_{act}}{D}\\right) R_a

    where :math:`R_a` is the extraterrestrial radiation
    (:attr:`extraterrestrial_solar_radiations`);
    :math:`D_{act}/D` is the relative sunshine duration as the ratio of the
    actual duration of sunshine (:math:`D_{act}`) on the maximum possible
    daylight hours (:math:`D`);
    and :math:`a_s` and :math:`b_s` two regression constants.

    In absence of clouds, :math:`R_s = (a_s+b_s) R_a \\approx 0.75 R_a`.

    Parameters
    ----------
    dates : str, datetime, np.datetime64, pd.datetime and lists thereof
        The date or dates.
    latitude : float
        Latitude in decimal degrees, negative for southern hemisphere.
    sunshine_hours :
        optional, default is None.
        If None, set equal to daylight hours.
        Average actual hours of sunshine [hr].
    a_s : float, optional
        Regression constant expressing the fraction of extraterrestrial solar
        radiation reaching the ground on overcast days.
        By default, ``a_s = 0.25``.
    b_s : float, optional
        Regression constant related to the amount of extraterrestrial solar
        radiation reaching the ground on clear day.
        By default, ``b_s = 0.5``.

    Examples
    --------
        >>> solar_shortwave_from_sunshine_hours('2001-02-19', 60)
        <Quantity(79.04049839603827, 'watt / meter ** 2')>
    """
    effsunshine = 1.0
    if sunshine_hours is not None:
        effsunshine = sunshine_hours / daylight_hours(dates, latitude)
    sol_rad = ((a_s + b_s * effsunshine) *
               extraterrestial_radiation(dates, latitude))
    return sol_rad


@exporter.export
def clearsky_solar_radiations(dates,
                              latitude,
                              z=0):
    r"""Returns the clear sky solar radiation :math:`R_{so}`
    [W.m\ :sup:`-2`.d\ :sup:`-1`], as:

    .. math::
       R_{so} = (0.75 + 2 10^{-5} z) R_a

    Parameters
    ----------
    z : {float}, optional
        Elevation from sea  level [m]

    Examples
    --------
        >>> clearsky_solar_radiations('2001-02-19', 60)
        <Quantity(79.04049839603827, 'watt / meter ** 2')>
    """
    csrad = (0.75 + 2e-5 * z) * extraterrestial_radiation(dates, latitude)
    return csrad
