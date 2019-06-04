# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Assorted tools in support of interpolation functionality."""

from __future__ import division

import urllib.request

import numpy as np

from ..package_tools import Exporter
from ..units import units

exporter = Exporter(globals())


def calc_kappa(spacing, kappa_star=5.052):
    r"""Calculate the kappa parameter for barnes interpolation.

    Parameters
    ----------
    spacing: float
        Average spacing between observations
    kappa_star: float
        Non-dimensional response parameter. Default 5.052.

    Returns
    -------
        kappa: float

    """
    return kappa_star * (2.0 * spacing / np.pi)**2


@exporter.export
def remove_observations_below_value(x, y, z, val=0):
    r"""Remove all x, y, and z where z is less than val.

    Will not destroy original values.

    Parameters
    ----------
    x: array_like
        x coordinate.
    y: array_like
        y coordinate.
    z: array_like
        Observation value.
    val: float
        Value at which to threshold z.

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        observation values less than val.

    """
    x_ = x[z >= val]
    y_ = y[z >= val]
    z_ = z[z >= val]

    return x_, y_, z_


@exporter.export
def remove_nan_observations(x, y, z):
    r"""Remove all x, y, and z where z is nan.

    Will not destroy original values.

    Parameters
    ----------
    x: array_like
        x coordinate
    y: array_like
        y coordinate
    z: array_like
        observation value

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        nan valued observations.

    """
    x_ = x[~np.isnan(z)]
    y_ = y[~np.isnan(z)]
    z_ = z[~np.isnan(z)]

    return x_, y_, z_


@exporter.export
def remove_repeat_coordinates(x, y, z):
    r"""Remove all x, y, and z where (x,y) is repeated and keep the first occurrence only.

    Will not destroy original values.

    Parameters
    ----------
    x: array_like
        x coordinate
    y: array_like
        y coordinate
    z: array_like
        observation value

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        repeated coordinates.

    """
    coords = []
    variable = []

    for (x_, y_, t_) in zip(x, y, z):
        if (x_, y_) not in coords:
            coords.append((x_, y_))
            variable.append(t_)

    coords = np.array(coords)

    x_ = coords[:, 0]
    y_ = coords[:, 1]

    z_ = np.array(variable)

    return x_, y_, z_


def barnes_weights(sq_dist, kappa, gamma):
    r"""Calculate the Barnes weights from squared distance values.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distances from interpolation point
        associated with each observation in meters.
    kappa: float
        Response parameter for barnes interpolation. Default None.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default None.

    Returns
    -------
    weights: (N, ) ndarray
        Calculated weights for the given observations determined by their distance
        to the interpolation point.

    """
    return np.exp(-1.0 * sq_dist / (kappa * gamma))


def cressman_weights(sq_dist, r):
    r"""Calculate the Cressman weights from squared distance values.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distances from interpolation point
        associated with each observation in meters.
    r: float
        Maximum distance an observation can be from an
        interpolation point to be considered in the inter-
        polation calculation.

    Returns
    -------
    weights: (N, ) ndarray
        Calculated weights for the given observations determined by their distance
        to the interpolation point.

    """
    return (r * r - sq_dist) / (r * r + sq_dist)


@exporter.export
def station_info(stid, with_units=True):
    r"""Provide information about weather stations.

    Parameters
    ----------
    stid: str or iterable object containing strs
        The ICAO or IATA code(s) for which station information is requested.
    with_units: bool
        Whether to include units for values that have them. Default True.

    Returns
    -------
    info: dict
        Information about the station(s) within a dictionary with these keys:
            'state': Two-character ID of the state/province where the station is located,
                      if applicable
            'name': The name of the station
            'lat': The latitude of the station [deg]
            'lon': The longitude of the station [deg]
            'elevation': The elevation of the station [m]
            'country': Two-character ID of the country where the station is located

    """
    # Provide a helper function for later usage
    def str2latlon(s):
        deg = float(s[:3])
        mn = float(s[-3:-1])
        if s[-1] == 'S' or s[-1] == 'W':
            deg = -deg
            mn = -mn
        res = deg + mn / 60.
        return res

    # Various constants describing the underlying data
    url = 'http://weather.rap.ucar.edu/surface/stations.txt'
    state_bnds = slice(0, 2)
    name_bnds = slice(3, 19)
    icao_bnds = slice(20, 24)
    iata_bnds = slice(26, 29)
    lat_bnds = slice(39, 45)
    lon_bnds = slice(47, 54)
    z_bnds = slice(55, 59)
    cntry_bnds = slice(81, 83)

    # Generalize to any number of IDs
    if isinstance(stid, str):
        stid = [stid]

    # Get the station dataset
    infile = urllib.request.urlopen(url)
    data = infile.readlines()

    for s in stid:
        s = s.upper()
        for line_bytes in data:
            line = line_bytes.decode('UTF-8')
            icao = line[icao_bnds]
            iata = line[iata_bnds]
            if len(s) == 3 and s in iata or len(s) == 4 and s in icao:
                state = line[state_bnds].strip()
                name = line[name_bnds].strip()
                lat = str2latlon(line[lat_bnds])
                lon = str2latlon(line[lon_bnds])
                z = float(line[z_bnds])
                cntry = line[cntry_bnds]

                if with_units:
                    lat = lat * units('deg')
                    lon = lon * units('deg')
                    z = z * units('m')

                break
        else:
            raise ValueError('Station {} not found!'.format(s))

        res = {'state': state, 'name': name, 'lat': lat, 'lon': lon, 'elevation': z,
               'country': cntry}
        yield res
