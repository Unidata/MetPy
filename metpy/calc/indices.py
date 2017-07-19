# Copyright (c) 2008-2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculation of various derived indices."""
import numpy as np

from .thermo import mixing_ratio, saturation_vapor_pressure
from .tools import get_layer
from ..constants import g, rho_l
from ..package_tools import Exporter
from ..units import check_units, units

exporter = Exporter(globals())


@exporter.export
@check_units('[temperature]', '[pressure]', '[pressure]')
def precipitable_water(dewpt, pressure, top=400 * units('hPa')):
    r"""Calculate precipitable water through the depth of a sounding.

    Default layer depth is sfc-400 hPa. Formula used is:

    .. math:: \frac{1}{pg} \int\limits_0^d x \,dp

    from [Tsonis2008]_, p. 170.

    Parameters
    ----------
    dewpt : `pint.Quantity`
        Atmospheric dewpoint profile
    pressure : `pint.Quantity`
        Atmospheric pressure profile
    top: `pint.Quantity`, optional
        The top of the layer, specified in pressure. Defaults to 400 hPa.

    Returns
    -------
    `pint.Quantity`
        The precipitable water in the layer

    """
    sort_inds = np.argsort(pressure[::-1])
    pressure = pressure[sort_inds]
    dewpt = dewpt[sort_inds]

    pres_layer, dewpt_layer = get_layer(pressure, dewpt, depth=pressure[0] - top)

    w = mixing_ratio(saturation_vapor_pressure(dewpt_layer), pres_layer)
    # Since pressure is in decreasing order, pw will be the negative of what we want.
    # Thus the *-1
    pw = -1. * (np.trapz(w.magnitude, pres_layer.magnitude) * (w.units * pres_layer.units) /
                (g * rho_l))
    return pw.to('millimeters')


def mean_wind_pressure_weighted(u, v, p, hgt, top, bottom=None, interp=False):
    r"""Calculate pressure-weighted mean wind through a layer.

    Layer top and bottom specified in meters AGL.

    Parameters
    ----------
    u : array-like
        U-component of wind.
    v : array-like
        V-component of wind.
    p : array-like
        Atmospheric pressure profile
    hgt : array-like
        Heights from sounding
    top: `pint.Quantity`
        The top of the layer in meters AGL
    bottom: `pint.Quantity`
        The bottom of the layer in meters AGL.
        Default is the surface.

    Returns
    -------
    `pint.Quantity`
        u_mean: u-component of layer mean wind, in m/s
    `pint.Quantity`
        v_mean: v-component of layer mean wind, in m/s

    """
    u = u.to('meters/second')
    v = v.to('meters/second')

    if bottom:
        depth_s = top - bottom
        bottom = bottom + hgt[0]
    else:
        depth_s = top

    if interp:
        dp = -1
        pressure_top = np.interp(top.magnitude, hgt.magnitude - hgt[0].magnitude,
                                 np.log(p.magnitude))
        pressure_top = np.exp(pressure_top)
        interp_levels = (np.arange(p[0].magnitude, pressure_top + dp, dp)) * units('hPa')
        u_int = log_interp(interp_levels, p, u)
        v_int = log_interp(interp_levels, p, v)
        h_int = log_interp(interp_levels, p, hgt)
        w_int = get_layer(interp_levels, u_int, v_int, heights=h_int,
                          bottom=bottom, depth=depth_s)
    else:
        w_int = get_layer(p, u, v, heights=hgt, bottom=bottom, depth=depth_s)

    u_mean = ma.average(w_int[1], weights = w_int[0]) * units('m/s')
    v_mean = ma.average(w_int[2], weights = w_int[0]) * units('m/s')

    return u_mean, v_mean
