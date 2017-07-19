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


@exporter.export
@check_units('[pressure]')
def mean_pressure_weighted(pressure, *args, **kwargs):
    r"""Calculate pressure-weighted mean of an arbitrary variable through a layer.

    Layer top and bottom specified in height or pressure.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile
    *args : `pint.Quantity`
        Parameters for which the pressure-weighted mean is to be calculated.
    heights : `pint.Quantity`, optional
        Heights from sounding. Standard atmosphere heights assumed (if needed)
        if no heights are given.
    bottom: `pint.Quantity`, optional
        The bottom of the layer in either the provided height coordinate
        or in pressure. Don't provide in meters AGL unless the provided
        height coordinate is meters AGL. Default is the first observation,
        assumed to be the surface.
    depth: `pint.Quantity`, optional
        The depth of the layer in meters or hPa.

    Returns
    -------
    `pint.Quantity`
        u_mean: u-component of layer mean wind.
    `pint.Quantity`
        v_mean: v-component of layer mean wind.

    """
    heights = kwargs.pop('heights', None)
    bottom = kwargs.pop('bottom', None)
    depth = kwargs.pop('depth', None)
    ret = []  # Returned variable means in layer
    layer_arg = get_layer(pressure, *args, heights=heights,
                          bottom=bottom, depth=depth)
    layer_p = layer_arg[0]
    layer_arg = layer_arg[1:]
    # Taking the integral of the weights (pressure) to feed into the weighting
    # function. Said integral works out to this function:
    pres_int = 0.5 * (layer_p[-1].magnitude**2 - layer_p[0].magnitude**2)
    for i, datavar in enumerate(args):
        arg_mean = np.trapz(layer_arg[i] * layer_p, x=layer_p) / pres_int
        ret.append(arg_mean * datavar.units)

    return ret
