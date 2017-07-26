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
def precipitable_water(dewpt, p, top=400 * units('hPa')):
    r"""Calculate precipitable water through the depth of a sounding.

    Default layer depth is sfc-400 hPa. Formula used is:

    .. math:: \frac{1}{pg} \int\limits_0^d x \,dp

    from [Tsonis2008]_, p. 170.

    Parameters
    ----------
    dewpt : `pint.Quantity`
        Atmospheric dewpoint profile
    p : `pint.Quantity`
        Atmospheric pressure profile
    top: `pint.Quantity`, optional
        The top of the layer, specified in pressure. Defaults to 400 hPa.

    Returns
    -------
    `pint.Quantity`
        The precipitable water in the layer

    """
    sort_inds = np.argsort(p[::-1])
    p = p[sort_inds]
    dewpt = dewpt[sort_inds]

    pres_layer, dewpt_layer = get_layer(p, dewpt, depth=p[0] - top)

    w = mixing_ratio(saturation_vapor_pressure(dewpt_layer), pres_layer)
    # Since pressure is in decreasing order, pw will be the negative of what we want.
    # Thus the *-1
    pw = -1. * (np.trapz(w.magnitude, pres_layer.magnitude) * (w.units * pres_layer.units) /
                (g * rho_l))
    return pw.to('millimeters')
