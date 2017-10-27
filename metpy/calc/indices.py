# Copyright (c) 2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculation of various derived indices."""
import numpy as np

from .thermo import mixing_ratio, saturation_vapor_pressure
from .tools import get_layer
from ..constants import g, rho_l
from ..package_tools import Exporter
from ..units import check_units, concatenate, units

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
    # Sort in decreasing pressure order
    sort_inds = np.argsort(pressure[::-1])
    pressure = pressure[sort_inds]
    heights = heights[sort_inds]
    ret = []  # Returned variable means in layer
    layer_arg = []
    for i, datavar in enumerate(args):
        datavar = datavar[sort_inds]
        layer_var = get_layer(pressure, datavar, heights=heights,
                              bottom=bottom, depth=depth)
        layer_p = layer_var[0]
        layer_arg.append(layer_var[1])
    # Taking the integral of the weights (pressure) to feed into the weighting
    # function. Said integral works out to this function:
    pres_int = 0.5 * (layer_p[-1].magnitude**2 - layer_p[0].magnitude**2)
    for i, datavar in enumerate(args):
        arg_mean = np.trapz(layer_arg[i] * layer_p, x=layer_p) / pres_int
        ret.append(arg_mean * datavar.units)

    return ret


@exporter.export
@check_units('[pressure]', '[speed]', '[speed]', '[length]')
def bunkers_storm_motion(pressure, u, v, heights):
    r"""Calculate the Bunkers right-mover and left-mover storm motions and sfc-6km mean flow.

    Uses the storm motion calculation from [Bunkers2000]_.

    Parameters
    ----------
    pressure : array-like
        Pressure from sounding
    u : array-like
        U component of the wind
    v : array-like
        V component of the wind
    heights : array-like
        Heights from sounding

    Returns
    -------
    right_mover: `pint.Quantity`
        U and v component of Bunkers RM storm motion
    left_mover: `pint.Quantity`
        U and v component of Bunkers LM storm motion
    wind_mean: `pint.Quantity`
        U and v component of sfc-6km mean flow

    """
    sort_inds = np.argsort(pressure[::-1])
    pressure = pressure[sort_inds]
    heights = heights[sort_inds]
    u = u[sort_inds]
    v = v[sort_inds]
    # mean wind from sfc-6km
    wind_mean = concatenate(mean_pressure_weighted(pressure, u, v, heights=heights,
                                                   depth=6000 * units('meter')))

    # mean wind from sfc-500m
    wind_500m = concatenate(mean_pressure_weighted(pressure, u, v, heights=heights,
                                                   depth=500 * units('meter')))

    # mean wind from 5.5-6km
    wind_5500m = concatenate(mean_pressure_weighted(pressure, u, v, heights=heights,
                                                    depth=500 * units('meter'),
                                                    bottom=heights[0] +
                                                    5500 * units('meter')))

    # Calculate the shear vector from sfc-500m to 5.5-6km
    shear = wind_5500m - wind_500m

    # Take the cross product of the wind shear and k, and divide by the vector magnitude and
    # multiply by the deviaton empirically calculated in Bunkers (2000) (7.5 m/s)
    shear_cross = concatenate([shear[1], -shear[0]])
    rdev = shear_cross * (7.5 * units('m/s').to(u.units) / np.hypot(*shear))

    # Add the deviations to the layer average wind to get the RM motion
    right_mover = wind_mean + rdev

    # Subtract the deviations to get the LM motion
    left_mover = wind_mean - rdev

    return right_mover, left_mover, wind_mean


@exporter.export
@check_units('[pressure]', '[speed]', '[speed]')
def bulk_shear(pressure, u, v, heights=None, bottom=None, depth=None):
    r"""Calculate bulk shear through a layer.

    Layer top and bottom specified in meters or pressure.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile
    u : `pint.Quantity`
        U-component of wind.
    v : `pint.Quantity`
        V-component of wind.
    height : `pint.Quantity`, optional
        Heights from sounding
    depth: `pint.Quantity`, optional
        The depth of the layer in meters or hPa
    bottom: `pint.Quantity`, optional
        The bottom of the layer in meters or hPa.
        If in meters, must be in the same coordinates as the given
        heights (i.e., don't use meters AGL unless given heights
        are in meters AGL.) Default is the surface (1st observation.)

    Returns
    -------
    u_shr: `pint.Quantity`
        u-component of layer bulk shear
    v_shr: `pint.Quantity`
        v-component of layer bulk shear

    """
    sort_inds = np.argsort(pressure[::-1])
    pressure = pressure[sort_inds]
    heights = heights[sort_inds]
    u = u[sort_inds]
    v = v[sort_inds]
    _, u_layer, v_layer = get_layer(pressure, u, v, heights=heights,
                                    bottom=bottom, depth=depth)

    u_shr = u_layer[-1] - u_layer[0]
    v_shr = v_layer[-1] - v_layer[0]

    return u_shr, v_shr


@exporter.export
def supercell_composite(mucape, effective_storm_helicity, effective_shear):
    r"""Calculate the supercell composite parameter.

    The supercell composite parameter is designed to identify
    environments favorable for the development of supercells,
    and is calculated using the formula developed by
    [Thompson2004]_:

    SCP = (mucape / 1000 J/kg) * (effective_storm_helicity / 50 m^2/s^2) *
          (effective_shear / 20 m/s)

    The effective_shear term is set to zero below 10 m/s and
    capped at 1 when effective_shear exceeds 20 m/s.

    Parameters
    ----------
    mucape : `pint.Quantity`
        Most-unstable CAPE
    effective_storm_helicity : `pint.Quantity`
        Effective-layer storm-relative helicity
    effective_shear : `pint.Quantity`
        Effective bulk shear

    Returns
    -------
    array-like
        supercell composite

    """
    effective_shear = np.clip(effective_shear, None, 20 * units('m/s'))
    effective_shear[effective_shear < 10 * units('m/s')] = 0 * units('m/s')
    effective_shear = effective_shear / (20 * units('m/s'))

    return ((mucape / (1000 * units('J/kg'))) *
            (effective_storm_helicity / (50 * units('m^2/s^2'))) *
            effective_shear).to('dimensionless')


@exporter.export
def significant_tornado(sbcape, sblcl, storm_helicity_1km, shear_6km):
    r"""Calculate the significant tornado parameter (fixed layer).

    The significant tornado parameter is designed to identify
    environments favorable for the production of significant
    tornadoes contingent upon the development of supercells.
    It's calculated according to the formula used on the SPC
    mesoanalysis page, updated in [Thompson2004]_:

    sigtor = (sbcape / 1500 J/kg) * ((2000 m - sblcl) / 1000 m) *
             (storm_helicity_1km / 150 m^s/s^2) * (shear_6km6 / 20 m/s)

    The sblcl term is set to zero when the lcl is above 2000m and
    capped at 1 when below 1000m, and the shr6 term is set to 0
    when shr6 is below 12.5 m/s and maxed out at 1.5 when shr6
    exceeds 30 m/s.

    Parameters
    ----------
    sbcape : `pint.Quantity`
        Surface-based CAPE
    sblcl : `pint.Quantity`
        Surface-based lifted condensation level
    storm_helicity_1km : `pint.Quantity`
        Surface-1km storm-relative helicity
    shear_6km : `pint.Quantity`
        Surface-6km bulk shear

    Returns
    -------
    array-like
        significant tornado parameter

    """
    sblcl = np.clip(sblcl, 1000 * units('meter'), 2000 * units('meter'))
    sblcl[sblcl > 2000 * units('meter')] = 0 * units('meter')
    sblcl = (2000. * units('meter') - sblcl) / (1000. * units('meter'))
    shear_6km = np.clip(shear_6km, None, 30 * units('m/s'))
    shear_6km[shear_6km < 12.5 * units('m/s')] = 0 * units('m/s')
    shear_6km = shear_6km / (20 * units('m/s'))

    return ((sbcape / (1500. * units('J/kg'))) *
            sblcl * (storm_helicity_1km / (150. * units('m^2/s^2'))) * shear_6km)
