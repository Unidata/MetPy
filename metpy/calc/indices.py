# Copyright (c) 2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculation of various derived indices."""
import numpy as np

from .thermo import mixing_ratio, saturation_vapor_pressure
from .tools import get_layer
from .. import constants as mpconsts
from ..package_tools import Exporter
from ..units import atleast_1d, check_units, concatenate, units
from ..xarray import preprocess_xarray

exporter = Exporter(globals())


@exporter.export
@preprocess_xarray
@check_units('[temperature]', '[pressure]', '[pressure]')
def precipitable_water(dewpt, pressure, bottom=None, top=None):
    r"""Calculate precipitable water through the depth of a sounding.

    Formula used is:

    .. math::  -\frac{1}{\rho_l g} \int\limits_{p_\text{bottom}}^{p_\text{top}} r dp

    from [Salby1996]_, p. 28.

    Parameters
    ----------
    dewpt : `pint.Quantity`
        Atmospheric dewpoint profile
    pressure : `pint.Quantity`
        Atmospheric pressure profile
    bottom: `pint.Quantity`, optional
        Bottom of the layer, specified in pressure. Defaults to None (highest pressure).
    top: `pint.Quantity`, optional
        The top of the layer, specified in pressure. Defaults to None (lowest pressure).

    Returns
    -------
    `pint.Quantity`
        The precipitable water in the layer

    """
    # Sort pressure and dewpoint to be in decreasing pressure order (increasing height)
    sort_inds = np.argsort(pressure)[::-1]
    pressure = pressure[sort_inds]
    dewpt = dewpt[sort_inds]

    if top is None:
        top = np.nanmin(pressure) * pressure.units

    if bottom is None:
        bottom = np.nanmax(pressure) * pressure.units

    pres_layer, dewpt_layer = get_layer(pressure, dewpt, bottom=bottom, depth=bottom - top)

    w = mixing_ratio(saturation_vapor_pressure(dewpt_layer), pres_layer)

    # Since pressure is in decreasing order, pw will be the opposite sign of that expected.
    pw = -1. * (np.trapz(w.magnitude, pres_layer.magnitude) * (w.units * pres_layer.units)
                / (mpconsts.g * mpconsts.rho_l))
    return pw.to('millimeters')


@exporter.export
@preprocess_xarray
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


@exporter.export
@preprocess_xarray
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
    # mean wind from sfc-6km
    wind_mean = concatenate(mean_pressure_weighted(pressure, u, v, heights=heights,
                                                   depth=6000 * units('meter')))

    # mean wind from sfc-500m
    wind_500m = concatenate(mean_pressure_weighted(pressure, u, v, heights=heights,
                                                   depth=500 * units('meter')))

    # mean wind from 5.5-6km
    wind_5500m = concatenate(mean_pressure_weighted(pressure, u, v, heights=heights,
                                                    depth=500 * units('meter'),
                                                    bottom=heights[0] + 5500 * units('meter')))

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
@preprocess_xarray
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
        The depth of the layer in meters or hPa. Defaults to 100 hPa.
    bottom: `pint.Quantity`, optional
        The bottom of the layer in height or pressure coordinates.
        If using a height, it must be in the same coordinates as the given
        heights (i.e., don't use meters AGL unless given heights
        are in meters AGL.) Defaults to the highest pressure or lowest height given.

    Returns
    -------
    u_shr: `pint.Quantity`
        u-component of layer bulk shear
    v_shr: `pint.Quantity`
        v-component of layer bulk shear

    """
    _, u_layer, v_layer = get_layer(pressure, u, v, heights=heights,
                                    bottom=bottom, depth=depth)

    u_shr = u_layer[-1] - u_layer[0]
    v_shr = v_layer[-1] - v_layer[0]

    return u_shr, v_shr


@exporter.export
@preprocess_xarray
@check_units('[energy] / [mass]', '[speed] * [speed]', '[speed]')
def supercell_composite(mucape, effective_storm_helicity, effective_shear):
    r"""Calculate the supercell composite parameter.

    The supercell composite parameter is designed to identify
    environments favorable for the development of supercells,
    and is calculated using the formula developed by
    [Thompson2004]_:

    .. math::  \text{SCP} = \frac{\text{MUCAPE}}{1000 \text{J/kg}} *
               \frac{\text{Effective SRH}}{50 \text{m}^2/\text{s}^2} *
               \frac{\text{Effective Shear}}{20 \text{m/s}}

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
    effective_shear = np.clip(atleast_1d(effective_shear), None, 20 * units('m/s'))
    effective_shear[effective_shear < 10 * units('m/s')] = 0 * units('m/s')
    effective_shear = effective_shear / (20 * units('m/s'))

    return ((mucape / (1000 * units('J/kg')))
            * (effective_storm_helicity / (50 * units('m^2/s^2')))
            * effective_shear).to('dimensionless')


@exporter.export
@preprocess_xarray
@check_units('[energy] / [mass]', '[length]', '[speed] * [speed]', '[speed]')
def significant_tornado(sbcape, surface_based_lcl_height, storm_helicity_1km, shear_6km):
    r"""Calculate the significant tornado parameter (fixed layer).

    The significant tornado parameter is designed to identify
    environments favorable for the production of significant
    tornadoes contingent upon the development of supercells.
    It's calculated according to the formula used on the SPC
    mesoanalysis page, updated in [Thompson2004]_:

    .. math::  \text{SIGTOR} = \frac{\text{SBCAPE}}{1500 \text{J/kg}} * \frac{(2000 \text{m} -
               \text{LCL}_\text{SB})}{1000 \text{m}} *
               \frac{SRH_{\text{1km}}}{150 \text{m}^\text{s}/\text{s}^2} *
               \frac{\text{Shear}_\text{6km}}{20 \text{m/s}}

    The lcl height is set to zero when the lcl is above 2000m and
    capped at 1 when below 1000m, and the shr6 term is set to 0
    when shr6 is below 12.5 m/s and maxed out at 1.5 when shr6
    exceeds 30 m/s.

    Parameters
    ----------
    sbcape : `pint.Quantity`
        Surface-based CAPE
    surface_based_lcl_height : `pint.Quantity`
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
    surface_based_lcl_height = np.clip(atleast_1d(surface_based_lcl_height),
                                       1000 * units.m, 2000 * units.m)
    surface_based_lcl_height[surface_based_lcl_height > 2000 * units.m] = 0 * units.m
    surface_based_lcl_height = ((2000. * units.m - surface_based_lcl_height)
                                / (1000. * units.m))
    shear_6km = np.clip(atleast_1d(shear_6km), None, 30 * units('m/s'))
    shear_6km[shear_6km < 12.5 * units('m/s')] = 0 * units('m/s')
    shear_6km /= 20 * units('m/s')

    return ((sbcape / (1500. * units('J/kg')))
            * surface_based_lcl_height
            * (storm_helicity_1km / (150. * units('m^2/s^2')))
            * shear_6km)


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[speed]', '[speed]', '[length]', '[speed]', '[speed]')
def critical_angle(pressure, u, v, heights, stormu, stormv):
    r"""Calculate the critical angle.

    The critical angle is the angle between the 10m storm-relative inflow vector
    and the 10m-500m shear vector. A critical angle near 90 degrees indicates
    that a storm in this environment on the indicated storm motion vector
    is likely ingesting purely streamwise vorticity into its updraft, and [Esterheld2008]_
    showed that significantly tornadic supercells tend to occur in environments
    with critical angles near 90 degrees.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Pressures from sounding.
    u : `pint.Quantity`
        U-component of sounding winds.
    v : `pint.Quantity`
        V-component of sounding winds.
    heights : `pint.Quantity`
        Heights from sounding.
    stormu : `pint.Quantity`
        U-component of storm motion.
    stormv : `pint.Quantity`
        V-component of storm motion.

    Returns
    -------
    `pint.Quantity`
        critical angle in degrees

    """
    # Convert everything to m/s
    u = u.to('m/s')
    v = v.to('m/s')
    stormu = stormu.to('m/s')
    stormv = stormv.to('m/s')

    sort_inds = np.argsort(pressure[::-1])
    pressure = pressure[sort_inds]
    heights = heights[sort_inds]
    u = u[sort_inds]
    v = v[sort_inds]

    # Calculate sfc-500m shear vector
    shr5 = bulk_shear(pressure, u, v, heights=heights, depth=500 * units('meter'))

    # Make everything relative to the sfc wind orientation
    umn = stormu - u[0]
    vmn = stormv - v[0]

    vshr = np.asarray([shr5[0].magnitude, shr5[1].magnitude])
    vsm = np.asarray([umn.magnitude, vmn.magnitude])
    angle_c = np.dot(vshr, vsm) / (np.linalg.norm(vshr) * np.linalg.norm(vsm))
    critical_angle = np.arccos(angle_c) * units('radian')

    return critical_angle.to('degrees')
