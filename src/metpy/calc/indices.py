# Copyright (c) 2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculation of various derived indices."""
import numpy as np

from .thermo import mixing_ratio, saturation_vapor_pressure
from .tools import _remove_nans, get_layer
from .. import constants as mpconsts
from ..package_tools import Exporter
from ..units import check_units, concatenate, units
from ..xarray import preprocess_and_wrap

exporter = Exporter(globals())


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[temperature]', bottom='[pressure]', top='[pressure]')
def precipitable_water(pressure, dewpoint, *, bottom=None, top=None):
    r"""Calculate precipitable water through the depth of a sounding.

    Formula used is:

    .. math::  -\frac{1}{\rho_l g} \int\limits_{p_\text{bottom}}^{p_\text{top}} r dp

    from [Salby1996]_, p. 28

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile

    dewpoint : `pint.Quantity`
        Atmospheric dewpoint profile

    bottom: `pint.Quantity`, optional
        Bottom of the layer, specified in pressure. Defaults to None (highest pressure).

    top: `pint.Quantity`, optional
        Top of the layer, specified in pressure. Defaults to None (lowest pressure).

    Returns
    -------
    `pint.Quantity`
        Precipitable water in the layer

    Examples
    --------
    >>> pressure = np.array([1000, 950, 900]) * units.hPa
    >>> dewpoint = np.array([20, 15, 10]) * units.degC
    >>> pw = precipitable_water(pressure, dewpoint)

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).

    .. versionchanged:: 1.0
       Signature changed from ``(dewpt, pressure, bottom=None, top=None)``

    """
    # Sort pressure and dewpoint to be in decreasing pressure order (increasing height)
    sort_inds = np.argsort(pressure)[::-1]
    pressure = pressure[sort_inds]
    dewpoint = dewpoint[sort_inds]

    pressure, dewpoint = _remove_nans(pressure, dewpoint)

    min_pressure = np.nanmin(pressure)
    max_pressure = np.nanmax(pressure)

    if top is None:
        top = min_pressure
    elif not min_pressure <= top <= max_pressure:
        raise ValueError(f'The pressure and dewpoint profile ranges from {max_pressure} to '
                         f'{min_pressure}, after removing missing values. {top} is outside '
                         'this range.')

    if bottom is None:
        bottom = max_pressure
    elif not min_pressure <= bottom <= max_pressure:
        raise ValueError(f'The pressure and dewpoint profile ranges from {max_pressure} to '
                         f'{min_pressure}, after removing missing values. {bottom} is outside '
                         'this range.')

    pres_layer, dewpoint_layer = get_layer(pressure, dewpoint, bottom=bottom,
                                           depth=bottom - top)

    w = mixing_ratio(saturation_vapor_pressure(dewpoint_layer), pres_layer)

    # Since pressure is in decreasing order, pw will be the opposite sign of that expected.
    pw = -np.trapz(w, pres_layer) / (mpconsts.g * mpconsts.rho_l)
    return pw.to('millimeters')


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]')
def mean_pressure_weighted(pressure, *args, height=None, bottom=None, depth=None):
    r"""Calculate pressure-weighted mean of an arbitrary variable through a layer.

    Layer top and bottom specified in height or pressure.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile

    args : `pint.Quantity`
        Parameters for which the pressure-weighted mean is to be calculated

    height : `pint.Quantity`, optional
        Heights from sounding. Standard atmosphere heights assumed (if needed)
        if no heights are given.

    bottom: `pint.Quantity`, optional
        The bottom of the layer in either the provided height coordinate
        or in pressure. Don't provide in meters AGL unless the provided
        height coordinate is meters AGL. Default is the first observation,
        assumed to be the surface.

    depth: `pint.Quantity`, optional
        Depth of the layer in meters or hPa

    Returns
    -------
    list of `pint.Quantity`
        list of layer mean value for each profile in args

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height``

    """
    # Split pressure profile from other variables to average
    pres_prof, *others = get_layer(pressure, *args, height=height, bottom=bottom, depth=depth)

    # Taking the integral of the weights (pressure) to feed into the weighting
    # function. Said integral works out to this function:
    pres_int = 0.5 * (pres_prof[-1] ** 2 - pres_prof[0] ** 2)

    # Perform integration on the profile for each variable
    return [np.trapz(var_prof * pres_prof, x=pres_prof) / pres_int for var_prof in others]


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[speed]', '[speed]', '[length]')
def bunkers_storm_motion(pressure, u, v, height):
    r"""Calculate the Bunkers right-mover and left-mover storm motions and sfc-6km mean flow.

    Uses the storm motion calculation from [Bunkers2000]_.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Pressure from sounding

    u : `pint.Quantity`
        U component of the wind

    v : `pint.Quantity`
        V component of the wind

    height : `pint.Quantity`
        Height from sounding

    Returns
    -------
    right_mover: `pint.Quantity`
        U and v component of Bunkers RM storm motion

    left_mover: `pint.Quantity`
        U and v component of Bunkers LM storm motion

    wind_mean: `pint.Quantity`
        U and v component of sfc-6km mean flow

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height``

    """
    # mean wind from sfc-6km
    _, u_mean, v_mean = get_layer(pressure, u, v, height=height,
                                  depth=units.Quantity(6000, 'meter'))
    wind_mean = units.Quantity([np.mean(u_mean).m, np.mean(v_mean).m], u_mean.units)

    # mean wind from sfc-500m
    _, u_500m, v_500m = get_layer(pressure, u, v, height=height,
                                  depth=units.Quantity(500, 'meter'))
    wind_500m = units.Quantity([np.mean(u_500m).m, np.mean(v_500m).m], u_500m.units)

    # mean wind from 5.5-6km
    _, u_5500m, v_5500m = get_layer(pressure, u, v, height=height,
                                    depth=units.Quantity(500, 'meter'),
                                    bottom=height[0] + units.Quantity(5500, 'meter'))
    wind_5500m = units.Quantity([np.mean(u_5500m).m, np.mean(v_5500m).m], u_5500m.units)

    # Calculate the shear vector from sfc-500m to 5.5-6km
    shear = wind_5500m - wind_500m

    # Take the cross product of the wind shear and k, and divide by the vector magnitude and
    # multiply by the deviation empirically calculated in Bunkers (2000) (7.5 m/s)
    shear_cross = concatenate([shear[1], -shear[0]])
    shear_mag = units.Quantity(np.hypot(*(arg.magnitude for arg in shear)), shear.units)
    rdev = shear_cross * (units.Quantity(7.5, 'm/s').to(u.units) / shear_mag)

    # Add the deviations to the layer average wind to get the RM motion
    right_mover = wind_mean + rdev

    # Subtract the deviations to get the LM motion
    left_mover = wind_mean - rdev

    return right_mover, left_mover, wind_mean


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[speed]', '[speed]')
def bulk_shear(pressure, u, v, height=None, bottom=None, depth=None):
    r"""Calculate bulk shear through a layer.

    Layer top and bottom specified in meters or pressure.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile

    u : `pint.Quantity`
        U-component of wind

    v : `pint.Quantity`
        V-component of wind

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
        U-component of layer bulk shear
    v_shr: `pint.Quantity`
        V-component of layer bulk shear

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height``

    """
    _, u_layer, v_layer = get_layer(pressure, u, v, height=height,
                                    bottom=bottom, depth=depth)

    u_shr = u_layer[-1] - u_layer[0]
    v_shr = v_layer[-1] - v_layer[0]

    return u_shr, v_shr


@exporter.export
@preprocess_and_wrap(wrap_like='mucape')
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
    `pint.Quantity`
        Supercell composite

    """
    effective_shear = np.clip(np.atleast_1d(effective_shear), None, units.Quantity(20, 'm/s'))
    effective_shear[effective_shear < units.Quantity(10, 'm/s')] = units.Quantity(0, 'm/s')
    effective_shear = effective_shear / units.Quantity(20, 'm/s')

    return ((mucape / units.Quantity(1000, 'J/kg'))
            * (effective_storm_helicity / units.Quantity(50, 'm^2/s^2'))
            * effective_shear).to('dimensionless')


@exporter.export
@preprocess_and_wrap(wrap_like='sbcape')
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
    `pint.Quantity`
        Significant tornado parameter

    """
    surface_based_lcl_height = np.clip(np.atleast_1d(surface_based_lcl_height),
                                       units.Quantity(1000., 'm'), units.Quantity(2000., 'm'))
    surface_based_lcl_height = ((units.Quantity(2000., 'm') - surface_based_lcl_height)
                                / units.Quantity(1000., 'm'))
    shear_6km = np.clip(np.atleast_1d(shear_6km), None, units.Quantity(30, 'm/s'))
    shear_6km[shear_6km < units.Quantity(12.5, 'm/s')] = units.Quantity(0, 'm/s')
    shear_6km /= units.Quantity(20, 'm/s')

    return ((sbcape / units.Quantity(1500., 'J/kg'))
            * surface_based_lcl_height
            * (storm_helicity_1km / units.Quantity(150., 'm^2/s^2'))
            * shear_6km)


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[speed]', '[speed]', '[length]', '[speed]', '[speed]')
def critical_angle(pressure, u, v, height, u_storm, v_storm):
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
        Pressures from sounding

    u : `pint.Quantity`
        U-component of sounding winds

    v : `pint.Quantity`
        V-component of sounding winds

    height : `pint.Quantity`
        Heights from sounding

    u_storm : `pint.Quantity`
        U-component of storm motion

    v_storm : `pint.Quantity`
        V-component of storm motion

    Returns
    -------
    `pint.Quantity`
        Critical angle in degrees

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height``

    """
    # Convert everything to m/s
    u = u.to('m/s')
    v = v.to('m/s')
    u_storm = u_storm.to('m/s')
    v_storm = v_storm.to('m/s')

    sort_inds = np.argsort(pressure[::-1])
    pressure = pressure[sort_inds]
    height = height[sort_inds]
    u = u[sort_inds]
    v = v[sort_inds]

    # Calculate sfc-500m shear vector
    shr5 = bulk_shear(pressure, u, v, height=height, depth=units.Quantity(500, 'meter'))

    # Make everything relative to the sfc wind orientation
    umn = u_storm - u[0]
    vmn = v_storm - v[0]

    vshr = np.asarray([shr5[0].magnitude, shr5[1].magnitude])
    vsm = np.asarray([umn.magnitude, vmn.magnitude])
    angle_c = np.dot(vshr, vsm) / (np.linalg.norm(vshr) * np.linalg.norm(vsm))
    critical_angle = units.Quantity(np.arccos(angle_c), 'radian')

    return critical_angle.to('degrees')
