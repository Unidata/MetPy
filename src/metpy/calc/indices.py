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
        Atmospheric pressure profile.

    args : `pint.Quantity`
        Parameters for which the weighted-continuous mean is to be calculated.

    height : `pint.Quantity`, optional
        Heights from sounding. Standard atmosphere heights assumed (if needed)
        if no heights are given.

    bottom: `pint.Quantity`, optional
        The bottom of the layer in either the provided height coordinate
        or in pressure. Don't provide in meters AGL unless the provided
        height coordinate is meters AGL. Default is the first observation,
        assumed to be the surface.

    depth: `pint.Quantity`, optional
        Depth of the layer in meters or hPa.

    Returns
    -------
    list of `pint.Quantity`
        list of layer mean value for each profile in args.

    See Also
    --------
    weighted_continuous_average

    Examples
    --------
    >>> from metpy.calc import mean_pressure_weighted
    >>> from metpy.units import units
    >>> p = [1000, 850, 700, 500] * units.hPa
    >>> T = [30, 15, 5, -5] * units.degC
    >>> mean_pressure_weighted(p, T)
    [<Quantity(298.54368, 'kelvin')>]

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
@check_units('[pressure]')
def weighted_continuous_average(pressure, *args, height=None, bottom=None, depth=None):
    r"""Calculate weighted-continuous mean of an arbitrary variable through a layer.

    Layer top and bottom specified in height or pressure.

    Formula based on that from [Holton2004]_ pg. 76 and the NCL function _wgt_vertical_n_

    .. math::  WCA = \frac{\int_{p_s}^{p} A dp}{\int_{p_s}^{p} dp},

    * :math:`WCA` is the weighted continuous average of a variable.
    * :math:`p` is the bottom pressure level.
    * :math:`p_s` is the top pressure level.
    * :math:`A` is the variable whose weighted continuous average is being calculated.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile.

    args : `pint.Quantity`
        Parameters for which the weighted-continuous mean is to be calculated.

    height : `pint.Quantity`, optional
        Heights from sounding. Standard atmosphere heights assumed (if needed)
        if no heights are given.

    bottom: `pint.Quantity`, optional
        The bottom of the layer in either the provided height coordinate
        or in pressure. Don't provide in meters AGL unless the provided
        height coordinate is meters AGL. Default is the first observation,
        assumed to be the surface.

    depth: `pint.Quantity`, optional
        Depth of the layer in meters or hPa.

    Returns
    -------
    list of `pint.Quantity`
        list of layer mean value for each profile in args.

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    """
    # Split pressure profile from other variables to average
    pres_prof, *others = get_layer(
        pressure, *args, height=height, bottom=bottom, depth=depth
    )

    return [np.trapz(var_prof, x=pres_prof) / (pres_prof[-1] - pres_prof[0])
            for var_prof in others]


@exporter.export
@preprocess_and_wrap()
@check_units('[pressure]', '[speed]', '[speed]', '[length]')
def bunkers_storm_motion(pressure, u, v, height):
    r"""Calculate right-mover and left-mover supercell storm motions using the Bunkers method.

    This is a physically based, shear-relative, and Galilean invariant method for predicting
    supercell motion. Full atmospheric profiles of wind components, as well as pressure and
    heights, need to be provided so that calculation can properly calculate the required
    surface to 6 km mean flow.

    The calculation in summary is (from [Bunkers2000]_):

    * surface to 6 km non-pressure-weighted mean wind
    * a deviation from the sfc to 6 km mean wind of 7.5 m s−1
    * a 5.5 to 6 km mean wind for the head of the vertical wind shear vector
    * a surface to 0.5 km mean wind for the tail of the vertical wind shear vector

    Parameters
    ----------
    pressure : `pint.Quantity`
        Pressure from full profile

    u : `pint.Quantity`
        Full profile of the U-component of the wind

    v : `pint.Quantity`
        Full profile of the V-component of the wind

    height : `pint.Quantity`
        Full profile of height

    Returns
    -------
    right_mover: (`pint.Quantity`, `pint.Quantity`)
        Scalar U- and V- components of Bunkers right-mover storm motion

    left_mover: (`pint.Quantity`, `pint.Quantity`)
        Scalar U- and V- components of Bunkers left-mover storm motion

    wind_mean: (`pint.Quantity`, `pint.Quantity`)
        Scalar U- and V- components of surface to 6 km mean flow

    Examples
    --------
    >>> from metpy.calc import bunkers_storm_motion, wind_components
    >>> from metpy.units import units
    >>> p = [1000, 925, 850, 700, 500, 400] * units.hPa
    >>> h = [250, 700, 1500, 3100, 5720, 7120] * units.meters
    >>> wdir = [165, 180, 190, 210, 220, 250] * units.degree
    >>> sped = [5, 15, 20, 30, 50, 60] * units.knots
    >>> u, v = wind_components(sped, wdir)
    >>> bunkers_storm_motion(p, u, v, h)
    (<Quantity([22.73539654 10.27331352], 'knot')>,
    <Quantity([ 7.27954821 34.99751866], 'knot')>,
    <Quantity([15.00747237 22.63541609], 'knot')>)

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

    Examples
    --------
    >>> from metpy.calc import bulk_shear, wind_components
    >>> from metpy.units import units
    >>> p = [1000, 925, 850, 700, 500] * units.hPa
    >>> wdir = [165, 180, 190, 210, 220] * units.degree
    >>> sped = [5, 15, 20, 30, 50] * units.knots
    >>> u, v = wind_components(sped, wdir)
    >>> bulk_shear(p, u, v)
    (<Quantity(2.41943319, 'knot')>, <Quantity(11.6920573, 'knot')>)

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

    Examples
    --------
    >>> from metpy.calc import supercell_composite
    >>> from metpy.units import units
    >>> supercell_composite(2500 * units('J/kg'), 125 * units('m^2/s^2'),
    ...                     50 * units.knot).to_base_units()
    <Quantity([6.25], 'dimensionless')>

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

    Examples
    --------
    >>> from metpy.calc import significant_tornado
    >>> from metpy.units import units
    >>> significant_tornado(3000 * units('J/kg'), 750 * units.meters,
    ...                     150 * units('m^2/s^2'), 25 * units.knot).to_base_units()
    <Quantity([1.28611111], 'dimensionless')>

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

    Examples
    --------
    >>> from metpy.calc import critical_angle, wind_components
    >>> from metpy.units import units
    >>> p = [1000, 925, 850, 700, 500, 400] * units.hPa
    >>> h = [250, 700, 1500, 3100, 5720, 7120] * units.meters
    >>> wdir = [165, 180, 190, 210, 220, 250] * units.degree
    >>> sped = [5, 15, 20, 30, 50, 60] * units.knots
    >>> u, v = wind_components(sped, wdir)
    >>> critical_angle(p, u, v, h, 7 * units.knots, 7 * units.knots)
    <Quantity(67.0942521, 'degree')>

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
