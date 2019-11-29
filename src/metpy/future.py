# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains updated functions that will be modified in a future release."""

import numpy as np

from . import constants as mpconsts
from .calc.kinematics import ensure_yx_order, geostrophic_wind
from .calc.thermo import mixing_ratio, saturation_vapor_pressure
from .calc.tools import _remove_nans, get_layer, get_layer_heights
from .package_tools import Exporter
from .units import check_units, units
from .xarray import preprocess_xarray


exporter = Exporter(globals())


@exporter.export
@preprocess_xarray
@ensure_yx_order
@check_units(f='[frequency]', u='[speed]', v='[speed]', dx='[length]', dy='[length]')
def ageostrophic_wind(heights, u, v, f, dx, dy, dim_order='yx'):
    r"""Calculate the ageostrophic wind given from the heights or geopotential.

    Parameters
    ----------
    heights : (M, N) ndarray
        The height or geopotential field.
    u : (M, N) `pint.Quantity`
        The u wind field.
    v : (M, N) `pint.Quantity`
        The u wind field.
    f : array_like
        The coriolis parameter.  This can be a scalar to be applied
        everywhere or an array of values.
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `heights` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `heights` along the applicable axis.

    Returns
    -------
    A 2-item tuple of arrays
        A tuple of the u-component and v-component of the ageostrophic wind.

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    This function contains an updated input variable order from the same function in the
    kinematics module. This version will be fully implemented in 1.0 and moved from the
    `future` module back to the `kinematics` module.

    """
    u_geostrophic, v_geostrophic = geostrophic_wind(heights, f, dx, dy, dim_order=dim_order)
    return u - u_geostrophic, v - v_geostrophic


@exporter.export
@preprocess_xarray
@check_units('[pressure]', '[temperature]', bottom='[pressure]', top='[pressure]')
def precipitable_water(pressure, dewpt, *, bottom=None, top=None):
    r"""Calculate precipitable water through the depth of a sounding.

    Formula used is:

    .. math::  -\frac{1}{\rho_l g} \int\limits_{p_\text{bottom}}^{p_\text{top}} r dp

    from [Salby1996]_, p. 28.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Atmospheric pressure profile
    dewpt : `pint.Quantity`
        Atmospheric dewpoint profile
    bottom: `pint.Quantity`, optional
        Bottom of the layer, specified in pressure. Defaults to None (highest pressure).
    top: `pint.Quantity`, optional
        The top of the layer, specified in pressure. Defaults to None (lowest pressure).

    Returns
    -------
    `pint.Quantity`
        The precipitable water in the layer

    Examples
    --------
    >>> pressure = np.array([1000, 950, 900]) * units.hPa
    >>> dewpoint = np.array([20, 15, 10]) * units.degC
    >>> pw = precipitable_water(pressure, dewpoint)

    """
    # Sort pressure and dewpoint to be in decreasing pressure order (increasing height)
    sort_inds = np.argsort(pressure)[::-1]
    pressure = pressure[sort_inds]
    dewpt = dewpt[sort_inds]

    pressure, dewpt = _remove_nans(pressure, dewpt)

    if top is None:
        top = np.nanmin(pressure.magnitude) * pressure.units

    if bottom is None:
        bottom = np.nanmax(pressure.magnitude) * pressure.units

    pres_layer, dewpt_layer = get_layer(pressure, dewpt, bottom=bottom, depth=bottom - top)

    w = mixing_ratio(saturation_vapor_pressure(dewpt_layer), pres_layer)

    # Since pressure is in decreasing order, pw will be the opposite sign of that expected.
    pw = -1. * (np.trapz(w.magnitude, pres_layer.magnitude) * (w.units * pres_layer.units)
                / (mpconsts.g * mpconsts.rho_l))
    return pw.to('millimeters')


@exporter.export
@preprocess_xarray
@check_units('[length]', '[speed]', '[speed]', '[length]',
             bottom='[length]', storm_u='[speed]', storm_v='[speed]')
def storm_relative_helicity(heights, u, v, depth, *, bottom=0 * units.m,
                            storm_u=0 * units('m/s'), storm_v=0 * units('m/s')):
    # Partially adapted from similar SharpPy code
    r"""Calculate storm relative helicity.

    Calculates storm relatively helicity following [Markowski2010]_ 230-231.

    .. math:: \int\limits_0^d (\bar v - c) \cdot \bar\omega_{h} \,dz

    This is applied to the data from a hodograph with the following summation:

    .. math:: \sum_{n = 1}^{N-1} [(u_{n+1} - c_{x})(v_{n} - c_{y}) -
                                  (u_{n} - c_{x})(v_{n+1} - c_{y})]

    Parameters
    ----------
    u : array-like
        u component winds
    v : array-like
        v component winds
    heights : array-like
        atmospheric heights, will be converted to AGL
    depth : number
        depth of the layer
    bottom : number
        height of layer bottom AGL (default is surface)
    storm_u : number
        u component of storm motion (default is 0 m/s)
    storm_v : number
        v component of storm motion (default is 0 m/s)

    Returns
    -------
    `pint.Quantity`
        positive storm-relative helicity
    `pint.Quantity`
        negative storm-relative helicity
    `pint.Quantity`
        total storm-relative helicity

    """
    _, u, v = get_layer_heights(heights, depth, u, v, with_agl=True, bottom=bottom)

    storm_relative_u = u - storm_u
    storm_relative_v = v - storm_v

    int_layers = (storm_relative_u[1:] * storm_relative_v[:-1]
                  - storm_relative_u[:-1] * storm_relative_v[1:])

    # Need to manually check for masked value because sum() on masked array with non-default
    # mask will return a masked value rather than 0. See numpy/numpy#11736
    positive_srh = int_layers[int_layers.magnitude > 0.].sum()
    if np.ma.is_masked(positive_srh):
        positive_srh = 0.0 * units('meter**2 / second**2')
    negative_srh = int_layers[int_layers.magnitude < 0.].sum()
    if np.ma.is_masked(negative_srh):
        negative_srh = 0.0 * units('meter**2 / second**2')

    return (positive_srh.to('meter ** 2 / second ** 2'),
            negative_srh.to('meter ** 2 / second ** 2'),
            (positive_srh + negative_srh).to('meter ** 2 / second ** 2'))
