# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains updated functions that will be modified in a future release."""

from .calc.kinematics import ensure_yx_order, geostrophic_wind
from .package_tools import Exporter
from .units import check_units
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
