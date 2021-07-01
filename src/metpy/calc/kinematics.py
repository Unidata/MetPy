# Copyright (c) 2009,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculation of kinematic parameters (e.g. divergence or vorticity)."""
import numpy as np

from . import coriolis_parameter
from .tools import first_derivative, get_layer_heights, gradient
from .. import constants as mpconsts
from ..package_tools import Exporter
from ..units import check_units, units
from ..xarray import add_grid_arguments_from_xarray, preprocess_and_wrap

exporter = Exporter(globals())


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='u')
@check_units('[speed]', '[speed]', dx='[length]', dy='[length]')
def vorticity(u, v, *, dx=None, dy=None, x_dim=-1, y_dim=-2):
    r"""Calculate the vertical vorticity of the horizontal wind.

    Parameters
    ----------
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the wind
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input. Keyword-only argument.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input. Keyword-only argument.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`. Keyword-only argument.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`. Keyword-only argument.

    Returns
    -------
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        vertical vorticity


    .. versionchanged:: 1.0
       Changed signature from ``(u, v, dx, dy)``

    See Also
    --------
    divergence

    """
    dudy = first_derivative(u, delta=dy, axis=y_dim)
    dvdx = first_derivative(v, delta=dx, axis=x_dim)
    return dvdx - dudy


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='u')
@check_units(dx='[length]', dy='[length]')
def divergence(u, v, *, dx=None, dy=None, x_dim=-1, y_dim=-2):
    r"""Calculate the horizontal divergence of a vector.

    Parameters
    ----------
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the vector
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the vector
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input. Keyword-only argument.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input. Keyword-only argument.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`. Keyword-only argument.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`. Keyword-only argument.

    Returns
    -------
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        The horizontal divergence


    .. versionchanged:: 1.0
       Changed signature from ``(u, v, dx, dy)``

    See Also
    --------
    vorticity

    """
    dudx = first_derivative(u, delta=dx, axis=x_dim)
    dvdy = first_derivative(v, delta=dy, axis=y_dim)
    return dudx + dvdy


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='u')
@check_units('[speed]', '[speed]', '[length]', '[length]')
def shearing_deformation(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2):
    r"""Calculate the shearing deformation of the horizontal wind.

    Parameters
    ----------
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the wind
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        Shearing Deformation


    .. versionchanged:: 1.0
       Changed signature from ``(u, v, dx, dy)``

    See Also
    --------
    stretching_deformation, total_deformation

    """
    dudy = first_derivative(u, delta=dy, axis=y_dim)
    dvdx = first_derivative(v, delta=dx, axis=x_dim)
    return dvdx + dudy


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='u')
@check_units('[speed]', '[speed]', '[length]', '[length]')
def stretching_deformation(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2):
    r"""Calculate the stretching deformation of the horizontal wind.

    Parameters
    ----------
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the wind
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        Stretching Deformation


    .. versionchanged:: 1.0
       Changed signature from ``(u, v, dx, dy)``

    See Also
    --------
    shearing_deformation, total_deformation

    """
    dudx = first_derivative(u, delta=dx, axis=x_dim)
    dvdy = first_derivative(v, delta=dy, axis=y_dim)
    return dudx - dvdy


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='u')
@check_units('[speed]', '[speed]', '[length]', '[length]')
def total_deformation(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2):
    r"""Calculate the horizontal total deformation of the horizontal wind.

    Parameters
    ----------
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the wind
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        Total Deformation

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    .. versionchanged:: 1.0
       Changed signature from ``(u, v, dx, dy)``

    See Also
    --------
    shearing_deformation, stretching_deformation

    """
    dudy, dudx = gradient(u, deltas=(dy, dx), axes=(y_dim, x_dim))
    dvdy, dvdx = gradient(v, deltas=(dy, dx), axes=(y_dim, x_dim))
    return np.sqrt((dvdx + dudy)**2 + (dudx - dvdy)**2)


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='scalar', broadcast=('scalar', 'u', 'v', 'w'))
def advection(
    scalar,
    u=None,
    v=None,
    w=None,
    *,
    dx=None,
    dy=None,
    dz=None,
    x_dim=-1,
    y_dim=-2,
    vertical_dim=-3
):
    r"""Calculate the advection of a scalar field by the wind.

    Parameters
    ----------
    scalar : `pint.Quantity` or `xarray.DataArray`
        Array (with N-dimensions) with the quantity to be advected. Use `xarray.DataArray` to
        have dimension ordering automatically determined, otherwise, use default
        [..., Z, Y, X] ordering or specify \*_dim keyword arguments.
    u, v, w : `pint.Quantity` or `xarray.DataArray` or None
        N-dimensional arrays with units of velocity representing the flow, with a component of
        the wind in each dimension. For 1D advection, use 1 positional argument (with `dx` for
        grid spacing and `x_dim` to specify axis if not the default of -1) or use 1 applicable
        keyword argument (u, v, or w) for proper physical dimension (with corresponding `d\*`
        for grid spacing and `\*_dim` to specify axis). For 2D/horizontal advection, use 2
        positional arguments in order for u and v winds respectively (with `dx` and `dy` for
        grid spacings and `x_dim` and `y_dim` keyword arguments to specify axes), or specify u
        and v as keyword arguments (grid spacings and axes likewise). For 3D advection,
        likewise use 3 positional arguments in order for u, v, and w winds respectively or
        specify u, v, and w as keyword arguments (either way, with `dx`, `dy`, `dz` for grid
        spacings and `x_dim`, `y_dim`, and `vertical_dim` for axes).
    dx, dy, dz: `pint.Quantity` or None, optional
        Grid spacing in applicable dimension(s). If using arrays, each array should have one
        item less than the size of `scalar` along the applicable axis. If `scalar` is an
        `xarray.DataArray`, these are automatically determined from its coordinates, and are
        therefore optional. Required if `scalar` is a `pint.Quantity`. These are keyword-only
        arguments.
    x_dim, y_dim, vertical_dim: int or None, optional
        Axis number in applicable dimension(s). Defaults to -1, -2, and -3 respectively for
        (..., Z, Y, X) dimension ordering. If `scalar` is an `xarray.DataArray`, these are
        automatically determined from its coordinates. These are keyword-only arguments.

    Returns
    -------
    `pint.Quantity` or `xarray.DataArray`
        An N-dimensional array containing the advection at all grid points.


    .. versionchanged:: 1.0
       Changed signature from ``(scalar, wind, deltas)``

    """
    return -sum(
        wind * first_derivative(scalar, axis=axis, delta=delta)
        for wind, delta, axis in (
            (u, dx, x_dim),
            (v, dy, y_dim),
            (w, dz, vertical_dim)
        )
        if wind is not None
    )


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(
    wrap_like='potential_temperature',
    broadcast=('potential_temperature', 'u', 'v')
)
@check_units('[temperature]', '[speed]', '[speed]', '[length]', '[length]')
def frontogenesis(potential_temperature, u, v, dx=None, dy=None, x_dim=-1, y_dim=-2):
    r"""Calculate the 2D kinematic frontogenesis of a temperature field.

    The implementation is a form of the Petterssen Frontogenesis and uses the formula
    outlined in [Bluestein1993]_ pg.248-253

    .. math:: F=\frac{1}{2}\left|\nabla \theta\right|[D cos(2\beta)-\delta]

    * :math:`F` is 2D kinematic frontogenesis
    * :math:`\theta` is potential temperature
    * :math:`D` is the total deformation
    * :math:`\beta` is the angle between the axis of dilitation and the isentropes
    * :math:`\delta` is the divergence

    Parameters
    ----------
    potential_temperature : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        Potential temperature
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the wind
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        2D Frontogenesis in [temperature units]/m/s

    Notes
    -----
    Conversion factor to go from [temperature units]/m/s to [temperature units/100km/3h]
    :math:`1.08e4*1.e5`

    .. versionchanged:: 1.0
       Changed signature from ``(thta, u, v, dx, dy, dim_order='yx')``

    """
    # Get gradients of potential temperature in both x and y
    ddy_theta = first_derivative(potential_temperature, delta=dy, axis=y_dim)
    ddx_theta = first_derivative(potential_temperature, delta=dx, axis=x_dim)

    # Compute the magnitude of the potential temperature gradient
    mag_theta = np.sqrt(ddx_theta**2 + ddy_theta**2)

    # Get the shearing, stretching, and total deformation of the wind field
    shrd = shearing_deformation(u, v, dx, dy, x_dim=x_dim, y_dim=y_dim)
    strd = stretching_deformation(u, v, dx, dy, x_dim=x_dim, y_dim=y_dim)
    tdef = total_deformation(u, v, dx, dy, x_dim=x_dim, y_dim=y_dim)

    # Get the divergence of the wind field
    div = divergence(u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim)

    # Compute the angle (beta) between the wind field and the gradient of potential temperature
    psi = 0.5 * np.arctan2(shrd, strd)
    beta = np.arcsin((-ddx_theta * np.cos(psi) - ddy_theta * np.sin(psi)) / mag_theta)

    return 0.5 * mag_theta * (tdef * np.cos(2 * beta) - div)


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like=('height', 'height'), broadcast=('height', 'latitude'))
@check_units(dx='[length]', dy='[length]', latitude='[dimensionless]')
def geostrophic_wind(height, dx=None, dy=None, latitude=None, x_dim=-1, y_dim=-2):
    r"""Calculate the geostrophic wind given from the height or geopotential.

    Parameters
    ----------
    height : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        The height or geopotential field.
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    latitude : `xarray.DataArray` or `pint.Quantity`
        The latitude, which is used to calculate the Coriolis parameter. Its dimensions must
        be broadcastable with those of height. Optional if `xarray.DataArray` with latitude
        coordinate used as input. Note that an argument without units is treated as
        dimensionless, which is equivalent to radians.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    A 2-item tuple of arrays
        A tuple of the u-component and v-component of the geostrophic wind.


    .. versionchanged:: 1.0
       Changed signature from ``(heights, f, dx, dy)``

    """
    f = coriolis_parameter(latitude)
    if height.dimensionality['[length]'] == 2.0:
        norm_factor = 1. / f
    else:
        norm_factor = mpconsts.g / f

    dhdy = first_derivative(height, delta=dy, axis=y_dim)
    dhdx = first_derivative(height, delta=dx, axis=x_dim)
    return -norm_factor * dhdy, norm_factor * dhdx


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(
    wrap_like=('height', 'height'),
    broadcast=('height', 'u', 'v', 'latitude')
)
@check_units(
    u='[speed]',
    v='[speed]',
    dx='[length]',
    dy='[length]',
    latitude='[dimensionless]'
)
def ageostrophic_wind(height, u, v, dx=None, dy=None, latitude=None, x_dim=-1, y_dim=-2):
    r"""Calculate the ageostrophic wind given from the height or geopotential.

    Parameters
    ----------
    height : (M, N) ndarray
        The height or geopotential field.
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        The u wind field.
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        The u wind field.
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    latitude : `xarray.DataArray` or `pint.Quantity`
        The latitude, which is used to calculate the Coriolis parameter. Its dimensions must
        be broadcastable with those of height. Optional if `xarray.DataArray` with latitude
        coordinate used as input. Note that an argument without units is treated as
        dimensionless, which is equivalent to radians.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    A 2-item tuple of arrays
        A tuple of the u-component and v-component of the ageostrophic wind


    .. versionchanged:: 1.0
       Changed signature from ``(heights, f, dx, dy, u, v, dim_order='yx')``

    """
    u_geostrophic, v_geostrophic = geostrophic_wind(
        height,
        dx,
        dy,
        latitude,
        x_dim=x_dim,
        y_dim=y_dim
    )
    return u - u_geostrophic, v - v_geostrophic


@exporter.export
def montgomery_streamfunction(height, temperature):
    r"""Compute the Montgomery Streamfunction on isentropic surfaces.

    The Montgomery Streamfunction is the streamfunction of the geostrophic wind on an
    isentropic surface. This quantity is proportional to the geostrophic wind in isentropic
    coordinates, and its gradient can be interpreted similarly to the pressure gradient in
    isobaric coordinates.

    Parameters
    ----------
    height : `pint.Quantity` or `xarray.DataArray`
        Array of geopotential height of isentropic surfaces
    temperature : `pint.Quantity` or `xarray.DataArray`
        Array of temperature on isentropic surfaces

    Returns
    -------
    stream_func : `pint.Quantity` or `xarray.DataArray`

    Notes
    -----
    The formula used is that from [Lackmann2011]_ p. 69.

    .. math:: \Psi = gZ + C_pT

    * :math:`\Psi` is Montgomery Streamfunction
    * :math:`g` is avg. gravitational acceleration on Earth
    * :math:`Z` is geopotential height of the isentropic surface
    * :math:`C_p` is specific heat at constant pressure for dry air
    * :math:`T` is temperature of the isentropic surface

    See Also
    --------
    get_isentropic_pressure, dry_static_energy

    """
    from . import dry_static_energy
    return dry_static_energy(height, temperature)


@exporter.export
@preprocess_and_wrap()
@check_units('[length]', '[speed]', '[speed]', '[length]',
             bottom='[length]', storm_u='[speed]', storm_v='[speed]')
def storm_relative_helicity(height, u, v, depth, *, bottom=None, storm_u=None, storm_v=None):
    # Partially adapted from similar SharpPy code
    r"""Calculate storm relative helicity.

    Calculates storm relatively helicity following [Markowski2010]_ pg.230-231

    .. math:: \int\limits_0^d (\bar v - c) \cdot \bar\omega_{h} \,dz

    This is applied to the data from a hodograph with the following summation:

    .. math:: \sum_{n = 1}^{N-1} [(u_{n+1} - c_{x})(v_{n} - c_{y}) -
                                  (u_{n} - c_{x})(v_{n+1} - c_{y})]

    Parameters
    ----------
    u : array-like
        U component winds

    v : array-like
        V component winds

    height : array-like
        Atmospheric height, will be converted to AGL

    depth : number
        Depth of the layer

    bottom : number
        Height of layer bottom AGL (default is surface)

    storm_u : number
        U component of storm motion (default is 0 m/s)

    storm_v : number
        V component of storm motion (default is 0 m/s)

    Returns
    -------
    `pint.Quantity`
        Positive storm-relative helicity

    `pint.Quantity`
        Negative storm-relative helicity

    `pint.Quantity`
        Total storm-relative helicity

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    .. versionchanged:: 1.0
       Renamed ``heights`` parameter to ``height`` and converted ``bottom``, ``storm_u``, and
       ``storm_v`` parameters to keyword-only arguments

    """
    if bottom is None:
        bottom = units.Quantity(0, 'm')
    if storm_u is None:
        storm_u = units.Quantity(0, 'm/s')
    if storm_v is None:
        storm_v = units.Quantity(0, 'm/s')

    _, u, v = get_layer_heights(height, depth, u, v, with_agl=True, bottom=bottom)

    storm_relative_u = u - storm_u
    storm_relative_v = v - storm_v

    int_layers = (storm_relative_u[1:] * storm_relative_v[:-1]
                  - storm_relative_u[:-1] * storm_relative_v[1:])

    # Need to manually check for masked value because sum() on masked array with non-default
    # mask will return a masked value rather than 0. See numpy/numpy#11736
    positive_srh = int_layers[int_layers.magnitude > 0.].sum()
    if np.ma.is_masked(positive_srh):
        positive_srh = units.Quantity(0.0, 'meter**2 / second**2')
    negative_srh = int_layers[int_layers.magnitude < 0.].sum()
    if np.ma.is_masked(negative_srh):
        negative_srh = units.Quantity(0.0, 'meter**2 / second**2')

    return (positive_srh.to('meter ** 2 / second ** 2'),
            negative_srh.to('meter ** 2 / second ** 2'),
            (positive_srh + negative_srh).to('meter ** 2 / second ** 2'))


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='u', broadcast=('u', 'v', 'latitude'))
@check_units('[speed]', '[speed]', '[length]', '[length]')
def absolute_vorticity(u, v, dx=None, dy=None, latitude=None, x_dim=-1, y_dim=-2):
    """Calculate the absolute vorticity of the horizontal wind.

    Parameters
    ----------
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the wind
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    latitude : `pint.Quantity`, optional
        Latitude of the wind data. Optional if `xarray.DataArray` with latitude/longitude
        coordinates used as input. Note that an argument without units is treated as
        dimensionless, which translates to radians.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        absolute vorticity


    .. versionchanged:: 1.0
       Changed signature from ``(u, v, dx, dy, lats, dim_order='yx')``

    """
    f = coriolis_parameter(latitude)
    relative_vorticity = vorticity(u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim)
    return relative_vorticity + f


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(
    wrap_like='potential_temperature',
    broadcast=('potential_temperature', 'pressure', 'u', 'v', 'latitude')
)
@check_units('[temperature]', '[pressure]', '[speed]', '[speed]',
             '[length]', '[length]', '[dimensionless]')
def potential_vorticity_baroclinic(
    potential_temperature,
    pressure,
    u,
    v,
    dx=None,
    dy=None,
    latitude=None,
    x_dim=-1,
    y_dim=-2,
    vertical_dim=-3
):
    r"""Calculate the baroclinic potential vorticity.

    .. math:: PV = -g \left(\frac{\partial u}{\partial p}\frac{\partial \theta}{\partial y}
              - \frac{\partial v}{\partial p}\frac{\partial \theta}{\partial x}
              + \frac{\partial \theta}{\partial p}(\zeta + f) \right)

    This formula is based on equation 4.5.93 [Bluestein1993]_

    Parameters
    ----------
    potential_temperature : (..., P, M, N) `xarray.DataArray` or `pint.Quantity`
        potential temperature
    pressure : (..., P, M, N) `xarray.DataArray` or `pint.Quantity`
        vertical pressures
    u : (..., P, M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the wind
    v : (..., P, M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    latitude : `pint.Quantity`, optional
        Latitude of the wind data. Optional if `xarray.DataArray` with latitude/longitude
        coordinates used as input. Note that an argument without units is treated as
        dimensionless, which translates to radians.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Z, Y, X] order).
        Automatically parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Z, Y, X] order).
        Automatically parsed from input if using `xarray.DataArray`.
    vertical_dim : int, optional
        Axis number of vertical dimension. Defaults to -3 (implying [..., Z, Y, X] order).
        Automatically parsed from input if using `xarray.DataArray`.

    Returns
    -------
    (..., P, M, N) `xarray.DataArray` or `pint.Quantity`
        baroclinic potential vorticity

    Notes
    -----
    The same function can be used for isobaric and isentropic PV analysis. Provide winds
    for vorticity calculations on the desired isobaric or isentropic surface. At least three
    layers of pressure/potential temperature are required in order to calculate the vertical
    derivative (one above and below the desired surface). The first two terms will be zero if
    isentropic level data is used due to the gradient of theta in both the x and y-directions
    will be zero since you are on an isentropic surface.

    This function expects pressure/isentropic level to increase with increasing array element
    (e.g., from higher in the atmosphere to closer to the surface. If the pressure array is
    one-dimensional, and not given as `xarray.DataArray`, p[:, None, None] can be used to make
    it appear multi-dimensional.)

    .. versionchanged:: 1.0
       Changed signature from ``(potential_temperature, pressure, u, v, dx, dy, lats)``

    """
    if (
        np.shape(potential_temperature)[vertical_dim] < 3
        or np.shape(pressure)[vertical_dim] < 3
        or np.shape(potential_temperature)[vertical_dim] != np.shape(pressure)[vertical_dim]
    ):
        raise ValueError('Length of potential temperature along the vertical axis '
                         '{} must be at least 3.'.format(vertical_dim))

    avor = absolute_vorticity(u, v, dx, dy, latitude, x_dim=x_dim, y_dim=y_dim)
    dthetadp = first_derivative(potential_temperature, x=pressure, axis=vertical_dim)

    if (
        (np.shape(potential_temperature)[y_dim] == 1)
        and (np.shape(potential_temperature)[x_dim] == 1)
    ):
        dthetady = units.Quantity(0, 'K/m')  # axis=y_dim only has one dimension
        dthetadx = units.Quantity(0, 'K/m')  # axis=x_dim only has one dimension
    else:
        dthetady = first_derivative(potential_temperature, delta=dy, axis=y_dim)
        dthetadx = first_derivative(potential_temperature, delta=dx, axis=x_dim)
    dudp = first_derivative(u, x=pressure, axis=vertical_dim)
    dvdp = first_derivative(v, x=pressure, axis=vertical_dim)

    return (-mpconsts.g * (dudp * dthetady - dvdp * dthetadx
                           + avor * dthetadp)).to('K * m**2 / (s * kg)')


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='height', broadcast=('height', 'u', 'v', 'latitude'))
@check_units('[length]', '[speed]', '[speed]', '[length]', '[length]', '[dimensionless]')
def potential_vorticity_barotropic(
    height,
    u,
    v,
    dx=None,
    dy=None,
    latitude=None,
    x_dim=-1,
    y_dim=-2
):
    r"""Calculate the barotropic (Rossby) potential vorticity.

    .. math:: PV = \frac{f + \zeta}{H}

    This formula is based on equation 7.27 [Hobbs2006]_.

    Parameters
    ----------
    height : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        atmospheric height
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the wind
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    latitude : `pint.Quantity`, optional
        Latitude of the wind data. Optional if `xarray.DataArray` with latitude/longitude
        coordinates used as input. Note that an argument without units is treated as
        dimensionless, which translates to radians.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        barotropic potential vorticity


    .. versionchanged:: 1.0
       Changed signature from ``(heights, u, v, dx, dy, lats, dim_order='yx')``

    """
    avor = absolute_vorticity(u, v, dx, dy, latitude, x_dim=x_dim, y_dim=y_dim)
    return (avor / height).to('meter**-1 * second**-1')


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(
    wrap_like=('u', 'u'),
    broadcast=('u', 'v', 'u_geostrophic', 'v_geostrophic', 'latitude')
)
@check_units('[speed]', '[speed]', '[speed]', '[speed]', '[length]', '[length]',
             '[dimensionless]')
def inertial_advective_wind(
    u,
    v,
    u_geostrophic,
    v_geostrophic,
    dx=None,
    dy=None,
    latitude=None,
    x_dim=-1,
    y_dim=-2
):
    r"""Calculate the inertial advective wind.

    .. math:: \frac{\hat k}{f} \times (\vec V \cdot \nabla)\hat V_g

    .. math:: \frac{\hat k}{f} \times \left[ \left( u \frac{\partial u_g}{\partial x} + v
              \frac{\partial u_g}{\partial y} \right) \hat i + \left( u \frac{\partial v_g}
              {\partial x} + v \frac{\partial v_g}{\partial y} \right) \hat j \right]

    .. math:: \left[ -\frac{1}{f}\left(u \frac{\partial v_g}{\partial x} + v
              \frac{\partial v_g}{\partial y} \right) \right] \hat i + \left[ \frac{1}{f}
              \left( u \frac{\partial u_g}{\partial x} + v \frac{\partial u_g}{\partial y}
              \right) \right] \hat j

    This formula is based on equation 27 of [Rochette2006]_.

    Parameters
    ----------
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the advecting wind
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the advecting wind
    u_geostrophic : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the geostrophic (advected) wind
    v_geostrophic : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the geostrophic (advected) wind
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    latitude : `pint.Quantity`, optional
        Latitude of the wind data. Optional if `xarray.DataArray` with latitude/longitude
        coordinates used as input. Note that an argument without units is treated as
        dimensionless, which translates to radians.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of inertial advective wind
    (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of inertial advective wind

    Notes
    -----
    Many forms of the inertial advective wind assume the advecting and advected
    wind to both be the geostrophic wind. To do so, pass the x and y components
    of the geostrophic wind for u and u_geostrophic/v and v_geostrophic.

    .. versionchanged:: 1.0
       Changed signature from ``(u, v, u_geostrophic, v_geostrophic, dx, dy, lats)``

    """
    f = coriolis_parameter(latitude)

    dugdy, dugdx = gradient(u_geostrophic, deltas=(dy, dx), axes=(y_dim, x_dim))
    dvgdy, dvgdx = gradient(v_geostrophic, deltas=(dy, dx), axes=(y_dim, x_dim))

    u_component = -(u * dvgdx + v * dvgdy) / f
    v_component = (u * dugdx + v * dugdy) / f

    return u_component, v_component


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(
    wrap_like=('u', 'u'),
    broadcast=('u', 'v', 'temperature', 'pressure', 'static_stability')
)
@check_units('[speed]', '[speed]', '[temperature]', '[pressure]', '[length]', '[length]')
def q_vector(
    u,
    v,
    temperature,
    pressure,
    dx=None,
    dy=None,
    static_stability=1,
    x_dim=-1,
    y_dim=-2
):
    r"""Calculate Q-vector at a given pressure level using the u, v winds and temperature.

    .. math:: \vec{Q} = (Q_1, Q_2)
                      =  - \frac{R}{\sigma p}\left(
                               \frac{\partial \vec{v}_g}{\partial x} \cdot \nabla_p T,
                               \frac{\partial \vec{v}_g}{\partial y} \cdot \nabla_p T
                           \right)

    This formula follows equation 5.7.55 from [Bluestein1992]_, and can be used with the
    the below form of the quasigeostrophic omega equation to assess vertical motion
    ([Bluestein1992]_ equation 5.7.54):

    .. math:: \left( \nabla_p^2 + \frac{f_0^2}{\sigma} \frac{\partial^2}{\partial p^2}
                  \right) \omega =
              - 2 \nabla_p \cdot \vec{Q} -
                  \frac{R}{\sigma p} \beta \frac{\partial T}{\partial x}

    Parameters
    ----------
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        x component of the wind (geostrophic in QG-theory)
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        y component of the wind (geostrophic in QG-theory)
    temperature : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        Array of temperature at pressure level
    pressure : `pint.Quantity`
        Pressure at level
    dx : `pint.Quantity`, optional
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    dy : `pint.Quantity`, optional
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis. Optional if `xarray.DataArray` with
        latitude/longitude coordinates used as input.
    static_stability : `pint.Quantity`, optional
        The static stability at the pressure level. Defaults to 1 if not given to calculate
        the Q-vector without factoring in static stability.
    x_dim : int, optional
        Axis number of x dimension. Defaults to -1 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.
    y_dim : int, optional
        Axis number of y dimension. Defaults to -2 (implying [..., Y, X] order). Automatically
        parsed from input if using `xarray.DataArray`.

    Returns
    -------
    tuple of (..., M, N) `xarray.DataArray` or `pint.Quantity`
        The components of the Q-vector in the u- and v-directions respectively


    .. versionchanged:: 1.0
       Changed signature from ``(u, v, temperature, pressure, dx, dy, static_stability=1)``

    See Also
    --------
    static_stability

    """
    dudy, dudx = gradient(u, deltas=(dy, dx), axes=(y_dim, x_dim))
    dvdy, dvdx = gradient(v, deltas=(dy, dx), axes=(y_dim, x_dim))
    dtempdy, dtempdx = gradient(temperature, deltas=(dy, dx), axes=(y_dim, x_dim))

    q1 = -mpconsts.Rd / (pressure * static_stability) * (dudx * dtempdx + dvdx * dtempdy)
    q2 = -mpconsts.Rd / (pressure * static_stability) * (dudy * dtempdx + dvdy * dtempdy)

    return q1.to_base_units(), q2.to_base_units()
