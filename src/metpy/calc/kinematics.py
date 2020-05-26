# Copyright (c) 2009,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculation of kinematic parameters (e.g. divergence or vorticity)."""
import functools
from inspect import signature
import warnings

import numpy as np
import xarray as xr

from . import coriolis_parameter
from .tools import first_derivative, get_layer_heights, gradient, grid_deltas_from_dataarray
from .. import constants as mpconsts
from ..cbook import iterable
from ..package_tools import Exporter
from ..units import check_units, concatenate, units
from ..xarray import preprocess_and_wrap

exporter = Exporter(globals())


def _stack(arrs):
    return concatenate([a[np.newaxis] if iterable(a) else a for a in arrs], axis=0)


def add_grid_arguments_from_xarray(func):
    """Fill in optional arguments like dx/dy from DataArray arguments."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Search for DataArray with valid latitude and longitude coordinates to find grid
        # deltas and any other needed parameter
        dataarray_arguments = [
            value for value in bound_args.arguments.values()
            if isinstance(value, xr.DataArray)
        ]
        grid_prototype = None
        for da in dataarray_arguments:
            if hasattr(da.metpy, 'latitude') and hasattr(da.metpy, 'longitude'):
                grid_prototype = da
                break

        # Fill in x_dim/y_dim
        if (
            grid_prototype is not None
            and 'x_dim' in bound_args.arguments
            and 'y_dim' in bound_args.arguments
        ):
            try:
                bound_args.arguments['x_dim'] = grid_prototype.metpy.find_axis_number('x')
                bound_args.arguments['y_dim'] = grid_prototype.metpy.find_axis_number('y')
            except AttributeError:
                # If axis number not found, fall back to default but warn.
                warnings.warn('Horizontal dimension numbers not found. Defaulting to '
                              '(..., Y, X) order.')

        # Fill in dx/dy
        if (
            'dx' in bound_args.arguments and bound_args.arguments['dx'] is None
            and 'dy' in bound_args.arguments and bound_args.arguments['dy'] is None
        ):
            if grid_prototype is not None:
                bound_args.arguments['dx'], bound_args.arguments['dy'] = (
                    grid_deltas_from_dataarray(grid_prototype, kind='actual')
                )
            else:
                raise ValueError('Must provide dx/dy arguments or input DataArray with '
                                 'latitude/longitude coordinates.')

        # Fill in latitude
        if 'latitude' in bound_args.arguments and bound_args.arguments['latitude'] is None:
            if grid_prototype is not None:
                bound_args.arguments['latitude'] = (
                    grid_prototype.metpy.latitude.metpy.unit_array
                )
            else:
                raise ValueError('Must provide latitude argument or input DataArray with '
                                 'latitude/longitude coordinates.')

        # Fill in Coriolis parameter
        if 'f' in bound_args.arguments and bound_args.arguments['f'] is None:
            if grid_prototype is not None:
                bound_args.arguments['f'] = coriolis_parameter(
                    grid_prototype.metpy.latitude.metpy.unit_array
                )
            else:
                raise ValueError('Must provide f argument or input DataArray with'
                                 'latitude/longitude coordinates.')

        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='u')
@check_units('[speed]', '[speed]', '[length]', '[length]')
def vorticity(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2):
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
        vertical vorticity

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
def divergence(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2):
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
        The horizontal divergence

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

    See Also
    --------
    shearing_deformation, stretching_deformation

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    dudy, dudx = gradient(u, deltas=(dy, dx), axes=(y_dim, x_dim))
    dvdy, dvdx = gradient(v, deltas=(dy, dx), axes=(y_dim, x_dim))
    return np.sqrt((dvdx + dudy)**2 + (dudx - dvdy)**2)


@exporter.export
def advection(scalar, wind, deltas):
    r"""Calculate the advection of a scalar field by the wind.

    The order of the dimensions of the arrays must match the order in which
    the wind components are given.  For example, if the winds are given [u, v],
    then the scalar and wind arrays must be indexed as x,y (which puts x as the
    rows, not columns).

    Parameters
    ----------
    scalar : N-dimensional array
        Array (with N-dimensions) with the quantity to be advected.
    wind : sequence of arrays
        Length M sequence of N-dimensional arrays.  Represents the flow,
        with a component of the wind in each dimension.  For example, for
        horizontal advection, this could be a list: [u, v], where u and v
        are each a 2-dimensional array.
    deltas : sequence of float or ndarray
        A (length M) sequence containing the grid spacing(s) in each dimension. If using
        arrays, in each array there should be one item less than the size of `scalar` along the
        applicable axis.

    Returns
    -------
    N-dimensional array
        An N-dimensional array containing the advection at all grid points.

    """
    # TODO: This requires custom handling with xarray, since it does not fit the API paradigm
    # of the rest of kinematics

    # This allows passing in a list of wind components or an array.
    wind = _stack(wind)

    # If we have more than one component, we need to reverse the order along the first
    # dimension so that the wind components line up with the
    # order of the gradients from the ..., y, x ordered array.
    if wind.ndim > scalar.ndim:
        wind = wind[::-1]

    # Gradient returns a list of derivatives along each dimension. We convert
    # this to an array with dimension as the first index. Reverse the deltas to line up
    # with the order of the dimensions.
    grad = _stack(gradient(scalar, deltas=deltas[::-1]))

    # Make them be at least 2D (handling the 1D case) so that we can do the
    # multiply and sum below
    grad, wind = np.atleast_2d(grad, wind)

    return (-grad * wind).sum(axis=0)


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
    outlined in [Bluestein1993]_ pg.248-253.

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

    """
    # Get gradients of potential temperature in both x and y
    ddy_thta = first_derivative(potential_temperature, delta=dy, axis=y_dim)
    ddx_thta = first_derivative(potential_temperature, delta=dx, axis=x_dim)

    # Compute the magnitude of the potential temperature gradient
    mag_thta = np.sqrt(ddx_thta**2 + ddy_thta**2)

    # Get the shearing, stretching, and total deformation of the wind field
    shrd = shearing_deformation(u, v, dx, dy, x_dim=x_dim, y_dim=y_dim)
    strd = stretching_deformation(u, v, dx, dy, x_dim=x_dim, y_dim=y_dim)
    tdef = total_deformation(u, v, dx, dy, x_dim=x_dim, y_dim=y_dim)

    # Get the divergence of the wind field
    div = divergence(u, v, dx, dy, x_dim=x_dim, y_dim=y_dim)

    # Compute the angle (beta) between the wind field and the gradient of potential temperature
    psi = 0.5 * np.arctan2(shrd, strd)
    beta = np.arcsin((-ddx_thta * np.cos(psi) - ddy_thta * np.sin(psi)) / mag_thta)

    return 0.5 * mag_thta * (tdef * np.cos(2 * beta) - div)


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like=('height', 'height'), broadcast=('height', 'f'))
@check_units(f='[frequency]', dx='[length]', dy='[length]')
def geostrophic_wind(height, f=None, dx=None, dy=None, x_dim=-1, y_dim=-2):
    r"""Calculate the geostrophic wind given from the height or geopotential.

    Parameters
    ----------
    height : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        The height or geopotential field.
    f : `pint.Quantity`, optional
        The Coriolis parameter. This can be a scalar to be applied everywhere or an array of
        values. Optional if `xarray.DataArray` with latitude/longitude coordinates used as
        input for height.
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
    A 2-item tuple of arrays
        A tuple of the u-component and v-component of the geostrophic wind.

    """
    if height.dimensionality['[length]'] == 2.0:
        norm_factor = 1. / f
    else:
        norm_factor = mpconsts.g / f

    dhdy = first_derivative(height, delta=dy, axis=y_dim)
    dhdx = first_derivative(height, delta=dx, axis=x_dim)
    return -norm_factor * dhdy, norm_factor * dhdx


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like=('height', 'height'), broadcast=('height', 'u', 'v', 'f'))
@check_units(f='[frequency]', u='[speed]', v='[speed]', dx='[length]', dy='[length]')
def ageostrophic_wind(height, u, v, f=None, dx=None, dy=None, x_dim=-1, y_dim=-2):
    r"""Calculate the ageostrophic wind given from the height or geopotential.

    Parameters
    ----------
    height : (M, N) ndarray
        The height or geopotential field.
    u : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        The u wind field.
    v : (..., M, N) `xarray.DataArray` or `pint.Quantity`
        The u wind field.
    f : `pint.Quantity`, optional
        The Coriolis parameter. This can be a scalar to be applied everywhere or an array of
        values. Optional if `xarray.DataArray` with latitude/longitude coordinates used as
        input for height.
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
    A 2-item tuple of arrays
        A tuple of the u-component and v-component of the ageostrophic wind.

    """
    u_geostrophic, v_geostrophic = geostrophic_wind(height, f, dx, dy)
    return u - u_geostrophic, v - v_geostrophic


@exporter.export
@preprocess_and_wrap(wrap_like='height', broadcast=('height', 'temperature'))
@check_units('[length]', '[temperature]')
def montgomery_streamfunction(height, temperature):
    r"""Compute the Montgomery Streamfunction on isentropic surfaces.

    The Montgomery Streamfunction is the streamfunction of the geostrophic wind on an
    isentropic surface. This quantity is proportional to the geostrophic wind in isentropic
    coordinates, and its gradient can be interpreted similarly to the pressure gradient in
    isobaric coordinates.

    Parameters
    ----------
    height : `pint.Quantity`
        Array of geopotential height of isentropic surfaces
    temperature : `pint.Quantity`
        Array of temperature on isentropic surfaces

    Returns
    -------
    stream_func : `pint.Quantity`

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
    get_isentropic_pressure

    """
    return (mpconsts.g * height) + (mpconsts.Cp_d * temperature)


@exporter.export
@preprocess_and_wrap()
@check_units('[length]', '[speed]', '[speed]', '[length]',
             bottom='[length]', storm_u='[speed]', storm_v='[speed]')
def storm_relative_helicity(height, u, v, depth, *, bottom=0 * units.m,
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
    height : array-like
        atmospheric height, will be converted to AGL
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

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    """
    _, u, v = get_layer_heights(height, depth, u, v, with_agl=True, bottom=bottom)

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


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='u')
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

    """
    f = coriolis_parameter(latitude)
    relative_vorticity = vorticity(u, v, dx, dy)
    return relative_vorticity + f


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(
    wrap_like='potential_temperature',
    broadcast=('potential_temperature', 'pressure', 'u', 'v')
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

    This formula is based on equation 4.5.93 [Bluestein1993]_.

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

    """
    if (
        np.shape(potential_temperature)[vertical_dim] < 3
        or np.shape(pressure)[vertical_dim] < 3
        or np.shape(potential_temperature)[vertical_dim] != np.shape(pressure)[vertical_dim]
    ):
        raise ValueError('Length of potential temperature along the vertical axis '
                         '{} must be at least 3.'.format(vertical_dim))

    avor = absolute_vorticity(u, v, dx, dy, latitude, x_dim=x_dim, y_dim=y_dim)
    dthtadp = first_derivative(potential_temperature, x=pressure, axis=vertical_dim)

    if (
        (np.shape(potential_temperature)[y_dim] == 1)
        and (np.shape(potential_temperature)[x_dim] == 1)
    ):
        dthtady = 0 * units.K / units.m  # axis=y_dim only has one dimension
        dthtadx = 0 * units.K / units.m  # axis=x_dim only has one dimension
    else:
        dthtady = first_derivative(potential_temperature, delta=dy, axis=y_dim)
        dthtadx = first_derivative(potential_temperature, delta=dx, axis=x_dim)
    dudp = first_derivative(u, x=pressure, axis=vertical_dim)
    dvdp = first_derivative(v, x=pressure, axis=vertical_dim)

    return (-mpconsts.g * (dudp * dthtady - dvdp * dthtadx
                           + avor * dthtadp)).to(units.kelvin * units.meter**2
                                                 / (units.second * units.kilogram))


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like='height', broadcast=('height', 'u', 'v'))
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

    """
    avor = absolute_vorticity(u, v, dx, dy, latitude, x_dim=x_dim, y_dim=y_dim)
    return (avor / height).to('meter**-1 * second**-1')


@exporter.export
@add_grid_arguments_from_xarray
@preprocess_and_wrap(wrap_like=('u', 'u'))
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
    of the geostrophic with for u and u_geostrophic/v and v_geostrophic.

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
                  \frac{R}{\sigma p} \beta \frac{\partial T}{\partial x}.

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
