# Copyright (c) 2009,2017,2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculation of kinematic parameters (e.g. divergence or vorticity)."""
from __future__ import division

import functools
import warnings

import numpy as np
from pyproj import Geod

from ..calc.tools import first_derivative, get_layer_heights, gradient
from ..cbook import is_string_like, iterable
from ..constants import Cp_d, g,Re
from ..deprecation import deprecated
from ..package_tools import Exporter
from ..units import atleast_2d, check_units, concatenate, units


exporter = Exporter(globals())



def _stack(arrs):
    return concatenate([a[np.newaxis] for a in arrs], axis=0)


def _is_x_first_dim(dim_order):
    """Determine whether x is the first dimension based on the value of dim_order."""
    if dim_order is None:
        warnings.warn('dim_order is using the default setting ("yx"). This changed in '
                      'version 0.7. It is recommended that you '
                      'specify the appropriate ordering ("xy", "yx") for your data by '
                      'passing the `dim_order` argument to the calculation.', UserWarning)
        dim_order = 'yx'
    return dim_order == 'xy'


def _check_and_flip(arr):
    """Transpose array or list of arrays if they are 2D."""
    if hasattr(arr, 'ndim'):
        if arr.ndim >= 2:
            return arr.T
        else:
            return arr
    elif not is_string_like(arr) and iterable(arr):
        return tuple(_check_and_flip(a) for a in arr)
    else:
        return arr


def ensure_yx_order(func):
    """Wrap a function to ensure all array arguments are y, x ordered, based on kwarg."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check what order we're given
        dim_order = kwargs.pop('dim_order', None)
        x_first = _is_x_first_dim(dim_order)

        # If x is the first dimension, flip (transpose) every array within the function args.
        if x_first:
            args = tuple(_check_and_flip(arr) for arr in args)
            for k, v in kwargs:
                kwargs[k] = _check_and_flip(v)

        ret = func(*args, **kwargs)

        # If we flipped on the way in, need to flip on the way out so that output array(s)
        # match the dimension order of the original input.
        if x_first:
            return _check_and_flip(ret)
        else:
            return ret

    # Inject a docstring for the dim_order argument into the function's docstring.
    dim_order_doc = """
    dim_order : str or ``None``, optional
        The ordering of dimensions in passed in arrays. Can be one of ``None``, ``'xy'``,
        or ``'yx'``. ``'xy'`` indicates that the dimension corresponding to x is the leading
        dimension, followed by y. ``'yx'`` indicates that x is the last dimension, preceded
        by y. ``None`` indicates that the default ordering should be assumed,
        which changed in version 0.7 from 'xy' to 'yx'. Can only be passed as a keyword
        argument, i.e. func(..., dim_order='xy')."""

    # Find the first blank line after the start of the parameters section
    params = wrapper.__doc__.find('Parameters')
    blank = wrapper.__doc__.find('\n\n', params)
    wrapper.__doc__ = wrapper.__doc__[:blank] + dim_order_doc + wrapper.__doc__[blank:]

    return wrapper


@exporter.export
@ensure_yx_order
def vorticity(u, v, dx, dy):
    r"""Calculate the vertical vorticity of the horizontal wind.

    Parameters
    ----------
    u : (M, N) ndarray
        x component of the wind
    v : (M, N) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (M, N) ndarray
        vertical vorticity

    See Also
    --------
    divergence

    """
    dudy = first_derivative(u, delta=dy, axis=0)
    dvdx = first_derivative(v, delta=dx, axis=1)
    return dvdx - dudy


@exporter.export
@deprecated('0.7', addendum=' This function has been renamed vorticity.',
            pending=False)
def v_vorticity(u, v, dx, dy, dim_order='xy'):
    """Wrap vorticity for deprecated v_vorticity function."""
    return vorticity(u, v, dx, dy, dim_order=dim_order)


v_vorticity.__doc__ = (vorticity.__doc__ +
                       '\n    .. deprecated:: 0.7.0\n        Function has been renamed to '
                       '`vorticity` and will be removed from MetPy in 0.9.0.')


@exporter.export
@ensure_yx_order
def divergence(u, v, dx, dy):
    r"""Calculate the horizontal divergence of the horizontal wind.

    Parameters
    ----------
    u : (M, N) ndarray
        x component of the wind
    v : (M, N) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (M, N) ndarray
        The horizontal divergence

    See Also
    --------
    vorticity

    """
    dudx = first_derivative(u, delta=dx, axis=1)
    dvdy = first_derivative(v, delta=dy, axis=0)
    return dudx + dvdy


@exporter.export
@deprecated('0.7', addendum=' This function has been replaced by divergence.',
            pending=False)
def h_convergence(u, v, dx, dy, dim_order='xy'):
    """Wrap divergence for deprecated convergence function."""
    return divergence(u, v, dx, dy, dim_order=dim_order)


h_convergence.__doc__ = (divergence.__doc__ +
                         '\n    .. deprecated:: 0.7.0\n        Function has been renamed to '
                         '`divergence` and will be removed from MetPy in 0.9.0.')


@exporter.export
@deprecated('0.7', addendum=' Use divergence and/or vorticity instead.',
            pending=False)
@ensure_yx_order
def convergence_vorticity(u, v, dx, dy, dim_order='xy'):
    r"""Calculate the horizontal divergence and vertical vorticity of the horizontal wind.

    Parameters
    ----------
    u : (M, N) ndarray
        x component of the wind
    v : (M, N) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    divergence, vorticity : tuple of (M, N) ndarrays
        The horizontal divergence and vertical vorticity, respectively

    See Also
    --------
    vorticity, divergence

    .. deprecated:: 0.7.0
        Function no longer has any performance benefit over individual calls to
        `divergence` and `vorticity` and will be removed from MetPy in 0.9.0.


    """
    dudx = first_derivative(u, delta=dx, axis=1)
    dudy = first_derivative(u, delta=dy, axis=0)
    dvdx = first_derivative(v, delta=dx, axis=1)
    dvdy = first_derivative(v, delta=dy, axis=0)
    return dudx + dvdy, dvdx - dudy


@exporter.export
@ensure_yx_order
def shearing_deformation(u, v, dx, dy):
    r"""Calculate the shearing deformation of the horizontal wind.

    Parameters
    ----------
    u : (M, N) ndarray
        x component of the wind
    v : (M, N) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (M, N) ndarray
        Shearing Deformation

    See Also
    --------
    stretching_deformation, total_deformation

    """
    dudy = first_derivative(u, delta=dy, axis=0)
    dvdx = first_derivative(v, delta=dx, axis=1)
    return dvdx + dudy


@exporter.export
@ensure_yx_order
def stretching_deformation(u, v, dx, dy):
    r"""Calculate the stretching deformation of the horizontal wind.

    Parameters
    ----------
    u : (M, N) ndarray
        x component of the wind
    v : (M, N) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (M, N) ndarray
        Stretching Deformation

    See Also
    --------
    shearing_deformation, total_deformation

    """
    dudx = first_derivative(u, delta=dx, axis=1)
    dvdy = first_derivative(v, delta=dy, axis=0)
    return dudx - dvdy


@exporter.export
@deprecated('0.7', addendum=' Use stretching_deformation and/or shearing_deformation instead.',
            pending=False)
@ensure_yx_order
def shearing_stretching_deformation(u, v, dx, dy):
    r"""Calculate the horizontal shearing and stretching deformation of the horizontal wind.

    Parameters
    ----------
    u : (M, N) ndarray
        x component of the wind
    v : (M, N) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    shearing, strectching : tuple of (M, N) ndarrays
        The horizontal shearing and stretching deformation, respectively

    See Also
    --------
    shearing_deformation, stretching_deformation


    .. deprecated:: 0.7.0
        Function no longer has any performance benefit over individual calls to
        `shearing_deformation` and `stretching_deformation` and will be removed from
        MetPy in 0.9.0.

    """
    dudx = first_derivative(u, delta=dx, axis=1)
    dudy = first_derivative(u, delta=dy, axis=0)
    dvdx = first_derivative(v, delta=dx, axis=1)
    dvdy = first_derivative(v, delta=dy, axis=0)
    return dvdx + dudy, dudx - dvdy


@exporter.export
@ensure_yx_order
def total_deformation(u, v, dx, dy):
    r"""Calculate the horizontal total deformation of the horizontal wind.

    Parameters
    ----------
    u : (M, N) ndarray
        x component of the wind
    v : (M, N) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (M, N) ndarray
        Total Deformation

    See Also
    --------
    shearing_deformation, stretching_deformation

    """
    dudx = first_derivative(u, delta=dx, axis=1)
    dudy = first_derivative(u, delta=dy, axis=0)
    dvdx = first_derivative(v, delta=dx, axis=1)
    dvdy = first_derivative(v, delta=dy, axis=0)
    return np.sqrt((dvdx + dudy)**2 + (dudx - dvdy)**2)


@exporter.export
@ensure_yx_order
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
        Length N sequence of N-dimensional arrays.  Represents the flow,
        with a component of the wind in each dimension.  For example, for
        horizontal advection, this could be a list: [u, v], where u and v
        are each a 2-dimensional array.
    deltas : sequence
        A (length N) sequence containing the grid spacing in each dimension.

    Returns
    -------
    N-dimensional array
        An N-dimensional array containing the advection at all grid points.

    """
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
    grad, wind = atleast_2d(grad, wind)

    return (-grad * wind).sum(axis=0)


@exporter.export
@ensure_yx_order
def frontogenesis(thta, u, v, dx, dy, dim_order='yx'):
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
    thta : (M, N) ndarray
        Potential temperature
    u : (M, N) ndarray
        x component of the wind
    v : (M, N) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (M, N) ndarray
        2D Frontogenesis in [temperature units]/m/s

    Notes
    -----
    Assumes dim_order='yx', unless otherwise specified.

    Conversion factor to go from [temperature units]/m/s to [temperature units/100km/3h]
    :math:`1.08e4*1.e5`

    """
    # Get gradients of potential temperature in both x and y
    ddy_thta = first_derivative(thta, delta=dy, axis=-2)
    ddx_thta = first_derivative(thta, delta=dx, axis=-1)

    # Compute the magnitude of the potential temperature gradient
    mag_thta = np.sqrt(ddx_thta**2 + ddy_thta**2)

    # Get the shearing, stretching, and total deformation of the wind field
    shrd = shearing_deformation(u, v, dx, dy, dim_order=dim_order)
    strd = stretching_deformation(u, v, dx, dy, dim_order=dim_order)
    tdef = total_deformation(u, v, dx, dy, dim_order=dim_order)

    # Get the divergence of the wind field
    div = divergence(u, v, dx, dy, dim_order=dim_order)

    # Compute the angle (beta) between the wind field and the gradient of potential temperature
    psi = 0.5 * np.arctan2(shrd, strd)
    beta = np.arcsin((-ddx_thta * np.cos(psi) - ddy_thta * np.sin(psi)) / mag_thta)

    return 0.5 * mag_thta * (tdef * np.cos(2 * beta) - div)


@exporter.export
@ensure_yx_order
def geostrophic_wind(heights, f, dx, dy):
    r"""Calculate the geostrophic wind given from the heights or geopotential.

    Parameters
    ----------
    heights : (M, N) ndarray
        The height field, with either leading dimensions of (x, y) or trailing dimensions
        of (y, x), depending on the value of ``dim_order``.
    f : array_like
        The coriolis parameter.  This can be a scalar to be applied
        everywhere or an array of values.
    dx : scalar
        The grid spacing in the x-direction
    dy : scalar
        The grid spacing in the y-direction

    Returns
    -------
    A 2-item tuple of arrays
        A tuple of the u-component and v-component of the geostrophic wind.

    """
    if heights.dimensionality['[length]'] == 2.0:
        norm_factor = 1. / f
    else:
        norm_factor = g / f

    dhdy = first_derivative(heights, delta=dy, axis=-2)
    dhdx = first_derivative(heights, delta=dx, axis=-1)
    return -norm_factor * dhdy, norm_factor * dhdx


@exporter.export
@ensure_yx_order
def ageostrophic_wind(heights, f, dx, dy, u, v, dim_order='yx'):
    r"""Calculate the ageostrophic wind given from the heights or geopotential.

    Parameters
    ----------
    heights : (M, N) ndarray
        The height field, with either leading dimensions of (x, y) or trailing dimensions
        of (y, x), depending on the value of ``dim_order``.
    f : array_like
        The coriolis parameter.  This can be a scalar to be applied
        everywhere or an array of values.
    dx : scalar
        The grid spacing in the x-direction
    dy : scalar
        The grid spacing in the y-direction
    u : (M, N) ndarray
        The u wind field, with either leading dimensions of (x, y) or trailing dimensions
        of (y, x), depending on the value of ``dim_order``.
    v : (M, N) ndarray
        The u wind field, with either leading dimensions of (x, y) or trailing dimensions
        of (y, x), depending on the value of ``dim_order``.

    Returns
    -------
    A 2-item tuple of arrays
        A tuple of the u-component and v-component of the ageostrophic wind.

    """
    u_geostrophic, v_geostrophic = geostrophic_wind(heights, f, dx, dy, dim_order=dim_order)
    return u - u_geostrophic, v - v_geostrophic


@exporter.export
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
    return (g * height) + (Cp_d * temperature)


@exporter.export
@check_units('[speed]', '[speed]', '[length]', '[length]', '[length]',
             '[speed]', '[speed]')
def storm_relative_helicity(u, v, heights, depth, bottom=0 * units.m,
                            storm_u=0 * units('m/s'), storm_v=0 * units('m/s')):
    # Partially adapted from similar SharpPy code
    r"""Calculate storm relative helicity.

    Calculates storm relatively helicity following [Markowski2010] 230-231.

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
    `pint.Quantity, pint.Quantity, pint.Quantity`
        positive, negative, total storm-relative helicity

    """
    _, u, v = get_layer_heights(heights, depth, u, v, with_agl=True, bottom=bottom)

    storm_relative_u = u - storm_u
    storm_relative_v = v - storm_v

    int_layers = (storm_relative_u[1:] * storm_relative_v[:-1] -
                  storm_relative_u[:-1] * storm_relative_v[1:])

    positive_srh = int_layers[int_layers.magnitude > 0.].sum()
    negative_srh = int_layers[int_layers.magnitude < 0.].sum()

    return (positive_srh.to('meter ** 2 / second ** 2'),
            negative_srh.to('meter ** 2 / second ** 2'),
            (positive_srh + negative_srh).to('meter ** 2 / second ** 2'))


@deprecated('0.8', addendum=' This function has been replaced by the signed delta distance'
                            'calculation lat_lon_grid_deltas and will be removed in MetPy'
                            ' 0.11.',
            pending=False)
@exporter.export
def lat_lon_grid_spacing(longitude, latitude, **kwargs):
    r"""Calculate the distance between grid points that are in a latitude/longitude format.

    Calculate the distance between grid points when the grid spacing is defined by
    delta lat/lon rather than delta x/y

    Parameters
    ----------
    longitude : array_like
        array of longitudes defining the grid
    latitude : array_like
        array of latitudes defining the grid
    kwargs
        Other keyword arguments to pass to :class:`~pyproj.Geod`

    Returns
    -------
     dx, dy: 2D arrays of distances between grid points in the x and y direction

    Notes
    -----
    Accepts, 1D or 2D arrays for latitude and longitude
    Assumes [Y, X] for 2D arrays

    .. deprecated:: 0.8.0
        Function has been replaced with the signed delta distance calculation
        `lat_lon_grid_deltas` and will be removed from MetPy in 0.11.0.

    """
    # Use the absolute value of the signed function replacing this
    dx, dy = lat_lon_grid_deltas(longitude, latitude, **kwargs)

    return np.abs(dx), np.abs(dy)


@exporter.export
def lat_lon_grid_deltas(longitude, latitude, **kwargs):
    r"""Calculate the delta between grid points that are in a latitude/longitude format.

    Calculate the signed delta distance between grid points when the grid spacing is defined by
    delta lat/lon rather than delta x/y

    Parameters
    ----------
    longitude : array_like
        array of longitudes defining the grid
    latitude : array_like
        array of latitudes defining the grid
    kwargs
        Other keyword arguments to pass to :class:`~pyproj.Geod`

    Returns
    -------
     dx, dy: 2D arrays of signed deltas between grid points in the x and y direction

    Notes
    -----
    Accepts, 1D or 2D arrays for latitude and longitude
    Assumes [Y, X] for 2D arrays

    """
    # Inputs must be the same number of dimensions
    if latitude.ndim != longitude.ndim:
        raise ValueError('Latitude and longitude must have the same number of dimensions.')

    # If we were given 1D arrays, make a mesh grid
    if latitude.ndim < 2:
        longitude, latitude = np.meshgrid(longitude, latitude)

    geod_args = {'ellps': 'sphere'}
    geod_args.update(**kwargs)
    g = Geod(**geod_args)

    forward_az, _, dy = g.inv(longitude[:-1, :], latitude[:-1, :], longitude[1:, :],
                              latitude[1:, :])
    dy[(forward_az < -90.) | (forward_az > 90.)] *= -1

    forward_az, _, dx = g.inv(longitude[:, :-1], latitude[:, :-1], longitude[:, 1:],
                              latitude[:, 1:])
    dx[(forward_az < 0.) | (forward_az > 180.)] *= -1

    return dx * units.meter, dy * units.meter

@exporter.export
def pv_isobaric_rossby(longitude, latitude, pressure,pressuretop,pressurebottom,tmp,tmptop,tmpbottom,u,utop,ubottom,v,vtop,vbottom):

    r"""Calculate the Potential Vorticitys and returns a scalar grid of PV values on an isobaric surface

    This particular implementation of PV is attributed to Rossby's equation and is given as

    gradient of (theta). (er x dvh/dp) + eta * delta theta/delta pressure
    where theta is potential tempeature, vh is the horizontal velocity and p is pressure
    Reference - Dynamics of the Atmosphere - a course in theoretical meteorology A. Bott

    Parameters
    ----------
    longitude : array_like
        array of longitudes defining the grid
    latitude : array_like
        array of latitudes defining the grid
    pressure : scalar value  - value in pascals for the isobaric surface in consideration

    pressuretop : scalar value - value in pascals for the  pressure surface above the surface  in consideration

    pressurebottom : scalar value - value in pascals for the  pressure surface below the surface  in consideration
    
    tmp : 2d array(lat,lon) - temperature units in  kelvin for the isobaric surface
    
    tmptop : 2d array(lat,lon) - temperature units in  kelvin for the isobaric surface above the current surface

    tmpbottom : 2d array(lat,lon) - temperature units in  kelvin for the isobaric surface below the current surface
    u : 2d array (lat,lon) - zonal velocity in m/s for the isobaric surface
    
    utop :2d array (lat,lon) - zonal velocity in m/s for the surface above the isobaric surface

    ubottom : 2d array (lat,lon) -zonal velocity in m/s for the surface below the isobaric surface

    v : 2d array (lat,lon) - meridional velocity in m/s for the isobaric surface
    
    vtop :2d array (lat,lon) - meridional velocity in m/s for the surface above the isobaric surface

    vbottom : 2d array (lat,lon) -meridional velocity in m/s for the surface below the isobaric surface

    Returns
    -------
     PV: 2d numpy array of potential vorticity


    """

    latLen = len(latitude)
    lonLen = len(longitude)
    
    vecfun = np.vectorize(potential_temperature) 
    theta = vecfunc(tmp,pressure)
    
    dpotdx = first_derivative(theta,longitude)
    
    dpotdy = first_derivative(theta,latitude)
    dpi = pressuretop - pressurebottom
    pv = np.empty((latLen,LonLen))
    
    dudp = (utop - ubottom)/dpi
    dvdp = (vtop - vbottom)/dpi
    dth = thetatop - thetabottom
    stability = dth/dpi

    dx,dy=lat_lon_grid_deltas(longitude, latitude)
    dx = Re * dx
    dy = Re * dy
    f = coriolis_parameter(np.deg2rad(lats)).to(units('1/sec'))
    absv = vorticity(u,v,dx,dy,dim_order='yx')+f
    vorcor = (dvdp*dpotdx)-(dudp*dpotdy)
    pv = -g* (absv*stability-vorcor)*10**6

    return pv
