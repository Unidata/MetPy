# Copyright (c) 2009,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculation of kinematic parameters (e.g. divergence or vorticity)."""
import numpy as np

from . import coriolis_parameter
from .tools import first_derivative, get_layer_heights, gradient
from .. import constants as mpconsts
from ..cbook import iterable
from ..package_tools import Exporter
from ..units import check_units, concatenate, units
from ..xarray import preprocess_and_wrap

exporter = Exporter(globals())


def _stack(arrs):
    return concatenate([a[np.newaxis] if iterable(a) else a for a in arrs], axis=0)


@exporter.export
@preprocess_and_wrap()
@check_units('[speed]', '[speed]', '[length]', '[length]')
def vorticity(u, v, dx, dy):
    r"""Calculate the vertical vorticity of the horizontal wind.

    Parameters
    ----------
    u : (M, N) `pint.Quantity`
        x component of the wind
    v : (M, N) `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.

    Returns
    -------
    (M, N) `pint.Quantity`
        vertical vorticity

    See Also
    --------
    divergence

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    dudy = first_derivative(u, delta=dy, axis=-2)
    dvdx = first_derivative(v, delta=dx, axis=-1)
    return dvdx - dudy


@exporter.export
@preprocess_and_wrap()
@check_units(dx='[length]', dy='[length]')
def divergence(u, v, dx, dy):
    r"""Calculate the horizontal divergence of a vector.

    Parameters
    ----------
    u : (M, N) `pint.Quantity`
        x component of the vector
    v : (M, N) `pint.Quantity`
        y component of the vector
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.

    Returns
    -------
    (M, N) `pint.Quantity`
        The horizontal divergence

    See Also
    --------
    vorticity

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    dudx = first_derivative(u, delta=dx, axis=-1)
    dvdy = first_derivative(v, delta=dy, axis=-2)
    return dudx + dvdy


@exporter.export
@preprocess_and_wrap()
@check_units('[speed]', '[speed]', '[length]', '[length]')
def shearing_deformation(u, v, dx, dy):
    r"""Calculate the shearing deformation of the horizontal wind.

    Parameters
    ----------
    u : (M, N) `pint.Quantity`
        x component of the wind
    v : (M, N) `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.

    Returns
    -------
    (M, N) `pint.Quantity`
        Shearing Deformation

    See Also
    --------
    stretching_deformation, total_deformation

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    dudy = first_derivative(u, delta=dy, axis=-2)
    dvdx = first_derivative(v, delta=dx, axis=-1)
    return dvdx + dudy


@exporter.export
@preprocess_and_wrap()
@check_units('[speed]', '[speed]', '[length]', '[length]')
def stretching_deformation(u, v, dx, dy):
    r"""Calculate the stretching deformation of the horizontal wind.

    Parameters
    ----------
    u : (M, N) `pint.Quantity`
        x component of the wind
    v : (M, N) `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.

    Returns
    -------
    (M, N) `pint.Quantity`
        Stretching Deformation

    See Also
    --------
    shearing_deformation, total_deformation

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    dudx = first_derivative(u, delta=dx, axis=-1)
    dvdy = first_derivative(v, delta=dy, axis=-2)
    return dudx - dvdy


@exporter.export
@preprocess_and_wrap()
@check_units('[speed]', '[speed]', '[length]', '[length]')
def total_deformation(u, v, dx, dy):
    r"""Calculate the horizontal total deformation of the horizontal wind.

    Parameters
    ----------
    u : (M, N) `pint.Quantity`
        x component of the wind
    v : (M, N) `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.

    Returns
    -------
    (M, N) `pint.Quantity`
        Total Deformation

    See Also
    --------
    shearing_deformation, stretching_deformation

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    dudy, dudx = gradient(u, deltas=(dy, dx), axes=(-2, -1))
    dvdy, dvdx = gradient(v, deltas=(dy, dx), axes=(-2, -1))
    return np.sqrt((dvdx + dudy)**2 + (dudx - dvdy)**2)


@exporter.export
@preprocess_and_wrap()
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
@preprocess_and_wrap()
@check_units('[temperature]', '[speed]', '[speed]', '[length]', '[length]')
def frontogenesis(potential_temperature, u, v, dx, dy):
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
    potential_temperature : (M, N) `pint.Quantity`
        Potential temperature
    u : (M, N) `pint.Quantity`
        x component of the wind
    v : (M, N) `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.

    Returns
    -------
    (M, N) `pint.Quantity`
        2D Frontogenesis in [temperature units]/m/s

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    Conversion factor to go from [temperature units]/m/s to [temperature units/100km/3h]
    :math:`1.08e4*1.e5`

    """
    # Get gradients of potential temperature in both x and y
    ddy_thta = first_derivative(potential_temperature, delta=dy, axis=-2)
    ddx_thta = first_derivative(potential_temperature, delta=dx, axis=-1)

    # Compute the magnitude of the potential temperature gradient
    mag_thta = np.sqrt(ddx_thta**2 + ddy_thta**2)

    # Get the shearing, stretching, and total deformation of the wind field
    shrd = shearing_deformation(u, v, dx, dy)
    strd = stretching_deformation(u, v, dx, dy)
    tdef = total_deformation(u, v, dx, dy)

    # Get the divergence of the wind field
    div = divergence(u, v, dx, dy)

    # Compute the angle (beta) between the wind field and the gradient of potential temperature
    psi = 0.5 * np.arctan2(shrd, strd)
    beta = np.arcsin((-ddx_thta * np.cos(psi) - ddy_thta * np.sin(psi)) / mag_thta)

    return 0.5 * mag_thta * (tdef * np.cos(2 * beta) - div)


@exporter.export
@preprocess_and_wrap()
@check_units(f='[frequency]', dx='[length]', dy='[length]')
def geostrophic_wind(height, f, dx, dy):
    r"""Calculate the geostrophic wind given from the height or geopotential.

    Parameters
    ----------
    height : (M, N) `pint.Quantity`
        The height field, with either leading dimensions of (x, y) or trailing dimensions
        of (y, x), depending on the value of ``dim_order``.
    f : array_like
        The coriolis parameter.  This can be a scalar to be applied
        everywhere or an array of values.
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `height` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `height` along the applicable axis.

    Returns
    -------
    A 2-item tuple of arrays, `pint.Quantity`
        A tuple of the u-component and v-component of the geostrophic wind.

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    if height.dimensionality['[length]'] == 2.0:
        norm_factor = 1. / f
    else:
        norm_factor = mpconsts.g / f

    dhdy = first_derivative(height, delta=dy, axis=-2)
    dhdx = first_derivative(height, delta=dx, axis=-1)
    return -norm_factor * dhdy, norm_factor * dhdx


@exporter.export
@preprocess_and_wrap()
@check_units(f='[frequency]', u='[speed]', v='[speed]', dx='[length]', dy='[length]')
def ageostrophic_wind(height, u, v, f, dx, dy):
    r"""Calculate the ageostrophic wind given from the height or geopotential.

    Parameters
    ----------
    height : (M, N) ndarray
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
        the size of `height` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `height` along the applicable axis.

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
    u_geostrophic, v_geostrophic = geostrophic_wind(height, f, dx, dy)
    return u - u_geostrophic, v - v_geostrophic


@exporter.export
@preprocess_and_wrap()
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
@preprocess_and_wrap()
@check_units('[speed]', '[speed]', '[length]', '[length]')
def absolute_vorticity(u, v, dx, dy, latitude):
    """Calculate the absolute vorticity of the horizontal wind.

    Parameters
    ----------
    u : (M, N) `pint.Quantity`
        x component of the wind
    v : (M, N) `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    latitude : (M, N) ndarray
        latitude of the wind data in radians or with appropriate unit information attached

    Returns
    -------
    (M, N) `pint.Quantity`
        absolute vorticity

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    f = coriolis_parameter(latitude)
    relative_vorticity = vorticity(u, v, dx, dy)
    return relative_vorticity + f


@exporter.export
@preprocess_and_wrap()
@check_units('[temperature]', '[pressure]', '[speed]', '[speed]',
             '[length]', '[length]', '[dimensionless]')
def potential_vorticity_baroclinic(potential_temperature, pressure, u, v, dx, dy, latitude):
    r"""Calculate the baroclinic potential vorticity.

    .. math:: PV = -g \left(\frac{\partial u}{\partial p}\frac{\partial \theta}{\partial y}
              - \frac{\partial v}{\partial p}\frac{\partial \theta}{\partial x}
              + \frac{\partial \theta}{\partial p}(\zeta + f) \right)

    This formula is based on equation 4.5.93 [Bluestein1993]_.

    Parameters
    ----------
    potential_temperature : (P, M, N) `pint.Quantity`
        potential temperature
    pressure : (P, M, N) `pint.Quantity`
        vertical pressures
    u : (P, M, N) `pint.Quantity`
        x component of the wind
    v : (P, M, N) `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    latitude : (M, N) ndarray
        latitude of the wind data in radians or with appropriate unit information attached

    Returns
    -------
    (P, M, N) `pint.Quantity`
        baroclinic potential vorticity

    Notes
    -----
    This function will only work with data that is in (P, Y, X) format. If your data
    is in a different order you will need to re-order your data in order to get correct
    results from this function.

    The same function can be used for isobaric and isentropic PV analysis. Provide winds
    for vorticity calculations on the desired isobaric or isentropic surface. At least three
    layers of pressure/potential temperature are required in order to calculate the vertical
    derivative (one above and below the desired surface). The first two terms will be zero if
    isentropic level data is used due to the gradient of theta in both the x and y-directions
    will be zero since you are on an isentropic surface.

    This function expects pressure/isentropic level to increase with increasing array element
    (e.g., from higher in the atmosphere to closer to the surface. If the pressure array is
    one-dimensional p[:, None, None] can be used to make it appear multi-dimensional.)

    """
    if ((np.shape(potential_temperature)[-3] < 3) or (np.shape(pressure)[-3] < 3)
       or (np.shape(potential_temperature)[-3] != (np.shape(pressure)[-3]))):
        raise ValueError('Length of potential temperature along the pressure axis '
                         '{} must be at least 3.'.format(-3))

    avor = absolute_vorticity(u, v, dx, dy, latitude)
    dthtadp = first_derivative(potential_temperature, x=pressure, axis=-3)

    if ((np.shape(potential_temperature)[-2] == 1)
       and (np.shape(potential_temperature)[-1] == 1)):
        dthtady = 0 * units.K / units.m  # axis=-2 only has one dimension
        dthtadx = 0 * units.K / units.m  # axis=-1 only has one dimension
    else:
        dthtady = first_derivative(potential_temperature, delta=dy, axis=-2)
        dthtadx = first_derivative(potential_temperature, delta=dx, axis=-1)
    dudp = first_derivative(u, x=pressure, axis=-3)
    dvdp = first_derivative(v, x=pressure, axis=-3)

    return (-mpconsts.g * (dudp * dthtady - dvdp * dthtadx
                           + avor * dthtadp)).to(units.kelvin * units.meter**2
                                                 / (units.second * units.kilogram))


@exporter.export
@preprocess_and_wrap()
@check_units('[length]', '[speed]', '[speed]', '[length]', '[length]', '[dimensionless]')
def potential_vorticity_barotropic(height, u, v, dx, dy, latitude):
    r"""Calculate the barotropic (Rossby) potential vorticity.

    .. math:: PV = \frac{f + \zeta}{H}

    This formula is based on equation 7.27 [Hobbs2006]_.

    Parameters
    ----------
    height : (M, N) `pint.Quantity`
        atmospheric height
    u : (M, N) `pint.Quantity`
        x component of the wind
    v : (M, N) `pint.Quantity`
        y component of the wind
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    latitude : (M, N) ndarray
        latitude of the wind data in radians or with appropriate unit information attached

    Returns
    -------
    (M, N) `pint.Quantity`
        barotropic potential vorticity

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    avor = absolute_vorticity(u, v, dx, dy, latitude)
    return (avor / height).to('meter**-1 * second**-1')


@exporter.export
@preprocess_and_wrap()
@check_units('[speed]', '[speed]', '[speed]', '[speed]', '[length]', '[length]',
             '[dimensionless]')
def inertial_advective_wind(u, v, u_geostrophic, v_geostrophic, dx, dy, latitude):
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
    u : (M, N) `pint.Quantity`
        x component of the advecting wind
    v : (M, N) `pint.Quantity`
        y component of the advecting wind
    u_geostrophic : (M, N) `pint.Quantity`
        x component of the geostrophic (advected) wind
    v_geostrophic : (M, N) `pint.Quantity`
        y component of the geostrophic (advected) wind
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    latitude : (M, N) ndarray
        latitude of the wind data in radians or with appropriate unit information attached

    Returns
    -------
    (M, N) `pint.Quantity`
        x component of inertial advective wind
    (M, N) `pint.Quantity`
        y component of inertial advective wind

    Notes
    -----
    Many forms of the inertial advective wind assume the advecting and advected
    wind to both be the geostrophic wind. To do so, pass the x and y components
    of the geostrophic with for u and u_geostrophic/v and v_geostrophic.

    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    f = coriolis_parameter(latitude)

    dugdy, dugdx = gradient(u_geostrophic, deltas=(dy, dx), axes=(-2, -1))
    dvgdy, dvgdx = gradient(v_geostrophic, deltas=(dy, dx), axes=(-2, -1))

    u_component = -(u * dvgdx + v * dvgdy) / f
    v_component = (u * dugdx + v * dugdy) / f

    return u_component, v_component


@exporter.export
@preprocess_and_wrap()
@check_units('[speed]', '[speed]', '[temperature]', '[pressure]', '[length]', '[length]')
def q_vector(u, v, temperature, pressure, dx, dy, static_stability=1):
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
    u : (M, N) `pint.Quantity`
        x component of the wind (geostrophic in QG-theory)
    v : (M, N) `pint.Quantity`
        y component of the wind (geostrophic in QG-theory)
    temperature : (M, N) `pint.Quantity`
        Array of temperature at pressure level
    pressure : `pint.Quantity`
        Pressure at level
    dx : `pint.Quantity`
        The grid spacing(s) in the x-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    dy : `pint.Quantity`
        The grid spacing(s) in the y-direction. If an array, there should be one item less than
        the size of `u` along the applicable axis.
    static_stability : `pint.Quantity`, optional
        The static stability at the pressure level. Defaults to 1 if not given to calculate
        the Q-vector without factoring in static stability.

    Returns
    -------
    tuple of (M, N) `pint.Quantity`
        The components of the Q-vector in the u- and v-directions respectively

    See Also
    --------
    static_stability

    Notes
    -----
    If inputs have more than two dimensions, they are assumed to have either leading dimensions
    of (x, y) or trailing dimensions of (y, x), depending on the value of ``dim_order``.

    """
    dudy, dudx = gradient(u, deltas=(dy, dx), axes=(-2, -1))
    dvdy, dvdx = gradient(v, deltas=(dy, dx), axes=(-2, -1))
    dtempdy, dtempdx = gradient(temperature, deltas=(dy, dx), axes=(-2, -1))

    q1 = -mpconsts.Rd / (pressure * static_stability) * (dudx * dtempdx + dvdx * dtempdy)
    q2 = -mpconsts.Rd / (pressure * static_stability) * (dudy * dtempdx + dvdy * dtempdy)

    return q1.to_base_units(), q2.to_base_units()
