import numpy as np
from ..package_tools import Exporter
from ..constants import g

exporter = Exporter(globals())


def _get_gradients(u, v, dx, dy):
    # Helper function for getting convergence and vorticity from 2D arrays
    dudx, dudy = np.gradient(u, dx, dy)
    dvdx, dvdy = np.gradient(v, dx, dy)
    return dudx, dudy, dvdx, dvdy


@exporter.export
def v_vorticity(u, v, dx, dy):
    r'''Calculate the vertical vorticity of the horizontal wind.

    The grid must have a constant spacing in each direction.

    Parameters
    ----------
    u : (X, Y) ndarray
        x component of the wind
    v : (X, Y) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (X, Y) ndarray
        vertical vorticity

    See Also
    --------
    h_convergence, convergence_vorticity
    '''

    _, dudy, dvdx, _ = _get_gradients(u, v, dx, dy)
    return dvdx - dudy


@exporter.export
def h_convergence(u, v, dx, dy):
    r'''Calculate the horizontal convergence of the horizontal wind.

    The grid must have a constant spacing in each direction.

    Parameters
    ----------
    u : (X, Y) ndarray
        x component of the wind
    v : (X, Y) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    (X, Y) ndarray
        The horizontal convergence

    See Also
    --------
    v_vorticity, convergence_vorticity
    '''

    dudx, _, _, dvdy = _get_gradients(u, v, dx, dy)
    return dudx + dvdy


@exporter.export
def convergence_vorticity(u, v, dx, dy):
    r'''Calculate the horizontal convergence and vertical vorticity of the
    horizontal wind.

    The grid must have a constant spacing in each direction.

    Parameters
    ----------
    u : (X, Y) ndarray
        x component of the wind
    v : (X, Y) ndarray
        y component of the wind
    dx : float
        The grid spacing in the x-direction
    dy : float
        The grid spacing in the y-direction

    Returns
    -------
    convergence, vorticity : tuple of (X, Y) ndarrays
        The horizontal convergence and vertical vorticity, respectively

    See Also
    --------
    v_vorticity, h_convergence

    Notes
    -----
    This is a convenience function that will do less work than calculating
    the horizontal convergence and vertical vorticity separately.
    '''

    dudx, dudy, dvdx, dvdy = _get_gradients(u, v, dx, dy)
    return dudx + dvdy, dvdx - dudy


@exporter.export
def advection(scalar, wind, deltas):
    r'''Calculate the advection of a scalar field by the wind.

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
    '''

    # Gradient returns a list of derivatives along each dimension.  We convert
    # this to an array with dimension as the first index
    grad = np.asarray(np.gradient(scalar, *deltas))

    # This allows passing in a list of wind components or an array
    wind = np.asarray(wind)

    # Make them be at least 2D (handling the 1D case) so that we can do the
    # multiply and sum below
    grad, wind = np.atleast_2d(grad, wind)

    return (-grad * wind).sum(axis=0)


@exporter.export
def geostrophic_wind(heights, f, dx, dy, geopotential=False):
    r'''Calculate the geostrophic wind given from the heights.

    Parameters
    ----------
    heights : (x,y) ndarray
        The height field, given with leading dimensions of x by y.  There
        can be trailing dimensions on the array. These are assumed in meters
        and will be scaled by gravity.
    f : array_like
        The coriolis parameter in s^-1.  This can be a scalar to be applied
        everywhere or an array of values.
    dx : scalar
        The grid spacing in the x-direction in meters.
    dy : scalar
        The grid spacing in the y-direction in meters.

    Returns
    -------
    A 2-item tuple of arrays
        A tuple of the x-component and y-component of the geostropic wind in
        m s^-1.

    Other Parameters
    ----------------
    geopotential : boolean, optional
        If true, the heights are assumed to actually be values of geopotential,
        in units of m^2 s^-2, and the values will not be scaled by gravity.
    '''

    if geopotential:
        norm_factor = 1. / f
    else:
        norm_factor = g / f

    # If heights is has more than 2 dimensions, we need to pass in some dummy
    # grid deltas so that we can still use np.gradient.  It may be better to
    # to loop in this case, but that remains to be done.
    deltas = [dx, dy]
    if heights.ndim > 2:
        deltas = deltas + [1.] * (heights.ndim - 2)

    grad = np.gradient(heights, *deltas)
    dx, dy = grad[0], grad[1]  # This throws away unused gradient components
    return -norm_factor * dy, norm_factor * dx
