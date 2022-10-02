# Copyright (c) 2008,2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Contains calculations related to turbulence and time series perturbations."""

import numpy as np

from .tools import make_take
from ..package_tools import Exporter
from ..xarray import preprocess_and_wrap

exporter = Exporter(globals())


@exporter.export
@preprocess_and_wrap(wrap_like='ts')
def get_perturbation(ts, axis=-1):
    r"""Compute the perturbation from the mean of a time series.

    Parameters
    ----------
    ts : array-like
         The time series from which you wish to find the perturbation
         time series (perturbation from the mean).

    Returns
    -------
    array-like
        The perturbation time series.

    Other Parameters
    ----------------
    axis : int
           The index of the time axis. Default is -1

    Notes
    -----
    The perturbation time series produced by this function is defined as
    the perturbations about the mean:

    .. math:: x(t)^{\prime} = x(t) - \overline{x(t)}

    """
    mean = ts.mean(axis=axis)[make_take(ts.ndim, axis)(None)]
    return ts - mean


@exporter.export
@preprocess_and_wrap(wrap_like='u')
def tke(u, v, w, perturbation=False, axis=-1):
    r"""Compute turbulence kinetic energy.

    Compute the turbulence kinetic energy (e) from the time series of the
    velocity components.

    Parameters
    ----------
    u : array-like
        The wind component along the x-axis
    v : array-like
        The wind component along the y-axis
    w : array-like
        The wind component along the z-axis
    perturbation : bool, optional
                   True if the `u`, `v`, and `w` components of wind speed
                   supplied to the function are perturbation velocities.
                   If False, perturbation velocities will be calculated by
                   removing the mean value from each component.

    Returns
    -------
    array-like
        The corresponding turbulence kinetic energy value

    Other Parameters
    ----------------
    axis : int
           The index of the time axis. Default is -1

    See Also
    --------
    get_perturbation : Used to compute perturbations if `perturbation`
                       is False.

    Notes
    -----
    Turbulence Kinetic Energy is computed as:

    .. math:: e = 0.5 \sqrt{\overline{u^{\prime2}} +
                            \overline{v^{\prime2}} +
                            \overline{w^{\prime2}}},

    where the velocity components

    .. math:: u^{\prime}, v^{\prime}, u^{\prime}

    are perturbation velocities. For more information on the subject, please
    see [Garratt1994]_.

    """
    if not perturbation:
        u = get_perturbation(u, axis=axis)
        v = get_perturbation(v, axis=axis)
        w = get_perturbation(w, axis=axis)

    u_cont = np.mean(u**2, axis=axis)
    v_cont = np.mean(v**2, axis=axis)
    w_cont = np.mean(w**2, axis=axis)

    return 0.5 * np.sqrt(u_cont + v_cont + w_cont)


@exporter.export
@preprocess_and_wrap(wrap_like='vel')
def kinematic_flux(vel, b, perturbation=False, axis=-1):
    r"""Compute the kinematic flux from two time series.

    Compute the kinematic flux from the time series of two variables `vel`
    and b. Note that to be a kinematic flux, at least one variable must be
    a component of velocity.

    Parameters
    ----------
    vel : array-like
        A component of velocity

    b : array-like
        May be a component of velocity or a scalar variable (e.g. Temperature)

    perturbation : bool, optional
        `True` if the `vel` and `b` variables are perturbations. If `False`, perturbations
        will be calculated by removing the mean value from each variable. Defaults to `False`.

    Returns
    -------
    array-like
        The corresponding kinematic flux

    Other Parameters
    ----------------
    axis : int, optional
           The index of the time axis, along which the calculations will be
           performed. Defaults to -1

    Notes
    -----
    A kinematic flux is computed as

    .. math:: \overline{u^{\prime} s^{\prime}}

    where at the prime notation denotes perturbation variables, and at least
    one variable is perturbation velocity. For example, the vertical kinematic
    momentum flux (two velocity components):

    .. math:: \overline{u^{\prime} w^{\prime}}

    or the vertical kinematic heat flux (one velocity component, and one
    scalar):

    .. math:: \overline{w^{\prime} T^{\prime}}

    If perturbation variables are passed into this function (i.e.
    `perturbation` is True), the kinematic flux is computed using the equation
    above.

    However, the equation above can be rewritten as

    .. math:: \overline{us} - \overline{u}~\overline{s}

    which is computationally more efficient. This is how the kinematic flux
    is computed in this function if `perturbation` is False.

    For more information on the subject, please see [Garratt1994]_.

    """
    kf = np.mean(vel * b, axis=axis)
    if not perturbation:
        kf -= np.mean(vel, axis=axis) * np.mean(b, axis=axis)
    return np.atleast_1d(kf)


@exporter.export
@preprocess_and_wrap(wrap_like='u')
def friction_velocity(u, w, v=None, perturbation=False, axis=-1):
    r"""Compute the friction velocity from the time series of velocity components.

    Compute the friction velocity from the time series of the x, z,
    and optionally y, velocity components.

    Parameters
    ----------
    u : array-like
        The wind component along the x-axis
    w : array-like
        The wind component along the z-axis
    v : array-like, optional
        The wind component along the y-axis.

    perturbation : bool, optional
                   True if the `u`, `w`, and `v` components of wind speed
                   supplied to the function are perturbation velocities.
                   If False, perturbation velocities will be calculated by
                   removing the mean value from each component.

    Returns
    -------
    array-like
        The corresponding friction velocity

    Other Parameters
    ----------------
    axis : int
           The index of the time axis. Default is -1

    See Also
    --------
    kinematic_flux : Used to compute the x-component and y-component
                     vertical kinematic momentum flux(es) used in the
                     computation of the friction velocity.

    Notes
    -----
    The Friction Velocity is computed as:

    .. math:: u_{*} = \sqrt[4]{\left(\overline{u^{\prime}w^{\prime}}\right)^2 +
                               \left(\overline{v^{\prime}w^{\prime}}\right)^2},

    where :math: \overline{u^{\prime}w^{\prime}} and
    :math: \overline{v^{\prime}w^{\prime}}
    are the x-component and y-components of the vertical kinematic momentum
    flux, respectively. If the optional v component of velocity is not
    supplied to the function, the computation of the friction velocity is
    reduced to

    .. math:: u_{*} = \sqrt[4]{\left(\overline{u^{\prime}w^{\prime}}\right)^2}

    For more information on the subject, please see [Garratt1994]_.

    """
    uw = kinematic_flux(u, w, perturbation=perturbation, axis=axis)
    kf = uw**2
    if v is not None:
        vw = kinematic_flux(v, w, perturbation=perturbation, axis=axis)
        kf += vw**2
    # the friction velocity is the 4th root of the kinematic momentum flux
    # As an optimization, first do inplace square root, then return the
    # square root of that. This is faster than np.power(..., 0.25)
    np.sqrt(kf, out=kf)
    return np.sqrt(kf)
