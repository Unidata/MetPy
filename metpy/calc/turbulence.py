import numpy as np
from ..package_tools import Exporter

exporter = Exporter(globals())


@exporter.export
def get_perturbation(ts, axis=-1):
    r"""Compute the perturbation from the mean of a time series.

    Parameters
    ----------
    ts : array_like
         The time series from which you wish to find the perturbation
         time series (perturbation from the mean).

    Returns
    -------
    array_like
        The perturbation time series.

    Other Parameters
    ----------------
    axis : int
           The index of the time axis. Default is -1

    Notes
    -----
    The perturbation time series produced by this funcation is defined as
    the perturbations about the mean:

    .. math:: x(t)^{\prime} = x(t) - \overline{x(t)}

    """
    slices = [slice(None)] * ts.ndim
    slices[axis] = None
    return ts - ts.mean(axis=axis)[slices]


@exporter.export
def tke(u, v, w, perturbation=False, axis=-1):
    r"""Compute the turbulence kinetic energy (e) from the time series of the
    velocity components.

    Parameters
    ----------
    u : array_like
        The wind component along the x-axis
    v : array_like
        The wind component along the y-axis
    w : array_like
        The wind component along the z-axis

    perturbation : {False, True}, optional
                   True if the u, v, and w components of wind speed supplied to
                   the function are perturbation velocities. If False,
                   perturbation velocities will be calculated by removing
                   the mean value from each component.

    Returns
    -------
    array_like
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

    are perturbation velocities. Fore more information on the subject, please
    see [1]_.

    References
    ----------
    .. [1] Garratt, J.R., 1994: The Atmospheric Boundary Layer. Cambridge
           University Press, 316 pp.

    """

    if not perturbation:
        u = get_perturbation(u, axis=axis)
        v = get_perturbation(v, axis=axis)
        w = get_perturbation(w, axis=axis)

    u_cont = np.mean(u * u, axis=axis)
    v_cont = np.mean(v * v, axis=axis)
    w_cont = np.mean(w * w, axis=axis)

    return 0.5 * np.sqrt(u_cont + v_cont + w_cont)
