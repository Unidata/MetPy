#!/usr/bin/python
# -*-coding:utf-8 -*-
"""
Contains a collection of boundary layer height estimations.


References:
-----------

[Col14]: Collaud Coen, M., Praz, C., Haefele, A., Ruffieux, D., Kaufmann, P., and Calpini, B. (2014): 
    Determination and climatology of the planetary boundary layer height above the Swiss plateau by in situ and remote sensing measurements as well as by the COSMO-2 model
    Atmos. Chem. Phys., 14, 13205–13221.

[HL06]: Hennemuth, B., & Lammert, A. (2006):
    Determination of the atmospheric boundary layer height from radiosonde and lidar backscatter.
    Boundary-Layer Meteorology, 120(1), 181-200.

[Guo16]: Guo, J., Miao, Y., Zhang, Y., Liu, H., Li, Z., Zhang, W., ... & Zhai, P. (2016):
    The climatology of planetary boundary layer height in China derived from radiosonde and reanalysis data.
    Atmos. Chem. Phys, 16(20), 13309-13319.

[Sei00]: Seidel, D. J., Ao, C. O., & Li, K. (2010):
    Estimating climatological planetary boundary layer heights from radiosonde observations: Comparison of methods and uncertainty analysis.
    Journal of Geophysical Research: Atmospheres, 115(D16).
    
[VH96]: Vogelezang, D. H. P., & Holtslag, A. A. M. (1996):
    Evaluation and model impacts of alternative boundary-layer height formulations.
    Boundary-Layer Meteorology, 81(3-4), 245-269.
"""

import numpy as np
from copy import deepcopy

import metpy.calc as mpcalc
import metpy.constants as mpconsts
from metpy.units import units


def smooth(val, span):
    """Function that calculates the moving average with a given span.
    The span is given in number of points on which the average is made.

    Parameters
    ----------
    val: array-like
        Array of values
    span: int
        Span of the moving average. The higher the smoother

    Returns
    -------
    smoothed_val: array-like
        Array of smoothed values
    """
    N = len(val)
    smoothed_val = deepcopy(val)
    for i in range(N):
        smoothed_val[i] = np.nanmean(val[i - min(span, i) : i + min(span, N - i)])

    return smoothed_val


def bulk_richardson_number(
    height,
    potential_temperature,
    u,
    v,
    idxfoot: int = 0,
    ustar=0 * units.meter_per_second,
):
    r"""Calculate the bulk Richardson number.

    See [VH96], eq. (3):

    .. math::   Ri = (g/\theta) * \frac{(\Delta z)(\Delta \theta)}
             {\left(\Delta u)^2 + (\Delta v)^2 + b(u_*)^2}

    Parameters
    ----------
    height : `pint.Quantity`
        Altitude (metres above ground) of the points in the profile
    potential_temperature : `pint.Quantity`
        Potential temperature profile
    u : `pint.Quantity`
        Zonal wind profile
    v : `pint.Quantity`
        Meridional wind profile
    idxfoot : int, optional
        The index of the foot point (first trusted measure), defaults to 0.

    Returns
    -------
    `pint.Quantity`
        Bulk Richardson number profile
    """

    u[0] = 0 * units.meter_per_second
    v[0] = 0 * units.meter_per_second

    Dtheta = potential_temperature - potential_temperature[idxfoot]
    Du = u - u[idxfoot]
    Dv = v - v[idxfoot]
    Dz = height - height[idxfoot]

    idx0 = Du**2 + Dv**2 + ustar**2 == 0
    if idx0.sum() > 0:
        bRi = np.ones_like(Dtheta) * np.nan * units.dimensionless
        bRi[~idx0] = (
            (mpconsts.g / potential_temperature[~idx0])
            * (Dtheta[~idx0] * Dz[~idx0])
            / (Du[~idx0] ** 2 + Dv[~idx0] ** 2 + ustar**2)
        )
    else:
        bRi = (
            (mpconsts.g / potential_temperature)
            * (Dtheta * Dz)
            / (Du**2 + Dv**2 + ustar**2)
        )

    return bRi


def blh_from_richardson_bulk(
    height,
    potential_temperature,
    u,
    v,
    smoothingspan: int = 10,
    idxfoot: int = 0,
    bri_threshold=0.25 * units.dimensionless,
    ustar=0.1 * units.meter_per_second,
):
    """Calculate atmospheric boundary layer height with the method of
    bulk Richardson number.

    It is the height where the bulk Richardson number exceeds a given threshold.
    See [VH96, Sei00, Col14, Guo16].

    Parameters
    ----------
    height : `pint.Quantity`
        Altitude (metres above ground) of the points in the profile
    potential_temperature : `pint.Quantity`
        Potential temperature profile
    u : `pint.Quantity`
        Zonal wind profile
    v : `pint.Quantity`
        Meridional wind profile
    smoothingspan : int, optional
        The amount of smoothing (number of points in moving average)
    idxfoot : int, optional
        The index of the foot point (first trusted measure), defaults to 0.
    bri_threshold : `pint.Quantity`, optional
        Threshold to exceed to get boundary layer top. Defaults to 0.25
    ustar : `pint.Quantity`, optional
        Additional friction term in [VH96]. Defaluts to 0.

    Returns
    -------
    blh : `pint.Quantity`
        Boundary layer height estimation
    """
    bRi = bulk_richardson_number(
        height,
        smooth(potential_temperature, smoothingspan),
        smooth(u, smoothingspan),
        smooth(v, smoothingspan),
        idxfoot=idxfoot,
        ustar=ustar,
    )

    height = height[~np.isnan(bRi)]
    bRi = bRi[~np.isnan(bRi)]

    if any(bRi > bri_threshold):
        iblh = np.where(bRi > bri_threshold)[0][0]
        blh = height[iblh]
    else:
        blh = np.nan * units.meter

    return blh


def blh_from_parcel(
    height,
    potential_temperature,
    smoothingspan: int = 5,
    theta0=None,
):
    """Calculate atmospheric boundary layer height with the parcel method.

    It is the height where the potential temperature profile reaches its
    foot value.
    See [Sei00, HL06, Col14].

    Parameters
    ----------
    height : `pint.Quantity`
        Altitude (metres above ground) of the points in the profile
    potential_temperature : `pint.Quantity`
        Potential temperature profile
    smoothingspan : int, optional
        The amount of smoothing (number of points in moving average)
    theta0 : `pint.Quantity`, optional
        Value of theta at the foot point (skip unstruted points or add extra term). If not provided, theta[0] is taken.

    Returns
    -------
    blh : `pint.Quantity`
        Boundary layer height estimation
    """
    potential_temperature = smooth(potential_temperature, smoothingspan)

    if theta0 is None:
        theta0 = potential_temperature[0]

    if any(potential_temperature > theta0):
        iblh = np.where(potential_temperature > theta0)[0][0]
        blh = height[iblh]
    else:
        blh = np.nan * units.meter

    return blh


def blh_from_humidity_gradient(
    height,
    humidity,
    smoothingspan: int = 5,
    idxfoot: int = 0,
):
    """Calculate atmospheric boundary layer height from the relative
    humidity gradient

    It is the height where the relative humidity or specific humidity gradient reaches a minimum.
    See [Sei00, HL06, Col14].

    Parameters
    ----------
    height : `pint.Quantity`
        Altitude (metres above ground) of the points in the profile
    humidity : `pint.Quantity`
        Humidity (relative or specific) profile
    smoothingspan : int, optional
        The amount of smoothing (number of points in moving average)
    idxfoot : int, optional
        The index of the foot point (first trusted measure), defaults to 0.

    Returns
    -------
    blh : `pint.Quantity`
        Boundary layer height estimation
    """

    dRHdz = mpcalc.first_derivative(smooth(humidity, smoothingspan), x=height)

    dRHdz = dRHdz[idxfoot:]
    height = height[idxfoot:]

    iblh = np.argmin(dRHdz)

    return height[iblh]


def blh_from_temperature_inversion(
    height,
    temperature,
    smoothingspan: int = 5,
    idxfoot: int = 0,
):
    """Calculate atmospheric boundary layer height from the inversion of
    absolute temperature gradient

    It is the height where the temperature gradient (absolute or potential) changes of sign.
    See [Col14].

    Parameters
    ----------
    height : `pint.Quantity`
        Altitude (metres above ground) of the points in the profile
    humidity : `pint.Quantity`
        Temperature (absolute or potential) profile
    smoothingspan : int, optional
        The amount of smoothing (number of points in moving average)
    idxfoot : int, optional
        The index of the foot point (first trusted measure), defaults to 0.

    Returns
    -------
    blh : `pint.Quantity`
        Boundary layer height estimation
    """

    dTdz = mpcalc.first_derivative(smooth(temperature, smoothingspan), x=height)

    dTdz = dTdz[idxfoot:]
    height = height[idxfoot:]

    if any(dTdz * dTdz[0] < 0):
        iblh = np.where(dTdz * dTdz[0] < 0)[0][0]
        blh = height[iblh]
    else:
        blh = np.nan * units.meter

    return blh
