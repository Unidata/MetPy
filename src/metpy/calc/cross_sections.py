# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculations related to cross-sections and respective vector components.

Compared to the rest of the calculations which are based around pint quantities, this module
is based around xarray DataArrays.
"""

import numpy as np
import xarray as xr

from .basic import coriolis_parameter
from .tools import first_derivative
from ..package_tools import Exporter
from ..units import units
from ..xarray import check_axis, check_matching_coordinates

exporter = Exporter(globals())


def distances_from_cross_section(cross):
    """Calculate the distances in the x and y directions along a cross-section.

    Parameters
    ----------
    cross : `xarray.DataArray`
        The input DataArray of a cross-section from which to obtain geometric distances in
        the x and y directions.

    Returns
    -------
    x, y : tuple of `xarray.DataArray`
        A tuple of the x and y distances as DataArrays

    """
    if check_axis(cross.metpy.x, 'longitude') and check_axis(cross.metpy.y, 'latitude'):
        # Use pyproj to obtain x and y distances
        g = cross.metpy.pyproj_crs.get_geod()
        lon = cross.metpy.x
        lat = cross.metpy.y

        forward_az, _, distance = g.inv(lon[0].values * np.ones_like(lon),
                                        lat[0].values * np.ones_like(lat),
                                        lon.values,
                                        lat.values)
        x = distance * np.sin(np.deg2rad(forward_az))
        y = distance * np.cos(np.deg2rad(forward_az))

        # Build into DataArrays
        x = xr.DataArray(units.Quantity(x, 'meter'), coords=lon.coords, dims=lon.dims)
        y = xr.DataArray(units.Quantity(y, 'meter'), coords=lat.coords, dims=lat.dims)

    elif check_axis(cross.metpy.x, 'x') and check_axis(cross.metpy.y, 'y'):

        # Simply return what we have
        x = cross.metpy.x
        y = cross.metpy.y

    else:
        raise AttributeError('Sufficient horizontal coordinates not defined.')

    return x, y


def latitude_from_cross_section(cross):
    """Calculate the latitude of points in a cross-section.

    Parameters
    ----------
    cross : `xarray.DataArray`
        The input DataArray of a cross-section from which to obtain latitudes

    Returns
    -------
    latitude : `xarray.DataArray`
        Latitude of points

    """
    y = cross.metpy.y
    if check_axis(y, 'latitude'):
        return y
    else:
        from pyproj import Proj
        latitude = Proj(cross.metpy.pyproj_crs)(
            cross.metpy.x.values,
            y.values,
            inverse=True,
            radians=False
        )[1]
        return xr.DataArray(units.Quantity(latitude, 'degrees_north'), coords=y.coords,
                            dims=y.dims)


@exporter.export
def unit_vectors_from_cross_section(cross, index='index'):
    r"""Calculate the unit tangent and unit normal vectors from a cross-section.

    Given a path described parametrically by :math:`\vec{l}(i) = (x(i), y(i))`, we can find
    the unit tangent vector by the formula:

    .. math:: \vec{T}(i) =
        \frac{1}{\sqrt{\left( \frac{dx}{di} \right)^2 + \left( \frac{dy}{di} \right)^2}}
        \left( \frac{dx}{di}, \frac{dy}{di} \right)

    From this, because this is a two-dimensional path, the normal vector can be obtained by a
    simple :math:`\frac{\pi}{2}` rotation.

    Parameters
    ----------
    cross : `xarray.DataArray`
        The input DataArray of a cross-section from which to obtain latitudes

    index : str or int, optional
        Denotes the index coordinate of the cross-section, defaults to 'index' as
        set by `metpy.interpolate.cross_section`

    Returns
    -------
    unit_tangent_vector, unit_normal_vector : tuple of `numpy.ndarray`
        Arrays describing the unit tangent and unit normal vectors (in x,y) for all points
        along the cross-section

    """
    x, y = distances_from_cross_section(cross)
    dx_di = first_derivative(x, axis=index).values
    dy_di = first_derivative(y, axis=index).values
    tangent_vector_mag = np.hypot(dx_di, dy_di)
    unit_tangent_vector = np.vstack([dx_di / tangent_vector_mag, dy_di / tangent_vector_mag])
    unit_normal_vector = np.vstack([-dy_di / tangent_vector_mag, dx_di / tangent_vector_mag])
    return unit_tangent_vector, unit_normal_vector


@exporter.export
@check_matching_coordinates
def cross_section_components(data_x, data_y, index='index'):
    r"""Obtain the tangential and normal components of a cross-section of a vector field.

    Parameters
    ----------
    data_x : `xarray.DataArray`
        The input DataArray of the x-component (in terms of data projection) of the vector
        field.

    data_y : `xarray.DataArray`
        The input DataArray of the y-component (in terms of data projection) of the vector
        field.

    index : str or int, optional
        Denotes the index coordinate of the cross-section, defaults to 'index' as
        set by `metpy.interpolate.cross_section`

    Returns
    -------
    component_tangential, component_normal: tuple of `xarray.DataArray`
        Components of the vector field in the tangential and normal directions, respectively

    See Also
    --------
    tangential_component, normal_component

    Notes
    -----
    The coordinates of `data_x` and `data_y` must match.

    """
    # Get the unit vectors
    unit_tang, unit_norm = unit_vectors_from_cross_section(data_x, index=index)

    # Take the dot products
    component_tang = data_x * unit_tang[0] + data_y * unit_tang[1]
    component_norm = data_x * unit_norm[0] + data_y * unit_norm[1]

    return component_tang, component_norm


@exporter.export
@check_matching_coordinates
def normal_component(data_x, data_y, index='index'):
    r"""Obtain the normal component of a cross-section of a vector field.

    Parameters
    ----------
    data_x : `xarray.DataArray`
        The input DataArray of the x-component (in terms of data projection) of the vector
        field.

    data_y : `xarray.DataArray`
        The input DataArray of the y-component (in terms of data projection) of the vector
        field.

    index : str or int, optional
        Denotes the index coordinate of the cross-section, defaults to 'index' as
        set by `metpy.interpolate.cross_section`

    Returns
    -------
    component_normal: `xarray.DataArray`
        Component of the vector field in the normal directions

    See Also
    --------
    cross_section_components, tangential_component

    Notes
    -----
    The coordinates of `data_x` and `data_y` must match.

    """
    # Get the unit vectors
    _, unit_norm = unit_vectors_from_cross_section(data_x, index=index)

    # Take the dot products
    component_norm = data_x * unit_norm[0] + data_y * unit_norm[1]

    # Reattach only reliable attributes after operation
    if 'grid_mapping' in data_x.attrs:
        component_norm.attrs['grid_mapping'] = data_x.attrs['grid_mapping']

    return component_norm


@exporter.export
@check_matching_coordinates
def tangential_component(data_x, data_y, index='index'):
    r"""Obtain the tangential component of a cross-section of a vector field.

    Parameters
    ----------
    data_x : `xarray.DataArray`
        The input DataArray of the x-component (in terms of data projection) of the vector
        field

    data_y : `xarray.DataArray`
        The input DataArray of the y-component (in terms of data projection) of the vector
        field

    index : str or int, optional
        Denotes the index coordinate of the cross-section, defaults to 'index' as
        set by `metpy.interpolate.cross_section`

    Returns
    -------
    component_tangential: `xarray.DataArray`
        Component of the vector field in the tangential directions

    See Also
    --------
    cross_section_components, normal_component

    Notes
    -----
    The coordinates of `data_x` and `data_y` must match.

    """
    # Get the unit vectors
    unit_tang, _ = unit_vectors_from_cross_section(data_x, index=index)

    # Take the dot products
    component_tang = data_x * unit_tang[0] + data_y * unit_tang[1]

    # Reattach only reliable attributes after operation
    if 'grid_mapping' in data_x.attrs:
        component_tang.attrs['grid_mapping'] = data_x.attrs['grid_mapping']

    return component_tang


@exporter.export
@check_matching_coordinates
def absolute_momentum(u, v, index='index'):
    r"""Calculate cross-sectional absolute momentum (also called pseudoangular momentum).

    The cross-sectional absolute momentum is calculated given u- and v-components of the wind
    along a 2 dimensional vertical cross-section. The coordinates of `u` and `v` must match.

    Parameters
    ----------
    u : `xarray.DataArray`
        The input DataArray of the x-component (in terms of data projection) of the wind.

    v : `xarray.DataArray`
        The input DataArray of the y-component (in terms of data projection) of the wind.

    index : str or int, optional
        Denotes the index coordinate of the cross-section, defaults to 'index' as
        set by `metpy.interpolate.cross_section`

    Returns
    -------
    absolute_momentum: `xarray.DataArray`
        Absolute momentum

    See Also
    --------
    metpy.interpolate.cross_section, cross_section_components

    Notes
    -----
    As given in [Schultz1999]_, absolute momentum (also called pseudoangular momentum) is
    given by:

    .. math:: M = v + fx

    where :math:`v` is the along-front component of the wind and :math:`x` is the cross-front
    distance. Applied to a cross-section taken perpendicular to the front, :math:`v` becomes
    the normal component of the wind and :math:`x` the tangential distance.

    If using this calculation in assessing symmetric instability, geostrophic wind should be
    used so that geostrophic absolute momentum :math:`\left(M_g\right)` is obtained, as
    described in [Schultz1999]_.

    .. versionchanged:: 1.0
       Renamed ``u_wind``, ``v_wind`` parameters to ``u``, ``v``

    """
    # Get the normal component of the wind
    norm_wind = normal_component(u.metpy.quantify(), v.metpy.quantify(), index=index)

    # Get other pieces of calculation (all as ndarrays matching shape of norm_wind)
    latitude = latitude_from_cross_section(norm_wind)
    _, latitude = xr.broadcast(norm_wind, latitude)
    f = coriolis_parameter(latitude)
    x, y = distances_from_cross_section(norm_wind)
    _, x, y = xr.broadcast(norm_wind, x, y)
    distance = np.hypot(x.metpy.quantify(), y.metpy.quantify())

    return (norm_wind + f * distance).metpy.convert_units('m/s')
