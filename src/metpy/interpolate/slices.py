# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for interpolating to a vertical slice/cross section through data."""

import numpy as np
import xarray as xr

from ..package_tools import Exporter
from ..units import is_quantity, units
from ..xarray import check_axis

exporter = Exporter(globals())


@exporter.export
def interpolate_to_slice(data, points, interp_type='linear'):
    r"""Obtain an interpolated slice through data using xarray.

    Utilizing the interpolation functionality in xarray, this function
    takes a slice of the given data (currently only regular grids are supported), which is
    given as an `xarray.DataArray` so that we can utilize its coordinate metadata.

    Parameters
    ----------
    data: `xarray.DataArray` or `xarray.Dataset`
        Three- (or higher) dimensional field(s) to interpolate. The DataArray (or each
        DataArray in the Dataset) must have been parsed by MetPy and include both an x and
        y coordinate dimension.
    points: (N, 2) array-like
        A list of x, y points in the data projection at which to interpolate the data
    interp_type: str, optional
        The interpolation method, either 'linear' or 'nearest' (see
        `xarray.DataArray.interp()` for details). Defaults to 'linear'.

    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        The interpolated slice of data, with new index dimension of size N.

    See Also
    --------
    cross_section

    """
    try:
        x, y = data.metpy.coordinates('x', 'y')
    except AttributeError:
        raise ValueError('Required coordinate information not available. Verify that '
                         'your data has been parsed by MetPy with proper x and y '
                         'dimension coordinates.')

    data_sliced = data.interp({
        x.name: xr.DataArray(points[:, 0], dims='index', attrs=x.attrs),
        y.name: xr.DataArray(points[:, 1], dims='index', attrs=y.attrs)
    }, method=interp_type)
    data_sliced.coords['index'] = range(len(points))

    # Bug in xarray: interp strips units
    if is_quantity(data.data) and not is_quantity(data_sliced.data):
        data_sliced.data = units.Quantity(data_sliced.data, data.data.units)

    return data_sliced


@exporter.export
def geodesic(crs, start, end, steps):
    r"""Construct a geodesic path between two points.

    This function acts as a wrapper for the geodesic construction available in ``pyproj``.

    Parameters
    ----------
    crs: `pyproj.crs.CRS`
        PyProj Coordinate Reference System to use for the output
    start: (2, ) array-like
        A latitude-longitude pair designating the start point of the geodesic (units are
        degrees north and degrees east).
    end: (2, ) array-like
        A latitude-longitude pair designating the end point of the geodesic (units are degrees
        north and degrees east).
    steps: int, optional
        The number of points along the geodesic between the start and the end point
        (including the end points).

    Returns
    -------
    `numpy.ndarray`
        The list of x, y points in the given CRS of length `steps` along the geodesic.

    See Also
    --------
    cross_section

    """
    from pyproj import Proj

    g = crs.get_geod()
    p = Proj(crs)

    # Geod.npts only gives points *in between* the start and end, and we want to include
    # the endpoints.
    geodesic = np.concatenate([
        np.array(start[::-1])[None],
        np.array(g.npts(start[1], start[0], end[1], end[0], steps - 2)),
        np.array(end[::-1])[None]
    ]).transpose()
    return np.stack(p(geodesic[0], geodesic[1], inverse=False, radians=False), axis=-1)


@exporter.export
def cross_section(data, start, end, steps=100, interp_type='linear'):
    r"""Obtain an interpolated cross-sectional slice through gridded data.

    Utilizing the interpolation functionality in xarray, this function takes a vertical
    cross-sectional slice along a geodesic through the given data on a regular grid, which is
    given as an `xarray.DataArray` so that we can utilize its coordinate and projection
    metadata.

    Parameters
    ----------
    data: `xarray.DataArray` or `xarray.Dataset`
        Three- (or higher) dimensional field(s) to interpolate. The DataArray (or each
        DataArray in the Dataset) must have been parsed by MetPy and include both an x and
        y coordinate dimension and the added `crs` coordinate.
    start: (2, ) array-like
        A latitude-longitude pair designating the start point of the cross section (units are
        degrees north and degrees east).
    end: (2, ) array-like
        A latitude-longitude pair designating the end point of the cross section (units are
        degrees north and degrees east).
    steps: int, optional
        The number of points along the geodesic between the start and the end point
        (including the end points) to use in the cross section. Defaults to 100.
    interp_type: str, optional
        The interpolation method, either 'linear' or 'nearest' (see
        `xarray.DataArray.interp()` for details). Defaults to 'linear'.

    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        The interpolated cross section, with new index dimension along the cross-section.

    See Also
    --------
    interpolate_to_slice, geodesic

    """
    if isinstance(data, xr.Dataset):
        # Recursively apply to dataset
        return data.map(cross_section, True, (start, end), steps=steps,
                        interp_type=interp_type)
    elif data.ndim == 0:
        # This has no dimensions, so it is likely a projection variable. In any case, there
        # are no data here to take the cross section with. Therefore, do nothing.
        return data
    else:

        # Get the projection and coordinates
        try:
            crs_data = data.metpy.pyproj_crs
            x = data.metpy.x
        except AttributeError:
            raise ValueError('Data missing required coordinate information. Verify that '
                             'your data have been parsed by MetPy with proper x and y '
                             'dimension coordinates and added crs coordinate of the '
                             'correct projection for each variable.')

        # Get the geodesic
        points_cross = geodesic(crs_data, start, end, steps)

        # Patch points_cross to match given longitude range, whether [0, 360) or (-180,  180]
        if check_axis(x, 'longitude') and (x > 180).any():
            points_cross[points_cross[:, 0] < 0, 0] += 360.

        # Return the interpolated data
        return interpolate_to_slice(data, points_cross, interp_type=interp_type)
