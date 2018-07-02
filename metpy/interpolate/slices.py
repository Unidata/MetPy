# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for interpolating to a vertical slice/cross section through data."""

import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from ..package_tools import Exporter
from ..xarray import CFConventionHandler

exporter = Exporter(globals())


@exporter.export
def cross_section(data, start, end, steps=100, interp_type='linear'):
    r"""Obtain an interpolated cross-sectional slice through gridded data.

    Utilizing the interpolation functionality in `metpy.interpolate`, this function takes a
    vertical cross-sectional slice along a geodesic through the given data on a regular grid,
    which is given as an `xarray.DataArray` so that we can utilize its coordinate and
    projection metadata.

    Parameters
    ----------
    data: `xarray.DataArray` or `xarray.Dataset`
        Three- (or higher) dimensional field(s) to interpolate (must have attached projection
        information).
    start: (2, ) array_like
        A latitude-longitude pair designating the start point of the cross section (units are
        degrees north and degrees east).
    end: (2, ) array_like
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
    interpolate_points

    """
    if isinstance(data, xr.Dataset):
        # Recursively apply to dataset
        return data.apply(cross_section, True, (start, end), steps=steps,
                          interp_type=interp_type)
    elif data.ndim == 0:
        # This has no dimensions, so it is likely a projection variable. In any case, there
        # are no data here to take the cross section with. Therefore, do nothing.
        return data
    else:
        from pyproj import Geod
        # Apply to this DataArray

        # Get the projection and coordinates
        crs_data = data.metpy.cartopy_crs
        x, y = data.metpy.coordinates('x', 'y')

        # Get the geodesic along which to take the cross section
        # Geod.npts only gives points *in between* the start and end, and we want to include
        # the endpoints.
        g = Geod(crs_data.proj4_init)
        geodesic = np.concatenate([
            np.array(start[::-1])[None],
            np.array(g.npts(start[1], start[0], end[1], end[0], steps - 2)),
            np.array(end[::-1])[None]
        ]).transpose()
        points_cross = crs_data.transform_points(ccrs.Geodetic(), *geodesic)[:, :2]

        # Patch points_cross to match given longitude range, whether [0, 360) or (-180,  180]
        if CFConventionHandler.check_axis(x, 'lon') and (x > 180).any():
            points_cross[points_cross[:, 0] < 0, 0] += 360.

        # Interpolate to geodesic slice, while adding new dimension coordinate 'index'
        data_cross = data.interp({
            x.name: xr.DataArray(points_cross[:, 0], dims='index', attrs=x.attrs),
            y.name: xr.DataArray(points_cross[:, 1], dims='index', attrs=y.attrs)
        }, method=interp_type)
        data_cross.coords['index'] = range(steps)

        return data_cross
