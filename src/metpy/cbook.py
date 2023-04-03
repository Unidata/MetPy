# Copyright (c) 2008,2015,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Collection of generally useful utility code from the cookbook."""

import os
from pathlib import Path

import numpy as np
import pooch

from . import __version__

pooch_kwargs = {'path': pooch.os_cache('metpy'), 'version': 'v' + __version__,
                'base_url': 'https://github.com/Unidata/MetPy/raw/{version}/staticdata/'}

# Check if we have the data available directly from a git checkout, either from the
# TEST_DATA_DIR variable, or looking relative to the path of this module's file. Use this
# to override Pooch's path and disable downloading from GitHub.
dev_data_path = os.environ.get('TEST_DATA_DIR', Path(__file__).parents[2] / 'staticdata')
if Path(dev_data_path).exists():
    pooch_kwargs['path'] = dev_data_path
    pooch_kwargs['version'] = None
    pooch_kwargs['base_url'] = pooch_kwargs['base_url'].format(version='main')
    if pooch.__version__ >= 'v1.6.0':
        pooch_kwargs['allow_updates'] = False

POOCH = pooch.create(version_dev='main', **pooch_kwargs)
POOCH.load_registry(Path(__file__).parent / 'static-data-manifest.txt')


def get_test_data(fname, as_file_obj=True, mode='rb'):
    """Access a file from MetPy's collection of test data."""
    path = POOCH.fetch(fname)
    # If we want a file object, open it, trying to guess whether this should be binary mode
    # or not
    if as_file_obj:
        return open(path, mode)  # noqa: SIM115

    return path


def example_data():
    """Create a sample xarray Dataset with 2D variables."""
    import xarray as xr

    # make data based on Matplotlib example data for wind barbs
    x, y = np.meshgrid(np.linspace(-3, 3, 25), np.linspace(-3, 3, 25))
    z = (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)

    # make u and v out of the z equation
    u = -np.diff(z[:, 1:], axis=0) * 100 + 10
    v = np.diff(z[1:, :], axis=1) * 100 + 10

    # make t as colder air to the north
    t = (np.linspace(15, 5, 24) * np.ones((24, 24))).T

    # Make lat/lon data over the mid-latitudes
    lats = np.linspace(30, 40, 24)
    lons = np.linspace(360 - 100, 360 - 90, 24)

    # place data into an xarray dataset object
    lat = xr.DataArray(lats, attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    lon = xr.DataArray(lons, attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    uwind = xr.DataArray(u, coords=(lat, lon), dims=['lat', 'lon'],
                         attrs={'standard_name': 'u-component_of_wind', 'units': 'm s-1'})
    vwind = xr.DataArray(v, coords=(lat, lon), dims=['lat', 'lon'],
                         attrs={'standard_name': 'u-component_of_wind', 'units': 'm s-1'})
    temperature = xr.DataArray(t, coords=(lat, lon), dims=['lat', 'lon'],
                               attrs={'standard_name': 'temperature', 'units': 'degC'})
    return xr.Dataset({'uwind': uwind,
                       'vwind': vwind,
                       'temperature': temperature})


class Registry:
    """Provide a generic function registry.

    This provides a class to instantiate, which then has a `register` method that can
    be used as a decorator on functions to register them under a particular name.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._registry = {}

    def register(self, name):
        """Register a callable with the registry under a particular name.

        Parameters
        ----------
        name : str
            The name under which to register a function

        Returns
        -------
        dec : Callable
            A decorator that takes a function and will register it under the name.

        """
        def dec(func):
            self._registry[name] = func
            return func
        return dec

    def __getitem__(self, name):
        """Return any callable registered under name."""
        return self._registry[name]


def broadcast_indices(indices, shape, axis):
    """Calculate index values to properly broadcast index array within data array.

    The purpose of this function is work around the challenges trying to work with arrays of
    indices that need to be "broadcast" against full slices for other dimensions.

    See usage in `interpolate_1d` or `isentropic_interpolation`.
    """
    ret = []
    ndim = len(shape)
    for dim in range(ndim):
        if dim == axis:
            ret.append(indices)
        else:
            broadcast_slice = [np.newaxis] * ndim
            broadcast_slice[dim] = slice(None)
            dim_inds = np.arange(shape[dim])
            ret.append(dim_inds[tuple(broadcast_slice)])
    return tuple(ret)


__all__ = ('Registry', 'broadcast_indices', 'get_test_data')
