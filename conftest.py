# Copyright (c) 2016,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Configure pytest for metpy."""

import os

import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import pooch
import pyproj
import pytest
import scipy
import traitlets
import xarray

import metpy.calc
import metpy.units

# Need to disable fallback before importing pint
os.environ['PINT_ARRAY_PROTOCOL_FALLBACK'] = '0'
import pint  # noqa: I100, E402

try:
    pooch_version = pooch.__version__
except AttributeError:
    pooch_version = pooch.version.full_version


def pytest_report_header(config, startdir):
    """Add dependency information to pytest output."""
    return (f'Dep Versions: Matplotlib {matplotlib.__version__}, '
            f'NumPy {numpy.__version__}, Pandas {pandas.__version__}, '
            f'Pint {pint.__version__}, Pooch {pooch_version}\n'
            f'\tPyProj {pyproj.__version__}, SciPy {scipy.__version__}, '
            f'Traitlets {traitlets.__version__}, Xarray {xarray.__version__}')


@pytest.fixture(autouse=True)
def doctest_available_modules(doctest_namespace):
    """Make modules available automatically to doctests."""
    doctest_namespace['metpy'] = metpy
    doctest_namespace['metpy.calc'] = metpy.calc
    doctest_namespace['np'] = numpy
    doctest_namespace['plt'] = matplotlib.pyplot
    doctest_namespace['units'] = metpy.units.units


@pytest.fixture()
def ccrs():
    """Provide access to the ``cartopy.crs`` module through a global fixture.

    Any testing function/fixture that needs access to ``cartopy.crs`` can simply add this to
    their parameter list.

    """
    return pytest.importorskip('cartopy.crs')


@pytest.fixture
def cfeature():
    """Provide access to the ``cartopy.feature`` module through a global fixture.

    Any testing function/fixture that needs access to ``cartopy.feature`` can simply add this
    to their parameter list.

    """
    return pytest.importorskip('cartopy.feature')


@pytest.fixture()
def test_da_lonlat():
    """Return a DataArray with a lon/lat grid and no time coordinate for use in tests."""
    data = numpy.linspace(300, 250, 3 * 4 * 4).reshape((3, 4, 4))
    ds = xarray.Dataset(
        {'temperature': (['isobaric', 'lat', 'lon'], data)},
        coords={
            'isobaric': xarray.DataArray(
                numpy.array([850., 700., 500.]),
                name='isobaric',
                dims=['isobaric'],
                attrs={'units': 'hPa'}
            ),
            'lat': xarray.DataArray(
                numpy.linspace(30, 40, 4),
                name='lat',
                dims=['lat'],
                attrs={'units': 'degrees_north'}
            ),
            'lon': xarray.DataArray(
                numpy.linspace(260, 270, 4),
                name='lon',
                dims=['lon'],
                attrs={'units': 'degrees_east'}
            )
        }
    )
    ds['temperature'].attrs['units'] = 'kelvin'

    return ds.metpy.parse_cf('temperature')


@pytest.fixture()
def test_da_xy():
    """Return a DataArray with a x/y grid and a time coordinate for use in tests."""
    data = numpy.linspace(300, 250, 3 * 3 * 4 * 4).reshape((3, 3, 4, 4))
    ds = xarray.Dataset(
        {'temperature': (['time', 'isobaric', 'y', 'x'], data),
         'lambert_conformal': ([], '')},
        coords={
            'time': xarray.DataArray(
                numpy.array([numpy.datetime64('2018-07-01T00:00'),
                             numpy.datetime64('2018-07-01T06:00'),
                             numpy.datetime64('2018-07-01T12:00')]),
                name='time',
                dims=['time']
            ),
            'isobaric': xarray.DataArray(
                numpy.array([850., 700., 500.]),
                name='isobaric',
                dims=['isobaric'],
                attrs={'units': 'hPa'}
            ),
            'y': xarray.DataArray(
                numpy.linspace(-1000, 500, 4),
                name='y',
                dims=['y'],
                attrs={'units': 'km'}
            ),
            'x': xarray.DataArray(
                numpy.linspace(0, 1500, 4),
                name='x',
                dims=['x'],
                attrs={'units': 'km'}
            )
        }
    )
    ds['temperature'].attrs = {
        'units': 'kelvin',
        'grid_mapping': 'lambert_conformal'
    }
    ds['lambert_conformal'].attrs = {
        'grid_mapping_name': 'lambert_conformal_conic',
        'standard_parallel': 50.0,
        'longitude_of_central_meridian': -107.0,
        'latitude_of_projection_origin': 50.0,
        'earth_shape': 'spherical',
        'earth_radius': 6367470.21484375
    }

    return ds.metpy.parse_cf('temperature')


@pytest.fixture()
def set_agg_backend():
    """Fixture to ensure the Agg backend is active."""
    prev_backend = matplotlib.pyplot.get_backend()
    try:
        matplotlib.pyplot.switch_backend('agg')
        yield
    finally:
        matplotlib.pyplot.switch_backend(prev_backend)


@pytest.fixture(params=['dask', 'xarray', 'masked', 'numpy'])
def array_type(request):
    """Return an array type for testing calc functions."""
    quantity = metpy.units.units.Quantity
    if request.param == 'dask':
        dask_array = pytest.importorskip('dask.array', reason='dask.array is not available')
        marker = request.node.get_closest_marker('xfail_dask')
        if marker is not None:
            request.applymarker(pytest.mark.xfail(reason=marker.args[0]))
        return lambda d, u, *, mask=None: quantity(dask_array.array(d), u)
    elif request.param == 'xarray':
        return lambda d, u, *, mask=None: xarray.DataArray(d, attrs={'units': u})
    elif request.param == 'masked':
        return lambda d, u, *, mask=None: quantity(numpy.ma.array(d, mask=mask), u)
    elif request.param == 'numpy':
        return lambda d, u, *, mask=None: quantity(numpy.array(d), u)
    else:
        raise ValueError(f'Unsupported array_type option {request.param}')


@pytest.fixture
def geog_data(request):
    """Create data to use for testing calculations on geographic coordinates."""
    # Generate a field of u and v on a lat/lon grid
    crs = pyproj.CRS(request.param)
    proj = pyproj.Proj(crs)
    a = numpy.arange(4)[None, :]
    arr = numpy.r_[a, a, a] * metpy.units.units('m/s')
    lons = numpy.array([-100, -90, -80, -70]) * metpy.units.units.degree
    lats = numpy.array([45, 55, 65]) * metpy.units.units.degree
    lon_arr, lat_arr = numpy.meshgrid(lons.m_as('degree'), lats.m_as('degree'))
    factors = proj.get_factors(lon_arr, lat_arr)

    return (crs, lons, lats, arr, arr, factors.parallel_scale, factors.meridional_scale,
            metpy.calc.lat_lon_grid_deltas(lons.m, numpy.zeros_like(lons.m),
                                           geod=crs.get_geod())[0][0],
            metpy.calc.lat_lon_grid_deltas(numpy.zeros_like(lats.m), lats.m,
                                           geod=crs.get_geod())[1][:, 0])
