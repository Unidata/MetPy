# Copyright (c) 2015,2016,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Collection of utilities for testing.

This includes:
* unit-aware test functions
* code for testing matplotlib figures
"""
import functools

import numpy as np
import numpy.testing
from pint import DimensionalityError
import pytest
import xarray as xr

import metpy.calc
from metpy.calc import wind_components
from metpy.cbook import get_test_data
from metpy.deprecation import MetpyDeprecationWarning
from .units import units

try:
    import dask.array
except ModuleNotFoundError:
    pass

def needs_cartopy(test_func):
    """Decorate a test function or fixture as requiring CartoPy.

    Will skip the decorated test, or any test using the decorated fixture, if ``cartopy`` is
    unable to be imported.
    """
    @functools.wraps(test_func)
    def wrapped(*args, **kwargs):
        pytest.importorskip('cartopy')
        return test_func(*args, **kwargs)
    return wrapped


def needs_pyproj(test_func):
    """Decorate a test function or fixture as requiring PyProj.

    Will skip the decorated test, or any test using the decorated fixture, if ``pyproj`` is
    unable to be imported.
    """
    @functools.wraps(test_func)
    def wrapped(*args, **kwargs):
        pytest.importorskip('pyproj')
        return test_func(*args, **kwargs)
    return wrapped


def get_upper_air_data(date, station):
    """Get upper air observations from the test data cache.

    Parameters
    ----------
    time : datetime
          The date and time of the desired observation.
    station : str
         The three letter ICAO identifier of the station for which data should be
         downloaded.

    Returns
    -------
        dict : upper air data

    """
    sounding_key = f'{date:%Y-%m-%dT%HZ}_{station}'
    sounding_files = {'2016-05-22T00Z_DDC': 'may22_sounding.txt',
                      '2013-01-20T12Z_OUN': 'jan20_sounding.txt',
                      '1999-05-04T00Z_OUN': 'may4_sounding.txt',
                      '2002-11-11T00Z_BNA': 'nov11_sounding.txt',
                      '2010-12-09T12Z_BOI': 'dec9_sounding.txt'}

    fname = sounding_files[sounding_key]
    fobj = get_test_data(fname)

    def to_float(s):
        # Remove all whitespace and replace empty values with NaN
        if not s.strip():
            s = 'nan'
        return float(s)

    # Skip dashes, column names, units, and more dashes
    for _ in range(4):
        fobj.readline()

    # Initiate lists for variables
    arr_data = []

    # Read all lines of data and append to lists only if there is some data
    for row in fobj:
        level = to_float(row[0:7])
        values = (to_float(row[7:14]), to_float(row[14:21]), to_float(row[21:28]),
                  to_float(row[42:49]), to_float(row[49:56]))

        if any(np.invert(np.isnan(values[1:]))):
            arr_data.append((level,) + values)

    p, z, t, td, direc, spd = np.array(arr_data).T

    p = p * units.hPa
    z = z * units.meters
    t = t * units.degC
    td = td * units.degC
    direc = direc * units.degrees
    spd = spd * units.knots

    u, v = wind_components(spd, direc)

    return {'pressure': p, 'height': z, 'temperature': t,
            'dewpoint': td, 'direction': direc, 'speed': spd, 'u_wind': u, 'v_wind': v}


def check_and_drop_units(actual, desired):
    r"""Check that the units on the passed in arrays are compatible; return the magnitudes.

    Parameters
    ----------
    actual : `pint.Quantity` or array-like

    desired : `pint.Quantity` or array-like

    Returns
    -------
    actual, desired
        array-like versions of `actual` and `desired` once they have been
        coerced to compatible units.

    Raises
    ------
    AssertionError
        If the units on the passed in objects are not compatible.

    """
    try:
        # Convert DataArrays to Quantities
        if isinstance(desired, xr.DataArray):
            desired = desired.metpy.unit_array
        if isinstance(actual, xr.DataArray):
            actual = actual.metpy.unit_array
        # If the desired result has units, add dimensionless units if necessary, then
        # ensure that this is compatible to the desired result.
        if hasattr(desired, 'units'):
            if not hasattr(actual, 'units'):
                actual = units.Quantity(actual, 'dimensionless')
            actual = actual.to(desired.units)
        # Otherwise, the desired result has no units. Convert the actual result to
        # dimensionless units if it is a united quantity.
        else:
            if hasattr(actual, 'units'):
                actual = actual.to('dimensionless')
    except DimensionalityError:
        raise AssertionError('Units are not compatible: {} should be {}'.format(
            actual.units, getattr(desired, 'units', 'dimensionless')))

    if hasattr(actual, 'magnitude'):
        actual = actual.magnitude
    if hasattr(desired, 'magnitude'):
        desired = desired.magnitude

    return actual, desired


def check_mask(actual, desired):
    """Check that two arrays have the same mask.

    This handles the fact that `~numpy.testing.assert_array_equal` ignores masked values
    in either of the arrays. This ensures that the masks are identical.
    """
    actual_mask = getattr(actual, 'mask', np.full(np.asarray(actual).shape, False))
    desired_mask = getattr(desired, 'mask', np.full(np.asarray(desired).shape, False))
    np.testing.assert_array_equal(actual_mask, desired_mask)


def assert_nan(value, units):
    """Check for nan with proper units."""
    value, _ = check_and_drop_units(value, np.nan * units)
    assert np.isnan(value)


def assert_almost_equal(actual, desired, decimal=7):
    """Check that values are almost equal, including units.

    Wrapper around :func:`numpy.testing.assert_almost_equal`
    """
    actual, desired = check_and_drop_units(actual, desired)
    numpy.testing.assert_almost_equal(actual, desired, decimal)


def assert_array_almost_equal(actual, desired, decimal=7):
    """Check that arrays are almost equal, including units.

    Wrapper around :func:`numpy.testing.assert_array_almost_equal`
    """
    actual, desired = check_and_drop_units(actual, desired)
    check_mask(actual, desired)
    numpy.testing.assert_array_almost_equal(actual, desired, decimal)


def assert_array_equal(actual, desired):
    """Check that arrays are equal, including units.

    Wrapper around :func:`numpy.testing.assert_array_equal`
    """
    actual, desired = check_and_drop_units(actual, desired)
    check_mask(actual, desired)
    numpy.testing.assert_array_equal(actual, desired)


def assert_xarray_allclose(actual, desired):
    """Check that the xarrays are almost equal, including coordinates and attributes."""
    xr.testing.assert_allclose(actual, desired)
    assert desired.metpy.coordinates_identical(actual)
    assert desired.attrs == actual.attrs


@pytest.fixture(scope='module', autouse=True)
def set_agg_backend():
    """Fixture to ensure the Agg backend is active."""
    import matplotlib.pyplot as plt
    prev_backend = plt.get_backend()
    try:
        plt.switch_backend('agg')
        yield
    finally:
        plt.switch_backend(prev_backend)


def check_and_silence_warning(warn_type):
    """Decorate a function to swallow some warning type, making sure they are present.

    This should be used on function tests to make sure the warnings are not printing in the
    tests, but checks that the warning is present and makes sure the function still works as
    intended.
    """
    def dec(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with pytest.warns(warn_type):
                return func(*args, **kwargs)
        return wrapper
    return dec


check_and_silence_deprecation = check_and_silence_warning(MetpyDeprecationWarning)


def scalar(name, values, truth, decimal):
    """Test a function using scalars."""
    func = eval(name)
    # Index by a tuple to select a single value from a (potential) ndarray
    args = [v[tuple(np.zeros_like(v.shape))] for v in values]
    truths = [t[tuple(np.zeros_like(t.shape))] for t in truth]

    results = func(*args)

    if isinstance(results, tuple):
        for t, r in zip(truths, results):
            assert_almost_equal(t, r, decimal)
    else:
        assert_almost_equal(*truths, results, decimal)


def array(name, values, truth, decimal):
    """Test a function using arrays."""
    func = eval(name)
    results = func(*values)

    _assert_array_results(truth, results, decimal)


def masked(name, values, truth, decimal):
    """Test a function using masked arrays."""
    func = eval(name)

    # Create a mask that is True/False for every other value
    mask = np.ones_like(values[0].m, dtype=bool)
    mask[::2] = False

    values_masked = [units.Quantity(np.ma.array(v.m, mask=mask), v.units) for v in values]
    truth_masked = [units.Quantity(np.ma.array(t.m, mask=mask), t.units) for t in truth]

    results = func(*values_masked)

    _assert_array_results(truth_masked, results, decimal)


def nans(name, values, truth, decimal):
    """Test a function using nans."""
    func = eval(name)

    # Create copies of values and truth, then add nans at every other index
    def assign_nans(arr):
        arr_nans = arr.copy()
        for i, a in enumerate(arr_nans):
            # Force the copied array to float64 so that it plays nicely with nans
            anans = a.copy().astype('float64')
            anans[::2] = np.nan
            arr_nans[i] = anans
        return arr_nans

    values_nans = assign_nans(values)
    truth_nans = assign_nans(truth)

    results = func(*values_nans)

    _assert_array_results(truth_nans, results, decimal)


def data_array(name, values, truth, decimal):
    """Test a function using data arrays."""
    func = eval(name)

    values_data_array = [xr.DataArray(v) for v in values]
    truth_data_array = [xr.DataArray(t) for t in truth]

    results = func(*values_data_array)

    _assert_array_results(truth_data_array, results, decimal)


def dask_array(name, values, truth, decimal):
    """Test a function using dask arrays."""
    func = eval(name)

    values_dask = [units.Quantity(dask.array.from_array(v.m), v.units) for v in values]
    truth_dask = [units.Quantity(dask.array.from_array(t.m), t.units) for t in truth]

    results = func(*values_dask)

    # Assert that the dask array objects are equal
    _assert_array_results(truth_dask, results, decimal)

    computed = results.compute()
    truth_unitless = [t.m for t in truth_dask]

    # Assert that the *computed* numpy arrays are equal
    _assert_array_results(truth_unitless, computed, decimal)


def _assert_array_results(truth, results, decimal):
    """Assert equality of an array-like data type."""
    if isinstance(results, tuple):
        for t, r in zip(truth, results):
            assert_array_almost_equal(t, r, decimal)
    else:
        assert_array_almost_equal(*truth, results, decimal)
