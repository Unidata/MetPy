# Copyright (c) 2015,2016,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Collection of utilities for testing.

This includes:
* unit-aware test functions
* code for testing matplotlib figures
"""
import contextlib
import functools
from importlib.metadata import PackageNotFoundError, requires, version
import operator as op
import re

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing
from packaging.version import Version
from pint import DimensionalityError
import pytest
import xarray as xr

from .calc import wind_components
from .cbook import get_test_data
from .deprecation import MetpyDeprecationWarning
from .units import units


def version_check(version_spec):
    """Return comparison between the active module and a requested version number.

    Will also validate specification against package metadata to alert if spec is irrelevant.

    Parameters
    ----------
    version_spec : str
        Module version specification to validate against installed package. Must take the form
        of `f'{module_name}{comparison_operator}{version_number}'` where `comparison_operator`
        must be one of `['==', '=', '!=', '<', '<=', '>', '>=']`, eg `'metpy>1.0'`.

    Returns
    -------
        bool : Whether the installed package validates against the provided specification
    """
    comparison_operators = {
        '==': op.eq, '=': op.eq, '!=': op.ne, '<': op.lt, '<=': op.le, '>': op.gt, '>=': op.ge,
    }

    # Match version_spec for groups of module name,
    # comparison operator, and requested module version
    module_name, comparison, version_number = _parse_version_spec(version_spec)

    # Check MetPy metadata for minimum required version of same package
    metadata_spec = _get_metadata_spec(module_name)
    _, _, minimum_version_number = _parse_version_spec(metadata_spec)

    try:
        installed_version = Version(version(module_name))
    except PackageNotFoundError:
        # Package not installed considered false condition for spec
        return False

    specified_version = Version(version_number)
    minimum_version = Version(minimum_version_number)

    if specified_version < minimum_version:
        raise ValueError(
            f'Specified {version_spec} outdated according to MetPy minimum {metadata_spec}.')

    try:
        return comparison_operators[comparison](installed_version, specified_version)
    except KeyError:
        raise ValueError(
            f'Comparison operator {comparison} not one of {list(comparison_operators)}.'
        ) from None


def _parse_version_spec(version_spec):
    """Parse module name, comparison, and version from pip-style package spec string.

    Parameters
    ----------
    version_spec : str
        Package spec to parse

    Returns
    -------
        tuple of str : Parsed specification groups of package name, comparison, and version

    """
    pattern = re.compile(r'(\w+)\s*([<>!=]+)\s*([\d\w.]+)')
    match = pattern.match(version_spec)

    if not match:
        raise ValueError(f'Invalid version specification {version_spec}.'
                         f'See version_check documentation for more information.')
    else:
        return match.groups()


def _get_metadata_spec(module_name):
    """Get package spec string for requested module from package metadata.

    Parameters
    ----------
    module_name : str
        Name of MetPy required package to look up

    Returns
    -------
        str : Package spec string for request module
    """
    return [entry for entry in requires('metpy') if module_name.lower() in entry.lower()][0]


def needs_module(module):
    """Decorate a test function or fixture as requiring a module.

    Will skip the decorated test, or any test using the decorated fixture, if ``module``
    is unable to be imported.
    """
    def dec(test_func):
        @functools.wraps(test_func)
        def wrapped(*args, **kwargs):
            pytest.importorskip(module)
            return test_func(*args, **kwargs)
        return wrapped
    return dec


needs_cartopy = needs_module('cartopy')


@contextlib.contextmanager
def autoclose_figure(*args, **kwargs):
    """Create a figure that is automatically closed when exiting a block.

    ``*args`` and ``**kwargs`` are forwarded onto the call to `plt.figure()`.
    """
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)


def get_upper_air_data(date, station):
    """Get upper air observations from the test data cache.

    Parameters
    ----------
    time : `~datetime.datetime`
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

    # Initiate lists for variables
    arr_data = []

    def to_float(s):
        # Remove all whitespace and replace empty values with NaN
        if not s.strip():
            s = 'nan'
        return float(s)

    with contextlib.closing(get_test_data(sounding_files[sounding_key])) as fobj:
        # Skip dashes, column names, units, and more dashes
        for _ in range(4):
            fobj.readline()

        # Read all lines of data and append to lists only if there is some data
        for row in fobj:
            level = to_float(row[0:7])
            values = (to_float(row[7:14]), to_float(row[14:21]), to_float(row[21:28]),
                      to_float(row[42:49]), to_float(row[49:56]))

            if any(np.invert(np.isnan(values[1:]))):
                arr_data.append((level,) + values)

    p, z, t, td, direc, spd = np.array(arr_data).T

    p = units.Quantity(p, 'hPa')
    z = units.Quantity(z, 'meters')
    t = units.Quantity(t, 'degC')
    td = units.Quantity(td, 'degC')
    direc = units.Quantity(direc, 'degrees')
    spd = units.Quantity(spd, 'knots')

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
        # dimensionless units if it is a quantity.
        else:
            if hasattr(actual, 'units'):
                actual = actual.to('dimensionless')
    except DimensionalityError:
        raise AssertionError('Units are not compatible: {} should be {}'.format(
            actual.units, getattr(desired, 'units', 'dimensionless'))) from None

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


def assert_nan(value, value_units):
    """Check for nan with proper units."""
    value, _ = check_and_drop_units(value, units.Quantity(np.nan, value_units))
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
