r'''Collection of utilities for testing

    Currently, this consists of unit-aware test functions
'''

# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy.testing
from pint import DimensionalityError
from .units import units


def check_and_drop_units(actual, desired):
    try:
        if not hasattr(desired, 'units'):
            actual = actual.to('dimensionless')
        elif not hasattr(actual, 'units'):
            actual = units.Quantity(actual, 'dimensionless')
        else:
            actual = actual.to(desired.units)
    except DimensionalityError:
        raise AssertionError('Units are not compatible: %s should be %s' %
                             (actual.units, desired.units))
    except AttributeError:
        pass

    if hasattr(actual, 'magnitude'):
        actual = actual.magnitude
    if hasattr(desired, 'magnitude'):
        desired = desired.magnitude

    return actual, desired


def assert_almost_equal(actual, desired, decimal=7):
    actual, desired = check_and_drop_units(actual, desired)
    numpy.testing.assert_almost_equal(actual, desired, decimal)


def assert_array_almost_equal(actual, desired, decimal=7):
    actual, desired = check_and_drop_units(actual, desired)
    numpy.testing.assert_array_almost_equal(actual, desired, decimal)


def assert_array_equal(actual, desired):
    actual, desired = check_and_drop_units(actual, desired)
    numpy.testing.assert_array_equal(actual, desired)
