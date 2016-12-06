# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Collection of utilities for testing.

This includes:
* unit-aware test functions
* code for testing matplotlib figures
"""

import numpy as np
import numpy.testing
from pint import DimensionalityError
import pytest

from .units import units


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
        raise AssertionError('Units are not compatible: {} should be {}'.format(actual.units,
                                                                                desired.units))
    except AttributeError:
        pass

    if hasattr(actual, 'magnitude'):
        actual = actual.magnitude
    if hasattr(desired, 'magnitude'):
        desired = desired.magnitude

    return actual, desired


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
    numpy.testing.assert_array_almost_equal(actual, desired, decimal)


def assert_array_equal(actual, desired):
    """Check that arrays are equal, including units.

    Wrapper around :func:`numpy.testing.assert_array_equal`
    """
    actual, desired = check_and_drop_units(actual, desired)
    numpy.testing.assert_array_equal(actual, desired)


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


@pytest.fixture(autouse=True)
def patch_round(monkeypatch):
    """Fixture to patch builtin round using numpy's.

    This works around the fact that built-in round changed between Python 2 and 3. This
    is probably not needed once we're testing on matplotlib 2.0, which has been updated
    to use numpy's throughout.
    """
    monkeypatch.setitem(__builtins__, 'round', np.round)
