# Copyright (c) 2015,2016,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Collection of utilities for data type testing.

Currently supported types include:
* scalar
* NaN
* NumPy array
* NumPy masked array
* Xarray DataArray

Planned support:
* Dask Array
"""
import inspect
import importlib

import pytest
import numpy as np
import xarray as xr

from metpy.units import units
from metpy.testing import assert_almost_equal, assert_array_almost_equal


def build_scenarios(test_data, module_ext):
    """Build scenarios for pytest_generate_tests from a test data dictionary.
    See https://docs.pytest.org/en/latest/example/parametrize.html#paramexamples
    """

    def build(funcname, values):

        # Set number of decimal places to pass to almost equal assertions
        decimal = values.pop('decimal', 4)

        module = importlib.import_module(module_ext)
        func = getattr(module, funcname)
        params = inspect.signature(func).parameters
        args = [v for key, v in values.items() if key in params]
        truth = [v for key, v in values.items() if key not in params]

        scenario = (
            funcname,
            {
                'func': func,
                'args': args,
                'truth': truth,
                'decimal': decimal,
            }
        )

        return scenario

    scenarios = [build(fname, values) for fname, values in test_data.items()]

    return scenarios


class IncorrectReturnsError(Exception):
    pass


def _check_length_results_and_truth(func, returns, truths):
    """Raise an exception if the number of return values from the tested function
    does not match the provided number of truth values
    """

    nreturn = len(returns) if isinstance(returns, tuple) else 1
    ntruth = len(truths)
    if nreturn != ntruth:
        msg = (
            f"Function {func.__module__}.{func.__name__} returned {nreturn} value(s),"
            f" but {ntruth} value(s) was/were provided as truth for testing"
        )

        raise IncorrectReturnsError(msg)


def scalar(func, args, truth, decimal):
    """Test a function using scalars."""

    # Index by a tuple to select a single value from an ndarray
    args_scalar = [a[tuple(np.zeros_like(a.shape))] for a in args]
    truths_scalar = [t[tuple(np.zeros_like(t.shape))] for t in truth]

    results = func(*args_scalar)

    _check_length_results_and_truth(func, results, truths_scalar)

    if isinstance(results, tuple):
        for t, r in zip(truths_scalar, results):
            assert_almost_equal(t, r, decimal)
    else:
        assert_almost_equal(*truths_scalar, results, decimal)


def array(func, args, truth, decimal):
    """Test a function using arrays."""
    results = func(*args)

    _check_length_results_and_truth(func, results, truth)

    _unpack_and_assert(truth, results, decimal)


def masked(func, args, truth, decimal):
    """Test a function using masked arrays."""

    # Create a mask that is True/False for every other value
    mask = np.ones_like(args[0].m, dtype=bool)
    mask[::2] = False

    args_masked = [units.Quantity(np.ma.array(a.m, mask=mask), a.units) for a in args]
    truth_masked = [units.Quantity(np.ma.array(t.m, mask=mask), t.units) for t in truth]

    results = func(*args_masked)

    _check_length_results_and_truth(func, results, truth)

    _unpack_and_assert(truth_masked, results, decimal)


def nans(func, args, truth, decimal):
    """Test a function using nans."""

    # Create copies of args and truth, then add nans at every other index
    def assign_nans(arr):
        arr_nans = arr.copy()
        for i, a in enumerate(arr_nans):
            # Force the copied array to float64 so that it plays nicely with nans
            anans = a.copy().astype('float64')
            anans[::2] = np.nan
            arr_nans[i] = anans
        return arr_nans

    args_nans = assign_nans(args)
    truth_nans = assign_nans(truth)

    results = func(*args_nans)

    _check_length_results_and_truth(func, results, truth)

    _unpack_and_assert(truth_nans, results, decimal)


def data_array(func, args, truth, decimal):
    """Test a function using data arrays."""

    args_data_array = [xr.DataArray(a) for a in args]
    truth_data_array = [xr.DataArray(t) for t in truth]

    results = func(*args_data_array)

    _check_length_results_and_truth(func, results, truth)

    _unpack_and_assert(truth_data_array, results, decimal)


def dask_arrays(func, args, truth, decimal):
    """Test a function using dask arrays."""
    dask = pytest.importorskip("dask", reason="Dask is not available")
    dask_array = pytest.importorskip(
        "dask.array", reason="Module dask.array is not available"
    )
    pint = pytest.importorskip(
        "pint",
        minversion="0.15",
        reason="Pint >=0.15 needed for Quantity wrapped Dask Arrays"
    )

    args_dask = [dask_array.from_array(a.m) * a.units for a in args]
    truth_dask = [dask_array.from_array(t.m) * t.units for t in truth]

    results = func(*args_dask)

    _check_length_results_and_truth(func, results, truth)

    if isinstance(results, tuple):
        computed = dask.compute(*results)
        persisted = dask.persist(*results)
    else:
        computed = results.compute()
        persisted = results.persist()

    _unpack_and_assert(truth_dask, results, decimal)
    _unpack_and_assert(truth, computed, decimal)
    _unpack_and_assert(truth_dask, persisted, decimal)


def _unpack_and_assert(truth, results, decimal):
    """Unpack and assert equality of an array-like data type."""
    if isinstance(results, tuple):
        for t, r in zip(truth, results):
            assert_array_almost_equal(t, r, decimal)
    else:
        assert_array_almost_equal(*truth, results, decimal)
