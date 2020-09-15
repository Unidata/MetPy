# Copyright (c) 2015,2016,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `data_type_testing` module."""

import pytest
import numpy as np

from metpy.units import units
from metpy.data_type_testing import (
    build_scenarios, scalar, array, masked, nans, data_array, dask_arrays
)
from metpy.testing import inc, add, subt, div, decimal, assert_array_equal


@pytest.fixture
def data():
    """A test data dictionary similar to what a contributor would add to."""

    test_data = {
        # Single arg, single truth
        'inc': {
            'x': np.arange(10) * units('dimensionless'),
            'res': np.arange(1, 11) * units('dimensionless'),
        },
        # Two args, single truth
        'add': {
            'x': np.arange(10) * units('dimensionless'),
            'y': np.arange(10, 20) * units('dimensionless'),
            'res': np.arange(10, 30, 2) * units('dimensionless'),
        },
        # Two args, two truths
        'subt': {
            'x': np.arange(10) * units('dimensionless'),
            'y': np.arange(10, 20) * units('dimensionless'),
            'res1': np.array([-10] * 10) * units('dimensionless'),
            'res2': np.array([10] * 10) * units('dimensionless'),
        },
        # Single arg, two truths
        'div': {
            'x': np.arange(10) * units('dimensionless'),
            'res1': np.arange(0, 5, step=0.5) * units('dimensionless'),
            'res2': np.arange(0, 2, step=0.2) * units('dimensionless'),
        },
        # Test decimal kwarg
        'decimal': {
            'x': np.arange(10) * units('dimensionless'),
            'res1': np.arange(0, 5, step=0.5) * units('dimensionless'),
            'res2': np.arange(0, 2, step=0.2) * units('dimensionless'),
            'decimal': 8,
        }
    }

    return test_data


@pytest.fixture
def scenarios():
    """A list of scenarios in a format that testscenarios is expecting."""

    scen = [
        # Single arg, single truth
        ('inc', {
            'func': inc,
            'args': [
                np.arange(10) * units('dimensionless'),
            ],
            'truth': [
                np.arange(1, 11) * units('dimensionless'),
            ],
            'decimal': 4,
        }),
        # Two args, single truth
        ('add', {
            'func': add,
            'args': [
                np.arange(10) * units('dimensionless'),
                np.arange(10, 20) * units('dimensionless'),
            ],
            'truth': [
                np.arange(10, 30, 2) * units('dimensionless'),
            ],
            'decimal': 4,
        }),
        # Two args, two truths
        ('subt', {
            'func': subt,
            'args': [
                np.arange(10) * units('dimensionless'),
                np.arange(10, 20) * units('dimensionless'),
            ],
            'truth': [
                np.array([-10] * 10) * units('dimensionless'),
                np.array([10] * 10) * units('dimensionless'),
            ],
            'decimal': 4,
        }),
        # Single arg, two truths
        ('div', {
            'func': div,
            'args': [
                np.arange(10) * units('dimensionless'),
            ],
            'truth': [
                np.arange(0, 5, step=0.5) * units('dimensionless'),
                np.arange(0, 2, step=0.2) * units('dimensionless'),
            ],
            'decimal': 4,
        }),
        # Test decimal kwarg
        ('decimal', {
            'func': decimal,
            'args': [
                np.arange(10) * units('dimensionless'),
            ],
            'truth': [
                np.arange(0, 5, step=0.5) * units('dimensionless'),
                np.arange(0, 2, step=0.2) * units('dimensionless'),
            ],
            'decimal': 8,
        })]

    return scen


def test_build_scenarios(scenarios, data):
    """Test that scenarios are built correctly from test data dictionaries."""

    scenarios_test = build_scenarios(data, module_ext='metpy.testing')

    assert isinstance(scenarios_test, list)

    for scen, scen_test in zip(scenarios, scenarios_test):
        assert scen[0] == scen_test[0]

        truthdict = scen[1]
        testdict = scen_test[1]
        for truthkey, testkey in zip(truthdict, testdict):
            assert truthkey == testkey

            if (
                isinstance(truthdict[truthkey], list) and
                isinstance(testdict[testkey], list)
            ):
                for trutharr, testarr in zip(truthdict[truthkey], testdict[testkey]):
                    assert_array_equal(trutharr, testarr)
            else:
                assert truthdict[truthkey] == testdict[testkey]


def test_scalar(scenarios):
    """Test the scalar test."""
    for scen in scenarios:
        scalar(**scen[1])


def test_nans(scenarios):
    """Test the nans test."""
    for scen in scenarios:
        nans(**scen[1])


def test_arrays(scenarios):
    """Test the arrays test."""
    for scen in scenarios:
        array(**scen[1])


def test_masked(scenarios):
    """Test the masked test."""
    for scen in scenarios:
        masked(**scen[1])


def test_data_array(scenarios):
    """Test the data_array test."""
    for scen in scenarios:
        data_array(**scen[1])


def test_dask_array(scenarios):
    """Test the dask_arrays test."""
    for scen in scenarios:
        dask_arrays(**scen[1])
