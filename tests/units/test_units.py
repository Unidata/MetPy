# Copyright (c) 2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Tests the operation of MetPy's unit support code."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from metpy.testing import assert_array_almost_equal, assert_array_equal, assert_nan
from metpy.units import (check_units, concatenate, is_quantity,
                         pandas_dataframe_to_unit_arrays, units)


def test_concatenate():
    """Test basic functionality of unit-aware concatenate."""
    result = concatenate((3 * units.meter, 400 * units.cm))
    assert_array_equal(result, np.array([3, 4]) * units.meter)
    assert not isinstance(result.m, np.ma.MaskedArray)


def test_concatenate_masked():
    """Test concatenate preserves masks."""
    d1 = units.Quantity(np.ma.array([1, 2, 3], mask=[False, True, False]), 'degC')
    result = concatenate((d1, 32 * units.degF))

    truth = np.ma.array([1, np.inf, 3, 0])
    truth[1] = np.ma.masked

    assert_array_almost_equal(result, units.Quantity(truth, 'degC'), 6)
    assert_array_equal(result.mask, np.array([False, True, False, False]))


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_axhline():
    r"""Ensure that passing a quantity to axhline does not error."""
    fig, ax = plt.subplots()
    ax.axhline(930 * units('mbar'))
    ax.set_ylim(900, 950)
    ax.set_ylabel('')
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_axvline():
    r"""Ensure that passing a quantity to axvline does not error."""
    fig, ax = plt.subplots()
    ax.axvline(0 * units('degC'))
    ax.set_xlim(-1, 1)
    ax.set_xlabel('')
    return fig


#
# Tests for unit-checking decorator
#


def unit_calc(temp, press, dens, mixing, unitless_const):
    r"""Stub calculation for testing unit checking."""


test_funcs = [
    check_units('[temperature]', '[pressure]', dens='[mass]/[volume]',
                mixing='[dimensionless]')(unit_calc),
    check_units(temp='[temperature]', press='[pressure]', dens='[mass]/[volume]',
                mixing='[dimensionless]')(unit_calc),
    check_units('[temperature]', '[pressure]', '[mass]/[volume]',
                '[dimensionless]')(unit_calc)]


@pytest.mark.parametrize('func', test_funcs, ids=['some kwargs', 'all kwargs', 'all pos'])
def test_good_units(func):
    r"""Test that unit checking passes good units regardless."""
    func(30 * units.degC, 1000 * units.mbar, 1.0 * units('kg/m^3'), 1, 5.)


test_params = [((30 * units.degC, 1000 * units.mb, 1 * units('kg/m^3'), 1, 5 * units('J/kg')),
                {}, [('press', '[pressure]', 'millibarn')]),
               ((30, 1000, 1.0, 1, 5.), {}, [('press', '[pressure]', 'none'),
                                             ('temp', '[temperature]', 'none'),
                                             ('dens', '[mass]/[volume]', 'none')]),
               ((30, 1000 * units.mbar),
                {'dens': 1.0 * units('kg / m'), 'mixing': 5 * units.m, 'unitless_const': 2},
                [('temp', '[temperature]', 'none'),
                 ('dens', '[mass]/[volume]', 'kilogram / meter'),
                 ('mixing', '[dimensionless]', 'meter')])]


@pytest.mark.parametrize('func', test_funcs, ids=['some kwargs', 'all kwargs', 'all pos'])
@pytest.mark.parametrize('args,kwargs,bad_parts', test_params,
                         ids=['one bad arg', 'all args no units', 'mixed args'])
def test_bad(func, args, kwargs, bad_parts):
    r"""Test that unit checking flags appropriate arguments."""
    with pytest.raises(ValueError) as exc:
        func(*args, **kwargs)

    message = str(exc.value)
    assert func.__name__ in message
    for param in bad_parts:
        assert '`{}` requires "{}" but given "{}"'.format(*param) in message

        # Should never complain about the const argument
        assert 'unitless_const' not in message


@pytest.mark.parametrize('func', test_funcs, ids=['some kwargs', 'all kwargs', 'all pos'])
def test_bad_masked_array(func):
    """Test getting a masked array-specific message when missing units."""
    with pytest.raises(ValueError) as exc:
        func(np.ma.array([30]), 1000 * units.mbar, 1.0 * units('kg/m^3'), 1, 5.)

    message = str(exc.value)
    assert 'units.Quantity' in message


def test_pandas_units_simple():
    """Simple unit attachment to two columns."""
    df = pd.DataFrame(data=[[1, 4], [2, 5], [3, 6]], columns=['cola', 'colb'])
    df_units = {'cola': 'kilometers', 'colb': 'degC'}
    res = pandas_dataframe_to_unit_arrays(df, column_units=df_units)
    cola_truth = np.array([1, 2, 3]) * units.km
    colb_truth = np.array([4, 5, 6]) * units.degC
    assert_array_equal(res['cola'], cola_truth)
    assert_array_equal(res['colb'], colb_truth)


@pytest.mark.filterwarnings("ignore:Pandas doesn't allow columns to be created")
def test_pandas_units_on_dataframe():
    """Unit attachment based on a units attribute to a dataframe."""
    df = pd.DataFrame(data=[[1, 4], [2, 5], [3, 6]], columns=['cola', 'colb'])
    df.units = {'cola': 'kilometers', 'colb': 'degC'}
    res = pandas_dataframe_to_unit_arrays(df)
    cola_truth = np.array([1, 2, 3]) * units.km
    colb_truth = np.array([4, 5, 6]) * units.degC
    assert_array_equal(res['cola'], cola_truth)
    assert_array_equal(res['colb'], colb_truth)


@pytest.mark.filterwarnings("ignore:Pandas doesn't allow columns to be created")
def test_pandas_units_on_dataframe_not_all_with_units():
    """Unit attachment with units attribute with a column with no units."""
    df = pd.DataFrame(data=[[1, 4], [2, 5], [3, 6]], columns=['cola', 'colb'])
    df.units = {'cola': 'kilometers'}
    res = pandas_dataframe_to_unit_arrays(df)
    cola_truth = np.array([1, 2, 3]) * units.km
    colb_truth = np.array([4, 5, 6])
    assert_array_equal(res['cola'], cola_truth)
    assert_array_equal(res['colb'], colb_truth)


def test_pandas_units_no_units_given():
    """Ensure unit attachment fails if no unit information is given."""
    df = pd.DataFrame(data=[[1, 4], [2, 5], [3, 6]], columns=['cola', 'colb'])
    with pytest.raises(ValueError):
        pandas_dataframe_to_unit_arrays(df)


def test_added_degrees_units():
    """Test that our added degrees units are present in the registry."""
    # Test equivalence of abbreviations/aliases to our defined names
    assert str(units('degrees_N').units) == 'degrees_north'
    assert str(units('degreesN').units) == 'degrees_north'
    assert str(units('degree_north').units) == 'degrees_north'
    assert str(units('degree_N').units) == 'degrees_north'
    assert str(units('degreeN').units) == 'degrees_north'
    assert str(units('degrees_E').units) == 'degrees_east'
    assert str(units('degreesE').units) == 'degrees_east'
    assert str(units('degree_east').units) == 'degrees_east'
    assert str(units('degree_E').units) == 'degrees_east'
    assert str(units('degreeE').units) == 'degrees_east'

    # Test equivalence of our defined units to base units
    assert units('degrees_north') == units('degrees')
    assert units('degrees_north').to_base_units().units == units.radian
    assert units('degrees_east') == units('degrees')
    assert units('degrees_east').to_base_units().units == units.radian


def test_is_quantity():
    """Test is_quantity properly works."""
    assert is_quantity(1 * units.m)
    assert not is_quantity(np.array([1]))


def test_is_quantity_multiple():
    """Test is_quantity with multiple inputs."""
    assert is_quantity(1 * units.m, np.array([4.]) * units.degree)
    assert not is_quantity(1 * units.second, np.array([5., 2.]))


def test_gpm_unit():
    """Test that the gpm unit does alias to meters."""
    x = 1 * units('gpm')
    assert str(x.units) == 'meter'


def test_assert_nan():
    """Test that assert_nan actually fails when not given a NaN."""
    with pytest.raises(AssertionError):
        assert_nan(1.0 * units.m, units.inches)


def test_assert_nan_checks_units():
    """Test that assert_nan properly checks units."""
    with pytest.raises(AssertionError):
        assert_nan(np.nan * units.m, units.second)


def test_percent_units():
    """Test that percent sign units are properly parsed and interpreted."""
    assert str(units('%').units) == 'percent'


@pytest.mark.parametrize(
    'unit_str,pint_unit',
    (
        # Validated against cf-units (UDUNITS-2 wrapper)
        ('m s-1', units.m / units.s),
        ('m2 s-2', units.m ** 2 / units.s ** 2),
        ('kg(-1)', units.kg),
        ('kg-1', units.kg ** -1),
        ('W m(-2)', units.m ** 3 * units.kg * units.s ** -3),
        ('W m-2', units.kg * units.s ** -3),
        ('(W m-2 um-1)-1', units.m * units.kg ** -1 * units.s ** 3),
        pytest.param(
            '(J kg-1)(m s-1)-1', units.m / units.s,
            marks=pytest.mark.xfail(reason='hgrecco/pint#1485')
        ),
        ('(J kg-1)(m s-1)(-1)', units.m ** 3 / units.s ** 3)
    )
)
def test_udunits_power_syntax(unit_str, pint_unit):
    """Test that UDUNITS style powers are properly parsed and interpreted."""
    assert units(unit_str).to_base_units().units == pint_unit
