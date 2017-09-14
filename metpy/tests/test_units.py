# Copyright (c) 2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Tests the operation of MetPy's unit support code."""

import sys

import matplotlib.pyplot as plt
import pytest

from metpy.testing import set_agg_backend  # noqa: F401
from metpy.units import check_units, units


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
    pass


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


@pytest.mark.skipif(sys.version_info < (3, 3), reason='Unit checking requires Python >= 3.3')
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
