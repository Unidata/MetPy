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
import pytest
import scipy
import traitlets
import xarray

import metpy.calc

# Need to disable fallback before importing pint
os.environ['PINT_ARRAY_PROTOCOL_FALLBACK'] = '0'
import pint  # noqa: I100, E402


def pytest_report_header(config, startdir):
    """Add dependency information to pytest output."""
    return (f'Dep Versions: Matplotlib {matplotlib.__version__}, '
            f'NumPy {numpy.__version__}, SciPy {scipy.__version__}, '
            f'Xarray {xarray.__version__}, Pint {pint.__version__}, '
            f'Pandas {pandas.__version__}, Traitlets {traitlets.__version__}, '
            f'Pooch {pooch.version.full_version}')


@pytest.fixture(autouse=True)
def doctest_available_modules(doctest_namespace):
    """Make modules available automatically to doctests."""
    doctest_namespace['metpy'] = metpy
    doctest_namespace['metpy.calc'] = metpy.calc
    doctest_namespace['plt'] = matplotlib.pyplot


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


# Automatically generate tests for data type support with testscenarios
# See https://docs.pytest.org/en/stable/example/parametrize.html#a-quick-port-of-testscenarios
def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []

    try:
        for scenario in metafunc.cls.scenarios:
            idlist.append(scenario[0])
            items = scenario[1].items()
            argnames = [x[0] for x in items]
            argvalues.append([x[1] for x in items])
        metafunc.parametrize(argnames, argvalues, ids=idlist)

    # Skip classes without a "scenarios" attribute
    except AttributeError:
        pass
